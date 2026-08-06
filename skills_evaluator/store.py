"""Revision storage — the corpus this cassette accumulates.

Every proposed revision is stored with the evaluation that motivated it and,
once a host reports back through the status hook, the human verdict:
``accepted`` or ``rejected``. That triple — (skill + evidence, proposed
revision, verdict) — is exactly the labeled trainset a GEPA optimizer needs,
and the ``skill_id`` column is the loose link back to tapes' skills table so
a skill's whole revision history hangs together. Loose on purpose: a
cassette owns tables only in its own schema and never foreign-keys into
core's.

Two implementations, per the cassette convention: Postgres when the
deployment supplies ``TAPES_DATABASE_URL`` (the cassette runs its own
migration in its own schema, which core neither creates nor reads), and an
in-memory store so the cassette is runnable — and testable — with nothing
configured. The memory store says so in every record's ``store`` field
rather than pretending durability it does not have.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

# The cassette's schema name is its cassette name; the hyphen makes quoting
# mandatory everywhere it reaches SQL.
SCHEMA = "skills-evaluator"

PROPOSED = "proposed"
ACCEPTED = "accepted"
REJECTED = "rejected"
DECIDED_STATUSES = (ACCEPTED, REJECTED)


class AlreadyDecidedError(Exception):
    """The revision already carries a different terminal status."""


@dataclass
class RevisionRecord:
    """One proposed skill revision and its lifecycle."""

    id: str
    skill_id: str
    ref: dict[str, Any] | None
    skill_name: str
    original_skill_md: str
    revised_skill_md: str
    rationale: str
    evaluation: dict[str, Any]
    status: str = PROPOSED
    status_reason: str = ""
    created_at: str = ""
    decided_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def new_revision_id() -> str:
    return str(uuid.uuid4())


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


class RevisionStore(Protocol):
    def kind(self) -> str: ...

    def insert(self, record: RevisionRecord) -> RevisionRecord: ...

    def get(self, revision_id: str) -> RevisionRecord | None: ...

    def set_status(
        self, revision_id: str, status: str, reason: str
    ) -> RevisionRecord | None: ...

    def list_for_skill(self, skill_id: str, limit: int) -> list[RevisionRecord]: ...


class MemoryRevisionStore:
    """The no-database fallback."""

    def __init__(self) -> None:
        self._records: dict[str, RevisionRecord] = {}

    def kind(self) -> str:
        return "memory"

    def insert(self, record: RevisionRecord) -> RevisionRecord:
        self._records[record.id] = record
        return record

    def get(self, revision_id: str) -> RevisionRecord | None:
        return self._records.get(revision_id)

    def set_status(
        self, revision_id: str, status: str, reason: str
    ) -> RevisionRecord | None:
        record = self._records.get(revision_id)
        if record is None:
            return None
        _check_transition(record, status)
        if record.status == PROPOSED:
            record.status = status
            record.status_reason = reason
            record.decided_at = utcnow()
        return record

    def list_for_skill(self, skill_id: str, limit: int) -> list[RevisionRecord]:
        matches = [r for r in self._records.values() if r.skill_id == skill_id]
        matches.sort(key=lambda r: r.created_at, reverse=True)
        return matches[:limit]


def _check_transition(record: RevisionRecord, status: str) -> None:
    """Only ``proposed`` revisions can be decided; re-asserting the same
    terminal status is idempotent, flipping it is refused — a corpus whose
    labels silently change is worse than no corpus."""
    if record.status in DECIDED_STATUSES and record.status != status:
        raise AlreadyDecidedError(
            f"revision {record.id} is already {record.status}"
        )


class PostgresRevisionStore:
    """Owns exactly one table in exactly one schema, and creates both itself.
    The deployment provisions the role and grants; core never sees the DDL."""

    def __init__(self, dsn: str) -> None:
        import psycopg

        self._psycopg = psycopg
        self._dsn = dsn
        self._migrate()

    def kind(self) -> str:
        return "postgres"

    def _connect(self):
        return self._psycopg.connect(self._dsn, autocommit=True)

    def _migrate(self) -> None:
        with self._connect() as conn:
            conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA}"')
            conn.execute(
                f'''CREATE TABLE IF NOT EXISTS "{SCHEMA}".revisions (
                    id                UUID PRIMARY KEY,
                    skill_id          TEXT NOT NULL DEFAULT '',
                    ref               JSONB,
                    skill_name        TEXT NOT NULL DEFAULT '',
                    original_skill_md TEXT NOT NULL,
                    revised_skill_md  TEXT NOT NULL,
                    rationale         TEXT NOT NULL DEFAULT '',
                    evaluation        JSONB NOT NULL,
                    status            TEXT NOT NULL DEFAULT 'proposed',
                    status_reason     TEXT NOT NULL DEFAULT '',
                    created_at        TIMESTAMPTZ NOT NULL,
                    decided_at        TIMESTAMPTZ
                )'''
            )
            conn.execute(
                f'CREATE INDEX IF NOT EXISTS revisions_skill_idx '
                f'ON "{SCHEMA}".revisions (skill_id, created_at DESC)'
            )

    _COLUMNS = (
        "id, skill_id, ref, skill_name, original_skill_md, revised_skill_md, "
        "rationale, evaluation, status, status_reason, created_at, decided_at"
    )

    def insert(self, record: RevisionRecord) -> RevisionRecord:
        with self._connect() as conn:
            conn.execute(
                f'INSERT INTO "{SCHEMA}".revisions ({self._COLUMNS}) '
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    record.id,
                    record.skill_id,
                    json.dumps(record.ref) if record.ref is not None else None,
                    record.skill_name,
                    record.original_skill_md,
                    record.revised_skill_md,
                    record.rationale,
                    json.dumps(record.evaluation),
                    record.status,
                    record.status_reason,
                    record.created_at,
                    record.decided_at,
                ),
            )
        return record

    def get(self, revision_id: str) -> RevisionRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                f'SELECT {self._COLUMNS} FROM "{SCHEMA}".revisions WHERE id = %s',
                (revision_id,),
            ).fetchone()
        return _from_row(row) if row else None

    def set_status(
        self, revision_id: str, status: str, reason: str
    ) -> RevisionRecord | None:
        record = self.get(revision_id)
        if record is None:
            return None
        _check_transition(record, status)
        if record.status == PROPOSED:
            decided_at = utcnow()
            with self._connect() as conn:
                conn.execute(
                    f'UPDATE "{SCHEMA}".revisions '
                    "SET status = %s, status_reason = %s, decided_at = %s "
                    "WHERE id = %s",
                    (status, reason, decided_at, revision_id),
                )
            record.status, record.status_reason = status, reason
            record.decided_at = decided_at
        return record

    def list_for_skill(self, skill_id: str, limit: int) -> list[RevisionRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                f'SELECT {self._COLUMNS} FROM "{SCHEMA}".revisions '
                "WHERE skill_id = %s ORDER BY created_at DESC LIMIT %s",
                (skill_id, limit),
            ).fetchall()
        return [_from_row(row) for row in rows]


def _from_row(row: tuple) -> RevisionRecord:
    return RevisionRecord(
        id=str(row[0]),
        skill_id=row[1],
        ref=row[2],
        skill_name=row[3],
        original_skill_md=row[4],
        revised_skill_md=row[5],
        rationale=row[6],
        evaluation=row[7],
        status=row[8],
        status_reason=row[9],
        created_at=row[10].isoformat() if hasattr(row[10], "isoformat") else str(row[10]),
        decided_at=(
            row[11].isoformat() if hasattr(row[11], "isoformat") else row[11]
        ),
    )


def open_store(dsn: str) -> RevisionStore:
    if dsn.strip():
        return PostgresRevisionStore(dsn.strip())
    return MemoryRevisionStore()
