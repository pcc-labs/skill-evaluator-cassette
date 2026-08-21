"""Revision storage — the corpus this cassette accumulates.

Every proposed revision is stored with the evaluation that motivated it and,
once a host reports back through the status hook, the human verdict:
``accepted`` or ``rejected``. That triple — (skill + evidence, proposed
revision, verdict) — is exactly the labeled trainset a GEPA optimizer needs.

``skill_id`` is the identity every row hangs off: a required reference to a
row in tapes' skills table (the server refuses requests without one, and
resolves the id against tapes before anything is stored). The reference is
deliberately not a SQL FOREIGN KEY — a cassette owns tables only in its own
schema and never foreign-keys into core's — but it is required and verified,
so a skill's evals and whole revision history join sanely on one key.

Two implementations, per the cassette convention: Postgres when the
deployment supplies ``TAPES_DATABASE_URL`` (the cassette runs its own
migration in its own schema, which core neither creates nor reads), and an
in-memory store so the cassette is runnable — and testable — with nothing
configured. The memory store says so in every record's ``store`` field
rather than pretending durability it does not have.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from yoyo import get_backend, read_migrations

# The cassette's schema name is its cassette name; the hyphen makes quoting
# mandatory everywhere it reaches SQL.
SCHEMA = "skills-evaluator"

PROPOSED = "proposed"
ACCEPTED = "accepted"
REJECTED = "rejected"
DECIDED_STATUSES = (ACCEPTED, REJECTED)


class AlreadyDecidedError(Exception):
    """The revision already carries a different terminal status."""


class EditedSpecError(Exception):
    """Regeneration would overwrite a human-edited spec; refused. A human
    edit is a statement about the metric, and the generator does not get to
    argue with it."""


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


ORIGIN_GENERATED = "generated"
ORIGIN_EDITED = "edited"


@dataclass
class EvalRecord:
    """One skill's eval spec — its intrinsic, human-editable metric.

    ``skill_id`` is the lookup identity: the tapes skills-table row this
    spec belongs to. One current spec per skill; ``origin`` records whether
    a human has taken it over. ``skill_name`` is a denormalized label for
    display, never a key.
    """

    id: str
    skill_id: str
    skill_name: str
    spec: dict[str, Any]
    source_sha256: str = ""
    origin: str = ORIGIN_GENERATED
    created_at: str = ""
    updated_at: str = ""

    @property
    def spec_sha256(self) -> str:
        return spec_sha(self.spec)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def spec_sha(spec: dict[str, Any]) -> str:
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


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

    def upsert_eval(self, record: EvalRecord, force: bool) -> EvalRecord: ...

    def get_eval(self, eval_id: str) -> EvalRecord | None: ...

    def get_eval_for_skill(self, skill_id: str) -> EvalRecord | None: ...

    def update_eval_spec(
        self, eval_id: str, spec: dict[str, Any]
    ) -> EvalRecord | None: ...


def _apply_eval_upsert(
    existing: EvalRecord | None, record: EvalRecord, force: bool
) -> EvalRecord | None:
    """Shared upsert policy: no spec → insert; generated spec → replace only
    with force; edited spec → never replaced by generation. Returns the
    record to keep, or None when the existing one stands."""
    if existing is None:
        return record
    if existing.origin == ORIGIN_EDITED:
        raise EditedSpecError(
            f"eval {existing.id} for skill {existing.skill_id!r} was human-edited; "
            "update it with PUT, not regeneration"
        )
    if not force:
        return None
    record.id = existing.id
    record.created_at = existing.created_at
    return record


class MemoryRevisionStore:
    """The no-database fallback."""

    def __init__(self) -> None:
        self._records: dict[str, RevisionRecord] = {}
        self._evals: dict[str, EvalRecord] = {}

    def kind(self) -> str:
        return "memory"

    def upsert_eval(self, record: EvalRecord, force: bool) -> EvalRecord:
        existing = self.get_eval_for_skill(record.skill_id)
        kept = _apply_eval_upsert(existing, record, force)
        if kept is None:
            return existing  # type: ignore[return-value]
        self._evals[kept.id] = kept
        return kept

    def get_eval(self, eval_id: str) -> EvalRecord | None:
        return self._evals.get(eval_id)

    def get_eval_for_skill(self, skill_id: str) -> EvalRecord | None:
        for record in self._evals.values():
            if record.skill_id == skill_id:
                return record
        return None

    def update_eval_spec(
        self, eval_id: str, spec: dict[str, Any]
    ) -> EvalRecord | None:
        record = self._evals.get(eval_id)
        if record is None:
            return None
        record.spec = spec
        record.origin = ORIGIN_EDITED
        record.updated_at = utcnow()
        return record

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


def _migration_dsn(dsn: str) -> str:
    """Select yoyo's Psycopg 3 backend and keep its ledger in our schema."""
    parsed = urlsplit(dsn)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["schema"] = f'"{SCHEMA}"'
    return urlunsplit(
        (
            "postgresql+psycopg",
            parsed.netloc,
            parsed.path,
            urlencode(query),
            parsed.fragment,
        )
    )


class PostgresRevisionStore:
    """Owns its schema and tables, applying versioned migrations at startup."""

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
        # Bootstrap only the namespace yoyo stores its own migration ledger in;
        # every cassette table and index belongs in a versioned migration.
        with self._connect() as conn:
            conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA}"')

        migrations = read_migrations(str(Path(__file__).with_name("migrations")))
        with get_backend(_migration_dsn(self._dsn)) as backend:
            with backend.lock():
                backend.apply_migrations(backend.to_apply(migrations))

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
        decided_at = utcnow()
        with self._connect() as conn:
            row = conn.execute(
                f'UPDATE "{SCHEMA}".revisions '
                "SET status = %s, status_reason = %s, decided_at = %s "
                "WHERE id = %s AND status = %s "
                f"RETURNING {self._COLUMNS}",
                (status, reason, decided_at, revision_id, PROPOSED),
            ).fetchone()
        if row:
            return _from_row(row)

        record = self.get(revision_id)
        if record is None:
            return None
        _check_transition(record, status)
        return record

    def list_for_skill(self, skill_id: str, limit: int) -> list[RevisionRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                f'SELECT {self._COLUMNS} FROM "{SCHEMA}".revisions '
                "WHERE skill_id = %s ORDER BY created_at DESC LIMIT %s",
                (skill_id, limit),
            ).fetchall()
        return [_from_row(row) for row in rows]

    _EVAL_COLUMNS = (
        "id, skill_id, skill_name, spec, source_sha256, origin, created_at, updated_at"
    )

    def upsert_eval(self, record: EvalRecord, force: bool) -> EvalRecord:
        with self._connect() as conn:
            row = conn.execute(
                f'INSERT INTO "{SCHEMA}".evals ({self._EVAL_COLUMNS}) '
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (skill_id) DO UPDATE SET "
                "spec = EXCLUDED.spec, source_sha256 = EXCLUDED.source_sha256, "
                "origin = EXCLUDED.origin, skill_name = EXCLUDED.skill_name, "
                "updated_at = EXCLUDED.updated_at "
                f'WHERE %s AND "{SCHEMA}".evals.origin = %s '
                f"RETURNING {self._EVAL_COLUMNS}",
                (
                    record.id,
                    record.skill_id,
                    record.skill_name,
                    json.dumps(record.spec),
                    record.source_sha256,
                    record.origin,
                    record.created_at,
                    record.updated_at,
                    force,
                    ORIGIN_GENERATED,
                ),
            ).fetchone()
        if row:
            return _eval_from_row(row)

        existing = self.get_eval_for_skill(record.skill_id)
        if existing is None:
            raise RuntimeError(f"eval upsert lost row for skill {record.skill_id!r}")
        if force and existing.origin == ORIGIN_EDITED:
            raise EditedSpecError(
                f"eval {existing.id} for skill {existing.skill_id!r} was human-edited; "
                "update it with PUT, not regeneration"
            )
        return existing

    def get_eval(self, eval_id: str) -> EvalRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                f'SELECT {self._EVAL_COLUMNS} FROM "{SCHEMA}".evals WHERE id = %s',
                (eval_id,),
            ).fetchone()
        return _eval_from_row(row) if row else None

    def get_eval_for_skill(self, skill_id: str) -> EvalRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                f'SELECT {self._EVAL_COLUMNS} FROM "{SCHEMA}".evals WHERE skill_id = %s',
                (skill_id,),
            ).fetchone()
        return _eval_from_row(row) if row else None

    def update_eval_spec(
        self, eval_id: str, spec: dict[str, Any]
    ) -> EvalRecord | None:
        record = self.get_eval(eval_id)
        if record is None:
            return None
        updated_at = utcnow()
        with self._connect() as conn:
            conn.execute(
                f'UPDATE "{SCHEMA}".evals '
                "SET spec = %s, origin = %s, updated_at = %s WHERE id = %s",
                (json.dumps(spec), ORIGIN_EDITED, updated_at, eval_id),
            )
        record.spec, record.origin, record.updated_at = spec, ORIGIN_EDITED, updated_at
        return record


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


def _eval_from_row(row: tuple) -> EvalRecord:
    return EvalRecord(
        id=str(row[0]),
        skill_id=row[1],
        skill_name=row[2],
        spec=row[3],
        source_sha256=row[4],
        origin=row[5],
        created_at=row[6].isoformat() if hasattr(row[6], "isoformat") else str(row[6]),
        updated_at=row[7].isoformat() if hasattr(row[7], "isoformat") else str(row[7]),
    )


def open_store(dsn: str) -> RevisionStore:
    if dsn.strip():
        return PostgresRevisionStore(dsn.strip())
    return MemoryRevisionStore()
