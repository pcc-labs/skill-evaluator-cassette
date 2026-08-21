from concurrent.futures import ThreadPoolExecutor
import os
from threading import Barrier
import uuid

import pytest

from skills_evaluator.store import (
    AlreadyDecidedError,
    EditedSpecError,
    EvalRecord,
    PostgresRevisionStore,
    RevisionRecord,
    SCHEMA,
    utcnow,
)

DSN = os.environ.get("TEST_POSTGRES_DSN", "")
pytestmark = pytest.mark.skipif(not DSN, reason="TEST_POSTGRES_DSN is not set")


@pytest.fixture
def store():
    import psycopg

    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute(f'DROP SCHEMA IF EXISTS "{SCHEMA}" CASCADE')
    opened = PostgresRevisionStore(DSN)
    yield opened
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute(f'DROP SCHEMA IF EXISTS "{SCHEMA}" CASCADE')


def revision() -> RevisionRecord:
    return RevisionRecord(
        id=str(uuid.uuid4()),
        skill_id="skill-1",
        ref=None,
        skill_name="Skill",
        original_skill_md="# Original",
        revised_skill_md="# Revised skill with enough content",
        rationale="r",
        evaluation={},
        created_at=utcnow(),
    )


def evaluation(source: str) -> EvalRecord:
    now = utcnow()
    return EvalRecord(
        id=str(uuid.uuid4()),
        skill_id="skill-1",
        skill_name="Skill",
        spec={"criteria": [], "cases": []},
        source_sha256=source,
        created_at=now,
        updated_at=now,
    )


def test_opposite_revision_decisions_have_one_winner(store):
    record = store.insert(revision())
    barrier = Barrier(2)

    def decide(status: str):
        barrier.wait()
        try:
            return store.set_status(record.id, status, status)
        except AlreadyDecidedError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(decide, ("accepted", "rejected")))

    assert sum(result is not None for result in results) == 1
    assert store.get(record.id).status in ("accepted", "rejected")


def test_concurrent_first_eval_generation_returns_the_persisted_identity(store):
    barrier = Barrier(2)

    def insert(record: EvalRecord):
        barrier.wait()
        return store.upsert_eval(record, force=False)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(insert, (evaluation("a"), evaluation("b"))))

    persisted = store.get_eval_for_skill("skill-1")
    assert persisted is not None
    assert {result.id for result in results} == {persisted.id}


def test_forced_generation_never_overwrites_an_edited_spec(store):
    record = store.upsert_eval(evaluation("before"), force=False)
    store.update_eval_spec(
        record.id,
        {"criteria": [{"id": "human", "description": "keep", "weight": 1}], "cases": []},
    )

    with pytest.raises(EditedSpecError):
        store.upsert_eval(evaluation("after"), force=True)

    kept = store.get_eval_for_skill("skill-1")
    assert kept.origin == "edited"
    assert kept.source_sha256 == "before"
    assert kept.spec["criteria"][0]["id"] == "human"
