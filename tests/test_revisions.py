from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from tests.conftest import seed_morning_skill
from skills_evaluator.server import create_app
from skills_evaluator.service import NoEvidenceError
from skills_evaluator.store import (
    AlreadyDecidedError,
    MemoryRevisionStore,
    RevisionRecord,
    new_revision_id,
    utcnow,
)
from skills_evaluator.tapes import SearchHit


@dataclass
class FakeReviser:
    calls: list[dict] = field(default_factory=list)

    def __call__(self, **kwargs):
        self.calls.append(kwargs)

        @dataclass
        class Proposal:
            revised_markdown: str = "# Morning catchup (revised)\n\nBetter steps."
            rationale: str = "Addressed the ambiguity finding using session s1."

        return Proposal()


class _FenceOnlyReviser:
    def __call__(self, **kwargs):
        @dataclass
        class Proposal:
            revised_markdown: str = "```"
            rationale: str = "r"

        return Proposal()


class _FencedReviser:
    def __call__(self, **kwargs):
        @dataclass
        class Proposal:
            revised_markdown: str = "```markdown\n# Morning catchup (revised)\n\nBetter steps.\n```"
            rationale: str = "r"

        return Proposal()


@pytest.fixture
def reviser() -> FakeReviser:
    return FakeReviser()


@pytest.fixture
def revising_service(service, reviser):
    service.reviser = reviser
    service.store = MemoryRevisionStore()
    return service


@pytest.fixture
def client(revising_service) -> TestClient:
    return TestClient(create_app(revising_service))


def seed_evidence(fake_tapes) -> None:
    fake_tapes.hits["morning-catchup Daily inbox triage"] = [
        SearchHit(session_id="s1", trace_id="t1", score=0.9)
    ]
    fake_tapes.transcripts["s1"] = "[user] catch me up on my inbox\n"


class TestServiceRevise:
    def test_revise_stores_a_proposed_record(
        self, revising_service, fake_tapes, reviser, request_fixture
    ):
        seed_evidence(fake_tapes)
        record = revising_service.revise(request_fixture)

        assert record.status == "proposed"
        assert record.revised_skill_md.startswith("# Morning catchup (revised)")
        assert record.evaluation["decision"] == "pass"
        assert record.original_skill_md == request_fixture.candidate.skill_md
        assert reviser.calls[0]["skill_name"] == "morning-catchup"
        assert "catch me up" in reviser.calls[0]["session_evidence"]
        assert revising_service.store.get(record.id) is not None

    def test_revise_refuses_without_evidence(self, revising_service, request_fixture):
        with pytest.raises(NoEvidenceError):
            revising_service.revise(request_fixture)

    def test_revise_refuses_a_fence_only_rewrite(
        self, revising_service, fake_tapes, reviser, request_fixture
    ):
        from skills_evaluator.service import RevisionFailedError

        seed_evidence(fake_tapes)
        reviser.calls  # keep the fake; override its proposal content
        revising_service.reviser = _FenceOnlyReviser()
        with pytest.raises(RevisionFailedError):
            revising_service.revise(request_fixture)

    def test_revise_unwraps_fenced_documents(
        self, revising_service, fake_tapes, request_fixture
    ):
        seed_evidence(fake_tapes)
        revising_service.reviser = _FencedReviser()
        record = revising_service.revise(request_fixture)
        assert record.revised_skill_md.startswith("# Morning catchup")
        assert "```" not in record.revised_skill_md


class TestStoreLifecycle:
    def make_record(self) -> RevisionRecord:
        return RevisionRecord(
            id=new_revision_id(),
            skill_id="sk-1",
            ref=None,
            skill_name="n",
            original_skill_md="# a",
            revised_skill_md="# b",
            rationale="r",
            evaluation={},
            created_at=utcnow(),
        )

    def test_decide_once_then_idempotent(self):
        store = MemoryRevisionStore()
        record = store.insert(self.make_record())

        decided = store.set_status(record.id, "accepted", "looks right")
        assert decided.status == "accepted"
        assert decided.decided_at is not None

        again = store.set_status(record.id, "accepted", "still right")
        assert again.status_reason == "looks right", "first decision wins"

        with pytest.raises(AlreadyDecidedError):
            store.set_status(record.id, "rejected", "changed my mind")

    def test_list_for_skill_newest_first(self):
        store = MemoryRevisionStore()
        first = self.make_record()
        second = self.make_record()
        second.created_at = "9999-01-01T00:00:00+00:00"
        store.insert(first)
        store.insert(second)

        listed = store.list_for_skill("sk-1", 10)
        assert [r.id for r in listed] == [second.id, first.id]
        assert store.list_for_skill("other", 10) == []


class TestRevisionEndpoints:
    def test_full_lifecycle(self, client, fake_tapes):
        skill_id = seed_morning_skill(fake_tapes)
        seed_evidence(fake_tapes)
        created = client.post(
            "/api/skills-evaluator/revisions",
            json={
                "ref": {"source": "test", "id": "prop-1"},
                "skill_id": skill_id,
                "candidate": {"skill_md": "# Morning catchup\n\nSteps."},
            },
        )
        assert created.status_code == 201
        body = created.json()
        assert body["status"] == "proposed"
        assert body["skill_id"] == skill_id
        assert body["ref"]["id"] == "prop-1"
        revision_id = body["id"]

        fetched = client.get(f"/api/skills-evaluator/revisions/{revision_id}")
        assert fetched.status_code == 200

        accepted = client.post(
            f"/api/skills-evaluator/revisions/{revision_id}/status",
            json={"status": "accepted", "reason": "applied upstream"},
        )
        assert accepted.status_code == 200
        assert accepted.json()["status"] == "accepted"

        flipped = client.post(
            f"/api/skills-evaluator/revisions/{revision_id}/status",
            json={"status": "rejected"},
        )
        assert flipped.status_code == 409

    def test_no_evidence_is_409(self, client, fake_tapes):
        skill_id = seed_morning_skill(fake_tapes)
        response = client.post(
            "/api/skills-evaluator/revisions", json={"skill_id": skill_id}
        )
        assert response.status_code == 409

    def test_missing_skill_id_is_400(self, client):
        response = client.post(
            "/api/skills-evaluator/revisions",
            json={"candidate": {"skill_md": "# s"}},
        )
        assert response.status_code == 400

    def test_status_validation_and_404(self, client):
        assert (
            client.post(
                "/api/skills-evaluator/revisions/nope/status",
                json={"status": "maybe"},
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/api/skills-evaluator/revisions/nope/status",
                json={"status": "accepted"},
            ).status_code
            == 404
        )
        assert client.get("/api/skills-evaluator/revisions/nope").status_code == 404

    def test_list_requires_skill_id(self, client):
        assert client.get("/api/skills-evaluator/revisions").status_code == 400

    def test_list_by_skill_id(self, client, fake_tapes, revising_service):
        from skills_evaluator.tapes import SkillRecord

        fake_tapes.skills["sk-1"] = SkillRecord(
            id="sk-1",
            name="morning-catchup",
            description="Daily inbox triage",
            content="# Morning catchup\n\nSteps.",
            originating_session_ids=["s1"],
        )
        fake_tapes.transcripts["s1"] = "[user] catch me up\n"

        created = client.post(
            "/api/skills-evaluator/revisions", json={"skill_id": "sk-1"}
        )
        assert created.status_code == 201

        listed = client.get("/api/skills-evaluator/revisions", params={"skill_id": "sk-1"})
        assert listed.status_code == 200
        assert [r["skill_id"] for r in listed.json()["items"]] == ["sk-1"]

    def test_openapi_declares_revision_paths(self, client):
        document = client.get("/openapi").json()
        for path in (
            "/api/skills-evaluator/revisions",
            "/api/skills-evaluator/revisions/{revision_id}",
            "/api/skills-evaluator/revisions/{revision_id}/status",
        ):
            assert path in document["paths"]
        tables = document["x-tapes-cassette"]["tables"]
        assert {"name": "revisions"} in tables
