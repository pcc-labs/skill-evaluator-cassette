from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.conftest import FakeSpecGenerator, FakeTapes
from skills_evaluator.server import create_app
from skills_evaluator.service import render_spec
from skills_evaluator.store import EditedSpecError, MemoryRevisionStore
from skills_evaluator.tapes import SearchHit, SkillRecord
from skills_evaluator.wire import EvaluateRequest

HAIKU_SKILL_ID = "sk-haiku"


def seed_haiku_skill(fake_tapes: FakeTapes) -> str:
    fake_tapes.skills[HAIKU_SKILL_ID] = SkillRecord(
        id=HAIKU_SKILL_ID,
        name="haiku-writer",
        description="Haiku writer skill",
        content="Write a classical haiku given the provided inputs.",
    )
    return HAIKU_SKILL_ID


def haiku_request() -> EvaluateRequest:
    # No inline candidate: the stored content itself is judged.
    return EvaluateRequest(skill_id=HAIKU_SKILL_ID)


@pytest.fixture
def spec_service(service, fake_module, fake_tapes):
    seed_haiku_skill(fake_tapes)
    service.store = MemoryRevisionStore()
    service.spec_generator = FakeSpecGenerator()
    return service


class TestRelevanceGate:
    def test_junk_hits_never_become_evidence(self, spec_service, fake_tapes):
        # The haiku regression: one weak inbox hit used to masquerade as
        # evidence and the skill got blamed for it.
        fake_tapes.hits["haiku-writer Haiku writer skill"] = [
            SearchHit(session_id="inbox-1", trace_id="t", score=0.22)
        ]
        fake_tapes.transcripts["inbox-1"] = "[user] catch me up on my inbox\n"

        response = spec_service.evaluate(haiku_request())

        assert response.mode == "spec", "gated out, judged on the spec instead"
        assert response.metrics.sessions_considered == 0
        assert response.metrics.spans_matched == 0
        assert response.metrics.spans_gated == 1

    def test_strong_hits_pass_the_gate(self, spec_service, fake_tapes, fake_module):
        fake_tapes.hits["haiku-writer Haiku writer skill"] = [
            SearchHit(session_id="s1", trace_id="t", score=0.6)
        ]
        fake_tapes.transcripts["s1"] = "[user] write me a haiku about winter\n"

        response = spec_service.evaluate(haiku_request())
        assert response.mode == "llm"
        assert response.metrics.spans_gated == 0


class TestSpecMode:
    def test_autogenerates_stores_and_judges_on_spec(
        self, spec_service, fake_module
    ):
        response = spec_service.evaluate(haiku_request())

        assert response.mode == "spec"
        assert response.metrics.spec_criteria == 2
        # The module saw the rendered criteria, not empty evidence framing.
        assert "inputs-defined" in fake_module.calls[0]["eval_spec"]
        assert fake_module.calls[0]["transcripts"] == []
        # The generated spec was stored, keyed by the tapes skill id.
        stored = spec_service.store.get_eval_for_skill(HAIKU_SKILL_ID)
        assert stored is not None and stored.origin == "generated"
        assert stored.source_sha256

    def test_spec_score_is_capped(self, spec_service, fake_module):
        fake_module.prediction.score = 1.0
        response = spec_service.evaluate(haiku_request())
        assert response.score == pytest.approx(0.9)

    def test_autogenerate_off_falls_back_to_no_evidence(self, spec_service):
        spec_service.config.spec_autogenerate = False
        response = spec_service.evaluate(haiku_request())
        assert response.mode == "no-evidence"
        assert response.score is None

    def test_stored_spec_joins_llm_mode_judgment(
        self, spec_service, fake_tapes, fake_module
    ):
        spec_service.generate_eval(haiku_request(), force=False)
        fake_tapes.hits["haiku-writer Haiku writer skill"] = [
            SearchHit(session_id="s1", trace_id="t", score=0.6)
        ]
        fake_tapes.transcripts["s1"] = "[user] haiku please\n"

        response = spec_service.evaluate(haiku_request())
        assert response.mode == "llm"
        assert "inputs-defined" in fake_module.calls[-1]["eval_spec"]
        assert response.metrics.spec_criteria == 2

    def test_generation_failure_degrades_to_no_evidence(self, spec_service):
        class Boom:
            def __call__(self, **kwargs):
                raise RuntimeError("generator down")

        spec_service.spec_generator = Boom()
        response = spec_service.evaluate(haiku_request())
        assert response.mode == "no-evidence"


class TestSpecStore:
    def test_edited_specs_are_never_regenerated(self, spec_service):
        record = spec_service.generate_eval(haiku_request(), force=False)
        spec_service.store.update_eval_spec(
            record.id, {"criteria": [{"id": "human", "kind": "content",
                                      "description": "d", "weight": 1}], "cases": []}
        )

        with pytest.raises(EditedSpecError):
            spec_service.generate_eval(haiku_request(), force=True)

        kept = spec_service.store.get_eval_for_skill(HAIKU_SKILL_ID)
        assert kept.origin == "edited"
        assert kept.spec["criteria"][0]["id"] == "human"

    def test_existing_generated_spec_returned_unless_forced(self, spec_service):
        first = spec_service.generate_eval(haiku_request(), force=False)
        again = spec_service.generate_eval(haiku_request(), force=False)
        assert again.id == first.id
        assert len(spec_service.spec_generator.calls) == 1

        forced = spec_service.generate_eval(haiku_request(), force=True)
        assert forced.id == first.id, "identity survives regeneration"
        assert len(spec_service.spec_generator.calls) == 2

    def test_generated_spec_refreshes_when_stored_skill_changes(
        self, spec_service, fake_tapes
    ):
        first = spec_service.generate_eval(haiku_request(), force=False)
        fake_tapes.skills[HAIKU_SKILL_ID].content = "Write a modern free-verse poem."

        spec_service.evaluate(haiku_request())

        refreshed = spec_service.store.get_eval_for_skill(HAIKU_SKILL_ID)
        assert refreshed.id == first.id
        assert refreshed.source_sha256 != first.source_sha256
        assert len(spec_service.spec_generator.calls) == 2

    def test_edited_spec_remains_canonical_after_skill_changes(
        self, spec_service, fake_tapes
    ):
        record = spec_service.generate_eval(haiku_request(), force=False)
        spec_service.store.update_eval_spec(
            record.id,
            {"criteria": [{"id": "human", "kind": "content", "description": "d", "weight": 1}], "cases": []},
        )
        fake_tapes.skills[HAIKU_SKILL_ID].content = "Changed content."

        spec_service.evaluate(haiku_request())

        kept = spec_service.store.get_eval_for_skill(HAIKU_SKILL_ID)
        assert kept.origin == "edited"
        assert kept.spec["criteria"][0]["id"] == "human"
        assert len(spec_service.spec_generator.calls) == 1


class TestSpecGroundedRevision:
    def test_revise_works_from_spec_alone(self, spec_service, request_fixture):
        from tests.test_revisions import FakeReviser

        reviser = FakeReviser()
        spec_service.reviser = reviser

        record = spec_service.revise(haiku_request())
        assert record.status == "proposed"
        assert record.evaluation["mode"] == "spec"
        assert "inputs-defined" in reviser.calls[0]["eval_spec"]


class TestEvalEndpoints:
    @pytest.fixture
    def client(self, spec_service) -> TestClient:
        return TestClient(create_app(spec_service))

    def test_generate_get_edit_lifecycle(self, client):
        created = client.post(
            "/api/skills-evaluator/evals",
            json={"skill_id": HAIKU_SKILL_ID},
        )
        assert created.status_code == 201
        body = created.json()
        assert body["origin"] == "generated"
        assert body["skill_id"] == HAIKU_SKILL_ID
        assert body["source_sha256"]
        assert len(body["spec"]["criteria"]) == 2
        eval_id = body["id"]

        by_skill = client.get(
            "/api/skills-evaluator/evals", params={"skill_id": HAIKU_SKILL_ID}
        )
        assert by_skill.status_code == 200
        assert by_skill.json()["id"] == eval_id

        update = {"spec": {"criteria": [
            {"id": "syllables", "kind": "output-property",
             "description": "5-7-5 structure", "weight": 3}
        ], "cases": []}}
        edited = client.put(
            f"/api/skills-evaluator/evals/{eval_id}",
            json=update,
        )
        assert edited.status_code == 200
        assert edited.json()["origin"] == "edited"
        assert edited.json()["spec_sha256"] != body["spec_sha256"]

        regen = client.post(
            "/api/skills-evaluator/evals?force=true",
            json={"skill_id": HAIKU_SKILL_ID},
        )
        assert regen.status_code == 409

    def test_generate_requires_skill_id(self, client):
        response = client.post(
            "/api/skills-evaluator/evals",
            json={"candidate": {"skill_md": "Write a classical haiku."}},
        )
        assert response.status_code == 400

    def test_get_unknown_is_404(self, client):
        assert client.get("/api/skills-evaluator/evals/nope").status_code == 404
        assert (
            client.get(
                "/api/skills-evaluator/evals", params={"skill_id": "ghost"}
            ).status_code
            == 404
        )

    def test_openapi_declares_eval_paths(self, client):
        document = client.get("/openapi").json()
        assert "/api/skills-evaluator/evals" in document["paths"]
        assert "/api/skills-evaluator/evals/{eval_id}" in document["paths"]
        tables = document["x-tapes-cassette"]["tables"]
        assert {"name": "evals"} in tables


def test_render_spec_format():
    text = render_spec(
        {
            "criteria": [
                {"id": "a", "kind": "structure", "description": "states inputs", "weight": 3}
            ],
            "cases": [{"scenario": "winter haiku", "expect": ["3 lines", "5-7-5"]}],
        }
    )
    assert "- [structure, weight=3] a: states inputs" in text
    assert "- case: winter haiku → expect: 3 lines; 5-7-5" in text
