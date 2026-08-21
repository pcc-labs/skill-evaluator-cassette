from __future__ import annotations

import pytest

from skills_evaluator.pipeline import RULESET_VERSION
from skills_evaluator.service import build_queries, merge_evidence_sessions
from skills_evaluator.tapes import (
    SearchHit,
    SearchUnavailableError,
    SkillNotFoundError,
    SkillRecord,
)
from skills_evaluator.wire import Bundle, BundleFile, EvaluateRequest


def hit(session_id: str, score: float) -> SearchHit:
    return SearchHit(session_id=session_id, trace_id="t-" + session_id, score=score)


class TestQueryDerivation:
    def test_name_description_and_headings(self, service, fake_tapes, request_fixture):
        service.evaluate(request_fixture)
        assert fake_tapes.queries == [
            "morning-catchup Daily inbox triage",
            "morning-catchup Triage inbox",
            "morning-catchup Draft replies",
        ]

    def test_caps_and_fallback(self):
        skill = SkillRecord(id="sk-x", name="x", description="very long " * 50, content="")
        queries = build_queries(skill, "## One\n## Two\n## Three\n## Four\n")
        assert len(queries) <= 3
        assert all(len(q) <= 200 for q in queries)

        bare = SkillRecord(id="sk-solo", name="solo", description="", content="")
        assert build_queries(bare, "no headings") == ["solo"]


class TestWithEvidence:
    @pytest.fixture(autouse=True)
    def seed(self, fake_tapes):
        fake_tapes.hits["morning-catchup Daily inbox triage"] = [
            hit("s1", 0.9),
            hit("s2", 0.7),
        ]
        fake_tapes.transcripts["s1"] = "[user] catch me up on my inbox\n"
        fake_tapes.transcripts["s2"] = "[user] triage my email\n"

    def test_feeds_module_and_returns_judgment(
        self, service, fake_module, request_fixture
    ):
        response = service.evaluate(request_fixture)

        assert len(fake_module.calls) == 1
        call = fake_module.calls[0]
        assert call["skill_name"] == "morning-catchup"
        assert [sid for sid, _ in call["transcripts"]] == ["s1", "s2"]

        assert response.decision == "pass"
        assert response.mode == "llm"
        assert response.score == pytest.approx(0.9)
        assert response.ref is not None and response.ref.id == "prop-1"
        assert response.evaluator_version == "0.0.0-test+" + RULESET_VERSION
        assert [f.rule_id for f in response.findings] == ["clarity.ambiguous"]
        assert response.metrics.sessions_considered == 2
        assert response.metrics.provenance_sessions == 0
        assert response.metrics.judge_model == "fake/judge"

    def test_baseline_forwarded_for_updates(self, service, fake_module, request_fixture):
        request_fixture.baseline = Bundle(
            skill_md="# Old skill",
            files=[BundleFile(path="reference.md", content="old context")],
        )
        service.evaluate(request_fixture)
        baseline = fake_module.calls[0]["baseline_markdown"]
        assert baseline.startswith("# Old skill")
        assert "### Support file: reference.md\nold context" in baseline

    def test_score_clamped(self, service, fake_module, request_fixture):
        fake_module.prediction.score = 3.7
        assert service.evaluate(request_fixture).score == 1.0
        fake_module.prediction.score = "not a number"
        assert service.evaluate(request_fixture).score is None

    def test_skips_unloadable_sessions(self, service, fake_tapes, request_fixture):
        del fake_tapes.transcripts["s1"]
        response = service.evaluate(request_fixture)
        assert response.metrics.sessions_considered == 1

    def test_module_failure_propagates(self, service, fake_module, request_fixture):
        fake_module.error = RuntimeError("provider down")
        with pytest.raises(RuntimeError, match="provider down"):
            service.evaluate(request_fixture)


class TestSkillResolution:
    @pytest.fixture(autouse=True)
    def seed(self, fake_tapes):
        fake_tapes.skills["sk-1"] = SkillRecord(
            id="sk-1",
            name="morning-catchup",
            description="Daily inbox triage",
            content="# Morning catchup\n\n## Triage inbox\n\nSteps.",
            originating_session_ids=["prov-1", "prov-2"],
        )
        for sid in ("prov-1", "prov-2", "s1"):
            fake_tapes.transcripts[sid] = f"[user] work in {sid}\n"

    def test_resolves_content_and_identity(self, service, fake_module):
        request = EvaluateRequest(skill_id="sk-1")
        response = service.evaluate(request)

        call = fake_module.calls[0]
        assert call["skill_name"] == "morning-catchup"
        assert "# Morning catchup" in call["skill_markdown"]
        assert response.mode == "llm"

    def test_provenance_outranks_search(self, service, fake_tapes, fake_module):
        fake_tapes.hits["morning-catchup Daily inbox triage"] = [hit("s1", 0.99)]

        response = service.evaluate(EvaluateRequest(skill_id="sk-1"))

        # max_sessions is 2: both provenance sessions win over the 0.99 hit.
        assert [sid for sid, _ in fake_module.calls[0]["transcripts"]] == [
            "prov-1",
            "prov-2",
        ]
        assert response.metrics.provenance_sessions == 2

    def test_inline_bundle_wins_over_stored(self, service, fake_module):
        request = EvaluateRequest(
            skill_id="sk-1",
            candidate=Bundle(
                skill_md="# Edited draft",
                files=[BundleFile(path="scripts/check.py", content="assert safe")],
            ),
        )
        service.evaluate(request)
        rendered = fake_module.calls[0]["skill_markdown"]
        assert rendered.startswith("# Edited draft")
        assert "### Support file: scripts/check.py\nassert safe" in rendered

    def test_explicit_empty_candidate_does_not_fall_back(self, service, fake_module):
        service.evaluate(EvaluateRequest(skill_id="sk-1", candidate=Bundle()))
        assert fake_module.calls[0]["skill_markdown"] == ""

    def test_unknown_skill_raises(self, service):
        with pytest.raises(SkillNotFoundError):
            service.evaluate(EvaluateRequest(skill_id="nope"))

    def test_provenance_survives_search_unavailable(
        self, service, fake_tapes, fake_module
    ):
        fake_tapes.search_error = SearchUnavailableError("not configured")
        response = service.evaluate(EvaluateRequest(skill_id="sk-1"))

        assert response.mode == "llm"
        assert response.metrics.provenance_sessions == 2
        assert any("span search is not configured" in f.message for f in response.findings)


class TestWithoutEvidence:
    def test_no_matches_is_an_honest_pass(self, service, fake_module, request_fixture):
        response = service.evaluate(request_fixture)

        assert fake_module.calls == []
        assert response.decision == "pass"
        assert response.mode == "no-evidence"
        assert response.score is None
        assert response.ref is not None and response.ref.source == "test"
        assert [f.rule_id for f in response.findings] == ["evidence.none"]
        assert response.metrics.sessions_considered == 0

    def test_search_unavailable_degrades(self, service, fake_tapes, request_fixture):
        fake_tapes.search_error = SearchUnavailableError("not configured")
        response = service.evaluate(request_fixture)
        assert response.mode == "no-evidence"
        assert "span search is not configured" in response.findings[0].message

    def test_other_search_errors_propagate(self, service, fake_tapes, request_fixture):
        fake_tapes.search_error = ConnectionError("refused")
        with pytest.raises(ConnectionError):
            service.evaluate(request_fixture)


class TestEvidenceMerge:
    def test_provenance_first_then_ranked_search_dedup(self):
        chosen = merge_evidence_sessions(
            ["p1", "", "p2", "p1"],
            [hit("s-low", 0.5), hit("p1", 0.99), hit("s-high", 0.9)],
            10,
        )
        assert chosen == ["p1", "p2", "s-high", "s-low"]

    def test_cap_applies_after_merge(self):
        chosen = merge_evidence_sessions(["p1"], [hit("s1", 0.9)], 1)
        assert chosen == ["p1"]
