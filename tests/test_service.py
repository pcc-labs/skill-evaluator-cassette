from __future__ import annotations

import pytest

from skills_evaluator.pipeline import RULESET_VERSION
from skills_evaluator.service import build_queries, rank_sessions
from skills_evaluator.tapes import SearchHit, SearchUnavailableError
from skills_evaluator.wire import Bundle, EvaluateRequest, SkillRef


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
        request = EvaluateRequest(
            skill=SkillRef(name="x", description="very long " * 50),
            candidate=Bundle(skill_md="## One\n## Two\n## Three\n## Four\n"),
        )
        queries = build_queries(request)
        assert len(queries) <= 3
        assert all(len(q) <= 200 for q in queries)

        bare = EvaluateRequest(
            skill=SkillRef(name="solo"), candidate=Bundle(skill_md="no headings")
        )
        assert build_queries(bare) == ["solo"]


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
        assert response.evaluator_version == "0.0.0-test+" + RULESET_VERSION
        assert [f.rule_id for f in response.findings] == ["clarity.ambiguous"]
        assert response.findings[0].line == 12
        assert response.metrics.sessions_considered == 2
        assert response.metrics.spans_matched == 2
        assert response.metrics.judge_model == "fake/judge"

    def test_baseline_forwarded_for_updates(self, service, fake_module, request_fixture):
        request_fixture.proposal.kind = "update"
        request_fixture.baseline = Bundle(skill_md="# Old skill")
        service.evaluate(request_fixture)
        assert fake_module.calls[0]["baseline_markdown"] == "# Old skill"

    def test_ranks_by_best_score_and_caps_sessions(
        self, service, fake_tapes, fake_module, request_fixture
    ):
        fake_tapes.hits["morning-catchup Triage inbox"] = [hit("s3", 0.95)]
        fake_tapes.transcripts["s3"] = "[user] sort out my backlog\n"

        response = service.evaluate(request_fixture)

        # max_sessions is 2: s3 (0.95) and s1 (0.9) win; s2 is cut.
        assert [sid for sid, _ in fake_module.calls[0]["transcripts"]] == ["s3", "s1"]
        assert response.metrics.sessions_considered == 2

    def test_skips_unloadable_sessions(self, service, fake_tapes, request_fixture):
        del fake_tapes.transcripts["s1"]
        response = service.evaluate(request_fixture)
        assert response.metrics.sessions_considered == 1

    def test_drops_hits_without_session_id(self, service, fake_tapes, request_fixture):
        fake_tapes.hits["morning-catchup Daily inbox triage"] = [
            hit("", 0.99),
            hit("s1", 0.9),
        ]
        response = service.evaluate(request_fixture)
        assert response.metrics.sessions_considered == 1

    def test_module_failure_propagates(self, service, fake_module, request_fixture):
        fake_module.error = RuntimeError("provider down")
        with pytest.raises(RuntimeError, match="provider down"):
            service.evaluate(request_fixture)


class TestWithoutEvidence:
    def test_no_matches_is_an_honest_pass(self, service, fake_module, request_fixture):
        response = service.evaluate(request_fixture)

        assert fake_module.calls == []
        assert response.decision == "pass"
        assert response.mode == "no-evidence"
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


class TestRanking:
    def test_deterministic_order(self):
        hits = [hit("b", 0.5), hit("a", 0.5), hit("c", 0.9)]
        assert rank_sessions(hits, 10) == ["c", "a", "b"]
