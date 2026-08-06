from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from skills_evaluator.pipeline import JudgeFinding
from skills_evaluator.service import EvaluationService, ServiceConfig
from skills_evaluator.tapes import SearchHit
from skills_evaluator.wire import Bundle, EvaluateRequest, ProposalRef, SkillRef


@dataclass
class FakeTapes:
    """Stands in for TapesClient: canned hits per query, canned transcripts."""

    hits: dict[str, list[SearchHit]] = field(default_factory=dict)
    transcripts: dict[str, str] = field(default_factory=dict)
    search_error: Exception | None = None
    queries: list[str] = field(default_factory=list)

    def search_spans(self, query: str, top_k: int) -> list[SearchHit]:
        self.queries.append(query)
        if self.search_error is not None:
            raise self.search_error
        return self.hits.get(query, [])

    def session_transcript(self, session_id: str) -> str:
        if session_id not in self.transcripts:
            raise ValueError(f"unknown session {session_id}")
        return self.transcripts[session_id]


@dataclass
class Prediction:
    summary: str = "The skill matches how the sessions actually went."
    decision: str = "pass"
    decision_reason: str = "supported by the evidence"
    findings: list = field(default_factory=list)
    triages: list = field(default_factory=list)


@dataclass
class FakeModule:
    """Stands in for the DSPy SkillEvaluator module."""

    prediction: Prediction = field(default_factory=Prediction)
    calls: list[dict] = field(default_factory=list)
    error: Exception | None = None

    def __call__(self, **kwargs) -> Prediction:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.prediction


@pytest.fixture
def fake_tapes() -> FakeTapes:
    return FakeTapes()


@pytest.fixture
def fake_module() -> FakeModule:
    return FakeModule(
        prediction=Prediction(
            findings=[
                JudgeFinding(
                    rule_id="clarity.ambiguous",
                    severity="info",
                    message="Step 3 could name the exact command.",
                    file="SKILL.md",
                    line=12,
                )
            ]
        )
    )


@pytest.fixture
def service(fake_tapes: FakeTapes, fake_module: FakeModule) -> EvaluationService:
    return EvaluationService(
        tapes=fake_tapes,  # type: ignore[arg-type]
        module=fake_module,
        config=ServiceConfig(
            top_k=5, max_sessions=2, judge_model="fake/judge", version="0.0.0-test"
        ),
    )


@pytest.fixture
def request_fixture() -> EvaluateRequest:
    return EvaluateRequest(
        proposal=ProposalRef(id="prop-1", kind="create", revision="v1"),
        skill=SkillRef(name="morning-catchup", description="Daily inbox triage"),
        candidate=Bundle(
            skill_md=(
                "# Morning catchup\n\n## Triage inbox\n\nSteps.\n\n"
                "## Draft replies\n\nMore steps."
            )
        ),
    )
