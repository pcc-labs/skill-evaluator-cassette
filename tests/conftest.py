from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from skills_evaluator.pipeline import JudgeFinding
from skills_evaluator.service import EvaluationService, ServiceConfig
from skills_evaluator.tapes import SearchHit, SkillNotFoundError, SkillRecord
from skills_evaluator.wire import Bundle, EvaluateRequest, Ref

MORNING_SKILL_ID = "sk-morning"
MORNING_SKILL_MD = (
    "# Morning catchup\n\n## Triage inbox\n\nSteps.\n\n"
    "## Draft replies\n\nMore steps."
)


def seed_morning_skill(fake_tapes: "FakeTapes") -> str:
    """Seeds the canonical test skill row. Every request names a stored
    skill now, so tests seed the row the way the plugin would have created
    it (POST /v1/skills) before evaluating."""
    fake_tapes.skills[MORNING_SKILL_ID] = SkillRecord(
        id=MORNING_SKILL_ID,
        name="morning-catchup",
        description="Daily inbox triage",
        content=MORNING_SKILL_MD,
    )
    return MORNING_SKILL_ID


@dataclass
class FakeTapes:
    """Stands in for TapesClient: canned hits per query, canned transcripts,
    canned skills."""

    hits: dict[str, list[SearchHit]] = field(default_factory=dict)
    transcripts: dict[str, str] = field(default_factory=dict)
    skills: dict[str, SkillRecord] = field(default_factory=dict)
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

    def get_skill(self, skill_id: str) -> SkillRecord:
        if skill_id not in self.skills:
            raise SkillNotFoundError(f"skill {skill_id} not found")
        return self.skills[skill_id]


@dataclass
class Prediction:
    summary: str = "The skill matches how the sessions actually went."
    decision: str = "pass"
    decision_reason: str = "supported by the evidence"
    score: float = 0.9
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


@dataclass
class FakeSpecGenerator:
    """Stands in for the DSPy EvalSpecGenerator module."""

    calls: list[dict] = field(default_factory=list)

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        from skills_evaluator.pipeline import SpecCase, SpecCriterion

        @dataclass
        class Proposal:
            criteria: list = field(
                default_factory=lambda: [
                    SpecCriterion(
                        id="inputs-defined",
                        kind="structure",
                        description="The skill names the inputs it expects.",
                        weight=3,
                    ),
                    SpecCriterion(
                        id="steps-concrete",
                        kind="content",
                        description="Steps are imperative and concrete.",
                        weight=2,
                    ),
                ]
            )
            cases: list = field(
                default_factory=lambda: [
                    SpecCase(scenario="typical run", expect=["follows every step"])
                ]
            )

        return Proposal()


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
def request_fixture(fake_tapes: FakeTapes) -> EvaluateRequest:
    return EvaluateRequest(
        ref=Ref(source="test", id="prop-1", revision="v1"),
        skill_id=seed_morning_skill(fake_tapes),
        candidate=Bundle(skill_md=MORNING_SKILL_MD),
    )
