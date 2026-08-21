"""The DSPy evaluation pipeline: trace triage, then skill judgment.

This is the two-stage shape from the DSPy exploration notes, applied to
evaluation rather than generation:

1. **Triage** (``SessionTriage``) — agent traces carry a lot of implicit
   supervision: tool-call errors, retries, loops, corrections, abandonment.
   A ChainOfThought pass over each session transcript recovers it as weak
   labels (outcome, failure mode, cited evidence) — so the judge reads
   annotated evidence, not raw logs.
2. **Judgment** (``SkillJudgment``) — one typed ChainOfThought call over the
   proposed skill, the baseline (for updates), and the annotated evidence,
   returning a structured verdict: summary, pass/revise decision, and
   attributed findings.

Because both stages are DSPy modules with typed signatures, the follow-on
from the notes — GEPA evolving the skill text (or these instructions
themselves) against a feedback metric built from triage labels — plugs in
without restructuring: this module IS the program GEPA would compile.
"""

from __future__ import annotations

from typing import Literal

import dspy
from pydantic import BaseModel

# The judgment rubric version, reported in evaluator_version so a rubric
# change is visible in stored evaluations even when the package version
# does not move.
RULESET_VERSION = "dspy-rules-2026-08"

# Prompt caps: the candidate must fit alongside the evidence without
# starving it; the baseline is context, not the subject, so it gets less.
MAX_CANDIDATE_PROMPT_CHARS = 20_000
MAX_BASELINE_PROMPT_CHARS = 8_000
MAX_TRANSCRIPT_CHARS = 30_000


class JudgeFinding(BaseModel):
    """One finding the judge attributes to a rule.

    rule_id is one of: accuracy.contradicted, completeness.missing-step,
    evidence.unsupported, safety.risky-instruction, clarity.ambiguous.
    """

    rule_id: str
    severity: Literal["info", "warn", "critical"]
    message: str
    file: str = ""
    line: int = 0


class SessionTriage(dspy.Signature):
    """Classify how a captured agent session actually went, from its
    transcript. [user] lines are the human's prompts, [assistant] lines the
    agent's responses, [tools] lines summarize tool usage. Look for the
    implicit signals: errors, retries, loops, user corrections, abandoned
    work, or a clean completion."""

    transcript: str = dspy.InputField()
    outcome: Literal["success", "partial", "failure"] = dspy.OutputField()
    failure_mode: str = dspy.OutputField(
        desc="short label for what went wrong; 'none' when the session succeeded"
    )
    evidence: str = dspy.OutputField(
        desc="one sentence citing the transcript for the outcome"
    )


class SkillJudgment(dspy.Signature):
    """Judge a proposed agent skill. Two grounds, use whichever are present:

    Session evidence (when provided): real captured sessions.
    - accuracy: does an instruction contradict what actually worked?
    - completeness: do the sessions show steps or recoveries the skill omits?
    - evidence: does the skill claim behavior the sessions do not support?
    - safety: does it instruct something the sessions show to be risky?
    - clarity: are the instructions concrete enough for an agent to follow?

    Eval spec (when provided): the skill's own stated criteria. Check EVERY
    criterion against the skill document one by one; higher weight matters
    more. A criterion the document does not satisfy IS a finding — use the
    criterion's id as the rule_id, severity warn for weight >= 2 (critical
    only for safety), info otherwise. A one-line skill rarely satisfies a
    multi-criterion spec; do not wave it through.

    When session evidence is '(none)', judge the skill on the spec and its
    own coherence — do NOT penalize the skill for the absence of evidence;
    that absence is reported elsewhere. Report at most 8 findings, most
    important first; do not invent problems. Decide "pass" when the skill is
    usable as-is, "revise" when the findings warrant changes."""

    skill_name: str = dspy.InputField()
    proposal_kind: Literal["create", "update"] = dspy.InputField()
    skill_markdown: str = dspy.InputField(desc="the proposed SKILL.md body")
    baseline_markdown: str = dspy.InputField(
        desc="the live skill an update proposal replaces; empty for create"
    )
    session_evidence: str = dspy.InputField(
        desc="session transcripts with triage verdicts; '(none)' when no "
        "captured sessions relate to this skill"
    )
    eval_spec: str = dspy.InputField(
        desc="the skill's eval criteria, one per line with weights; "
        "'(none)' when no spec exists"
    )
    summary: str = dspy.OutputField(
        desc="2-4 sentences: overall verdict and how well the evidence covers the skill"
    )
    decision: Literal["pass", "revise"] = dspy.OutputField()
    decision_reason: str = dspy.OutputField(desc="one sentence")
    score: float = dspy.OutputField(
        desc="quality estimate in [0.0, 1.0]: 1.0 = accurate, complete, and "
        "fully supported by the evidence; below 0.5 = the findings warrant revision"
    )
    findings: list[JudgeFinding] = dspy.OutputField()


class SkillRevisionProposal(dspy.Signature):
    """Rewrite a skill so the evaluation findings are addressed, grounded
    ONLY in what the session evidence and the eval criteria support. Keep
    everything that was right; change only what a finding, a criterion, or
    the evidence justifies; never invent steps nothing supports. Preserve
    the document's markdown structure and voice.

    The revised document is an instruction for an agent, NOT an evaluation
    report: never write meta-commentary about evidence, evaluation, or
    assessability into it (no "insufficient evidence to...", no "cannot be
    evaluated..."). If a finding cannot be addressed from the available
    grounds, leave that part of the skill unchanged and say so in the
    rationale instead. The rationale must attribute every material change
    to a finding, a criterion, or specific evidence."""

    skill_name: str = dspy.InputField()
    skill_markdown: str = dspy.InputField(desc="the current SKILL.md document")
    support_files: str = dspy.InputField(
        desc="read-only support-file context; '(none)' when the bundle has none"
    )
    findings: str = dspy.InputField(desc="the evaluation's findings, one per line")
    session_evidence: str = dspy.InputField(
        desc="session transcripts with triage annotations; '(none)' when absent"
    )
    eval_spec: str = dspy.InputField(
        desc="the skill's eval criteria, one per line; '(none)' when absent"
    )
    revised_markdown: str = dspy.OutputField(desc="the complete revised document")
    rationale: str = dspy.OutputField(
        desc="2-4 sentences attributing each material change to a finding, "
        "criterion, or evidence"
    )


class SpecCriterion(BaseModel):
    """One checkable expectation for the skill."""

    id: str
    kind: Literal["structure", "content", "output-property"]
    description: str
    weight: int


class SpecCase(BaseModel):
    """One concrete scenario with observable expectations."""

    scenario: str
    expect: list[str]


class EvalSpecGeneration(dspy.Signature):
    """Derive an eval spec — checkable criteria and concrete test cases —
    from a skill document's own claims, plus observed behavior when session
    evidence exists.

    Criteria kinds: 'structure' (the document states its triggers, inputs,
    preconditions, steps), 'content' (instructions are concrete, imperative,
    non-contradictory), 'output-property' (verifiable properties of what an
    agent following the skill should produce — the domain's formal rules,
    e.g. a haiku's 5-7-5). Derive output-properties from what the skill
    implies ('classical haiku' implies syllable structure) and from failure
    modes visible in the evidence. 4-8 criteria, weights 1-3 (3 = essential),
    each independently checkable; 1-3 cases with observable expectations."""

    skill_name: str = dspy.InputField()
    skill_description: str = dspy.InputField()
    skill_markdown: str = dspy.InputField()
    session_evidence: str = dspy.InputField(
        desc="triaged transcripts when any exist; '(none)' otherwise"
    )
    criteria: list[SpecCriterion] = dspy.OutputField()
    cases: list[SpecCase] = dspy.OutputField()


class EvalSpecGenerator(dspy.Module):
    """Drafts a skill's intrinsic metric. Its output is a starting point:
    humans edit the stored spec, and edits win permanently."""

    def __init__(self) -> None:
        super().__init__()
        self.generate = dspy.ChainOfThought(EvalSpecGeneration)

    def forward(
        self,
        skill_name: str,
        skill_description: str,
        skill_markdown: str,
        session_evidence: str,
    ) -> dspy.Prediction:
        proposal = self.generate(
            skill_name=skill_name,
            skill_description=skill_description,
            skill_markdown=skill_markdown[:MAX_CANDIDATE_PROMPT_CHARS],
            session_evidence=session_evidence or "(none)",
        )
        return dspy.Prediction(criteria=proposal.criteria, cases=proposal.cases)


class SkillReviser(dspy.Module):
    """Proposes a revision of a skill from its evaluation. Kept separate
    from SkillEvaluator so evaluation stays cheap and pure; revision is an
    explicit, stored act — the unit the accept/reject corpus is built from."""

    def __init__(self) -> None:
        super().__init__()
        self.revise = dspy.ChainOfThought(SkillRevisionProposal)

    def forward(
        self,
        skill_name: str,
        skill_markdown: str,
        support_files: str,
        findings: list[JudgeFinding],
        session_evidence: str,
        eval_spec: str = "",
    ) -> dspy.Prediction:
        rendered = "\n".join(
            f"- [{f.severity}] {f.rule_id}: {f.message}" for f in findings
        ) or "- (no findings; tighten wording only where the grounds justify it)"
        proposal = self.revise(
            skill_name=skill_name,
            skill_markdown=skill_markdown[:MAX_CANDIDATE_PROMPT_CHARS],
            support_files=support_files[:MAX_CANDIDATE_PROMPT_CHARS] or "(none)",
            findings=rendered,
            session_evidence=session_evidence or "(none)",
            eval_spec=eval_spec or "(none)",
        )
        return dspy.Prediction(
            revised_markdown=proposal.revised_markdown,
            rationale=proposal.rationale,
        )


class SkillEvaluator(dspy.Module):
    """Triage each session, then judge the skill against the annotated
    evidence. Returns a Prediction carrying the judgment fields plus the
    per-session triage verdicts (useful metrics, and the raw material for a
    future GEPA feedback function)."""

    def __init__(self) -> None:
        super().__init__()
        self.triage = dspy.ChainOfThought(SessionTriage)
        self.judge = dspy.ChainOfThought(SkillJudgment)

    def forward(
        self,
        skill_name: str,
        skill_markdown: str,
        baseline_markdown: str,
        transcripts: list[tuple[str, str]],
        eval_spec: str = "",
    ) -> dspy.Prediction:
        triages: list[dict[str, str]] = []
        annotated: list[str] = []
        for session_id, transcript in transcripts:
            verdict = self.triage(transcript=transcript)
            triages.append(
                {
                    "session_id": session_id,
                    "outcome": verdict.outcome,
                    "failure_mode": verdict.failure_mode,
                    "evidence": verdict.evidence,
                }
            )
            annotated.append(
                f"### Session {session_id} — triage: {verdict.outcome}"
                f" (failure_mode: {verdict.failure_mode}; {verdict.evidence})\n"
                f"{transcript}"
            )

        judgment = self.judge(
            skill_name=skill_name,
            # An update is fully encoded by the baseline's presence; there
            # is no separate proposal-kind concept on the wire.
            proposal_kind="update" if baseline_markdown.strip() else "create",
            skill_markdown=skill_markdown[:MAX_CANDIDATE_PROMPT_CHARS],
            baseline_markdown=baseline_markdown[:MAX_BASELINE_PROMPT_CHARS],
            session_evidence="\n---\n".join(annotated) or "(none)",
            eval_spec=eval_spec or "(none)",
        )
        return dspy.Prediction(
            summary=judgment.summary,
            decision=judgment.decision,
            decision_reason=judgment.decision_reason,
            score=judgment.score,
            findings=judgment.findings,
            triages=triages,
        )
