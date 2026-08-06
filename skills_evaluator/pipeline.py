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
    """Judge a proposed agent skill against evidence from real captured
    sessions. Judge only what the skill text and the evidence support:

    - accuracy: does an instruction contradict what actually worked?
    - completeness: do the sessions show steps or recoveries the skill omits?
    - evidence: does the skill claim behavior the sessions do not support?
    - safety: does it instruct something the sessions show to be risky?
    - clarity: are the instructions concrete enough for an agent to follow?

    Report at most 8 findings, most important first; do not invent problems.
    If the evidence covers the skill only partially, say so in the summary
    instead of guessing. Decide "pass" when the skill is usable as-is,
    "revise" when the findings warrant changes."""

    skill_name: str = dspy.InputField()
    proposal_kind: Literal["create", "update"] = dspy.InputField()
    skill_markdown: str = dspy.InputField(desc="the proposed SKILL.md body")
    baseline_markdown: str = dspy.InputField(
        desc="the live skill an update proposal replaces; empty for create"
    )
    session_evidence: str = dspy.InputField(
        desc="session transcripts, each annotated with a triage verdict"
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
    ONLY in what the session evidence supports. Keep everything that was
    right; change only what a finding or the evidence justifies; never
    invent steps the evidence does not show. Preserve the document's
    markdown structure and voice. The rationale must attribute every
    material change to a finding or to specific evidence."""

    skill_name: str = dspy.InputField()
    skill_markdown: str = dspy.InputField(desc="the current skill document")
    findings: str = dspy.InputField(desc="the evaluation's findings, one per line")
    session_evidence: str = dspy.InputField(
        desc="session transcripts with triage annotations"
    )
    revised_markdown: str = dspy.OutputField(desc="the complete revised document")
    rationale: str = dspy.OutputField(
        desc="2-4 sentences attributing each material change to a finding or evidence"
    )


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
        findings: list[JudgeFinding],
        session_evidence: str,
    ) -> dspy.Prediction:
        rendered = "\n".join(
            f"- [{f.severity}] {f.rule_id}: {f.message}" for f in findings
        ) or "- (no findings; tighten wording only where the evidence justifies it)"
        proposal = self.revise(
            skill_name=skill_name,
            skill_markdown=skill_markdown[:MAX_CANDIDATE_PROMPT_CHARS],
            findings=rendered,
            session_evidence=session_evidence,
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
            session_evidence="\n---\n".join(annotated),
        )
        return dspy.Prediction(
            summary=judgment.summary,
            decision=judgment.decision,
            decision_reason=judgment.decision_reason,
            score=judgment.score,
            findings=judgment.findings,
            triages=triages,
        )
