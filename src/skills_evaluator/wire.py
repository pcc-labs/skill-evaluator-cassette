"""The wire contract of POST /evaluate.

Field names are snake_case like the rest of the tapes surface; the OpenClaw
plugin maps the Gateway's camelCase hook event into this shape and the
response back out. The shapes mirror OpenClaw's
PluginHookSkillProposalEvaluateResult so the plugin's mapping stays a
rename, not a transformation.

Every response field is clamped inside OpenClaw's evaluator caps (200
findings, 8000-char summary, 4000-char message, ...) because OpenClaw's
normalization is all-or-nothing per field: one oversized array voids the
whole judgment.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# OpenClaw's normalization caps (service-evaluation.ts), minus headroom we
# choose for ourselves (50 findings is plenty for a useful review).
MAX_FINDINGS = 50
MAX_SUMMARY_CHARS = 8_000
MAX_MESSAGE_CHARS = 4_000
MAX_RULE_ID_CHARS = 256
MAX_REASON_CHARS = 2_000

SEVERITIES = ("info", "warn", "critical")


class ProposalRef(BaseModel):
    """Identity of the Skill Workshop proposal revision under evaluation."""

    id: str = ""
    kind: str = "create"
    revision: str = ""
    revision_sha256: str = ""


class SkillRef(BaseModel):
    """The skill a proposal targets."""

    name: str = ""
    description: str = ""


class BundleFile(BaseModel):
    """One support file in a proposal bundle."""

    path: str
    content: str


class Bundle(BaseModel):
    """A skill bundle snapshot: the SKILL.md body plus support files."""

    skill_md: str = ""
    files: list[BundleFile] = Field(default_factory=list)


class EvaluateRequest(BaseModel):
    """One skill proposal to judge against captured sessions."""

    proposal: ProposalRef = Field(default_factory=ProposalRef)
    skill: SkillRef = Field(default_factory=SkillRef)
    candidate: Bundle = Field(default_factory=Bundle)
    baseline: Bundle | None = None


class Finding(BaseModel):
    """One observation the judge attributed to a rule."""

    rule_id: str = "judge.finding"
    severity: str = "info"
    message: str
    file: str = ""
    line: int = 0


class Metrics(BaseModel):
    """Quantifies the evidence behind a judgment."""

    sessions_considered: int = 0
    spans_matched: int = 0
    mean_search_score: float = 0.0
    judge_model: str = ""
    transcript_chars: int = 0


class EvaluateResponse(BaseModel):
    """The judgment for one proposal revision.

    ``decision`` is only ever "pass" or "revise": a block gates apply inside
    OpenClaw, and an LLM judgment grounded in partial evidence has not
    earned that authority.
    """

    summary: str = ""
    findings: list[Finding] = Field(default_factory=list)
    metrics: Metrics = Field(default_factory=Metrics)
    decision: str = "pass"
    decision_reason: str = ""
    evaluator_version: str = ""
    mode: str = "llm"


def normalize_findings(raw: list[Finding]) -> list[Finding]:
    """Bounds a judge's findings: capped count, valid severities, non-empty
    messages, sane line numbers. Protects the whole judgment — OpenClaw
    discards a result whole when one field is out of bounds."""
    findings: list[Finding] = []
    for finding in raw:
        if len(findings) == MAX_FINDINGS:
            break
        message = finding.message.strip()
        if not message:
            continue
        severity = finding.severity.strip().lower()
        if severity == "warning":
            severity = "warn"
        if severity not in SEVERITIES:
            severity = "info"
        findings.append(
            Finding(
                rule_id=(finding.rule_id.strip() or "judge.finding")[:MAX_RULE_ID_CHARS],
                severity=severity,
                message=message[:MAX_MESSAGE_CHARS],
                file=finding.file.strip(),
                line=finding.line if finding.line >= 1 else 0,
            )
        )
    return findings


def normalize_decision(decision: str, findings: list[Finding]) -> str:
    """Keeps the judge inside this cassette's authority: only "pass" and
    "revise" leave the process; "block" preserves its intent as "revise";
    anything else falls back to what the findings imply."""
    match decision.strip().lower():
        case "pass":
            return "pass"
        case "revise" | "block":
            return "revise"
    if any(finding.severity != "info" for finding in findings):
        return "revise"
    return "pass"
