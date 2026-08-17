"""The wire contract of POST /evaluate.

The contract is host-neutral and anchored on the tapes skills table: every
request names a stored skill row by ``skill_id`` — identity, provenance,
and history all hang off that row. The optional ``candidate`` bundle is the
*proposed* document for that row (a workshop draft, an edit under review);
when absent, the stored content itself is judged. ``baseline`` frames the
evaluation as an update replacing it, and ``ref`` is opaque correlation
metadata echoed back verbatim.

Hosts conform through thin adapters. Paper platform callers already hold a
skill_id. The OpenClaw Gateway plugin evaluates greenfield workshop
proposals with no tapes row yet — so it creates one first through the
skills cassette API (``POST /v1/cassettes/skills``) and evaluates against that id.

Field names are snake_case like the rest of the tapes surface. Every
response field is bounded: the caps happen to sit inside OpenClaw's
evaluator limits (200 findings, 8000-char summary, ...) so that adapter
stays a rename, but they are just sane limits for any host.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

MAX_FINDINGS = 50
MAX_SUMMARY_CHARS = 8_000
MAX_MESSAGE_CHARS = 4_000
MAX_RULE_ID_CHARS = 256
MAX_REASON_CHARS = 2_000

SEVERITIES = ("info", "warn", "critical")


class Ref(BaseModel):
    """Opaque correlation metadata: who is asking, about what revision.
    Echoed back verbatim in the response; never interpreted."""

    source: str = ""
    id: str = ""
    revision: str = ""
    revision_sha256: str = ""


class BundleFile(BaseModel):
    """One support file in a skill bundle."""

    path: str
    content: str


class Bundle(BaseModel):
    """A skill bundle snapshot: the SKILL.md body plus support files."""

    skill_md: str = ""
    files: list[BundleFile] = Field(default_factory=list)


class EvaluateRequest(BaseModel):
    """One skill to judge against captured sessions.

    ``skill_id`` is required: the tapes skill row is the unit of identity,
    and the cassette resolves it itself — name, description, content, and
    the provenance sessions that seed the evidence. ``candidate`` optionally
    carries the proposed document replacing the stored content (a workshop
    draft under review); when ``baseline`` is present the evaluation is
    framed as an update replacing it. There is no separate "kind" field.
    """

    ref: Ref | None = None
    skill_id: str
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
    provenance_sessions: int = 0
    spans_matched: int = 0
    spans_gated: int = 0
    mean_search_score: float = 0.0
    spec_criteria: int = 0
    judge_model: str = ""
    transcript_chars: int = 0


class EvaluateResponse(BaseModel):
    """The judgment for one skill document.

    ``decision`` is only ever "pass" or "revise": whether either gates
    anything is the host's decision, not this evaluator's. ``score`` is the
    judge's 0..1 quality estimate — the shape a GEPA-style optimizer wants
    as its metric — and is null when there was no evidence to score
    against.
    """

    ref: Ref | None = None
    summary: str = ""
    findings: list[Finding] = Field(default_factory=list)
    metrics: Metrics = Field(default_factory=Metrics)
    decision: str = "pass"
    decision_reason: str = ""
    score: float | None = None
    evaluator_version: str = ""
    mode: str = "llm"


class EvalCriterion(BaseModel):
    """One checkable expectation an eval spec holds a skill to."""

    id: str
    kind: str = "content"  # structure | content | output-property
    description: str
    weight: int = 1


class EvalCase(BaseModel):
    """One concrete scenario with the properties a skill-following agent's
    output should satisfy."""

    scenario: str
    expect: list[str] = Field(default_factory=list)


class EvalSpec(BaseModel):
    """A skill's intrinsic metric: criteria (and optional cases) it can be
    judged against with zero session history. Human-editable — editing the
    spec is editing the metric."""

    criteria: list[EvalCriterion] = Field(default_factory=list)
    cases: list[EvalCase] = Field(default_factory=list)


class EvalResponse(BaseModel):
    """One stored eval spec and its provenance."""

    id: str
    skill_id: str = ""
    skill_name: str = ""
    origin: str = "generated"  # generated | edited
    spec: EvalSpec = Field(default_factory=EvalSpec)
    spec_sha256: str = ""
    created_at: str = ""
    updated_at: str = ""


class EvalUpdateRequest(BaseModel):
    """A human edit: the replacement spec. Origin flips to `edited` and the
    spec is never silently regenerated afterwards."""

    spec: EvalSpec


class RevisionStatusRequest(BaseModel):
    """The status hook body: a host reporting what a human decided about a
    proposed revision. This is the labeling event the corpus is built from."""

    status: str
    reason: str = ""


class RevisionResponse(BaseModel):
    """One stored revision: the proposed rewrite, the evaluation that
    motivated it, and its lifecycle status."""

    id: str
    skill_id: str = ""
    ref: Ref | None = None
    skill_name: str = ""
    status: str = "proposed"
    status_reason: str = ""
    revised_skill_md: str = ""
    rationale: str = ""
    evaluation: EvaluateResponse = Field(default_factory=EvaluateResponse)
    created_at: str = ""
    decided_at: str | None = None


class RevisionListResponse(BaseModel):
    """Revisions for one skill, newest first."""

    items: list[RevisionResponse] = Field(default_factory=list)


def normalize_findings(raw: list[Finding]) -> list[Finding]:
    """Bounds a judge's findings: capped count, valid severities, non-empty
    messages, sane line numbers. Protects the whole judgment for hosts
    (like OpenClaw) that discard a result whole when one field is out of
    bounds."""
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
    """Keeps the judge inside this evaluator's authority: only "pass" and
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


def normalize_score(raw: object) -> float | None:
    """Clamps the judge's quality estimate into 0..1; anything unusable
    becomes null rather than a made-up number."""
    try:
        score = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if score != score:  # NaN
        return None
    return min(1.0, max(0.0, score))
