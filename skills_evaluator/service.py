"""Orchestration around the DSPy pipeline: skill resolution, query
derivation, evidence gathering (with the relevance gate), the tiered
llm / spec / no-evidence modes, spec generation, and normalization into the
wire contract. Everything model-shaped lives in ``pipeline``; everything
tapes-shaped lives in ``tapes``; this module is the seam between them and
the one the tests exercise with fakes."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

import httpx

from .pipeline import MAX_TRANSCRIPT_CHARS, RULESET_VERSION, JudgeFinding
from .store import (
    EvalRecord,
    RevisionRecord,
    RevisionStore,
    new_revision_id,
    utcnow,
)
from .tapes import SearchHit, SearchUnavailableError, TapesClient
from .wire import (
    MAX_REASON_CHARS,
    MAX_SUMMARY_CHARS,
    EvaluateRequest,
    EvaluateResponse,
    Finding,
    Metrics,
    normalize_decision,
    normalize_findings,
    normalize_score,
)

logger = logging.getLogger(__name__)

MAX_QUERIES = 3
MAX_QUERY_CHARS = 200

VERSION = "0.3.0"


class NoEvidenceError(Exception):
    """A revision was requested but there is neither session evidence nor an
    eval spec to ground it. Rewriting a skill from nothing would be exactly
    the ungrounded guessing this cassette exists to prevent."""


class RevisionFailedError(Exception):
    """The reviser returned no usable document. Storing an empty or
    fence-only rewrite would poison the corpus with a 'proposal' no human
    could meaningfully accept or reject."""


# A revised document shorter than this is not a document.
MIN_REVISION_CHARS = 20


@dataclass
class ServiceConfig:
    top_k: int = 5
    max_sessions: int = 3
    # Hits below this similarity score are noise, not evidence: a 0.22
    # inbox-session "match" for a haiku skill must never reach the judge.
    min_search_score: float = 0.35
    # When a skill has no evidence and no spec, draft-and-store a spec
    # inline so day-zero skills still get judged on their own terms.
    spec_autogenerate: bool = True
    # A spec-mode score is a self-consistency claim, not a behavioral one;
    # the cap keeps behavioral evidence strictly outranking it.
    spec_score_cap: float = 0.9
    judge_model: str = ""
    version: str = VERSION


@dataclass
class _Grounds:
    """What one evaluation stood on."""

    transcripts: list[tuple[str, str]]
    spec_text: str
    eval_record: EvalRecord | None


@dataclass
class EvaluationService:
    """Turns one skill document plus tapes telemetry into a judgment.

    ``module``, ``reviser``, and ``spec_generator`` are callables with the
    corresponding DSPy modules' forward signatures — real programs in
    production, stubs in tests.
    """

    tapes: TapesClient
    module: object
    config: ServiceConfig = field(default_factory=ServiceConfig)
    reviser: object | None = None
    store: RevisionStore | None = None
    spec_generator: object | None = None

    # ------------------------------------------------------------------
    # Evaluate

    def evaluate(self, request: EvaluateRequest) -> EvaluateResponse:
        response, _ = self._evaluate_full(request)
        return response

    def _evaluate_full(
        self, request: EvaluateRequest
    ) -> tuple[EvaluateResponse, _Grounds]:
        provenance_ids = self._resolve_skill(request)

        hits, gated_out, search_note = self._gather_hits(request)
        session_ids = merge_evidence_sessions(
            provenance_ids, hits, self.config.max_sessions
        )
        transcripts = self._build_transcripts(session_ids)
        loaded_ids = {sid for sid, _ in transcripts}

        eval_record = self._resolve_spec(request, transcripts)
        spec_text = render_spec(eval_record.spec) if eval_record else ""
        grounds = _Grounds(transcripts, spec_text, eval_record)

        metrics = Metrics(
            sessions_considered=len(transcripts),
            provenance_sessions=sum(1 for sid in provenance_ids if sid in loaded_ids),
            spans_matched=len(hits),
            spans_gated=gated_out,
            mean_search_score=round(sum(h.score for h in hits) / len(hits), 6)
            if hits
            else 0.0,
            spec_criteria=len((eval_record.spec.get("criteria") or []))
            if eval_record
            else 0,
            judge_model=self.config.judge_model,
            transcript_chars=sum(len(text) for _, text in transcripts),
        )

        if not transcripts and not spec_text:
            return (
                self._finish(request, self._no_evidence_response(metrics, search_note)),
                grounds,
            )

        prediction = self.module(
            skill_name=request.skill.name,
            skill_markdown=request.candidate.skill_md,
            baseline_markdown=request.baseline.skill_md if request.baseline else "",
            transcripts=transcripts,
            eval_spec=spec_text,
        )

        findings = normalize_findings(
            [_to_wire_finding(f) for f in (prediction.findings or [])]
        )
        score = normalize_score(getattr(prediction, "score", None))
        mode = "llm"
        if not transcripts:
            # Judged purely against the skill's own criteria: honest label,
            # capped score, so behavior always outranks self-consistency.
            mode = "spec"
            if score is not None:
                score = min(score, self.config.spec_score_cap)

        response = EvaluateResponse(
            summary=str(prediction.summary or "").strip()[:MAX_SUMMARY_CHARS],
            findings=findings,
            metrics=metrics,
            decision=normalize_decision(str(prediction.decision or ""), findings),
            decision_reason=str(prediction.decision_reason or "").strip()[
                :MAX_REASON_CHARS
            ],
            score=score,
            evaluator_version=f"{self.config.version}+{RULESET_VERSION}",
            mode=mode,
        )
        if search_note:
            response.findings.insert(0, _note_finding(search_note))
        return self._finish(request, response), grounds

    # ------------------------------------------------------------------
    # Revise

    def revise(self, request: EvaluateRequest) -> RevisionRecord:
        """Evaluates the skill, proposes a grounded revision, and stores it
        as ``proposed`` — the unit the status hook later labels. Session
        evidence and eval criteria are both legitimate grounds; having
        neither is the only refusal."""
        if self.reviser is None or self.store is None:
            raise RuntimeError("revision pipeline is not configured")

        response, grounds = self._evaluate_full(request)
        if response.mode == "no-evidence":
            raise NoEvidenceError(
                "no session evidence or eval spec relates to this skill; "
                "refusing to propose an ungrounded revision"
            )

        evidence = "\n---\n".join(text for _, text in grounds.transcripts)
        proposal = self.reviser(
            skill_name=request.skill.name,
            skill_markdown=request.candidate.skill_md,
            findings=[
                JudgeFinding(
                    rule_id=f.rule_id,
                    severity=f.severity,  # type: ignore[arg-type]
                    message=f.message,
                    file=f.file,
                    line=f.line,
                )
                for f in response.findings
            ],
            session_evidence=evidence,
            eval_spec=grounds.spec_text,
        )

        revised = _strip_fences(str(proposal.revised_markdown or ""))
        if len(revised) < MIN_REVISION_CHARS:
            raise RevisionFailedError(
                "reviser produced no usable document "
                f"({len(revised)} chars after stripping fences); "
                "try a more capable judge model"
            )

        record = RevisionRecord(
            id=new_revision_id(),
            skill_id=request.skill_id,
            ref=request.ref.model_dump() if request.ref else None,
            skill_name=request.skill.name,
            original_skill_md=request.candidate.skill_md,
            revised_skill_md=revised,
            rationale=str(proposal.rationale or "").strip()[: MAX_REASON_CHARS * 2],
            evaluation=response.model_dump(),
            created_at=utcnow(),
        )
        return self.store.insert(record)

    # ------------------------------------------------------------------
    # Eval specs

    def generate_eval(self, request: EvaluateRequest, force: bool) -> EvalRecord:
        """Drafts and stores an eval spec for the skill, seeded with
        whatever evidence exists. Refuses to regenerate over a human-edited
        spec (the store enforces it); returns the existing spec unchanged
        when one exists and force is off."""
        if self.spec_generator is None or self.store is None:
            raise RuntimeError("eval spec pipeline is not configured")

        provenance_ids = self._resolve_skill(request)
        skill_key = eval_skill_key(request)
        if not skill_key:
            raise ValueError("a skill_id or skill.name is required to key an eval spec")

        existing = self.store.get_eval_for_key(skill_key)
        if existing is not None and not force:
            return existing

        hits, _, _ = self._gather_hits(request)
        session_ids = merge_evidence_sessions(
            provenance_ids, hits, self.config.max_sessions
        )
        transcripts = self._build_transcripts(session_ids)
        evidence = "\n---\n".join(text for _, text in transcripts)

        proposal = self.spec_generator(
            skill_name=request.skill.name,
            skill_description=request.skill.description,
            skill_markdown=request.candidate.skill_md,
            session_evidence=evidence,
        )
        spec = {
            "criteria": [c.model_dump() for c in (proposal.criteria or [])],
            "cases": [c.model_dump() for c in (proposal.cases or [])],
        }
        now = utcnow()
        record = EvalRecord(
            id=str(uuid.uuid4()),
            skill_key=skill_key,
            skill_id=request.skill_id,
            skill_name=request.skill.name,
            spec=spec,
            created_at=now,
            updated_at=now,
        )
        return self.store.upsert_eval(record, force)

    def _resolve_spec(
        self, request: EvaluateRequest, transcripts: list[tuple[str, str]]
    ) -> EvalRecord | None:
        """Finds the skill's stored spec; when there is none AND no session
        evidence, optionally drafts one inline so a day-zero skill still
        gets judged on its own terms. Generation failure degrades to the
        no-evidence path rather than failing the evaluation."""
        if self.store is None:
            return None
        skill_key = eval_skill_key(request)
        if not skill_key:
            return None
        record = self.store.get_eval_for_key(skill_key)
        if record is not None:
            return record
        if transcripts or not self.config.spec_autogenerate:
            return None
        if self.spec_generator is None:
            return None
        try:
            return self.generate_eval(request, force=False)
        except Exception:  # noqa: BLE001 - degrade, never fail the evaluation
            logger.exception("inline spec generation failed for %s", skill_key)
            return None

    # ------------------------------------------------------------------
    # Evidence plumbing

    def _resolve_skill(self, request: EvaluateRequest) -> list[str]:
        """Fills the candidate from tapes when the caller sent a skill_id
        instead of inline content, and returns the skill's provenance
        session ids — the sessions it was generated from, which outrank
        anything search can find."""
        if not request.skill_id:
            return []
        record = self.tapes.get_skill(request.skill_id)
        if not request.candidate.skill_md.strip():
            request.candidate.skill_md = record.content
        if not request.skill.name:
            request.skill.name = record.name
        if not request.skill.description:
            request.skill.description = record.description
        return record.originating_session_ids

    def _gather_hits(
        self, request: EvaluateRequest
    ) -> tuple[list[SearchHit], int, str]:
        """Runs every derived query through span search and drops hits below
        the relevance gate — weak similarity is noise, not evidence. Returns
        (surviving hits, gated-out count, degradation note)."""
        raw: list[SearchHit] = []
        for query in build_queries(request):
            try:
                raw.extend(self.tapes.search_spans(query, self.config.top_k))
            except SearchUnavailableError:
                return [], 0, (
                    "span search is not configured on this tapes deployment; "
                    "evidence is limited to the skill's provenance sessions"
                )
        hits = [h for h in raw if h.score >= self.config.min_search_score]
        return hits, len(raw) - len(hits), ""

    def _build_transcripts(self, session_ids: list[str]) -> list[tuple[str, str]]:
        """Renders the chosen sessions' transcripts, truncating at a session
        boundary within the same budget skill generation uses. A session
        whose projection fails to load is skipped: partial evidence with a
        correct count beats no judgment."""
        transcripts: list[tuple[str, str]] = []
        total = 0
        for session_id in session_ids:
            try:
                text = self.tapes.session_transcript(session_id)
            except (httpx.HTTPError, ValueError) as error:
                logger.warning(
                    "skipping session transcript", extra={"session_id": session_id}
                )
                logger.debug("transcript error: %s", error)
                continue
            if transcripts and total + len(text) > MAX_TRANSCRIPT_CHARS:
                break
            transcripts.append((session_id, text))
            total += len(text)
        return transcripts

    def _no_evidence_response(
        self, metrics: Metrics, search_note: str
    ) -> EvaluateResponse:
        """The honest answer when nothing grounds a judgment: no findings
        against the skill, a null score, and a clearly-labeled absence of
        evidence rather than an endorsement."""
        note = search_note or (
            "no captured sessions relate to this skill and no eval spec "
            "exists; nothing to judge it against"
        )
        return EvaluateResponse(
            summary=f"No grounds: {note}.",
            findings=[_note_finding(note)],
            metrics=metrics,
            decision="pass",
            decision_reason="nothing contradicts the skill",
            score=None,
            evaluator_version=f"{self.config.version}+{RULESET_VERSION}",
            mode="no-evidence",
        )

    def _finish(
        self, request: EvaluateRequest, response: EvaluateResponse
    ) -> EvaluateResponse:
        response.ref = request.ref
        return response


def _strip_fences(text: str) -> str:
    """Unwraps a document a model returned inside a markdown code fence."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def eval_skill_key(request: EvaluateRequest) -> str:
    """The identity an eval spec is stored under: the tapes skill id when
    the skill is stored, otherwise the skill name (an OpenClaw proposal has
    no tapes id, but its name is its workshop identity)."""
    return request.skill_id.strip() or request.skill.name.strip()


def render_spec(spec: dict) -> str:
    """Renders a stored spec for a prompt: one criterion per line with kind
    and weight, then the cases."""
    lines: list[str] = []
    for criterion in spec.get("criteria") or []:
        lines.append(
            f"- [{criterion.get('kind', 'content')}, weight={criterion.get('weight', 1)}] "
            f"{criterion.get('id', '')}: {criterion.get('description', '')}"
        )
    for case in spec.get("cases") or []:
        expects = "; ".join(case.get("expect") or [])
        lines.append(f"- case: {case.get('scenario', '')} → expect: {expects}")
    return "\n".join(lines)


def build_queries(request: EvaluateRequest) -> list[str]:
    """Derives the span-search queries from the skill: its own
    name/description first, then its leading section headings — the terms an
    agent doing this work would have used."""
    queries: list[str] = []

    def add(query: str) -> None:
        query = query.strip()[:MAX_QUERY_CHARS]
        if query and query not in queries:
            queries.append(query)

    add(f"{request.skill.name} {request.skill.description}".strip())
    for heading in _leading_headings(
        request.candidate.skill_md, MAX_QUERIES - len(queries)
    ):
        add(f"{request.skill.name} {heading}")
    if not queries:
        add(request.skill.name)
    return queries


def _leading_headings(markdown: str, n: int) -> list[str]:
    headings: list[str] = []
    for line in markdown.splitlines():
        if n <= 0:
            break
        stripped = line.strip()
        if stripped.startswith("## ") and stripped[3:].strip():
            headings.append(stripped[3:].strip())
            n -= 1
    return headings


def merge_evidence_sessions(
    provenance_ids: list[str], hits: list[SearchHit], max_sessions: int
) -> list[str]:
    """Chooses the evidence sessions: provenance first (the sessions the
    skill was generated from, in their stored order), then search hits
    ranked by best score, deduplicated. Hits without a session id cannot
    produce a transcript and are dropped."""
    chosen: list[str] = []
    seen: set[str] = set()
    for session_id in provenance_ids:
        if session_id and session_id not in seen:
            chosen.append(session_id)
            seen.add(session_id)

    best: dict[str, float] = {}
    for hit in hits:
        if not hit.session_id or hit.session_id in seen:
            continue
        if hit.score > best.get(hit.session_id, float("-inf")):
            best[hit.session_id] = hit.score
    chosen.extend(sorted(best, key=lambda sid: (-best[sid], sid)))

    return chosen[:max_sessions]


def _to_wire_finding(raw: object) -> Finding:
    if isinstance(raw, JudgeFinding):
        return Finding(
            rule_id=raw.rule_id,
            severity=raw.severity,
            message=raw.message,
            file=raw.file,
            line=raw.line,
        )
    if isinstance(raw, dict):
        return Finding(
            rule_id=str(raw.get("rule_id", "")),
            severity=str(raw.get("severity", "")),
            message=str(raw.get("message", "")),
            file=str(raw.get("file", "")),
            line=int(raw.get("line", 0) or 0),
        )
    return Finding(message=str(raw))


def _note_finding(note: str) -> Finding:
    return Finding(rule_id="evidence.none", severity="info", message=note)
