"""Orchestration around the DSPy pipeline: skill resolution, query
derivation, evidence gathering, the honest no-evidence path, and
normalization into the wire contract. Everything model-shaped lives in
``pipeline``; everything tapes-shaped lives in ``tapes``; this module is
the seam between them and the one the tests exercise with fakes."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx

from .pipeline import MAX_TRANSCRIPT_CHARS, RULESET_VERSION, JudgeFinding
from .store import RevisionRecord, RevisionStore, new_revision_id, utcnow
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


class NoEvidenceError(Exception):
    """A revision was requested but no session evidence exists to ground
    it. Rewriting a skill from nothing would be exactly the ungrounded
    guessing this cassette exists to prevent."""

logger = logging.getLogger(__name__)

MAX_QUERIES = 3
MAX_QUERY_CHARS = 200

VERSION = "0.2.0"


@dataclass
class ServiceConfig:
    top_k: int = 5
    max_sessions: int = 3
    judge_model: str = ""
    version: str = VERSION


@dataclass
class EvaluationService:
    """Turns one skill document plus tapes telemetry into a judgment.

    ``module`` is any callable with SkillEvaluator's forward signature —
    the DSPy program in production, a stub in tests.
    """

    tapes: TapesClient
    module: object
    config: ServiceConfig = field(default_factory=ServiceConfig)
    # reviser is any callable with SkillReviser's forward signature; store
    # is where proposed revisions and their accept/reject labels accumulate.
    # Both are optional so an evaluate-only deployment stays minimal.
    reviser: object | None = None
    store: RevisionStore | None = None

    def evaluate(self, request: EvaluateRequest) -> EvaluateResponse:
        response, _ = self._evaluate_full(request)
        return response

    def revise(self, request: EvaluateRequest) -> RevisionRecord:
        """Evaluates the skill, proposes an evidence-grounded revision, and
        stores it as ``proposed`` — the unit the status hook later labels."""
        if self.reviser is None or self.store is None:
            raise RuntimeError("revision pipeline is not configured")

        response, transcripts = self._evaluate_full(request)
        if not transcripts:
            raise NoEvidenceError(
                "no session evidence relates to this skill; "
                "refusing to propose an ungrounded revision"
            )

        evidence = "\n---\n".join(text for _, text in transcripts)
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
        )

        record = RevisionRecord(
            id=new_revision_id(),
            skill_id=request.skill_id,
            ref=request.ref.model_dump() if request.ref else None,
            skill_name=request.skill.name,
            original_skill_md=request.candidate.skill_md,
            revised_skill_md=str(proposal.revised_markdown or "").strip(),
            rationale=str(proposal.rationale or "").strip()[:MAX_REASON_CHARS * 2],
            evaluation=response.model_dump(),
            created_at=utcnow(),
        )
        return self.store.insert(record)

    def _evaluate_full(
        self, request: EvaluateRequest
    ) -> tuple[EvaluateResponse, list[tuple[str, str]]]:
        provenance_ids = self._resolve_skill(request)

        hits, search_note = self._gather_hits(request)
        session_ids = merge_evidence_sessions(
            provenance_ids, hits, self.config.max_sessions
        )
        transcripts = self._build_transcripts(session_ids)
        loaded_ids = {sid for sid, _ in transcripts}

        metrics = Metrics(
            sessions_considered=len(transcripts),
            provenance_sessions=sum(
                1 for sid in provenance_ids if sid in loaded_ids
            ),
            spans_matched=len(hits),
            mean_search_score=round(sum(h.score for h in hits) / len(hits), 6)
            if hits
            else 0.0,
            judge_model=self.config.judge_model,
            transcript_chars=sum(len(text) for _, text in transcripts),
        )

        if not transcripts:
            return (
                self._finish(request, self._no_evidence_response(metrics, search_note)),
                transcripts,
            )

        prediction = self.module(
            skill_name=request.skill.name,
            skill_markdown=request.candidate.skill_md,
            baseline_markdown=request.baseline.skill_md if request.baseline else "",
            transcripts=transcripts,
        )

        findings = normalize_findings(
            [_to_wire_finding(f) for f in (prediction.findings or [])]
        )
        response = EvaluateResponse(
            summary=str(prediction.summary or "").strip()[:MAX_SUMMARY_CHARS],
            findings=findings,
            metrics=metrics,
            decision=normalize_decision(str(prediction.decision or ""), findings),
            decision_reason=str(prediction.decision_reason or "").strip()[
                :MAX_REASON_CHARS
            ],
            score=normalize_score(getattr(prediction, "score", None)),
            evaluator_version=f"{self.config.version}+{RULESET_VERSION}",
            mode="llm",
        )
        if search_note:
            response.findings.insert(0, _note_finding(search_note))
        return self._finish(request, response), transcripts

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

    def _gather_hits(self, request: EvaluateRequest) -> tuple[list[SearchHit], str]:
        """Runs every derived query through span search. A deployment
        without search (503) is a degraded state to report, not an error;
        any other failure means the core API is broken and the evaluation
        cannot honestly proceed."""
        hits: list[SearchHit] = []
        for query in build_queries(request):
            try:
                hits.extend(self.tapes.search_spans(query, self.config.top_k))
            except SearchUnavailableError:
                return [], (
                    "span search is not configured on this tapes deployment; "
                    "evidence is limited to the skill's provenance sessions"
                )
        return hits, ""

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
        """The honest answer when nothing in tapes relates to the skill: no
        findings against it, a null score, and a clearly-labeled absence of
        evidence rather than an endorsement."""
        note = search_note or (
            "no captured sessions relate to this skill; "
            "nothing to judge it against"
        )
        return EvaluateResponse(
            summary=f"No session evidence: {note}.",
            findings=[_note_finding(note)],
            metrics=metrics,
            decision="pass",
            decision_reason="no session evidence contradicts the skill",
            score=None,
            evaluator_version=f"{self.config.version}+{RULESET_VERSION}",
            mode="no-evidence",
        )

    def _finish(
        self, request: EvaluateRequest, response: EvaluateResponse
    ) -> EvaluateResponse:
        response.ref = request.ref
        return response


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
