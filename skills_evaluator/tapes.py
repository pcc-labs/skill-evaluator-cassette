"""Read-only client for the tapes core API.

The evaluator needs two things from tapes: span search to find the sessions
a proposed skill is about, and the trace surface to render those sessions
as transcripts. The transcript rendering is a port of tapes'
``pkg/skill/transcript.go`` — the same code path skill generation uses —
so the judge sees the actual conversation, not the harness's shadow
traffic: only main-thread conversation-spine llm spans and tool summaries,
synthetic turns dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx

REQUEST_TIMEOUT_SECONDS = 30.0


class SearchUnavailableError(Exception):
    """Tapes answered 503: span search is not configured on this deployment.
    The evaluator degrades to a no-evidence judgment instead of failing."""


class SkillNotFoundError(Exception):
    """Tapes answered 404 for a skill_id: nothing to evaluate."""


@dataclass
class SearchHit:
    """One span-search result reduced to what evaluation uses."""

    session_id: str
    trace_id: str
    score: float
    user_prompt: str = ""
    snippet: str = ""


@dataclass
class SkillRecord:
    """The slice of a tapes skill (GET /v1/skills/{id}) evaluation uses.
    ``originating_session_ids`` is the provenance — the sessions the skill
    was generated from — and the strongest evidence to judge it against."""

    id: str
    name: str
    description: str
    content: str
    originating_session_ids: list[str] = field(default_factory=list)


@dataclass
class TraceSummary:
    """One user-visible turn header of a session."""

    trace_id: str
    user_prompt: str
    response_preview: str
    synthetic: str


@dataclass
class Span:
    """The slice of a span the transcript renderer consumes."""

    kind: str
    name: str
    call_kind: str
    thread_id: str
    texts: list[str] = field(default_factory=list)


class TapesClient:
    """HTTP client against a running tapes core API (e.g. http://127.0.0.1:8081)."""

    def __init__(self, base_url: str, client: httpx.Client | None = None) -> None:
        base = base_url.strip().rstrip("/")
        if base and "://" not in base:
            base = "http://" + base
        self.base_url = base
        self._client = client or httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS)

    def search_spans(self, query: str, top_k: int) -> list[SearchHit]:
        response = self._client.get(
            f"{self.base_url}/v1/cassettes/search/spans",
            params={"query": query, "top_k": str(top_k)},
        )
        if response.status_code == 503:
            raise SearchUnavailableError(response.text)
        response.raise_for_status()
        results = response.json().get("results") or []
        return [
            SearchHit(
                session_id=hit.get("session_id", ""),
                trace_id=hit.get("trace_id", ""),
                score=float(hit.get("score", 0.0)),
                user_prompt=hit.get("user_prompt", ""),
                snippet=hit.get("snippet", ""),
            )
            for hit in results
        ]

    def get_skill(self, skill_id: str) -> SkillRecord:
        """Resolves a platform skill by id. The skills surface is camelCase
        (it predates the rest of the API's snake_case; the console owns it)."""
        response = self._client.get(f"{self.base_url}/v1/skills/{skill_id}")
        if response.status_code == 404:
            raise SkillNotFoundError(f"skill {skill_id} not found")
        response.raise_for_status()
        body = response.json()
        return SkillRecord(
            id=body.get("id", skill_id),
            name=body.get("name", ""),
            description=body.get("description", ""),
            content=body.get("content", ""),
            originating_session_ids=[
                sid for sid in (body.get("originatingSessionIds") or []) if sid
            ],
        )

    def trace_summaries(self, session_id: str) -> list[TraceSummary]:
        response = self._client.get(
            f"{self.base_url}/v1/traces", params={"session_id": session_id}
        )
        response.raise_for_status()
        items = response.json().get("items") or []
        return [
            TraceSummary(
                trace_id=item.get("trace_id", ""),
                user_prompt=item.get("user_prompt", ""),
                response_preview=item.get("response_preview", ""),
                synthetic=item.get("synthetic", ""),
            )
            for item in items
        ]

    def trace_spans(self, trace_id: str) -> list[Span]:
        response = self._client.get(f"{self.base_url}/v1/traces/{trace_id}")
        response.raise_for_status()
        spans = response.json().get("spans") or []
        parsed: list[Span] = []
        for span in spans:
            texts = [
                block["text"]
                for block in (span.get("output") or [])
                if isinstance(block, dict) and block.get("text")
            ]
            parsed.append(
                Span(
                    kind=span.get("kind", ""),
                    name=span.get("name", ""),
                    call_kind=span.get("call_kind", ""),
                    thread_id=span.get("thread_id", ""),
                    texts=texts,
                )
            )
        return parsed

    def session_transcript(self, session_id: str) -> str:
        """Renders the ``[user]`` / ``[assistant]`` / ``[tools]`` transcript
        for one session, synthetic turns dropped."""
        lines: list[str] = []
        for turn in self.trace_summaries(session_id):
            if turn.synthetic:
                continue
            if turn.user_prompt:
                lines.append(f"[user] {turn.user_prompt}")
            try:
                spans = self.trace_spans(turn.trace_id)
            except httpx.HTTPError:
                spans = []
            if not _render_spine(lines, spans) and turn.response_preview:
                lines.append(f"[assistant] {turn.response_preview}")
        if not lines:
            raise ValueError(f"no turns in session {session_id}")
        return "\n".join(lines) + "\n"


def _render_spine(lines: list[str], spans: list[Span]) -> bool:
    """Walks one turn's spans in order, emitting an ``[assistant]`` line per
    conversation-spine llm span and a ``[tools]`` summary for the tool calls
    between them. Offshoot/injected call kinds and subagent threads are
    skipped. Returns whether any assistant text was written."""
    wrote = False
    pending: dict[str, int] = {}
    order: list[str] = []

    def flush_tools() -> None:
        nonlocal pending, order
        if order:
            parts = [
                f"{name} ×{pending[name]}" if pending[name] > 1 else name
                for name in order
            ]
            lines.append(f"[tools] {', '.join(parts)}")
        pending, order = {}, []

    for span in spans:
        if span.kind == "tool":
            if span.thread_id:
                continue
            if span.name not in pending:
                order.append(span.name)
                pending[span.name] = 0
            pending[span.name] += 1
        elif span.kind == "llm":
            if span.call_kind != "main" or span.thread_id:
                continue
            text = "\n".join(span.texts)
            if not text:
                continue
            flush_tools()
            lines.append(f"[assistant] {text}")
            wrote = True
    flush_tools()
    return wrote
