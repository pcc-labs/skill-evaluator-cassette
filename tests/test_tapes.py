from __future__ import annotations

import httpx
import pytest

from skills_evaluator.tapes import SearchUnavailableError, TapesClient


def make_client(handler) -> TapesClient:
    return TapesClient(
        "http://tapes.test", httpx.Client(transport=httpx.MockTransport(handler))
    )


def test_search_spans_maps_hits_and_503():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/cassettes/search/spans"
        assert request.url.params["query"] == "inbox triage"
        assert request.url.params["top_k"] == "5"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "trace_id": "t1",
                        "span_id": "sp1",
                        "session_id": "s1",
                        "score": 0.8,
                        "user_prompt": "u",
                        "snippet": "sn",
                    }
                ]
            },
        )

    hits = make_client(handler).search_spans("inbox triage", 5)
    assert hits[0].session_id == "s1"
    assert hits[0].score == pytest.approx(0.8)

    unavailable = make_client(lambda _: httpx.Response(503, text="no embedder"))
    with pytest.raises(SearchUnavailableError):
        unavailable.search_spans("q", 5)


def test_get_skill_uses_tapes_skills_api():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/skills/skill-1"
        return httpx.Response(
            200,
            json={
                "id": "skill-1",
                "name": "Triage",
                "description": "Triage an inbox",
                "content": "# Triage",
                "originatingSessionIds": ["session-1"],
            },
        )

    skill = make_client(handler).get_skill("skill-1")
    assert skill.name == "Triage"
    assert skill.originating_session_ids == ["session-1"]


def test_transcript_renders_spine_and_tools():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/traces":
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "trace_id": "t1",
                            "user_prompt": "catch me up",
                            "response_preview": "preview",
                            "synthetic": "",
                        },
                        {
                            "trace_id": "t2",
                            "user_prompt": "replayed",
                            "response_preview": "",
                            "synthetic": "resume-replay",
                        },
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "spans": [
                    {"kind": "tool", "name": "read_email", "thread_id": ""},
                    {"kind": "tool", "name": "read_email", "thread_id": ""},
                    {"kind": "tool", "name": "sub_tool", "thread_id": "sub-1"},
                    {
                        "kind": "llm",
                        "call_kind": "main",
                        "thread_id": "",
                        "output": [{"text": "Here is your inbox summary."}],
                    },
                    {
                        "kind": "llm",
                        "call_kind": "offshoot:title-gen",
                        "thread_id": "",
                        "output": [{"text": "A Title"}],
                    },
                ]
            },
        )

    transcript = make_client(handler).session_transcript("s1")
    assert transcript == (
        "[user] catch me up\n"
        "[tools] read_email ×2\n"
        "[assistant] Here is your inbox summary.\n"
    )


def test_transcript_falls_back_to_preview():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/traces":
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "trace_id": "t1",
                            "user_prompt": "hi",
                            "response_preview": "the preview",
                            "synthetic": "",
                        }
                    ]
                },
            )
        return httpx.Response(200, json={"spans": []})

    transcript = make_client(handler).session_transcript("s1")
    assert transcript == "[user] hi\n[assistant] the preview\n"


def test_empty_session_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"items": []})

    with pytest.raises(ValueError, match="no turns"):
        make_client(handler).session_transcript("s1")
