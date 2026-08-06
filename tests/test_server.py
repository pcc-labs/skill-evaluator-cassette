from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from skills_evaluator.server import create_app, load_manifest


@pytest.fixture
def client(service) -> TestClient:
    return TestClient(create_app(service))


def test_manifest_is_valid_cassette_metadata():
    manifest = load_manifest()
    assert manifest["kind"] == "cassette/v1alpha1"
    assert manifest["cassette"]["name"] == "skills-evaluator"
    assert manifest["api"] == {
        "health": "/ping",
        "openapi": "/openapi",
        "prefix_path": "api",
    }
    assert manifest["depends"] == {"core": "v1", "views": []}
    secret_keys = [c["key"] for c in manifest["config"] if c.get("secret")]
    assert secret_keys == ["llm.api_key"]


def test_ping(client):
    assert client.get("/ping").json() == {
        "status": "ok",
        "cassette": "skills-evaluator",
    }


def test_openapi_embeds_manifest_and_declares_prefixed_path(client):
    document = client.get("/openapi").json()
    assert document["x-tapes-cassette"] == load_manifest()
    assert "/api/skills-evaluator/evaluate" in document["paths"]
    assert "EvaluateRequest" in document["components"]["schemas"]


def test_evaluate_requires_skill_md_or_skill_id(client):
    response = client.post(
        "/api/skills-evaluator/evaluate",
        json={"skill": {"name": "n"}, "candidate": {"skill_md": "  "}},
    )
    assert response.status_code == 400
    assert "skill_id" in response.json()["detail"]


def test_unknown_skill_id_is_404(client):
    response = client.post(
        "/api/skills-evaluator/evaluate", json={"skill_id": "nope"}
    )
    assert response.status_code == 404


def test_evaluate_returns_judgment(client, fake_tapes):
    fake_tapes.hits["n "] = []
    response = client.post(
        "/api/skills-evaluator/evaluate",
        json={
            "proposal": {"id": "p1", "kind": "create"},
            "skill": {"name": "n"},
            "candidate": {"skill_md": "# skill"},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "pass"
    assert body["mode"] == "no-evidence"
    assert body["metrics"]["sessions_considered"] == 0


def test_unconfigured_llm_reports_503():
    client = TestClient(create_app(None, "no API key for provider 'anthropic'"))
    response = client.post(
        "/api/skills-evaluator/evaluate",
        json={"skill": {"name": "n"}, "candidate": {"skill_md": "# s"}},
    )
    assert response.status_code == 503
    assert "no API key" in response.json()["detail"]


def test_service_failure_is_502(client, fake_tapes):
    fake_tapes.search_error = ConnectionError("refused")
    response = client.post(
        "/api/skills-evaluator/evaluate",
        json={"skill": {"name": "n"}, "candidate": {"skill_md": "# s"}},
    )
    assert response.status_code == 502
    assert "evaluation failed" in response.json()["detail"]
