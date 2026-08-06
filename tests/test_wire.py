from __future__ import annotations

from skills_evaluator.wire import (
    MAX_FINDINGS,
    MAX_MESSAGE_CHARS,
    MAX_RULE_ID_CHARS,
    EvaluateRequest,
    EvaluateResponse,
    Finding,
    normalize_decision,
    normalize_findings,
)


def test_request_parses_snake_case():
    request = EvaluateRequest.model_validate_json(
        b'{"ref": {"source": "openclaw", "id": "p1", "revision_sha256": "abc"},'
        b' "skill": {"name": "n", "description": "d"},'
        b' "candidate": {"skill_md": "# s", "files": [{"path": "a.md", "content": "x"}]},'
        b' "baseline": {"skill_md": "# old"}}'
    )
    assert request.ref is not None and request.ref.revision_sha256 == "abc"
    assert request.candidate.files[0].path == "a.md"
    assert request.baseline is not None and request.baseline.skill_md == "# old"


def test_request_accepts_skill_id_alone():
    request = EvaluateRequest.model_validate_json(b'{"skill_id": "sk-1"}')
    assert request.skill_id == "sk-1"
    assert request.candidate.skill_md == ""


def test_response_serializes_snake_case():
    encoded = EvaluateResponse(
        summary="s", findings=[Finding(message="m")]
    ).model_dump_json()
    for key in (
        '"summary"',
        '"findings"',
        '"rule_id"',
        '"severity"',
        '"metrics"',
        '"sessions_considered"',
        '"provenance_sessions"',
        '"decision"',
        '"score"',
        '"evaluator_version"',
        '"mode"',
    ):
        assert key in encoded


def test_normalize_findings_caps_and_defaults():
    raw = [
        Finding(rule_id="r" * 500, severity="URGENT!!", message="m" * 5_000, line=-3)
    ] * (MAX_FINDINGS + 20) + [Finding(message="   ")]
    findings = normalize_findings(raw)

    assert len(findings) == MAX_FINDINGS
    assert len(findings[0].rule_id) == MAX_RULE_ID_CHARS
    assert len(findings[0].message) == MAX_MESSAGE_CHARS
    assert findings[0].severity == "info"
    assert findings[0].line == 0


def test_normalize_findings_fills_missing_rule_id():
    findings = normalize_findings([Finding(rule_id="  ", message="real")])
    assert findings[0].rule_id == "judge.finding"


def test_normalize_decision_never_blocks():
    warn = [Finding(severity="warn", message="m")]
    assert normalize_decision("block", []) == "revise"
    assert normalize_decision("pass", warn) == "pass"
    assert normalize_decision("nonsense", warn) == "revise"
    assert normalize_decision("", []) == "pass"


def test_normalize_score():
    from skills_evaluator.wire import normalize_score

    assert normalize_score(0.7) == 0.7
    assert normalize_score(7) == 1.0
    assert normalize_score(-1) == 0.0
    assert normalize_score("0.4") == 0.4
    assert normalize_score("high") is None
    assert normalize_score(None) is None
    assert normalize_score(float("nan")) is None
