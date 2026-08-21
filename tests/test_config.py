from skills_evaluator.config import load_settings


def test_operator_core_url_wins_over_legacy_setting(monkeypatch):
    monkeypatch.setenv("CASSETTE_CORE_URL", "http://tapes-api:8091")
    monkeypatch.setenv("CASSETTE_TAPES_BASE_URL", "http://legacy:8081")

    assert load_settings().tapes_base_url == "http://tapes-api:8091"
