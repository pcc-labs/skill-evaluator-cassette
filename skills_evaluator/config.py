"""Environment configuration, following the cassette convention: every
setting in cassette.toml arrives as a CASSETTE_* environment variable (dots
become underscores: llm.api_key -> CASSETTE_LLM_API_KEY). The cassette runs
with none of them set — an LLM credential is required to evaluate, but its
absence keeps /evaluate at 503 rather than refusing to start."""

from __future__ import annotations

import os
from dataclasses import dataclass

import dspy

DEFAULT_LISTEN = "0.0.0.0:9978"
DEFAULT_TAPES_BASE_URL = "http://127.0.0.1:8081"
DEFAULT_PROVIDER = "anthropic"

# Default judge model per provider, mirroring tapes' skill generator.
PROVIDER_DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
    "ollama": "llama3.2",
}


class LLMUnconfiguredError(Exception):
    """No usable judge model could be built from the environment."""


@dataclass
class Settings:
    listen: str
    tapes_base_url: str
    search_top_k: int
    max_sessions: int
    search_min_score: float
    spec_autogenerate: bool
    llm_provider: str
    llm_model: str
    llm_api_key: str
    llm_base_url: str

    @property
    def host(self) -> str:
        host, _, _ = self.listen.rpartition(":")
        return host or "0.0.0.0"

    @property
    def port(self) -> int:
        _, _, port = self.listen.rpartition(":")
        return int(port) if port.isdigit() else 9978

    @property
    def judge_model_label(self) -> str:
        model = self.llm_model or PROVIDER_DEFAULT_MODELS.get(self.llm_provider, "")
        return f"{self.llm_provider}/{model or 'default'}"


def load_settings() -> Settings:
    return Settings(
        listen=_env("CASSETTE_LISTEN", DEFAULT_LISTEN),
        tapes_base_url=_env(
            "CASSETTE_CORE_URL",
            _env("CASSETTE_TAPES_BASE_URL", DEFAULT_TAPES_BASE_URL),
        ),
        search_top_k=_env_int("CASSETTE_SEARCH_TOP_K", 5),
        max_sessions=_env_int("CASSETTE_MAX_SESSIONS", 3),
        search_min_score=_env_float("CASSETTE_SEARCH_MIN_SCORE", 0.35),
        spec_autogenerate=_env_bool("CASSETTE_SPEC_AUTOGENERATE", True),
        llm_provider=_env("CASSETTE_LLM_PROVIDER", DEFAULT_PROVIDER).lower(),
        llm_model=_env("CASSETTE_LLM_MODEL", ""),
        llm_api_key=_env("CASSETTE_LLM_API_KEY", ""),
        llm_base_url=_env("CASSETTE_LLM_BASE_URL", ""),
    )


def build_lm(settings: Settings) -> dspy.LM:
    """Builds the judge LM from settings, raising LLMUnconfiguredError when
    a required credential is missing. Model strings follow litellm's
    provider/model convention."""
    provider = settings.llm_provider
    model = settings.llm_model or PROVIDER_DEFAULT_MODELS.get(provider, "")

    match provider:
        case "ollama":
            return dspy.LM(
                f"ollama_chat/{model}",
                api_base=settings.llm_base_url or "http://localhost:11434",
                api_key="",
            )
        case "anthropic" | "openai":
            api_key = settings.llm_api_key or os.environ.get(
                "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY",
                "",
            )
            if not api_key:
                raise LLMUnconfiguredError(
                    f"no API key for provider {provider!r}: set CASSETTE_LLM_API_KEY "
                    f"or {'ANTHROPIC_API_KEY' if provider == 'anthropic' else 'OPENAI_API_KEY'}"
                )
            kwargs: dict[str, object] = {"api_key": api_key}
            if settings.llm_base_url:
                kwargs["api_base"] = settings.llm_base_url
            return dspy.LM(f"{provider}/{model}", **kwargs)
        case _:
            raise LLMUnconfiguredError(f"unsupported provider {provider!r}")


def _env(key: str, default: str) -> str:
    return os.environ.get(key, "").strip() or default


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "").strip()
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key, "").strip()
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if 0.0 <= value <= 1.0 else default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default
