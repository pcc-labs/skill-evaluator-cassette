"""skills-evaluator: a tapes cassette that evaluates OpenClaw Skill
Workshop proposals against captured session data with a DSPy pipeline."""

from __future__ import annotations

import logging


def main() -> None:
    """Start the cassette server. Configuration is environment-only; see
    cassette.toml for the schema (CASSETTE_* variables)."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    logger = logging.getLogger("skills_evaluator")

    import dspy
    import uvicorn

    from .config import LLMUnconfiguredError, build_lm, load_settings
    from .pipeline import SkillEvaluator
    from .server import create_app
    from .service import EvaluationService, ServiceConfig
    from .tapes import TapesClient

    settings = load_settings()

    service = None
    llm_error = ""
    try:
        dspy.configure(lm=build_lm(settings))
        service = EvaluationService(
            tapes=TapesClient(settings.tapes_base_url),
            module=SkillEvaluator(),
            config=ServiceConfig(
                top_k=settings.search_top_k,
                max_sessions=settings.max_sessions,
                judge_model=settings.judge_model_label,
            ),
        )
        logger.info(
            "judge configured: %s; tapes at %s",
            settings.judge_model_label,
            settings.tapes_base_url,
        )
    except LLMUnconfiguredError as error:
        # A configuration state, not a crash: discovery and health stay
        # inspectable, and /evaluate reports 503 with the reason.
        llm_error = str(error)
        logger.warning("llm unavailable; /evaluate will report 503: %s", error)

    app = create_app(service, llm_error)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")
