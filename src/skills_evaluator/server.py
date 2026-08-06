"""The cassette's HTTP surface.

Four things make this process a tapes cassette:

1. ``GET /ping`` answers 200 — the ``api.health`` anchor core probes.
2. ``GET /openapi`` serves an OpenAPI 3.0 document — core fetches,
   admits (on the embedded ``x-tapes-cassette`` manifest), and aggregates it.
3. The API lives under ``/api/skills-evaluator`` — the declared
   ``prefix_path`` core strips and republishes as
   ``/v1/cassettes/skills-evaluator``.
4. Configuration arrives entirely through the environment.

The manifest is authored once, in ``cassette.toml``, read with stdlib
tomllib and embedded verbatim — the authored and served manifests are the
same bytes by construction. The request/response schemas in the document
are generated from the pydantic wire models, so a renamed field cannot
keep its old name in the contract.
"""

from __future__ import annotations

import logging
import tomllib
from importlib import resources
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .service import EvaluationService
from .tapes import SkillNotFoundError
from .wire import EvaluateRequest, EvaluateResponse

logger = logging.getLogger(__name__)

# Bounds an /evaluate body: OpenClaw caps a proposal bundle at 8 MiB;
# double that leaves room for JSON encoding overhead.
MAX_REQUEST_BYTES = 16 << 20


def load_manifest() -> dict[str, Any]:
    raw = (
        resources.files("skills_evaluator").joinpath("cassette.toml").read_bytes()
    )
    return tomllib.loads(raw.decode())


def create_app(
    service: EvaluationService | None,
    llm_error: str = "",
) -> FastAPI:
    """Builds the app. ``service`` is None when no LLM credential resolved
    at startup — the cassette stays up and answers discovery, and /evaluate
    explains what is missing."""
    manifest = load_manifest()
    name = manifest["cassette"]["name"]
    prefix = f"/api/{name}"
    document = _openapi_document(manifest, prefix)

    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)

    @app.get("/ping")
    def ping() -> dict[str, str]:
        return {"status": "ok", "cassette": name}

    @app.get("/openapi")
    def openapi() -> JSONResponse:
        return JSONResponse(document)

    @app.post(f"{prefix}/evaluate", response_model=EvaluateResponse)
    async def evaluate(request: Request) -> EvaluateResponse:
        if service is None:
            raise HTTPException(
                status_code=503,
                detail="evaluation is not configured: no LLM credential resolved"
                + (f" ({llm_error})" if llm_error else ""),
            )
        body = await request.body()
        if len(body) > MAX_REQUEST_BYTES:
            raise HTTPException(status_code=413, detail="request body too large")
        try:
            parsed = EvaluateRequest.model_validate_json(body)
        except ValueError as error:
            raise HTTPException(
                status_code=400, detail=f"body must be a valid request: {error}"
            ) from error
        if not parsed.candidate.skill_md.strip() and not parsed.skill_id.strip():
            raise HTTPException(
                status_code=400,
                detail="either candidate.skill_md or skill_id is required",
            )
        try:
            response = service.evaluate(parsed)
        except SkillNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except Exception as error:  # noqa: BLE001 - the boundary reports, never hides
            logger.exception(
                "evaluation failed for skill %s",
                parsed.skill_id or parsed.skill.name,
            )
            raise HTTPException(
                status_code=502, detail=f"evaluation failed: {error}"
            ) from error
        logger.info(
            "evaluated skill %s: decision=%s score=%s findings=%d sessions=%d",
            parsed.skill_id or parsed.skill.name or (parsed.ref.id if parsed.ref else ""),
            response.decision,
            response.score,
            len(response.findings),
            response.metrics.sessions_considered,
        )
        return response

    return app


def _openapi_document(manifest: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Renders the OpenAPI 3.0 document core fetches: every path under the
    admitted prefix, schemas reflected from the wire models, and the
    manifest riding as the ``x-tapes-cassette`` root extension."""
    schemas: dict[str, Any] = {}

    def schema_ref(model: type) -> dict[str, Any]:
        schema = model.model_json_schema(
            ref_template="#/components/schemas/{model}"
        )
        schemas.update(schema.pop("$defs", {}))
        schemas[model.__name__] = schema
        return {"$ref": f"#/components/schemas/{model.__name__}"}

    request_ref = schema_ref(EvaluateRequest)
    response_ref = schema_ref(EvaluateResponse)

    return {
        "openapi": "3.0.3",
        "info": {
            "title": "Skills Evaluator Cassette",
            "description": manifest["cassette"]["description"],
            "version": manifest["cassette"]["version"],
        },
        "x-tapes-cassette": manifest,
        "paths": {
            f"{prefix}/evaluate": {
                "post": {
                    "operationId": "evaluateSkill",
                    "summary": "Judge one skill document against captured session evidence",
                    "description": (
                        "Accepts the skill inline or as a tapes skill_id (resolved "
                        "with its provenance sessions as seed evidence), finds "
                        "related sessions via span search, triages each transcript, "
                        "and runs a DSPy judgment over the annotated evidence. "
                        "Returns findings, metrics, a 0..1 score, and a pass/revise "
                        "decision. Host adapters (e.g. the OpenClaw Gateway plugin) "
                        "conform their events to this shape."
                    ),
                    "tags": [manifest["cassette"]["name"]],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {"schema": request_ref}
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "The judgment for this proposal revision",
                            "content": {
                                "application/json": {"schema": response_ref}
                            },
                        }
                    },
                }
            }
        },
        "components": {"schemas": schemas},
    }
