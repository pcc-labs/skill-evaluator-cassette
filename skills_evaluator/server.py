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

import asyncio
import logging
import tomllib
from importlib import resources
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .service import EvaluationService, NoEvidenceError, RevisionFailedError
from .store import (
    ACCEPTED,
    REJECTED,
    AlreadyDecidedError,
    EditedSpecError,
    EvalRecord,
    RevisionRecord,
)
from .tapes import SkillNotFoundError
from .wire import (
    EvaluateRequest,
    EvaluateResponse,
    EvalResponse,
    EvalSpec,
    EvalUpdateRequest,
    Ref,
    RevisionListResponse,
    RevisionResponse,
    RevisionStatusRequest,
)

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
        parsed = parse_evaluate_body(body)
        try:
            # The pipeline is synchronous (DSPy + httpx); run it off the
            # event loop so /ping and concurrent requests stay responsive
            # during a long judge call.
            response = await asyncio.to_thread(service.evaluate, parsed)
        except SkillNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except Exception as error:  # noqa: BLE001 - the boundary reports, never hides
            logger.exception("evaluation failed for skill %s", parsed.skill_id)
            raise HTTPException(
                status_code=502, detail=f"evaluation failed: {error}"
            ) from error
        logger.info(
            "evaluated skill %s: decision=%s score=%s findings=%d sessions=%d",
            parsed.skill_id,
            response.decision,
            response.score,
            len(response.findings),
            response.metrics.sessions_considered,
        )
        return response

    def parse_evaluate_body(body: bytes) -> EvaluateRequest:
        try:
            parsed = EvaluateRequest.model_validate_json(body)
        except ValueError as error:
            raise HTTPException(
                status_code=400, detail=f"body must be a valid request: {error}"
            ) from error
        if not parsed.skill_id.strip():
            raise HTTPException(
                status_code=400,
                detail="skill_id is required: every evaluation is anchored on a "
                "tapes skill row (create one with POST /v1/skills first)",
            )
        return parsed

    @app.post(f"{prefix}/revisions", response_model=RevisionResponse, status_code=201)
    async def create_revision(request: Request) -> RevisionResponse:
        if service is None:
            raise HTTPException(
                status_code=503,
                detail="revision is not configured: no LLM credential resolved"
                + (f" ({llm_error})" if llm_error else ""),
            )
        body = await request.body()
        if len(body) > MAX_REQUEST_BYTES:
            raise HTTPException(status_code=413, detail="request body too large")
        parsed = parse_evaluate_body(body)
        try:
            # Synchronous pipeline off the event loop, same as /evaluate:
            # a long judge+reviser call must not block /ping or discovery.
            record = await asyncio.to_thread(service.revise, parsed)
        except SkillNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except NoEvidenceError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except RevisionFailedError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        except Exception as error:  # noqa: BLE001 - the boundary reports, never hides
            logger.exception(
                "revision failed for skill %s", parsed.skill_id or parsed.skill.name
            )
            raise HTTPException(
                status_code=502, detail=f"revision failed: {error}"
            ) from error
        logger.info(
            "proposed revision %s for skill %s", record.id, record.skill_name
        )
        return _revision_response(record)

    @app.get(f"{prefix}/revisions", response_model=RevisionListResponse)
    def list_revisions(skill_id: str = "", limit: int = 20) -> RevisionListResponse:
        if service is None or service.store is None:
            raise HTTPException(status_code=503, detail="revision store unavailable")
        if not skill_id.strip():
            raise HTTPException(status_code=400, detail="skill_id is required")
        records = service.store.list_for_skill(skill_id.strip(), max(1, min(limit, 100)))
        return RevisionListResponse(items=[_revision_response(r) for r in records])

    @app.get(prefix + "/revisions/{revision_id}", response_model=RevisionResponse)
    def get_revision(revision_id: str) -> RevisionResponse:
        if service is None or service.store is None:
            raise HTTPException(status_code=503, detail="revision store unavailable")
        record = service.store.get(revision_id)
        if record is None:
            raise HTTPException(status_code=404, detail="revision not found")
        return _revision_response(record)

    @app.post(prefix + "/revisions/{revision_id}/status", response_model=RevisionResponse)
    async def set_revision_status(revision_id: str, request: Request) -> RevisionResponse:
        """The labeling hook: a host reports the human verdict on a proposed
        revision. Accepted/rejected pairs are the corpus a future GEPA run
        trains on, so a decided label can be re-asserted but never flipped."""
        if service is None or service.store is None:
            raise HTTPException(status_code=503, detail="revision store unavailable")
        try:
            update = RevisionStatusRequest.model_validate_json(await request.body())
        except ValueError as error:
            raise HTTPException(
                status_code=400, detail=f"body must be a valid status update: {error}"
            ) from error
        status = update.status.strip().lower()
        if status not in (ACCEPTED, REJECTED):
            raise HTTPException(
                status_code=400, detail='status must be "accepted" or "rejected"'
            )
        try:
            record = service.store.set_status(revision_id, status, update.reason.strip())
        except AlreadyDecidedError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        if record is None:
            raise HTTPException(status_code=404, detail="revision not found")
        logger.info("revision %s labeled %s", revision_id, status)
        return _revision_response(record)

    @app.post(f"{prefix}/evals", response_model=EvalResponse, status_code=201)
    async def create_eval(request: Request, force: bool = False) -> EvalResponse:
        """Generate (or return) the skill's eval spec. Never regenerates
        over a human-edited spec; force only replaces a generated one."""
        if service is None:
            raise HTTPException(
                status_code=503,
                detail="eval generation is not configured: no LLM credential resolved"
                + (f" ({llm_error})" if llm_error else ""),
            )
        parsed = parse_evaluate_body(await request.body())
        try:
            record = await asyncio.to_thread(service.generate_eval, parsed, force)
        except SkillNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except EditedSpecError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except Exception as error:  # noqa: BLE001 - the boundary reports, never hides
            logger.exception("eval generation failed for %s", parsed.skill_id)
            raise HTTPException(
                status_code=502, detail=f"eval generation failed: {error}"
            ) from error
        logger.info("eval spec %s stored for %s", record.id, record.skill_id)
        return _eval_response(record)

    @app.get(f"{prefix}/evals", response_model=EvalResponse)
    def get_eval_for_skill(skill_id: str = "") -> EvalResponse:
        if service is None or service.store is None:
            raise HTTPException(status_code=503, detail="eval store unavailable")
        key = skill_id.strip()
        if not key:
            raise HTTPException(status_code=400, detail="skill_id is required")
        record = service.store.get_eval_for_skill(key)
        if record is None:
            raise HTTPException(status_code=404, detail=f"no eval spec for {key!r}")
        return _eval_response(record)

    @app.get(prefix + "/evals/{eval_id}", response_model=EvalResponse)
    def get_eval(eval_id: str) -> EvalResponse:
        if service is None or service.store is None:
            raise HTTPException(status_code=503, detail="eval store unavailable")
        record = service.store.get_eval(eval_id)
        if record is None:
            raise HTTPException(status_code=404, detail="eval spec not found")
        return _eval_response(record)

    @app.put(prefix + "/evals/{eval_id}", response_model=EvalResponse)
    async def update_eval(eval_id: str, request: Request) -> EvalResponse:
        """The human edit: replace the spec. Origin flips to `edited` and
        generation can never overwrite it again — editing the spec is
        editing the metric, and the human's metric wins."""
        if service is None or service.store is None:
            raise HTTPException(status_code=503, detail="eval store unavailable")
        try:
            update = EvalUpdateRequest.model_validate_json(await request.body())
        except ValueError as error:
            raise HTTPException(
                status_code=400, detail=f"body must be a valid spec update: {error}"
            ) from error
        record = service.store.update_eval_spec(eval_id, update.spec.model_dump())
        if record is None:
            raise HTTPException(status_code=404, detail="eval spec not found")
        logger.info("eval spec %s edited (origin now %s)", eval_id, record.origin)
        return _eval_response(record)

    return app


def _eval_response(record: EvalRecord) -> EvalResponse:
    return EvalResponse(
        id=record.id,
        skill_id=record.skill_id,
        skill_name=record.skill_name,
        origin=record.origin,
        spec=EvalSpec.model_validate(record.spec),
        spec_sha256=record.spec_sha256,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _revision_response(record: RevisionRecord) -> RevisionResponse:
    return RevisionResponse(
        id=record.id,
        skill_id=record.skill_id,
        ref=Ref.model_validate(record.ref) if record.ref else None,
        skill_name=record.skill_name,
        status=record.status,
        status_reason=record.status_reason,
        revised_skill_md=record.revised_skill_md,
        rationale=record.rationale,
        evaluation=EvaluateResponse.model_validate(record.evaluation),
        created_at=record.created_at,
        decided_at=record.decided_at,
    )


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
    revision_ref = schema_ref(RevisionResponse)
    revision_list_ref = schema_ref(RevisionListResponse)
    status_ref = schema_ref(RevisionStatusRequest)
    eval_ref = schema_ref(EvalResponse)
    eval_update_ref = schema_ref(EvalUpdateRequest)

    return {
        "openapi": "3.0.3",
        "info": {
            "title": "Skills Evaluator Cassette",
            "description": manifest["cassette"]["description"],
            "version": manifest["cassette"]["version"],
        },
        "x-tapes-cassette": manifest,
        "paths": {
            f"{prefix}/evals": {
                "post": {
                    "operationId": "generateSkillEval",
                    "summary": "Generate (or return) a skill's eval spec",
                    "description": (
                        "Drafts checkable criteria and test cases from the skill's "
                        "own claims, seeded with session evidence when any exists. "
                        "Returns the existing spec unchanged unless force=true; "
                        "never regenerates over a human-edited spec (409)."
                    ),
                    "tags": [manifest["cassette"]["name"]],
                    "parameters": [
                        {
                            "name": "force",
                            "in": "query",
                            "schema": {"type": "boolean", "default": False},
                        }
                    ],
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {"schema": request_ref}},
                    },
                    "responses": {
                        "201": {
                            "description": "The stored eval spec",
                            "content": {"application/json": {"schema": eval_ref}},
                        }
                    },
                },
                "get": {
                    "operationId": "getSkillEvalByKey",
                    "summary": "Fetch the eval spec for a skill",
                    "tags": [manifest["cassette"]["name"]],
                    "parameters": [
                        {
                            "name": "skill_id",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                        },
                    ],
                    "responses": {
                        "200": {
                            "description": "The skill's eval spec",
                            "content": {"application/json": {"schema": eval_ref}},
                        }
                    },
                },
            },
            f"{prefix}/evals/{{eval_id}}": {
                "get": {
                    "operationId": "getSkillEval",
                    "summary": "Fetch one eval spec",
                    "tags": [manifest["cassette"]["name"]],
                    "parameters": [
                        {
                            "name": "eval_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "The eval spec",
                            "content": {"application/json": {"schema": eval_ref}},
                        }
                    },
                },
                "put": {
                    "operationId": "editSkillEval",
                    "summary": "Replace a spec's criteria (the human edit)",
                    "description": (
                        "Origin flips to `edited`; generation can never overwrite "
                        "an edited spec. Editing the spec is editing the metric."
                    ),
                    "tags": [manifest["cassette"]["name"]],
                    "parameters": [
                        {
                            "name": "eval_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {"schema": eval_update_ref}},
                    },
                    "responses": {
                        "200": {
                            "description": "The edited eval spec",
                            "content": {"application/json": {"schema": eval_ref}},
                        }
                    },
                },
            },
            f"{prefix}/revisions": {
                "post": {
                    "operationId": "proposeSkillRevision",
                    "summary": "Evaluate a skill and propose an evidence-grounded revision",
                    "description": (
                        "Runs the evaluation pipeline, then rewrites the skill so "
                        "the findings are addressed, grounded only in the session "
                        "evidence. The revision is stored as `proposed` with the "
                        "evaluation that motivated it; 409 when no session evidence "
                        "exists to ground a rewrite."
                    ),
                    "tags": [manifest["cassette"]["name"]],
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {"schema": request_ref}},
                    },
                    "responses": {
                        "201": {
                            "description": "The stored revision proposal",
                            "content": {"application/json": {"schema": revision_ref}},
                        }
                    },
                },
                "get": {
                    "operationId": "listSkillRevisions",
                    "summary": "List a skill's revisions, newest first",
                    "tags": [manifest["cassette"]["name"]],
                    "parameters": [
                        {
                            "name": "skill_id",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "limit",
                            "in": "query",
                            "schema": {"type": "integer", "default": 20},
                        },
                    ],
                    "responses": {
                        "200": {
                            "description": "The skill's revision history",
                            "content": {
                                "application/json": {"schema": revision_list_ref}
                            },
                        }
                    },
                },
            },
            f"{prefix}/revisions/{{revision_id}}": {
                "get": {
                    "operationId": "getSkillRevision",
                    "summary": "Fetch one revision and its status",
                    "tags": [manifest["cassette"]["name"]],
                    "parameters": [
                        {
                            "name": "revision_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "The revision",
                            "content": {"application/json": {"schema": revision_ref}},
                        }
                    },
                }
            },
            f"{prefix}/revisions/{{revision_id}}/status": {
                "post": {
                    "operationId": "setSkillRevisionStatus",
                    "summary": "Report the human verdict on a proposed revision",
                    "description": (
                        "The labeling hook hosts call when a suggestion is accepted "
                        "or rejected. A decided label can be re-asserted but never "
                        "flipped (409): accepted/rejected pairs are the corpus a "
                        "GEPA optimizer trains on."
                    ),
                    "tags": [manifest["cassette"]["name"]],
                    "parameters": [
                        {
                            "name": "revision_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {"schema": status_ref}},
                    },
                    "responses": {
                        "200": {
                            "description": "The revision with its new status",
                            "content": {"application/json": {"schema": revision_ref}},
                        }
                    },
                }
            },
            f"{prefix}/evaluate": {
                "post": {
                    "operationId": "evaluateSkill",
                    "summary": "Judge one skill document against captured session evidence",
                    "description": (
                        "Requires a tapes skill_id — every evaluation is anchored "
                        "on a skills-table row, resolved with its provenance "
                        "sessions as seed evidence. An inline candidate bundle, "
                        "when sent, is the proposed document replacing the stored "
                        "content. Finds related sessions via span search, triages "
                        "each transcript, and runs a DSPy judgment over the "
                        "annotated evidence. Returns findings, metrics, a 0..1 "
                        "score, and a pass/revise decision. Host adapters (e.g. "
                        "the OpenClaw Gateway plugin) create a skill row through "
                        "POST /v1/skills for greenfield proposals, then conform "
                        "their events to this shape."
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
