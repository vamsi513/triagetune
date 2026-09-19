"""Local HTTP service for saved banking-intent classification."""

from __future__ import annotations

import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Callable, Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.security import APIKeyHeader
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from src.inference import (
    InferenceSettings,
    InputTooLong,
    InvalidModelOutput,
    RoutingEngine,
)
from src.routing import build_routes


class ClassifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class ClassifyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: str = Field(min_length=1)
    priority: Literal["urgent", "standard", "low"]
    team: Literal[
        "card_support", "payments", "transfers", "cash_atm",
        "account_identity", "top_up", "foreign_exchange",
    ]
    routing_policy: Literal["provisional_project_mapping"]


class AgentAssistResponse(BaseModel):
    suggested_category: str | None
    suggestion_status: Literal["available", "invalid_model_output", "input_too_long"]
    review_required: Literal[True]
    review_status: Literal["pending_human_review"]


class AgentAssistCategoriesResponse(BaseModel):
    categories: list[str]
    review_required: Literal[True]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model: str
    model_revision: str
    adapter_loaded: bool
    category_count: int
    device: str
    dtype: str
    model_load_seconds: float
    adapter_size_bytes: int
    adapter_weight_bytes: int
    max_input_tokens: int
    max_new_tokens: int
    unknown_request_handling: str
    priority_and_team: str
    routing_policy: Literal["provisional_project_mapping", "not_enabled"]
    provisional_routing_enabled: bool
    agent_assist_enabled: bool


AGENT_ASSIST_KEY_HEADER = APIKeyHeader(name="X-TriageTune-Key", auto_error=False)
REVIEWER_DIR = Path(__file__).resolve().parent / "reviewer"


def create_app(
    engine_factory: Callable[[], RoutingEngine] | None = None,
    enable_provisional_routing: bool | None = None,
    enable_agent_assist: bool | None = None,
    agent_assist_key: str | None = None,
) -> FastAPI:
    if engine_factory is None:
        engine_factory = lambda: RoutingEngine(InferenceSettings.from_environment())
    if enable_provisional_routing is None:
        enable_provisional_routing = os.getenv("TRIAGETUNE_ENABLE_PROVISIONAL_ROUTING") == "1"
    if enable_agent_assist is None:
        enable_agent_assist = os.getenv("TRIAGETUNE_ENABLE_AGENT_ASSIST") == "1"
    if enable_agent_assist and enable_provisional_routing:
        raise ValueError("agent assist and provisional routing cannot be enabled together")
    if enable_agent_assist:
        agent_assist_key = agent_assist_key or os.getenv("TRIAGETUNE_AGENT_ASSIST_KEY")
        if not agent_assist_key or len(agent_assist_key) < 32 or not agent_assist_key.isascii():
            raise ValueError("agent assist requires an ASCII key of at least 32 characters")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = await run_in_threadpool(engine_factory)
        app.state.routes = build_routes(app.state.engine.allowed_categories)
        yield
        del app.state.engine
        del app.state.routes

    service = FastAPI(title="TriageTune", version="0.2.0", lifespan=lifespan)

    def require_agent_assist(key: str | None) -> None:
        if not enable_agent_assist:
            raise HTTPException(status_code=503, detail="agent assist is disabled")
        if key is None or not secrets.compare_digest(key, agent_assist_key):
            raise HTTPException(status_code=401, detail="invalid agent-assist credential")

    @service.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", model_loaded=True)

    @service.get("/model-info", response_model=ModelInfoResponse)
    async def model_info() -> ModelInfoResponse:
        return ModelInfoResponse.model_validate({
            **service.state.engine.info(),
            "priority_and_team": (
                "provisional_project_mapping" if enable_provisional_routing else "not_returned"
            ),
            "routing_policy": (
                "provisional_project_mapping" if enable_provisional_routing else "not_enabled"
            ),
            "provisional_routing_enabled": enable_provisional_routing,
            "agent_assist_enabled": enable_agent_assist,
        })

    @service.post("/agent-assist", response_model=AgentAssistResponse)
    async def agent_assist(
        request: ClassifyRequest,
        key: Annotated[str | None, Depends(AGENT_ASSIST_KEY_HEADER)],
    ) -> AgentAssistResponse:
        require_agent_assist(key)
        try:
            category = await run_in_threadpool(service.state.engine.classify, request.text)
        except InputTooLong:
            return AgentAssistResponse(
                suggested_category=None,
                suggestion_status="input_too_long",
                review_required=True,
                review_status="pending_human_review",
            )
        except InvalidModelOutput:
            category = None
        if category not in service.state.engine.allowed_categories:
            category = None
        return AgentAssistResponse(
            suggested_category=category,
            suggestion_status="available" if category else "invalid_model_output",
            review_required=True,
            review_status="pending_human_review",
        )

    @service.get("/agent-assist/categories", response_model=AgentAssistCategoriesResponse)
    async def agent_assist_categories(
        key: Annotated[str | None, Depends(AGENT_ASSIST_KEY_HEADER)],
    ) -> AgentAssistCategoriesResponse:
        require_agent_assist(key)
        return AgentAssistCategoriesResponse(
            categories=sorted(service.state.engine.allowed_categories),
            review_required=True,
        )

    @service.get("/reviewer", include_in_schema=False)
    async def reviewer() -> FileResponse:
        if not enable_agent_assist:
            raise HTTPException(status_code=503, detail="agent assist is disabled")
        return FileResponse(
            REVIEWER_DIR / "index.html",
            media_type="text/html",
            headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": (
                    "default-src 'none'; script-src 'self'; style-src 'self'; "
                    "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
                ),
                "Referrer-Policy": "no-referrer",
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
            },
        )

    @service.get("/reviewer/reviewer.css", include_in_schema=False)
    async def reviewer_css() -> FileResponse:
        if not enable_agent_assist:
            raise HTTPException(status_code=503, detail="agent assist is disabled")
        return FileResponse(
            REVIEWER_DIR / "reviewer.css",
            media_type="text/css",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    @service.get("/reviewer/reviewer.js", include_in_schema=False)
    async def reviewer_js() -> FileResponse:
        if not enable_agent_assist:
            raise HTTPException(status_code=503, detail="agent assist is disabled")
        return FileResponse(
            REVIEWER_DIR / "reviewer.js",
            media_type="text/javascript",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    @service.post("/classify", response_model=ClassifyResponse)
    async def classify(request: ClassifyRequest) -> ClassifyResponse:
        if not enable_provisional_routing:
            raise HTTPException(
                status_code=503,
                detail="provisional routing is disabled; explicit local opt-in is required",
            )
        try:
            category = await run_in_threadpool(service.state.engine.classify, request.text)
        except InputTooLong as error:
            raise HTTPException(status_code=413, detail=str(error)) from error
        except InvalidModelOutput as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        if category not in service.state.engine.allowed_categories:
            raise HTTPException(status_code=422, detail="model output is not an approved category")
        route = service.state.routes[category]
        return ClassifyResponse(
            category=category,
            priority=route.priority,
            team=route.team,
            routing_policy="provisional_project_mapping",
        )

    return service


app = create_app()
