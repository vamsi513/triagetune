"""Local HTTP service for saved banking-intent classification."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Annotated, Callable, Literal

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
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
    routing_policy: Literal["provisional_project_mapping"]
    provisional_routing_enabled: bool


def create_app(
    engine_factory: Callable[[], RoutingEngine] | None = None,
    enable_provisional_routing: bool | None = None,
) -> FastAPI:
    if engine_factory is None:
        engine_factory = lambda: RoutingEngine(InferenceSettings.from_environment())
    if enable_provisional_routing is None:
        enable_provisional_routing = os.getenv("TRIAGETUNE_ENABLE_PROVISIONAL_ROUTING") == "1"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = await run_in_threadpool(engine_factory)
        app.state.routes = build_routes(app.state.engine.allowed_categories)
        yield
        del app.state.engine
        del app.state.routes

    service = FastAPI(title="TriageTune", version="0.1.0", lifespan=lifespan)

    @service.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", model_loaded=True)

    @service.get("/model-info", response_model=ModelInfoResponse)
    async def model_info() -> ModelInfoResponse:
        return ModelInfoResponse.model_validate({
            **service.state.engine.info(),
            "priority_and_team": "provisional_project_mapping",
            "routing_policy": "provisional_project_mapping",
            "provisional_routing_enabled": enable_provisional_routing,
        })

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
