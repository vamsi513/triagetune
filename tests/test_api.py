from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.main import create_app
from src.inference import InputTooLong, InvalidModelOutput, load_categories
from src.routing import build_routes


ALL_CATEGORIES = set(
    load_categories(Path(__file__).resolve().parents[1] / "reports" / "lora_adapter_test.json")
)
TEST_AGENT_KEY = "a" * 32


class FakeEngine:
    allowed_categories = ALL_CATEGORIES

    def info(self):
        return {
            "model": "test-model",
            "model_revision": "test-revision",
            "adapter_loaded": True,
            "category_count": 77,
            "device": "cpu",
            "dtype": "float32",
            "model_load_seconds": 0.1,
            "adapter_size_bytes": 100,
            "adapter_weight_bytes": 80,
            "max_input_tokens": 1024,
            "max_new_tokens": 32,
            "unknown_request_handling": "not_supported",
            "priority_and_team": "not_approved",
        }

    def classify(self, text: str):
        if text == "too many tokens":
            raise InputTooLong("rendered input exceeds the configured token limit")
        if text == "invalid output":
            raise InvalidModelOutput("model output is not an approved category")
        if text == "unexpected category":
            return "invented"
        if text in self.allowed_categories:
            return text
        return "card_swallowed"


def test_health_and_model_info():
    with TestClient(create_app(engine_factory=FakeEngine, enable_provisional_routing=True)) as client:
        assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
        info = client.get("/model-info")
        assert info.status_code == 200
        assert info.json()["category_count"] == 77
        assert info.json()["priority_and_team"] == "provisional_project_mapping"
        assert info.json()["provisional_routing_enabled"] is True


def test_classification_and_request_validation():
    with TestClient(create_app(engine_factory=FakeEngine, enable_provisional_routing=True)) as client:
        assert client.post("/classify", json={"text": "  card swallowed  "}).json() == {
            "category": "card_swallowed",
            "priority": "standard",
            "team": "card_support",
            "routing_policy": "provisional_project_mapping",
        }
        assert client.post("/classify", json={"text": " "}).status_code == 422
        assert client.post("/classify", json={"text": "x" * 2001}).status_code == 422
        assert client.post("/classify", json={"text": "card", "extra": 1}).status_code == 422


def test_invalid_generation_and_context_limit_fail_closed():
    with TestClient(create_app(engine_factory=FakeEngine, enable_provisional_routing=True)) as client:
        assert client.post("/classify", json={"text": "invalid output"}).status_code == 422
        assert client.post("/classify", json={"text": "unexpected category"}).status_code == 422
        assert client.post("/classify", json={"text": "too many tokens"}).status_code == 413


def test_every_approved_category_gets_its_expected_route():
    routes = build_routes(ALL_CATEGORIES)
    with TestClient(create_app(engine_factory=FakeEngine, enable_provisional_routing=True)) as client:
        for category, route in routes.items():
            response = client.post("/classify", json={"text": category})
            assert response.status_code == 200
            assert response.json() == {
                "category": category,
                "priority": route.priority,
                "team": route.team,
                "routing_policy": "provisional_project_mapping",
            }


def test_provisional_routing_is_disabled_without_opt_in(monkeypatch):
    monkeypatch.delenv("TRIAGETUNE_ENABLE_PROVISIONAL_ROUTING", raising=False)
    with TestClient(create_app(engine_factory=FakeEngine)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/model-info").json()["provisional_routing_enabled"] is False
        response = client.post("/classify", json={"text": "card_swallowed"})
        assert response.status_code == 503


def test_provisional_routing_requires_exact_environment_opt_in(monkeypatch):
    monkeypatch.setenv("TRIAGETUNE_ENABLE_PROVISIONAL_ROUTING", "true")
    with TestClient(create_app(engine_factory=FakeEngine)) as client:
        assert client.post("/classify", json={"text": "card_swallowed"}).status_code == 503
    monkeypatch.setenv("TRIAGETUNE_ENABLE_PROVISIONAL_ROUTING", "1")
    with TestClient(create_app(engine_factory=FakeEngine)) as client:
        assert client.post("/classify", json={"text": "card_swallowed"}).status_code == 200


def test_agent_assist_is_disabled_without_opt_in():
    with TestClient(create_app(engine_factory=FakeEngine, enable_agent_assist=False)) as client:
        assert client.get("/model-info").json()["agent_assist_enabled"] is False
        response = client.post("/agent-assist", json={"text": "card_swallowed"})
        assert response.status_code == 503


def test_agent_assist_requires_a_strong_key_and_cannot_enable_routing():
    with pytest.raises(ValueError, match="at least 32"):
        create_app(
            engine_factory=FakeEngine,
            enable_agent_assist=True,
            agent_assist_key="too-short",
        )
    with pytest.raises(ValueError, match="cannot be enabled together"):
        create_app(
            engine_factory=FakeEngine,
            enable_agent_assist=True,
            agent_assist_key=TEST_AGENT_KEY,
            enable_provisional_routing=True,
        )


def test_agent_assist_requires_authentication_and_only_suggests_a_category():
    with TestClient(create_app(
        engine_factory=FakeEngine,
        enable_agent_assist=True,
        agent_assist_key=TEST_AGENT_KEY,
    )) as client:
        info = client.get("/model-info").json()
        assert info["agent_assist_enabled"] is True
        assert info["routing_policy"] == "not_enabled"
        assert info["priority_and_team"] == "not_returned"
        assert client.post("/classify", json={"text": "card_swallowed"}).status_code == 503
        assert client.post("/agent-assist", json={"text": "card_swallowed"}).status_code == 401
        assert client.post(
            "/agent-assist",
            headers={"X-TriageTune-Key": "wrong-key"},
            json={"text": "card_swallowed"},
        ).status_code == 401
        response = client.post(
            "/agent-assist",
            headers={"X-TriageTune-Key": TEST_AGENT_KEY},
            json={"text": "card_swallowed"},
        )
        assert response.status_code == 200
        assert response.json() == {
            "suggested_category": "card_swallowed",
            "suggestion_status": "available",
            "review_required": True,
            "review_status": "pending_human_review",
        }


def test_agent_assist_failures_still_require_human_review():
    with TestClient(create_app(
        engine_factory=FakeEngine,
        enable_agent_assist=True,
        agent_assist_key=TEST_AGENT_KEY,
    )) as client:
        for text, expected_status in (
            ("invalid output", "invalid_model_output"),
            ("unexpected category", "invalid_model_output"),
            ("too many tokens", "input_too_long"),
        ):
            response = client.post(
                "/agent-assist",
                headers={"X-TriageTune-Key": TEST_AGENT_KEY},
                json={"text": text},
            )
            assert response.status_code == 200
            assert response.json() == {
                "suggested_category": None,
                "suggestion_status": expected_status,
                "review_required": True,
                "review_status": "pending_human_review",
            }
