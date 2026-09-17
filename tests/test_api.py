from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.main import create_app
from src.inference import InputTooLong, InvalidModelOutput


class FakeEngine:
    allowed_categories = {"card_swallowed"}

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
        return "card_swallowed"


def test_health_and_model_info():
    with TestClient(create_app(engine_factory=FakeEngine)) as client:
        assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
        info = client.get("/model-info")
        assert info.status_code == 200
        assert info.json()["category_count"] == 77
        assert info.json()["priority_and_team"] == "not_approved"


def test_classification_and_request_validation():
    with TestClient(create_app(engine_factory=FakeEngine)) as client:
        assert client.post("/classify", json={"text": "  card swallowed  "}).json() == {
            "category": "card_swallowed"
        }
        assert client.post("/classify", json={"text": " "}).status_code == 422
        assert client.post("/classify", json={"text": "x" * 2001}).status_code == 422
        assert client.post("/classify", json={"text": "card", "extra": 1}).status_code == 422


def test_invalid_generation_and_context_limit_fail_closed():
    with TestClient(create_app(engine_factory=FakeEngine)) as client:
        assert client.post("/classify", json={"text": "invalid output"}).status_code == 422
        assert client.post("/classify", json={"text": "unexpected category"}).status_code == 422
        assert client.post("/classify", json={"text": "too many tokens"}).status_code == 413
