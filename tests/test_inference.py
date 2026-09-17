from __future__ import annotations

import contextlib
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import parse_raw_output
from src.inference import InferenceSettings, InputTooLong, InvalidModelOutput, RoutingEngine


class FakeTokenIds:
    def __init__(self, length: int):
        self.shape = (1, length)


class FakeInputs(dict):
    def to(self, device: str):
        assert device == "cpu"
        return self


class FakeGenerated:
    def __getitem__(self, key):
        assert key[0] == slice(None)
        return "output tokens"


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self, output: str, input_length: int):
        self.output = output
        self.input_length = input_length

    def apply_chat_template(self, conversation, tokenize: bool, add_generation_prompt: bool):
        assert conversation[1]["content"] == "My PIN is blocked."
        assert not tokenize and add_generation_prompt
        return "rendered prompt"

    def __call__(self, rendered: str, return_tensors: str):
        assert rendered == "rendered prompt" and return_tensors == "pt"
        return FakeInputs(input_ids=FakeTokenIds(self.input_length))

    def batch_decode(self, output_tokens, skip_special_tokens: bool):
        assert output_tokens == "output tokens" and skip_special_tokens
        return [self.output]


class FakeModel:
    def __init__(self):
        self.calls = 0

    def generate(self, **kwargs):
        assert kwargs["do_sample"] is False
        assert kwargs["max_new_tokens"] == 32
        self.calls += 1
        return FakeGenerated()


class FakeTorch:
    @staticmethod
    def inference_mode():
        return contextlib.nullcontext()


def make_engine(output: str, input_length: int = 5):
    engine = RoutingEngine.__new__(RoutingEngine)
    engine.settings = InferenceSettings(max_input_tokens=6)
    engine.system_prompt = "Pick one category."
    engine.allowed_categories = {"pin_blocked"}
    engine._parse_raw_output = parse_raw_output
    engine._lock = threading.Lock()
    engine._torch = FakeTorch()
    engine.device = "cpu"
    engine.tokenizer = FakeTokenizer(output, input_length)
    engine.model = FakeModel()
    return engine


def test_inference_returns_only_approved_category():
    engine = make_engine('{"category":"pin_blocked"}')
    assert engine.classify("My PIN is blocked.") == "pin_blocked"
    assert engine.model.calls == 1


@pytest.mark.parametrize("output", ['{"category":"invented"}', "not json"])
def test_inference_rejects_unapproved_or_malformed_output(output: str):
    engine = make_engine(output)
    with pytest.raises(InvalidModelOutput):
        engine.classify("My PIN is blocked.")


def test_inference_rejects_oversized_rendered_input_before_generation():
    engine = make_engine('{"category":"pin_blocked"}', input_length=7)
    with pytest.raises(InputTooLong):
        engine.classify("My PIN is blocked.")
    assert engine.model.calls == 0
