"""Load the saved routing adapter and return validated category predictions."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


class InvalidModelOutput(ValueError):
    """The model response is not one of the approved categories."""


class InputTooLong(ValueError):
    """The rendered input exceeds the configured context limit."""


def load_categories(report_path: Path) -> tuple[str, ...]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    categories = tuple(sorted(report["metrics"]["per_class"]))
    if len(categories) != 77:
        raise ValueError("category reference must contain 77 categories")
    return categories


def adapter_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


@dataclass(frozen=True)
class InferenceSettings:
    adapter_dir: Path = PROJECT_ROOT / "artifacts" / "lora_adapter"
    cache_dir: Path = PROJECT_ROOT / ".cache" / "huggingface"
    report_path: Path = PROJECT_ROOT / "reports" / "lora_adapter_test.json"
    max_input_tokens: int = 1024
    max_new_tokens: int = 32

    @classmethod
    def from_environment(cls) -> "InferenceSettings":
        return cls(
            adapter_dir=Path(os.getenv("TRIAGETUNE_ADAPTER_DIR", cls.adapter_dir)),
            cache_dir=Path(os.getenv("TRIAGETUNE_CACHE_DIR", cls.cache_dir)),
            report_path=Path(os.getenv("TRIAGETUNE_REPORT_PATH", cls.report_path)),
            max_input_tokens=int(os.getenv("TRIAGETUNE_MAX_INPUT_TOKENS", "1024")),
            max_new_tokens=int(os.getenv("TRIAGETUNE_MAX_NEW_TOKENS", "32")),
        )


class RoutingEngine:
    def __init__(self, settings: InferenceSettings) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from src.evaluate import build_system_prompt, choose_device, parse_raw_output

        self.settings = settings
        self.categories = load_categories(settings.report_path)
        self.allowed_categories = set(self.categories)
        self.system_prompt = build_system_prompt(list(self.categories))
        self._parse_raw_output = parse_raw_output
        self._lock = threading.Lock()
        self._torch = torch
        self.adapter_size_bytes = adapter_bytes(settings.adapter_dir)
        self.adapter_weight_bytes = (
            settings.adapter_dir / "adapter_model.safetensors"
        ).stat().st_size
        device, dtype = choose_device(torch)
        self.device = device
        self.dtype = str(dtype).removeprefix("torch.")

        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.adapter_dir, local_files_only=True
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            DEFAULT_MODEL,
            revision=DEFAULT_REVISION,
            cache_dir=settings.cache_dir,
            local_files_only=True,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.model = PeftModel.from_pretrained(base, settings.adapter_dir)
        self.model.to(device)
        self.model.eval()
        self.load_seconds = time.perf_counter() - started

    def info(self) -> dict[str, Any]:
        return {
            "model": DEFAULT_MODEL,
            "model_revision": DEFAULT_REVISION,
            "adapter_loaded": True,
            "category_count": len(self.categories),
            "device": self.device,
            "dtype": self.dtype,
            "model_load_seconds": self.load_seconds,
            "adapter_size_bytes": self.adapter_size_bytes,
            "adapter_weight_bytes": self.adapter_weight_bytes,
            "max_input_tokens": self.settings.max_input_tokens,
            "max_new_tokens": self.settings.max_new_tokens,
            "unknown_request_handling": "not_supported",
            "priority_and_team": "provisional_project_mapping",
        }

    def classify(self, text: str) -> str:
        conversation = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text},
        ]
        rendered = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(rendered, return_tensors="pt")
        if inputs["input_ids"].shape[1] > self.settings.max_input_tokens:
            raise InputTooLong("rendered input exceeds the configured token limit")
        inputs = inputs.to(self.device)
        with self._lock, self._torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.settings.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        output_tokens = generated[:, inputs["input_ids"].shape[1] :]
        raw_output = self.tokenizer.batch_decode(
            output_tokens, skip_special_tokens=True
        )[0]
        parsed = self._parse_raw_output(raw_output, self.allowed_categories)
        if not parsed["label_valid"]:
            raise InvalidModelOutput("model output is not an approved category")
        return parsed["predicted_category"]
