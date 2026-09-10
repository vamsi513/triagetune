#!/usr/bin/env python3
"""Fine-tune and save a response-only LoRA routing adapter."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import torch
import transformers
from mlflow import MlflowClient
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
MAX_SEQUENCE_LENGTH = 544
SEED = 42


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, default=project_root / "data" / "processed" / "train.csv")
    parser.add_argument("--validation-file", type=Path, default=project_root / "data" / "processed" / "validation.csv")
    parser.add_argument("--cache-dir", type=Path, default=project_root / ".cache" / "huggingface")
    parser.add_argument("--output-dir", type=Path, default=project_root / "checkpoints" / "lora_training")
    parser.add_argument("--adapter-dir", type=Path, default=project_root / "artifacts" / "lora_adapter")
    parser.add_argument("--report-file", type=Path, default=project_root / "reports" / "lora_training.json")
    parser.add_argument("--history-file", type=Path, default=project_root / "reports" / "lora_training_history.json")
    parser.add_argument("--tracking-dir", type=Path, default=project_root / "mlruns")
    parser.add_argument("--resume-from-checkpoint", type=Path, help="Resume an interrupted full run from an existing checkpoint.")
    parser.add_argument("--smoke-test", action="store_true", help="Run two disposable optimizer steps without saving artifacts.")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"text", "category"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns: {sorted(required)}")
        rows = [{"text": row["text"], "category": row["category"]} for row in reader]
    if not rows:
        raise ValueError(f"{path} contains no records")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def system_prompt(labels: list[str]) -> str:
    return (
        "Classify one banking support request. Respond with exactly one JSON object "
        'using this schema: {"category":"one_allowed_category"}. '
        "Do not add markdown, explanation, or extra keys. The allowed categories are: "
        + ", ".join(labels)
    )


class ResponseOnlyDataset(Dataset[dict[str, list[int]]]):
    """Tokenized conversations with prompt tokens excluded from the loss."""

    def __init__(self, rows: list[dict[str, str]], tokenizer: Any, prompt: str, max_length: int) -> None:
        self.examples: list[dict[str, list[int]]] = []
        self.lengths: list[int] = []
        self.response_lengths: list[int] = []
        for row in rows:
            prompt_messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": row["text"]},
            ]
            full_messages = [
                *prompt_messages,
                {
                    "role": "assistant",
                    "content": json.dumps({"category": row["category"]}, separators=(",", ":")),
                },
            ]
            prompt_ids = tokenizer.apply_chat_template(
                prompt_messages, tokenize=True, add_generation_prompt=True
            )["input_ids"]
            full_ids = tokenizer.apply_chat_template(
                full_messages, tokenize=True, add_generation_prompt=False
            )["input_ids"]
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise ValueError("chat template response boundary is not a stable prefix")
            if len(full_ids) > max_length:
                raise ValueError(f"encoded record has {len(full_ids)} tokens, exceeding {max_length}")
            response_length = len(full_ids) - len(prompt_ids)
            if response_length <= 0:
                raise ValueError("encoded record contains no supervised response tokens")
            self.examples.append(
                {
                    "input_ids": full_ids,
                    "attention_mask": [1] * len(full_ids),
                    "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :],
                }
            )
            self.lengths.append(len(full_ids))
            self.response_lengths.append(response_length)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


@dataclass
class ResponseOnlyCollator:
    pad_token_id: int
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        longest = max(len(item["input_ids"]) for item in features)
        padded_length = math.ceil(longest / self.pad_to_multiple_of) * self.pad_to_multiple_of
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            pad_count = padded_length - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [self.pad_token_id] * pad_count)
            batch["attention_mask"].append(item["attention_mask"] + [0] * pad_count)
            batch["labels"].append(item["labels"] + [-100] * pad_count)
        return {name: torch.tensor(values, dtype=torch.long) for name, values in batch.items()}


class WallClockCallback(TrainerCallback):
    def __init__(self, limit_seconds: float) -> None:
        self.limit_seconds = limit_seconds
        self.started_at: float | None = None
        self.stopped_for_time = False

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        self.started_at = time.monotonic()
        return control

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if self.started_at is not None and time.monotonic() - self.started_at >= self.limit_seconds:
            control.should_training_stop = True
            control.should_save = True
            self.stopped_for_time = True
        return control


class MlflowMetricCallback(TrainerCallback):
    def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kwargs: Any) -> Any:
        numeric = {
            key: float(value)
            for key, value in (logs or {}).items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        }
        if numeric and mlflow.active_run() is not None:
            mlflow.log_metrics(numeric, step=int(state.global_step))
        return control


def choose_device() -> tuple[str, torch.dtype]:
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def summarize_lengths(values: list[int]) -> dict[str, int]:
    ordered = sorted(values)
    return {
        "minimum": ordered[0],
        "median": ordered[len(ordered) // 2],
        "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "p99": ordered[math.ceil(0.99 * len(ordered)) - 1],
        "maximum": ordered[-1],
    }


def build_training_arguments(output_dir: Path, smoke_test: bool, device: str) -> TrainingArguments:
    common: dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 30,
        "optim": "adamw_torch",
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "fp16": device == "cuda",
        "seed": SEED,
        "data_seed": SEED,
        "report_to": "none",
        "remove_unused_columns": False,
        "dataloader_pin_memory": False,
        "logging_first_step": True,
    }
    if smoke_test:
        common.update(max_steps=2, eval_strategy="no", save_strategy="no", logging_steps=1)
    else:
        common.update(
            num_train_epochs=2,
            eval_strategy="steps",
            eval_steps=250,
            save_strategy="steps",
            save_steps=250,
            save_total_limit=2,
            save_only_model=True,
            logging_steps=25,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
    return TrainingArguments(**common)


def file_manifest(directory: Path, project_root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": relative_path(path, project_root),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]


def run(args: argparse.Namespace) -> None:
    project_root = Path(__file__).resolve().parents[1]
    set_seed(SEED)
    train_rows = load_records(args.train_file)
    validation_rows = load_records(args.validation_file)
    labels = sorted({row["category"] for row in train_rows})
    if len(labels) != 77 or set(labels) != {row["category"] for row in validation_rows}:
        raise ValueError("training and validation files must contain the same 77 categories")
    if args.smoke_test:
        train_rows = train_rows[:32]
        validation_rows = validation_rows[:16]

    device, dtype = choose_device()
    if device == "cpu" and not args.smoke_test:
        raise RuntimeError("the full run requires an available accelerated device")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, cache_dir=args.cache_dir
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt = system_prompt(labels)
    train_dataset = ResponseOnlyDataset(train_rows, tokenizer, prompt, MAX_SEQUENCE_LENGTH)
    validation_dataset = ResponseOnlyDataset(validation_rows, tokenizer, prompt, MAX_SEQUENCE_LENGTH)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        cache_dir=args.cache_dir,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    model.enable_input_require_grads()
    trainable_parameters, total_parameters = model.get_nb_trainable_parameters()
    print(
        json.dumps(
            {
                "mode": "smoke_test" if args.smoke_test else "full_training",
                "device": device,
                "dtype": str(dtype).removeprefix("torch."),
                "training_records": len(train_dataset),
                "validation_records": len(validation_dataset),
                "trainable_parameters": trainable_parameters,
                "total_parameters": total_parameters,
            }
        ),
        flush=True,
    )

    temporary_output: tempfile.TemporaryDirectory[str] | None = None
    if args.smoke_test:
        temporary_output = tempfile.TemporaryDirectory(prefix="triagetune-smoke-")
        output_dir = Path(temporary_output.name)
    else:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

    training_args = build_training_arguments(output_dir, args.smoke_test, device)
    wall_clock = WallClockCallback(3 * 60 * 60)
    callbacks: list[TrainerCallback] = [wall_clock]
    if not args.smoke_test:
        callbacks.extend(
            [
                EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.002),
                MlflowMetricCallback(),
            ]
        )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=ResponseOnlyCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    if args.smoke_test:
        started = time.perf_counter()
        result = trainer.train()
        print(
            json.dumps(
                {
                    "smoke_test": "passed",
                    "optimizer_steps": trainer.state.global_step,
                    "train_loss": result.metrics.get("train_loss"),
                    "seconds": time.perf_counter() - started,
                    "saved_artifacts": False,
                }
            ),
            flush=True,
        )
        temporary_output.cleanup()
        return

    args.tracking_dir.mkdir(parents=True, exist_ok=True)
    tracking_database = args.tracking_dir / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{tracking_database.resolve()}")
    client = MlflowClient()
    if client.get_experiment_by_name("triagetune") is None:
        client.create_experiment(
            "triagetune",
            artifact_location=(args.tracking_dir / "artifacts").resolve().as_uri(),
        )
    mlflow.set_experiment("triagetune")
    run_started = time.perf_counter()
    with mlflow.start_run(run_name="banking77-lora") as active_run:
        mlflow.log_params(
            {
                "model": MODEL_NAME,
                "model_revision": MODEL_REVISION,
                "train_records": len(train_dataset),
                "validation_records": len(validation_dataset),
                "max_sequence_length": MAX_SEQUENCE_LENGTH,
                "lora_rank": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "target_modules": "q_proj,k_proj,v_proj,o_proj",
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 8,
                "effective_batch_size": 16,
                "learning_rate": 2e-4,
                "scheduler": "cosine",
                "warmup_steps": 30,
                "epochs": 2,
                "seed": SEED,
                "device": device,
                "dtype": str(dtype).removeprefix("torch."),
                "response_only_loss": True,
            }
        )
        train_result = trainer.train(
            resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
        )
        if wall_clock.stopped_for_time and trainer.state.best_metric is not None:
            evaluation = {"eval_loss": trainer.state.best_metric}
        else:
            evaluation = trainer.evaluate()
        elapsed_seconds = time.perf_counter() - run_started

        if args.adapter_dir.exists():
            shutil.rmtree(args.adapter_dir)
        args.adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.adapter_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.adapter_dir)
        manifest = file_manifest(args.adapter_dir, project_root)
        if not any(item["path"].endswith("adapter_model.safetensors") for item in manifest):
            raise RuntimeError("adapter weights were not saved")
        forbidden_weight_names = {"model.safetensors", "pytorch_model.bin"}
        if any(Path(item["path"]).name in forbidden_weight_names for item in manifest):
            raise RuntimeError("a complete model weight file was written unexpectedly")

        report = {
            "run": "banking77_lora_training",
            "mlflow": {
                "experiment": "triagetune",
                "run_id": active_run.info.run_id,
                "tracking_directory": relative_path(args.tracking_dir, project_root),
            },
            "runtime": {
                "device": device,
                "dtype": str(dtype).removeprefix("torch."),
                "automatic_mixed_precision": device == "cuda",
                "elapsed_seconds": elapsed_seconds,
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
            },
            "data": {
                "train_file": relative_path(args.train_file, project_root),
                "train_sha256": sha256_file(args.train_file),
                "training_records": len(train_dataset),
                "validation_file": relative_path(args.validation_file, project_root),
                "validation_sha256": sha256_file(args.validation_file),
                "validation_records": len(validation_dataset),
                "categories": len(labels),
                "sequence_lengths": summarize_lengths(train_dataset.lengths),
                "response_lengths": summarize_lengths(train_dataset.response_lengths),
            },
            "configuration": {
                "model": MODEL_NAME,
                "model_revision": MODEL_REVISION,
                "max_sequence_length": MAX_SEQUENCE_LENGTH,
                "response_only_loss": True,
                "prompt_tokens_masked": True,
                "lora": {
                    "rank": 16,
                    "alpha": 32,
                    "dropout": 0.05,
                    "bias": "none",
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                },
                "trainable_parameters": trainable_parameters,
                "total_parameters_with_adapter": total_parameters,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 8,
                "effective_batch_size": 16,
                "learning_rate": 2e-4,
                "scheduler": "cosine",
                "warmup_steps": 30,
                "weight_decay": 0.01,
                "gradient_clip_norm": 1.0,
                "gradient_checkpointing": True,
                "maximum_epochs": 2,
                "evaluation_steps": 250,
                "checkpoint_steps": 250,
                "early_stopping_patience": 2,
                "early_stopping_minimum_improvement": 0.002,
                "wall_clock_limit_seconds": 10800,
                "seed": SEED,
            },
            "results": {
                "optimizer_steps": trainer.state.global_step,
                "completed_epochs": trainer.state.epoch,
                "train_loss": train_result.metrics.get("train_loss"),
                "validation_loss": evaluation.get("eval_loss"),
                "best_validation_loss": trainer.state.best_metric,
                "best_checkpoint": relative_path(Path(trainer.state.best_model_checkpoint), project_root)
                if trainer.state.best_model_checkpoint
                else None,
                "stopped_for_wall_clock_limit": wall_clock.stopped_for_time,
            },
            "saved_model": {
                "format": "LoRA adapter and tokenizer",
                "directory": relative_path(args.adapter_dir, project_root),
                "complete_base_model_weights_saved": False,
                "files": manifest,
                "total_bytes": sum(item["bytes"] for item in manifest),
            },
        }
        args.report_file.parent.mkdir(parents=True, exist_ok=True)
        args.report_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        args.history_file.write_text(json.dumps(trainer.state.log_history, indent=2) + "\n", encoding="utf-8")
        mlflow.log_metrics(
            {
                "final_train_loss": float(train_result.metrics["train_loss"]),
                "final_validation_loss": float(evaluation["eval_loss"]),
                "elapsed_seconds": elapsed_seconds,
            },
            step=int(trainer.state.global_step),
        )
        mlflow.log_artifact(str(args.report_file), artifact_path="reports")
        mlflow.log_artifact(str(args.history_file), artifact_path="reports")
        mlflow.log_artifacts(str(args.adapter_dir), artifact_path="adapter")
        print(json.dumps(report["results"], indent=2), flush=True)
        print(json.dumps(report["saved_model"], indent=2), flush=True)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
