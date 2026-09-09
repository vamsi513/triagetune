#!/usr/bin/env python3
"""Evaluate an unchanged instruction-tuned model on validation data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from train_baseline import classification_metrics, portable_path, sha256_file


DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=project_root / "data" / "processed" / "validation.csv",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=project_root / "reports" / "base_model_validation_outputs.jsonl",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=project_root / "reports" / "base_model_validation.json",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=project_root / ".cache" / "huggingface",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard an existing output checkpoint instead of resuming it.",
    )
    return parser.parse_args()


def load_validation(path: Path, limit: int | None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"text", "category"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns: {sorted(required)}")
        rows = [{"text": row["text"], "category": row["category"]} for row in reader]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"{path} contains no selected records")
    return rows


def build_system_prompt(labels: list[str]) -> str:
    label_text = ", ".join(labels)
    return (
        "Classify one banking support request. Respond with exactly one JSON object "
        'using this schema: {"category":"one_allowed_category"}. '
        "Do not add markdown, explanation, or extra keys. The allowed categories are: "
        f"{label_text}"
    )


def prompt_hash(system_prompt: str) -> str:
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()


def load_checkpoint(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                row = json.loads(line)
                if row.get("row_index") != len(rows):
                    raise ValueError(
                        f"non-contiguous checkpoint row at line {line_number}: {path}"
                    )
                rows.append(row)
    return rows


def parse_raw_output(raw_output: str, allowed_labels: set[str]) -> dict[str, Any]:
    json_valid = False
    schema_valid = False
    label_valid = False
    predicted_category = None
    try:
        parsed = json.loads(raw_output)
        json_valid = True
    except (json.JSONDecodeError, TypeError):
        parsed = None

    if (
        isinstance(parsed, dict)
        and set(parsed) == {"category"}
        and isinstance(parsed["category"], str)
    ):
        schema_valid = True
        predicted_category = parsed["category"]
        label_valid = predicted_category in allowed_labels
    return {
        "json_valid": json_valid,
        "schema_valid": schema_valid,
        "label_valid": label_valid,
        "predicted_category": predicted_category,
    }


def choose_device(torch_module: Any) -> tuple[str, Any]:
    if torch_module.backends.mps.is_available():
        return "mps", torch_module.float16
    if torch_module.cuda.is_available():
        return "cuda", torch_module.float16
    return "cpu", torch_module.float32


def generate_outputs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    project_root = Path(__file__).resolve().parents[1]
    validation = load_validation(args.validation_file, args.limit)
    labels = sorted({row["category"] for row in validation})
    if args.limit is not None and len(labels) != 77:
        full_validation = load_validation(args.validation_file, None)
        labels = sorted({row["category"] for row in full_validation})
    allowed_labels = set(labels)
    system_prompt = build_system_prompt(labels)
    current_prompt_hash = prompt_hash(system_prompt)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite and args.output_file.exists():
        args.output_file.unlink()
    completed = load_checkpoint(args.output_file)
    resumed_rows = len(completed)
    if len(completed) > len(validation):
        raise ValueError("checkpoint contains more rows than this evaluation")
    for index, row in enumerate(completed):
        if (
            row.get("model") != args.model
            or row.get("prompt_sha256") != current_prompt_hash
            or row.get("text") != validation[index]["text"]
            or row.get("actual_category") != validation[index]["category"]
        ):
            raise ValueError("checkpoint does not match the requested evaluation")

    device, dtype = choose_device(torch)
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    model_load_seconds = time.perf_counter() - load_started
    model_revision = getattr(model.config, "_commit_hash", None)

    generation_seconds = 0.0
    with args.output_file.open("a", encoding="utf-8") as output_handle:
        for start in range(len(completed), len(validation), args.batch_size):
            batch = validation[start : start + args.batch_size]
            conversations = [
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": row["text"]},
                ]
                for row in batch
            ]
            rendered = tokenizer.apply_chat_template(
                conversations, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(rendered, return_tensors="pt", padding=True).to(device)
            generation_started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generation_seconds += time.perf_counter() - generation_started
            new_tokens = generated[:, inputs["input_ids"].shape[1] :]
            raw_outputs = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for offset, (source, raw_output) in enumerate(zip(batch, raw_outputs)):
                parsed = parse_raw_output(raw_output, allowed_labels)
                row = {
                    "row_index": start + offset,
                    "model": args.model,
                    "model_revision": model_revision,
                    "prompt_sha256": current_prompt_hash,
                    "text": source["text"],
                    "actual_category": source["category"],
                    "raw_output": raw_output,
                    **parsed,
                    "correct": parsed["label_valid"]
                    and parsed["predicted_category"] == source["category"],
                }
                output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                completed.append(row)
            output_handle.flush()
            print(f"completed {len(completed)}/{len(validation)}", flush=True)

    runtime = {
        "device": device,
        "dtype": str(dtype).removeprefix("torch."),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "model_load_seconds_this_process": model_load_seconds,
        "generation_seconds_this_process": generation_seconds,
        "resumed_rows": resumed_rows,
    }
    configuration = {
        "model": args.model,
        "model_revision": model_revision,
        "model_parameters_reported": "1.54B",
        "model_license": "Apache-2.0",
        "weights_modified": False,
        "prompt_sha256": current_prompt_hash,
        "prompt": system_prompt,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "parsing": "json.loads on the complete raw output; no extraction or cleanup",
        "validation_file": portable_path(args.validation_file, project_root),
        "validation_sha256": sha256_file(args.validation_file),
        "output_file": portable_path(args.output_file, project_root),
    }
    return completed, {"runtime": runtime, "configuration": configuration}


def build_report(
    rows: list[dict[str, Any]], metadata: dict[str, Any]
) -> dict[str, Any]:
    labels = sorted({row["actual_category"] for row in rows})
    predictions = [
        row["predicted_category"] if row["label_valid"] else "__invalid_output__"
        for row in rows
    ]
    expected = [row["actual_category"] for row in rows]
    metrics = classification_metrics(expected, predictions, labels)
    total = len(rows)
    json_valid = sum(bool(row["json_valid"]) for row in rows)
    schema_valid = sum(bool(row["schema_valid"]) for row in rows)
    label_valid = sum(bool(row["label_valid"]) for row in rows)
    predicted_counts = Counter(predictions)
    return {
        "evaluation": "unchanged_base_model_validation",
        "target": "category",
        "records": total,
        **metadata,
        "metrics": metrics,
        "structured_output": {
            "json_valid": json_valid,
            "json_invalid": total - json_valid,
            "invalid_json_rate": (total - json_valid) / total,
            "schema_valid": schema_valid,
            "schema_invalid": total - schema_valid,
            "invalid_schema_rate": (total - schema_valid) / total,
            "allowed_label_valid": label_valid,
            "allowed_label_invalid": total - label_valid,
            "invalid_allowed_label_rate": (total - label_valid) / total,
        },
        "prediction_counts": dict(sorted(predicted_counts.items())),
    }


def main() -> None:
    args = parse_args()
    rows, metadata = generate_outputs(args)
    report = build_report(rows, metadata)
    args.report_file.parent.mkdir(parents=True, exist_ok=True)
    args.report_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = {
        "records": report["records"],
        "accuracy": report["metrics"]["accuracy"],
        "macro_f1": report["metrics"]["macro_f1"],
        **report["structured_output"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
