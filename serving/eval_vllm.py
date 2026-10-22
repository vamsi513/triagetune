#!/usr/bin/env python3
"""Score the reserved test set through the vLLM server and compare it to the
recorded Hugging Face results.

The prompt, the decoding settings and the label parsing are taken from
src/evaluate.py so a difference in accuracy cannot be explained by this script
scoring differently. The run refuses to start if the system prompt it builds
does not hash to the digest stored in the saved evaluation report.

Nothing here rewrites the recorded Hugging Face results. Output goes to
results/serving_preds.jsonl and results/serving_accuracy.json only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serving.serving_common import (  # noqa: E402
    DEFAULT_ADAPTER_DIR,
    DEFAULT_MODEL,
    DEFAULT_MODEL_REVISION,
    DEFAULT_REFERENCE_OUTPUTS,
    DEFAULT_REFERENCE_REPORT,
    DEFAULT_TEST_FILE,
    LORA_MODULE_NAME,
    PROJECT_ROOT,
    RESULTS_DIR,
    build_system_prompt,
    classification_metrics,
    effective_labels,
    load_reference_predictions,
    load_validation,
    parse_raw_output,
    portable_path,
    prompt_hash,
    reference_prompt_hash,
    sha256_file,
)

# The README reports 80.49% for the saved adapter. A gap wider than this many
# points is treated as a finding rather than as noise.
ACCURACY_TOLERANCE_POINTS = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8800")
    parser.add_argument("--model", default=LORA_MODULE_NAME)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument(
        "--mode",
        choices=("completions", "chat"),
        default="completions",
        help=(
            "completions renders the prompt locally with the adapter tokenizer, "
            "which is what src/evaluate.py does. chat lets the server apply its "
            "own chat template, which is the realistic serving path but can "
            "differ from the recorded run."
        ),
    )
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--reference-report", type=Path, default=DEFAULT_REFERENCE_REPORT
    )
    parser.add_argument(
        "--reference-outputs", type=Path, default=DEFAULT_REFERENCE_OUTPUTS
    )
    parser.add_argument(
        "--preds-file", type=Path, default=RESULTS_DIR / "serving_preds.jsonl"
    )
    parser.add_argument(
        "--accuracy-file", type=Path, default=RESULTS_DIR / "serving_accuracy.json"
    )
    args = parser.parse_args()
    if args.concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    return args


def render_prompts(adapter_dir: Path, system_prompt: str, texts: list[str]) -> list[str]:
    """Render each request with the adapter tokenizer, as src/evaluate.py does."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, local_files_only=True)
    conversations = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]
        for text in texts
    ]
    rendered = tokenizer.apply_chat_template(
        conversations, tokenize=False, add_generation_prompt=True
    )
    if isinstance(rendered, str):
        rendered = [rendered]
    if len(rendered) != len(texts):
        raise ValueError("chat template did not render one prompt per request")
    return list(rendered)


async def fetch_one(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    system_prompt: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    if args.mode == "completions":
        path = "/v1/completions"
        payload: dict[str, Any] = {
            "model": args.model,
            "prompt": record["rendered_prompt"],
            "temperature": 0,
            "max_tokens": args.max_new_tokens,
        }
    else:
        path = "/v1/chat/completions"
        payload = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": record["text"]},
            ],
            "temperature": 0,
            "max_tokens": args.max_new_tokens,
        }

    started = time.perf_counter()
    try:
        response = await client.post(path, json=payload)
    except httpx.HTTPError as error:
        return {
            "latency_seconds": time.perf_counter() - started,
            "status_code": None,
            "request_error": type(error).__name__,
            "raw_output": "",
        }
    latency = time.perf_counter() - started
    if response.status_code != 200:
        return {
            "latency_seconds": latency,
            "status_code": response.status_code,
            "request_error": f"http_{response.status_code}",
            "raw_output": "",
        }
    body = response.json()
    choice = body["choices"][0]
    raw_output = (
        choice["text"] if args.mode == "completions" else choice["message"]["content"]
    )
    usage = body.get("usage") or {}
    return {
        "latency_seconds": latency,
        "status_code": 200,
        "request_error": None,
        "raw_output": raw_output or "",
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


async def run_requests(
    args: argparse.Namespace,
    system_prompt: str,
    allowed_labels: set[str],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Send every record and write predictions in dataset order as they land."""
    rows: list[dict[str, Any]] = []
    args.preds_file.parent.mkdir(parents=True, exist_ok=True)
    limits = httpx.Limits(
        max_connections=args.concurrency, max_keepalive_connections=args.concurrency
    )
    async with httpx.AsyncClient(
        base_url=args.base_url, timeout=args.timeout, limits=limits
    ) as client:
        with args.preds_file.open("w", encoding="utf-8") as handle:
            for start in range(0, len(records), args.concurrency):
                chunk = records[start : start + args.concurrency]
                responses = await asyncio.gather(
                    *(fetch_one(client, args, system_prompt, record) for record in chunk)
                )
                for record, result in zip(chunk, responses):
                    parsed = parse_raw_output(result["raw_output"], allowed_labels)
                    row = {
                        "row_index": record["row_index"],
                        "serving_runtime": "vllm",
                        "mode": args.mode,
                        "model": DEFAULT_MODEL,
                        "model_revision": DEFAULT_MODEL_REVISION,
                        "lora_module": args.model,
                        "text": record["text"],
                        "actual_category": record["actual_category"],
                        "raw_output": result["raw_output"],
                        **parsed,
                        "correct": parsed["label_valid"]
                        and parsed["predicted_category"] == record["actual_category"],
                        "status_code": result["status_code"],
                        "request_error": result["request_error"],
                        "finish_reason": result.get("finish_reason"),
                        "prompt_tokens": result.get("prompt_tokens"),
                        "completion_tokens": result.get("completion_tokens"),
                        "latency_seconds": result["latency_seconds"],
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows.append(row)
                handle.flush()
                print(f"completed {len(rows)}/{len(records)}", flush=True)
    return rows


def compare_with_reference(
    rows: list[dict[str, Any]], reference: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    """Count where vLLM and the recorded Hugging Face run predicted differently."""
    compared = 0
    disagreements = 0
    examples: list[dict[str, Any]] = []
    both_correct = 0
    only_vllm_correct = 0
    only_reference_correct = 0
    neither_correct = 0
    for row in rows:
        saved = reference.get(row["row_index"])
        if saved is None:
            continue
        if saved["text"] != row["text"]:
            raise ValueError(
                f"row {row['row_index']} text differs from the saved prediction"
            )
        compared += 1
        serving_label = row["predicted_category"] if row["label_valid"] else None
        saved_label = saved["predicted_category"] if saved["label_valid"] else None
        if serving_label != saved_label:
            disagreements += 1
            if len(examples) < 25:
                examples.append(
                    {
                        "row_index": row["row_index"],
                        "text": row["text"],
                        "actual_category": row["actual_category"],
                        "vllm_prediction": serving_label,
                        "huggingface_prediction": saved_label,
                        "vllm_raw_output": row["raw_output"],
                        "huggingface_raw_output": saved["raw_output"],
                    }
                )
        if row["correct"] and saved["correct"]:
            both_correct += 1
        elif row["correct"]:
            only_vllm_correct += 1
        elif saved["correct"]:
            only_reference_correct += 1
        else:
            neither_correct += 1
    return {
        "compared_examples": compared,
        "disagreements": disagreements,
        "disagreement_rate": disagreements / compared if compared else None,
        "both_correct": both_correct,
        "only_vllm_correct": only_vllm_correct,
        "only_huggingface_correct": only_reference_correct,
        "neither_correct": neither_correct,
        "disagreement_examples": examples,
    }


def build_report(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    labels: list[str],
    system_prompt: str,
    expected_prompt_hash: str,
    reference: dict[int, dict[str, Any]],
    reference_report: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
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
    errors = Counter(
        row["request_error"] for row in rows if row["request_error"] is not None
    )
    truncated = sum(row.get("finish_reason") == "length" for row in rows)

    reference_accuracy = reference_report["metrics"]["accuracy"]
    reference_macro_f1 = reference_report["metrics"]["macro_f1"]
    accuracy_delta_points = (metrics["accuracy"] - reference_accuracy) * 100
    beyond_tolerance = abs(accuracy_delta_points) > ACCURACY_TOLERANCE_POINTS

    comparison = compare_with_reference(rows, reference)
    comparison["reference_outputs_file"] = portable_path(
        args.reference_outputs, PROJECT_ROOT
    )

    findings: list[str] = []
    if beyond_tolerance:
        findings.append(
            f"vLLM accuracy differs from the recorded Hugging Face accuracy by "
            f"{accuracy_delta_points:+.2f} points, more than the "
            f"{ACCURACY_TOLERANCE_POINTS} point tolerance. See "
            "serving/profile_notes.md for the investigation."
        )
    if errors:
        findings.append(
            f"{sum(errors.values())} of {total} requests failed: {dict(errors)}."
        )
    if truncated:
        findings.append(
            f"{truncated} responses stopped on the token limit rather than on an "
            "end-of-sequence token, so their output may be cut off."
        )

    return {
        "evaluation": "vllm_lora_adapter_test",
        "target": "category",
        "records": total,
        "elapsed_seconds": elapsed_seconds,
        "configuration": {
            "serving_runtime": "vllm",
            "base_url": args.base_url,
            "lora_module": args.model,
            "request_mode": args.mode,
            "model": DEFAULT_MODEL,
            "model_revision": DEFAULT_MODEL_REVISION,
            "adapter_directory": portable_path(args.adapter_dir, PROJECT_ROOT),
            "adapter_sha256": sha256_file(
                args.adapter_dir / "adapter_model.safetensors"
            ),
            "temperature": 0,
            "max_new_tokens": args.max_new_tokens,
            "concurrency": args.concurrency,
            "prompt": system_prompt,
            "prompt_sha256": prompt_hash(system_prompt),
            "prompt_sha256_matches_recorded_run": prompt_hash(system_prompt)
            == expected_prompt_hash,
            "parsing": "json.loads on the complete raw output; no extraction or cleanup",
            "data_split": "test",
            "data_file": portable_path(args.data_file, PROJECT_ROOT),
            "data_sha256": sha256_file(args.data_file),
            "limit": args.limit,
        },
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
        "request_health": {
            "failed_requests": sum(errors.values()),
            "failures_by_kind": dict(sorted(errors.items())),
            "responses_stopped_on_token_limit": truncated,
        },
        "comparison_with_recorded_huggingface_run": {
            "reference_report": portable_path(args.reference_report, PROJECT_ROOT),
            "reference_accuracy": reference_accuracy,
            "reference_macro_f1": reference_macro_f1,
            "vllm_accuracy": metrics["accuracy"],
            "vllm_macro_f1": metrics["macro_f1"],
            "accuracy_delta_points": accuracy_delta_points,
            "accuracy_tolerance_points": ACCURACY_TOLERANCE_POINTS,
            "accuracy_outside_tolerance": beyond_tolerance,
            **comparison,
        },
        "findings": findings,
        "prediction_counts": dict(sorted(Counter(predictions).items())),
        "notes": [
            "The recorded Hugging Face results in reports/ are unchanged.",
            "Scoring reuses parse_raw_output and classification_metrics from src/.",
        ],
    }


def main() -> None:
    args = parse_args()
    records = load_validation(args.data_file, args.limit)
    labels = effective_labels(records)
    if args.limit is not None and len(labels) != 77:
        labels = effective_labels(load_validation(args.data_file, None))
    allowed_labels = set(labels)
    system_prompt = build_system_prompt(labels)
    expected_prompt_hash = reference_prompt_hash(args.reference_report)
    if prompt_hash(system_prompt) != expected_prompt_hash:
        raise SystemExit(
            "the system prompt does not match the recorded evaluation "
            f"(built {prompt_hash(system_prompt)}, recorded {expected_prompt_hash}); "
            "refusing to report a comparison against different prompts"
        )

    requests = [
        {
            "row_index": index,
            "text": row["text"],
            "actual_category": row["category"],
        }
        for index, row in enumerate(records)
    ]
    if args.mode == "completions":
        rendered = render_prompts(
            args.adapter_dir, system_prompt, [row["text"] for row in requests]
        )
        for row, prompt in zip(requests, rendered):
            row["rendered_prompt"] = prompt

    started = time.perf_counter()
    rows = asyncio.run(run_requests(args, system_prompt, allowed_labels, requests))
    elapsed = time.perf_counter() - started

    reference = load_reference_predictions(args.reference_outputs)
    reference_report = json.loads(args.reference_report.read_text(encoding="utf-8"))
    report = build_report(
        args,
        rows,
        labels,
        system_prompt,
        expected_prompt_hash,
        reference,
        reference_report,
        elapsed,
    )
    args.accuracy_file.parent.mkdir(parents=True, exist_ok=True)
    args.accuracy_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    summary = {
        "records": report["records"],
        "accuracy": report["metrics"]["accuracy"],
        "macro_f1": report["metrics"]["macro_f1"],
        "invalid_allowed_label_rate": report["structured_output"][
            "invalid_allowed_label_rate"
        ],
        "accuracy_delta_points": report["comparison_with_recorded_huggingface_run"][
            "accuracy_delta_points"
        ],
        "disagreements": report["comparison_with_recorded_huggingface_run"][
            "disagreements"
        ],
        "failed_requests": report["request_health"]["failed_requests"],
        "findings": report["findings"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
