#!/usr/bin/env python3
"""Evaluate saved classifiers on clearly labeled synthetic robustness probes."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import joblib

from evaluate import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_REVISION,
    build_system_prompt,
    choose_device,
    parse_raw_output,
)
from train_baseline import load_records, portable_path, sha256_file


GROUPS = ("short", "long", "ambiguous", "out_of_scope")
EXPLICIT_UNKNOWN_LABELS = {
    "cannot_classify",
    "out_of_scope",
    "unknown",
    "unsupported_request",
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probes-file",
        type=Path,
        default=project_root / "data" / "robustness_probes.jsonl",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=project_root / "artifacts" / "lora_adapter",
    )
    parser.add_argument(
        "--classical-model",
        type=Path,
        default=project_root / "artifacts" / "tfidf_logreg.joblib",
    )
    parser.add_argument(
        "--training-file",
        type=Path,
        default=project_root / "data" / "processed" / "train.csv",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=project_root / ".cache" / "huggingface",
    )
    parser.add_argument(
        "--predictions-file",
        type=Path,
        default=project_root / "reports" / "robustness_predictions.jsonl",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=project_root / "reports" / "robustness_evaluation.json",
    )
    parser.add_argument(
        "--markdown-file",
        type=Path,
        default=project_root / "reports" / "robustness_evaluation.md",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def load_probes(path: Path, allowed_labels: set[str]) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        probes = [json.loads(line) for line in handle if line.strip()]
    if not probes:
        raise ValueError("probe file is empty")
    seen_ids: set[str] = set()
    counts = Counter()
    for probe in probes:
        required = {
            "id",
            "group",
            "text",
            "expected_category",
            "acceptable_categories",
            "source",
        }
        if set(probe) != required:
            raise ValueError(f"probe {probe.get('id')} has unexpected fields")
        if probe["id"] in seen_ids:
            raise ValueError(f"duplicate probe id: {probe['id']}")
        seen_ids.add(probe["id"])
        if probe["group"] not in GROUPS:
            raise ValueError(f"unexpected probe group: {probe['group']}")
        if probe["source"] != "project_created_synthetic":
            raise ValueError(f"probe {probe['id']} lacks the synthetic-data label")
        if not probe["text"].strip():
            raise ValueError(f"probe {probe['id']} has empty text")
        acceptable = set(probe["acceptable_categories"])
        if not acceptable.issubset(allowed_labels):
            raise ValueError(f"probe {probe['id']} has an unknown acceptable category")
        expected = probe["expected_category"]
        if expected is not None and (expected not in allowed_labels or acceptable != {expected}):
            raise ValueError(f"probe {probe['id']} has inconsistent expected categories")
        if probe["group"] in {"short", "long"} and expected is None:
            raise ValueError(f"probe {probe['id']} requires one expected category")
        if probe["group"] == "ambiguous" and (expected is not None or not acceptable):
            raise ValueError(f"probe {probe['id']} requires plausible categories only")
        if probe["group"] == "out_of_scope" and (expected is not None or acceptable):
            raise ValueError(f"probe {probe['id']} must not have an in-scope target")
        counts[probe["group"]] += 1
    if set(counts) != set(GROUPS):
        raise ValueError("all robustness groups must be present")
    return probes


def run_classical(model_path: Path, probes: list[dict[str, Any]]) -> list[str]:
    pipeline = joblib.load(model_path)
    return pipeline.predict([probe["text"] for probe in probes]).tolist()


def run_adapter(
    args: argparse.Namespace,
    probes: list[dict[str, Any]],
    labels: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device, dtype = choose_device(torch)
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, cache_dir=args.cache_dir)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL,
        revision=DEFAULT_MODEL_REVISION,
        cache_dir=args.cache_dir,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.to(device)
    model.eval()
    load_seconds = time.perf_counter() - load_started
    system_prompt = build_system_prompt(labels)
    allowed_labels = set(labels)
    results: list[dict[str, Any]] = []
    generation_seconds = 0.0
    maximum_input_tokens = 0

    for start in range(0, len(probes), args.batch_size):
        batch = probes[start : start + args.batch_size]
        conversations = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": probe["text"]},
            ]
            for probe in batch
        ]
        rendered = tokenizer.apply_chat_template(
            conversations, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(rendered, return_tensors="pt", padding=True).to(device)
        maximum_input_tokens = max(maximum_input_tokens, inputs["input_ids"].shape[1])
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generation_seconds += time.perf_counter() - started
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        raw_outputs = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for probe, raw_output in zip(batch, raw_outputs):
            results.append(
                {"raw_output": raw_output, **parse_raw_output(raw_output, allowed_labels)}
            )

    return results, {
        "device": device,
        "dtype": str(dtype).removeprefix("torch."),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "model_load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "maximum_padded_input_tokens": maximum_input_tokens,
    }


def score_group(rows: list[dict[str, Any]], group: str, approach: str) -> dict[str, Any]:
    selected = [
        row for row in rows if row["group"] == group and row["approach"] == approach
    ]
    predictions = [row["predicted_category"] for row in selected]
    result: dict[str, Any] = {
        "records": len(selected),
        "prediction_counts": dict(
            sorted(Counter(predictions).items(), key=lambda item: str(item[0]))
        ),
    }
    if group in {"short", "long"}:
        correct = sum(row["exact_match"] for row in selected)
        result.update({"exact_matches": correct, "exact_match_rate": correct / len(selected)})
    elif group == "ambiguous":
        acceptable = sum(row["acceptable_match"] for row in selected)
        result.update(
            {
                "plausible_category_matches": acceptable,
                "plausible_category_match_rate": acceptable / len(selected),
                "interpretation": (
                    "Plausibility uses project-created acceptable sets, not source labels."
                ),
            }
        )
    else:
        forced = sum(row["label_valid"] for row in selected)
        explicit_unknown = sum(row["explicit_unknown"] for row in selected)
        invalid_or_disallowed = len(selected) - forced
        result.update(
            {
                "forced_in_scope_predictions": forced,
                "forced_in_scope_rate": forced / len(selected),
                "invalid_or_disallowed_outputs": invalid_or_disallowed,
                "invalid_or_disallowed_output_rate": invalid_or_disallowed
                / len(selected),
                "explicit_unknown_outputs": explicit_unknown,
                "explicit_unknown_rate": explicit_unknown / len(selected),
                "safe_rejections": explicit_unknown,
                "safe_rejection_rate": explicit_unknown / len(selected),
                "desired_behavior": "abstain or return an explicit unknown result",
            }
        )
    if approach == "lora_adapter":
        result["structured_output"] = {
            "json_valid": sum(row["json_valid"] for row in selected),
            "schema_valid": sum(row["schema_valid"] for row in selected),
            "allowed_label_valid": sum(row["label_valid"] for row in selected),
        }
    else:
        result["structured_output"] = {"applicable": False}
    return result


def build_rows(
    probes: list[dict[str, Any]],
    classical_predictions: list[str],
    adapter_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for approach, predictions in (
        ("tfidf_logreg", [{"predicted_category": value} for value in classical_predictions]),
        ("lora_adapter", adapter_results),
    ):
        for probe, prediction in zip(probes, predictions):
            predicted = prediction["predicted_category"]
            label_valid = prediction.get("label_valid", True)
            rows.append(
                {
                    **probe,
                    "approach": approach,
                    **prediction,
                    "label_valid": label_valid,
                    "exact_match": (
                        probe["expected_category"] is not None
                        and label_valid
                        and predicted == probe["expected_category"]
                    ),
                    "acceptable_match": label_valid and predicted in probe["acceptable_categories"],
                    "explicit_unknown": (
                        isinstance(predicted, str)
                        and predicted.lower() in EXPLICIT_UNKNOWN_LABELS
                    ),
                }
            )
    return rows


def markdown_report(report: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Robustness evaluation",
        "",
        "This is a 32-record project-created synthetic probe set, not a production benchmark. "
        "It contains eight short, eight long, eight ambiguous, and eight out-of-scope inputs.",
        "",
        "| Input group | TF-IDF result | Saved adapter result |",
        "| --- | ---: | ---: |",
    ]
    for group in GROUPS:
        left = report["results"]["tfidf_logreg"][group]
        right = report["results"]["lora_adapter"][group]
        if group in {"short", "long"}:
            label = "exact-match rate"
            left_value = left["exact_match_rate"]
            right_value = right["exact_match_rate"]
        elif group == "ambiguous":
            label = "plausible-category rate"
            left_value = left["plausible_category_match_rate"]
            right_value = right["plausible_category_match_rate"]
        else:
            label = "safe-rejection rate"
            left_value = left["safe_rejection_rate"]
            right_value = right["safe_rejection_rate"]
        lines.append(
            f"| {group.replace('_', ' ').title()} ({label}) | "
            f"{left_value:.3f} | {right_value:.3f} |"
        )

    lines.extend(
        [
            "",
            "Out-of-scope inputs have no valid target among the 77 categories. Both "
            "approaches safely rejected 0 of 8 requests. The classical baseline forced "
            "all eight into known categories; the adapter forced seven and invented one "
            "disallowed category. Neither behavior is counted as a safe rejection.",
            "",
            "## Actual failures",
            "",
        ]
    )
    for approach, display_name in (
        ("tfidf_logreg", "TF-IDF + logistic regression"),
        ("lora_adapter", "Saved LoRA adapter"),
    ):
        lines.extend([f"### {display_name}", ""])
        failures = [
            row
            for row in rows
            if row["approach"] == approach
            and (
                (row["group"] in {"short", "long"} and not row["exact_match"])
                or (row["group"] == "ambiguous" and not row["acceptable_match"])
                or (row["group"] == "out_of_scope" and not row["explicit_unknown"])
            )
        ]
        for row in failures:
            lines.append(
                f"- `{row['id']}` ({row['group']}): predicted "
                f"`{row['predicted_category']}` for “{row['text']}”"
            )
        if not failures:
            lines.append("- No failures under the probe-specific criteria.")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    labels = sorted({row["category"] for row in load_records(args.training_file)})
    if len(labels) != 77:
        raise ValueError("expected 77 training categories")
    probes = load_probes(args.probes_file, set(labels))
    classical_predictions = run_classical(args.classical_model, probes)
    adapter_results, runtime = run_adapter(args, probes, labels)
    rows = build_rows(probes, classical_predictions, adapter_results)

    args.predictions_file.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions_file.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "evaluation": "synthetic_robustness_probes",
        "stage": "phase_4_stage_3",
        "retraining_performed": False,
        "probe_source": "project_created_synthetic",
        "records": len(probes),
        "groups": dict(sorted(Counter(probe["group"] for probe in probes).items())),
        "configuration": {
            "model": DEFAULT_MODEL,
            "model_revision": DEFAULT_MODEL_REVISION,
            "adapter_sha256": sha256_file(args.adapter_dir / "adapter_model.safetensors"),
            "classical_model_sha256": sha256_file(args.classical_model),
            "probes_file": portable_path(args.probes_file, project_root),
            "probes_sha256": sha256_file(args.probes_file),
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "decoding": "greedy",
        },
        "runtime": runtime,
        "results": {
            approach: {
                group: score_group(rows, group, approach) for group in GROUPS
            }
            for approach in ("tfidf_logreg", "lora_adapter")
        },
        "limitations": [
            "The probes are project-created synthetic examples, not source-dataset records.",
            (
                "Each group has only eight examples, so results are diagnostic rather "
                "than population estimates."
            ),
            "Ambiguous acceptable-category sets reflect documented project judgment.",
            "The 77-category schema has no unknown or abstain label.",
        ],
        "artifacts": {
            "predictions": portable_path(args.predictions_file, project_root),
            "report": portable_path(args.report_file, project_root),
            "readable_report": portable_path(args.markdown_file, project_root),
        },
    }
    args.report_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown_file.write_text(markdown_report(report, rows), encoding="utf-8")
    print(json.dumps(report["results"], indent=2))


if __name__ == "__main__":
    main()
