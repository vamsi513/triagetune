#!/usr/bin/env python3
"""Measure exploratory rejection signals on the synthetic scope probes."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib

from src.inference import InferenceSettings, InvalidModelOutput, RoutingEngine


ROOT = Path(__file__).resolve().parents[1]
PROBES = ROOT / "data" / "synthetic" / "unknown_request_probes.jsonl"
TRAIN = ROOT / "data" / "processed" / "train.csv"
CLASSICAL = ROOT / "artifacts" / "tfidf_logreg.joblib"
ROWS_REPORT = ROOT / "reports" / "unknown_rejection_predictions.jsonl"
SUMMARY_REPORT = ROOT / "reports" / "unknown_rejection_evaluation.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_threshold(rows: list[dict[str, Any]], signal: str) -> dict[str, Any]:
    values = sorted({row[signal] for row in rows})
    candidates = [values[0] - 1.0]
    candidates += [(left + right) / 2 for left, right in zip(values, values[1:])]
    candidates.append(values[-1] + 1.0)
    ranked = []
    for threshold in candidates:
        unknown_rejected = sum(
            row["scope"] == "out_of_scope"
            and (not row["model_label_valid"] or row[signal] < threshold)
            for row in rows
        )
        known_rejected = sum(
            row["scope"] == "in_scope"
            and (not row["model_label_valid"] or row[signal] < threshold)
            for row in rows
        )
        unknown_count = sum(row["scope"] == "out_of_scope" for row in rows)
        known_count = len(rows) - unknown_count
        balanced_accuracy = (
            unknown_rejected / unknown_count + (known_count - known_rejected) / known_count
        ) / 2
        ranked.append((balanced_accuracy, -known_rejected, unknown_rejected, threshold))
    best = max(ranked)
    return {
        "threshold": best[3],
        "calibration_balanced_accuracy": best[0],
        "selection": "maximize calibration balanced accuracy; then minimize known rejections",
    }


def score_decisions(
    rows: list[dict[str, Any]], signal: str | None, threshold: float | None
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        rejected = not row["model_label_valid"] or (
            signal is not None and row[signal] < threshold
        )
        outcome = (
            "unknown_rejected" if rejected else "unknown_accepted"
        ) if row["scope"] == "out_of_scope" else (
            "known_rejected" if rejected else "known_accepted"
        )
        counts[outcome] += 1
        groups[row["group"]][outcome] += 1
    return {
        "confusion_counts": dict(sorted(counts.items())),
        "group_counts": {key: dict(sorted(value.items())) for key, value in sorted(groups.items())},
        "unknown_rejection_rate": counts["unknown_rejected"]
        / (counts["unknown_rejected"] + counts["unknown_accepted"]),
        "known_false_rejection_rate": counts["known_rejected"]
        / (counts["known_rejected"] + counts["known_accepted"]),
    }


def run() -> None:
    probes = [json.loads(line) for line in PROBES.read_text(encoding="utf-8").splitlines()]
    group_positions: Counter[str] = Counter()
    for probe in probes:
        group_positions[probe["group"]] += 1
        probe["split"] = "calibration" if group_positions[probe["group"]] <= 4 else "held_out"

    pipeline = joblib.load(CLASSICAL)
    texts = [probe["text"] for probe in probes]
    probabilities = pipeline.predict_proba(texts)
    vectorizer = pipeline.named_steps["tfidf"]
    with TRAIN.open(newline="", encoding="utf-8") as handle:
        training_texts = [row["text"] for row in csv.DictReader(handle)]
    training_vectors = vectorizer.transform(training_texts)
    probe_vectors = vectorizer.transform(texts)
    confidence = probabilities.max(axis=1)
    similarity = (probe_vectors @ training_vectors.T).max(axis=1).toarray().ravel()

    engine = RoutingEngine(InferenceSettings())
    rows = []
    inference_started = time.perf_counter()
    for index, (probe, model_confidence, nearest_similarity) in enumerate(
        zip(probes, confidence, similarity), start=1
    ):
        started = time.perf_counter()
        try:
            predicted_category = engine.classify(probe["text"])
            label_valid = True
        except InvalidModelOutput:
            predicted_category = None
            label_valid = False
        row = {
            **probe,
            "model_predicted_category": predicted_category,
            "model_label_valid": label_valid,
            "model_category_correct": (
                label_valid and predicted_category == probe["expected_category"]
                if probe["scope"] == "in_scope" else None
            ),
            "classical_max_probability": float(model_confidence),
            "nearest_training_cosine": float(nearest_similarity),
            "model_inference_seconds": time.perf_counter() - started,
        }
        rows.append(row)
        print(f"{index}/{len(probes)} {probe['group']} {probe['split']}", flush=True)
    inference_seconds = time.perf_counter() - inference_started

    calibration = [row for row in rows if row["split"] == "calibration"]
    held_out = [row for row in rows if row["split"] == "held_out"]
    baseline = {
        "calibration": score_decisions(calibration, None, None),
        "held_out": score_decisions(held_out, None, None),
    }
    candidates = {}
    for signal in ("classical_max_probability", "nearest_training_cosine"):
        selection = choose_threshold(calibration, signal)
        candidates[signal] = {
            **selection,
            "calibration": score_decisions(calibration, signal, selection["threshold"]),
            "held_out": score_decisions(held_out, signal, selection["threshold"]),
        }

    summary = {
        "evaluation": "synthetic_unknown_request_exploration",
        "retraining_performed": False,
        "rejection_enabled_in_service": False,
        "records": len(rows),
        "calibration_records": len(calibration),
        "held_out_records": len(held_out),
        "source": "project_created_synthetic",
        "inputs": {
            "probes_sha256": sha256(PROBES),
            "classical_model_sha256": sha256(CLASSICAL),
            "adapter_weights_sha256": sha256(
                InferenceSettings().adapter_dir / "adapter_model.safetensors"
            ),
        },
        "runtime": {
            "device": engine.device,
            "dtype": engine.dtype,
            "model_load_seconds": engine.load_seconds,
            "total_model_inference_seconds": inference_seconds,
        },
        "model_category_results": {
            "known_exact": sum(row["model_category_correct"] is True for row in rows),
            "known_total": sum(row["scope"] == "in_scope" for row in rows),
            "unknown_forced_known": sum(
                row["scope"] == "out_of_scope" and row["model_label_valid"] for row in rows
            ),
            "unknown_total": sum(row["scope"] == "out_of_scope" for row in rows),
        },
        "invalid_output_only": baseline,
        "candidate_signals": candidates,
        "limitations": [
            "All probes are synthetic and manually authored; this is not production performance.",
            "Each signal uses only 16 calibration and 16 held-out examples.",
            "The held-out examples share authorship and patterns with calibration examples.",
            "No threshold is activated in the service from this exploratory study.",
        ],
    }
    ROWS_REPORT.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    SUMMARY_REPORT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    run()
