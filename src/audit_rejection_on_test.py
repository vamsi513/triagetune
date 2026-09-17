#!/usr/bin/env python3
"""Audit frozen synthetic-calibrated rejection thresholds on saved test outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[1]


def run() -> None:
    with (ROOT / "data" / "processed" / "test.csv").open(newline="", encoding="utf-8") as handle:
        test_rows = list(csv.DictReader(handle))
    outputs = [
        json.loads(line)
        for line in (ROOT / "reports" / "lora_adapter_test_outputs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    if len(test_rows) != 3080 or len(outputs) != len(test_rows):
        raise ValueError("test data and saved model outputs do not have the expected size")
    for index, (row, output) in enumerate(zip(test_rows, outputs)):
        if (
            output["row_index"] != index
            or output["text"] != row["text"]
            or output["actual_category"] != row["category"]
        ):
            raise ValueError(f"test record mismatch at row {index}")

    study = json.loads(
        (ROOT / "reports" / "unknown_rejection_evaluation.json").read_text(encoding="utf-8")
    )
    pipeline = joblib.load(ROOT / "artifacts" / "tfidf_logreg.joblib")
    texts = [row["text"] for row in test_rows]
    confidence = pipeline.predict_proba(texts).max(axis=1)
    vectorizer = pipeline.named_steps["tfidf"]
    with (ROOT / "data" / "processed" / "train.csv").open(newline="", encoding="utf-8") as handle:
        train_texts = [row["text"] for row in csv.DictReader(handle)]
    similarity = (
        vectorizer.transform(texts) @ vectorizer.transform(train_texts).T
    ).max(axis=1).toarray().ravel()

    signals = {
        "classical_max_probability": confidence,
        "nearest_training_cosine": similarity,
    }
    candidates = {}
    for signal, scores in signals.items():
        threshold = study["candidate_signals"][signal]["threshold"]
        rejected = [
            bool(not output["label_valid"] or score < threshold)
            for output, score in zip(outputs, scores)
        ]
        accepted = len(outputs) - sum(rejected)
        accepted_correct = sum(
            output["correct"] and not reject for output, reject in zip(outputs, rejected)
        )
        candidates[signal] = {
            "threshold_from_synthetic_calibration": threshold,
            "valid_banking_requests_rejected": sum(rejected),
            "valid_banking_requests_rejected_rate": sum(rejected) / len(outputs),
            "valid_banking_requests_accepted": accepted,
            "accepted_correct_category": accepted_correct,
            "accepted_category_accuracy": accepted_correct / accepted if accepted else None,
            "correct_category_requests_rejected": sum(
                output["correct"] and reject
                for output, reject in zip(outputs, rejected)
            ),
        }
    baseline_rejected = sum(not output["label_valid"] for output in outputs)
    report = {
        "evaluation": "frozen_threshold_audit_on_published_banking_test_split",
        "records": len(outputs),
        "thresholds_selected_on": "16 project-created synthetic calibration requests",
        "threshold_tuning_on_test_performed": False,
        "model_retraining_or_regeneration_performed": False,
        "invalid_output_only_valid_banking_requests_rejected": baseline_rejected,
        "candidate_signals": candidates,
        "interpretation": (
            "Every test request belongs to a known banking category; rejection here is a false "
            "rejection. This audit cannot measure unknown-request detection."
        ),
    }
    (ROOT / "reports" / "unknown_rejection_test_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
