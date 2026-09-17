#!/usr/bin/env python3
"""Audit frozen rejection rules on independently authored external intents."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import joblib

from src.inference import InferenceSettings, InvalidModelOutput, RoutingEngine


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / ".cache" / "clinc_oos_data_full.json"
EXPECTED_SHA256 = "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0"
SOURCE_REVISION = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
NONBANKING_LABELS = {
    "alarm", "book_flight", "calendar_update", "cook_time", "directions",
    "jump_start", "meal_suggestion", "play_music", "recipe",
    "restaurant_reservation", "smart_home", "timer", "tire_pressure",
    "traffic", "translate", "weather",
}
FINANCIAL_ADJACENT_LABELS = {
    "credit_score", "improve_credit_score", "insurance_change", "taxes",
}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_external() -> list[dict[str, str]]:
    if file_hash(EXTERNAL) != EXPECTED_SHA256:
        raise ValueError("external source checksum does not match the pinned revision")
    data = json.loads(EXTERNAL.read_text(encoding="utf-8"))
    selected = NONBANKING_LABELS | FINANCIAL_ADJACENT_LABELS
    rows = [
        {"text": text, "label": label,
         "group": "financial_adjacent" if label in FINANCIAL_ADJACENT_LABELS else "nonbanking"}
        for text, label in data["test"]
        if label in selected
    ]
    counts = Counter(row["label"] for row in rows)
    if set(counts) != selected or any(count != 30 for count in counts.values()):
        raise ValueError("external challenge set must contain 30 rows per selected label")
    return rows


def run() -> None:
    rows = load_external()
    study = json.loads(
        (ROOT / "reports" / "unknown_rejection_evaluation.json").read_text(encoding="utf-8")
    )
    pipeline = joblib.load(ROOT / "artifacts" / "tfidf_logreg.joblib")
    texts = [row["text"] for row in rows]
    confidence = pipeline.predict_proba(texts).max(axis=1)
    vectorizer = pipeline.named_steps["tfidf"]
    with (ROOT / "data" / "processed" / "train.csv").open(newline="", encoding="utf-8") as handle:
        training_texts = [row["text"] for row in csv.DictReader(handle)]
    similarity = (
        vectorizer.transform(texts) @ vectorizer.transform(training_texts).T
    ).max(axis=1).toarray().ravel()

    score_arrays = {
        "classical_max_probability": confidence,
        "nearest_training_cosine": similarity,
    }
    candidates = {}
    for name, scores in score_arrays.items():
        threshold = study["candidate_signals"][name]["threshold"]
        by_group = defaultdict(Counter)
        for row, score in zip(rows, scores):
            by_group[row["group"]]["rejected" if score < threshold else "accepted"] += 1
        candidates[name] = {
            "frozen_threshold": threshold,
            "rejected": int(sum(score < threshold for score in scores)),
            "accepted": int(sum(score >= threshold for score in scores)),
            "by_group": {group: dict(counts) for group, counts in sorted(by_group.items())},
            "decision_rule": "score below threshold; model-output validity not included",
        }

    sample = []
    for label in sorted(NONBANKING_LABELS | FINANCIAL_ADJACENT_LABELS):
        sample.extend(row for row in rows if row["label"] == label)  # preserve source order
    selected_sample = []
    seen = Counter()
    for row in sample:
        if seen[row["label"]] < 2:
            selected_sample.append(row)
            seen[row["label"]] += 1
    engine = RoutingEngine(InferenceSettings())
    sample_counts = defaultdict(Counter)
    for index, row in enumerate(selected_sample, start=1):
        try:
            engine.classify(row["text"])
            outcome = "forced_banking_category"
        except InvalidModelOutput:
            outcome = "invalid_or_disallowed_output"
        sample_counts[row["group"]][outcome] += 1
        print(f"{index}/{len(selected_sample)} {row['group']}", flush=True)

    report = {
        "evaluation": "external_scope_challenge",
        "source": {
            "repository": "https://github.com/clinc/oos-eval",
            "revision": SOURCE_REVISION,
            "file": "data/data_full.json",
            "sha256": EXPECTED_SHA256,
            "license": "CC BY 3.0",
            "attribution": "Larson et al., An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction, 2019",
        },
        "selection": {
            "split": "test",
            "nonbanking_labels": sorted(NONBANKING_LABELS),
            "financial_adjacent_labels": sorted(FINANCIAL_ADJACENT_LABELS),
            "records_per_label": 30,
            "total_records": len(rows),
            "scope_note": "Label-level selection only; individual examples have not been independently reviewed against BANKING77.",
        },
        "thresholds_selected_on_external_data": False,
        "candidate_scores_without_model_output_check": candidates,
        "saved_model_sample": {
            "selection": "first two source-order records from each selected label",
            "records": len(selected_sample),
            "by_group": {group: dict(counts) for group, counts in sorted(sample_counts.items())},
            "device": engine.device,
            "dtype": engine.dtype,
        },
        "rejection_enabled_in_service": False,
        "limitations": [
            "External intents were labeled for a different application, not this banking router.",
            "Some financially adjacent examples may overlap a supported banking category.",
            "The 40-record model sample is deterministic but small and not a random draw.",
            "This dataset does not represent real requests to this service.",
        ],
    }
    target = ROOT / "reports" / "external_scope_evaluation.json"
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    run()
