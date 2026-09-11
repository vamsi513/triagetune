#!/usr/bin/env python3
"""Evaluate the frozen classical baselines on the reserved test split."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib

from train_baseline import classification_metrics, load_records, portable_path, sha256_file


def confusion_data(
    expected: list[str], predicted: list[str], labels: list[str]
) -> tuple[list[list[int]], list[dict[str, int | str]]]:
    positions = {label: index for index, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for actual, prediction in zip(expected, predicted):
        matrix[positions[actual]][positions[prediction]] += 1
    confusions = [
        {"actual": actual, "predicted": prediction, "count": matrix[i][j]}
        for i, actual in enumerate(labels)
        for j, prediction in enumerate(labels)
        if actual != prediction and matrix[i][j]
    ]
    confusions.sort(key=lambda row: (-int(row["count"]), str(row["actual"]), str(row["predicted"])))
    return matrix, confusions


def write_matrix(path: Path, labels: list[str], matrix: list[list[int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["actual\\predicted", *labels])
        for label, counts in zip(labels, matrix):
            writer.writerow([label, *counts])


def write_predictions(
    path: Path,
    records: list[dict[str, str]],
    predictions: list[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_index", "text", "actual", "predicted", "correct"],
            lineterminator="\n",
        )
        writer.writeheader()
        for index, (record, prediction) in enumerate(zip(records, predictions)):
            writer.writerow(
                {
                    "row_index": index,
                    "text": record["text"],
                    "actual": record["category"],
                    "predicted": prediction,
                    "correct": record["category"] == prediction,
                }
            )


def build_report(
    name: str,
    train_file: Path,
    test_file: Path,
    expected: list[str],
    predicted: list[str],
    labels: list[str],
    matrix_file: Path,
    predictions_file: Path,
    project_root: Path,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    matrix, confusions = confusion_data(expected, predicted, labels)
    write_matrix(matrix_file, labels, matrix)
    write_predictions(predictions_file, load_records(test_file), predicted)
    return {
        "evaluation": name,
        "target": "category",
        "records": len(expected),
        "training_records": len(load_records(train_file)),
        "number_of_classes": len(labels),
        "configuration": configuration,
        "inputs": {
            "train_file": portable_path(train_file, project_root),
            "train_sha256": sha256_file(train_file),
            "test_file": portable_path(test_file, project_root),
            "test_sha256": sha256_file(test_file),
        },
        "metrics": classification_metrics(expected, predicted, labels),
        "structured_output": {"applicable": False, "invalid_json_rate": None},
        "top_confusions": confusions[:20],
        "artifacts": {
            "confusion_matrix_file": portable_path(matrix_file, project_root),
            "predictions_file": portable_path(predictions_file, project_root),
        },
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    train_file = project_root / "data" / "processed" / "train.csv"
    test_file = project_root / "data" / "processed" / "test.csv"
    reports_dir = project_root / "reports"
    train_records = load_records(train_file)
    test_records = load_records(test_file)
    expected = [row["category"] for row in test_records]
    labels = sorted({row["category"] for row in train_records} | set(expected))

    counts = Counter(row["category"] for row in train_records)
    largest = max(counts.values())
    majority_category = sorted(label for label, count in counts.items() if count == largest)[0]
    majority_predictions = [majority_category] * len(test_records)
    majority_report = build_report(
        "majority_class_test",
        train_file,
        test_file,
        expected,
        majority_predictions,
        labels,
        reports_dir / "majority_test_confusion_matrix.csv",
        reports_dir / "majority_test_predictions.csv",
        project_root,
        {
            "selection_source": "training_split_only",
            "majority_category": majority_category,
            "majority_training_count": largest,
        },
    )
    (reports_dir / "majority_test.json").write_text(
        json.dumps(majority_report, indent=2) + "\n", encoding="utf-8"
    )

    model_file = project_root / "artifacts" / "tfidf_logreg.joblib"
    pipeline = joblib.load(model_file)
    tfidf_predictions = pipeline.predict([row["text"] for row in test_records]).tolist()
    tfidf_report = build_report(
        "tfidf_logistic_regression_test",
        train_file,
        test_file,
        expected,
        tfidf_predictions,
        labels,
        reports_dir / "tfidf_logreg_test_confusion_matrix.csv",
        reports_dir / "tfidf_logreg_test_predictions.csv",
        project_root,
        {
            "model_file": portable_path(model_file, project_root),
            "model_sha256": sha256_file(model_file),
            "training_source": "unchanged validation-stage fitted pipeline",
        },
    )
    (reports_dir / "tfidf_logreg_test.json").write_text(
        json.dumps(tfidf_report, indent=2) + "\n", encoding="utf-8"
    )

    summaries = {
        "majority": {
            key: majority_report["metrics"][key]
            for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1")
        },
        "tfidf_logreg": {
            key: tfidf_report["metrics"][key]
            for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1")
        },
    }
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
