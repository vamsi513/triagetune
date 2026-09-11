#!/usr/bin/env python3
"""Validate test artifacts and build the final comparison report."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


REPORTS = (
    ("majority", "Majority class", "majority_test.json"),
    ("tfidf_logreg", "TF-IDF + logistic regression", "tfidf_logreg_test.json"),
    ("base_model", "Unchanged base model", "base_model_test.json"),
    ("lora_adapter", "Saved LoRA adapter", "lora_adapter_test.json"),
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_row_count(path: Path) -> int:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def matrix_total(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.reader(handle)
        next(rows)
        return sum(sum(int(value) for value in row[1:]) for row in rows)


def validate_report(project_root: Path, report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    per_class = metrics["per_class"]
    records = report["records"]
    if records != 3080 or metrics["total"] != records:
        raise ValueError(f"unexpected record total in {report['evaluation']}")
    if len(per_class) != 77 or sum(row["support"] for row in per_class.values()) != records:
        raise ValueError(f"unexpected class coverage in {report['evaluation']}")
    if abs(metrics["accuracy"] - metrics["correct"] / records) > 1e-12:
        raise ValueError(f"accuracy mismatch in {report['evaluation']}")
    for metric_name in ("precision", "recall", "f1"):
        expected = sum(row[metric_name] for row in per_class.values()) / len(per_class)
        if abs(metrics[f"macro_{metric_name}"] - expected) > 1e-12:
            raise ValueError(f"macro {metric_name} mismatch in {report['evaluation']}")

    artifacts = report["artifacts"]
    predictions_name = artifacts.get("predictions_file") or artifacts.get("raw_outputs_file")
    predictions_path = project_root / predictions_name
    matrix_path = project_root / artifacts["confusion_matrix_file"]
    if artifact_row_count(predictions_path) != records:
        raise ValueError(f"prediction count mismatch in {report['evaluation']}")
    if matrix_total(matrix_path) != records:
        raise ValueError(f"confusion-matrix total mismatch in {report['evaluation']}")

    configuration = report.get("configuration", {})
    inputs = report.get("inputs", {})
    return configuration.get("data_sha256") or inputs["test_sha256"]


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    reports_dir = project_root / "reports"
    approaches: list[dict[str, Any]] = []
    test_hashes: set[str] = set()

    for identifier, name, filename in REPORTS:
        report = read_json(reports_dir / filename)
        test_hashes.add(validate_report(project_root, report))
        metrics = report["metrics"]
        structured = report["structured_output"]
        approaches.append(
            {
                "id": identifier,
                "name": name,
                "report": f"reports/{filename}",
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "correct": metrics["correct"],
                "invalid_json_rate": structured["invalid_json_rate"],
                "invalid_allowed_label_rate": structured.get("invalid_allowed_label_rate"),
            }
        )

    if len(test_hashes) != 1:
        raise ValueError("test data fingerprints do not match")
    by_id = {row["id"]: row for row in approaches}
    adapter = by_id["lora_adapter"]
    base = by_id["base_model"]
    classical = by_id["tfidf_logreg"]
    output = {
        "evaluation": "final_test_comparison",
        "stage": "phase_4_stage_1",
        "first_reserved_test_evaluation": True,
        "test_records": 3080,
        "number_of_classes": 77,
        "test_sha256": test_hashes.pop(),
        "approaches": approaches,
        "ranking_by_macro_f1": [
            row["id"]
            for row in sorted(approaches, key=lambda item: item["macro_f1"], reverse=True)
        ],
        "comparisons": {
            "adapter_minus_base_accuracy": adapter["accuracy"] - base["accuracy"],
            "adapter_minus_base_macro_f1": adapter["macro_f1"] - base["macro_f1"],
            "adapter_minus_tfidf_accuracy": adapter["accuracy"] - classical["accuracy"],
            "adapter_minus_tfidf_macro_f1": adapter["macro_f1"] - classical["macro_f1"],
            "adapter_minus_base_invalid_json_rate": (
                adapter["invalid_json_rate"] - base["invalid_json_rate"]
            ),
        },
        "validation": {
            "detailed_reports": len(approaches),
            "prediction_rows_per_approach": 3080,
            "confusion_matrix_total_per_approach": 3080,
            "per_class_entries_per_approach": 77,
        },
    }
    output_path = reports_dir / "final_test_evaluation.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
