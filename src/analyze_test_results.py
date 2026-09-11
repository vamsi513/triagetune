#!/usr/bin/env python3
"""Measure test uncertainty and analyze saved prediction errors."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


APPROACHES = {
    "majority": {
        "name": "Majority class",
        "predictions": "reports/majority_test_predictions.csv",
        "report": "reports/majority_test.json",
        "format": "csv",
    },
    "tfidf_logreg": {
        "name": "TF-IDF + logistic regression",
        "predictions": "reports/tfidf_logreg_test_predictions.csv",
        "report": "reports/tfidf_logreg_test.json",
        "format": "csv",
    },
    "base_model": {
        "name": "Unchanged base model",
        "predictions": "reports/base_model_test_outputs.jsonl",
        "report": "reports/base_model_test.json",
        "format": "jsonl",
    },
    "lora_adapter": {
        "name": "Saved LoRA adapter",
        "predictions": "reports/lora_adapter_test_outputs.jsonl",
        "report": "reports/lora_adapter_test.json",
        "format": "jsonl",
    },
}
METRIC_NAMES = ("accuracy", "macro_precision", "macro_recall", "macro_f1")
INVALID_LABEL = "__invalid_or_disallowed__"


def load_rows(path: Path, file_format: str) -> list[dict[str, Any]]:
    if file_format == "csv":
        with path.open(newline="", encoding="utf-8") as handle:
            source = list(csv.DictReader(handle))
        return [
            {
                "row_index": int(row["row_index"]),
                "text": row["text"],
                "actual": row["actual"],
                "predicted": row["predicted"],
                "correct": row["correct"] == "True",
                "json_valid": None,
                "schema_valid": None,
                "label_valid": True,
            }
            for row in source
        ]

    with path.open(encoding="utf-8") as handle:
        source = [json.loads(line) for line in handle if line.strip()]
    return [
        {
            "row_index": row["row_index"],
            "text": row["text"],
            "actual": row["actual_category"],
            "predicted": row["predicted_category"] if row["label_valid"] else INVALID_LABEL,
            "correct": row["correct"],
            "json_valid": row["json_valid"],
            "schema_valid": row["schema_valid"],
            "label_valid": row["label_valid"],
        }
        for row in source
    ]


def validate_alignment(rows_by_approach: dict[str, list[dict[str, Any]]]) -> None:
    reference = rows_by_approach["majority"]
    if len(reference) != 3080:
        raise ValueError("expected 3,080 saved test predictions")
    for identifier, rows in rows_by_approach.items():
        if len(rows) != len(reference):
            raise ValueError(f"row-count mismatch for {identifier}")
        for position, (expected, row) in enumerate(zip(reference, rows)):
            identity = (position, expected["text"], expected["actual"])
            if (row["row_index"], row["text"], row["actual"]) != identity:
                raise ValueError(f"row alignment mismatch for {identifier} at {position}")


def metrics_from_matrices(matrices: np.ndarray, class_count: int) -> dict[str, np.ndarray]:
    diagonal = np.arange(class_count)
    true_positive = matrices[:, diagonal, diagonal]
    predicted_total = matrices[:, :, :class_count].sum(axis=1)
    actual_total = matrices.sum(axis=2)
    precision = np.divide(
        true_positive,
        predicted_total,
        out=np.zeros_like(true_positive, dtype=float),
        where=predicted_total != 0,
    )
    recall = np.divide(
        true_positive,
        actual_total,
        out=np.zeros_like(true_positive, dtype=float),
        where=actual_total != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    return {
        "accuracy": true_positive.sum(axis=1) / actual_total.sum(axis=1),
        "macro_precision": precision.mean(axis=1),
        "macro_recall": recall.mean(axis=1),
        "macro_f1": f1.mean(axis=1),
    }


def stratified_bootstrap(
    actual: list[str],
    predictions: dict[str, list[str]],
    samples: int,
    seed: int,
    batch_size: int = 250,
) -> dict[str, dict[str, np.ndarray]]:
    labels = sorted(set(actual))
    label_to_index = {label: index for index, label in enumerate(labels)}
    actual_codes = np.array([label_to_index[label] for label in actual], dtype=np.int16)
    prediction_codes = {
        identifier: np.array(
            [label_to_index.get(label, len(labels)) for label in values], dtype=np.int16
        )
        for identifier, values in predictions.items()
    }
    positions = [np.flatnonzero(actual_codes == index) for index in range(len(labels))]
    if len({len(group) for group in positions}) != 1:
        raise ValueError("stratified bootstrap expects equal class support")
    support = len(positions[0])
    actual_template = np.repeat(np.arange(len(labels), dtype=np.int16), support)
    predicted_class_count = len(labels) + 1
    matrix_size = len(labels) * predicted_class_count
    rng = np.random.default_rng(seed)
    output = {
        identifier: {metric: np.empty(samples, dtype=float) for metric in METRIC_NAMES}
        for identifier in predictions
    }

    for start in range(0, samples, batch_size):
        size = min(batch_size, samples - start)
        bootstrap_indices = np.concatenate(
            [
                group[rng.integers(0, len(group), size=(size, support))]
                for group in positions
            ],
            axis=1,
        )
        replicate_offsets = np.arange(size, dtype=np.int64)[:, None] * matrix_size
        actual_offsets = actual_template[None, :] * predicted_class_count
        for identifier, encoded in prediction_codes.items():
            flat_positions = replicate_offsets + actual_offsets + encoded[bootstrap_indices]
            matrices = np.bincount(
                flat_positions.ravel(), minlength=size * matrix_size
            ).reshape(size, len(labels), predicted_class_count)
            batch_metrics = metrics_from_matrices(matrices, len(labels))
            for metric_name, values in batch_metrics.items():
                output[identifier][metric_name][start : start + size] = values
    return output


def confidence_interval(values: np.ndarray) -> list[float]:
    lower, upper = np.quantile(values, [0.025, 0.975], method="linear")
    return [float(lower), float(upper)]


def error_summary(
    rows: list[dict[str, Any]], report: dict[str, Any], top_count: int = 15
) -> dict[str, Any]:
    errors = [row for row in rows if not row["correct"]]
    confusion_counts = Counter((row["actual"], row["predicted"]) for row in errors)
    top_confusions = []
    for (actual, predicted), count in sorted(
        confusion_counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
    )[:top_count]:
        examples = [
            {"row_index": row["row_index"], "text": row["text"]}
            for row in errors
            if row["actual"] == actual and row["predicted"] == predicted
        ][:3]
        top_confusions.append(
            {"actual": actual, "predicted": predicted, "count": count, "examples": examples}
        )

    per_class = report["metrics"]["per_class"]
    lowest = sorted(per_class.items(), key=lambda item: (item[1]["f1"], item[0]))[:10]
    invalid = sum(row["predicted"] == INVALID_LABEL for row in errors)
    return {
        "total_errors": len(errors),
        "valid_label_classification_errors": len(errors) - invalid,
        "invalid_or_disallowed_outputs": invalid,
        "top_confusions": top_confusions,
        "lowest_f1_classes": [
            {"category": category, **values} for category, values in lowest
        ],
        "zero_f1_class_count": sum(values["f1"] == 0 for values in per_class.values()),
        "classes_below_0_50_f1": sum(values["f1"] < 0.5 for values in per_class.values()),
        "structured_output": report["structured_output"],
    }


def correctness_overlap(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> dict[str, int]:
    left_correct = np.array([row["correct"] for row in left], dtype=bool)
    right_correct = np.array([row["correct"] for row in right], dtype=bool)
    return {
        "both_correct": int(np.sum(left_correct & right_correct)),
        "left_only_correct": int(np.sum(left_correct & ~right_correct)),
        "right_only_correct": int(np.sum(~left_correct & right_correct)),
        "both_incorrect": int(np.sum(~left_correct & ~right_correct)),
    }


def class_comparison(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for category in sorted(left):
        difference = left[category]["f1"] - right[category]["f1"]
        rows.append(
            {
                "category": category,
                "left_f1": left[category]["f1"],
                "right_f1": right[category]["f1"],
                "difference": difference,
            }
        )
    tolerance = 1e-12
    return {
        "left_better_classes": sum(row["difference"] > tolerance for row in rows),
        "right_better_classes": sum(row["difference"] < -tolerance for row in rows),
        "tied_classes": sum(abs(row["difference"]) <= tolerance for row in rows),
        "largest_left_advantages": sorted(
            rows, key=lambda row: (-row["difference"], row["category"])
        )[:10],
        "largest_right_advantages": sorted(
            rows, key=lambda row: (row["difference"], row["category"])
        )[:10],
    }


def paired_difference(
    left_id: str,
    right_id: str,
    reports: dict[str, dict[str, Any]],
    bootstraps: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    output: dict[str, Any] = {"left": left_id, "right": right_id}
    for metric_name in ("accuracy", "macro_f1"):
        point = (
            reports[left_id]["metrics"][metric_name]
            - reports[right_id]["metrics"][metric_name]
        )
        differences = bootstraps[left_id][metric_name] - bootstraps[right_id][metric_name]
        interval = confidence_interval(differences)
        output[metric_name] = {
            "difference": point,
            "confidence_interval_95": interval,
            "interval_excludes_zero": interval[0] > 0 or interval[1] < 0,
        }
    return output


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Final test uncertainty and error analysis",
        "",
        "Intervals use 10,000 deterministic stratified bootstrap samples. Each sample "
        "draws 40 records with replacement from every one of the 77 categories. Paired "
        "comparisons reuse the same sampled rows for both approaches.",
        "",
        "## Metric intervals",
        "",
        "| Approach | Accuracy (95% CI) | Macro F1 (95% CI) |",
        "| --- | ---: | ---: |",
    ]
    for identifier, values in report["confidence_intervals"].items():
        accuracy = values["accuracy"]
        macro_f1 = values["macro_f1"]
        lines.append(
            f"| {APPROACHES[identifier]['name']} | {accuracy['point']:.6f} "
            f"[{accuracy['confidence_interval_95'][0]:.6f}, "
            f"{accuracy['confidence_interval_95'][1]:.6f}] | {macro_f1['point']:.6f} "
            f"[{macro_f1['confidence_interval_95'][0]:.6f}, "
            f"{macro_f1['confidence_interval_95'][1]:.6f}] |"
        )

    lines.extend(["", "## Paired comparisons", ""])
    for comparison in report["paired_comparisons"]:
        left = APPROACHES[comparison["left"]]["name"]
        right = APPROACHES[comparison["right"]]["name"]
        accuracy = comparison["accuracy"]
        macro_f1 = comparison["macro_f1"]
        lines.append(
            f"- {left} minus {right}: accuracy {accuracy['difference']:.6f} "
            f"(95% CI [{accuracy['confidence_interval_95'][0]:.6f}, "
            f"{accuracy['confidence_interval_95'][1]:.6f}]); macro F1 "
            f"{macro_f1['difference']:.6f} (95% CI "
            f"[{macro_f1['confidence_interval_95'][0]:.6f}, "
            f"{macro_f1['confidence_interval_95'][1]:.6f}])."
        )

    lines.extend(["", "## Error concentrations", ""])
    for identifier in ("tfidf_logreg", "lora_adapter", "base_model"):
        analysis = report["error_analysis"][identifier]
        lines.append(f"### {APPROACHES[identifier]['name']}")
        lines.append("")
        lines.append(
            f"Errors: {analysis['total_errors']}; invalid or disallowed outputs: "
            f"{analysis['invalid_or_disallowed_outputs']}; categories below 0.50 F1: "
            f"{analysis['classes_below_0_50_f1']}."
        )
        lines.append("")
        lines.append("Most frequent confusion pairs:")
        lines.append("")
        for row in analysis["top_confusions"][:5]:
            lines.append(
                f"- `{row['actual']}` → `{row['predicted']}`: {row['count']} records"
            )
        weakest = ", ".join(
            f"`{row['category']}` ({row['f1']:.3f})"
            for row in analysis["lowest_f1_classes"][:5]
        )
        lines.extend(["", f"Lowest five class F1 values: {weakest}.", ""])
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    rows_by_approach = {
        identifier: load_rows(project_root / details["predictions"], details["format"])
        for identifier, details in APPROACHES.items()
    }
    validate_alignment(rows_by_approach)
    reports = {
        identifier: json.loads(
            (project_root / details["report"]).read_text(encoding="utf-8")
        )
        for identifier, details in APPROACHES.items()
    }
    actual = [row["actual"] for row in rows_by_approach["majority"]]
    predictions = {
        identifier: [row["predicted"] for row in rows]
        for identifier, rows in rows_by_approach.items()
    }
    samples = 10_000
    seed = 42
    bootstraps = stratified_bootstrap(actual, predictions, samples=samples, seed=seed)

    intervals = {}
    for identifier, metric_samples in bootstraps.items():
        intervals[identifier] = {}
        for metric_name, values in metric_samples.items():
            intervals[identifier][metric_name] = {
                "point": reports[identifier]["metrics"][metric_name],
                "confidence_interval_95": confidence_interval(values),
            }

    output = {
        "analysis": "final_test_uncertainty_and_errors",
        "stage": "phase_4_stage_2",
        "source": "saved_phase_4_stage_1_predictions",
        "retraining_performed": False,
        "records": len(actual),
        "classes": len(set(actual)),
        "bootstrap": {
            "method": "stratified percentile bootstrap with paired row samples",
            "samples": samples,
            "seed": seed,
            "confidence_level": 0.95,
            "records_sampled_per_class": 40,
        },
        "confidence_intervals": intervals,
        "paired_comparisons": [
            paired_difference("tfidf_logreg", "lora_adapter", reports, bootstraps),
            paired_difference("lora_adapter", "base_model", reports, bootstraps),
            paired_difference("tfidf_logreg", "base_model", reports, bootstraps),
        ],
        "error_analysis": {
            identifier: error_summary(rows_by_approach[identifier], reports[identifier])
            for identifier in APPROACHES
        },
        "correctness_overlap": {
            "tfidf_logreg_vs_lora_adapter": correctness_overlap(
                rows_by_approach["tfidf_logreg"], rows_by_approach["lora_adapter"]
            ),
            "lora_adapter_vs_base_model": correctness_overlap(
                rows_by_approach["lora_adapter"], rows_by_approach["base_model"]
            ),
        },
        "adapter_vs_tfidf_per_class": class_comparison(
            reports["lora_adapter"]["metrics"]["per_class"],
            reports["tfidf_logreg"]["metrics"]["per_class"],
        ),
        "artifacts": {
            "machine_readable": "reports/test_error_analysis.json",
            "human_readable": "reports/test_error_analysis.md",
        },
    }
    reports_dir = project_root / "reports"
    (reports_dir / "test_error_analysis.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    (reports_dir / "test_error_analysis.md").write_text(
        markdown_report(output), encoding="utf-8"
    )
    print(json.dumps({"status": "complete", **output["bootstrap"]}, indent=2))


if __name__ == "__main__":
    main()
