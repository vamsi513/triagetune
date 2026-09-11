from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate import output_lock, write_confusion_matrix
from train_baseline import classification_metrics


def test_classification_metrics_include_macro_precision_and_recall() -> None:
    metrics = classification_metrics(["a", "a", "b"], ["a", "b", "b"], ["a", "b"])

    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["macro_precision"] == pytest.approx(0.75)
    assert metrics["macro_recall"] == pytest.approx(0.75)
    assert metrics["macro_f1"] == pytest.approx(2 / 3)


def test_confusion_matrix_handles_predictions_outside_actual_subset(tmp_path: Path) -> None:
    path = tmp_path / "matrix.csv"
    rows = [
        {"actual_category": "a", "predicted_category": "b", "label_valid": True},
        {"actual_category": "a", "predicted_category": None, "label_valid": False},
    ]

    write_confusion_matrix(path, rows)

    with path.open(newline="", encoding="utf-8") as handle:
        matrix = list(csv.reader(handle))
    assert matrix[0] == ["actual\\predicted", "a", "b", "__invalid_output__"]
    assert matrix[1] == ["a", "0", "1", "1"]


def test_output_lock_rejects_a_second_writer(tmp_path: Path) -> None:
    output = tmp_path / "outputs.jsonl"

    with output_lock(output):
        with pytest.raises(RuntimeError, match="already writes"):
            with output_lock(output):
                pass
