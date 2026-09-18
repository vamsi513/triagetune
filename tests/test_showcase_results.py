"""Keep the lightweight showcase grounded in saved test predictions."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from src import showcase_results


def test_showcase_reproduces_saved_correctness_overlap() -> None:
    rows = showcase_results.load_comparison()
    assert len(rows) == 3080
    assert Counter(row["outcome"] for row in rows) == {
        (True, True): 2224,
        (False, True): 384,
        (True, False): 255,
        (False, False): 217,
    }


def test_showcase_examples_cover_all_outcomes_without_reusing_categories() -> None:
    examples = showcase_results.choose_examples(showcase_results.load_comparison())
    assert [label for label, _ in examples] == [label for _, label in showcase_results.OUTCOMES]
    assert len({row["actual"] for _, row in examples}) == 4


def test_showcase_rejects_misaligned_saved_predictions(tmp_path: Path, monkeypatch) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "lora_adapter_test_outputs.jsonl").write_text(
        json.dumps({
            "row_index": 0,
            "text": "Example request",
            "actual_category": "pin_blocked",
            "predicted_category": "pin_blocked",
            "correct": True,
        }) + "\n",
        encoding="utf-8",
    )
    with (reports / "tfidf_logreg_test_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["row_index", "text", "actual", "predicted", "correct"]
        )
        writer.writeheader()
        writer.writerow({
            "row_index": 0,
            "text": "Different request",
            "actual": "pin_blocked",
            "predicted": "pin_blocked",
            "correct": "True",
        })
    monkeypatch.setattr(showcase_results, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="disagree"):
        showcase_results.load_comparison()
