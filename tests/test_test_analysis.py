from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyze_test_results import confidence_interval, correctness_overlap, stratified_bootstrap


def test_stratified_bootstrap_is_deterministic() -> None:
    actual = ["a", "a", "b", "b"]
    predictions = {
        "first": ["a", "b", "b", "b"],
        "second": ["a", "a", "a", "b"],
    }

    first = stratified_bootstrap(actual, predictions, samples=50, seed=7, batch_size=13)
    second = stratified_bootstrap(actual, predictions, samples=50, seed=7, batch_size=13)

    for approach in predictions:
        for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
            assert np.array_equal(first[approach][metric], second[approach][metric])


def test_confidence_interval_uses_percentiles() -> None:
    values = np.arange(101, dtype=float)
    assert confidence_interval(values) == [2.5, 97.5]


def test_correctness_overlap_counts_all_cases() -> None:
    left = [{"correct": value} for value in (True, True, False, False)]
    right = [{"correct": value} for value in (True, False, True, False)]

    assert correctness_overlap(left, right) == {
        "both_correct": 1,
        "left_only_correct": 1,
        "right_only_correct": 1,
        "both_incorrect": 1,
    }
