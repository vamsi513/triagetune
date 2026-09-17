from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate_unknown_rejection import choose_threshold, score_decisions


def test_threshold_is_selected_on_calibration_rows_only():
    rows = [
        {"scope": "in_scope", "group": "known", "model_label_valid": True, "score": 0.8},
        {"scope": "in_scope", "group": "known", "model_label_valid": True, "score": 0.7},
        {"scope": "out_of_scope", "group": "unknown", "model_label_valid": True, "score": 0.2},
        {"scope": "out_of_scope", "group": "unknown", "model_label_valid": False, "score": 0.9},
    ]
    selection = choose_threshold(rows, "score")
    result = score_decisions(rows, "score", selection["threshold"])
    assert result["confusion_counts"] == {"known_accepted": 2, "unknown_rejected": 2}
    assert result["unknown_rejection_rate"] == 1.0
    assert result["known_false_rejection_rate"] == 0.0


def test_invalid_output_only_counts_forced_known_as_missed_unknown():
    rows = [
        {"scope": "in_scope", "group": "known", "model_label_valid": False},
        {"scope": "out_of_scope", "group": "unknown", "model_label_valid": True},
    ]
    result = score_decisions(rows, None, None)
    assert result["confusion_counts"] == {
        "known_rejected": 1,
        "unknown_accepted": 1,
    }
