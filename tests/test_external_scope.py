from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate_external_scope import FINANCIAL_ADJACENT_LABELS, NONBANKING_LABELS


ROOT = Path(__file__).resolve().parents[1]


def test_external_challenge_report_matches_frozen_selection():
    report = json.loads((ROOT / "reports" / "external_scope_evaluation.json").read_text())
    selection = report["selection"]
    assert not (NONBANKING_LABELS & FINANCIAL_ADJACENT_LABELS)
    assert selection["total_records"] == 600
    assert selection["records_per_label"] == 30
    assert set(selection["nonbanking_labels"]) == NONBANKING_LABELS
    assert set(selection["financial_adjacent_labels"]) == FINANCIAL_ADJACENT_LABELS
    for result in report["candidate_scores_without_model_output_check"].values():
        assert result["accepted"] + result["rejected"] == 600
    assert report["saved_model_sample"]["records"] == 40
    assert report["rejection_enabled_in_service"] is False
