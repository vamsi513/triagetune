"""Catch stale reports or data changes before sharing evaluation results."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_data_and_test_fingerprint_match_recorded_values():
    assert digest(ROOT / "data" / "raw" / "train.csv") == (
        "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b"
    )
    assert digest(ROOT / "data" / "raw" / "test.csv") == (
        "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d"
    )
    report = json.loads((ROOT / "reports" / "lora_adapter_test.json").read_text())
    assert digest(ROOT / "data" / "processed" / "test.csv") == report["configuration"]["data_sha256"]


def test_saved_model_test_rows_match_metrics_and_test_source():
    with (ROOT / "data" / "processed" / "test.csv").open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    outputs = [
        json.loads(line)
        for line in (ROOT / "reports" / "lora_adapter_test_outputs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    report = json.loads((ROOT / "reports" / "lora_adapter_test.json").read_text())
    assert len(source) == len(outputs) == report["metrics"]["total"] == 3080
    for index, (row, output) in enumerate(zip(source, outputs)):
        assert output["row_index"] == index
        assert output["text"] == row["text"]
        assert output["actual_category"] == row["category"]
    correct = sum(bool(row["correct"]) for row in outputs)
    assert correct == report["metrics"]["correct"]
    assert correct / len(outputs) == report["metrics"]["accuracy"]
    assert sum(not row["label_valid"] for row in outputs) == 8


def test_classical_test_rows_match_reported_accuracy():
    with (ROOT / "reports" / "tfidf_logreg_test_predictions.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        predictions = list(csv.DictReader(handle))
    report = json.loads((ROOT / "reports" / "tfidf_logreg_test.json").read_text())
    correct = sum(row["correct"] == "True" for row in predictions)
    assert len(predictions) == report["metrics"]["total"] == 3080
    assert correct == report["metrics"]["correct"]
    assert correct / len(predictions) == report["metrics"]["accuracy"]


def test_scope_reports_have_consistent_record_totals():
    synthetic = json.loads(
        (ROOT / "reports" / "unknown_rejection_evaluation.json").read_text()
    )
    rows = [
        json.loads(line)
        for line in (ROOT / "reports" / "unknown_rejection_predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == synthetic["records"] == 32
    assert sum(row["scope"] == "in_scope" for row in rows) == 16
    assert sum(row["scope"] == "out_of_scope" for row in rows) == 16
    external = json.loads((ROOT / "reports" / "external_scope_evaluation.json").read_text())
    assert external["selection"]["total_records"] == 600
    for result in external["candidate_scores_without_model_output_check"].values():
        assert result["accepted"] + result["rejected"] == 600
