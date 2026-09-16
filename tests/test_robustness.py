from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate_robustness import build_rows, load_probes, score_group


def probe(identifier: str, group: str, expected: str | None, acceptable: list[str]):
    return {
        "id": identifier,
        "group": group,
        "text": identifier,
        "expected_category": expected,
        "acceptable_categories": acceptable,
        "source": "project_created_synthetic",
    }


def test_load_probes_requires_synthetic_label(tmp_path: Path) -> None:
    probes = [
        probe("short", "short", "a", ["a"]),
        probe("long", "long", "a", ["a"]),
        probe("ambiguous", "ambiguous", None, ["a"]),
        probe("outside", "out_of_scope", None, []),
    ]
    probes[0]["source"] = "unspecified"
    path = tmp_path / "probes.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in probes), encoding="utf-8")

    with pytest.raises(ValueError, match="synthetic-data label"):
        load_probes(path, {"a"})


def test_out_of_scope_invented_label_is_not_safe_rejection() -> None:
    probes = [probe("outside", "out_of_scope", None, [])]
    adapter_results = [
        {
            "raw_output": '{"category":"invented"}',
            "json_valid": True,
            "schema_valid": True,
            "label_valid": False,
            "predicted_category": "invented",
        }
    ]
    rows = build_rows(probes, ["a"], adapter_results)

    result = score_group(rows, "out_of_scope", "lora_adapter")

    assert result["forced_in_scope_predictions"] == 0
    assert result["invalid_or_disallowed_outputs"] == 1
    assert result["explicit_unknown_outputs"] == 0
    assert result["safe_rejections"] == 0


def test_known_probe_exact_match_scoring() -> None:
    probes = [probe("short", "short", "a", ["a"])]
    adapter_results = [
        {
            "raw_output": '{"category":"a"}',
            "json_valid": True,
            "schema_valid": True,
            "label_valid": True,
            "predicted_category": "a",
        }
    ]
    rows = build_rows(probes, ["b"], adapter_results)

    classical = score_group(rows, "short", "tfidf_logreg")
    adapter = score_group(rows, "short", "lora_adapter")

    assert classical["exact_match_rate"] == 0.0
    assert adapter["exact_match_rate"] == 1.0
