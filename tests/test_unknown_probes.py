from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference import load_categories


ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_unknown_request_probes_are_consistent():
    allowed = set(load_categories(ROOT / "reports" / "lora_adapter_test.json"))
    rows = [
        json.loads(line)
        for line in (ROOT / "data" / "synthetic" / "unknown_request_probes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 32
    assert len({row["text"] for row in rows}) == len(rows)
    assert all(row["synthetic"] is True for row in rows)
    assert Counter(row["scope"] for row in rows) == {
        "in_scope": 16,
        "out_of_scope": 16,
    }
    assert all(
        row["expected_category"] in allowed
        if row["scope"] == "in_scope"
        else row["expected_category"] is None
        for row in rows
    )
    assert Counter(row["group"] for row in rows) == {
        "in_scope_direct": 8,
        "in_scope_paraphrase": 8,
        "out_of_scope_unrelated": 8,
        "out_of_scope_financial_adjacent": 8,
    }
