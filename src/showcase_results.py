#!/usr/bin/env python3
"""Explore saved BANKING77 test predictions without loading model weights."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTCOMES = (
    ((True, True), "both correct"),
    ((False, True), "TF-IDF only correct"),
    ((True, False), "adapter only correct"),
    ((False, False), "neither correct"),
)


def load_comparison() -> list[dict[str, object]]:
    adapter_path = ROOT / "reports" / "lora_adapter_test_outputs.jsonl"
    classical_path = ROOT / "reports" / "tfidf_logreg_test_predictions.csv"
    adapter = [json.loads(line) for line in adapter_path.read_text(encoding="utf-8").splitlines()]
    with classical_path.open(newline="", encoding="utf-8") as handle:
        classical = list(csv.DictReader(handle))
    if not adapter or len(adapter) != len(classical):
        raise ValueError("Saved prediction files are empty or have different row counts")

    rows: list[dict[str, object]] = []
    for index, (model_row, baseline_row) in enumerate(zip(adapter, classical)):
        if (
            model_row["row_index"] != index
            or int(baseline_row["row_index"]) != index
            or model_row["text"] != baseline_row["text"]
            or model_row["actual_category"] != baseline_row["actual"]
        ):
            raise ValueError(f"Saved predictions disagree at row {index}")
        model_correct = model_row["correct"] is True
        baseline_correct = baseline_row["correct"] == "True"
        if model_correct != (model_row["predicted_category"] == model_row["actual_category"]):
            raise ValueError(f"Adapter correctness disagrees at row {index}")
        if baseline_correct != (baseline_row["predicted"] == baseline_row["actual"]):
            raise ValueError(f"Baseline correctness disagrees at row {index}")
        rows.append({
            "index": index,
            "text": model_row["text"],
            "actual": model_row["actual_category"],
            "adapter": model_row["predicted_category"],
            "baseline": baseline_row["predicted"],
            "outcome": (model_correct, baseline_correct),
        })
    return rows


def choose_examples(rows: list[dict[str, object]]) -> list[tuple[str, dict[str, object]]]:
    chosen: list[tuple[str, dict[str, object]]] = []
    used_categories: set[str] = set()
    for outcome, label in OUTCOMES:
        candidates = [row for row in rows if row["outcome"] == outcome]
        if not candidates:
            continue
        row = next(
            (candidate for candidate in candidates if candidate["actual"] not in used_categories),
            candidates[0],
        )
        chosen.append((label, row))
        used_categories.add(str(row["actual"]))
    return chosen


def main() -> None:
    rows = load_comparison()
    counts = Counter(row["outcome"] for row in rows)
    adapter_correct = sum(count for (model, _), count in counts.items() if model)
    baseline_correct = sum(count for (_, baseline), count in counts.items() if baseline)

    print("TriageTune recorded test-results explorer — not live inference")
    print("Source: public BANKING77 test split (CC BY 4.0); no weights or network needed")
    print(f"Rows: {len(rows):,}")
    print(f"TF-IDF accuracy: {baseline_correct / len(rows):.2%} ({baseline_correct:,}/{len(rows):,})")
    print(f"LoRA adapter accuracy: {adapter_correct / len(rows):.2%} ({adapter_correct:,}/{len(rows):,})")
    print("\nCorrectness overlap:")
    for outcome, label in OUTCOMES:
        print(f"  {label}: {counts[outcome]:,}")

    print("\nFirst matching examples, preferring distinct actual categories:")
    for label, row in choose_examples(rows):
        print(f"  {label} (row {row['index']}): {row['text']}")
        print(f"    actual={row['actual']} | TF-IDF={row['baseline']} | adapter={row['adapter']}")
    print("\nThese examples are illustrative, not a separate performance estimate.")


if __name__ == "__main__":
    main()
