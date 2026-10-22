"""Checks for the pure serving helpers.

These run without a GPU, a network connection or a model download. Anything
that needs the vLLM server or the adapter weights is out of scope here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serving.serving_common import (
    BENCH_CONCURRENCIES,
    BENCH_SAMPLE_SIZE,
    BENCH_WARMUP_REQUESTS,
    build_system_prompt,
    effective_labels,
    latency_summary,
    load_reference_predictions,
    parse_raw_output,
    percentile,
    prompt_hash,
    reference_prompt_hash,
    sample_requests,
)
from src.evaluate import parse_raw_output as evaluate_parse_raw_output
from src.load_test_service import percentile as load_test_percentile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECORDED_REPORT = PROJECT_ROOT / "reports" / "lora_adapter_test.json"
RECORDED_OUTPUTS = PROJECT_ROOT / "reports" / "lora_adapter_test_outputs.jsonl"


def make_records(count: int) -> list[dict[str, str]]:
    return [
        {"text": f"request {index}", "category": f"category_{index % 7}"}
        for index in range(count)
    ]


def test_percentile_is_the_existing_implementation() -> None:
    assert percentile is load_test_percentile


@pytest.mark.parametrize(
    ("fraction", "expected"),
    [(0.0, 1.0), (0.5, 3.0), (1.0, 5.0)],
)
def test_percentile_interpolates_between_ordered_samples(
    fraction: float, expected: float
) -> None:
    assert percentile([5.0, 1.0, 3.0, 2.0, 4.0], fraction) == pytest.approx(expected)


def test_percentile_handles_a_single_sample() -> None:
    assert percentile([0.25], 0.99) == pytest.approx(0.25)


def test_latency_summary_reports_median_p95_and_p99() -> None:
    latencies = [index / 100 for index in range(1, 101)]

    summary = latency_summary(latencies)

    assert summary["median_seconds"] == pytest.approx(percentile(latencies, 0.50))
    assert summary["p95_seconds"] == pytest.approx(percentile(latencies, 0.95))
    assert summary["p99_seconds"] == pytest.approx(percentile(latencies, 0.99))
    assert summary["min_seconds"] == pytest.approx(0.01)
    assert summary["max_seconds"] == pytest.approx(1.0)
    assert (
        summary["median_seconds"] <= summary["p95_seconds"] <= summary["p99_seconds"]
    )


def test_latency_summary_rejects_an_empty_measurement() -> None:
    with pytest.raises(ValueError, match="at least one measurement"):
        latency_summary([])


def test_request_sampling_is_reproducible_for_one_seed() -> None:
    records = make_records(500)

    first = sample_requests(records, 50, seed=20251022)
    second = sample_requests(records, 50, seed=20251022)

    assert first == second
    assert [row["row_index"] for row in first] == sorted(
        row["row_index"] for row in first
    )


def test_request_sampling_differs_between_seeds() -> None:
    records = make_records(500)

    first = sample_requests(records, 50, seed=1)
    second = sample_requests(records, 50, seed=2)

    assert first != second


def test_request_sampling_draws_without_replacement_and_keeps_the_label() -> None:
    records = make_records(300)

    sampled = sample_requests(records, 300, seed=7)

    indexes = [row["row_index"] for row in sampled]
    assert len(set(indexes)) == 300
    for row in sampled:
        source = records[row["row_index"]]
        assert row["text"] == source["text"]
        assert row["actual_category"] == source["category"]


@pytest.mark.parametrize("count", [0, -1, 301])
def test_request_sampling_rejects_impossible_counts(count: int) -> None:
    with pytest.raises(ValueError):
        sample_requests(make_records(300), count, seed=7)


def test_label_parsing_is_the_evaluation_implementation() -> None:
    assert parse_raw_output is evaluate_parse_raw_output


def test_label_parsing_accepts_only_the_exact_schema() -> None:
    allowed = {"pin_blocked"}

    assert parse_raw_output('{"category":"pin_blocked"}', allowed) == {
        "json_valid": True,
        "schema_valid": True,
        "label_valid": True,
        "predicted_category": "pin_blocked",
    }


@pytest.mark.parametrize(
    "raw_output",
    [
        '{"category":"invented_category"}',
        '{"category":"pin_blocked","confidence":0.9}',
        '```json\n{"category":"pin_blocked"}\n```',
        "pin_blocked",
        "",
    ],
)
def test_label_parsing_refuses_to_repair_unusable_output(raw_output: str) -> None:
    parsed = parse_raw_output(raw_output, {"pin_blocked"})

    assert parsed["label_valid"] is False


def test_system_prompt_matches_the_recorded_evaluation_digest() -> None:
    """The serving prompt must hash to the digest of the recorded run.

    If this fails, a serving accuracy number cannot be compared with the
    80.49% recorded for the saved adapter, because the prompts differ.
    """
    labels = effective_labels(
        {"text": "", "category": label}
        for label in json.loads(RECORDED_REPORT.read_text(encoding="utf-8"))["metrics"][
            "per_class"
        ]
    )

    assert len(labels) == 77
    assert prompt_hash(build_system_prompt(labels)) == reference_prompt_hash(
        RECORDED_REPORT
    )


def test_recorded_predictions_load_as_a_contiguous_index() -> None:
    reference = load_reference_predictions(RECORDED_OUTPUTS)

    assert len(reference) == 3080
    assert sorted(reference) == list(range(3080))
    assert reference[0]["actual_category"] == "card_arrival"


def test_recorded_predictions_reject_an_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "serving_preds.jsonl"
    empty.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="no saved predictions"):
        load_reference_predictions(empty)


def test_benchmark_defaults_match_the_documented_plan() -> None:
    assert BENCH_SAMPLE_SIZE == 300
    assert BENCH_WARMUP_REQUESTS == 20
    assert BENCH_CONCURRENCIES == (1, 4, 16, 32)
