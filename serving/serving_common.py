"""Shared pure helpers for the vLLM serving benchmark.

Everything here runs without a GPU, without network access, and without
downloading model weights, so it can be unit tested in CI. Prompt building,
label parsing, metric computation and percentile math are imported from the
existing modules rather than reimplemented, so the serving path scores
identically to `src/evaluate.py`.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_MODEL_REVISION,
    build_system_prompt,
    load_validation,
    parse_raw_output,
    prompt_hash,
)
from src.load_test_service import percentile  # noqa: E402
from src.train_baseline import classification_metrics, portable_path, sha256_file  # noqa: E402


DEFAULT_TEST_FILE = PROJECT_ROOT / "data" / "processed" / "test.csv"
DEFAULT_ADAPTER_DIR = PROJECT_ROOT / "artifacts" / "lora_adapter"
DEFAULT_REFERENCE_REPORT = PROJECT_ROOT / "reports" / "lora_adapter_test.json"
DEFAULT_REFERENCE_OUTPUTS = PROJECT_ROOT / "reports" / "lora_adapter_test_outputs.jsonl"
RESULTS_DIR = PROJECT_ROOT / "results"

LORA_MODULE_NAME = "triagetune-banking77-lora"
BENCH_SAMPLE_SEED = 20251022
BENCH_SAMPLE_SIZE = 300
BENCH_WARMUP_REQUESTS = 20
BENCH_CONCURRENCIES = (1, 4, 16, 32)

__all__ = [
    "BENCH_CONCURRENCIES",
    "BENCH_SAMPLE_SEED",
    "BENCH_SAMPLE_SIZE",
    "BENCH_WARMUP_REQUESTS",
    "DEFAULT_ADAPTER_DIR",
    "DEFAULT_MODEL",
    "DEFAULT_MODEL_REVISION",
    "DEFAULT_REFERENCE_OUTPUTS",
    "DEFAULT_REFERENCE_REPORT",
    "DEFAULT_TEST_FILE",
    "LORA_MODULE_NAME",
    "PROJECT_ROOT",
    "RESULTS_DIR",
    "build_system_prompt",
    "classification_metrics",
    "latency_summary",
    "load_reference_predictions",
    "load_validation",
    "parse_raw_output",
    "percentile",
    "portable_path",
    "prompt_hash",
    "reference_prompt_hash",
    "sample_requests",
    "sha256_file",
]


def sample_requests(
    records: Sequence[dict[str, str]], count: int, seed: int
) -> list[dict[str, Any]]:
    """Pick `count` records without replacement, reproducibly for a given seed.

    The chosen row indices are returned in ascending dataset order so two runs
    with the same seed send byte-identical requests in the same sequence. The
    row index is carried through so a prediction can be traced back to the
    exact test record.
    """
    if count <= 0:
        raise ValueError("sample count must be positive")
    if count > len(records):
        raise ValueError(
            f"cannot sample {count} records from a set of {len(records)}"
        )
    chosen = sorted(random.Random(seed).sample(range(len(records)), count))
    return [
        {
            "row_index": index,
            "text": records[index]["text"],
            "actual_category": records[index]["category"],
        }
        for index in chosen
    ]


def latency_summary(latencies: Sequence[float]) -> dict[str, float]:
    """Median, p95 and p99 latency plus the bounds, in seconds."""
    if not latencies:
        raise ValueError("latency summary requires at least one measurement")
    return {
        "median_seconds": percentile(list(latencies), 0.50),
        "p95_seconds": percentile(list(latencies), 0.95),
        "p99_seconds": percentile(list(latencies), 0.99),
        "min_seconds": min(latencies),
        "max_seconds": max(latencies),
    }


def reference_prompt_hash(report_path: Path) -> str:
    """The prompt digest recorded by the saved Hugging Face evaluation."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    return report["configuration"]["prompt_sha256"]


def load_reference_predictions(path: Path) -> dict[int, dict[str, Any]]:
    """Index the saved Hugging Face per-example outputs by row index."""
    rows: dict[int, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[int(row["row_index"])] = row
    if not rows:
        raise ValueError(f"{path} contains no saved predictions")
    return rows


def effective_labels(records: Iterable[dict[str, str]]) -> list[str]:
    """The sorted label set the system prompt is built from.

    `src/evaluate.py` derives the allowed categories from the data file it is
    scoring, so the serving path does the same to keep the prompt identical.
    """
    return sorted({row["category"] for row in records})
