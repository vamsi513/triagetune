"""Keep the distribution card aligned with the saved evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.evaluate import build_system_prompt, prompt_hash
from src.package_adapter import ADAPTER_FILES, checked_sources


ROOT = Path(__file__).resolve().parents[1]


def test_model_card_labels_and_prompt_match_test_evaluation():
    labels = json.loads((ROOT / "model_card" / "labels.json").read_text(encoding="utf-8"))
    with (ROOT / "data" / "processed" / "test.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        expected = sorted({row["category"] for row in csv.DictReader(handle)})
    report = json.loads(
        (ROOT / "reports" / "lora_adapter_test.json").read_text(encoding="utf-8")
    )
    assert labels == expected
    assert len(labels) == 77
    assert prompt_hash(build_system_prompt(labels)) == report["configuration"]["prompt_sha256"]


def test_model_card_keeps_license_and_release_status_explicit():
    card = (ROOT / "model_card" / "README.md").read_text(encoding="utf-8")
    assert card.startswith("---\n")
    metadata = card.split("---\n", 2)[1]
    assert "base_model: Qwen/Qwen2.5-1.5B-Instruct" in metadata
    assert "base_model_relation: adapter" in metadata
    assert "license: apache-2.0" in metadata
    assert "draft model card" not in card
    assert "repository remains private" not in card
    assert 'snapshot_download(repo_id="Vamsi513/triagetune-banking77-lora")' in card
    assert (ROOT / "model_card" / "LICENSE").is_file()
    assert "not" in card.lower() and "automatic support routing" in card


def test_package_sources_match_recorded_adapter_files():
    sources = checked_sources()
    assert set(sources) == {"README.md", "LICENSE", "labels.json", *ADAPTER_FILES}
