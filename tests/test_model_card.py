"""Keep the distribution card aligned with the saved evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from src import package_adapter
from src.evaluate import build_system_prompt, prompt_hash
from src.package_adapter import ADAPTER_FILES


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


def test_package_sources_match_recorded_adapter_files(tmp_path: Path, monkeypatch):
    adapter_dir = tmp_path / "artifacts" / "lora_adapter"
    adapter_dir.mkdir(parents=True)
    saved_files = []
    for name in ADAPTER_FILES:
        path = adapter_dir / name
        content = f"test content for {name}".encode("utf-8")
        path.write_bytes(content)
        saved_files.append({
            "path": str(path),
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        })

    card_dir = tmp_path / "model_card"
    card_dir.mkdir()
    for name in ("README.md", "LICENSE", "labels.json"):
        (card_dir / name).write_bytes((ROOT / "model_card" / name).read_bytes())
    labels = json.loads((card_dir / "labels.json").read_text(encoding="utf-8"))
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "lora_training.json").write_text(
        json.dumps({"saved_model": {"files": saved_files}}), encoding="utf-8"
    )
    (reports_dir / "lora_adapter_test.json").write_text(
        json.dumps({"configuration": {"prompt_sha256": prompt_hash(build_system_prompt(labels))}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(package_adapter, "ROOT", tmp_path)
    sources = package_adapter.checked_sources()
    assert set(sources) == {"README.md", "LICENSE", "labels.json", *ADAPTER_FILES}
    destination = tmp_path / "checked_package"
    package_adapter.build_package(destination)
    assert {path.name for path in destination.iterdir()} == set(sources)
    assert all(
        package_adapter.sha256(destination / name) == package_adapter.sha256(source)
        for name, source in sources.items()
    )
