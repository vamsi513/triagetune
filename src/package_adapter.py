#!/usr/bin/env python3
"""Assemble a checked local adapter package without publishing it."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

try:
    from .evaluate import build_system_prompt, prompt_hash
except ImportError:
    from evaluate import build_system_prompt, prompt_hash


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_sources() -> dict[str, Path]:
    report = json.loads((ROOT / "reports" / "lora_training.json").read_text())
    evaluation = json.loads((ROOT / "reports" / "lora_adapter_test.json").read_text())
    saved = {Path(item["path"]).name: item for item in report["saved_model"]["files"]}
    source_dir = ROOT / "artifacts" / "lora_adapter"
    for name in ADAPTER_FILES:
        path = source_dir / name
        expected = saved[name]
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or linked adapter file: {path}")
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Adapter file differs from training report: {path}")

    card = ROOT / "model_card" / "README.md"
    license_file = ROOT / "model_card" / "LICENSE"
    label_file = ROOT / "model_card" / "labels.json"
    labels = json.loads(label_file.read_text(encoding="utf-8"))
    if not isinstance(labels, list) or len(labels) != 77 or labels != sorted(set(labels)):
        raise ValueError("Model card labels must be 77 distinct sorted names")
    if prompt_hash(build_system_prompt(labels)) != evaluation["configuration"]["prompt_sha256"]:
        raise ValueError("Model card labels do not match the evaluated prompt")
    content = card.read_text(encoding="utf-8")
    if not content.startswith("---\n") or "license: apache-2.0" not in content.split("---\n", 2)[1]:
        raise ValueError("Card metadata must declare the approved adapter license")
    if "This LoRA adapter classifies" not in content or "snapshot_download" not in content:
        raise ValueError("Card must include release-ready description and download instructions")
    if "draft model card" in content or "repository remains private" in content:
        raise ValueError("Card contains temporary release wording")
    license_text = license_file.read_text(encoding="utf-8")
    if "Apache License" not in license_text or "END OF TERMS AND CONDITIONS" not in license_text:
        raise ValueError("Approved adapter license text is missing")
    return {"README.md": card, "LICENSE": license_file, "labels.json": label_file} | {
        name: source_dir / name for name in ADAPTER_FILES
    }


def build_package(destination: Path) -> None:
    sources = checked_sources()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing package: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="adapter-package-", dir=destination.parent))
    try:
        for name, source in sources.items():
            shutil.copy2(source, staging / name)
            if sha256(staging / name) != sha256(source):
                raise ValueError(f"Copy verification failed: {name}")
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "huggingface_package",
    )
    args = parser.parse_args()
    build_package(args.output_dir)
    print(f"Local adapter package ready: {args.output_dir}")


if __name__ == "__main__":
    main()
