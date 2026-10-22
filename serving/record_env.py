#!/usr/bin/env python3
"""Record the GPU, driver, CUDA and package versions for one benchmark run."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serving.serving_common import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_MODEL_REVISION,
    RESULTS_DIR,
)


TRACKED_PACKAGES = (
    "vllm",
    "torch",
    "transformers",
    "peft",
    "tokenizers",
    "httpx",
    "matplotlib",
    "numpy",
)

NVIDIA_SMI_FIELDS = (
    "name",
    "driver_version",
    "memory.total",
    "compute_cap",
)


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def nvidia_smi_gpus() -> list[dict[str, str]] | None:
    """Query the installed GPUs, or return None when nvidia-smi is absent."""
    if shutil.which("nvidia-smi") is None:
        return None
    query = ",".join(NVIDIA_SMI_FIELDS)
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    gpus = []
    for line in completed.stdout.strip().splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == len(NVIDIA_SMI_FIELDS):
            gpus.append(dict(zip(NVIDIA_SMI_FIELDS, values)))
    return gpus or None


def cuda_runtime() -> dict[str, Any]:
    """CUDA details reported by torch, without requiring a CUDA build."""
    try:
        import torch
    except ImportError:
        return {"torch_installed": False}
    available = torch.cuda.is_available()
    details: dict[str, Any] = {
        "torch_installed": True,
        "torch_version": torch.__version__,
        "cuda_built_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        ),
        "cuda_available": available,
        "device_count": torch.cuda.device_count() if available else 0,
    }
    if available:
        details["devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": ".".join(
                    str(part) for part in torch.cuda.get_device_capability(index)
                ),
                "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ]
    return details


def build_record() -> dict[str, Any]:
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        },
        "gpus_from_nvidia_smi": nvidia_smi_gpus(),
        "cuda": cuda_runtime(),
        "packages": package_versions(),
        "model": {
            "base_model": DEFAULT_MODEL,
            "base_model_revision": DEFAULT_MODEL_REVISION,
        },
        "notes": [
            "gpus_from_nvidia_smi is null when nvidia-smi is not on PATH.",
            "Benchmark numbers hold for this hardware only.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-file", type=Path, default=RESULTS_DIR / "serving_env.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = build_record()
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
