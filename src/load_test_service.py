#!/usr/bin/env python3
"""Measure concurrent loopback inference latency and process memory."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx
import psutil


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--text", default="I cannot use my PIN.")
    parser.add_argument(
        "--report-file",
        type=Path,
        default=project_root / "reports" / "serving_load_test.json",
    )
    args = parser.parse_args()
    if args.requests <= 0 or args.concurrency <= 0:
        raise ValueError("request count and concurrency must be positive")
    return args


async def run(args: argparse.Namespace) -> dict[str, Any]:
    process = psutil.Process(args.pid)
    samples: list[int] = []
    monitoring = True

    async def monitor_memory() -> None:
        while monitoring:
            try:
                samples.append(process.memory_info().rss)
            except psutil.NoSuchProcess:
                break
            await asyncio.sleep(0.05)

    async with httpx.AsyncClient(base_url=args.url, timeout=60) as client:
        health = await client.get("/health")
        health.raise_for_status()
        info_response = await client.get("/model-info")
        info_response.raise_for_status()
        info = info_response.json()
        warmup = await client.post("/classify", json={"text": args.text})
        warmup.raise_for_status()
        expected = warmup.json()
        semaphore = asyncio.Semaphore(args.concurrency)

        async def one_request() -> tuple[float, int, dict[str, Any]]:
            async with semaphore:
                started = time.perf_counter()
                response = await client.post("/classify", json={"text": args.text})
                latency = time.perf_counter() - started
                return latency, response.status_code, response.json()

        monitor = asyncio.create_task(monitor_memory())
        started = time.perf_counter()
        results = await asyncio.gather(
            *(one_request() for _ in range(args.requests))
        )
        elapsed = time.perf_counter() - started
        monitoring = False
        await monitor

    latencies = [row[0] for row in results]
    success = sum(status == 200 and body == expected for _, status, body in results)
    memory_start = samples[0] if samples else process.memory_info().rss
    memory_peak = max(samples) if samples else memory_start
    return {
        "environment": "local_loopback",
        "concurrent_clients": args.concurrency,
        "measured_requests": args.requests,
        "warmup_requests": 1,
        "successful_matching_responses": success,
        "failed_or_mismatched_responses": args.requests - success,
        "request_text": args.text,
        "expected_response": expected,
        "p50_seconds": percentile(latencies, 0.50),
        "p95_seconds": percentile(latencies, 0.95),
        "min_seconds": min(latencies),
        "max_seconds": max(latencies),
        "elapsed_seconds": elapsed,
        "throughput_requests_per_second": args.requests / elapsed,
        "server_pid": args.pid,
        "server_rss_bytes_at_start": memory_start,
        "server_rss_bytes_peak": memory_peak,
        "server_rss_bytes_at_end": process.memory_info().rss,
        "server_memory_sampling_seconds": 0.05,
        "model_load_seconds": info["model_load_seconds"],
        "adapter_size_bytes": info["adapter_size_bytes"],
        "adapter_weight_bytes": info["adapter_weight_bytes"],
        "device": info["device"],
        "dtype": info["dtype"],
        "notes": [
            "Latency is end-to-end HTTP time after one warmup request.",
            "The model is protected by a lock; requests may queue under concurrency.",
            "RSS is process memory and excludes accelerator allocations.",
        ],
    }


def main() -> None:
    args = parse_args()
    report = asyncio.run(run(args))
    args.report_file.parent.mkdir(parents=True, exist_ok=True)
    args.report_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
