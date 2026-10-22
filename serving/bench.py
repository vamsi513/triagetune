#!/usr/bin/env python3
"""Load test the vLLM server and the Hugging Face FastAPI endpoint.

Both backends receive the same 300 requests, sampled from the reserved test
set with a fixed seed, at concurrency 1, 4, 16 and 32, with 20 warmup requests
per run excluded from the measurement. Run them on the same machine in the same
session, otherwise the comparison measures the machine rather than the runtime.

Outputs: results/serving_bench.json, results/serving_bench.md and
results/serving_bench.png.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serving.eval_vllm import render_prompts  # noqa: E402
from serving.serving_common import (  # noqa: E402
    BENCH_CONCURRENCIES,
    BENCH_SAMPLE_SEED,
    BENCH_SAMPLE_SIZE,
    BENCH_WARMUP_REQUESTS,
    DEFAULT_ADAPTER_DIR,
    DEFAULT_MODEL,
    DEFAULT_MODEL_REVISION,
    DEFAULT_REFERENCE_REPORT,
    DEFAULT_TEST_FILE,
    LORA_MODULE_NAME,
    PROJECT_ROOT,
    RESULTS_DIR,
    build_system_prompt,
    effective_labels,
    latency_summary,
    load_validation,
    portable_path,
    prompt_hash,
    reference_prompt_hash,
    sample_requests,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8800")
    parser.add_argument("--huggingface-url", default="http://127.0.0.1:8765")
    parser.add_argument(
        "--backends",
        default="vllm,huggingface",
        help="comma separated subset of vllm,huggingface",
    )
    parser.add_argument("--model", default=LORA_MODULE_NAME)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument(
        "--reference-report", type=Path, default=DEFAULT_REFERENCE_REPORT
    )
    parser.add_argument(
        "--vllm-mode", choices=("completions", "chat"), default="completions"
    )
    parser.add_argument("--requests", type=int, default=BENCH_SAMPLE_SIZE)
    parser.add_argument("--warmup", type=int, default=BENCH_WARMUP_REQUESTS)
    parser.add_argument("--seed", type=int, default=BENCH_SAMPLE_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--concurrency",
        default=",".join(str(value) for value in BENCH_CONCURRENCIES),
        help="comma separated concurrency levels",
    )
    parser.add_argument(
        "--json-file", type=Path, default=RESULTS_DIR / "serving_bench.json"
    )
    parser.add_argument(
        "--markdown-file", type=Path, default=RESULTS_DIR / "serving_bench.md"
    )
    parser.add_argument(
        "--chart-file", type=Path, default=RESULTS_DIR / "serving_bench.png"
    )
    args = parser.parse_args()
    args.backend_list = [name.strip() for name in args.backends.split(",") if name.strip()]
    unknown = set(args.backend_list) - {"vllm", "huggingface"}
    if unknown:
        raise ValueError(f"unknown backends: {sorted(unknown)}")
    if not args.backend_list:
        raise ValueError("at least one backend is required")
    args.concurrency_levels = [
        int(value) for value in args.concurrency.split(",") if value.strip()
    ]
    if any(level <= 0 for level in args.concurrency_levels):
        raise ValueError("concurrency levels must be positive")
    if args.warmup < 0:
        raise ValueError("warmup count cannot be negative")
    return args


def build_payload(
    backend: str, args: argparse.Namespace, system_prompt: str, record: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    if backend == "huggingface":
        return "/classify", {"text": record["text"]}
    if args.vllm_mode == "completions":
        return "/v1/completions", {
            "model": args.model,
            "prompt": record["rendered_prompt"],
            "temperature": 0,
            "max_tokens": args.max_new_tokens,
        }
    return "/v1/chat/completions", {
        "model": args.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": record["text"]},
        ],
        "temperature": 0,
        "max_tokens": args.max_new_tokens,
    }


async def one_request(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    backend: str,
    args: argparse.Namespace,
    system_prompt: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    path, payload = build_payload(backend, args, system_prompt, record)
    async with semaphore:
        started = time.perf_counter()
        try:
            response = await client.post(path, json=payload)
        except httpx.HTTPError as error:
            return {
                "latency_seconds": time.perf_counter() - started,
                "status_code": None,
                "outcome": f"transport_{type(error).__name__}",
            }
        latency = time.perf_counter() - started
    if response.status_code == 200:
        outcome = "ok"
    elif response.status_code in (413, 422):
        # The Hugging Face endpoint returns these when the model output is not
        # an approved category or the prompt is too long. They are real
        # non-200 responses, counted as errors, but kept distinct from
        # timeouts and transport failures.
        outcome = f"rejected_{response.status_code}"
    else:
        outcome = f"http_{response.status_code}"
    return {
        "latency_seconds": latency,
        "status_code": response.status_code,
        "outcome": outcome,
    }


async def run_phase(
    client: httpx.AsyncClient,
    backend: str,
    args: argparse.Namespace,
    system_prompt: str,
    records: list[dict[str, Any]],
    concurrency: int,
) -> tuple[list[dict[str, Any]], float]:
    semaphore = asyncio.Semaphore(concurrency)
    started = time.perf_counter()
    results = await asyncio.gather(
        *(
            one_request(client, semaphore, backend, args, system_prompt, record)
            for record in records
        )
    )
    return list(results), time.perf_counter() - started


async def run_backend(
    backend: str,
    base_url: str,
    args: argparse.Namespace,
    system_prompt: str,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    limits = httpx.Limits(
        max_connections=max(args.concurrency_levels),
        max_keepalive_connections=max(args.concurrency_levels),
    )
    async with httpx.AsyncClient(
        base_url=base_url, timeout=args.timeout, limits=limits
    ) as client:
        for concurrency in args.concurrency_levels:
            if args.warmup:
                warmup_records = [
                    records[index % len(records)] for index in range(args.warmup)
                ]
                warmup_results, _ = await run_phase(
                    client, backend, args, system_prompt, warmup_records, concurrency
                )
                warmup_errors = sum(
                    result["outcome"] != "ok" for result in warmup_results
                )
            else:
                warmup_errors = 0

            results, elapsed = await run_phase(
                client, backend, args, system_prompt, records, concurrency
            )
            latencies = [result["latency_seconds"] for result in results]
            outcomes = Counter(result["outcome"] for result in results)
            errors = len(results) - outcomes["ok"]
            run = {
                "backend": backend,
                "base_url": base_url,
                "concurrency": concurrency,
                "measured_requests": len(results),
                "warmup_requests_excluded": args.warmup,
                "warmup_errors": warmup_errors,
                "elapsed_seconds": elapsed,
                "requests_per_second": len(results) / elapsed if elapsed else None,
                "error_count": errors,
                "outcomes": dict(sorted(outcomes.items())),
                **latency_summary(latencies),
            }
            runs.append(run)
            print(
                f"{backend} concurrency {concurrency}: "
                f"{run['requests_per_second']:.2f} req/s, "
                f"p95 {run['p95_seconds']:.3f}s, errors {errors}",
                flush=True,
            )
    return runs


def markdown_table(runs: list[dict[str, Any]]) -> str:
    lines = [
        "# Serving benchmark",
        "",
        "Same machine, same requests, same session. Latency is end-to-end HTTP",
        "time measured by the client. Warmup requests are excluded.",
        "",
        "| Backend | Concurrency | Requests | Median (s) | p95 (s) | p99 (s) | Req/s | Errors |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        throughput = run["requests_per_second"]
        lines.append(
            "| {backend} | {concurrency} | {measured} | {median:.3f} | {p95:.3f} | "
            "{p99:.3f} | {rps} | {errors} |".format(
                backend=run["backend"],
                concurrency=run["concurrency"],
                measured=run["measured_requests"],
                median=run["median_seconds"],
                p95=run["p95_seconds"],
                p99=run["p99_seconds"],
                rps=f"{throughput:.2f}" if throughput is not None else "n/a",
                errors=run["error_count"],
            )
        )
    non_ok = {
        run["backend"]: {
            outcome: count
            for outcome, count in run["outcomes"].items()
            if outcome != "ok"
        }
        for run in runs
        if run["error_count"]
    }
    if non_ok:
        lines += ["", "Non-200 responses by backend:", ""]
        for backend, outcomes in sorted(non_ok.items()):
            lines.append(f"- {backend}: {outcomes}")
    lines.append("")
    return "\n".join(lines)


def write_chart(path: Path, runs: list[dict[str, Any]]) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import ScalarFormatter
    except ImportError:
        print("matplotlib is not installed; skipping the chart", flush=True)
        return False

    backends = sorted({run["backend"] for run in runs})
    figure, (latency_axis, throughput_axis) = plt.subplots(1, 2, figsize=(11, 4.5))
    for backend in backends:
        series = sorted(
            (run for run in runs if run["backend"] == backend),
            key=lambda run: run["concurrency"],
        )
        levels = [run["concurrency"] for run in series]
        latency_axis.plot(
            levels, [run["p95_seconds"] for run in series], marker="o", label=backend
        )
        throughput_axis.plot(
            levels,
            [run["requests_per_second"] for run in series],
            marker="o",
            label=backend,
        )
    for axis, title, ylabel in (
        (latency_axis, "p95 latency versus concurrency", "p95 latency (seconds)"),
        (throughput_axis, "Throughput versus concurrency", "Requests per second"),
    ):
        axis.set_title(title)
        axis.set_xlabel("Concurrent requests")
        axis.set_ylabel(ylabel)
        axis.set_xscale("log", base=2)
        axis.set_xticks(sorted({run["concurrency"] for run in runs}))
        axis.get_xaxis().set_major_formatter(ScalarFormatter())
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return True


def main() -> None:
    args = parse_args()
    records = load_validation(args.data_file, None)
    labels = effective_labels(records)
    system_prompt = build_system_prompt(labels)
    expected_prompt_hash = reference_prompt_hash(args.reference_report)
    if prompt_hash(system_prompt) != expected_prompt_hash:
        raise SystemExit(
            "the system prompt does not match the recorded evaluation; "
            "refusing to benchmark a different prompt"
        )

    sampled = sample_requests(records, args.requests, args.seed)
    if "vllm" in args.backend_list and args.vllm_mode == "completions":
        rendered = render_prompts(
            args.adapter_dir, system_prompt, [row["text"] for row in sampled]
        )
        for row, prompt in zip(sampled, rendered):
            row["rendered_prompt"] = prompt

    runs: list[dict[str, Any]] = []
    for backend in args.backend_list:
        base_url = args.vllm_url if backend == "vllm" else args.huggingface_url
        runs += asyncio.run(
            run_backend(backend, base_url, args, system_prompt, sampled)
        )

    chart_written = write_chart(args.chart_file, runs)
    report = {
        "benchmark": "serving_latency_and_throughput",
        "configuration": {
            "backends": args.backend_list,
            "vllm_url": args.vllm_url,
            "vllm_request_mode": args.vllm_mode,
            "huggingface_url": args.huggingface_url,
            "lora_module": args.model,
            "model": DEFAULT_MODEL,
            "model_revision": DEFAULT_MODEL_REVISION,
            "sampled_requests": args.requests,
            "sample_seed": args.seed,
            "warmup_requests_excluded_per_run": args.warmup,
            "concurrency_levels": args.concurrency_levels,
            "temperature": 0,
            "max_new_tokens": args.max_new_tokens,
            "client_timeout_seconds": args.timeout,
            "data_file": portable_path(args.data_file, PROJECT_ROOT),
            "data_sha256": sha256_file(args.data_file),
            "prompt_sha256": prompt_hash(system_prompt),
            "sampled_row_indexes": [row["row_index"] for row in sampled],
        },
        "runs": runs,
        "artifacts": {
            "markdown_file": portable_path(args.markdown_file, PROJECT_ROOT),
            "chart_file": (
                portable_path(args.chart_file, PROJECT_ROOT) if chart_written else None
            ),
            "environment_file": "results/serving_env.json",
        },
        "notes": [
            "Latency is end-to-end HTTP time from the client, not server-side time.",
            "Both backends ran on the same machine against the same sampled requests.",
            "Results hold for this hardware only; see results/serving_env.json.",
        ],
    }
    args.json_file.parent.mkdir(parents=True, exist_ok=True)
    args.json_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown_file.write_text(markdown_table(runs), encoding="utf-8")
    print(markdown_table(runs))


if __name__ == "__main__":
    main()
