# Draft README section for the vLLM serving benchmark

This is the draft of the `README.md` section for this benchmark. It is kept
here instead of in `README.md` because the measurement table is still empty:
no benchmark run has happened yet, and the README records measurements only
after the corresponding command has been run.

Insert it after `## Local serving: Phase 5` and before
`## Internal agent-assist prototype` once `results/serving_bench.json` and
`results/serving_accuracy.json` exist. Fill the table from those files only.

## Draft

```markdown
## vLLM serving benchmark

The saved adapter can also be served with vLLM's OpenAI compatible server
instead of the FastAPI endpoint in `api/main.py`. vLLM loads the adapter as a
LoRA module against the same pinned base-model revision, so the served weights
are the ones the recorded results were produced with.

This benchmark is GPU only. vLLM needs a CUDA or ROCm device, and the
dependencies are pinned separately in `serving/requirements-vllm.txt` because
vLLM pins its own torch build. `serving/RUN_ON_GPU.md` has the exact steps for
a free Colab or Kaggle T4.

```bash
# GPU host, from the project root
pip install -r serving/requirements-vllm.txt -r requirements-serving.txt
VLLM_GPU_FRACTION=0.45 serving/serve_vllm.sh --background
TRIAGETUNE_ENABLE_PROVISIONAL_ROUTING=1 \
  python3 -m uvicorn api.main:app --host 127.0.0.1 --port 8765 &
python3 serving/eval_vllm.py   # accuracy on the reserved test set
python3 serving/bench.py       # latency and throughput, both backends
```

`serving/eval_vllm.py` scores all 3,080 reserved test requests at temperature
0, reusing the prompt builder and the output parser from `src/evaluate.py`. It
refuses to start unless the system prompt it builds hashes to the digest stored
in `reports/lora_adapter_test.json`, so a serving accuracy number can only be
compared against the recorded 80.49% when the prompts are identical. It also
counts the examples where vLLM and the saved Hugging Face predictions disagree.
Nothing in `reports/` is overwritten; output goes to `results/`.

`serving/bench.py` sends the same 300 requests, sampled from the test set with
a fixed seed, to both backends at concurrency 1, 4, 16 and 32, discarding 20
warmup requests per run.

| Backend | Concurrency | Median (s) | p95 (s) | p99 (s) | Req/s | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| | | | | | | |

Accuracy, hardware, and the reading of where vLLM did and did not help are in
`serving/profile_notes.md`. The FastAPI endpoint serializes generation behind a
lock, so its latency at concurrency 16 and 32 reflects queueing rather than
model speed. Results come from one GPU and do not generalize to other hardware.
```

## Note on test dependencies

`serving/serving_common.py` reuses `percentile` from `src/load_test_service.py`
rather than keeping a second copy of the same math. That module imports
`psutil` at module scope, so `psutil>=6.0,<8` is required for
`tests/test_serving_helpers.py` to collect. It is listed in
`requirements-test.txt` for that reason, and it pulls in no model or training
dependencies.
