# vLLM serving benchmark: profile notes

## Status: nothing has been benchmarked

No numbers are reported here because no run has happened. The machine this
code was written on is an Apple M5 with no NVIDIA GPU:

- `nvidia-smi` is not installed and not on `PATH`.
- `python3 -c "import torch"` fails with `ModuleNotFoundError: No module named
  'torch'`, and the host has no CUDA device for torch to find in any case.

vLLM requires a CUDA or ROCm GPU, so neither the accuracy run nor the load test
could be started. `results/` is therefore empty: there is no
`serving_env.json`, `serving_accuracy.json` or `serving_bench.json` to read
from yet. `serving/RUN_ON_GPU.md` has the exact steps to produce them on a free
Colab or Kaggle T4.

The only thing verified on this machine is that the scoring code is faithful to
the recorded evaluation. `serving/eval_vllm.py` was pointed at a local stub that
replayed the saved per-example outputs from
`reports/lora_adapter_test_outputs.jsonl`, and it reproduced the recorded
figures exactly: accuracy 0.8048701298701298, macro F1 0.804990809472458, and
an invalid output rate of 0.0025974025974025974, with zero disagreements
against the saved predictions. That is a check on this script's arithmetic and
parsing, not a measurement of vLLM, and it says nothing at all about latency or
throughput.

## Fill this in after a real run

Replace this section with the numbers from the result files. Report only what
the files contain, and keep the hardware line next to the numbers.

- Hardware, driver, CUDA and package versions: copy from
  `results/serving_env.json`.
- Accuracy, macro F1 and invalid output rate: from
  `results/serving_accuracy.json`.
- Disagreements against the saved Hugging Face predictions: the
  `disagreements` field in the same file, out of 3,080 compared examples.
- Latency and throughput per concurrency level and per backend: from
  `results/serving_bench.json` and the table in `results/serving_bench.md`.

Answer these specific questions in prose:

1. Where was vLLM faster, and by how much? Give the ratio at each concurrency
   level, not just the best one.
2. Where was vLLM not faster? At concurrency 1 the comparison is one request
   at a time against a model already resident in the FastAPI process, so vLLM's
   continuous batching has nothing to work with and its scheduling overhead can
   leave it slower. If that happened, say so.
3. Did the Hugging Face endpoint hit errors or timeouts at concurrency 16 or
   32? It serializes generation behind a single lock
   (`src/inference.py`, `self._lock`), so queued requests can exceed the
   client timeout. Record the error count and the breakdown from the
   `outcomes` field, and separate client timeouts from the HTTP 422 responses
   that the endpoint returns when the model output is not an approved category.
4. Did vLLM accuracy match the recorded 80.49%? If the gap is wider than 0.5
   points, `results/serving_accuracy.json` records a finding. Investigate in
   this order, and do not change anything in `reports/`:
   - Prompt. `serving/eval_vllm.py` refuses to run unless the system prompt it
     builds hashes to the digest stored in `reports/lora_adapter_test.json`,
     so a prompt text difference is already ruled out before any request is
     sent. What is not ruled out is the chat template: `--mode chat` lets the
     server render the conversation, and vLLM's template may not be the one in
     the adapter directory. Rerun with `--mode completions`, which renders
     locally with the adapter tokenizer exactly as `src/evaluate.py` does, and
     compare the two.
   - Tokenization. `serving/serve_vllm.sh` passes `--tokenizer` pointing at the
     adapter directory for this reason. Check that `results/serving_env.json`
     shows the same `transformers` version the recorded run used, and compare
     the `prompt_tokens` counts in `results/serving_preds.jsonl` against the
     prompt lengths in `reports/lora_training.json` (median 446, maximum 532).
   - Dtype. The recorded run used float16 on the Mac accelerator
     (`choose_device` in `src/evaluate.py` returns float16 for both mps and
     cuda). The T4 has no bfloat16, so `--dtype half` keeps this aligned. If
     the server was started with a different dtype, that is the first thing to
     rule out.
   - Decoding. Both paths use temperature 0 and 32 new tokens, but greedy
     decoding is not bit identical across kernels and batch shapes, so a small
     number of disagreements on near-tie examples is expected rather than a
     bug. The saved run already produces 8 invalid outputs out of 3,080, which
     shows how close some of these decisions are.

## State the hardware

Results from one GPU do not generalize. A T4 is a 2018 card with 16 GB and no
bfloat16; an A10G, L4 or A100 will reorder these numbers, and the throughput
gap in particular depends on how much KV cache the card can hold. Always print
the hardware line from `results/serving_env.json` next to any number taken from
these files, and do not quote a latency or throughput figure from this project
without it.
