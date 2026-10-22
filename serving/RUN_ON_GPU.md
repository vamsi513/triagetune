# Running the vLLM serving benchmark on a free T4

Nothing in `results/` has been produced yet. The machine this code was written
on is an Apple M5 with no NVIDIA GPU, so `nvidia-smi` is absent and vLLM cannot
run on it at all. These are the exact steps to produce the numbers on a free
Google Colab or Kaggle T4.

The T4 is compute capability 7.5 and has no bfloat16 support, so every command
below uses `--dtype half` (the default in `serving/serve_vllm.sh`).

## What gets downloaded

| Item | Approximate size |
| --- | ---: |
| `Qwen/Qwen2.5-1.5B-Instruct` at the pinned revision | 3.1 GB |
| `Vamsi513/triagetune-banking77-lora` adapter | 29 MB |

Both are public and need no access token. Do not paste a token into these
notebooks; nothing here requires one.

## Colab

1. Open a new notebook, then `Runtime` > `Change runtime type` > `T4 GPU`.
2. Confirm the GPU is attached:

   ```
   !nvidia-smi
   ```

   If this prints an error, the runtime has no GPU and the rest will fail.

3. Clone and install. vLLM pins its own `torch` build, so this replaces the
   torch that Colab ships with and the runtime has to be restarted afterwards:

   ```
   !git clone https://github.com/vamsi513/triagetune.git
   %cd triagetune
   !pip install -q -r serving/requirements-vllm.txt -r requirements-serving.txt
   ```

   After it finishes, use `Runtime` > `Restart session`, then `%cd triagetune`
   again. Skipping the restart leaves the old torch loaded in the process.

4. Download the base model at the pinned revision and the adapter:

   ```
   !python3 - <<'PY'
   from huggingface_hub import snapshot_download

   snapshot_download(
       repo_id="Vamsi513/triagetune-banking77-lora",
       local_dir="artifacts/lora_adapter",
       token=False,
   )
   snapshot_download(
       repo_id="Qwen/Qwen2.5-1.5B-Instruct",
       revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
       cache_dir=".cache/huggingface",
       token=False,
   )
   PY
   ```

5. Start vLLM in the background and wait for it to report healthy. The memory
   fraction is lowered from the default 0.90 so the Hugging Face endpoint can
   hold its own weights on the same 15 GB card at the same time:

   ```
   !VLLM_GPU_FRACTION=0.45 serving/serve_vllm.sh --background
   ```

   This writes `results/serving_env.json`, plus
   `results/serving_vllm_server.log` and `results/serving_vllm_server.pid`.
   First startup includes the model download and the CUDA graph capture, so
   allow several minutes. If it reports that the server exited, read the log
   before rerunning.

6. Start the Hugging Face FastAPI endpoint in a second background process.
   `TRIAGETUNE_ENABLE_PROVISIONAL_ROUTING=1` is the opt-in switch from the
   README; it is required or `/classify` returns HTTP 503:

   ```
   !nohup env TRIAGETUNE_ENABLE_PROVISIONAL_ROUTING=1 \
       python3 -m uvicorn api.main:app --host 127.0.0.1 --port 8765 \
       > results/serving_huggingface_server.log 2>&1 &
   !sleep 90 && curl -fsS http://127.0.0.1:8765/health
   ```

   The adapter loads at startup, so `/health` only answers once the weights
   are resident. If `curl` fails, wait longer and retry before moving on.

7. Score the reserved test set through vLLM. All 3,080 requests run, so this
   is the long step:

   ```
   !python3 serving/eval_vllm.py
   ```

   Writes `results/serving_preds.jsonl` and `results/serving_accuracy.json`,
   and prints accuracy, macro F1, the invalid output rate, the gap against the
   recorded 80.49%, and the number of examples where vLLM and the saved
   Hugging Face predictions disagree.

8. Run the load test against both backends:

   ```
   !python3 serving/bench.py
   ```

   Writes `results/serving_bench.json`, `results/serving_bench.md` and
   `results/serving_bench.png`.

9. Fill in `serving/profile_notes.md` from those files, and copy
   `results/serving_env.json` into the notes so the hardware is recorded next
   to the numbers.

## Kaggle

The steps are identical with three changes:

- Set `Accelerator` to `GPU T4 x2` in the notebook settings. vLLM uses one GPU
  here; there is no tensor parallel flag in `serving/serve_vllm.sh`.
- Turn `Internet` on in the notebook settings, otherwise the clone and the
  model download both fail.
- Work under `/kaggle/working`, which is writable and persists for the session.
  `/kaggle/input` is read only.

## If vLLM will not start on the T4

Check `results/serving_vllm_server.log` first, then try these in order:

1. `VLLM_EXTRA_ARGS="--enforce-eager"` skips CUDA graph capture. It lowers
   throughput but cuts startup memory, and it is the usual fix for an
   out of memory failure during startup.
2. Lower `VLLM_GPU_FRACTION` further, for example `0.35`.
3. `VLLM_MAX_MODEL_LEN=768`. The recorded prompts are around 450 tokens, with
   a maximum of 532 in training, so 768 still covers the test set and leaves
   more room for the KV cache.
4. If the pinned `vllm==0.32.0` itself fails on sm_75, try `vllm==0.31.0`.
   I have not run either version on a T4, so I cannot say which works; record
   whichever one you used in `results/serving_env.json`, which the launch
   script writes from the installed packages.

## Honest limits of this setup

- Lowering the vLLM memory fraction to 0.45 shrinks the KV cache, which limits
  how many sequences vLLM can batch. That understates vLLM's throughput
  advantage at concurrency 16 and 32. If you want vLLM's best number, run
  `serving/bench.py --backends vllm` alone at `VLLM_GPU_FRACTION=0.90` after
  stopping the FastAPI endpoint, and say in the notes that the two backends
  were then not measured concurrently.
- A free Colab or Kaggle T4 is shared, throttled hardware. Latency there is
  noisier than on a dedicated GPU, and the session can be reclaimed mid run.
- One GPU is one data point. These numbers do not carry over to an A10G, an
  L4, an A100, or to CPU serving.
