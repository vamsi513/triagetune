#!/usr/bin/env bash
#
# Start vLLM's OpenAI compatible server for the saved TriageTune adapter.
#
# The base model and its revision match src/evaluate.py and src/inference.py,
# so the served weights are the same ones the recorded Hugging Face results
# were produced with. The tokenizer is loaded from the adapter directory for
# the same reason: that is the tokenizer src/evaluate.py uses, and it carries
# the chat template the recorded prompt digest depends on.
#
# Usage:
#   serving/serve_vllm.sh                 # foreground, Ctrl-C to stop
#   serving/serve_vllm.sh --background    # background, waits for /health
#
# Environment overrides:
#   VLLM_PORT           default 8800
#   VLLM_HOST           default 127.0.0.1
#   VLLM_DTYPE          default half  (T4 and other sm_75 GPUs have no bfloat16)
#   VLLM_MAX_MODEL_LEN  default 1024  (matches TRIAGETUNE_MAX_INPUT_TOKENS)
#   VLLM_GPU_FRACTION   default 0.90
#   VLLM_EXTRA_ARGS     extra flags appended verbatim
#   TRIAGETUNE_ADAPTER_DIR  default artifacts/lora_adapter
#   TRIAGETUNE_CACHE_DIR    default .cache/huggingface

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
BASE_REVISION="989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
LORA_NAME="triagetune-banking77-lora"
LORA_RANK=16

ADAPTER_DIR="${TRIAGETUNE_ADAPTER_DIR:-$PROJECT_ROOT/artifacts/lora_adapter}"
CACHE_DIR="${TRIAGETUNE_CACHE_DIR:-$PROJECT_ROOT/.cache/huggingface}"
HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8800}"
DTYPE="${VLLM_DTYPE:-half}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-1024}"
GPU_FRACTION="${VLLM_GPU_FRACTION:-0.90}"
RESULTS_DIR="$PROJECT_ROOT/results"
LOG_FILE="$RESULTS_DIR/serving_vllm_server.log"
PID_FILE="$RESULTS_DIR/serving_vllm_server.pid"

BACKGROUND=0
if [ "${1:-}" = "--background" ]; then
  BACKGROUND=1
  shift
fi

if [ ! -d "$ADAPTER_DIR" ]; then
  echo "adapter directory not found: $ADAPTER_DIR" >&2
  echo "download it first; see serving/RUN_ON_GPU.md" >&2
  exit 1
fi
if [ ! -f "$ADAPTER_DIR/adapter_model.safetensors" ]; then
  echo "adapter weights not found: $ADAPTER_DIR/adapter_model.safetensors" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; vLLM needs an NVIDIA GPU" >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR"

echo "recording environment to results/serving_env.json"
python3 serving/record_env.py --output-file "$RESULTS_DIR/serving_env.json" >/dev/null

# --lora-modules also accepts a JSON object per module in recent vLLM
# releases; the name=path form is the stable short spelling.
ARGS=(
  --model "$BASE_MODEL"
  --revision "$BASE_REVISION"
  --tokenizer "$ADAPTER_DIR"
  --download-dir "$CACHE_DIR"
  --served-model-name "$BASE_MODEL"
  --enable-lora
  --lora-modules "$LORA_NAME=$ADAPTER_DIR"
  --max-lora-rank "$LORA_RANK"
  --max-loras 1
  --dtype "$DTYPE"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_FRACTION"
  --host "$HOST"
  --port "$PORT"
  --no-enable-log-requests
)
if [ -n "${VLLM_EXTRA_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  ARGS+=(${VLLM_EXTRA_ARGS})
fi

echo "base model     $BASE_MODEL"
echo "base revision  $BASE_REVISION"
echo "adapter        $LORA_NAME (rank $LORA_RANK)"
echo "endpoint       http://$HOST:$PORT/v1"

if [ "$BACKGROUND" -eq 0 ]; then
  exec python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
fi

python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}" >"$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" >"$PID_FILE"
echo "server pid $SERVER_PID, log results/serving_vllm_server.log"

for _ in $(seq 1 120); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "server exited during startup; see results/serving_vllm_server.log" >&2
    exit 1
  fi
  if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    echo "server is healthy"
    exit 0
  fi
  sleep 5
done

echo "server did not become healthy within 600 seconds" >&2
exit 1
