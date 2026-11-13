#!/usr/bin/env bash
# RFC-0001 CUDA interface: implemented, pending real-device verification.
# Run on Linux with a supported CUDA environment and prepared Qwen3 weights.
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a local Qwen3-0.6B model directory}"
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model directory does not exist: $MODEL_PATH" >&2
  exit 1
fi
if ! command -v nano-xpu >/dev/null 2>&1; then
  echo "nano-xpu is unavailable. Install the project with the server extra in a CUDA environment." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec nano-xpu serve "$MODEL_PATH" \
  --served-model-name "${SERVED_MODEL_NAME:-qwen3-0.6b}" \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-8000}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
  --distributed-port "${DISTRIBUTED_PORT:-2333}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-16384}" \
  --max-num-seqs "${MAX_NUM_SEQS:-512}" \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}" \
  --enforce-eager
