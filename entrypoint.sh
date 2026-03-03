#!/bin/bash
# entrypoint.sh – translates HF Endpoints env vars → vllm serve flags
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/repository}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated}"
TENSOR_PARALLEL="${TENSOR_PARALLEL_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
DTYPE="${DTYPE:-bfloat16}"
PORT="${PORT:-80}"

echo "==> Starting vLLM ${VLLM_VERSION:-unknown} on port ${PORT}"
echo "    model=${MODEL_PATH}  tp=${TENSOR_PARALLEL}  dtype=${DTYPE}"

exec vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size "${TENSOR_PARALLEL}" \
    --dtype "${DTYPE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --model-impl transformers \
    --trust-remote-code \
    --port "${PORT}" \
    --host 0.0.0.0 \
    --uvicorn-log-level info
