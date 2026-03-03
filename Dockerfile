# ── Qwen3.5-MoE custom container for HF Inference Endpoints ──────────────────
# Fixes: "Transformers does not recognize model_type `qwen3_5_moe`"
# Base: vllm/vllm-openai 0.16.0 (latest stable, CUDA 12, Python 3.12)
# Listening port: 80
# Model mount:    /repository  (HF Endpoints convention)
# ─────────────────────────────────────────────────────────────────────────────
FROM vllm/vllm-openai:v0.16.0

# Transformers 5.2.0 (stable) adds qwen3_5_moe to MODEL_TYPE_TO_CONFIG_CLASS.
# accelerate  – required by Transformers for device_map / MoE dispatch
# safetensors – fast weight loading
# sentencepiece / tiktoken – Qwen tokenizer
RUN pip install --no-cache-dir \
        "transformers==5.2.0" \
        "accelerate>=0.34.0" \
        "safetensors>=0.4.3" \
        "sentencepiece>=0.2.0" \
        "tiktoken>=0.7.0"

# vLLM's built-in /health route is available at GET /health (HTTP 200 when ready).
# HF Endpoints polls that path, so no extra sidecar is needed.

# Copy the entrypoint that translates HF env vars → vllm serve flags.
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 80

ENTRYPOINT ["/entrypoint.sh"]
