# ── Qwen3.5-MoE custom container for HF Inference Endpoints ──────────────────
# Fixes: "Transformers does not recognize model_type `qwen3_5_moe`"
# Base: vllm/vllm-openai 0.8.5 (stable, ships CUDA 12.4, Python 3.12)
# Listening port: 80
# Model mount:    /repository  (HF Endpoints convention)
# ─────────────────────────────────────────────────────────────────────────────
FROM vllm/vllm-openai:v0.8.5

# Upgrade Transformers from git main so qwen3_5_moe is recognised.
# Pin to a commit once HF ships a stable release that includes the arch.
# accelerate  – required by Transformers for device_map / MoE dispatch
# safetensors – fast weight loading
# sentencepiece / tiktoken – Qwen tokenizer
RUN pip install --no-cache-dir \
        "git+https://github.com/huggingface/transformers.git" \
        "accelerate>=0.34.0" \
        "safetensors>=0.4.3" \
        "sentencepiece>=0.2.0" \
        "tiktoken>=0.7.0" \
    && pip check

# vLLM's built-in /health route is available at GET /health (HTTP 200 when ready).
# HF Endpoints polls that path, so no extra sidecar is needed.

# Copy the entrypoint that translates HF env vars → vllm serve flags.
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 80

ENTRYPOINT ["/entrypoint.sh"]
