# Qwen3.5-MoE on HF Inference Endpoints – Full Deployment Guide

---

## 1. Root Cause & Verification Checklist

### Root Cause
`vllm/vllm-openai:latest` (v0.14.1 at time of failure) ships with a **Transformers release that
predates the `qwen3_5_moe` architecture registration** (~4.52 needed, shipped ~4.51). When vLLM
calls `AutoConfig.from_pretrained("/repository")` it reads `config.json → model_type: qwen3_5_moe`
and raises a `ValidationError` because no mapping exists in the installed Transformers.

### Verification Checklist
- [ ] Inside the failing container: `python -c "import transformers; print(transformers.__version__)"`  
      → should be ≥ 4.52 / git-main post-March 2025 once `qwen3_5_moe` was merged.
- [ ] `cat /repository/config.json | python -m json.tool | grep model_type`  
      → confirms `"model_type": "qwen3_5_moe"`.
- [ ] `python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('/repository', trust_remote_code=False)"`  
      → no error in fixed image.
- [ ] `curl -s http://localhost:80/health` → `{"status":"ok"}` after vLLM starts.
- [ ] `curl -s http://localhost:80/v1/models | python -m json.tool` → lists served model name.

---

## 2. Dockerfile

See `./Dockerfile` in this directory.

Key decisions:
| Decision | Rationale |
|---|---|
| `vllm/vllm-openai:v0.8.5` | Last stable tag with CUDA 12.4; avoids float drift in `latest` |
| `transformers @ git+main` | Only guaranteed way to get `qwen3_5_moe` until a release ships it |
| `--trust-remote-code` | Qwen models sometimes need it for custom tokenizer/config code |
| `bfloat16` dtype | Qwen3.5 MoE was trained in bf16; avoids NaN issues with fp16 |
| `tensor_parallel_size=2` | Matches 2× A100-80 GB or 2× H100-80 GB; adjust per HW |
| port 80 | HF Endpoints requires HTTP on port 80 (no TLS termination by container) |
| `/health` | vLLM exposes `GET /health` natively → HF health probe just works |

---

## 3. Container Registry & Push (GitHub Container Registry — zero setup)

GHCR is free, built into GitHub, and needs **no secrets to configure** — the workflow
uses the automatic `GITHUB_TOKEN`.

### Steps
1. Create a GitHub repo (public or private).
2. Push `Dockerfile`, `entrypoint.sh`, and `.github/workflows/build-push.yml` to `main`.
3. GitHub Actions builds the image and pushes it to `ghcr.io/<your-username>/qwen3-5-moe-vllm:latest`.
4. **(One-time, if repo is private):** go to your GitHub profile → Packages → select
   the package → Package Settings → **Change visibility → Public**, or configure HF
   Endpoints with a GitHub PAT (see below).

**No secrets, no registry accounts, no CLI tools.**

Image URL: `ghcr.io/<your-github-username>/qwen3-5-moe-vllm:latest`

### Private image? Give HF Endpoints access
If you keep the package private, HF Endpoints needs credentials to pull:
- **Username:** your GitHub username
- **Password:** a GitHub **Personal Access Token (classic)** with `read:packages` scope
  (Settings → Developer settings → Personal access tokens → Tokens (classic) → Generate).

---

## 4. HF Endpoints UI Configuration

Navigate to: https://ui.endpoints.huggingface.co → **New Endpoint**

### Step-by-step

| Field | Value |
|---|---|
| **Model repository** | `huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated` |
| **Endpoint name** | `huihui-qwen3-5-35b-a3b` (or your choice) |
| **Cloud / Region** | Any (pick closest to your users) |
| **Instance type** | `nvidia-a100-80gb × 2` or `nvidia-h100-80gb × 2` |
| **Container type** | **Custom** (toggle from "Managed") |

#### Custom Container fields (appear after selecting Custom)
| Field | Value |
|---|---|
| **Container image URL** | `ghcr.io/<your-github-username>/qwen3-5-moe-vllm:latest` |
| **Registry username** | *(blank if public; GitHub username if private)* |
| **Registry password** | *(blank if public; GitHub PAT with `read:packages` if private)* |
| **Port** | `80` |
| **Health route** | `/health` |

#### Environment Variables (add in the Env Vars section)
| Key | Value |
|---|---|
| `MODEL_PATH` | `/repository` |
| `SERVED_MODEL_NAME` | `huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated` |
| `TENSOR_PARALLEL_SIZE` | `2` |
| `DTYPE` | `bfloat16` |
| `MAX_MODEL_LEN` | `32768` |
| `PORT` | `80` |
| `HF_TOKEN` | Your HF token (if model is gated) |

Click **Create Endpoint** → wait for status → **Running**.

---

## 5. Failure-Mode Playbook

If the endpoint fails after applying the fix, check these 8 causes in order:

### Cause 1 – Wrong vLLM / Transformers version still installed
**Log line**: `does not recognize this architecture` or `KeyError: 'qwen3_5_moe'`  
**Fix**: Verify `pip show transformers` inside container shows git-main build date > March 2025.
Add `RUN pip show transformers` to Dockerfile build to surface the version in build logs.

### Cause 2 – GPU count mismatch (`tensor_parallel_size` > available GPUs)
**Log line**: `AssertionError: tensor_parallel_size (2) > number of GPUs (1)`  
**Fix**: Change `TENSOR_PARALLEL_SIZE` env var to match actual GPU count, or upgrade instance type.

### Cause 3 – OOM / CUDA out of memory
**Log line**: `torch.cuda.OutOfMemoryError` or `CUDA error: out of memory`  
**Fix**: Set `MAX_MODEL_LEN=16384` (or lower). Add `--gpu-memory-utilization 0.90` to entrypoint.

### Cause 4 – flash-attn missing / wrong version
**Log line**: `ImportError: flash_attn` or `FlashAttentionError`  
**Fix**: Add to Dockerfile:
```dockerfile
RUN pip install flash-attn --no-build-isolation
```
Or force eager attention: add `--enforce-eager` to vllm serve flags in entrypoint.

### Cause 5 – trust_remote_code not set
**Log line**: `Loading ... requires you to execute the configuration file … trust_remote_code=True`  
**Fix**: `--trust-remote-code` is already in the entrypoint; confirm it reaches vllm serve.

### Cause 6 – Tokenizer mismatch / sentencepiece missing
**Log line**: `sentencepiece` ImportError or `tiktoken` version error  
**Fix**: Dockerfile already installs both; if still failing pin versions:
```
sentencepiece==0.2.0
tiktoken==0.7.0
```

### Cause 7 – HF model not accessible (gated / private)
**Log line**: `401 Client Error` or `Repository not found`  
**Fix**: Set `HF_TOKEN` env var in HF Endpoints UI with a token that has read access to the model.

### Cause 8 – Health check timing out before model loads
**Log line**: HF Endpoints shows `Unhealthy` despite vLLM still loading weights  
**Fix**: In HF Endpoints settings increase the **startup timeout** (default 300 s). For a 35B MoE model on 2× A100, allow at least 600 s. If no UI control exists, add a health wrapper that returns 200 immediately:
```bash
# In entrypoint.sh, before exec vllm serve …
python3 -c "
import http.server, threading, os
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'ok')
    def log_message(self, *a): pass
t = threading.Thread(target=http.server.HTTPServer(('', 8099), H).serve_forever, daemon=True)
t.start()
" &
```
Then configure `/health` on port 8099 in HF UI while vLLM serves on port 80.  
(Better: HF Endpoints uses the same port; just wait for vLLM `/health` to go live.)

---

## 6. Final Copy/Paste Blocks

### Dockerfile
```dockerfile
FROM vllm/vllm-openai:v0.8.5

RUN pip install --no-cache-dir \
        "git+https://github.com/huggingface/transformers.git" \
        "accelerate>=0.34.0" \
        "safetensors>=0.4.3" \
        "sentencepiece>=0.2.0" \
        "tiktoken>=0.7.0" \
    && pip check

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 80
ENTRYPOINT ["/entrypoint.sh"]
```

### entrypoint.sh
```bash
#!/bin/bash
set -euo pipefail
MODEL_PATH="${MODEL_PATH:-/repository}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated}"
TENSOR_PARALLEL="${TENSOR_PARALLEL_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
DTYPE="${DTYPE:-bfloat16}"
PORT="${PORT:-80}"

exec vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size "${TENSOR_PARALLEL}" \
    --dtype "${DTYPE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --trust-remote-code \
    --port "${PORT}" \
    --host 0.0.0.0
```

### No secrets needed for GitHub Actions
The workflow uses the built-in `GITHUB_TOKEN` — nothing to configure.

### If GHCR package is private, set these in HF Endpoints
```
Registry username  = <your-github-username>
Registry password  = <GitHub PAT with read:packages scope>
```

### HF Endpoints UI Checklist
- [ ] Model repo: `huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated`
- [ ] Container type: **Custom**
- [ ] Image URL: `ghcr.io/<your-github-username>/qwen3-5-moe-vllm:latest`
- [ ] Registry username: *(blank if public)*
- [ ] Registry password: *(blank if public)*
- [ ] Port: `80`
- [ ] Health route: `/health`
- [ ] Env `MODEL_PATH` = `/repository`
- [ ] Env `SERVED_MODEL_NAME` = `huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated`
- [ ] Env `TENSOR_PARALLEL_SIZE` = `2`
- [ ] Env `DTYPE` = `bfloat16`
- [ ] Env `MAX_MODEL_LEN` = `32768`
- [ ] Env `PORT` = `80`
- [ ] Env `HF_TOKEN` = (your token if model is gated)

### Validation commands (run after endpoint is Running)
```bash
BASE=https://<your-endpoint>.endpoints.huggingface.cloud
TOKEN=hf_xxx

# Health
curl -s "${BASE}/health"

# Models
curl -s -H "Authorization: Bearer ${TOKEN}" "${BASE}/v1/models" | python -m json.tool

# Chat completions
curl -s "${BASE}/v1/chat/completions" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated",
    "messages": [{"role":"user","content":"Hello, who are you?"}],
    "max_tokens": 128
  }' | python -m json.tool
```
