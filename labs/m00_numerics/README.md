# Module 0 lab — Numbers, tokens, and memory

> Build a memory calculator you trust, then confront it with a real allocator.
> Scaffold for `Module 00 — Numbers, Tokens, and Memory.md`. **Status: implemented — first real-hardware run welcome (try it on your GPU; see `run_lab.py`).**

**Server (pick one):** the calculator is **pure math — runs on your laptop, no GPU**.
To *validate* it, rent any single **24 GB** GPU for ~20 min: **RunPod** (RTX 4090,
~$0.3–0.4/hr) or **Vast.ai** (cheapest 3090/4090). Full table: [`../README.md`](../README.md).

## Steps (→ the deliverables)
1. Implement `mem_estimate(model_config, weight_dtype, kv_dtype, context_len, batch)` →
   predicted VRAM split into **weights / KV / overhead** (Module 0 §0.3–0.4).
2. Produce the **params↔bytes table** for 3 models (Llama-3.1-8B, Qwen2.5-14B, Mistral-7B)
   across BF16 / INT8 / INT4.
3. *(GPU)* Load one model in vLLM at two precisions; read the **reserved VRAM** and the
   **KV-cache blocks** it reports; compare to your estimate.
4. Explain the gap (allocator rounding, paging block size, CUDA-graph capture).

## Deliverable
`mem_estimate()` validated against measured VRAM within **~15%**, plus the params↔bytes table.
*(This `mem_estimate()` is reused by Labs 2, 3, 5, 7 — promote it into `common/` when it works.)*

## Setup (validation box)
```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
nvidia-smi            # reserved VRAM; the server logs report KV-cache blocks
```

## Skeleton — `run_lab.py`
```python
def mem_estimate(cfg, weight_dtype, kv_dtype, context_len, batch):
    bpp = {"bf16": 2, "fp16": 2, "fp8": 1, "int8": 1, "int4": 0.5}
    weights  = cfg["params_b"] * 1e9 * bpp[weight_dtype]
    kv       = 2 * cfg["layers"] * cfg["kv_heads"] * cfg["head_dim"] * bpp[kv_dtype] * context_len * batch
    overhead = 1.5e9
    return {"weights": weights, "kv": kv, "overhead": overhead, "total": weights + kv + overhead}
# TODO 2: params<->bytes table across BF16/INT8/INT4
# TODO 3: on the GPU box, compare total to vLLM's reserved VRAM (target: within 15%)
```
