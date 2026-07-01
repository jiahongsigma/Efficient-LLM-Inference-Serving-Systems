# Module 2 lab — Attention, the KV cache, and where the memory goes

> Measure the cache, fit the size law, watch context bind the roofline.
> Scaffold for `Module 02 — Attention, the KV Cache, and Where the Memory Goes.md`. **Status: implemented — first real-hardware run welcome (try it on your GPU; see `run_lab.py`).**

**Server (pick one):** **1× 24–48 GB** GPU (A10 / L40S / 4090) — **RunPod** or **Lambda**.
A 48 GB card lets you push context to OOM comfortably. Full table: [`../README.md`](../README.md).

## Steps (→ the deliverables)
1. Extend Lab 0's `mem_estimate` **KV term**; serve an 8B model; measure reserved KV across
   context {2K, 8K, 32K} × batch {1, 8, 32}; **fit the KV law**; predict and confirm OOM.
2. **Traffic, not just footprint:** hold batch fixed, sweep *context* {2K…64K}; measure decode
   throughput falling as each step rereads a bigger cache (§2.2 — the Module 1 link).
3. `--kv-cache-dtype fp8`: per-token KV bytes halve; record footprint + throughput + accuracy.
4. **MHA vs GQA:** compute KV-bytes/token analytically for a full-MHA model (Llama-2-7B) vs a
   GQA model (Llama-3.1-8B); validate against each model's measured footprint; relate to `H/G`.
5. *(kernel)* micro-benchmark naive `softmax(QKᵀ)·V` vs `scaled_dot_product_attention`;
   plot `max_memory_allocated()` vs sequence length (O(S²) vs O(S)).

## Deliverable
Extended `mem_estimate` (KV validated vs OOM within ~10%), the context-throughput curve,
the FP16-vs-FP8 KV comparison, the MHA-vs-GQA table, and the linear-vs-quadratic attention plot.

## Setup
```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000 --max-model-len 65536
# step 3: add --kv-cache-dtype fp8
```

## Skeleton — `run_lab.py`
```python
from common.bench import OpenAIEndpoint, compute_metrics
from common.traffic import build_long_doc_qa   # long contexts for the sweep
# Step 1/2: sweep context and batch; read reserved KV from the server /metrics; fit KV_total
# Step 5: a standalone torch micro-bench (no server) for the O(S) vs O(S^2) memory plot
```
