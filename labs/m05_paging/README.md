# Module 5 lab — Serving-time memory management

> Reclaim the memory, reuse the prefix — on the *right* workload.
> Scaffold for `Module 05 — Serving-Time Memory Management.md`. **Status: implemented — first real-hardware run welcome (try it on your GPU; see `run_lab.py`).**
> *(Fully supported by `common/` today — the cheapest lab to wire after the built two.)*

**Server (pick one):** **1× 24–48 GB** GPU (A10 / L40S / 4090) — **RunPod** or **Lambda**.
Full table: [`../README.md`](../README.md).

## Steps (→ the deliverables)
1. **Paging vs contiguous:** at fixed KV memory, compare achievable **batch** + throughput under
   PagedAttention vs the `max_model_len`-reservation baseline `mem_estimate` predicts. Show
   several× more concurrency, lifting throughput toward Module 1's ceiling.
2. **Prefix sharing — and why the workload decides:** on **long-doc-QA** (shared document),
   prefix caching off vs on; **sweep `prefix_share_ratio`** and show the gain scale. Then run
   the *same* test on **ShareGPT** and show ≈ zero benefit (**INVARIANT 4**).
3. **Memory pressure:** drive concurrency past KV capacity; observe **preemption** (swap vs
   recompute); measure its latency cost.

## Deliverable
Paging-vs-contiguous achievable-batch/throughput; the prefix-cache **gain-vs-sharing-ratio**
curve on long-doc-QA **beside** the ShareGPT null result; the preemption latency cost.

## Setup
```bash
vllm serve <model> --port 8000 --enable-prefix-caching     # step 1 baseline: --no-enable-prefix-caching
```

## Skeleton — `run_lab.py`
```python
from common.bench import OpenAIEndpoint, run_open_loop, compute_metrics
from common.traffic import build_long_doc_qa, build_sharegpt
EP = OpenAIEndpoint("http://localhost:8000", "<model>")
# Step 2: for r in [0,0.25,0.5,0.75,1.0]:
#   reqs = build_long_doc_qa(200, prefix_share_ratio=r, rate_or_trace=...); m = compute_metrics(run_open_loop(...))
#   record m.cache_hit_rate, m.throughput_req_s   -> then repeat on build_sharegpt (expect cache_hit ~0)
```
