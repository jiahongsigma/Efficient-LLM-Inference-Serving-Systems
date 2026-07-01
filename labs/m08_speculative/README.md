# Module 8 lab — Attacking the sequential dependency

> Measure the speculative speedup, and find where it stops paying.
> Scaffold for `Module 08 — Attacking the Sequential Dependency.md`. **Status: implemented — first real-hardware run welcome (try it on your GPU; see `run_lab.py`).**

**Server (pick one):** **1× 24–48 GB** GPU (draft + target both fit) — **RunPod** or **Lambda**.
A single L40S/A100 holds an 8B target + a 1B draft. Full table: [`../README.md`](../README.md).

## Steps (→ the deliverables)
1. **Speedup vs acceptance:** run draft+target speculative decoding at fixed `k`; measure
   end-to-end decode speedup and the realized acceptance rate `α` on realistic prompts. Sweep
   `k`, find the optimum, confirm the `(1−α^{k+1})/(1−α)` expected-tokens relation.
2. **Where it stops paying (the centerpiece):** hold the method fixed and **sweep batch size**;
   show speedup large at batch 1 and **fading to ~0 (or negative) at high batch** — batching
   already consumed the idle FLOPs (M1). Conclude speculation is a *latency*, not throughput, win.
3. **Self-speculation:** compare a separate draft model vs **EAGLE / Medusa**; show higher `α`,
   no second model to maintain.
4. **Disaggregation:** co-located prefill+decode vs **disaggregated** pools on a long-prefill
   workload; measure interference removed + the **KV-transfer cost**.

## Deliverable
The **speedup-vs-acceptance** curve with optimal `k`, **and** the **speedup-vs-batch** curve
marking where speculation stops paying; the self-speculation comparison.

## Setup
```bash
vllm serve <8B target> --port 8000 \
  --speculative-model <1B draft> --num-speculative-tokens 5
```

## Skeleton — `run_lab.py`
```python
from common.bench import OpenAIEndpoint, run_open_loop, compute_metrics
from common.traffic import build_sharegpt
# Step 2: for batch in (1, 4, 16, 64): drive that concurrency; record decode speedup vs a no-spec baseline
#         (point at a spec server and a non-spec server on two ports; compare tokens/s)
```
