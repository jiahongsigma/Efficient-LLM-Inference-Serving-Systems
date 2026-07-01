# Module 6 lab — Multi-GPU parallelism

> Make the communication cost visible.
> Scaffold for `Module 06 — Multi-GPU Parallelism.md`. **Status: implemented — first real-hardware run welcome (try it on your GPU; see `run_lab.py`).**

**Server (pick one):** **2–4× A100/H100 with NVLink** (TP needs NVLink; PP tolerates PCIe).
**Lambda** (8×A100/H100 NVLink nodes) or **CoreWeave / Crusoe** (multi-GPU H100). For the
TP-collapses-on-PCIe contrast, a box where you can pin ranks across NVLink vs PCIe helps.
Full table: [`../README.md`](../README.md).

## Steps (→ the deliverables)
1. Serve a model that won't fit on one GPU at **TP=2** and **TP=4** (open-loop load). Measure
   throughput + **single-request latency**; plot scaling vs the ideal linear line — the gap *is*
   the communication cost. Classify the binding resource (compute vs interconnect).
2. **The interconnect decides:** measure all-reduce time vs interconnect BW (NVLink ~900 GB/s vs
   PCIe ~64 GB/s, Appendix A). If you can place TP ranks across PCIe, show throughput collapse.
3. **TP vs PP character:** TP lowers single-request latency; PP needs many in-flight microbatches
   to amortize the `(p−1)/(m+p−1)` bubble and does **not** lower single-request latency.
4. **Dense vs MoE:** contrast a dense model under TP with a Mixtral-class MoE under **expert
   parallelism**; observe the all-to-all dispatch/combine cost.

## Deliverable
The **parallel-scaling curve** (throughput + latency vs TP degree) with the **binding resource
named at each point**; the all-reduce-time-vs-interconnect relation; the TP-vs-PP contrast.

## Setup
```bash
vllm serve <70B model> --port 8000 --tensor-parallel-size 2   # then 4
```

## Skeleton — `run_lab.py`
```python
from common.bench import OpenAIEndpoint, run_open_loop, compute_metrics
from common.traffic import build_sharegpt
# Step 1: for tp in (2, 4): point at the TP=tp server; m = compute_metrics(run_open_loop(build_sharegpt(...)))
#         record throughput + single-request latency; plot vs ideal-linear (gap = comm cost)
```
