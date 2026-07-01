# Module 11 lab — Frameworks, the API layer, and resilience

> Build the gateway, then break it on purpose.
> Scaffold for `Module 11 — Frameworks, the API Layer, and Resilience.md`. **Status: implemented — first real-hardware run welcome (try it on your GPU; see `run_lab.py` + `gateway.py`).**

**Server (pick one):** **two backends** — either **2× small GPU pods** (one vLLM + one SGLang) on
**RunPod** / **Lambda**, or one GPU running two small models on two ports. The gateway itself is
CPU and can run on your laptop in front of the remote engines. Full table: [`../README.md`](../README.md).

## Steps (→ the deliverables)
1. **Build the gateway:** stand up **vLLM and SGLang** behind a minimal (~150-line) OpenAI-compatible
   gateway with a failover chain, auth, a **capability-parity guard** (reject a tool-call request a
   backend can't serve), and **telemetry** logging metadata-only TTFT/TPOT/tokens/status.
2. **The failover gap (centerpiece, INVARIANT 1):** under **open-loop** load, **kill a backend
   mid-stream**. Measure in-flight requests lost, the **p99 spike during the failure window**
   (not the average), and recovery time.
3. **Load shedding:** drive past the knee; compare admitting-everything (collapse) vs
   admission-control (reject excess) — show shedding **protects goodput**.
4. *(frontier)* **KV-aware routing:** round-robin vs routing to the replica holding the cached
   prefix; show cache hits + lower TTFT.

## Deliverable
The gateway **and** a `resilience.md` quantifying each failure mode — the failover gap, in-flight
loss, recovery time — measured **open-loop, on the tail**; plus the load-shedding goodput comparison.

## Setup
```bash
vllm serve <model> --port 8000                         # backend A
python -m sglang.launch_server --model-path <model> --port 8001   # backend B
```

## Skeleton — `run_lab.py` + `gateway.py`
```python
# gateway.py: ~150-line FastAPI /v1/chat/completions that proxies to [A, B] with failover + telemetry
from common.bench import OpenAIEndpoint, SimEndpoint, FaultSpec, run_open_loop, compute_metrics
# Step 2: point run_open_loop at the gateway; schedule FaultSpec(at_time=…, kind="kill_backend")
#         — against real backends, kill the process out of band at that time; report p99 in the window
```
