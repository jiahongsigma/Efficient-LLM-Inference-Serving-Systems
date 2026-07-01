# Module 1 lab — The roofline of a transformer forward pass

> Place measured prefill and decode points on a real roofline.
> Scaffold for `Module 01 — The Roofline of a Transformer Forward Pass.md`. **Status: implemented — first real-hardware run welcome (try it on your GPU; see `run_lab.py`).**

**Server (pick one):** **1× H100 80GB** — you need *real* achieved HBM bandwidth and SM
utilization, which only real silicon gives. **Lambda** (H100 on-demand) or **RunPod**
(H100 PCIe/SXM). Nsight Compute is easiest on a box where you have root. Full table:
[`../README.md`](../README.md).

## Steps (→ the deliverables)
1. Serve an 8B model (vLLM). Instrument GPU telemetry: `nvidia-smi dmon` (SM util +
   achieved HBM bandwidth) or **Nsight Compute** for a kernel view.
2. **Isolate phases:** long-prompt / 1-token output ⇒ **prefill**; short-prompt /
   long-output ⇒ **decode**. Record FLOP/s and achieved bandwidth for each.
3. **Plot the roofline:** draw the `P` and `B` ceilings (Appendix A), place the two points.
   Decode should sit deep in the memory-bound region (60–80% achieved BW).
4. **Sweep batch** {1,4,16,64}; watch the operating point climb toward the ridge.
5. Validate batch-1 per-token latency against the `weight_bytes / B` bound (§1.3).

## Deliverable
A roofline plot with measured prefill+decode points, the decode achieved-bandwidth figure,
and the batch-sweep curve showing arithmetic intensity rising with batch.

## Setup
```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
nvidia-smi dmon -s u    # or: ncu --set full python your_probe.py
```

## Skeleton — `run_lab.py`
```python
from common.bench import OpenAIEndpoint, Request, run_open_loop, compute_metrics
EP = OpenAIEndpoint("http://localhost:8000", "meta-llama/Llama-3.1-8B-Instruct")
# Step 2: prefill probe = long prompt + max_tokens=1; decode probe = short prompt + max_tokens=512
# Step 4: batch sweep — drive N concurrent decode probes, read achieved BW from dmon alongside
# (the harness gives you TTFT/TPOT/throughput; pair them with nvidia-smi/Nsight numbers)
```
