# Module 3 lab — Quantization: shrinking the numerator

> Quantize, and prove the *mechanism*, not just the speedup.
> Scaffold for `Module 03 — Quantization - Shrinking the Numerator.md`. **Status: implemented — first real-hardware run welcome (try it on your GPU; see `run_lab.py`).**

**Server (pick one):** **1× 24–48 GB Ada/Hopper** GPU — FP8 needs Ada+ (L40S / 4090 / H100);
INT4 runs anywhere. **RunPod** (L40S / 4090) or **Vast.ai** (cheapest 4090). Full table:
[`../README.md`](../README.md).

## Steps (→ the deliverables)
1. **Footprint:** serve one model at FP16, INT8, INT4 (GPTQ/AWQ), FP8; confirm `mem_estimate`
   predicts each weight footprint from `bytes_per_param`.
2. **The mechanism split (the centerpiece):** with Module 1's phase isolation, measure
   **prefill tok/s** *and* **decode tok/s** for FP16 vs **weight-only INT4** vs **FP8 w+a**.
   At low batch: INT4 speeds decode, ~not prefill; FP8 speeds both. **Sweep decode batch** and
   find where INT4's decode win erodes (§3.2 refinement).
3. **Per-task quality (reuse `common.eval`):** FP16/AWQ/GPTQ/FP8 across MMLU, GSM8K, HumanEval,
   IFEval (+ a harder discriminator); tabulate **per task**, never averaged.
4. *(optional)* hook a forward pass; plot per-channel activation magnitude (the outliers).

## Deliverable
A **quality × throughput × memory** table — throughput split into **prefill/decode**, quality
**per task** — plus the workload→scheme recommendation and the batch at which it flips.

## Setup
```bash
vllm serve <model> --port 8000 --quantization awq      # then gptq, fp8; FP16 baseline = no flag
```

## Skeleton — `run_lab.py`
```python
from common.bench import OpenAIEndpoint, Request, run_open_loop, compute_metrics
from common.eval import score_suite, quant_delta      # per-task, never averaged (INVARIANT 6)
# Step 2: prefill probe (long prompt, 1 tok) vs decode probe (short prompt, long output), each config
# Step 3: run the eval suite per config; quant_delta(fp16, candidate) per task
```
