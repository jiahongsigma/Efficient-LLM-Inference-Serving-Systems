# Module 7 lab — Long-context serving

> Push to the wall, recover headroom, and measure retrieval honestly.
> Scaffold for `Module 07 — Long-Context Serving.md`. **Status: implemented — first real-hardware run welcome (try it on your GPU; see `run_lab.py`).**

**Server (pick one):** **1× 80 GB H100** (a single 128K-context KV is ~16 GB before weights) —
**Lambda** (H100) or **RunPod** (H100 SXM). Full table: [`../README.md`](../README.md).

## Steps (→ the deliverables)
1. **Hit the wall:** push context up on long-doc-QA until OOM; confirm it matches `mem_estimate`'s
   KV prediction; profile to show **attention compute now dominates** weight reads (M1 crossover).
2. **Recover headroom three ways:** (a) **eviction** (StreamingLLM sinks + window / H2O),
   (b) **KV quantization** (FP8→INT4 KV), (c) **sparse attention** if available; record KV
   recovered + context extension for each.
3. **The retrieval trap (the honesty centerpiece, INVARIANT 5):** score eviction on **both**
   average long-doc-QA **and needle-in-haystack with the needle in the evicted region**. Show
   eviction looking ~free on perplexity but **failing needle retrieval**; KV-quant survives.
4. *(M6 payoff)* if it still won't fit, enable Ring-Attention-style **sequence parallelism**.

## Deliverable
An **accuracy × context × memory** table across {full, eviction, KV-quant, sparse}, accuracy
reported **both** as average long-doc-QA **and** needle-in-haystack retrieval by depth.

## Setup
```bash
vllm serve <model> --port 8000 --max-model-len 131072       # step 2: --kv-cache-dtype fp8, etc.
```

## Skeleton — `run_lab.py`
```python
from common.bench import OpenAIEndpoint, run_open_loop, compute_metrics
from common.traffic import build_needle, build_long_doc_qa
from common.eval import score_needle                         # retrieval by depth (INVARIANT 5)
# Step 3: build_needle(n, context_len=128000, needle_depth_fraction=0.7)  # needle in the evicted region
#         score_needle(results).by_depth   -> compare full vs eviction vs KV-quant
```
