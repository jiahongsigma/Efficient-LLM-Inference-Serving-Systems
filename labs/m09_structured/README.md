# Module 9 lab — Structured / constrained decoding

> Guarantee the structure, then check what it cost.
> Scaffold for `Module 09 — Structured and Constrained Decoding.md`. **Status: implemented — first real-hardware run welcome (try it on your GPU; see `run_lab.py`).**

**Server (pick one):** **1× 24–48 GB** GPU — **RunPod** or **Lambda**. vLLM (guided decoding) or
**SGLang** (first-class structured generation / XGrammar). Full table: [`../README.md`](../README.md).

## Steps (→ the deliverables)
1. **The guarantee (INVARIANT 6 corollary):** generate JSON-schema output with and without
   constrained decoding on the same prompts; score with `common.eval.score_json_schema`.
   Unconstrained < 100% (parse failures, schema violations); constrained = 100% by construction.
2. **The two costs:** measure the **compile cost** (first request with a fresh grammar) and the
   **per-step TPOT overhead** (constrained vs unconstrained); show caching the compiled grammar
   amortizes the compile cost.
3. **The quality trap (centerpiece):** on a reasoning-plus-structure task, compare (a) forcing
   JSON from the first token vs (b) **reason-in-prose-then-constrain**; score answer quality both
   ways with `score_suite`. Show constraining too early can *lower quality at 100% adherence*.
4. *(frontier/interactions)* XGrammar vs naive FSM overhead; a batch mixing grammars; constrained
   + speculative (the draft must be grammar-aware or `α` collapses).

## Deliverable
The adherence table (→100%); the compile + per-step overhead with the caching amortization; the
**quality comparison** (forced vs reason-then-constrain) proving adherence ≠ quality.

## Setup
```bash
python -m sglang.launch_server --model-path <model> --port 8000   # or: vllm serve <model> (guided decoding)
```

## Skeleton — `run_lab.py`
```python
from common.bench import OpenAIEndpoint, Request, run_open_loop, compute_metrics
from common.eval import score_json_schema, score_suite
SCHEMA = {"type": "object", "required": ["name", "age"],
          "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
# Step 1: same prompts, with vs without guided_json=SCHEMA (via Request.meta["extra_body"]); compare valid_fraction
# Step 3: reasoning task, forced-JSON vs reason-then-constrain; score_suite both ways
```
