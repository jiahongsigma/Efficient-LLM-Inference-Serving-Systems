# Module 12 lab — How to benchmark without lying to yourself

> Make the same system win, then lose.
> Scaffold for `Module 12 — How to Benchmark Without Lying to Yourself.md`. **Status: implemented — first real-hardware run welcome (try it on your GPU; see `run_lab.py`).**
> *(Fully supported by `common/` today — reuses every earlier lab's result and harness piece.)*

**Server (pick one):** **reuse any single-GPU box** from an earlier lab (RunPod / Lambda). For the
trace-replay scenario a hosted OpenAI-compatible API (**Together** / **Fireworks**) also works,
since you only need an endpoint. Full table: [`../README.md`](../README.md).

## Steps (→ the deliverables)
1. **Win, then lose (centerpiece):** take one earlier conclusion ("INT4 is faster", "prefix caching
   helps") and construct **two defensible benchmarks** — one where it wins, one where it loses — by
   changing only traffic / metric / batch / eval task.
2. **Coordinated omission:** measure the *same* system's tail **closed-loop** vs
   **open-loop-fixed-schedule**; show the closed-loop p99 is dramatically optimistic; quantify the gap.
3. **Statistical rigor + determinism:** run several times, with/without warmup; report variance + CI;
   demonstrate **temp-0 batched non-determinism** with `determinism_check` (and batch-invariant kernels
   restoring it, if available).
4. **Quality + performance:** exhibit a "faster" config that is only faster because it is worse —
   pair its speed with its quality on an *un-saturated* eval.

## Deliverable
A **methodology note** showing the *same system* "winning" and "losing" under defensible-but-different
choices, plus the coordinated-omission gap, the variance/CI/non-determinism demo, and a quality+perf pairing.

## Skeleton — `run_lab.py`
```python
from common.bench import OpenAIEndpoint, run_open_loop, run_closed_loop, compute_metrics, determinism_check, SLO
from common.traffic import build_sharegpt, build_long_doc_qa
EP = OpenAIEndpoint("http://localhost:8000", "<model>")
# Step 2: closed = compute_metrics(run_closed_loop(reqs, EP, concurrency=64))
#         open   = compute_metrics(run_open_loop(reqs, EP, schedule=LAMBDA))   # fixed-schedule
#         compare e2e_p99  ->  closed-loop is the comfortable lie
# Step 3: determinism_check(EP, prompt, n_runs=20, vary_batch=True)
```
