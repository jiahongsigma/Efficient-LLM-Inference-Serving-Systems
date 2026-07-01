# Module 4 lab — Batching and scheduling

> Find the knee, the honest way. The runnable companion to
> `Module 04 — Batching and Scheduling.md`, driving the `common/` harness
> **open-loop throughout**.

Point the knee/goodput sweep at a real engine with `--endpoint http://host:port
--model NAME`. (A local `--endpoint sim` dev mode exists for wiring the script
without a GPU, but the real numbers come from the server below.)

**Server (pick one):** a single **24–48 GB** GPU — **RunPod** (RTX 4090 / L40S) or
**Lambda** (A10 / A100). The knee & goodput sweep needs a real engine; exps 1 & 4
are simulator-modelled (a real engine is always continuous, and chunked prefill is
a launch flag). Full per-lab provider table: [`../README.md`](../README.md).

## Run it

```bash
pip install -r common/requirements.txt
python labs/m04_batching/run_lab.py            # full dry-run, no GPU (~1 min)
python labs/m04_batching/run_lab.py --exp 2    # just the knee + goodput sweep
```

Artifacts (CSV always; PNG if matplotlib is present) land in `labs/m04_batching/out/`.

### Against a real engine (the metered run)

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000 --enable-chunked-prefill
python labs/m04_batching/run_lab.py --exp 2 \
    --endpoint http://localhost:8000 --model meta-llama/Llama-3.1-8B-Instruct \
    --lam-sweep 2,4,8,16,24,32,48 --slo-e2e 5
```

The **knee/goodput sweep (exp 2/3)** drives the real engine and finds its real
operating point. **Exp 1** (static vs continuous) and **exp 4** (chunked prefill)
are simulator-modelled, because a production engine is *always* continuous and
chunked-prefill is a launch flag (`--enable-chunked-prefill`) — to see exp 4 for
real, run the sweep with the flag on vs off and compare decode TPOT.

## The four deliverables

| # | Experiment | Confirms | Output |
|---|---|---|---|
| 1 | **static vs continuous** — length-variant open-loop stream into each | §4.2–4.3: static's tail explodes (head-of-line blocking + no mid-flight admission); continuous holds | `exp1_static_vs_continuous.{csv,png}` |
| 2 | **the knee** — sweep offered load λ | §4.4: throughput saturates, p99 explodes; name the saturating wall | `exp23_knee_goodput.{csv,png}` |
| 3 | **goodput** — same sweep, against an SLO | §4.4: goodput collapses past the knee even while throughput plateaus; run near-but-below it | `exp23_knee_goodput.{csv,png}` |
| 4 | **chunked prefill** — decode-heavy stream + a big prefill | §4.5: a giant prefill stalls in-flight decode TPOT; chunking flattens it (motivates disaggregation, M8) | `exp4_chunked_prefill.{csv,png}` |

## Self-check (known-good shape, `--endpoint sim`, default seed)

Absolute numbers depend on the `SimEndpoint` cost knobs (and the event loop's
~1 ms timer granularity); the **shape** is what should match:

- **Exp 1:** static `p99` ≈ **2–3×** continuous `p99` at similar throughput — short
  requests wait for the batch's longest member.
- **Exp 2:** throughput rises with λ, then **saturates** (here ~200 req/s); `p99`
  rises gently, then **explodes** past the knee (~λ=250).
- **Exp 3:** goodput tracks throughput up to **λ\*** (~200), then **collapses** as
  p99 breaks the SLO while raw throughput stays flat.
- **Exp 4:** decode TPOT with chunked prefill **off** ≈ **2–3×** higher than **on**.

## Key flags

`--exp all|1|2|3|4`, `--concurrency`, `--decode-ms`, `--gen-cap`, `--lam-sweep`,
`--slo-e2e`, `--rate1`, `--long-tokens`/`--short-tokens`/`--long-every` (exp 1),
`--big-prompt-tokens`/`--chunk-tokens` (exp 4), `--seed`, `--out`.

## How it uses the harness

- `run_open_loop` everywhere — INVARIANT 2: a closed-loop client would self-throttle
  and **hide the knee**.
- `SimEndpoint` (continuous batching) vs **`StaticBatchEndpoint`** (the broken
  baseline: fixed batches held to the slowest member, one batch at a time).
- `SimEndpoint(model_prefill_interference=True, chunked_prefill=…)` models a giant
  prefill starving in-flight decode, and chunked prefill interleaving it.
- `compute_metrics(..., slo=SLO(e2e=…), warmup=…)` for TTFT/TPOT/p99 and **goodput**.
