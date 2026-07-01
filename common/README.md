# `common/` — the shared lab harness

The package every lab imports. It defines load generation, metrics, workloads,
and quality scorers **once**, so no lab re-implements (or fudges) them. The
seven correctness invariants (mapped to their tests in the table below) are the
acceptance criteria — if `common` upholds them, every lab's headline number is
honest by construction.

## Install & test

```bash
pip install -r common/requirements.txt
pytest common/tests            # 20 tests, ~5s, no GPU
# or, without pytest:
python -m common.tests.test_harness
```

## Two endpoints: real and simulated

Everything talks to a backend through `Endpoint.stream`. There are two:

- **`OpenAIEndpoint(base_url, model)`** — a real `/v1/chat/completions` server
  (vLLM, SGLang, or your gateway), streamed over httpx. This is what the labs
  meter against a rented GPU.
- **`SimEndpoint(...)`** — a deterministic, in-process **cost model** (not a real
  model). It simulates a prefix cache, a concurrency limit (so open-loop
  overload actually queues — the knee), and fault injection. Use it to develop
  and unit-test a lab with **no GPU**, then flip to `OpenAIEndpoint` for the
  real measurement. The entire test suite runs on `SimEndpoint`.

```python
import asyncio
from common.bench import OpenAIEndpoint, SimEndpoint, run_open_loop, compute_metrics, SLO
from common.traffic import build_sharegpt

# --- real engine: vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000 ---
ep = OpenAIEndpoint("http://localhost:8000", "meta-llama/Llama-3.1-8B-Instruct")

# --- or dry-run with no GPU ---
# ep = SimEndpoint(max_concurrency=8)

async def main():
    reqs = build_sharegpt(200, rate_or_trace=20.0, seed=0)         # open-loop Poisson @ 20 req/s
    results = await run_open_loop(reqs, ep, api_key="…")
    m = compute_metrics(results, slo=SLO(e2e=5.0, ttft=0.5), warmup=10)
    print(f"ttft_p99={m.ttft_p99:.3f}s  e2e_p99={m.e2e_p99:.3f}s  goodput={m.goodput_req_s:.1f} req/s")

asyncio.run(main())
```

The drivers are backend-agnostic: swap `SimEndpoint` ↔ `OpenAIEndpoint` and
nothing else changes.

## The agentic loop (Module 10)

```python
from common.bench import run_agentic_session, compute_session_metrics
from common.traffic import build_agentic_sessions

stable = build_agentic_sessions(8, turns=20, increment_tokens=600, system_tokens=1500,
                                tool_latency=2.0, prefix_stable=True, seed=0)
perturbed = build_agentic_sessions(8, turns=20, increment_tokens=600, system_tokens=1500,
                                   tool_latency=2.0, prefix_stable=False, seed=0)   # silent cache death

srs = [await run_agentic_session(s, ep, kv_policy="hold") for s in stable]
sm = compute_session_metrics(srs)
# sm.latency_breakdown -> {"model":…, "tool":…, "reprefill":…}, sm.cache_hit_rate, sm.slot_utilization
```

`prefix_stable=False` injects a per-turn perturbation so cross-turn cache hits
silently vanish — re-prefill jumps from `O(T)` to `O(T²)` with no error raised,
exactly Module 10 §10.2. Compare the two runs' `cache_hit_rate` and
`reprefill_tokens_total`.

## Invariant → test map

| # | Invariant | Test(s) in `tests/test_harness.py` |
|---|---|---|
| 1 | latency from `intended_send_time` (coordinated omission) | `test_inv1_latency_from_intended_not_actual`, `test_inv1_queue_wait_counts_toward_latency` |
| 2 | open-loop default; closed-loop only for the M12 contrast | `test_inv2_open_loop_reveals_knee_closed_hides` |
| 3 | warmup discarded; CI over ≥3 repeats | `test_inv3_warmup_discarded`, `test_inv3_aggregate_runs_reports_ci` |
| 4 | prefix caching tested on long-doc-QA **and** ShareGPT | `test_inv4_prefix_cache_helps_only_with_shared_prefix` |
| 5 | long-context accuracy = needle retrieval, not perplexity | `test_inv5_needle_scored_by_depth`, `test_inv5_builder_plants_needle` |
| 6 | quality per task, never one mean; saturation flagged | `test_inv6_per_task_and_saturation_and_delta` |
| 7 | agentic measured end-to-end & decomposed, with cache-hit rate | `test_inv7_agentic_decomposition_and_cache_fragility`, `test_inv7_kv_policy_offload_frees_the_slot` |

Plus: structured-output adherence (M9), determinism under batch change (M12),
fault injection + failover (M11), and SLO-aware goodput.

## Pointing at a real engine — notes

- **`cached_prefix_tokens`** is read from the engine's streamed `usage`
  (`prompt_tokens_details.cached_tokens`). Launch vLLM with
  `--enable-prefix-caching` (and `stream_options.include_usage`, which the client
  sets) for the M5/M10 cache-hit metrics to be populated.
- **Fault injection** against a real backend can't self-injure: kill the engine
  process (or a gateway backend) out of band at the planned `at_time`; `inject`
  raises `NotImplementedError` to remind you. Against `SimEndpoint` it's applied
  in-process so the failover story is testable offline.
- The `SimEndpoint` cost-model constants (`*_ms_per_tok`, `max_concurrency`) are
  for *shape*, not absolute fidelity — trust the real endpoint for numbers.
