# Module 10 lab — Tool-use / agentic serving

> Make the cache miss, and watch the cost explode. The runnable companion to
> `Module 10 — Tool-Use and Agentic Serving.md`, driving the `common/` harness.

This lab produces the four Lab-10 deliverables. It runs **end-to-end on a laptop
with no GPU** (default `--endpoint sim` uses the harness `SimEndpoint`), and
points at a real engine by changing one flag — nothing else in the lab changes.

**Server (pick one):** a single **24–48 GB** GPU — **RunPod** (RTX 4090 / L40S) or
**Lambda** (A10 / A100), launched with `--enable-prefix-caching`. Full per-lab
provider table: [`../README.md`](../README.md).

## Run it

```bash
pip install -r common/requirements.txt        # httpx, numpy, jsonschema, matplotlib optional
python labs/m10_agentic/run_lab.py              # full dry-run, no GPU (~30s)
python labs/m10_agentic/run_lab.py --exp 2      # just the silent-collapse experiment
```

Artifacts (CSV always; PNG if matplotlib is present) land in `labs/m10_agentic/out/`.

### Against a real engine (the metered run)

```bash
# on the GPU box:
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000 --enable-prefix-caching
# (the harness sets stream_options.include_usage so cached_prefix_tokens is populated)

python labs/m10_agentic/run_lab.py \
    --endpoint http://localhost:8000 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --turns 20 --increment 600 --system-tokens 1500 --tool-latency 2.0
```

The agent loop, tool waits, and re-prefill accounting are identical; only the
numbers change because they now come from real silicon. (For Experiment 1's
genuine cache-OFF curve on a real engine you relaunch without
`--enable-prefix-caching`; on `sim` the flag is toggled in-process, and the lab
falls back to a perturbed prefix as the off-proxy when it can't toggle.)

## The four deliverables

| # | Experiment | Confirms | Output |
|---|---|---|---|
| 1 | **re-prefill scaling** — cache effective vs defeated, swept over turns `T` | §10.1: caching turns cumulative re-prefill from `O(T²)` into `O(T)` | `exp1_reprefill.{csv,png}` |
| 2 | **silent cache collapse** — stable vs perturbed prefix on the *same* cache-on engine | §10.2: a broken prefix silently reverts to `O(T²)` **with no error raised** | `exp2_collapse.{csv,png}` |
| 3 | **tool-latency capacity** — slot duty cycle vs `T_tool`, hold vs offload | §10.3: tool latency gates GPU occupancy; offloading the paused KV frees the slot | `exp3_capacity.{csv,png}` |
| 4 | **end-to-end decomposition** — model / tool / re-prefill fractions | §10.4: the model is usually *not* the bottleneck | `exp4_breakdown.{csv,png}` |

## Self-check (known-good shape, `--endpoint sim`, default seed)

Absolute numbers depend on the `SimEndpoint` cost knobs — what matters is the
**shape**, which should match:

- **Exp 1:** cache-on re-prefill ≈ linear in `T`; cache-off ≈ quadratic; the
  ratio climbs with `T` (≈1.7× at T=2 → ≈13× at T=24).
- **Exp 2:** stable cache-hit ≈ 0.86, perturbed ≈ 0.00; perturbed re-prefills
  ≈ 7× more tokens; **errors raised = 0** (the whole point).
- **Exp 3:** hold duty cycle falls 1.0 → ~0.09 as `T_tool` grows; offload stays
  ~0.8; offload capacity multiplier rises toward ~9× at `T_tool`=0.2s.
- **Exp 4:** tool ≫ model; model(decode) is a small slice (~13%).

## Key flags

`--exp all|1|2|3|4`, `--n-sessions`, `--turns`, `--increment`, `--system-tokens`,
`--tool-latency`, `--concurrency`, `--t-sweep`, `--tool-sweep`, `--seed`, `--out`.
Simulator-only cost knobs (ignored for real endpoints): `--gen-cap`,
`--prefill-ms`, `--decode-ms`.

## How it uses the harness (INVARIANT 7)

- `common.traffic.build_agentic_sessions(..., prefix_stable=…)` — the stable vs
  perturbed sessions.
- `common.bench.run_agentic_session(session, endpoint, kv_policy="hold"|"offload")`
  — the generate → tool → resume loop with per-turn prefill/cache/decode/tool.
- `common.bench.compute_session_metrics(...)` — the **decomposed** model / tool /
  re-prefill breakdown, cache-hit rate, and slot duty cycle — never model
  TTFT/TPOT alone.
