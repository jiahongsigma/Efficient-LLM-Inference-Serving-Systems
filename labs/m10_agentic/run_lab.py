#!/usr/bin/env python3
"""Lab 10 — Make the cache miss, and watch the cost explode (Module 10).

Drives the ``common/`` harness to produce the four Lab-10 deliverables:

  1. re-prefill curve              — cache effective O(T) vs defeated O(T²)
  2. silent cache collapse         — stable vs perturbed prefix, NO error raised
  3. tool-latency capacity curve   — slot duty cycle vs T_tool, hold vs offload
  4. end-to-end decomposition      — model / tool / re-prefill fractions

Runs end-to-end on your laptop with NO GPU (the default ``--endpoint sim`` uses
the harness's in-process ``SimEndpoint``). Point it at a real engine with
``--endpoint http://host:port --model NAME`` and nothing else changes.

    python labs/m10_agentic/run_lab.py                 # full dry-run, no GPU
    python labs/m10_agentic/run_lab.py --exp 2         # one experiment
    python labs/m10_agentic/run_lab.py \\
        --endpoint http://localhost:8000 \\
        --model meta-llama/Llama-3.1-8B-Instruct      # real engine
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import statistics
import sys

# make ``common`` importable however this script is launched
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from common.bench import (  # noqa: E402
    OpenAIEndpoint,
    SimEndpoint,
    compute_session_metrics,
    run_agentic_session,
)
from common.traffic import build_agentic_sessions  # noqa: E402

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAVE_MPL = True
except Exception:  # pragma: no cover
    HAVE_MPL = False


# --------------------------------------------------------------------------- #
# Endpoint + session helpers
# --------------------------------------------------------------------------- #
def make_endpoint(args, *, cache: bool = True):
    """A fresh endpoint. For the simulator, ``cache`` toggles the prefix cache
    (the genuine engine-cache off/on of Experiment 1). A real engine's cache is
    a launch flag (``--enable-prefix-caching``); see the README."""
    if args.endpoint == "sim":
        return SimEndpoint(
            max_concurrency=args.concurrency,
            gen_token_cap=args.gen_cap,
            prefill_ms_per_tok=args.prefill_ms,
            decode_ms_per_tok=args.decode_ms,
            enable_prefix_cache=cache,
        )
    return OpenAIEndpoint(args.endpoint, args.model)


def sessions(args, *, turns, prefix_stable, tool_latency, n=None, seed=0):
    return build_agentic_sessions(
        n if n is not None else args.n_sessions,
        turns=turns,
        increment_tokens=args.increment,
        system_tokens=args.system_tokens,
        tool_latency=tool_latency,
        tool_latency_jitter=args.tool_jitter,
        prefix_stable=prefix_stable,
        seed=seed,
    )


async def run_all(endpoint, sess, kv_policy):
    return await asyncio.gather(*[run_agentic_session(s, endpoint, kv_policy=kv_policy) for s in sess])


def write_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# Experiment 1 — the re-prefill lever: O(T) vs O(T²)  (§10.1)
# --------------------------------------------------------------------------- #
async def exp1_reprefill(args, outdir):
    print("\n[1] Re-prefill scaling — cache effective (O(T)) vs defeated (O(T²))")
    is_sim = args.endpoint == "sim"
    rows = []
    for T in args.t_sweep:
        # cache effective: stable prefix on a cache-on endpoint
        sr_on = await run_all(make_endpoint(args, cache=True),
                              sessions(args, turns=T, prefix_stable=True, tool_latency=0.0), "hold")
        on = statistics.mean(s.total_reprefill_tokens for s in sr_on)

        # cache defeated/off: the simulator truly disables the cache; a real
        # engine cannot toggle per-call, so we defeat it with a perturbed prefix
        if is_sim:
            ep_off = make_endpoint(args, cache=False)
            sess_off = sessions(args, turns=T, prefix_stable=True, tool_latency=0.0)
        else:
            ep_off = make_endpoint(args, cache=True)
            sess_off = sessions(args, turns=T, prefix_stable=False, tool_latency=0.0)
        sr_off = await run_all(ep_off, sess_off, "hold")
        off = statistics.mean(s.total_reprefill_tokens for s in sr_off)

        rows.append((T, round(on, 1), round(off, 1), round(off / on, 2) if on else float("nan")))
        print(f"    T={T:>3}  cache-on={on:>9.0f}  cache-off={off:>9.0f}  ratio={rows[-1][3]}x")

    write_csv(os.path.join(outdir, "exp1_reprefill.csv"),
              ["turns", "reprefill_cache_on", "reprefill_cache_off", "ratio"], rows)
    if HAVE_MPL:
        T = [r[0] for r in rows]
        plt.figure(figsize=(6, 4))
        plt.plot(T, [r[1] for r in rows], "o-", label="cache effective ≈ O(T)")
        plt.plot(T, [r[2] for r in rows], "s-", label="cache defeated ≈ O(T²)")
        plt.xlabel("turns T"); plt.ylabel("cumulative re-prefill tokens / session")
        plt.title("Lab 10.1 — re-prefill: cache turns O(T²) into O(T)")
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "exp1_reprefill.png"), dpi=130); plt.close()
    return rows


# --------------------------------------------------------------------------- #
# Experiment 2 — the silent collapse: stable vs perturbed prefix  (§10.2)
# --------------------------------------------------------------------------- #
async def exp2_silent_collapse(args, outdir):
    print("\n[2] Silent cache collapse — stable vs perturbed prefix (same cache-on engine)")
    rows = []
    detail = {}
    for label, stable in (("stable", True), ("perturbed", False)):
        ep = make_endpoint(args, cache=True)
        sr = await run_all(ep, sessions(args, turns=args.turns, prefix_stable=stable,
                                        tool_latency=args.tool_latency), "hold")
        sm = compute_session_metrics(sr)
        n_turns = sum(len(s.turns) for s in sr)
        n_errors = sum(1 for s in sr for t in s.turns if t.decode_tokens == 0)
        rows.append((label, round(sm.cache_hit_rate, 3), sm.reprefill_tokens_total,
                     round(statistics.mean(s.e2e_latency for s in sr), 4), n_turns, n_errors))
        detail[label] = sm
        print(f"    {label:>9}: cache_hit={sm.cache_hit_rate:.2f}  "
              f"reprefill_tokens={sm.reprefill_tokens_total:>8}  errors_raised={n_errors}")

    blow = rows[1][2] / rows[0][2] if rows[0][2] else float("nan")
    print(f"    -> perturbed re-prefills {blow:.1f}x more tokens, with {rows[1][5]} errors raised "
          f"(the failure is silent — visible only on the bill).")
    write_csv(os.path.join(outdir, "exp2_collapse.csv"),
              ["prefix", "cache_hit_rate", "reprefill_tokens", "e2e_mean_s", "turns", "errors"], rows)
    if HAVE_MPL:
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        labels = [r[0] for r in rows]
        ax[0].bar(labels, [r[1] for r in rows], color=["#2a9d8f", "#e76f51"])
        ax[0].set_title("cache-hit rate"); ax[0].set_ylim(0, 1)
        ax[1].bar(labels, [r[2] for r in rows], color=["#2a9d8f", "#e76f51"])
        ax[1].set_title("re-prefill tokens (silent blow-up)")
        fig.suptitle("Lab 10.2 — break the prefix, the cost explodes with no error")
        plt.tight_layout(); plt.savefig(os.path.join(outdir, "exp2_collapse.png"), dpi=130); plt.close()
    return rows


# --------------------------------------------------------------------------- #
# Experiment 3 — tool-latency capacity: slot duty cycle vs T_tool  (§10.3)
# --------------------------------------------------------------------------- #
async def exp3_tool_capacity(args, outdir):
    print("\n[3] Tool-latency capacity — slot duty cycle, hold vs offload (concurrent sessions)")
    rows = []
    for ttool in args.tool_sweep:
        point = {"tool": ttool}
        for policy in ("hold", "offload"):
            ep = make_endpoint(args, cache=True)
            sess = sessions(args, turns=args.cap_turns, prefix_stable=True,
                            tool_latency=ttool, n=args.concurrency)
            sr = await run_all(ep, sess, policy)
            sm = compute_session_metrics(sr)
            point[policy] = sm.slot_utilization
            point[f"{policy}_slot_held"] = statistics.mean(s.slot_held_time for s in sr)
        cap_mult = point["hold_slot_held"] / point["offload_slot_held"] if point["offload_slot_held"] else float("nan")
        rows.append((ttool, round(point["hold"], 3), round(point["offload"], 3), round(cap_mult, 2)))
        print(f"    T_tool={ttool:>5}s  duty(hold)={point['hold']:.2f}  "
              f"duty(offload)={point['offload']:.2f}  offload capacity≈{cap_mult:.1f}x more sessions/slot")

    write_csv(os.path.join(outdir, "exp3_capacity.csv"),
              ["tool_latency_s", "duty_hold", "duty_offload", "offload_capacity_mult"], rows)
    if HAVE_MPL:
        x = [r[0] for r in rows]
        plt.figure(figsize=(6, 4))
        plt.plot(x, [r[1] for r in rows], "o-", label="hold KV (duty ≈ T_gen/(T_gen+T_tool))")
        plt.plot(x, [r[2] for r in rows], "s-", label="offload KV during tool")
        plt.xlabel("tool latency T_tool (s)"); plt.ylabel("slot duty cycle (GPU-busy / slot-held)")
        plt.title("Lab 10.3 — tool latency gates the GPU; offload frees the slot")
        plt.ylim(0, 1.05); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "exp3_capacity.png"), dpi=130); plt.close()
    return rows


# --------------------------------------------------------------------------- #
# Experiment 4 — end-to-end decomposition: model / tool / re-prefill  (§10.4)
# --------------------------------------------------------------------------- #
async def exp4_decomposition(args, outdir):
    print("\n[4] End-to-end decomposition — the model is usually NOT the bottleneck")
    ep = make_endpoint(args, cache=True)
    sr = await run_all(ep, sessions(args, turns=args.turns, prefix_stable=True,
                                    tool_latency=args.tool_latency), "hold")
    sm = compute_session_metrics(sr)
    bd = sm.latency_breakdown
    rows = [("model(decode)", round(bd["model"], 3)),
            ("tool", round(bd["tool"], 3)),
            ("reprefill", round(bd["reprefill"], 3))]
    for k, v in rows:
        print(f"    {k:>14}: {v*100:5.1f}%")
    print(f"    -> halving model TPOT moves session latency by at most ~{bd['model']*100:.0f}%.")
    write_csv(os.path.join(outdir, "exp4_breakdown.csv"), ["component", "fraction"], rows)
    if HAVE_MPL:
        plt.figure(figsize=(5, 5))
        plt.pie([r[1] for r in rows], labels=[r[0] for r in rows], autopct="%1.0f%%",
                colors=["#264653", "#e76f51", "#e9c46a"])
        plt.title("Lab 10.4 — where an agent's latency actually goes")
        plt.tight_layout(); plt.savefig(os.path.join(outdir, "exp4_breakdown.png"), dpi=130); plt.close()
    return rows


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Lab 10 — agentic serving on the common/ harness")
    p.add_argument("--endpoint", default="sim", help="'sim' (no GPU) or a base URL")
    p.add_argument("--model", default="", help="model name for a real endpoint")
    p.add_argument("--exp", default="all", help="which experiment: all|1|2|3|4")
    p.add_argument("--n-sessions", type=int, default=6, dest="n_sessions")
    p.add_argument("--turns", type=int, default=12)
    p.add_argument("--cap-turns", type=int, default=6, dest="cap_turns")
    p.add_argument("--increment", type=int, default=120)
    p.add_argument("--system-tokens", type=int, default=300, dest="system_tokens")
    p.add_argument("--tool-latency", type=float, default=0.05, dest="tool_latency")
    p.add_argument("--tool-jitter", type=float, default=0.0, dest="tool_jitter")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--t-sweep", default="2,4,8,16,24", dest="t_sweep")
    p.add_argument("--tool-sweep", default="0,0.02,0.05,0.1,0.2", dest="tool_sweep")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    # SimEndpoint cost knobs (ignored for real endpoints)
    p.add_argument("--gen-cap", type=int, default=8, dest="gen_cap")
    p.add_argument("--prefill-ms", type=float, default=0.05, dest="prefill_ms")
    p.add_argument("--decode-ms", type=float, default=0.5, dest="decode_ms")
    a = p.parse_args()
    a.t_sweep = [int(x) for x in a.t_sweep.split(",") if x]
    a.tool_sweep = [float(x) for x in a.tool_sweep.split(",") if x != ""]
    return a


async def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    where = "SimEndpoint (no GPU)" if args.endpoint == "sim" else f"{args.endpoint} [{args.model}]"
    print(f"Lab 10 — agentic serving | endpoint: {where} | out: {args.out}")
    if not HAVE_MPL:
        print("  (matplotlib not found — writing CSV + tables only, no PNGs)")

    exps = {"1": exp1_reprefill, "2": exp2_silent_collapse,
            "3": exp3_tool_capacity, "4": exp4_decomposition}
    todo = list(exps) if args.exp == "all" else [args.exp]
    for k in todo:
        await exps[k](args, args.out)

    print(f"\nDone. Artifacts (CSV{' + PNG' if HAVE_MPL else ''}) in {args.out}")
    print("Deliverables: exp1 re-prefill curve | exp2 silent collapse | "
          "exp3 tool-latency capacity | exp4 latency decomposition")


if __name__ == "__main__":
    asyncio.run(main())
