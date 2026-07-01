#!/usr/bin/env python3
"""Lab 4 — Find the knee, the honest way (Module 4).

Drives the ``common/`` harness, open-loop throughout, to produce the four Lab-4
deliverables:

  1. static vs continuous batching   — the tail explodes under static; continuous holds
  2. the latency–throughput knee      — sweep offered load, mark where p99 explodes
  3. goodput vs load                  — the honest operating point (near but below the knee)
  4. chunked prefill on/off           — a big prefill stalls decode TPOT; chunking flattens it

Runs end-to-end on a laptop with NO GPU (default ``--endpoint sim``). Point the
knee/goodput experiments at a real engine with ``--endpoint http://host:port
--model NAME``; experiments 1 and 4 are simulator-modelled (a real engine is
always continuous, and chunked-prefill is a server launch flag — see README).

    python labs/m04_batching/run_lab.py            # full dry-run, no GPU
    python labs/m04_batching/run_lab.py --exp 2    # just the knee sweep
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from common.bench import (  # noqa: E402
    OpenAIEndpoint,
    Request,
    SimEndpoint,
    SLO,
    StaticBatchEndpoint,
    compute_metrics,
    run_open_loop,
)
from common.traffic import build_sharegpt, poisson_schedule  # noqa: E402

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAVE_MPL = True
except Exception:  # pragma: no cover
    HAVE_MPL = False


def write_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def continuous_endpoint(args):
    if args.endpoint == "sim":
        return SimEndpoint(max_concurrency=args.concurrency, decode_ms_per_tok=args.decode_ms,
                           prefill_ms_per_tok=args.prefill_ms, gen_token_cap=args.gen_cap)
    return OpenAIEndpoint(args.endpoint, args.model)


# --------------------------------------------------------------------------- #
# Experiment 1 — static vs continuous batching (§4.2–4.3)
# --------------------------------------------------------------------------- #
async def exp1_static_vs_continuous(args, outdir):
    print("\n[1] Static vs continuous batching — open-loop, length-variant stream")

    def stream():
        times = poisson_schedule(args.n1, args.rate1, args.seed)
        out = []
        for i, t in enumerate(times):
            long = (i % args.long_every == 0)
            out.append(Request(id=f"r{i}", messages=[{"role": "user", "content": "prompt"}],
                               max_tokens=(args.long_tokens if long else args.short_tokens),
                               intended_send_time=t, meta={"long": long}))
        return out

    cont = continuous_endpoint(args)
    cont_m = compute_metrics(await run_open_loop(stream(), cont), warmup=args.warmup)
    stat = StaticBatchEndpoint(batch_size=args.concurrency, decode_ms_per_tok=args.decode_ms,
                               prefill_ms_per_tok=args.prefill_ms, gen_token_cap=args.gen_cap)
    stat_m = compute_metrics(await run_open_loop(stream(), stat), warmup=args.warmup)
    await stat.aclose()  # stop the batch coordinator (no dangling task)

    rows = [("continuous", cont_m.e2e_p50, cont_m.e2e_p95, cont_m.e2e_p99, cont_m.throughput_req_s),
            ("static", stat_m.e2e_p50, stat_m.e2e_p95, stat_m.e2e_p99, stat_m.throughput_req_s)]
    for name, p50, p95, p99, tp in rows:
        print(f"    {name:>10}: p50={p50*1e3:6.1f}ms  p95={p95*1e3:6.1f}ms  p99={p99*1e3:6.1f}ms  "
              f"throughput={tp:5.0f} req/s")
    print(f"    -> static p99 is {stat_m.e2e_p99/cont_m.e2e_p99:.1f}x continuous's: head-of-line "
          f"blocking + no mid-flight admission. Short requests wait for the batch's longest member.")
    write_csv(os.path.join(outdir, "exp1_static_vs_continuous.csv"),
              ["mode", "e2e_p50_s", "e2e_p95_s", "e2e_p99_s", "throughput_req_s"], rows)
    if HAVE_MPL:
        import numpy as _np
        x = _np.arange(3)
        plt.figure(figsize=(6, 4))
        plt.bar(x - 0.2, [cont_m.e2e_p50 * 1e3, cont_m.e2e_p95 * 1e3, cont_m.e2e_p99 * 1e3],
                width=0.4, label="continuous", color="#2a9d8f")
        plt.bar(x + 0.2, [stat_m.e2e_p50 * 1e3, stat_m.e2e_p95 * 1e3, stat_m.e2e_p99 * 1e3],
                width=0.4, label="static", color="#e76f51")
        plt.xticks(x, ["p50", "p95", "p99"]); plt.ylabel("end-to-end latency (ms)")
        plt.title("Lab 4.1 — static batching's tail explodes; continuous holds")
        plt.legend(); plt.grid(True, axis="y", alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "exp1_static_vs_continuous.png"), dpi=130); plt.close()
    return rows


# --------------------------------------------------------------------------- #
# Experiments 2 & 3 — the knee and goodput (§4.4)
# --------------------------------------------------------------------------- #
async def exp23_knee_and_goodput(args, outdir):
    print("\n[2/3] Latency–throughput knee + goodput — sweep offered load λ (open-loop)")
    rows = []
    for lam in args.lam_sweep:
        reqs = build_sharegpt(args.n2, rate_or_trace=lam, length_profile="short", seed=args.seed)
        m = compute_metrics(await run_open_loop(reqs, continuous_endpoint(args)),
                            slo=SLO(e2e=args.slo_e2e), warmup=args.warmup)
        rows.append((lam, m.throughput_req_s, m.e2e_p99, m.goodput_req_s))
        print(f"    λ={lam:5.0f}/s  throughput={m.throughput_req_s:5.0f}/s  "
              f"p99={m.e2e_p99*1e3:7.1f}ms  goodput={m.goodput_req_s:5.0f}/s")

    # knee ≈ where throughput stops tracking λ; operating point = λ that maximizes goodput
    sat = max(r[1] for r in rows)
    knee = next((r[0] for r in rows if r[2] > args.slo_e2e), rows[-1][0])  # p99 first breaks the SLO
    best = max(rows, key=lambda r: r[3])  # goodput-maximizing operating point
    print(f"    -> throughput saturates near {sat:.0f} req/s; knee ≈ λ={knee:.0f}/s; "
          f"goodput maximized at λ={best[0]:.0f}/s ({best[3]:.0f} good req/s). Run near-but-below the knee.")
    print(f"    -> saturating wall (sim): the {args.concurrency}-way concurrency cap (the KV-capacity "
          f"analogue). On a real engine, name it: compute (M1) / KV-bandwidth (M2) / KV-capacity (M2/5).")
    write_csv(os.path.join(outdir, "exp23_knee_goodput.csv"),
              ["offered_lambda", "throughput_req_s", "e2e_p99_s", "goodput_req_s"], rows)
    if HAVE_MPL:
        lam = [r[0] for r in rows]
        fig, ax1 = plt.subplots(figsize=(6.5, 4))
        ax1.plot(lam, [r[1] for r in rows], "o-", color="#264653", label="throughput")
        ax1.plot(lam, [r[3] for r in rows], "^--", color="#2a9d8f", label="goodput (SLO)")
        ax1.set_xlabel("offered load λ (req/s)"); ax1.set_ylabel("req/s")
        ax2 = ax1.twinx()
        ax2.plot(lam, [r[2] * 1e3 for r in rows], "s-", color="#e76f51", label="p99 latency")
        ax2.set_ylabel("p99 latency (ms)")
        ax1.axvline(best[0], ls=":", color="gray", label="_nolegend_")
        ax1.text(best[0], ax1.get_ylim()[1] * 0.02, " λ*", color="gray")
        ax1.set_title("Lab 4.2/4.3 — the latency–throughput knee", fontsize=11)
        lines = [l for l in ax1.get_lines() + ax2.get_lines() if not l.get_label().startswith("_")]
        ax1.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=8)
        ax1.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "exp23_knee_goodput.png"), dpi=130); plt.close()
    return rows


# --------------------------------------------------------------------------- #
# Experiment 4 — chunked prefill flattens decode TPOT (§4.5)
# --------------------------------------------------------------------------- #
async def exp4_chunked_prefill(args, outdir):
    print("\n[4] Chunked prefill — a big prefill stalls in-flight decode; chunking flattens it (sim)")
    if args.endpoint != "sim":
        print("    (modelled in the simulator; on a real engine relaunch vLLM with/without "
              "--enable-chunked-prefill and compare decode TPOT.)")

    def workload():
        decoders = [Request(id=f"d{i}", messages=[{"role": "user", "content": "short prompt"}],
                           max_tokens=args.decoder_tokens, intended_send_time=0.0, meta={"role": "decoder"})
                    for i in range(args.n_decoders)]
        bigs = [Request(id=f"big{j}", messages=[{"role": "user", "content": "x " * args.big_prompt_tokens}],
                        max_tokens=2, intended_send_time=0.005 + 0.04 * j, meta={"role": "prefill"})
                for j in range(args.n_bigs)]
        return decoders + bigs

    rows = []
    for chunked in (False, True):
        ep = SimEndpoint(max_concurrency=args.n_decoders + args.n_bigs + 2, decode_ms_per_tok=args.decode_ms,
                         prefill_ms_per_tok=0.1, base_ms=0.5, model_prefill_interference=True,
                         chunked_prefill=chunked, prefill_chunk_tokens=args.chunk_tokens, enable_prefix_cache=False)
        res = await run_open_loop(workload(), ep)
        decoders = [r for r in res if r.meta.get("role") == "decoder"]
        m = compute_metrics(decoders)
        rows.append(("on" if chunked else "off", m.tpot_p50, m.tpot_p99))
        print(f"    chunked prefill {('on ' if chunked else 'off')}: decode TPOT "
              f"p50={m.tpot_p50*1e3:6.2f}ms  p99={m.tpot_p99*1e3:6.2f}ms")
    print(f"    -> chunked prefill cuts decode TPOT p99 by {rows[0][2]/max(rows[1][2],1e-9):.1f}x: "
          f"the giant prefill no longer starves in-flight decodes. (Motivates disaggregation, Module 8.)")
    write_csv(os.path.join(outdir, "exp4_chunked_prefill.csv"),
              ["chunked_prefill", "decode_tpot_p50_s", "decode_tpot_p99_s"], rows)
    if HAVE_MPL:
        x = np.arange(2)
        plt.figure(figsize=(5.5, 4))
        plt.bar(x - 0.2, [rows[0][1] * 1e3, rows[1][1] * 1e3], width=0.4, label="TPOT p50", color="#264653")
        plt.bar(x + 0.2, [rows[0][2] * 1e3, rows[1][2] * 1e3], width=0.4, label="TPOT p99", color="#e76f51")
        plt.xticks(x, ["chunked off", "chunked on"]); plt.ylabel("decode TPOT (ms)")
        plt.title("Lab 4.4 — chunked prefill flattens the decode-TPOT spike")
        plt.legend(); plt.grid(True, axis="y", alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "exp4_chunked_prefill.png"), dpi=130); plt.close()
    return rows


def parse_args():
    p = argparse.ArgumentParser(description="Lab 4 — find the knee, on the common/ harness")
    p.add_argument("--endpoint", default="sim", help="'sim' (no GPU) or a base URL")
    p.add_argument("--model", default="", help="model name for a real endpoint")
    p.add_argument("--exp", default="all", help="which experiment: all|1|2|3|4 (2 and 3 share a sweep)")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--decode-ms", type=float, default=1.0, dest="decode_ms")
    p.add_argument("--prefill-ms", type=float, default=0.05, dest="prefill_ms")
    p.add_argument("--gen-cap", type=int, default=32, dest="gen_cap")
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    # exp1
    p.add_argument("--n1", type=int, default=80)
    p.add_argument("--rate1", type=float, default=120.0)
    p.add_argument("--short-tokens", type=int, default=8, dest="short_tokens")
    p.add_argument("--long-tokens", type=int, default=96, dest="long_tokens")
    p.add_argument("--long-every", type=int, default=6, dest="long_every")
    # exp2/3
    p.add_argument("--n2", type=int, default=200)
    p.add_argument("--lam-sweep", default="50,100,150,200,250,300,400,550", dest="lam_sweep")
    p.add_argument("--slo-e2e", type=float, default=0.12, dest="slo_e2e")
    # exp4
    p.add_argument("--n-decoders", type=int, default=8, dest="n_decoders")
    p.add_argument("--decoder-tokens", type=int, default=24, dest="decoder_tokens")
    p.add_argument("--n-bigs", type=int, default=1, dest="n_bigs")
    p.add_argument("--big-prompt-tokens", type=int, default=1000, dest="big_prompt_tokens")
    p.add_argument("--chunk-tokens", type=int, default=256, dest="chunk_tokens")
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    a = p.parse_args()
    a.lam_sweep = [float(x) for x in a.lam_sweep.split(",") if x]
    return a


async def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    where = "SimEndpoint (no GPU)" if args.endpoint == "sim" else f"{args.endpoint} [{args.model}]"
    print(f"Lab 4 — find the knee | endpoint: {where} | out: {args.out}")
    if not HAVE_MPL:
        print("  (matplotlib not found — writing CSV + tables only, no PNGs)")

    todo = ["1", "2", "4"] if args.exp == "all" else [args.exp]
    if "1" in todo:
        await exp1_static_vs_continuous(args, args.out)
    if "2" in todo or "3" in todo:
        await exp23_knee_and_goodput(args, args.out)
    if "4" in todo:
        await exp4_chunked_prefill(args, args.out)

    print(f"\nDone. Artifacts (CSV{' + PNG' if HAVE_MPL else ''}) in {args.out}")
    print("Deliverables: exp1 static-vs-continuous tail | exp2/3 knee + goodput | exp4 chunked-prefill TPOT")


if __name__ == "__main__":
    asyncio.run(main())
