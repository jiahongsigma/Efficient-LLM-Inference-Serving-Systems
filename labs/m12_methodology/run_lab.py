#!/usr/bin/env python3
"""Lab 12 — How to benchmark without lying to yourself (Module 12).

!!! First real-hardware run — never executed on a GPU here, but fully harness-supported
    (the common/ pieces it calls are tested). See how it does on your setup. !!!

  python labs/m12_methodology/run_lab.py --endpoint http://localhost:8000 --model <id>
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import statistics
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

from common.bench import (  # noqa: E402
    SLO,
    OpenAIEndpoint,
    aggregate_runs,
    compute_metrics,
    determinism_check,
    run_closed_loop,
    run_open_loop,
)
from common.traffic import build_sharegpt  # noqa: E402


def _csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


async def coordinated_omission(args, ep, outdir):
    """Closed-loop self-throttles and reports a falsely-good tail; open-loop with a
    fixed schedule exposes the true tail (§12.4)."""
    print("\n[1] Coordinated omission — closed-loop vs open-loop tail")
    openm = compute_metrics(
        await run_open_loop(build_sharegpt(args.n, rate_or_trace=args.rate, length_profile="short", seed=args.seed),
                            ep, timeout=args.timeout), warmup=args.warmup)
    closedm = compute_metrics(
        await run_closed_loop(build_sharegpt(args.n, rate_or_trace=args.rate, length_profile="short", seed=args.seed),
                              ep, concurrency=args.concurrency), warmup=args.warmup)
    print(f"    open-loop   p99 = {openm.e2e_p99*1e3:8.1f} ms")
    print(f"    closed-loop p99 = {closedm.e2e_p99*1e3:8.1f} ms   "
          f"(optimistic by {openm.e2e_p99/max(closedm.e2e_p99,1e-9):.1f}x — the comfortable lie)")
    _csv(os.path.join(outdir, "coordinated_omission.csv"),
         ["loop", "e2e_p50_s", "e2e_p99_s"],
         [("open", round(openm.e2e_p50, 4), round(openm.e2e_p99, 4)),
          ("closed", round(closedm.e2e_p50, 4), round(closedm.e2e_p99, 4))])


async def determinism(args, ep, outdir):
    """Temp-0 batched inference is not reproducible unless kernels are batch-invariant (§12.5)."""
    print("\n[2] Determinism at temperature 0 (vary batch composition)")
    rep = await determinism_check(ep, args.prompt, n_runs=args.det_runs, vary_batch=True, timeout=args.timeout)
    print(f"    identical_fraction = {rep.identical_fraction:.2f}  "
          f"({'reproducible' if rep.identical_fraction == 1.0 else 'NOT reproducible — batch composition changed the output'})")
    _csv(os.path.join(outdir, "determinism.csv"),
         ["n_runs", "identical_fraction"], [(args.det_runs, round(rep.identical_fraction, 3))])


async def variance(args, ep, outdir):
    """A single run of a single number is not a measurement (§12.5)."""
    print("\n[3] Variance / CI over repeats")
    runs = []
    for s in range(args.repeats):
        reqs = build_sharegpt(args.n, rate_or_trace=args.rate, length_profile="short", seed=s)
        runs.append(compute_metrics(await run_open_loop(reqs, ep, slo=SLO(e2e=args.slo_e2e), timeout=args.timeout),
                                    warmup=args.warmup))
    agg = aggregate_runs(runs)
    print(f"    e2e_p99 = {agg.mean['e2e_p99']*1e3:.1f} ± {agg.ci95['e2e_p99']*1e3:.1f} ms (95% CI, n={agg.n_runs})")
    print(f"    goodput = {agg.mean['goodput_req_s']:.1f} ± {agg.ci95['goodput_req_s']:.1f} req/s")
    _csv(os.path.join(outdir, "variance.csv"),
         ["metric", "mean", "ci95"],
         [("e2e_p99_s", agg.mean["e2e_p99"], agg.ci95["e2e_p99"]),
          ("goodput_req_s", agg.mean["goodput_req_s"], agg.ci95["goodput_req_s"])])


async def main_async(args):
    os.makedirs(args.out, exist_ok=True)
    ep = OpenAIEndpoint(args.endpoint, args.model)
    await coordinated_omission(args, ep, args.out)
    await determinism(args, ep, args.out)
    await variance(args, ep, args.out)
    print("\n[4] Win-then-lose (manual): take one earlier conclusion and build two defensible "
          "benchmarks — one where it wins, one where it loses — by changing only traffic / metric / batch.")
    print(f"\nDone. Artifacts in {args.out}")


def main():
    p = argparse.ArgumentParser(description="Lab 12 — methodology")
    p.add_argument("--endpoint", default="http://localhost:8000")
    p.add_argument("--model", default="")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--rate", type=float, default=20.0)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--det-runs", type=int, default=20, dest="det_runs")
    p.add_argument("--prompt", default="List three prime numbers between 10 and 30.")
    p.add_argument("--slo-e2e", type=float, default=5.0, dest="slo_e2e")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
