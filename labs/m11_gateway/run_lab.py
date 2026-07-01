#!/usr/bin/env python3
"""Lab 11 — Frameworks, the API layer, and resilience (Module 11).

Drives the gateway (gateway.py) and breaks it on purpose: kills a backend
mid-stream and measures the user-visible failover gap on the TAIL, open-loop.

!!! First real-hardware run — never executed on a GPU here; run it on yours and see how it does. !!!

  # 1) bring up two backends + the gateway:
  GATEWAY_BACKENDS="a=http://localhost:8000,b=http://localhost:8001" \
    uvicorn gateway:app --port 8080      # (run from labs/m11_gateway/)
  # 2) drive it and kill backend 'a' mid-run:
  python labs/m11_gateway/run_lab.py --endpoint http://localhost:8080 --model <id> --kill a
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys

import httpx

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

from common.bench import OpenAIEndpoint, compute_metrics, run_open_loop  # noqa: E402
from common.traffic import build_sharegpt  # noqa: E402


def _csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def _p99(xs):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    import math
    return xs[min(len(xs) - 1, math.ceil(0.99 * len(xs)) - 1)]


async def main_async(args):
    os.makedirs(args.out, exist_ok=True)
    ep = OpenAIEndpoint(args.endpoint, args.model)  # the GATEWAY url

    async def kill_then_revive():
        await asyncio.sleep(args.fault_at)
        async with httpx.AsyncClient() as c:
            await c.post(f"{args.endpoint}/admin/kill", params={"backend": args.kill})
            print(f"    [fault] killed backend '{args.kill}' at t={args.fault_at}s")
            if args.revive_after > 0:
                await asyncio.sleep(args.revive_after)
                await c.post(f"{args.endpoint}/admin/kill", params={"backend": "revive"})

    print("\n[resilience] open-loop stream through the gateway; kill a backend mid-run")
    reqs = build_sharegpt(args.n, rate_or_trace=args.rate, length_profile="short", seed=args.seed)
    killer = asyncio.create_task(kill_then_revive())
    results = await run_open_loop(reqs, ep, timeout=args.timeout)
    await killer

    window = [r for r in results if args.fault_at <= r.intended_send_time <= args.fault_at + args.window]
    outside = [r for r in results if r not in window]
    in_p99 = _p99([r.e2e for r in window])
    out_p99 = _p99([r.e2e for r in outside])
    lost = sum(1 for r in results if not r.ok)
    print(f"    p99 OUTSIDE the failure window = {out_p99*1e3:8.1f} ms")
    print(f"    p99 DURING  the failure window = {in_p99*1e3:8.1f} ms   "
          f"(spike {in_p99/max(out_p99,1e-9):.1f}x)")
    print(f"    in-flight requests lost (errors) = {lost}  of {len(results)}")
    print("    -> failover is NOT seamless: the average hides this; report the tail in the window.")
    _csv(os.path.join(args.out, "failover_gap.csv"),
         ["metric", "value"],
         [("p99_outside_s", round(out_p99, 4)), ("p99_in_window_s", round(in_p99, 4)),
          ("inflight_lost", lost), ("n", len(results))])
    print("\n[load shedding] (manual) drive past the knee with GATEWAY_MAX_INFLIGHT set; compare "
          "admitting-everything (collapse) vs shedding (503s) — shedding protects served-request goodput.")
    print(f"\nDone. Artifacts in {args.out}  (per-request telemetry: gateway's telemetry.jsonl)")


def main():
    p = argparse.ArgumentParser(description="Lab 11 — gateway resilience")
    p.add_argument("--endpoint", default="http://localhost:8080", help="the GATEWAY url")
    p.add_argument("--model", default="")
    p.add_argument("--kill", default="a", help="backend id to kill mid-run")
    p.add_argument("--fault-at", type=float, default=3.0, dest="fault_at")
    p.add_argument("--window", type=float, default=2.0, help="failure-window width (s) for the tail")
    p.add_argument("--revive-after", type=float, default=0.0, dest="revive_after")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--rate", type=float, default=40.0)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
