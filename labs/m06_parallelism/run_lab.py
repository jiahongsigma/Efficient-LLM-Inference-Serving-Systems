#!/usr/bin/env python3
"""Lab 6 — Multi-GPU parallelism (Module 6).

!!! First real-hardware run — never executed on a GPU here, so run it on yours and see how it does across environments. The all-reduce time itself needs Nsight/NCCL;
    here we measure the SCALING curve (throughput + single-request latency) and the
    gap from ideal-linear, which IS the communication cost. Run once per TP degree. !!!

  vllm serve <70B> --port 8000 --tensor-parallel-size 2
  python labs/m06_parallelism/run_lab.py --endpoint http://localhost:8000 --model <id> --tp 2
  vllm serve <70B> --port 8000 --tensor-parallel-size 4
  python labs/m06_parallelism/run_lab.py --endpoint http://localhost:8000 --model <id> --tp 4
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

from common.bench import OpenAIEndpoint, Request, compute_metrics, run_open_loop  # noqa: E402
from common.traffic import build_sharegpt  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


def _upsert(path, header, row):
    rows = {}
    if os.path.exists(path):
        with open(path) as f:
            for x in list(csv.reader(f))[1:]:
                if x:
                    rows[x[0]] = x
    rows[str(row[0])] = [str(c) for c in row]
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows.values())
    return sorted(rows.values(), key=lambda r: float(r[0]))


async def run(args):
    os.makedirs(args.out, exist_ok=True)
    ep = OpenAIEndpoint(args.endpoint, args.model)
    print(f"\n[TP={args.tp}] throughput + single-request latency")
    m = compute_metrics(await run_open_loop(
        build_sharegpt(args.n, rate_or_trace=args.rate, length_profile="short", seed=args.seed),
        ep, timeout=args.timeout), warmup=args.warmup)
    single = (await run_open_loop([Request(id="s", messages=[{"role": "user", "content": "Write an essay."}],
                                           max_tokens=256, intended_send_time=0.0)], ep, timeout=args.timeout))[0]
    print(f"    throughput={m.throughput_tok_s:8.0f} tok/s   single-request latency={single.e2e*1e3:7.1f} ms")

    rows = _upsert(os.path.join(args.out, "scaling.csv"),
                   ["tp", "throughput_tok_s", "single_latency_s"],
                   (args.tp, round(m.throughput_tok_s, 1), round(single.e2e, 4)))

    if HAVE_MPL and len(rows) >= 2:
        tps = [float(r[0]) for r in rows]
        tput = [float(r[1]) for r in rows]
        base_tp, base = tps[0], tput[0]
        ideal = [base * (t / base_tp) for t in tps]
        plt.figure(figsize=(6, 4))
        plt.plot(tps, tput, "o-", label="measured throughput")
        plt.plot(tps, ideal, "k--", label="ideal linear")
        plt.xlabel("tensor-parallel degree"); plt.ylabel("throughput (tok/s)")
        plt.title("Lab 6 — scaling is sub-linear; the gap is the communication cost")
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(args.out, "scaling.png"), dpi=130); plt.close()
    print(f"\nDone (TP={args.tp}). Re-run at another TP degree; the gap from ideal-linear in "
          f"{args.out}/scaling.png is the comm cost. Name the binding wall: compute vs interconnect.")


def main():
    p = argparse.ArgumentParser(description="Lab 6 — parallelism")
    p.add_argument("--endpoint", default="http://localhost:8000")
    p.add_argument("--model", default="")
    p.add_argument("--tp", type=int, required=True, help="tensor-parallel degree of this server")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--rate", type=float, default=20.0)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
