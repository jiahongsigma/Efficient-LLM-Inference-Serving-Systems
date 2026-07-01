#!/usr/bin/env python3
"""Lab 7 — Long-context serving (Module 7).

!!! First real-hardware run — never executed on a GPU here, so run it on yours and see how it does across environments. Eviction / KV-quant are launch-time configs,
    so run once per server state. Scores NEEDLE RETRIEVAL by depth, not perplexity
    (INVARIANT 5). !!!

  vllm serve <model> --port 8000 --max-model-len 131072                    # full
  python labs/m07_longcontext/run_lab.py --endpoint http://localhost:8000 --model <id> --label full --ctx 128000
  vllm serve <model> --port 8000 --max-model-len 131072 --kv-cache-dtype fp8   # KV-quant
  python labs/m07_longcontext/run_lab.py --endpoint http://localhost:8000 --model <id> --label kvquant_fp8 --ctx 128000
  # (eviction backends: StreamingLLM / H2O forks — label eviction)
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

from common.bench import OpenAIEndpoint, run_open_loop  # noqa: E402
from common.eval import score_needle  # noqa: E402
from common.traffic import build_needle  # noqa: E402


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


async def run(args):
    os.makedirs(args.out, exist_ok=True)
    ep = OpenAIEndpoint(args.endpoint, args.model)
    print(f"\n[{args.label}] needle-in-haystack retrieval by depth at ctx={args.ctx}")

    # plant the needle at several depths (incl. deep, where eviction drops it)
    all_res = []
    for depth in args.depths:
        rs = build_needle(args.n_per_depth, context_len=args.ctx, needle_depth_fraction=depth, seed=args.seed)
        all_res += await run_open_loop(rs, ep, timeout=args.timeout)
    score = score_needle(all_res)

    print(f"    overall retrieval = {score.retrieval_rate:.2f}")
    for bucket, rate in score.by_depth.items():
        print(f"      depth {bucket}: {rate:.2f}")
    print("    -> eviction often looks ~free on average but FAILS deep needles; KV-quant keeps all "
          "tokens and survives (§7.7). Report retrieval, not perplexity.")
    header = ["label", "ctx", "overall"] + [f"depth_{b}" for b in sorted(score.by_depth)]
    row = [args.label, args.ctx, round(score.retrieval_rate, 3)] + [round(score.by_depth[b], 3) for b in sorted(score.by_depth)]
    _upsert(os.path.join(args.out, "needle_by_depth.csv"), header, row)
    print(f"\nDone ({args.label}). Re-run per server config; rows accumulate in "
          f"{args.out}/needle_by_depth.csv — the accuracy×context×memory table.")


def main():
    p = argparse.ArgumentParser(description="Lab 7 — long context")
    p.add_argument("--endpoint", default="http://localhost:8000")
    p.add_argument("--model", default="")
    p.add_argument("--label", required=True, help="server config: full / eviction / kvquant_fp8 / ...")
    p.add_argument("--ctx", type=int, default=128000)
    p.add_argument("--depths", default="0.1,0.3,0.5,0.7,0.9")
    p.add_argument("--n-per-depth", type=int, default=20, dest="n_per_depth")
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    args = p.parse_args()
    args.depths = [float(x) for x in args.depths.split(",") if x != ""]
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
