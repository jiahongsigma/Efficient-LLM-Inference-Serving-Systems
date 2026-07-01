#!/usr/bin/env python3
"""Lab 5 — Serving-time memory management (Module 5).

!!! First real-hardware run — never executed on a GPU here, so run it on yours and see how it does across environments. The common/ harness API it calls IS tested,
    and this lab is fully harness-supported, so expect few surprises. !!!

Prefix caching is a launch flag, so the cleanest path is to run this once per
server state and let it accumulate:

  # server A: vllm serve <model> --enable-prefix-caching
  python labs/m05_paging/run_lab.py --endpoint http://localhost:8000 --model <id> --label cache_on
  # server B (relaunch): vllm serve <model> --no-enable-prefix-caching
  python labs/m05_paging/run_lab.py --endpoint http://localhost:8000 --model <id> --label cache_off

Each run sweeps the prefix-sharing ratio on long-doc-QA AND runs the ShareGPT null
control (INVARIANT 4), appending to the same CSVs.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

from common.bench import OpenAIEndpoint, compute_metrics, run_open_loop  # noqa: E402
from common.traffic import build_long_doc_qa, build_sharegpt  # noqa: E402

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
            data = list(csv.reader(f))[1:]
        for x in data:
            if x:
                rows[tuple(x[:2])] = x
    rows[(str(row[0]), str(row[1]))] = [str(c) for c in row]
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows.values())


async def run(args, outdir):
    ep = OpenAIEndpoint(args.endpoint, args.model)
    print(f"\n[prefix sharing] label={args.label} — sweep share ratio on long-doc-QA")
    for ratio in args.share_sweep:
        reqs = build_long_doc_qa(args.n, doc_tokens=args.doc_tokens, prefix_share_ratio=ratio,
                                 rate_or_trace=args.rate, seed=args.seed)
        m = compute_metrics(await run_open_loop(reqs, ep, timeout=args.timeout), warmup=args.warmup)
        _upsert(os.path.join(outdir, "prefix_sweep.csv"),
                ["label", "share_ratio", "cache_hit_rate", "ttft_p50_s", "throughput_req_s"],
                (args.label, ratio, round(m.cache_hit_rate, 3), round(m.ttft_p50, 4), round(m.throughput_req_s, 1)))
        print(f"    share={ratio:.2f}  cache_hit={m.cache_hit_rate:.2f}  "
              f"ttft_p50={m.ttft_p50*1e3:6.1f}ms  tput={m.throughput_req_s:5.1f} req/s")

    print(f"[null control] label={args.label} — ShareGPT (no shared prefix; expect cache_hit≈0)")
    sg = build_sharegpt(args.n, rate_or_trace=args.rate, length_profile="short", seed=args.seed)
    m = compute_metrics(await run_open_loop(sg, ep, timeout=args.timeout), warmup=args.warmup)
    _upsert(os.path.join(outdir, "sharegpt_null.csv"),
            ["label", "workload", "cache_hit_rate", "throughput_req_s"],
            (args.label, "sharegpt", round(m.cache_hit_rate, 3), round(m.throughput_req_s, 1)))
    print(f"    ShareGPT cache_hit={m.cache_hit_rate:.2f}  tput={m.throughput_req_s:5.1f} req/s "
          f"-> measuring prefix caching here would falsely declare it useless (INVARIANT 4)")

    _plot(outdir)


def _plot(outdir):
    path = os.path.join(outdir, "prefix_sweep.csv")
    if not (HAVE_MPL and os.path.exists(path)):
        return
    import collections
    series = collections.defaultdict(list)
    with open(path) as f:
        for r in list(csv.reader(f))[1:]:
            series[r[0]].append((float(r[1]), float(r[2])))
    plt.figure(figsize=(6, 4))
    for label, pts in series.items():
        pts.sort()
        plt.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=label)
    plt.xlabel("prefix-sharing ratio"); plt.ylabel("cache-hit rate"); plt.legend()
    plt.title("Lab 5 — prefix-cache gain scales with sharing (only on long-doc-QA)")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "prefix_cache.png"), dpi=130); plt.close()


def main():
    p = argparse.ArgumentParser(description="Lab 5 — paging / prefix caching")
    p.add_argument("--endpoint", default="http://localhost:8000")
    p.add_argument("--model", default="")
    p.add_argument("--label", default="cache_on", help="server state label (cache_on / cache_off)")
    p.add_argument("--share-sweep", default="0,0.25,0.5,0.75,1.0", dest="share_sweep")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--doc-tokens", type=int, default=2000, dest="doc_tokens")
    p.add_argument("--rate", type=float, default=10.0)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    args = p.parse_args()
    args.share_sweep = [float(x) for x in args.share_sweep.split(",") if x != ""]
    os.makedirs(args.out, exist_ok=True)
    asyncio.run(run(args, args.out))
    print(f"\nDone. Artifacts in {args.out}")


if __name__ == "__main__":
    main()
