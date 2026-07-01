#!/usr/bin/env python3
"""Lab 8 — Attacking the sequential dependency (Module 8).

!!! First real-hardware run — never executed on a GPU here, so run it on yours and see how it does across environments. Acceptance rate α is read from the engine's
    metrics (vLLM exposes spec-decode stats); here we measure the end-to-end decode
    SPEEDUP and its batch-dependence, the centerpiece. Run once per server state. !!!

  vllm serve <8B> --port 8000                                            # no speculation
  python labs/m08_speculative/run_lab.py --endpoint http://localhost:8000 --model <id> --label nospec
  vllm serve <8B> --port 8000 --speculative-model <1B> --num-speculative-tokens 5   # speculation
  python labs/m08_speculative/run_lab.py --endpoint http://localhost:8000 --model <id> --label spec
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

from common.bench import OpenAIEndpoint, Request, run_open_loop  # noqa: E402

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
                    rows[(x[0], x[1])] = x
    rows[(str(row[0]), str(row[1]))] = [str(c) for c in row]
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows.values())


async def _decode_tok_s(ep, batch, decode_tokens, timeout):
    reqs = [Request(id=f"d{i}", messages=[{"role": "user", "content": "Write a long essay about LLM serving."}],
                    max_tokens=decode_tokens, intended_send_time=0.0) for i in range(batch)]
    res = await run_open_loop(reqs, ep, timeout=timeout)
    ok = [r for r in res if r.ok and r.first_token_time is not None]
    toks = sum(r.completion_tokens for r in ok)
    span = (max(r.end_time for r in ok) - min(r.first_token_time for r in ok)) if ok else 0
    return toks / span if span > 0 else float("nan")


async def run(args):
    os.makedirs(args.out, exist_ok=True)
    ep = OpenAIEndpoint(args.endpoint, args.model)
    print(f"\n[{args.label}] decode throughput vs batch (speculation is a low-batch win)")
    path = os.path.join(args.out, "spec_by_batch.csv")
    for bs in args.batch_sweep:
        tps = await _decode_tok_s(ep, bs, args.decode_tokens, args.timeout)
        _upsert(path, ["label", "batch", "decode_tok_s"], (args.label, bs, round(tps, 1)))
        print(f"    batch={bs:>3}  decode={tps:8.1f} tok/s")

    # if both labels are present, draw the speedup curve
    if HAVE_MPL and os.path.exists(path):
        data = {}
        with open(path) as f:
            for r in list(csv.reader(f))[1:]:
                data[(r[0], int(r[1]))] = float(r[2])
        labels = {k[0] for k in data}
        if {"spec", "nospec"} <= labels:
            batches = sorted({k[1] for k in data})
            speedup = [data.get(("spec", b), float("nan")) / data.get(("nospec", b), float("nan")) for b in batches]
            plt.figure(figsize=(6, 4))
            plt.plot(batches, speedup, "o-")
            plt.axhline(1.0, ls=":", color="gray")
            plt.xlabel("batch size"); plt.ylabel("speculative speedup (×)")
            plt.title("Lab 8 — speculation helps at low batch, fades as batching fills the FLOPs")
            plt.grid(True, alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(args.out, "spec_speedup.png"), dpi=130); plt.close()
            print(f"    -> speedup-vs-batch: {[round(s,2) for s in speedup]} (should fall toward 1.0 at high batch).")
    print(f"\nDone ({args.label}). Run the other label too; spec_speedup.png shows where it stops paying. "
          "Read α from the engine's spec-decode metrics.")


def main():
    p = argparse.ArgumentParser(description="Lab 8 — speculative decoding")
    p.add_argument("--endpoint", default="http://localhost:8000")
    p.add_argument("--model", default="")
    p.add_argument("--label", required=True, choices=["spec", "nospec"])
    p.add_argument("--batch-sweep", default="1,4,16,64", dest="batch_sweep")
    p.add_argument("--decode-tokens", type=int, default=256, dest="decode_tokens")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    args = p.parse_args()
    args.batch_sweep = [int(x) for x in args.batch_sweep.split(",") if x]
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
