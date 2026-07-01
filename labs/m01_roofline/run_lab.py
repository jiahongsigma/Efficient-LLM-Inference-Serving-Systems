#!/usr/bin/env python3
"""Lab 1 — The roofline of a transformer forward pass (Module 1).

!!! First real-hardware run — never executed on a GPU here, so run it on yours and see how it does across environments. Achieved *bandwidth* (the honest y-axis) needs
    Nsight Compute; here we derive FLOP/s from measured tok/s and place the points by
    arithmetic intensity. Pair with `nvidia-smi dmon -s u` for SM utilization. !!!

  python labs/m01_roofline/run_lab.py --endpoint http://localhost:8000 --model <id> \
      --params-b 8.03 --peak-bf16-tflops 989 --hbm-tbs 3.35
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


def _csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


async def _one(ep, prompt, max_tokens, timeout, batch=1):
    reqs = [Request(id=f"r{i}", messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens, intended_send_time=0.0) for i in range(batch)]
    return await run_open_loop(reqs, ep, timeout=timeout)


async def run(args):
    os.makedirs(args.out, exist_ok=True)
    ep = OpenAIEndpoint(args.endpoint, args.model)
    N = args.params_b * 1e9
    P = args.peak_bf16_tflops * 1e12
    B = args.hbm_tbs * 1e12
    ridge = P / B
    print(f"\nRidge point I* = P/B = {ridge:.0f} FLOP/byte  (P={args.peak_bf16_tflops} TFLOP/s, B={args.hbm_tbs} TB/s)")

    # --- phase isolation ---
    long_prompt = "word " * args.prompt_tokens
    pre = (await _one(ep, long_prompt, 1, args.timeout))[0]
    dec = (await _one(ep, "Write a very long detailed essay about transformers.", args.decode_tokens, args.timeout))[0]
    prefill_tok_s = (pre.prompt_tokens / pre.ttft) if pre.ttft else float("nan")
    decode_tok_s = (1.0 / dec.tpot) if dec.tpot else float("nan")
    print(f"\n[phases]  prefill ≈ {prefill_tok_s:8.0f} tok/s (I≈{pre.prompt_tokens})   "
          f"decode ≈ {decode_tok_s:6.1f} tok/s (I≈1)")

    # --- the batch-1 decode bound: time/token >= weight_bytes / B (§1.3) ---
    weight_bytes = 2 * N  # bf16
    bound_ms = weight_bytes / B * 1e3
    print(f"[bound]   batch-1 decode floor = weight_bytes/B = {bound_ms:.2f} ms/token; "
          f"measured TPOT = {(dec.tpot or float('nan'))*1e3:.2f} ms/token")

    # --- batch sweep: decode throughput climbs the roofline ---
    print("\n[batch sweep] aggregate decode throughput vs batch")
    rows = []
    for bs in args.batch_sweep:
        res = await _one(ep, "Write a long essay.", args.decode_tokens, args.timeout, batch=bs)
        ok = [r for r in res if r.ok and r.first_token_time is not None]
        toks = sum(r.completion_tokens for r in ok)
        span = (max(r.end_time for r in ok) - min(r.first_token_time for r in ok)) if ok else 0
        agg = toks / span if span > 0 else float("nan")
        rows.append((bs, round(agg, 1)))
        print(f"    batch={bs:>3}  aggregate decode = {agg:8.1f} tok/s")
    _csv(os.path.join(args.out, "batch_sweep.csv"), ["batch", "agg_decode_tok_s"], rows)

    # --- roofline plot: ceilings + the two measured operating points ---
    if HAVE_MPL:
        import numpy as np
        I = np.logspace(-1, 3, 200)
        plt.figure(figsize=(6.5, 4.5))
        plt.loglog(I, np.minimum(P, I * B), "k-", lw=2, label="roofline min(P, I·B)")
        plt.axvline(ridge, ls=":", color="gray"); plt.text(ridge, P * 0.5, " ridge")
        # decode point: I≈1, achieved FLOP/s ≈ 2N · decode_tok/s
        plt.scatter([1], [2 * N * decode_tok_s], c="#e76f51", s=60, zorder=5, label="decode (I≈1)")
        plt.scatter([pre.prompt_tokens], [2 * N * prefill_tok_s], c="#2a9d8f", s=60, zorder=5, label="prefill")
        plt.xlabel("arithmetic intensity (FLOP/byte)"); plt.ylabel("achieved FLOP/s")
        plt.title("Lab 1 — prefill is compute-bound, decode is memory-bound")
        plt.legend(fontsize=8); plt.grid(True, which="both", alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(args.out, "roofline.png"), dpi=130); plt.close()
    print(f"\nDone. Artifacts in {args.out}. In your writeup: why a higher-FLOP/s GPU at the same B "
          "would not speed up batch-1 decode.")


def main():
    p = argparse.ArgumentParser(description="Lab 1 — roofline")
    p.add_argument("--endpoint", default="http://localhost:8000")
    p.add_argument("--model", default="")
    p.add_argument("--params-b", type=float, default=8.03, dest="params_b")
    p.add_argument("--peak-bf16-tflops", type=float, default=989.0, dest="peak_bf16_tflops")
    p.add_argument("--hbm-tbs", type=float, default=3.35, dest="hbm_tbs")
    p.add_argument("--prompt-tokens", type=int, default=4096, dest="prompt_tokens")
    p.add_argument("--decode-tokens", type=int, default=256, dest="decode_tokens")
    p.add_argument("--batch-sweep", default="1,4,16,64", dest="batch_sweep")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    args = p.parse_args()
    args.batch_sweep = [int(x) for x in args.batch_sweep.split(",") if x]
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
