#!/usr/bin/env python3
"""Lab 2 — Attention, the KV cache, and where the memory goes (Module 2).

!!! First real-hardware run — never executed on a GPU here, so run it on yours and see how it
    does across environments; sanity-check the engine flags + /metrics field names on first run.
    The common/ harness API it calls IS tested. !!!

  # GPU box, server up (vllm serve <model> --max-model-len 65536):
  python labs/m02_kvcache/run_lab.py --endpoint http://localhost:8000 --model <hf-id> --cfg llama-3.1-8b
  python labs/m02_kvcache/run_lab.py --micro          # standalone attention-memory micro-bench (needs torch+CUDA)
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
from common.mem import MODELS, kv_budget, mem_estimate  # noqa: E402
from common.traffic import build_long_doc_qa  # noqa: E402

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


async def context_sweep(args, outdir):
    """Hold batch fixed, sweep context; decode throughput should fall as each step
    rereads a bigger KV cache (§2.2 — the Module 1 link). Predict OOM via kv_budget."""
    print("\n[1/2] KV law + context-throughput sweep")
    cfg = MODELS[args.cfg]
    ep = OpenAIEndpoint(args.endpoint, args.model)
    rows = []
    for ctx in args.ctx_sweep:
        budget = kv_budget(cfg, vram_gb=args.vram_gb, context_len=ctx)
        pred_gb = mem_estimate(cfg, context_len=ctx, batch=args.batch).gb()["total"]
        # short questions over a ctx-token document; decode-heavy so throughput reflects KV traffic
        reqs = build_long_doc_qa(args.n, doc_tokens=ctx, prefix_share_ratio=0.0,
                                 rate_or_trace=args.rate, seed=args.seed)
        for r in reqs:
            r.max_tokens = 128
        try:
            m = compute_metrics(await run_open_loop(reqs, ep, timeout=args.timeout), warmup=args.warmup)
            tput = m.throughput_tok_s
            # OOM surfaces as errored requests (run_open_loop doesn't raise), so read n_error
            note = f"errors={m.n_error}/{m.n}" if m.n_error else ""
        except ValueError as exc:  # no successful results at all (server down / total OOM)
            tput = float("nan"); note = f"FAILED: {exc}"
        rows.append((ctx, round(pred_gb, 1), budget["max_concurrency"], round(tput, 1), note))
        print(f"    ctx={ctx:>6}  predicted={pred_gb:5.1f} GB  max_conc≈{budget['max_concurrency']:>4}  "
              f"decode_tput={tput:8.1f} tok/s  {note}")
    _csv(os.path.join(outdir, "context_sweep.csv"),
         ["ctx", "predicted_total_gb", "max_concurrency", "decode_tput_tok_s", "note"], rows)
    if HAVE_MPL:
        ok = [r for r in rows if r[3] == r[3]]  # drop NaN
        if ok:
            plt.figure(figsize=(6, 4))
            plt.plot([r[0] for r in ok], [r[3] for r in ok], "o-")
            plt.xlabel("context length (tokens)"); plt.ylabel("decode throughput (tok/s)")
            plt.title("Lab 2 — long context descends the roofline (fixed batch)")
            plt.grid(True, alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(outdir, "context_throughput.png"), dpi=130); plt.close()
    print("    -> find the ctx where the server OOMs; compare to the kv_budget prediction (target ~10%).")


def micro_attention(outdir):
    """Standalone: naive softmax(QKᵀ)·V (materializes S×S) vs FlashAttention-style SDPA.
    Plots peak memory vs sequence length — O(S²) vs O(S) (§2.4). Needs torch + CUDA."""
    print("\n[micro] attention memory: naive O(S²) vs SDPA O(S)")
    import torch  # local import so the harness path needs no torch
    import torch.nn.functional as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = 64
    rows = []
    for S in (512, 1024, 2048, 4096, 8192):
        out = {}
        for kind in ("naive", "sdpa"):
            q = torch.randn(1, 8, S, d, device=dev, dtype=torch.float16)
            k = torch.randn_like(q); v = torch.randn_like(q)
            if dev == "cuda":
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            if kind == "naive":
                a = torch.softmax((q @ k.transpose(-1, -2)) / d ** 0.5, dim=-1) @ v  # S×S materialized
            else:
                a = F.scaled_dot_product_attention(q, k, v)
            del a
            out[kind] = (torch.cuda.max_memory_allocated() / 1e6) if dev == "cuda" else float("nan")
        rows.append((S, round(out["naive"], 1), round(out["sdpa"], 1)))
        print(f"    S={S:>5}  naive={out['naive']:8.1f} MB   sdpa={out['sdpa']:8.1f} MB")
    _csv(os.path.join(outdir, "attention_memory.csv"), ["seq_len", "naive_mb", "sdpa_mb"], rows)
    if HAVE_MPL and dev == "cuda":
        plt.figure(figsize=(6, 4))
        plt.plot([r[0] for r in rows], [r[1] for r in rows], "o-", label="naive  O(S²)")
        plt.plot([r[0] for r in rows], [r[2] for r in rows], "s-", label="SDPA  O(S)")
        plt.xlabel("sequence length"); plt.ylabel("peak memory (MB)"); plt.legend()
        plt.title("Lab 2 — FlashAttention never materializes the S×S matrix")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "attention_memory.png"), dpi=130); plt.close()


def main():
    p = argparse.ArgumentParser(description="Lab 2 — KV cache")
    p.add_argument("--endpoint", default="http://localhost:8000")
    p.add_argument("--model", default="")
    p.add_argument("--cfg", default="llama-3.1-8b", choices=list(MODELS), help="mem-math config")
    p.add_argument("--ctx-sweep", default="2048,8192,32768,65536", dest="ctx_sweep")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--rate", type=float, default=8.0)
    p.add_argument("--vram-gb", type=float, default=48.0, dest="vram_gb")
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--micro", action="store_true", help="run only the attention-memory micro-bench")
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    args = p.parse_args()
    args.ctx_sweep = [int(x) for x in args.ctx_sweep.split(",") if x]
    os.makedirs(args.out, exist_ok=True)
    if args.micro:
        micro_attention(args.out)
    else:
        asyncio.run(context_sweep(args, args.out))
    print(f"\nDone. Artifacts in {args.out}")


if __name__ == "__main__":
    main()
