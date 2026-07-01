#!/usr/bin/env python3
"""Lab 0 — Numbers, tokens, and memory (Module 0).

The `mem_estimate()` math lives in `common/mem.py` and is unit-tested, so the
tables below run ANYWHERE (no GPU). The `--validate` step compares the estimate
to a real vLLM allocator and needs the GPU box.

!!! The tables are verified (pure math); the --validate path (nvidia-smi parsing) hasn't
    been run on a GPU here — sanity-check it on your first run. !!!

  python labs/m00_numerics/run_lab.py                                  # tables, no GPU
  python labs/m00_numerics/run_lab.py --validate --model llama-3.1-8b  # on the GPU box, server up
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

from common.mem import MODELS, bytes_per_param, mem_estimate  # noqa: E402


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def params_bytes_table(outdir):
    print("\n[1] params <-> bytes (weights only)")
    rows = []
    for name, cfg in MODELS.items():
        for dt in ("bf16", "int8", "int4"):
            gb = cfg["params_b"] * bytes_per_param(dt)
            rows.append((name, dt, round(gb, 2)))
            print(f"    {name:14s} {dt:5s}  {gb:6.1f} GB")
    _write_csv(os.path.join(outdir, "params_bytes.csv"), ["model", "dtype", "weights_gb"], rows)


def serve_memory_table(outdir, ctx, batch):
    print(f"\n[2] serve-time memory  (context={ctx}, batch={batch})")
    rows = []
    for name, cfg in MODELS.items():
        g = mem_estimate(cfg, context_len=ctx, batch=batch).gb()
        rows.append((name, ctx, batch, round(g["weights"], 2), round(g["kv"], 2), round(g["total"], 2)))
        print(f"    {name:14s} weights={g['weights']:5.1f} + KV={g['kv']:5.1f} + oh={g['overhead']:.1f} = {g['total']:6.1f} GB")
    _write_csv(os.path.join(outdir, "serve_memory.csv"),
               ["model", "ctx", "batch", "weights_gb", "kv_gb", "total_gb"], rows)


def validate(args, outdir):
    print("\n[3] validate vs the real allocator (GPU box; server already serving --model)")
    cfg = MODELS[args.model]
    est = mem_estimate(cfg, weight_dtype=args.weight_dtype, kv_dtype=args.kv_dtype,
                       context_len=args.ctx, batch=args.batch)
    used_mb = _nvidia_smi_used_mb()
    pred_mb = est.total / 1e6
    err = abs(used_mb - pred_mb) / used_mb * 100 if used_mb else float("nan")
    print(f"    predicted total = {pred_mb/1e3:5.1f} GB   nvidia-smi used = {used_mb/1e3:5.1f} GB   error = {err:.1f}%")
    print("    target: within ~15%. Also compare vLLM's reported KV-cache *blocks* "
          "(startup log or GET /metrics) against the KV term.")
    _write_csv(os.path.join(outdir, "validation.csv"),
               ["model", "predicted_gb", "measured_gb", "error_pct"],
               [(args.model, round(pred_mb / 1e3, 2), round(used_mb / 1e3, 2), round(err, 1))])


def _nvidia_smi_used_mb() -> float:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).decode()
    return float(out.strip().splitlines()[0])


def main():
    p = argparse.ArgumentParser(description="Lab 0 — memory calculator")
    p.add_argument("--validate", action="store_true", help="compare to nvidia-smi on the GPU box")
    p.add_argument("--model", default="llama-3.1-8b", choices=list(MODELS))
    p.add_argument("--weight-dtype", default="bf16", dest="weight_dtype")
    p.add_argument("--kv-dtype", default="fp16", dest="kv_dtype")
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--vram-gb", type=float, default=24.0, dest="vram_gb")
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    params_bytes_table(args.out)
    serve_memory_table(args.out, args.ctx, args.batch)
    if args.validate:
        validate(args, args.out)
    print(f"\nDone. CSVs in {args.out}")


if __name__ == "__main__":
    main()
