#!/usr/bin/env python3
"""Lab 3 — Quantization: shrinking the numerator (Module 3).

!!! First real-hardware run — never executed on a GPU here, so run it on yours and see how it does across environments. The accuracy probe below is a tiny built-in
    set so the script runs; swap in the full suite (lm-eval / HF datasets) for a
    real number. Quant config is a launch flag, so run once per server state. !!!

  # relaunch the server per config, run once each (rows accumulate by --label):
  vllm serve <model> --port 8000                          # FP16 baseline
  python labs/m03_quantization/run_lab.py --endpoint http://localhost:8000 --model <id> --label fp16 --params-b 8.03
  vllm serve <model> --port 8000 --quantization awq       # then INT4
  python labs/m03_quantization/run_lab.py --endpoint http://localhost:8000 --model <id> --label int4_awq --params-b 4.0
  vllm serve <model> --port 8000 --quantization fp8       # then FP8
  python labs/m03_quantization/run_lab.py --endpoint http://localhost:8000 --model <id> --label fp8 --params-b 8.03
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
from common.eval import score_suite  # noqa: E402

# Tiny built-in accuracy probe — REPLACE with the real suite (MMLU/GSM8K/HumanEval/IFEval).
EVAL = [
    {"task": "gsm8k", "q": "What is 17 times 4? End with the number.", "expected": "68"},
    {"task": "gsm8k", "q": "A box has 12 apples; you remove 5 then add 8. How many? End with the number.", "expected": "15"},
    {"task": "gsm8k", "q": "What is 144 divided by 12? End with the number.", "expected": "12"},
    {"task": "mmlu", "q": "Which is a noble gas? A) Oxygen B) Argon C) Sodium D) Iron. Answer with one letter.", "expected": "B"},
    {"task": "mmlu", "q": "2+2 equals? A) 3 B) 5 C) 4 D) 22. Answer with one letter.", "expected": "C"},
]


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


async def _probe(ep, prompt, max_tokens, timeout):
    return (await run_open_loop([Request(id="p", messages=[{"role": "user", "content": prompt}],
                                         max_tokens=max_tokens, intended_send_time=0.0)], ep, timeout=timeout))[0]


async def run(args):
    os.makedirs(args.out, exist_ok=True)
    ep = OpenAIEndpoint(args.endpoint, args.model)
    print(f"\n[{args.label}] phase isolation + per-task accuracy")

    # mechanism split: prefill (long prompt, 1 tok) vs decode (short prompt, long output)
    pre = await _probe(ep, "word " * args.prompt_tokens, 1, args.timeout)
    dec = await _probe(ep, "Write a long essay.", args.decode_tokens, args.timeout)
    prefill_tok_s = (pre.prompt_tokens / pre.ttft) if pre.ttft else float("nan")
    decode_tok_s = (1.0 / dec.tpot) if dec.tpot else float("nan")

    # per-task accuracy (NEVER averaged — INVARIANT 6)
    reqs = [Request(id=f"e{i}", messages=[{"role": "user", "content": e["q"]}], max_tokens=64,
                    intended_send_time=i * 0.05, meta={"task": e["task"], "expected": e["expected"]})
            for i, e in enumerate(EVAL)]
    suite = score_suite(await run_open_loop(reqs, ep, timeout=args.timeout))
    accs = {t: round(s.score, 3) for t, s in suite.items()}

    print(f"    prefill={prefill_tok_s:8.0f} tok/s   decode={decode_tok_s:6.1f} tok/s   accuracy={accs}")
    print("    -> weight-only INT4 should speed DECODE but ~not prefill (low batch); FP8 speeds both (§3.2).")
    _upsert(os.path.join(args.out, "quant.csv"),
            ["label", "params_b", "prefill_tok_s", "decode_tok_s", "gsm8k", "mmlu"],
            (args.label, args.params_b, round(prefill_tok_s, 1), round(decode_tok_s, 1),
             accs.get("gsm8k", ""), accs.get("mmlu", "")))
    print(f"\nDone ({args.label}). Re-run after relaunching the server at another precision; "
          f"rows accumulate in {args.out}/quant.csv (quality × throughput per config).")


def main():
    p = argparse.ArgumentParser(description="Lab 3 — quantization")
    p.add_argument("--endpoint", default="http://localhost:8000")
    p.add_argument("--model", default="")
    p.add_argument("--label", required=True, help="config label, e.g. fp16 / int4_awq / fp8")
    p.add_argument("--params-b", type=float, default=8.03, dest="params_b")
    p.add_argument("--prompt-tokens", type=int, default=4096, dest="prompt_tokens")
    p.add_argument("--decode-tokens", type=int, default=256, dest="decode_tokens")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
