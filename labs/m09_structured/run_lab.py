#!/usr/bin/env python3
"""Lab 9 — Structured / constrained decoding (Module 9).

!!! First real-hardware run — never executed on a GPU here, so run it on yours and see how it does across environments. The guided-decoding request field differs
    by engine: vLLM uses `guided_json` (extra_body); SGLang uses `response_format`
    with a json_schema. Set --engine accordingly and see how it does on your setup. !!!

  python labs/m09_structured/run_lab.py --endpoint http://localhost:8000 --model <id> --engine vllm
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
from common.eval import score_json_schema  # noqa: E402

SCHEMA = {
    "type": "object",
    "required": ["name", "age", "city"],
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "city": {"type": "string"}},
}


def _guided_extra_body(engine: str) -> dict:
    if engine == "vllm":
        return {"guided_json": SCHEMA}
    if engine == "sglang":
        return {"response_format": {"type": "json_schema",
                                    "json_schema": {"name": "person", "schema": SCHEMA}}}
    raise ValueError("engine must be vllm or sglang")


def _csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def _prompts(n):
    return [f"Give me one fictional person as JSON with keys name, age, city. Variation #{i}."
            for i in range(n)]


async def adherence_and_overhead(args, ep, outdir):
    """Unconstrained gives 'mostly valid'; constrained is valid by construction (§9.2).
    Also compare per-step decode latency (TPOT) — the cost of the per-step mask (§9.4)."""
    print("\n[1/2] Adherence + per-step overhead")
    eb = _guided_extra_body(args.engine)

    def reqs(constrained):
        return [Request(id=f"{'c' if constrained else 'u'}{i}",
                        messages=[{"role": "user", "content": pr}], max_tokens=128,
                        intended_send_time=i / args.rate,
                        meta={"extra_body": eb} if constrained else {})
                for i, pr in enumerate(_prompts(args.n))]

    res_u = await run_open_loop(reqs(False), ep, timeout=args.timeout)
    res_c = await run_open_loop(reqs(True), ep, timeout=args.timeout)
    s_u, s_c = score_json_schema(res_u, SCHEMA), score_json_schema(res_c, SCHEMA)
    m_u, m_c = compute_metrics(res_u, warmup=args.warmup), compute_metrics(res_c, warmup=args.warmup)

    print(f"    unconstrained: valid={s_u.valid_fraction:.2f}  parse_fail={s_u.parse_failure_rate:.2f}  "
          f"TPOT_p50={m_u.tpot_p50*1e3:.2f}ms")
    print(f"    constrained:   valid={s_c.valid_fraction:.2f}  parse_fail={s_c.parse_failure_rate:.2f}  "
          f"TPOT_p50={m_c.tpot_p50*1e3:.2f}ms")
    print(f"    -> constrained should be 1.00 valid by construction; per-step overhead = "
          f"{(m_c.tpot_p50 - m_u.tpot_p50)*1e3:+.2f}ms/token.")
    _csv(os.path.join(outdir, "adherence_overhead.csv"),
         ["mode", "valid_fraction", "parse_failure_rate", "tpot_p50_s"],
         [("unconstrained", round(s_u.valid_fraction, 3), round(s_u.parse_failure_rate, 3), round(m_u.tpot_p50, 5)),
          ("constrained", round(s_c.valid_fraction, 3), round(s_c.parse_failure_rate, 3), round(m_c.tpot_p50, 5))])
    print("\n[3] Quality trap (manual): on a reasoning+structure task, compare forcing JSON from the "
          "first token vs reason-in-prose-then-constrain the final span; score answer quality with "
          "common.eval.score_suite. Adherence can be 100% while quality drops (§9.5).")


def main():
    p = argparse.ArgumentParser(description="Lab 9 — structured decoding")
    p.add_argument("--endpoint", default="http://localhost:8000")
    p.add_argument("--model", default="")
    p.add_argument("--engine", default="vllm", choices=["vllm", "sglang"])
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--rate", type=float, default=20.0)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ep = OpenAIEndpoint(args.endpoint, args.model)
    asyncio.run(adherence_and_overhead(args, ep, args.out))
    print(f"\nDone. Artifacts in {args.out}")


if __name__ == "__main__":
    main()
