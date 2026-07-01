#!/usr/bin/env python3
"""infra/smoke.py — is my GPU box wired up correctly?

A 30-second check before running any lab: send ONE streaming request through the
common/ harness to a live engine and print what came back. If this works, the
whole harness works — so every lab's request/stream/metrics path is good, and any
remaining failure is a per-lab engine flag, not the plumbing.

  # after: vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
  python infra/smoke.py --endpoint http://localhost:8000 --model meta-llama/Llama-3.1-8B-Instruct

Exit code 0 = wired up; non-zero = something's off (with a hint).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from common.bench import OpenAIEndpoint, Request, stream_chat  # noqa: E402


async def main_async(args) -> int:
    ep = OpenAIEndpoint(args.endpoint, args.model)
    req = Request(id="smoke", messages=[{"role": "user", "content": args.prompt}],
                  max_tokens=args.max_tokens)
    print(f">> {args.endpoint}  [{args.model or '(no --model!)'}]  — sending one request...")
    r = await stream_chat(ep, req, api_key=(args.api_key or None), timeout=args.timeout)

    print(f"   status            : {r.status}")
    if r.status != "ok":
        print("   !! no completion streamed. Check: is the server up (curl "
              f"{args.endpoint}/v1/models)? is --model the EXACT served id? is --api-key set if required?")
        return 1
    print(f"   TTFT              : {r.ttft * 1e3:.0f} ms" if r.ttft is not None else "   TTFT              : n/a")
    print(f"   TPOT              : {r.tpot * 1e3:.1f} ms/token" if r.tpot is not None else "   TPOT              : n/a")
    print(f"   prompt tokens     : {r.prompt_tokens}")
    print(f"   completion tokens : {r.completion_tokens}")
    print(f"   cached prefix     : {r.cached_prefix_tokens}  (>0 only with --enable-prefix-caching + a shared prefix)")
    print(f"   output            : {r.output_text!r}")
    if r.completion_tokens == 0 or not r.output_text:
        print("   !! connected but no tokens came back — usually a wrong model id or a server-side error.")
        return 1
    print(">> OK — the harness reaches the engine. You're cleared to run labs (start with M5 / M12).")
    return 0


def main():
    p = argparse.ArgumentParser(description="smoke-test the common/ harness against a live engine")
    p.add_argument("--endpoint", default="http://localhost:8000")
    p.add_argument("--model", default="", help="the EXACT model id the engine serves")
    p.add_argument("--prompt", default="Say hello in exactly three words.")
    p.add_argument("--max-tokens", type=int, default=16, dest="max_tokens")
    p.add_argument("--api-key", default="", dest="api_key")
    p.add_argument("--timeout", type=float, default=60.0)
    raise SystemExit(asyncio.run(main_async(p.parse_args())))


if __name__ == "__main__":
    main()
