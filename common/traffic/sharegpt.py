"""ShareGPT-style traffic — realistic length variance, NO shared prefix.

The negative control for prefix caching (INVARIANT 4 / Module 5): every prompt
is unique, so prefix caching must show ≈ no benefit here. Synthesises content by
default; pass ``source`` to replay a real trace (one JSON conversation per line).
"""

from __future__ import annotations

import json

import numpy as np

from ..bench.schema import Request
from ._synth import approx_tokens, make_text
from .schedules import poisson_schedule, trace_schedule

_PROFILES = {
    # (prompt_mu, prompt_sigma, out_mu, out_sigma) in log-space over tokens
    "default": (5.2, 0.8, 4.6, 0.7),  # ~180 prompt / ~100 out, heavy tail
    "short": (4.0, 0.5, 3.5, 0.5),
    "long": (6.5, 0.7, 5.5, 0.6),
}


def build_sharegpt(
    n: int,
    *,
    rate_or_trace,
    length_profile: str = "default",
    seed: int = 0,
    source: str | None = None,
) -> list[Request]:
    rng = np.random.default_rng(seed)
    times = _times(n, rate_or_trace, seed)

    if source is not None:
        convos = _load_source(source, n)
        reqs = []
        for i, (prompt, out_len) in enumerate(convos):
            reqs.append(Request(
                id=f"sg-{i}", messages=[{"role": "user", "content": prompt}],
                max_tokens=out_len, intended_send_time=times[i],
                meta={"prefix_group": None, "task": None},
            ))
        return reqs

    mu_p, sig_p, mu_o, sig_o = _PROFILES[length_profile]
    reqs = []
    for i in range(n):
        p_tok = int(np.clip(rng.lognormal(mu_p, sig_p), 8, 8000))
        o_tok = int(np.clip(rng.lognormal(mu_o, sig_o), 4, 4000))
        # unique leading marker => no two prompts share a prefix
        prompt = make_text(p_tok, prefix=f"s{i}q{rng.integers(1_000_000)}")
        reqs.append(Request(
            id=f"sg-{i}", messages=[{"role": "user", "content": prompt}],
            max_tokens=o_tok, intended_send_time=times[i],
            meta={"prefix_group": None, "task": None, "prompt_tokens": approx_tokens(prompt)},
        ))
    return reqs


def _times(n: int, rate_or_trace, seed: int) -> list[float]:
    if isinstance(rate_or_trace, (int, float)) and not isinstance(rate_or_trace, bool):
        return poisson_schedule(n, float(rate_or_trace), seed)
    if isinstance(rate_or_trace, str):
        return trace_schedule(rate_or_trace)[:n]
    if isinstance(rate_or_trace, (list, tuple)):
        return list(rate_or_trace)[:n]
    raise TypeError("rate_or_trace must be a rate (number), a trace path, or a list of times")


def _load_source(source: str, n: int):
    out = []
    with open(source) as fh:
        for line in fh:
            if len(out) >= n:
                break
            obj = json.loads(line)
            msgs = obj.get("conversations") or obj.get("messages") or []
            prompt = next((m.get("value") or m.get("content", "") for m in msgs), "")
            out.append((prompt, obj.get("max_tokens", 256)))
    return out
