"""Needle-in-haystack — retrieval at a known depth.

Plants a known fact at ``needle_depth_fraction`` of a ``context_len``-token
context and asks for it. ``meta["expected"]`` and ``meta["needle_depth"]`` drive
retrieval scoring. Put the needle where eviction drops it. This is the workload
behind INVARIANT 5: long-context accuracy is needle retrieval, not perplexity
(Module 7 §7.7).
"""

from __future__ import annotations

import numpy as np

from ..bench.schema import Request
from ._synth import make_text


def build_needle(
    n: int,
    *,
    context_len: int,
    needle_depth_fraction: float = 0.5,
    seed: int = 0,
    rate: float = 1.0,
) -> list[Request]:
    rng = np.random.default_rng(seed)
    reqs = []
    for i in range(n):
        secret = int(rng.integers(100_000, 1_000_000))
        depth = float(np.clip(needle_depth_fraction, 0.0, 1.0))
        before = max(1, int(context_len * depth))
        after = max(1, context_len - before)
        needle = f"The magic number is {secret}."
        context = f"{make_text(before)} {needle} {make_text(after)}"
        reqs.append(Request(
            id=f"ndl-{i}",
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": "What is the magic number? Answer with the number only."},
            ],
            max_tokens=16, intended_send_time=float(i) / max(rate, 1e-9),
            meta={"task": "needle", "expected": str(secret), "needle_depth": depth},
        ))
    return reqs
