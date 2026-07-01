"""Arrival schedules — when each request is *intended* to be sent.

All times are seconds from run start. Builders stamp these onto
``Request.intended_send_time``; ``run_open_loop`` honours them regardless of
server state (the basis of INVARIANT 1/2).
"""

from __future__ import annotations

import numpy as np


def poisson_schedule(n: int, rate: float, seed: int = 0) -> list[float]:
    """``n`` Poisson arrivals at ``rate`` req/s. First arrival at t=0."""
    if n <= 0:
        return []
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(1.0 / max(rate, 1e-9), size=n)
    times = np.cumsum(gaps) - gaps[0]  # shift so the first request is at t=0
    return times.tolist()


def trace_schedule(trace_file: str) -> list[float]:
    """Arrival times (seconds from start, one per line) from a real trace file."""
    times: list[float] = []
    with open(trace_file) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                times.append(float(line.split(",")[0]))
    return times


def burst_schedule(
    n: int,
    *,
    base_rate: float,
    burst_every: float,
    burst_size: int,
    seed: int = 0,
) -> list[float]:
    """A decode-heavy baseline (Poisson at ``base_rate``) with a cluster of
    ``burst_size`` near-simultaneous arrivals every ``burst_every`` seconds —
    the shape used to expose prefill/decode interference (Modules 4, 8)."""
    if n <= 0:
        return []
    rng = np.random.default_rng(seed)
    base_n = max(0, n - burst_size * max(1, int(n // max(burst_size, 1) // 4)))
    base = poisson_schedule(base_n, base_rate, seed)
    times = list(base)
    t = burst_every
    while len(times) < n:
        jitter = rng.uniform(0, 0.01, size=min(burst_size, n - len(times)))
        times.extend((t + jitter).tolist())
        t += burst_every
    return sorted(times[:n])
