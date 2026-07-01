"""Metric definitions — defined ONCE here so no lab re-implements (or fudges) them.

INVARIANT 1: latencies come from ``Result.e2e`` / ``Result.ttft`` (measured from
``intended_send_time``).
INVARIANT 3: ``compute_metrics`` discards the first ``warmup`` requests; headline
numbers come from ``aggregate_runs`` over ≥3 repeats with a confidence interval.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import numpy as np

from .client import stream_chat
from .schema import Request, Result, SessionResult


@dataclass
class SLO:
    e2e: float | None = None  # seconds
    ttft: float | None = None  # seconds


@dataclass
class Metrics:
    ttft_p50: float
    ttft_p95: float
    ttft_p99: float
    tpot_p50: float
    tpot_p95: float
    tpot_p99: float
    e2e_p50: float
    e2e_p95: float
    e2e_p99: float
    throughput_tok_s: float
    throughput_req_s: float
    goodput_req_s: float
    cache_hit_rate: float
    n: int
    n_error: int


# field names aggregate_runs averages across repeats
_NUMERIC_FIELDS = [
    "ttft_p50", "ttft_p95", "ttft_p99",
    "tpot_p50", "tpot_p95", "tpot_p99",
    "e2e_p50", "e2e_p95", "e2e_p99",
    "throughput_tok_s", "throughput_req_s", "goodput_req_s", "cache_hit_rate",
]


@dataclass
class MetricsWithCI:
    """mean ± 95% half-width per metric, across repeated runs (INVARIANT 3)."""

    n_runs: int
    mean: dict[str, float]
    ci95: dict[str, float]  # half-width; report mean ± ci95

    def __str__(self) -> str:
        rows = [f"  {k:18s} {self.mean[k]:.4g} ± {self.ci95[k]:.2g}" for k in self.mean]
        return f"MetricsWithCI(n_runs={self.n_runs})\n" + "\n".join(rows)


def _pct(xs: list[float], q: float) -> float:
    return float(np.percentile(xs, q)) if xs else float("nan")


def compute_metrics(results: list[Result], *, slo: SLO | None = None, warmup: int = 0) -> Metrics:
    """Reduce one run's results to metrics. Discards the first ``warmup`` requests
    (ordered by intended send time) to drop cold start."""
    kept = sorted(results, key=lambda r: r.intended_send_time)[warmup:]
    if not kept:
        raise ValueError("no results left after warmup")

    e2e = [r.e2e for r in kept]  # includes errors — they count toward the tail (INV 1)
    ttft = [r.ttft for r in kept if r.ttft is not None]
    tpot = [r.tpot for r in kept if r.tpot is not None]

    # wall window = from the first intended send to the last completion
    window = max(r.end_time for r in kept) - min(r.intended_send_time for r in kept)
    window = max(window, 1e-9)

    tok = sum(r.completion_tokens for r in kept if r.ok)
    n_ok = sum(1 for r in kept if r.ok)

    if slo is not None:
        def meets(r: Result) -> bool:
            if not r.ok:
                return False
            if slo.e2e is not None and r.e2e > slo.e2e:
                return False
            if slo.ttft is not None and (r.ttft is None or r.ttft > slo.ttft):
                return False
            return True

        good = sum(1 for r in kept if meets(r))
    else:
        good = n_ok

    prompt_total = sum(r.prompt_tokens for r in kept)
    cached_total = sum(r.cached_prefix_tokens for r in kept)

    return Metrics(
        ttft_p50=_pct(ttft, 50), ttft_p95=_pct(ttft, 95), ttft_p99=_pct(ttft, 99),
        tpot_p50=_pct(tpot, 50), tpot_p95=_pct(tpot, 95), tpot_p99=_pct(tpot, 99),
        e2e_p50=_pct(e2e, 50), e2e_p95=_pct(e2e, 95), e2e_p99=_pct(e2e, 99),
        throughput_tok_s=tok / window,
        throughput_req_s=n_ok / window,
        goodput_req_s=good / window,
        cache_hit_rate=(cached_total / prompt_total) if prompt_total else 0.0,
        n=len(kept),
        n_error=sum(1 for r in kept if not r.ok),
    )


def aggregate_runs(metrics_list: list[Metrics]) -> MetricsWithCI:
    """Mean ± 95% CI across repeated runs. A single run is not a measurement."""
    if not metrics_list:
        raise ValueError("need at least one run")
    k = len(metrics_list)
    mean: dict[str, float] = {}
    ci95: dict[str, float] = {}
    for fld in _NUMERIC_FIELDS:
        xs = np.array([getattr(m, fld) for m in metrics_list], dtype=float)
        mean[fld] = float(np.nanmean(xs))
        if k > 1:
            sd = float(np.nanstd(xs, ddof=1))
            ci95[fld] = 1.96 * sd / np.sqrt(k)
        else:
            ci95[fld] = 0.0
    return MetricsWithCI(n_runs=k, mean=mean, ci95=ci95)


# --------------------------------------------------------------------------- #
# Agentic session metrics (Module 10) — INVARIANT 7.
# --------------------------------------------------------------------------- #
@dataclass
class SessionMetrics:
    latency_breakdown: dict  # {"model":x, "tool":y, "reprefill":z} as fractions summing to 1
    cache_hit_rate: float
    slot_utilization: float  # total_model_time / slot_held_time
    reprefill_tokens_total: int


def compute_session_metrics(session_results: list[SessionResult], *, warmup: int = 0) -> SessionMetrics:
    """Decompose agent cost into model(decode) / tool / re-prefill, with cache-hit
    rate alongside — never model TTFT/TPOT alone (INVARIANT 7, Module 10 §10.4)."""
    kept = session_results[warmup:]
    if not kept:
        raise ValueError("no sessions left after warmup")

    decode = sum(t.decode_time for s in kept for t in s.turns)
    prefill = sum(t.prefill_time for s in kept for t in s.turns)
    tool = sum(s.total_tool_time for s in kept)
    total = decode + prefill + tool
    total = total if total > 0 else 1e-9

    prompt_total = sum(t.cached_prefix_tokens + t.prefill_tokens for s in kept for t in s.turns)
    cached_total = sum(t.cached_prefix_tokens for s in kept for t in s.turns)

    model_time = sum(s.total_model_time for s in kept)
    slot_held = sum(s.slot_held_time for s in kept)

    return SessionMetrics(
        latency_breakdown={
            "model": decode / total,
            "tool": tool / total,
            "reprefill": prefill / total,
        },
        cache_hit_rate=(cached_total / prompt_total) if prompt_total else 0.0,
        slot_utilization=(model_time / slot_held) if slot_held else 0.0,
        reprefill_tokens_total=sum(s.total_reprefill_tokens for s in kept),
    )


# --------------------------------------------------------------------------- #
# Determinism (Module 12 §12.5/§12.8).
# --------------------------------------------------------------------------- #
@dataclass
class DeterminismReport:
    identical_fraction: float  # fraction of runs bitwise-identical to the first
    divergence_examples: list[str] = field(default_factory=list)


async def determinism_check(
    endpoint,
    prompt: str,
    *,
    n_runs: int = 8,
    vary_batch: bool = True,
    seed: int = 0,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> DeterminismReport:
    """Send the same temperature-0 prompt ``n_runs`` times. With ``vary_batch``
    each call advertises a different batch composition; an engine whose kernels
    are not batch-invariant will diverge. Demonstrates that "temperature 0" is
    not, by itself, reproducible."""
    outputs: list[str] = []
    for i in range(n_runs):
        meta = {"batch_tag": i if vary_batch else 0}
        req = Request(id=f"det-{i}", messages=[{"role": "user", "content": prompt}],
                      max_tokens=32, meta=meta)
        res = await stream_chat(endpoint, req, api_key=api_key, timeout=timeout)
        outputs.append(res.output_text)

    first = outputs[0]
    identical = sum(1 for o in outputs if o == first)
    divergent = sorted({o for o in outputs if o != first})
    return DeterminismReport(
        identical_fraction=identical / len(outputs),
        divergence_examples=divergent[:5],
    )
