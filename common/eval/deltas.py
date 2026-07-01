"""Per-task accuracy deltas (Module 3).

A quantization/eviction change is judged per task, never on an average — so a
regression concentrated in reasoning/code is visible even when the mean looks
flat (INVARIANT 6).
"""

from __future__ import annotations


def quant_delta(baseline: dict, candidate: dict) -> dict[str, float]:
    """``candidate_score - baseline_score`` per task. Inputs are
    ``{task: TaskScore}`` dicts from ``score_suite`` (or raw ``{task: float}``)."""

    def _val(x):
        return x.score if hasattr(x, "score") else float(x)

    tasks = set(baseline) | set(candidate)
    out: dict[str, float] = {}
    for t in sorted(tasks):
        if t in baseline and t in candidate:
            out[t] = _val(candidate[t]) - _val(baseline[t])
    return out
