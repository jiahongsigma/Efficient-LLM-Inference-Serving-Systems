"""Needle retrieval scoring (Module 7, INVARIANT 5).

Retrieval rate overall and bucketed by planted depth — so an eviction method's
catastrophic failure *in the evicted region* is visible beside its memory win,
instead of being hidden by a good average/perplexity number.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class NeedleScore:
    retrieval_rate: float
    by_depth: dict[str, float] = field(default_factory=dict)  # depth bucket -> rate
    n: int = 0


def _bucket(depth: float, n_buckets: int = 5) -> str:
    b = min(n_buckets - 1, int(depth * n_buckets))
    lo, hi = b / n_buckets, (b + 1) / n_buckets
    return f"{lo:.1f}-{hi:.1f}"


def score_needle(results, n_buckets: int = 5) -> NeedleScore:
    found_by_bucket: dict[str, list[int]] = defaultdict(list)
    total_found = 0
    n = 0
    for r in results:
        if not getattr(r, "ok", True):
            continue
        expected = str(r.meta.get("expected", "")).strip()
        if not expected:
            continue
        n += 1
        hit = int(expected in r.output_text)
        total_found += hit
        found_by_bucket[_bucket(float(r.meta.get("needle_depth", 0.0)), n_buckets)].append(hit)
    by_depth = {b: (sum(v) / len(v) if v else 0.0) for b, v in sorted(found_by_bucket.items())}
    return NeedleScore(retrieval_rate=(total_found / n) if n else 0.0, by_depth=by_depth, n=n)
