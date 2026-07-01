"""Long-document QA — a controllable *shared* prefix.

A fixed document is the shared prefix; many questions follow. ``prefix_share_ratio``
is the swept knob (Module 5): the fraction of questions that reuse the shared doc
(the rest get a unique doc, so they cannot hit the cache). This is the positive
control that, paired with ``build_sharegpt`` (the null), satisfies INVARIANT 4.
Also the long-context base for Module 7.
"""

from __future__ import annotations

import numpy as np

from ..bench.schema import Request
from ._synth import make_text
from .schedules import poisson_schedule, trace_schedule


def build_long_doc_qa(
    n_questions: int,
    *,
    doc=None,
    doc_tokens: int = 2000,
    prefix_share_ratio: float = 1.0,
    rate_or_trace=1.0,
    seed: int = 0,
) -> list[Request]:
    rng = np.random.default_rng(seed)
    shared_doc = doc if doc is not None else make_text(doc_tokens, prefix="DOCUMENT")
    times = _times(n_questions, rate_or_trace, seed)

    reqs = []
    for i in range(n_questions):
        shares = rng.random() < prefix_share_ratio
        if shares:
            doc_text, group = shared_doc, "doc"
        else:
            doc_text, group = make_text(doc_tokens, prefix=f"DOC{i}u{rng.integers(1_000_000)}"), f"doc-{i}"
        question = make_text(24, prefix=f"q{i}")
        reqs.append(Request(
            id=f"ldq-{i}",
            messages=[{"role": "system", "content": doc_text}, {"role": "user", "content": question}],
            max_tokens=64, intended_send_time=times[i],
            meta={"prefix_group": group, "task": "long_doc_qa", "shares_prefix": shares},
        ))
    return reqs


def _times(n, rate_or_trace, seed):
    if isinstance(rate_or_trace, (int, float)) and not isinstance(rate_or_trace, bool):
        return poisson_schedule(n, float(rate_or_trace), seed)
    if isinstance(rate_or_trace, str):
        return trace_schedule(rate_or_trace)[:n]
    if isinstance(rate_or_trace, (list, tuple)):
        return list(rate_or_trace)[:n]
    raise TypeError("rate_or_trace must be a rate, a trace path, or a list of times")
