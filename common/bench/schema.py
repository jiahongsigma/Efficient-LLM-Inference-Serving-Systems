"""Core record types for the lab harness.

These dataclasses are the *only* place request/result shapes are defined; every
driver and metric consumes them. Times on ``Result`` are seconds measured from
the run's ``t0`` (the moment the driver started), so they compose across
requests issued at different wall-clock moments.

INVARIANT 1 (coordinated-omission-correct latency) lives here as the ``e2e``
and ``ttft`` properties: both are measured from ``intended_send_time``, never
from ``actual_send_time``. See ``common/bench/metrics.py`` and Module 12 §12.4.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Request:
    """One OpenAI-style chat request, pre-stamped with when it *should* be sent."""

    id: str
    messages: list[dict]
    max_tokens: int = 128
    intended_send_time: float = 0.0  # seconds from run start — SET BY THE SCHEDULE
    meta: dict = field(default_factory=dict)  # prefix_group, needle_depth, task, expected, ...


@dataclass
class Result:
    """The outcome of one request. All *_time fields are seconds from run start."""

    id: str
    intended_send_time: float
    actual_send_time: float
    first_token_time: float | None
    end_time: float
    prompt_tokens: int
    completion_tokens: int
    cached_prefix_tokens: int = 0  # tokens served from prefix cache (engine usage)
    status: str = "ok"  # "ok" | "error" | "timeout" | "dropped"
    output_text: str = ""
    meta: dict = field(default_factory=dict)

    # --- INVARIANT 1: latency is measured from intended_send_time ---
    @property
    def e2e(self) -> float:
        """End-to-end latency. Counts the full queueing wait of a late request."""
        return self.end_time - self.intended_send_time

    @property
    def ttft(self) -> float | None:
        """Time to first token, from intended_send_time. None if no token arrived."""
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.intended_send_time

    @property
    def tpot(self) -> float | None:
        """Mean inter-token latency. None if fewer than 2 output tokens."""
        if self.first_token_time is None or self.completion_tokens <= 1:
            return None
        return (self.end_time - self.first_token_time) / (self.completion_tokens - 1)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@dataclass
class TurnRecord:
    """One turn of a multi-turn agentic session (Module 10)."""

    turn: int
    prefill_tokens: int  # tokens actually prefilled this turn (post-cache)
    cached_prefix_tokens: int  # tokens served from cache this turn
    decode_tokens: int
    model_time: float  # prefill + decode for this turn
    tool_time: float  # simulated tool execution latency
    cache_hit: bool  # did this turn's prefix hit the cache
    # Split of model_time, derived from measured TTFT (prefill) vs the rest (decode).
    # Additive to the spec so compute_session_metrics can report the model/tool/
    # re-prefill breakdown of Module 10 §10.4 without a second cost model.
    prefill_time: float = 0.0
    decode_time: float = 0.0


@dataclass
class SessionResult:
    """Aggregate of one agentic session (Module 10)."""

    session_id: str
    turns: list[TurnRecord]
    e2e_latency: float  # Σ (model_time + tool_time) over turns
    total_model_time: float
    total_tool_time: float
    total_reprefill_tokens: int  # Σ prefill_tokens (post-cache) over turns
    cache_hit_rate: float  # cached_prefix_tokens / total prompt tokens over the session
    slot_held_time: float  # wall-time the KV slot was occupied (depends on kv_policy)
