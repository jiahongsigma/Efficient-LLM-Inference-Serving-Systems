"""Multi-turn agentic sessions (Module 10).

Each session is a ``T``-turn tool-calling loop. Per turn the context grows by
~``increment_tokens`` (the model's tool call + the tool's result), and the tool
takes a simulated ``tool_latency``. The decisive knob is ``prefix_stable``:

* ``True``  — the prefix is strictly append-only and byte-identical turn-to-turn,
  so every turn after the first hits the cross-turn cache (re-prefill ≈ Δ).
* ``False`` — a per-turn timestamp is injected into the *system head*, so the
  leading tokens change every turn and the whole prefix silently misses
  (re-prefill ≈ full context) — Module 10 §10.2.

``run_agentic_session`` consumes the precomputed ``turn_messages`` /
``assistant_outputs`` so the driver stays oblivious to the perturbation logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ._synth import make_text


@dataclass
class Session:
    session_id: str
    turn_messages: list[list[dict]]  # full context presented at each turn
    assistant_outputs: list[str]  # what the model "generates" each turn (the tool call)
    decode_tokens: int
    tool_latency: float
    tool_latency_jitter: float = 0.0
    prefix_stable: bool = True
    restore_cost: float = 0.005  # KV restore cost per turn under kv_policy="offload"
    meta: dict = field(default_factory=dict)

    @property
    def turns(self) -> int:
        return len(self.turn_messages)


def build_agentic_sessions(
    n_sessions: int,
    *,
    turns: int,
    increment_tokens: int,
    system_tokens: int,
    tool_latency: float,
    tool_latency_jitter: float = 0.0,
    prefix_stable: bool = True,
    seed: int = 0,
    decode_tokens: int = 32,
) -> list[Session]:
    """Build ``n_sessions`` sessions. ``decode_tokens`` (the per-turn model output,
    additive to the spec signature) is part of the per-turn growth, mirroring
    §10.5 where Δ = model output + tool result."""
    tool_tokens = max(1, increment_tokens - decode_tokens)
    sessions: list[Session] = []

    for s in range(n_sessions):
        base_head = make_text(system_tokens, prefix="SYSTEM tools=[read,write,run]")
        user0 = make_text(16, prefix=f"sess{s}-task")
        # per-turn tool call (decode) and tool result, fixed across turns so the
        # only thing that changes the prefix is the (optional) injected timestamp
        assistant = [make_text(decode_tokens, prefix=f"call{j}") for j in range(turns)]
        tool_result = [make_text(tool_tokens, prefix=f"result{j}") for j in range(turns)]

        turn_messages: list[list[dict]] = []
        for ti in range(turns):
            head = base_head if prefix_stable else f"{base_head} [ts=s{s}t{ti}]"
            ctx: list[dict] = [{"role": "system", "content": head},
                               {"role": "user", "content": user0}]
            for j in range(ti):  # prior turns: model output + tool result
                ctx.append({"role": "assistant", "content": assistant[j]})
                ctx.append({"role": "tool", "content": tool_result[j]})
            turn_messages.append(ctx)

        sessions.append(Session(
            session_id=f"agent-{s}",
            turn_messages=turn_messages,
            assistant_outputs=assistant,
            decode_tokens=decode_tokens,
            tool_latency=tool_latency,
            tool_latency_jitter=tool_latency_jitter,
            prefix_stable=prefix_stable,
            meta={"increment_tokens": increment_tokens, "system_tokens": system_tokens},
        ))
    return sessions
