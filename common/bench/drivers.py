"""Load drivers: open-loop, closed-loop, and the agentic session loop.

INVARIANT 2: ``run_open_loop`` is the default for any latency/goodput claim. It
issues each request at its ``intended_send_time`` regardless of whether earlier
requests have finished, so a saturated server's queue actually builds and the
true tail/knee appear. ``run_closed_loop`` is the self-throttling contrast used
only for the Module 12 coordinated-omission demonstration.
"""

from __future__ import annotations

import asyncio
import collections

import numpy as np

from .client import stream_chat
from .faults import FaultSpec, inject
from .schema import Request, Result, SessionResult, TurnRecord


async def run_open_loop(
    requests: list[Request],
    endpoint,
    *,
    schedule=None,
    seed: int = 0,
    fault: FaultSpec | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> list[Result]:
    """Issue every request at its scheduled time, no matter the server state.

    ``schedule`` may be ``None`` (use each request's existing
    ``intended_send_time``, as set by the traffic builder), a number (override
    with a fresh Poisson schedule at that rate), or ``"trace"`` (use existing).
    ``fault`` (Module 11) is applied at ``fault.at_time`` mid-run.
    """
    loop = asyncio.get_event_loop()
    if isinstance(schedule, (int, float)) and not isinstance(schedule, bool):
        from ..traffic.schedules import poisson_schedule

        times = poisson_schedule(len(requests), float(schedule), seed)
        for r, t in zip(requests, times):
            r.intended_send_time = t

    t0 = loop.time()

    async def _one(req: Request) -> Result:
        delay = req.intended_send_time - (loop.time() - t0)
        if delay > 0:
            await asyncio.sleep(delay)
        return await stream_chat(endpoint, req, api_key=api_key, timeout=timeout, t0=t0)

    tasks = [loop.create_task(_one(r)) for r in requests]

    if fault is not None:
        async def _fault_task() -> None:
            await asyncio.sleep(max(0.0, fault.at_time))
            await inject(fault, endpoint)

        loop.create_task(_fault_task())

    results = await asyncio.gather(*tasks)
    return list(results)


async def run_closed_loop(
    requests: list[Request],
    endpoint,
    *,
    concurrency: int,
    seed: int = 0,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> list[Result]:
    """Fixed pool of ``concurrency`` workers; each sends its next only after the
    previous returns. Self-throttling: it can never offer more load than the
    server serves, so it hides the knee and under-reports the tail (Module 12).
    """
    loop = asyncio.get_event_loop()
    t0 = loop.time()
    queue: collections.deque[Request] = collections.deque(requests)
    results: list[Result] = []
    qlock = asyncio.Lock()

    async def worker() -> None:
        while True:
            async with qlock:
                if not queue:
                    return
                req = queue.popleft()
            # closed-loop measures *service time*: there is no schedule, so the
            # "intended" send is the actual send. This is exactly why its tail is
            # optimistic — the omission is coordinated.
            req.intended_send_time = loop.time() - t0
            res = await stream_chat(endpoint, req, api_key=api_key, timeout=timeout, t0=t0)
            results.append(res)

    await asyncio.gather(*[worker() for _ in range(concurrency)])
    return results


async def run_agentic_session(
    session,
    endpoint,
    *,
    kv_policy: str = "hold",
    seed: int = 0,
    api_key: str | None = None,
    timeout: float = 60.0,
) -> SessionResult:
    """Run one multi-turn agent: per turn, generate → (structured) tool call →
    sleep the simulated tool latency → resume on the grown context (Module 10).

    ``kv_policy``:
      * ``"hold"``  — the session's KV occupies its slot through the tool wait;
        ``slot_held_time`` includes tool time (low duty cycle, §10.3).
      * ``"offload"`` — KV is freed during the wait and restored on resume;
        ``slot_held_time`` excludes tool time but adds a per-turn restore cost.

    The model/decode/prefill split per turn is derived from the *measured*
    timings (TTFT = prefill, the rest = decode), so it is backend-agnostic.
    """
    loop = asyncio.get_event_loop()
    t0 = loop.time()
    rng = np.random.default_rng((seed * 1_000_003) ^ (abs(hash(session.session_id)) & 0xFFFFFFFF))
    restore = float(getattr(session, "restore_cost", 0.005))

    turns: list[TurnRecord] = []
    total_prompt = 0
    total_cached = 0

    for ti in range(session.turns):
        req = Request(
            id=f"{session.session_id}:t{ti}",
            messages=session.turn_messages[ti],
            max_tokens=session.decode_tokens,
            meta={"sim_output": session.assistant_outputs[ti]},
        )
        res = await stream_chat(endpoint, req, api_key=api_key, timeout=timeout, t0=t0)

        model_time = res.end_time - res.actual_send_time
        if res.first_token_time is not None:
            prefill_time = res.first_token_time - res.actual_send_time
            decode_time = res.end_time - res.first_token_time
        else:
            prefill_time, decode_time = model_time, 0.0

        tool_time = max(0.0, session.tool_latency + rng.normal(0.0, session.tool_latency_jitter))
        await asyncio.sleep(tool_time)  # the tool runs in wall-clock either way

        prefill_tokens = max(0, res.prompt_tokens - res.cached_prefix_tokens)
        turns.append(
            TurnRecord(
                turn=ti,
                prefill_tokens=prefill_tokens,
                cached_prefix_tokens=res.cached_prefix_tokens,
                decode_tokens=res.completion_tokens,
                model_time=model_time,
                tool_time=tool_time,
                cache_hit=res.cached_prefix_tokens > 0,
                prefill_time=prefill_time,
                decode_time=decode_time,
            )
        )
        total_prompt += res.prompt_tokens
        total_cached += res.cached_prefix_tokens

    e2e = sum(t.model_time + t.tool_time for t in turns)
    if kv_policy == "hold":
        slot_held = sum(t.model_time + t.tool_time for t in turns)
    elif kv_policy == "offload":
        slot_held = sum(t.model_time + restore for t in turns)
    else:
        raise ValueError(f"unknown kv_policy {kv_policy!r}")

    return SessionResult(
        session_id=session.session_id,
        turns=turns,
        e2e_latency=e2e,
        total_model_time=sum(t.model_time for t in turns),
        total_tool_time=sum(t.tool_time for t in turns),
        total_reprefill_tokens=sum(t.prefill_tokens for t in turns),
        cache_hit_rate=(total_cached / total_prompt) if total_prompt else 0.0,
        slot_held_time=slot_held,
    )
