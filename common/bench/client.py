"""The OpenAI-compatible streaming client and its simulated twin.

Everything talks to a backend through the ``Endpoint.stream`` async generator,
which yields ``StreamEvent``s (token deltas, then one usage event). Two
implementations:

* ``OpenAIEndpoint`` — real ``/v1/chat/completions`` over httpx with SSE
  streaming; used by the labs against vLLM / SGLang / the gateway.
* ``SimEndpoint`` — a deterministic in-process simulator that models a prefix
  cache, a concurrency limit (so open-loop overload actually queues — the
  knee), and fault injection. It lets the whole harness be unit-tested with no
  GPU, and lets a lab be dry-run before metering a real accelerator.

``stream_chat`` is identical for both: it times the first token (→ TTFT) and the
stream end (→ TPOT/E2E), accumulates usage (including ``cached_prefix_tokens``),
and turns any timeout/error into a counted ``Result`` rather than an exception.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol, runtime_checkable

from .faults import FaultError, FaultSpec
from .schema import Request, Result


def approx_tokens(text: str) -> int:
    """~4 chars/token (Module 0). Good enough for sizing synthetic workloads."""
    return max(1, len(text) // 4)


@dataclass
class StreamEvent:
    kind: str  # "token" | "usage"
    text: str = ""
    usage: dict = field(default_factory=dict)


@runtime_checkable
class Endpoint(Protocol):
    backend_id: str

    def stream(
        self, request: Request, *, api_key: str | None = None, timeout: float = 30.0
    ) -> AsyncIterator[StreamEvent]: ...


# --------------------------------------------------------------------------- #
# The timing/Result logic — shared by every endpoint.
# --------------------------------------------------------------------------- #
async def stream_chat(
    endpoint: Endpoint,
    request: Request,
    *,
    api_key: str | None = None,
    timeout: float = 30.0,
    t0: float | None = None,
) -> Result:
    """Drive one request to completion and return a fully-timed ``Result``.

    ``t0`` is the run clock origin (``loop.time()`` at run start) so all times in
    the returned Result are seconds-from-run-start and comparable across
    requests. A timeout or backend error still yields a Result (with ``status``
    set and ``end_time`` recorded) so it counts toward the tail.
    """
    loop = asyncio.get_event_loop()
    if t0 is None:
        t0 = loop.time()
    actual_send = loop.time() - t0
    first: float | None = None
    chunks: list[str] = []
    streamed = 0
    usage: dict = {}
    status = "ok"
    try:
        async with asyncio.timeout(timeout):
            async for ev in endpoint.stream(request, api_key=api_key, timeout=timeout):
                if ev.kind == "token":
                    if first is None:
                        first = loop.time() - t0
                    chunks.append(ev.text)
                    streamed += 1
                elif ev.kind == "usage":
                    usage = ev.usage
    except (asyncio.TimeoutError, TimeoutError):
        status = "timeout"
    except FaultError:
        status = "error"
    except Exception:
        status = "error"
    end = loop.time() - t0

    prompt_tokens = int(usage.get("prompt_tokens", 0))
    cached = int(usage.get("cached_prefix_tokens", 0))
    completion = int(usage.get("completion_tokens", streamed))
    return Result(
        id=request.id,
        intended_send_time=request.intended_send_time,
        actual_send_time=actual_send,
        first_token_time=first,
        end_time=end,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion,
        cached_prefix_tokens=cached,
        status=status,
        output_text="".join(chunks),
        meta=request.meta,
    )


# --------------------------------------------------------------------------- #
# Real backend: OpenAI-compatible SSE streaming.
# --------------------------------------------------------------------------- #
class OpenAIEndpoint:
    """A real ``/v1/chat/completions`` backend (vLLM, SGLang, or a gateway)."""

    def __init__(self, base_url: str, model: str, *, backend_id: str | None = None, transport=None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.backend_id = backend_id or base_url
        self._transport = transport  # for tests: inject an httpx.MockTransport

    async def stream(
        self, request: Request, *, api_key: str | None = None, timeout: float = 30.0
    ) -> AsyncIterator[StreamEvent]:
        import httpx  # imported lazily so the sim path needs no network stack

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": self.model,
            "messages": request.messages,
            "max_tokens": request.max_tokens,
            "stream": True,
            # ask engines to include usage in the final SSE chunk
            "stream_options": {"include_usage": True},
            **request.meta.get("extra_body", {}),
        }
        async with httpx.AsyncClient(timeout=timeout, transport=self._transport) as client:
            async with client.stream(
                "POST", f"{self.base_url}/v1/chat/completions", json=payload, headers=headers
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    obj = json.loads(data)
                    choices = obj.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta", {})
                        piece = delta.get("content")
                        if piece:
                            yield StreamEvent("token", text=piece)
                    if obj.get("usage"):
                        u = obj["usage"]
                        details = u.get("prompt_tokens_details") or {}
                        yield StreamEvent(
                            "usage",
                            usage={
                                "prompt_tokens": u.get("prompt_tokens", 0),
                                "completion_tokens": u.get("completion_tokens", 0),
                                "cached_prefix_tokens": details.get("cached_tokens", 0),
                            },
                        )


# --------------------------------------------------------------------------- #
# Simulated backend: deterministic, cache-aware, concurrency-limited, faultable.
# --------------------------------------------------------------------------- #
class SimEndpoint:
    """In-process simulator. Not a model — a *cost model* for testing the harness.

    Models the three things the invariants depend on:
      * a **prefix cache** keyed on the leading messages (so M5/M10 cache-hit
        accounting is exercised),
      * a **concurrency limit** (a semaphore): requests beyond ``max_concurrency``
        queue, and the wait is real awaited time, so open-loop overload reveals
        the knee while closed-loop self-throttles (INV 2),
      * **faults**: ``apply_fault`` kills/slows a backend; a request being served
        on a killed backend raises ``FaultError`` mid-stream (INV 1 corollary).

    All durations are tiny by default so test suites run in well under a second.
    """

    def __init__(
        self,
        *,
        base_ms: float = 0.5,
        prefill_ms_per_tok: float = 0.01,
        decode_ms_per_tok: float = 0.5,
        max_concurrency: int = 8,
        backends: tuple[str, ...] = ("sim-0",),
        failover_penalty_ms: float = 20.0,
        enable_prefix_cache: bool = True,
        nondeterministic: bool = False,
        gen_token_cap: int | None = None,
        model_prefill_interference: bool = False,
        chunked_prefill: bool = True,
        prefill_chunk_tokens: int = 512,
        seed: int = 0,
    ):
        self.base_ms = base_ms
        self.prefill_ms_per_tok = prefill_ms_per_tok
        self.decode_ms_per_tok = decode_ms_per_tok
        self.backends = list(backends)
        self.backend_id = backends[0]
        self.failover_penalty_ms = failover_penalty_ms
        self.enable_prefix_cache = enable_prefix_cache
        self.nondeterministic = nondeterministic
        self.gen_token_cap = gen_token_cap
        self.model_prefill_interference = model_prefill_interference
        self.chunked_prefill = chunked_prefill
        self.prefill_chunk_tokens = prefill_chunk_tokens
        self._seed = seed
        self._sema = asyncio.Semaphore(max_concurrency)
        self._gpu_free = asyncio.Event()  # clear == a non-chunked prefill is hogging the GPU
        self._gpu_free.set()
        self._cache: dict[int, int] = {}  # prefix-hash -> cumulative tokens
        self._dead: set[str] = set()
        self._slow: dict[str, float] = {}  # backend -> added seconds
        self._rr = 0

    # -- fault control -------------------------------------------------------
    def apply_fault(self, fault: FaultSpec) -> None:
        if fault.kind == "kill_backend":
            self._dead.add(fault.target)
        elif fault.kind == "slow_backend":
            self._slow[fault.target] = fault.severity
        else:
            raise ValueError(f"unknown fault kind {fault.kind!r}")

    def _route(self) -> str | None:
        live = [b for b in self.backends if b not in self._dead]
        if not live:
            return None
        b = live[self._rr % len(live)]
        self._rr += 1
        return b

    # -- prefix cache --------------------------------------------------------
    def _cached_prefix_tokens(self, messages: list[dict]) -> tuple[int, list[tuple[int, int]]]:
        """Return (cached_tokens, prefix_entries_to_store) for these messages.

        Cached = the longest leading run of messages whose hash is already in the
        cache. ``prefix_entries_to_store`` are every leading-prefix (hash, cumtok)
        for this request, recorded after serving so a later identical prefix hits.
        """
        cum = 0
        cached = 0
        entries: list[tuple[int, int]] = []
        running_key = 0
        for m in messages:
            cum += approx_tokens(m.get("content", ""))
            running_key = hash((running_key, m.get("role", ""), m.get("content", "")))
            entries.append((running_key, cum))
            if self.enable_prefix_cache and running_key in self._cache:
                cached = cum  # this whole prefix is cached
        return cached, entries

    async def _do_prefill(self, prefill_s: float, prompt_tokens: int) -> None:
        """Sleep out the prefill, modelling prefill↔decode interference (Module 4 §4.5).

        A *large* prefill run as one un-chunked step starves every in-flight decode
        for its whole duration (decode stalls on ``_gpu_free``). Chunked prefill
        interleaves decode steps with the prefill chunks, so decode is not starved —
        modelled here as simply not seizing the GPU. Small prefills never starve."""
        big = self.model_prefill_interference and prompt_tokens > self.prefill_chunk_tokens
        if big and not self.chunked_prefill:
            self._gpu_free.clear()
            try:
                await asyncio.sleep(prefill_s)
            finally:
                self._gpu_free.set()
        else:
            await asyncio.sleep(prefill_s)

    async def stream(
        self, request: Request, *, api_key: str | None = None, timeout: float = 30.0
    ) -> AsyncIterator[StreamEvent]:
        backend = self._route()
        if backend is None:
            raise FaultError("all backends dead")

        async with self._sema:  # concurrency limit → queueing under overload
            # re-check liveness after (possibly) waiting in the queue
            if backend in self._dead:
                raise FaultError(f"backend {backend} died while queued")

            cached, entries = self._cached_prefix_tokens(request.messages)
            prompt_tokens = sum(approx_tokens(m.get("content", "")) for m in request.messages)
            uncached = max(0, prompt_tokens - cached)

            prefill_s = (self.base_ms + self.prefill_ms_per_tok * uncached) / 1000.0
            prefill_s += self._slow.get(backend, 0.0)
            await self._do_prefill(prefill_s, uncached)

            # a kill that lands during prefill must surface as an error
            if backend in self._dead:
                raise FaultError(f"backend {backend} killed mid-request")

            # record this request's prefixes for future cache hits
            if self.enable_prefix_cache:
                for k, tok in entries:
                    self._cache[k] = tok

            # output text: caller-supplied (for eval tests) or a default marker
            text = request.meta.get("sim_output")
            if text is None:
                text = " ".join(f"w{i}" for i in range(request.max_tokens))
            if self.nondeterministic and request.meta.get("batch_tag") is not None:
                # batch composition perturbs the result (Module 12 §12.5); prepend so
                # the perturbation survives any generation cap
                text = f"<batch={request.meta['batch_tag']}> {text}"

            words = text.split(" ") if text else []
            if self.gen_token_cap is not None:
                words = words[: self.gen_token_cap]  # keep test runs fast
            per_tok = self.decode_ms_per_tok / 1000.0
            for w in words:
                if self.model_prefill_interference:
                    await self._gpu_free.wait()  # a non-chunked prefill stalls decode here
                await asyncio.sleep(per_tok)
                if backend in self._dead:
                    raise FaultError(f"backend {backend} killed mid-stream")
                yield StreamEvent("token", text=w + " ")

            yield StreamEvent(
                "usage",
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": len(words),
                    "cached_prefix_tokens": cached,
                },
            )


class StaticBatchEndpoint:
    """Static (fixed) batching — the broken baseline of Module 4.

    Requests are admitted in groups of ``batch_size`` and the whole batch is held
    until its slowest member finishes (lockstep decode over the longest output).
    Two consequences the lab measures:

      * **head-of-line blocking** — a short request returns only when the batch's
        *longest* member is done, so its latency is inflated to the batch time;
      * **no mid-flight admission** — a request arriving while a batch runs waits
        for the entire batch to drain before it can even start.

    A single coordinator runs batches strictly one at a time; that serialization
    is what makes the tail explode under a length-variant open-loop stream, while
    continuous batching (``SimEndpoint``) holds. Same cost model as ``SimEndpoint``.
    """

    def __init__(
        self,
        *,
        batch_size: int = 16,
        max_wait_s: float = 0.05,
        base_ms: float = 0.5,
        prefill_ms_per_tok: float = 0.01,
        decode_ms_per_tok: float = 0.5,
        gen_token_cap: int | None = None,
        backend_id: str = "static",
    ):
        self.backend_id = backend_id
        self.batch_size = batch_size
        self.max_wait_s = max_wait_s
        self.base_ms = base_ms
        self.prefill_ms_per_tok = prefill_ms_per_tok
        self.decode_ms_per_tok = decode_ms_per_tok
        self.gen_token_cap = gen_token_cap
        self._queue: asyncio.Queue | None = None
        self._runner: asyncio.Task | None = None

    def _cost(self, request: Request):
        prompt = sum(approx_tokens(m.get("content", "")) for m in request.messages)
        n_out = request.max_tokens
        if self.gen_token_cap is not None:
            n_out = min(n_out, self.gen_token_cap)
        prefill_s = (self.base_ms + self.prefill_ms_per_tok * prompt) / 1000.0
        decode_s = self.decode_ms_per_tok * n_out / 1000.0
        return prompt, n_out, prefill_s, decode_s

    def _ensure_runner(self) -> None:
        loop = asyncio.get_event_loop()
        if self._queue is None:
            self._queue = asyncio.Queue()
        if self._runner is None or self._runner.done():
            self._runner = loop.create_task(self._run_loop())

    async def aclose(self) -> None:
        """Stop the batch coordinator. Call when done to avoid a dangling task."""
        if self._runner is not None and not self._runner.done():
            self._runner.cancel()
            try:
                await self._runner
            except asyncio.CancelledError:
                pass
        self._runner = None

    async def _run_loop(self) -> None:
        loop = asyncio.get_event_loop()
        q = self._queue
        try:
            while True:
                batch = [await q.get()]  # block until at least one request
                deadline = loop.time() + self.max_wait_s
                while len(batch) < self.batch_size:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    try:
                        batch.append(await asyncio.wait_for(q.get(), remaining))
                    except (asyncio.TimeoutError, TimeoutError):
                        break
                costs = [self._cost(r) for (r, _) in batch]
                batch_prefill = max(c[2] for c in costs)
                t_batch = batch_prefill + max(c[3] for c in costs)  # lockstep over slowest
                for (r, fut), c in zip(batch, costs):
                    if not fut.done():
                        fut.set_result((batch_prefill, t_batch, c[0], c[1]))
                await asyncio.sleep(t_batch)  # hold admission for the whole batch
        except asyncio.CancelledError:
            return

    async def stream(
        self, request: Request, *, api_key: str | None = None, timeout: float = 30.0
    ) -> AsyncIterator[StreamEvent]:
        self._ensure_runner()
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        await self._queue.put((request, fut))
        batch_prefill, t_batch, prompt_tokens, n_out = await fut  # wait for our batch to start
        start = loop.time()

        await asyncio.sleep(batch_prefill)  # first token only after the (padded) batch prefill
        per_tok = self.decode_ms_per_tok / 1000.0
        for i in range(n_out):
            if i > 0:
                await asyncio.sleep(per_tok)
            yield StreamEvent("token", text=f"w{i} ")
        # lockstep padding: the response is not returned until the batch finishes
        remaining = t_batch - (loop.time() - start)
        if remaining > 0:
            await asyncio.sleep(remaining)
        yield StreamEvent(
            "usage",
            usage={"prompt_tokens": prompt_tokens, "completion_tokens": n_out, "cached_prefix_tokens": 0},
        )
