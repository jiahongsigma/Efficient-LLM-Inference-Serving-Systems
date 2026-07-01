# Module 11 — Frameworks, the API Layer, and Resilience

*Prerequisites: Modules 1–8 (the engine *capabilities* — batching, paging, quantization, parallelism, long-context, speculation, disaggregation). A capability is not a service. This module is the operational layer that packages those capabilities into something dependable: the engines, the interface that exposes them, the gateway in front, the observability that lets you measure them (feeding Module 12), and the resilience that keeps them up. It is where the algorithms meet operations — and where production systems actually fail.* 

---

## In plain English

**Why this matters.** A fast engine is not a dependable service. The day your backend crashes mid-response, or traffic spikes past capacity, is the day you learn that the operations layer — not the algorithms — is what keeps you online. For actually self-hosting, this is the most important module.

**What this module gives you.** The engines (vLLM, SGLang), the standard API they all expose, a small readable gateway to put in front (routing, failover, authentication, metrics), and how to survive failures gracefully instead of going dark.

**How it works (the intuition).** Put a thin, auditable front door over your engines that watches every request, reroutes when one dies, and politely turns away excess load instead of letting everything grind to a halt. Then you deliberately break it on purpose to measure how bad a real failure actually feels to users.

---

## 11.1 A capability is not a service

Modules 1–8 made a forward pass fast and a model fit. But "vLLM can serve a quantized 70B at high throughput" is not yet a service you can depend on. Five operational layers stand between a capable engine and production:

1. **package** the capabilities into an engine (vLLM / SGLang);
2. **expose** them through a stable interface (OpenAI-compatible);
3. **front** them with a gateway (routing, failover, auth, telemetry);
4. **observe** them (the metrics Module 12 will measure);
5. **make them dependable** (resilience, graceful degradation).

These concerns are under-taught relative to the algorithms, and they are where outages come from. This module is each layer.

---

## 11.2 The engines, architecturally

The two engines (full notes: `Appendix C — The Modern Serving Stack.md`) differ in what they optimize:

- **vLLM** — the throughput-optimized general engine: PagedAttention (Module 5) + continuous batching (Module 4), broad quantization and parallelism support. Its 2025 **V1 re-architecture** rebuilt the scheduler and execution loop for lower overhead and cleaner async.
- **SGLang** — optimized for prefix sharing (RadixAttention, Module 5) and structured/constrained generation, with a programming frontend.

Both implement the Modules 1–8 techniques; the choice is workload-driven (prefix-heavy / structured → SGLang; general throughput → vLLM), and both sit behind the same interface, which is what makes them swappable.

---

## 11.3 The OpenAI-compatible interface — the integration contract

The field converged on the OpenAI `/v1/chat/completions` schema (with SSE streaming, tool calling, JSON mode, logprobs) as the **de-facto standard**. Its value is decoupling: the application talks to a stable contract, and you can swap vLLM ↔ SGLang ↔ a hosted API **without changing client code**. This contract is precisely what makes the gateway pattern (§11.4) possible — a uniform front door over heterogeneous backends.

The catch, which the gateway must handle: **not every backend supports every feature** of the schema (tool calling, JSON mode, vision, `n>1`). The interface being uniform does not make the *capabilities* uniform — hence capability parity (§11.4).

---

## 11.4 The gateway — the operational front door

You put a thin **gateway** in front of the engine(s) to handle what the engine does not:

- **routing** across replicas and models;
- **failover** (a primary→fallback chain);
- **auth and rate limiting** (who may call, how often);
- **telemetry** (the per-request metrics Module 12 needs);
- **capability parity** — refuse to silently route a tool-calling request to a backend that cannot do tools (§11.3); declare each backend's capabilities and reject rather than degrade.

The design principle is **minimal and auditable**: a few hundred lines, explicit backend allowlist, **no payload logging** (security/privacy), hard timeouts, constant-time key checks. A sprawling third-party gateway trades a small attack surface for a large one; a gateway you can read end-to-end is the safer choice. (This is the gateway you build in the lab; the same pattern recurs whenever you need a provider-agnostic, failover-capable front door.)

---

## 11.5 Observability — you cannot operate or measure what you cannot see

The gateway must log, **per request, metadata only** (never payloads): route, backend, prompt/completion tokens, **TTFT, TPOT**, total latency, status. Streamed responses must **accumulate usage**, since the SSE stream does not carry token counts by default. This telemetry is the raw material for Module 12's measurement and economics — **Module 11 produces the metrics Module 12 analyzes** (a deliberate dependency).

Engine-level signals matter too: **KV-cache utilization, running batch size, queue depth, preemption rate** (Modules 4–5). These tell you *which wall you are hitting* — a high preemption rate means KV capacity is the bottleneck (Module 5); a deep queue at low GPU utilization means you are latency-bound, not throughput-bound (Module 4). Without these, you are tuning blind.

---

## 11.6 Resilience and graceful degradation

Production failure modes: a backend crashes; a backend is slow (timeout); a backend returns errors; the GPU OOMs; a request exceeds limits; **load exceeds capacity**. The responses:

- **Failover** — route to another backend (a primary→fallback chain).
- **Load shedding / admission control** — this is **Module 4's knee operationalized**: past the saturation knee, *reject or queue* excess load rather than admit it, so the requests you *do* serve stay fast (goodput protected) instead of everything slowing together. Shedding load is how you defend goodput under overload.
- **Graceful degradation** — fall back to a smaller/faster model or drop optional features under pressure.
- **Circuit breakers and backoff** — stop hammering a failed backend; retry with backoff.

The honest part, and the lab's centerpiece: **failover is not seamless.** When a backend dies, its **in-flight requests are lost** (streamed responses are cut mid-token), and there is a **latency spike** while the gateway detects the failure and reroutes. You must *measure* this user-visible gap — and measure it correctly: it is a **tail-latency event under load**, so an average-only or closed-loop measurement (Module 4) would hide exactly the thing you are trying to quantify.

---

## 11.7 Where the frontier is now

Single-engine serving behind a simple gateway is the baseline; production has moved to **fleet orchestration**:

- **Orchestration layers** above the engines — **NVIDIA Dynamo** (disaggregated-serving orchestration, 2025), **llm-d** (Kubernetes-native distributed inference, 2025), vLLM Production Stack, Ray Serve, KServe — manage many replicas, autoscaling, and disaggregated prefill/decode pools (Module 8).
- **KV-aware / prefix-aware routing** — route a request to the replica whose cache **already holds its prefix** (Modules 5, 7), turning the router into a cache-aware scheduler (Mooncake-style). This couples the routing layer to the KV cache, which the naive round-robin gateway ignores.
- **Disaggregation orchestration** — managing separate prefill and decode pools (Module 8) with KV transfer between them.

The gateway is evolving from a dumb proxy into a **KV-aware, disaggregation-aware fleet router** — teaching only "vLLM behind a proxy" would miss it.

---

## 11.8 The picture to carry forward

- **A capability is not a service**: package (engine) → expose (OpenAI interface) → front (gateway) → observe → make dependable.
- The **OpenAI-compatible interface** decouples app from engine and enables the gateway; the gateway adds routing, **failover**, auth, telemetry, and **capability parity**, kept minimal and auditable.
- **Observability is mandatory** — metadata-only per-request metrics (feeding Module 12) plus engine signals (which wall you hit).
- **Resilience** = failover + **load shedding (Module 4's knee)** + graceful degradation; **failover is not seamless**, and its gap must be measured honestly (open-loop, tail).
- The frontier is **fleet orchestration with KV-aware routing** (Dynamo, llm-d).

---

## Going Deeper (appendix) — scheduler internals

The engine's scheduler is the continuous-batching loop (Module 4) plus three policies:

- **Admission control** — decide whether to start a waiting request now or hold it: starting a request reserves KV (Module 5) and may force preemption; under load you queue rather than admit (the load-shedding decision, §11.6).
- **Preemption / swapping** (Module 5) — when a running sequence needs a block and none is free, preempt a victim by *swapping* its KV to host memory or *recomputing* it on resumption — the cost trade depends on sequence length.
- **Priority and fairness** — production schedulers support request priorities (interactive vs batch) and fairness across tenants, which interacts with admission and preemption: a high-priority arrival may preempt a low-priority running request.

These three turn the bare batching loop of Module 4 into a scheduler that behaves sanely under contention — the operational complement to Module 4's throughput story.

---

## Lab 11 — Build the gateway, then break it on purpose

**Context — what this builds on (checklist).** *Reuse:* the `common/bench` **open-loop** generator and its metric definitions (Modules 4, 12) — the gateway emits these; the capability-parity concept (tool-call support, Modules 9–10). *Test the new insight:* the operational layer (gateway + observability + resilience), with the **failover gap** as the centerpiece — not a restated throughput run. *Exercise prior quantities:* TTFT/TPOT (Modules 1, 4), preemption signals (Module 5), the knee (Module 4). *Frontier:* KV-aware routing. *Consistency:* measure failover under **open-loop** load and report the **tail during the failure**, not the average.

1. **Build the gateway.** Stand up **vLLM and SGLang** behind a minimal (~150-line) OpenAI-compatible gateway with: a failover chain, auth, a **capability-parity guard** (Modules 9–10 — reject a tool-call request a backend can't serve, don't degrade), and **telemetry middleware** logging metadata-only TTFT/TPOT/tokens/status (accumulating usage on streams). Verify a request routes and is logged. *Confirms: §11.4–11.5; reuses the Module 12 metrics.*

2. **Resilience — the failover gap (centerpiece).** Under open-loop load, **kill a backend mid-stream**. Measure: in-flight requests lost (streams cut), the **tail-latency spike** during detect-and-reroute, and recovery time. Report the **p99 during the failure window**, not the average — show why the average hides it. *Confirms: §11.6 — failover is not seamless; measure honestly.*

3. **Load shedding (Module 4's knee, operationalized).** Drive load past the knee; compare (a) admitting everything (server collapses, goodput craters) vs (b) admission control / load shedding (reject excess); show shedding keeps served-request latency low and **protects goodput**. *Confirms: §11.6 — goodput as an operational policy.*

4. **(Frontier) KV-aware routing.** On a shared-prefix workload (Modules 5, 7), compare round-robin routing across replicas vs **prefix-aware routing** (route to the replica holding the cached prefix); show the prefix-aware router lands cache hits and lower TTFT. Note the orchestration layers (Dynamo, llm-d) that do this at scale. *Confirms: §11.7.*

**Deliverable:** the gateway (committed to `m11_serving_gateway/`) **and** a `resilience.md` quantifying each failure mode — the failover gap, in-flight loss, recovery time — measured under open-loop load and reported on the **tail**; plus the load-shedding goodput comparison and the KV-aware routing TTFT win. **Mastery test — defend in one sentence each:** *why a capability is not a service; why the OpenAI-compatible interface enables the gateway pattern; what the gateway must do and why telemetry is metadata-only; why failover is not seamless and how to measure its gap honestly (open-loop, tail, in-flight loss); how load shedding protects goodput; and what KV-aware routing does.* *Feeds:* Module 12 (the telemetry this gateway emits is what Module 12 measures and where it teaches you not to lie).

**Reading:** the vLLM (incl. the V1 re-architecture) and SGLang documentation and papers. Current frontier: *NVIDIA Dynamo* and *llm-d* (2025) for orchestration; Mooncake (2024, Module 5) for KV-aware routing.