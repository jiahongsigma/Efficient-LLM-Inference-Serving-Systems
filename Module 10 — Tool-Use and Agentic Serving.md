# Module 10 — Tool-Use / Agentic Serving

*Prerequisites: Module 5 (prefix caching — the lever), Module 7 (long context), Module 9 (the tool call is structured output). This module is short and deliberately concrete: agentic serving has **one** real mechanism (cross-turn prefix reuse, a corollary of Module 5) and **two** practical realities that determine whether your agent costs 1× or 20× — neither of which is a new algorithm, both of which are where real deployments fail. We skip the concept tour and go to the engineering that decides the bill.*

---

## In plain English

**Why this matters.** An agent calls a tool, reads the result, calls another — re-reading its whole growing conversation every single step. Done naively that costs ~20× more than it should, and the overspend is *silent*: no error, just a bigger bill at the end of the month.

**What this module gives you.** The one trick that keeps an agent cheap (reuse the cached conversation instead of reprocessing it each turn), the two real-world ways that trick quietly breaks, and why the model itself is usually *not* the slow part.

**How it works (the intuition).** Picture a meeting where, before every new sentence, someone re-reads the entire transcript aloud — unless you keep notes and read only the new line. Those notes are the prefix cache; anything that perturbs the transcript (an injected timestamp, reordered tool output) tears them up and you're back to re-reading everything. And while the agent waits on a slow tool, the GPU just sits there holding the seat, idle.

---

## 10.1 The loop, and where the cost is

An agent is a multi-step session: generate → emit a tool call (structured, Module 9) → pause → execute the tool → append the result → resume, repeated `T` times. Each turn re-processes the **entire accumulated context**, and prefill is the quadratic-attention cost (Modules 1, 7). So the cost question is entirely: **how much do you re-prefill each turn?**

Without caching, turn `t` re-prefills the whole context (~`t·Δ` tokens for per-turn increment `Δ`), so cumulative prefill is `Δ·Σt = O(T²)`. With prefix caching (Module 5), turn `t`'s context is turn `t−1`'s plus `Δ`, so you re-prefill only `Δ` and reuse the cached rest: cumulative `O(T)`. The ratio is `~(T+1)/2` — a 40-turn agent re-prefills **~20× less** with caching. That single factor is the whole economic story, and it is just Module 5 applied to turns. Everything that follows is about *whether you actually get that 20×*.

---

## 10.2 Reality #1: cache hits are fragile and fail silently

The `O(T²)→O(T)` win requires the cached prefix to be **byte-identical** turn-to-turn — the engine keys the cache on the exact token prefix (Module 5). In production this breaks constantly, and **silently**:

- **Non-deterministic tool output** — a tool returns JSON with unstable key order, varying float formatting, or an embedded timestamp; the appended text differs run-to-run, so next turn's prefix changes.
- **Injected per-turn metadata** — the framework prepends the current time, a request ID, or a turn counter into the context.
- **History rewriting** — agents summarize, truncate, or reorder prior turns to manage context length (Module 7); any rewrite changes the prefix.
- **System-prompt drift** — an A/B test or a dynamically assembled system prompt changes the very first tokens, invalidating *every* downstream cache entry.

Any of these flips a cache hit to a miss, and the turn silently falls back to the `O(t·Δ)` full re-prefill. There is **no error** — just a cost and latency blowup discovered on the bill or a p99 graph. The practical discipline: **stable, strictly append-only prefixes; deterministic tool-result serialization; the system prompt fixed for the session's life; and cache-hit-rate tracked as a first-class metric** (Module 9's lesson that a thing can succeed on its surface metric while failing on what matters, applied to cost). A high-quality agent stack is, in large part, the discipline of not perturbing the prefix.

---

## 10.3 Reality #2: tool latency gates the GPU, not model speed

When the server pauses for a tool, the session's KV cache sits in GPU memory while the model does nothing. So if you **hold** the KV through the tool call, a session occupies its slot for `T_gen + T_tool` per turn. With a slow tool this is brutal:

```
GPU-busy fraction ≈ T_gen / (T_gen + T_tool)
```

A 0.5 s generation followed by a 5 s API call runs the GPU slot at **~9% GPU-busy** — the other 91% is the slot holding *idle* KV waiting on the network (decode keeps the GPU *occupied* but, per Module 1, barely *uses* it). With `S` slots that fit the KV, throughput is `~S/(T_gen+T_tool)` sessions/s, **gated by tool latency**, not by anything in Modules 1–9. Optimizing the model's TPOT here is nearly pointless.

The concrete decisions: **offload the paused session's KV** to free the slot for another session during the tool call (Module 5's swap/recompute; pay a restore cost on resume), or **make tools fast/async** so `T_tool` stops dominating. Which one depends on `T_tool`, concurrency, and the restore cost — a capacity calculation you do, not a concept you cite.

---

## 10.4 The model is usually not the bottleneck

Putting §10.2 and §10.3 together: end-to-end agent latency is `Σ_turns (model_generation + tool_execution + re_prefill)`, and **tool execution and re-prefill routinely dominate the model's decode**. You can halve the model's TPOT and barely move the agent's latency or cost if 80% of it is tool time and cache-missed re-prefill. So measuring agent serving by model TTFT/TPOT (Modules 1, 8) **misrepresents the workload** — you must measure **end-to-end session latency and cost, decomposed into model / tool / re-prefill, with the cross-turn cache-hit rate alongside**. This is the only honest scorecard for an agent (Module 12).

The tool call itself must be schema-valid, which is constrained decoding (Module 9) — free sampling produces mostly-valid calls, and one malformed call breaks the loop; reason-then-constrain applies. That is the one piece of new machinery agents add, and Module 9 already covered it.

*(The productized forms of all this — prompt-caching APIs, KV offload, disaggregated re-prefill pools — are just §10.2–10.3 packaged. They help only to the extent you respect the prefix-stability and tool-latency engineering above; buying the feature does not buy the discipline.)*

---

## 10.5 A worked example: one 15-turn coding agent

Numbers make the impacts concrete. Take a coding agent on **Llama-3.1-8B / H100** (~50% MFU → ~495 effective TFLOP/s), illustrative but realistic:

- **System prompt + tool definitions:** 1,500 tokens (the fixed head).
- **Per turn:** the model generates ~150 tokens (reasoning + a tool call), the tool returns ~450 tokens (file contents, test output) → **increment `Δ` ≈ 600 tokens/turn**.
- **Turns:** `T = 15`. **Final context:** `1,500 + 14·600 = 9,900` tokens.
- **Timings:** generation `T_gen ≈ 1 s/turn` (150 tok at ~150 tok/s); tool execution `T_tool ≈ 2 s/turn` (a test run / API call).

**Re-prefill (§10.1).** Cumulative prefill tokens over the session:
- *No cache:* `Σ_{t=1}^{15}[1500 + (t−1)·600] = 85,500` tokens.
- *Cached:* `1,500 + 14·600 = 9,900` tokens — an **8.6× reduction**.
At `2·N·tokens` FLOPs on 8B, that is **~2.8 s vs ~0.3 s** of GPU prefill compute per session.

**Three scenarios, per session:**

| | No cache, hold KV | Cached, hold KV | Cached + offload during tools |
|---|---|---|---|
| Prefill (GPU-busy, compute-bound) | ~2.8 s | ~0.3 s | ~0.3 s |
| Decode (GPU-busy, memory-bound) | ~15 s | ~15 s | ~15 s |
| Tool wait (GPU idle) | ~30 s | ~30 s | ~30 s |
| **Session latency** | ~47.8 s | ~45.3 s | ~45.3 s |
| **GPU-busy (total)** | ~17.8 s | ~15.3 s | ~15.3 s |
| **Slot held** | ~47.8 s | ~45.3 s | **~15.3 s** |
| **Slot duty cycle** | ~37% | ~34% | ~100% (backfilled) |

**What each impact looks like in these numbers:**

- **Latency decomposition (§10.4).** In the cached/hold case (~45 s): **tool ~66%, decode ~33%, prefill <1%.** The model is a *third* of the latency; halving its TPOT shaves ~16% off the session. *The bottleneck is the tools.*
- **Tool-latency-gated GPU (§10.3).** Holding KV, each session runs its slot at a **~34% duty cycle** — the 30 s of tool waits are idle KV. **Offloading during tool calls frees the slot**, letting ~3× more sessions share the GPU (slot held drops 45 s → 15 s). That 3× is real capacity, not a tuning knob.
- **Silent cache death (§10.2).** If an injected timestamp or non-deterministic tool-JSON breaks the prefix, every turn misses and prefill compute reverts **0.3 s → 2.8 s (8.6×)**. Note *where it hurts*: session latency rises only ~5% (tools still dominate), but **GPU compute per session rises ~16%, with no error** — so the damage is a **cost/throughput** problem, invisible on a single-session latency graph and visible only on the bill or aggregate GPU-hours. Across 1,000 sessions, that is **~40 wasted GPU-minutes**, silently.

The example also shows the *non-obvious* truth: with slow tools, a broken cache barely moves user latency, which is exactly why it goes unnoticed — you have to be watching cache-hit rate and GPU-hours, not latency, to catch it.

---

## 10.6 The picture to carry forward

- Agent cost is a **re-prefill** question; prefix caching (Module 5) turns cumulative `O(T²)` into `O(T)` (~20× at 40 turns) — *if you get the cache hit*.
- **Cache hits are fragile and fail silently** (unstable tool output, injected metadata, history rewriting, system-prompt drift); stable append-only prefixes and a tracked cache-hit rate are the discipline.
- **Tool latency gates GPU utilization**, not model speed; the decision is hold-vs-offload the paused KV, or make tools fast.
- **Measure end-to-end and decomposed** (model / tool / re-prefill + cache-hit rate) — the model is usually not the bottleneck.
- The only new machinery agents add over the rest of the course is the **structured tool call (Module 9)**.

---

## Lab 10 — Make the cache miss, and watch the cost explode

**Context — what this builds on (checklist).** *Reuse:* `common/bench` metrics and `common/eval`'s schema scorer (Module 9); Module 5's prefix caching. **Needs** a small **multi-turn session builder** in `common/traffic` (a session of `T` tool calls with per-turn increment `Δ` and a configurable simulated tool latency `T_tool`) — an addition to the `common/` spec. *Test the new insight:* the *concrete* realities — cache fragility and tool-latency capacity — not a concept tour. *Exercise prior quantities:* Module 5 (caching), Module 7 (context growth), Module 9 (structured call). *Consistency:* measure end-to-end and **cache-hit rate**, never model-only.

1. **The re-prefill lever.** Run a `T`-turn session with prefix caching off vs on; measure cumulative re-prefill. Off ≈ `O(T²)`; on ≈ `O(T)`. *Confirms: §10.1.*

2. **Break the cache silently (the practical centerpiece).** Run two otherwise-identical agent loops: one with a **stable, append-only prefix**, one with a realistic **perturbation** (inject a per-turn timestamp, or non-deterministic tool-result JSON ordering). Measure the **cache-hit rate** and the resulting cost/latency: the perturbed run silently collapses to the `O(T²)` path with **no error raised**. *Confirms: §10.2 — the #1 practical failure.*

3. **Tool-latency capacity.** Hold KV during tool calls and **sweep `T_tool`** under many concurrent sessions; plot GPU-busy fraction and session throughput, showing them collapse as `T_tool` dominates (`util ≈ T_gen/(T_gen+T_tool)`). Then enable **paused-KV offload** and show the slot freed for other sessions (at a restore cost). *Confirms: §10.3.*

4. **End-to-end decomposition.** Decompose total session latency into model / tool-execution / re-prefill; show the model is often the small part. *Confirms: §10.4.*

**Deliverable:** the re-prefill curve (off `O(T²)` vs on `O(T)`); the **cache-hit-rate-vs-prefix-stability** result showing a silent cost blowup with no error; the **tool-latency capacity curve** (`util ≈ T_gen/(T_gen+T_tool)`) with offload recovery; and the end-to-end decomposition. **Mastery test — defend in one sentence each:** *the `O(T²)`→`O(T)` re-prefill and what it depends on; the concrete ways a cross-turn cache hit silently breaks, and the metric that catches it; why tool latency, not model speed, gates GPU utilization, and the hold-vs-offload decision; and why agent serving must be measured end-to-end and decomposed.* *Feeds:* Module 11 (a session's continuation must route to where its prefix is cached) and Module 12 (decomposed measurement).

**Reading:** vLLM/SGLang automatic-prefix-caching docs (the cache-key and stability rules are the practical core); a provider prompt-caching API's documentation, read specifically for *what invalidates the cache*. Background: Module 5.