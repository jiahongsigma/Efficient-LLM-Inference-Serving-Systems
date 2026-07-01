# Module 4 — Batching and Scheduling

*Prerequisites: Module 1 (batching raises arithmetic intensity → the throughput lever), Module 2 (KV-bandwidth and KV-capacity re-bind batching at scale), Module 3 (quantization raises the ceiling batching pushes against). Modules 1–3 made a single forward pass efficient. This module is the pivot of the course: serving a **stream** of independent requests that arrive at different times and have wildly different lengths. That is a **scheduling** problem, and the naive answer is broken.*

---

## In plain English

**Why this matters.** Serving one request at a time wastes almost all of your expensive GPU. But real users don't arrive politely together — they show up at random moments wanting wildly different amounts of text. Serving them well is a scheduling problem, and the obvious approach is badly broken.

**What this module gives you.** Why naive "wait for the whole group to finish" batching collapses, how *continuous* batching fixes it, and how to find the operating point between speed-per-user and total capacity.

**How it works (the intuition).** Picture a bus that won't leave until its slowest passenger is done — newcomers wait outside, everyone's stuck behind one person. Continuous batching is a bus people hop on and off of at every stop. The job is to find the load level just *before* the queue explodes.

---

## 4.1 The pivot: one forward pass is not a server

Module 1 said "batch B sequences and you climb the roofline." True — but it quietly assumed B fixed sequences that start and finish together. Real traffic does not look like that:

- requests **arrive continuously** at times you do not control;
- their **output lengths vary by 100×** (one wants 8 tokens, the next wants 2,000);
- they have **different prompt lengths** (different prefill costs).

A server must keep the GPU saturated (Module 1's throughput argument) *while* admitting and retiring requests on the fly. Modules 1–3 gave you an efficient forward pass; they did not give you a way to schedule a stream onto it. That is this module.

---

## 4.2 Why static batching is broken

The obvious approach — **static batching** — collects a batch, runs `generate()` until every member finishes, returns, repeats. Three failures, all rooted in length variance and arrival timing:

1. **Head-of-line blocking.** The batch runs until its *longest* member finishes. A request needing 8 tokens sits in a batch with one needing 2,000 — its slot is held, its compute wasted, for ~250× longer than its own work. Short requests pay the long requests' latency.
2. **No mid-flight admission.** A request arriving one step after the batch starts cannot join — it waits for the *entire* batch to drain before it even begins prefill. Under streaming load this is fatal to tail latency.
3. **Prefill padding waste** if prompts are padded to a common length.

So static batching forces a lose-lose: large batches give high throughput but catastrophic tail latency (everyone waits for the longest, newcomers wait for the whole batch); small batches give better latency but drop back into Module 1's idle-FLOP regime. There is no good operating point.

---

## 4.3 Continuous (iteration-level) batching — the fix

**Orca** (OSDI 2022) made the batching decision at the granularity of a **single decode iteration** rather than a whole request. After *every* forward step:

- sequences that emitted EOS or hit max length **leave** the batch, freeing their slot immediately;
- **waiting requests join** the running batch on the next step.

The batch composition is now *fluid* — finished requests exit, new ones enter, every iteration. The consequences directly fix §4.2: no slot is wasted holding a finished sequence (head-of-line blocking gone), and a newcomer joins within ~one iteration instead of waiting for a batch to drain. The GPU stays saturated under streaming load. This is **iteration-level scheduling**: each step, decide which requests run.

Two connections this module must make explicit, because they are why continuous batching is not free:

- **It requires dynamic KV memory (→ Module 5).** The set of active sequences changes every step, each holding a *growing* KV cache (Module 2). Allocating and freeing that cache, per step, without fragmenting memory is exactly what PagedAttention solves — which is why vLLM shipped continuous batching and paging *together*. Module 4 decides *which* sequences run; Module 5 manages *where their KV lives*.
- **Prefill and decode interfere.** A newly admitted request needs a compute-heavy prefill while running requests need cheap decode steps. If the scheduler runs a long prefill in one step, every in-flight decode **stalls** behind it — a latency spike for all of them. This tension is the seam where the modern frontier lives (§4.5).

---

## 4.4 The latency–throughput frontier

Even with continuous batching, a fundamental tradeoff remains, and finding its operating point is the serving engineer's core job.

Raising concurrency (more sequences in flight) raises throughput — Module 1: higher batch, higher arithmetic intensity. But it also raises per-token latency (each decode step now processes more sequences) and, past a point, induces queueing. Sweep the offered load upward and you get the characteristic shape:

- **Throughput** rises, then **saturates** — and it saturates at `min(compute ceiling, KV-bandwidth ceiling)` from Modules 1–2, with the achievable batch itself **capped by KV capacity** (Module 2 / Module 5). Three distinct walls; know which one you hit.
- **Latency** rises gently, then **explodes** past the saturation point (the queue builds because requests arrive faster than the server drains them).

The knee between those regimes is the operating point. To make it precise you need the right metric: **goodput** — the throughput of requests that *meet a latency SLO*. Past the knee, raw throughput may plateau looking healthy while goodput **collapses**, because latencies now violate the SLO. The job is to run **near but below the knee**, where goodput is maximized. (The queueing math behind the knee — why latency blows up as `1/(1−ρ)` — is in Going Deeper.)

---

## 4.5 Where the frontier is now

Continuous batching itself is **settled** — vLLM, SGLang, TGI, and TensorRT-LLM all do it; it is the baseline, not the frontier. The live problem is the prefill–decode interference of §4.3, and the field has moved through two stages:

- **Chunked prefill (Sarathi-Serve, 2024), now the default in vLLM.** Split a long prefill into bounded chunks and interleave them with decode steps, so no single iteration is dominated by prefill and the decode stall flattens. It trades a little prefill efficiency for a large tail-latency improvement.
- **Prefill–decode disaggregation (DistServe, Splitwise, 2024).** Run prefill and decode on *separate* GPU pools entirely, since Module 1 showed they have opposite resource profiles (compute-bound vs memory-bound) and interfere when co-located. This is the natural endpoint of the interference story and is developed in Module 8.

So the arc is Orca (iteration-level batching, 2022) → chunked prefill (interleave to reduce interference, 2024) → disaggregation (separate entirely, 2024). Teaching only Orca would leave you two years behind the production default.

---

## 4.6 The picture to carry forward

- Modules 1–3 made a forward pass efficient; **serving a stream is a scheduling problem** they did not solve.
- **Static batching is broken** by length variance and arrival timing (head-of-line blocking, no mid-flight admission).
- **Continuous (iteration-level) batching** is the fix — fluid batch composition — and it *requires* dynamic KV management (Module 5) and *exposes* prefill–decode interference.
- The **latency–throughput frontier** has a knee; throughput saturates at the compute/KV-bandwidth ceiling (Modules 1–2) and batch is capped by KV capacity; you operate near-but-below the knee, measured by **goodput**.
- The frontier is **chunked prefill → disaggregation** (Module 8), not vanilla continuous batching.

---

## Going Deeper (appendix) — the serving queue

**Little's law** (exact, distribution-free): `L = λ·W` — average number of requests in the system equals arrival rate × average time in system. For serving, the average concurrency (effective batch) equals arrival rate × average latency. Concretely: to sustain an average batch of 32 at 2 s average latency, arrivals must be ~16 req/s. This single identity ties the three quantities you measure (concurrency, throughput, latency).

**Why the knee exists.** Model service as a queue with utilization `ρ = λ/μ` (arrival rate over capacity). For an M/M/1 queue the mean sojourn time is `W = 1/(μ − λ) = (1/μ)/(1 − ρ)`: as `λ → μ`, `ρ → 1`, and `W` blows up like `1/(1−ρ)`. That `1/(1−ρ)` singularity *is* the latency knee. LLM serving is not literally M/M/1 — service times are coupled through the shared batch and arrivals are not Poisson — but the qualitative blow-up as utilization approaches one is robust and is what the latency curve shows.

**Goodput.** Formally, `goodput = throughput × P(latency ≤ SLO)`. It is the only throughput number that is operationally honest, because it penalizes the regime past the knee where requests are served but too slowly to count.

**Open- vs closed-loop (this governs the lab, and previews Module 12).** A **closed-loop** client (fixed pool, each client sends its next request only after the previous returns) *self-throttles*: it cannot offer more load than the server can serve, so it **never reveals the knee** and systematically **under-reports tail latency** (coordinated omission — a slow response delays the next request, so the slowness is never sampled again). An **open-loop** client (requests arrive at rate `λ` independent of server state, e.g. Poisson) is what real traffic looks like and what exposes both the knee and the true tail. **Serving benchmarks must be open-loop**; a closed-loop result is a comfortable lie.

---

## Lab 4 — Find the knee, the honest way

**Context — what this builds on (checklist).** *Reuse:* the `common/bench` load generator and its metric definitions (TTFT/TPOT/goodput) and the `common/traffic` ShareGPT builder — do not write a new load generator; and Module 1's roofline ceilings / Module 2's KV bounds to *explain* where throughput saturates. *Test the new insight:* the §4.2–4.4 scheduling story (static vs continuous; the latency knee; goodput), not a restated batch sweep. *Exercise prior quantities:* identify *which* of the three walls (compute, KV-bandwidth, KV-capacity) your saturation hits. *Frontier:* chunked prefill on/off. *Methodology consistency:* **open-loop generation** throughout — the one thing that makes the knee visible.

1. **Static vs continuous under a stream (the new insight).** Drive an **open-loop** ShareGPT stream with realistic length variance into (a) static batching and (b) continuous batching. Compare throughput and p50/p95/**p99** latency. Static batching's tail should explode from head-of-line blocking and no mid-flight admission; continuous batching should hold. *Confirms: §4.2–4.3 — why continuous batching exists.*

2. **The latency–throughput Pareto front and the knee.** Sweep offered load `λ` upward (open-loop); plot throughput and p99 latency vs `λ`. Mark the knee where latency explodes and throughput saturates, and **identify which ceiling you hit** — compute (Module 1), KV-bandwidth (Module 2), or KV-capacity cap on batch (Module 2/5). Optionally re-run with an INT4/FP8 model (Module 3) and show the ceiling — and thus the knee — move. *Confirms: §4.4, and ties saturation back to Modules 1–3.*

3. **Goodput — the operating point.** Fix a latency SLO; plot **goodput** vs `λ`; show it collapsing past the knee even while raw throughput plateaus, and locate the `λ` that maximizes goodput. *Confirms: run near-but-below the knee.*

4. **Frontier: prefill–decode interference.** Inject occasional long-prompt requests into a decode-heavy stream. With **chunked prefill off**, watch per-token decode latency (TPOT) spike whenever a big prefill runs; turn **chunked prefill on** and show the spikes flatten. *Confirms: §4.5; motivates disaggregation (Module 8).*

**Deliverable:** the static-vs-continuous tail-latency comparison; the latency–throughput Pareto curve with the knee marked **and the saturating wall named**; the goodput-vs-load curve with the optimal operating point; and the chunked-prefill before/after on decode TPOT — **all from an open-loop generator**, with one sentence justifying why a closed-loop run would have hidden the knee. **Mastery test — defend in one sentence each:** *why static batching collapses under length-variant streaming load; what continuous batching schedules and why it needs dynamic KV memory; why latency explodes at the knee (`1/(1−ρ)`); why a serving benchmark must be open-loop; and what causes decode stalls and how chunked prefill fixes them.* *Feeds:* Module 5 (the dynamic KV memory continuous batching demands) and Module 8 (disaggregation).

**Reading:** Yu et al., *Orca* (OSDI 2022). Current frontier: Agrawal et al., *Sarathi-Serve / chunked prefill* (2024); Zhong et al., *DistServe* (2024) and *Splitwise* (2024) for disaggregation. Background: any queueing-theory text on M/M/1 and Little's law.