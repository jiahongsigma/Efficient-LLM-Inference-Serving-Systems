# Module 12 — How to Benchmark Without Lying to Yourself

*Prerequisites: every prior module, because every prior lab produced numbers — and this module is deliberately last so it lands on you *after* you have already generated misleading ones. By now you have seen prefix caching look useless on the wrong traffic (Module 5), eviction look free on the wrong metric (Module 7), quantization look free on a saturated benchmark (Module 3), and speculation help at the wrong batch (Module 8). This module names the pattern, formalizes it, and arms you to distrust any single number — including your own.*

---

## In plain English

**Why this matters.** Every earlier lab produced numbers — and numbers lie easily. The same system can be made to look like a winner or a loser just by changing how you test it. This module teaches you to distrust benchmarks, including your own.

**What this module gives you.** Precise definitions of the metrics that matter (and which one your app actually cares about), why your *test traffic* decides the result more than the engine does, the load-testing trap that hides your real worst-case latency, and why every speed claim needs a quality claim beside it.

**How it works (the intuition).** A benchmark is a little model of your real workload, so a convenient-but-wrong test gives a confident-but-wrong answer. You'll take one earlier result and make it both "win" and "lose" using defensible-but-different choices — the inoculation against fooling yourself.

---

## 12.1 Why this is last, and the thesis

A measurement is not a fact; it is the output of a benchmark, and **a benchmark is a *model* of a workload.** Like any model, it can be wrong, incomplete, or chosen adversarially. The course's recurring discovery, now made explicit:

| Module | Same system, opposite conclusion, because… |
|---|---|
| 3 | quantization is "free" on saturated MMLU, harmful on hard reasoning — *benchmark difficulty* |
| 5 | prefix caching is useless on ShareGPT, transformative on long-doc-QA — *traffic shape* |
| 7 | eviction is free on perplexity, fails on needle-in-haystack — *eval task* |
| 8 | speculation helps at batch 1, hurts at batch 64 — *operating regime* |

The skill is not "running a benchmark"; it is knowing which choices are honest **for your workload**, and being skeptical of everyone else's numbers (including your past self's). The whole module is the set of choices that decide the answer.

---

## 12.2 The metrics — and which one your application actually cares about

Define them precisely, because reporting the wrong one is the easiest way to mislead:

- **TTFT** (time to first token) — prefill latency; the *perceived responsiveness* of a streaming app.
- **TPOT / ITL** (per-output-token / inter-token latency) — decode speed; the streaming *reading rate*.
- **End-to-end latency** — total; what a *non-streaming* (batch, classification) caller sees.
- **Throughput** (tokens/s, req/s) — capacity, the *cost* driver.
- **Goodput** (throughput meeting an SLO, Module 4) — the *honest* capacity metric.
- **Tail (p95/p99)** vs **average** — the tail is the user experience; the average hides it.

Different applications care about different metrics: a chatbot lives or dies on **TTFT and TPOT tails**; a batch pipeline cares about **throughput/cost**; a classifier cares about **end-to-end p99**. **Reporting only throughput, or only the average, is a way to hide** — and it is the most common one. Report the metric the *workload* cares about, on the *tail*.

---

## 12.3 The workload is the benchmark

The §12.1 table has one root cause: **the traffic distribution determines the result more than the engine does.** The variables that flip conclusions are the prefix-sharing ratio (Module 5), the prompt/output length distribution (Modules 2, 7), the arrival pattern (Module 4), the batch/concurrency regime (Modules 3, 8), and the eval-task difficulty (Modules 3, 7). A synthetic or borrowed benchmark that differs from your traffic on any of these can **invert your decision**.

The discipline: characterize *your* traffic (Module 0's harness lesson) and replay *that* — its real prefix-sharing, lengths, arrivals, and batch regime — not a convenient public dataset that happens to be at hand. ShareGPT is a fine stress trace; it is a terrible proxy for a RAG workload.

---

## 12.4 Open vs closed loop, and coordinated omission

Module 4's load-generation point, now formalized because it is the deepest latency-measurement trap:

- **Closed-loop** (fixed client pool, each sends its next request only after the previous returns) **self-throttles** — it cannot offer more load than the server serves, so it **never reveals the knee** and reports a falsely good tail.
- **Open-loop** (requests arrive at rate λ independent of server state — Poisson or a trace) exposes both the knee and the true tail.

**Coordinated omission** is the subtle killer (Gil Tene): when a response is slow, the *next* request is delayed by exactly that slowness, so the bad latency is sampled **once and never compounds** — the measured tail is far better than reality. The fix is open-loop with requests scheduled at **fixed wall-clock times regardless of when responses arrive**, counting the full queueing delay of any request that could not be sent on time. Measure latency any other way and your p99 is fiction. (Formal treatment in Going Deeper.)

---

## 12.5 Statistical rigor

A single run of a single number is not a measurement:

- **Warmup.** Discard the first requests — cold start (weight load, CUDA-graph capture, cache fill) measures the wrong thing. Report steady state.
- **Variance and intervals.** Run multiple times; report mean ± confidence interval (or median + IQR), never a lone number. Serving has real run-to-run variance.
- **Non-determinism at temperature 0.** Batched inference is **not deterministic even at temp 0** — batch composition changes floating-point reduction order (Modules 0, 3), so the *same* prompt yields different outputs depending on its batchmates. Reproducibility therefore requires either **batch-invariant kernels** (§12.8) or honestly accepting and reporting the variance. "Set temperature 0 for reproducibility" is, by itself, false.
- **Sample size for tails.** A stable p99 needs many samples; a p99 from 100 requests is noise.

---

## 12.6 Quality and performance are one measurement

Every speed technique in this course that *can* trade quality — quantization (Module 3), eviction (Module 7), approximate self-speculation (Module 8) — means **a performance number without a quality number is meaningless: you can always go faster by being wrong.** The Module 3 and Module 7 deliverables paired them deliberately (quality × throughput, accuracy × memory). The quality side has its own traps: **contamination** (benchmark data leaked into training), **saturation** (Module 3 — an easy benchmark hides the regression), and the gap between *benchmark* quality and *your-task* quality. Report both axes, on an un-saturated, un-contaminated eval that resembles your task.

---

## 12.7 Standardized benchmarks — value and limits

**MLPerf Inference** is the audited industry standard. Its scenarios encode much of this module's discipline:

- **Offline** — maximize throughput, no latency constraint (the cost-ceiling number).
- **Server** — **Poisson (open-loop) arrivals with TTFT/TPOT latency SLOs**, scoring the request rate sustainable within the SLO — i.e. goodput, measured correctly.
- **Single-/Multi-stream** — latency under fixed concurrency.

The Server scenario is, in effect, this module enforced. Its **value** is standardized, audited, apples-to-apples comparison; its **limit** is that it is a *fixed* workload — excellent for vendor comparison, not a substitute for benchmarking *your* traffic (§12.3). Use it to compare systems; use your own trace to make your decision.

---

## 12.8 Where the frontier is now

- **MLPerf Inference LLM scenarios** (v4/v5) — Llama-2-70B, Mixtral, and reasoning workloads under the Server scenario's SLO discipline.
- **Production trace replay** — benchmarking against real arrival/length distributions (e.g. published Azure LLM-inference traces) rather than synthetic ShareGPT.
- **Reproducible inference** — **batch-invariant kernels** (2025 work on defeating LLM-inference nondeterminism) make serving bitwise-reproducible at a throughput cost, finally letting "temperature 0" mean what people thought it meant.
- **The reproducibility / contamination crisis** — leaderboard contamination and unfair comparison have made the community far more skeptical of single quality scores; the methodological response (held-out, private, or freshly-generated evals) is itself a frontier.

The trajectory across the board is **from synthetic/average/closed-loop toward trace-based/tail/open-loop/quality-aware/reproducible** — which is the whole of this module.

---

## 12.9 The picture to carry forward

- A **benchmark is a model of a workload**; the choices (metric, traffic, loop type, batch, eval task) determine the conclusion, so the same system can be made to win or lose.
- Report the **metric the workload cares about, on the tail** — not throughput-only or average-only.
- **Benchmark on your traffic** (Module 3/5/7/8 all reduce to this).
- **Open-loop, fixed-schedule** generation, or your tail is fiction (coordinated omission).
- **Warmup, variance/CI, and temp-0 non-determinism** are mandatory; quality and performance are **one** measurement.
- Standardized benchmarks (MLPerf Server) compare systems; **your trace** makes your decision.
- Distrust any single number — including your own.

---

## Going Deeper (appendix) — coordinated omission and batch-invariant determinism

**Coordinated omission, formally.** Suppose requests are *intended* at times `t, t+δ, t+2δ, …`. In a closed-loop or naive harness, if the request at `t` takes `T ≫ δ`, the harness only issues the next request at `t+T` — so the requests that *would* have arrived during `[t, t+T]` are never issued, and the latency that those omitted requests would have experienced (mostly queueing) is never recorded. The result systematically under-counts the tail. The correction is to issue requests on the *fixed* schedule regardless of completion, and for any request issued late, add the time it spent waiting to be issued to its measured latency. This is why an honest harness records `latency = (response_time − intended_send_time)`, not `(response_time − actual_send_time)`.

**Batch-invariant determinism.** Standard attention/GEMM kernels reduce in an order that depends on batch shape, so the floating-point result for a given sequence depends on its batchmates → non-determinism at temp 0. **Batch-invariant kernels** fix the reduction order to be independent of batch composition (e.g. fixed split-K, deterministic reductions), yielding bitwise-identical outputs regardless of what else is in the batch, at some throughput cost. This is the implementation behind a genuinely reproducible `temperature=0` serving path — the option Module 0's audit/bulk split assumed exists.

---

## Lab 12 — Make the same system win, then lose

**Context — what this builds on (checklist).** *Reuse:* the `common/bench` generator and **every prior lab's result** — this lab re-examines them. *Test the new insight:* that benchmark *choices* determine the conclusion — the meta-experiment, not a new serving technique. *Exercise prior quantities:* the traps from Modules 3, 5, 7, 8, the metrics from Modules 1/4, the determinism from Module 0/3. *Frontier:* an MLPerf-Server-style scenario or trace replay. *Consistency:* this lab is the course's methodology synthesis.

1. **Make a system win, then lose (the centerpiece).** Take one earlier conclusion (e.g. "engine A beats B", "INT4 is faster", "prefix caching helps") and construct **two defensible benchmarks** — one where it wins, one where it loses — by changing only the traffic / metric / batch / eval task. Document each choice as defensible-but-different. *Confirms: §12.1 — the benchmark determines the answer.*

2. **Coordinated omission.** Measure the *same* system's tail latency closed-loop vs open-loop-with-fixed-schedule; show the closed-loop p99 is dramatically optimistic, and quantify the gap. *Confirms: §12.4.*

3. **Statistical rigor + determinism.** Run a benchmark several times, with and without warmup; report variance and a CI; demonstrate temp-0 batched non-determinism (same input, different output under different batchmates), and — if available — batch-invariant kernels restoring it. *Confirms: §12.5, §12.8.*

4. **Quality + performance.** Exhibit a "faster" config that is only faster because it is worse, by pairing its speed number with its quality number on an *un-saturated* eval; show a saturated benchmark hiding the same regression. *Confirms: §12.6.*

5. **(Frontier) MLPerf-Server / trace replay.** Run an open-loop Poisson + TTFT/TPOT-SLO scenario (or a production trace) and contrast its conclusion with the synthetic ShareGPT result. *Confirms: §12.7–12.8.*

**Deliverable:** a **methodology note** that shows the *same system* "winning" and "losing" under defensible-but-different benchmark choices (the meta-lesson), plus the coordinated-omission open-vs-closed tail gap, the variance/CI/non-determinism demonstration, and a quality+performance pairing. **Mastery test — defend in one sentence each:** *which metric a given application cares about and why average/throughput-only hides; why the benchmark is a model of the workload and traffic determines the result; coordinated omission and why open-loop fixed-schedule is required for honest tails; why temp-0 batched inference is non-deterministic and what reproducibility requires; and why a performance number without a quality number is meaningless.* *Feeds:* the **capstone** — every claim in your final project must survive this module's scrutiny.

**Reading:** the **MLPerf Inference** rules (especially the Server scenario); Gil Tene's *coordinated omission* talk/notes; the 2025 work on **batch-invariant / deterministic LLM inference**. Background: any prior lab of yours, re-read with this module's skepticism.