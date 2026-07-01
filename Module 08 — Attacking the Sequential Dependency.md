# Module 8 — Attacking the Sequential Dependency

*Prerequisites: Module 1 (decode is memory-bound and strictly serial — token t+1 needs token t), Module 4 (prefill blocks decode; continuous batching), Module 5 (cross-instance KV), Module 7 (long prefill is expensive). Everything so far took autoregressive decode's one-token-at-a-time nature as given. This module attacks it. The centerpiece — speculative decoding — works precisely *because* decode is memory-bound (Module 1), and its correctness is a clean, exact result worth proving in full.* 

---

## In plain English

**Why this matters.** Even with perfect batching, a single user's reply is produced one word at a time, in strict order — and that serial chain sets a hard floor on how fast any one response can be. This module attacks that floor.

**What this module gives you.** Speculative decoding (a small fast model guesses ahead and the big model checks all the guesses in one shot), plus chunked prefill and prefill/decode disaggregation — and a clean guarantee that the output is *exactly* what the big model would have said anyway.

**How it works (the intuition).** Like autocomplete that drafts the next few words while a careful editor verifies them in a single pass: accept the good guesses, fix the first wrong one. It's nearly free while the GPU is idle waiting on memory — and stops paying once the GPU is already busy.

---

## 8.1 The last bottleneck: single-request latency is serial

Modules 4–5 maximized *throughput* across many requests (batching, scheduling, memory). But a **single** request's latency was untouched: generating N tokens is N sequential forward passes, each at memory-bound latency, and **batching does not help one request's latency** — it helps how many requests you serve at once. For latency-critical use (a single user waiting), the serial decode is the wall.

Three distinct sequential dependencies can be broken, and this module breaks all three:

1. token t+1 depends on token t → **speculative decoding**;
2. a long prefill blocks in-flight decodes (Module 4) → **chunked prefill**;
3. prefill and decode share hardware despite opposite profiles (Module 1) → **disaggregation**.

---

## 8.2 Speculative decoding — and why it is nearly free

A cheap **draft** model autoregressively proposes `k` tokens. The expensive **target** model then **verifies all k in a single forward pass** (processing them in parallel, like a mini-prefill). Accepted tokens are kept; the first rejection truncates and a corrected token is resampled.

Why the verification is nearly free is the Module 1 payoff: the target's forward pass is **memory-bound** — its cost is dominated by reading all the weights from HBM once, and processing `k` tokens in that pass adds almost no time because the extra FLOPs are tiny next to the weight read. **So the target produces up to k tokens for the price of roughly one memory-bound pass.** Speculative decoding *monetizes the idle FLOPs Module 1 identified.* (This is also why its benefit evaporates at high batch — §8.4.)

Two parameters govern it: the **acceptance rate α** (how often draft tokens are accepted — a function of draft-target alignment and how predictable the text is) and the **draft length k**.

---

## 8.3 Correctness — the output is *exactly* the target distribution

The remarkable property: speculative decoding produces tokens distributed **exactly** as the target model would have produced them — not an approximation. The mechanism is **modified rejection sampling**. For one position, with target distribution `p` and draft distribution `q`:

1. Draft samples `x ~ q(x)`.
2. **Accept** `x` with probability `min(1, p(x)/q(x))`.
3. If **rejected**, resample `x` from the residual `p'(x) = (p(x) − q(x))₊ / Σ_x (p(x) − q(x))₊`.

**Proof that the emitted `x` ~ `p`.** The probability of finally emitting a specific token `x`:

`P(emit x) = P(drafted x and accepted) + P(rejected) · p'(x)`

The first term: `q(x) · min(1, p(x)/q(x)) = min(q(x), p(x))`.

The total rejection mass: `β = Σ_{x'} q(x')·(1 − min(1, p(x')/q(x'))) = Σ_{x'} (q(x') − p(x'))₊ = Σ_{x'} (p(x') − q(x'))₊` (the positive and negative parts of `p − q` have equal mass since both sum to 1). So the second term is `β · p'(x) = (p(x) − q(x))₊`.

Therefore `P(emit x) = min(q(x), p(x)) + (p(x) − q(x))₊`. Two cases:
- if `p(x) ≥ q(x)`: `= q(x) + (p(x) − q(x)) = p(x)`;
- if `p(x) < q(x)`: `= p(x) + 0 = p(x)`.

Either way `P(emit x) = p(x)`. **∎** The draft model affects *speed*, never the output distribution. For `k` tokens, verify left-to-right, applying the test at each position (the single verification pass already produced the target's distribution at every position), and on the first rejection resample from the residual there and stop — every emitted token is exactly target-distributed. If all `k` are accepted, the verification pass also yields one **bonus** token from the target's distribution at the next position.

---

## 8.4 The expected-speedup law — and where it dies

With per-token acceptance `α` (iid approximation) and draft length `k`, the expected tokens produced per target forward pass is (Leviathan et al.):

```
E[tokens per target pass] = (1 − α^(k+1)) / (1 − α)
```

(Check `k=1`: `(1−α²)/(1−α) = 1 + α` — accept (prob α) → 2 tokens incl. the bonus; reject → 1 resampled. ✓) If the draft costs `c` target-passes per token, the per-iteration cost is `≈ 1 + k·c`, so

```
speedup ≈ (1 − α^(k+1)) / [ (1 − α)(1 + k·c) ]
```

Behaviors: speedup rises with `α`; as `α → 1`, tokens `→ k+1`; there is an **optimal k** (too large wastes draft compute on tokens that will be rejected). Full derivation in Going Deeper.

**The crucial regime caveat (consistency with Modules 1, 3, 4).** The "free verification" rests entirely on the target pass being *memory-bound* (§8.2). At **high batch**, Module 1's arithmetic intensity has already risen — the FLOPs are **no longer idle**, batching consumed them — so verifying `k` tokens × a large batch now adds *real* compute, and speculation's advantage **shrinks toward zero or goes negative**. Speculative decoding is therefore a **low-batch / latency-critical** technique; in a saturated high-throughput regime it does not pay. This is the same arithmetic-intensity lesson as Module 3's quantization split, and the lab measures exactly where it dies.

---

## 8.5 Chunked prefill — breaking prefill-blocks-decode

Introduced in Module 4: a long prefill in one scheduler step stalls all in-flight decodes. Splitting the prefill into bounded chunks interleaved with decode steps removes the stall — another sequential dependency broken, now the default in vLLM (Sarathi-Serve). See Module 4 §4.5; it belongs in this module's taxonomy as the second of the three dependencies.

---

## 8.6 Prefill–decode disaggregation — separating opposite profiles

Module 1 showed prefill is compute-bound and decode is memory-bound — opposite profiles — and Module 4 showed co-locating them causes interference. The radical fix: run prefill and decode on **separate GPU pools**, each tuned to its profile (prefill pool for compute throughput; decode pool for memory bandwidth and large batch). DistServe and Splitwise (2024) established this.

The cost, and where Module 5 returns: after the prefill pool produces the **KV cache**, it must be **transferred** to the decode pool — a cross-instance KV migration whose cost is `KV_size / interconnect_bandwidth` (Modules 2, 6) and must be hidden or amortized. The payoffs are large: **no interference**, **independent scaling** of prefill vs decode capacity to match the workload's prefill/decode ratio, and each pool can run a **different parallel config** (Module 6 — e.g. heavier TP for compute-bound prefill). Module 7's long-prefill workloads benefit most: the heavy quadratic prefill goes to a dedicated, compute-optimized pool instead of stalling a shared GPU.

---

## 8.7 Where the frontier is now

- **Self-speculation — no separate draft model.** Maintaining a well-aligned draft model is hard, so the frontier derives the draft from the target itself: **Medusa** (extra decoding heads, 2024) and **EAGLE / EAGLE-2 / EAGLE-3** (a lightweight autoregressive head on the target's features, 2024–25 — current state of the art, high acceptance), plus n-gram **lookahead decoding**. These raise `α` and remove the second-model maintenance burden, often using **tree-structured** speculation (verify several candidate continuations at once).
- **Disaggregation in production.** DistServe/Splitwise (2024) → mainstream (Mooncake, vLLM disaggregated serving), built on the cross-instance KV stores of Module 5.

Teaching only draft-model speculation (2023) and co-located serving would miss that self-speculation and disaggregated pools are now the production reality.

---

## 8.8 The picture to carry forward

- **Single-request latency is serial decode**, which batching (Modules 4–5) does not fix — three sequential dependencies remain.
- **Speculative decoding** verifies `k` draft tokens in one memory-bound pass (free *because* decode is memory-bound, Module 1), producing the **exact** target distribution, with speedup `≈ (1−α^(k+1))/((1−α)(1+kc))` — but the freeness **dies at high batch**, so it is a latency technique.
- **Chunked prefill** (Module 4) and **disaggregation** break the other two dependencies; disaggregation separates Module 1's opposite profiles at the cost of a **KV transfer** (Module 5).
- The frontier is **self-speculation** (EAGLE/Medusa) and **production disaggregation**.

---

## Going Deeper (appendix) — the speedup derivation and tree speculation

**Expected tokens.** Let the per-token acceptance be `α`, iid. Over `k` drafted tokens you accept a prefix until the first rejection. The expected number of accepted draft tokens is `Σ_{j=1}^{k} α^j = α(1−α^k)/(1−α)`, and you always emit one more (the resampled token at the rejection, or the bonus token if all `k` accept), giving `E[tokens] = 1 + α(1−α^k)/(1−α) = (1−α^{k+1})/(1−α)`. Dividing by per-iteration cost `1 + kc` gives the speedup; maximizing over `k` yields the optimal draft length, which falls as `α` falls (low alignment → short drafts).

**Tree speculation.** Instead of one linear draft of `k` tokens, propose a **tree** of candidate continuations and verify them together with a specialized (tree) attention mask. This raises the expected accepted length per pass for the same target cost (more chances for a match), and is what Medusa/EAGLE use. The correctness argument of §8.3 applies along the accepted path.

---

## Lab 8 — Measure the speedup, and find where it stops paying

**Context — what this builds on (checklist).** *Reuse:* the `common/bench` **open-loop** generator and Module 1's prefill/decode isolation and Module 4's latency/throughput methodology; `mem_estimate` for the disaggregation KV-transfer size. *Test the new insight:* the speculative speedup-vs-acceptance curve and its **batch-dependence**, not a restated throughput run. *Exercise prior quantities:* Module 1 (the free verification *and* its high-batch death), Module 4/5 (disaggregation + KV transfer), Module 7 (long-prefill disaggregation benefit). *Frontier:* EAGLE/Medusa self-speculation. *Consistency:* sweep batch — speculation must be shown to **fade at high batch**, and `α` measured on realistic prompts.

1. **Speedup vs acceptance (the new insight).** Run draft+target speculative decoding at fixed `k`; measure end-to-end decode speedup and the realized acceptance rate `α` on realistic prompts. Sweep `k`, find the optimal, and confirm the `(1−α^{k+1})/(1−α)` expected-tokens relation. *Confirms: §8.2–8.4.*

2. **Where it stops paying — the batch sweep (the consistency centerpiece).** Hold the method fixed and sweep batch size; measure speculative speedup at each. Show it is large at batch 1 and **fades to ~0 (or negative) at high batch**, because batching already consumed the idle FLOPs (Module 1). Conclude that speculation is a latency, not throughput, technique. *Confirms: §8.4 — the regime where it helps.*

3. **Frontier: self-speculation.** Compare a separate draft model against **EAGLE/Medusa** (target-derived draft); show higher `α` and no second model to maintain. *Confirms: §8.7.*

4. **Disaggregation (Modules 4/5/7 payoff).** Compare co-located prefill+decode against **disaggregated** pools on a long-prefill workload; measure the interference removed, the **KV-transfer cost** (Module 5; predict its size with `mem_estimate`), and independent prefill/decode scaling. *Confirms: §8.6.*

**Deliverable:** the **speedup-vs-acceptance curve** with the optimal `k`, **and** the **speedup-vs-batch curve marking where speculation stops paying**; the self-speculation comparison; and the disaggregation interference + KV-transfer measurement. **Mastery test — defend in one sentence each:** *why single-request latency is the serial-decode wall that batching doesn't fix; why verifying `k` draft tokens is nearly free, and why that freeness dies at high batch; the rejection-sampling argument that the output is exactly the target distribution; the α–k tradeoff; and what disaggregation separates and what it costs.* *Feeds:* Module 11 (a serving stack must expose all of this) and Module 12 (measuring it without lying — especially the batch caveat).

**Reading:** Leviathan et al., *Fast Inference from Transformers via Speculative Decoding* (2023); Chen et al., *Accelerating LLM Decoding with Speculative Sampling* (2023). Current frontier: *Medusa* (2024), *EAGLE / EAGLE-2 / EAGLE-3* (2024–25) for self-speculation; *DistServe* and *Splitwise* (2024) for disaggregation.