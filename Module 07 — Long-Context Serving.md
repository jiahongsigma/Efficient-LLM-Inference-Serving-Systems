# Module 7 — Long-Context Serving

*Prerequisites: Module 2 (the KV size law — the thing that now explodes), Module 1 (the attention-compute crossover — it finally bites here), Module 3 (KV quantization, the third target), Module 5 (eviction as a memory-management policy), Module 6 (sequence/context parallelism). This module is the regime where the KV cache, not the weights, is unambiguously the enemy, and where full, exact, dense attention becomes unaffordable. Every technique here relaxes one of those three words.*

---

## In plain English

**Why this matters.** Feed a model a 200-page document and everything you've learned breaks at once: memory explodes, every new word gets slower, and the attention math that used to be cheap becomes the bottleneck. Long context is its own hard regime, not just "more of the same."

**What this module gives you.** Four ways to cope — throw away old tokens (eviction), store them cheaply (KV quantization), look at fewer of them (sparse attention), or replace the mechanism entirely (linear / state-space models) — and, crucially, how to tell whether you quietly broke the model's memory.

**How it works (the intuition).** All four buy room by trading away some faithfulness. The trap: a model can look perfect on average yet completely forget one fact buried mid-document. So you test it like hide-and-seek — plant a fact deep in the text and check it can still be found.

---

## 7.1 The long-context regime is qualitatively different

Return to Module 2's example (Llama-3-8B-class, ~0.125 MB/token of KV). At 8K context that was 1 GB per sequence. At **128K context it is 16 GB per sequence** — a single sequence's KV now exceeds a fifth of an 80 GB GPU, before weights. Three resources blow up at once:

- **KV memory** (Module 2): linear in context — at 128K it dwarfs the weights.
- **KV bandwidth** (Module 2's traffic axis): every decode step rereads the *entire* KV cache, so per-token decode cost grows with context.
- **Attention compute** (Module 1's crossover): prefill attention is `O(S²)`; decode attention reads `O(S)` KV per step. At long `S` the attention term **overtakes the weight-read term** — this is exactly the crossover Module 1 flagged, now dominant. Decode becomes attention/KV-bound, and prefilling 128K tokens is a quadratic-attention problem.

So the single-GPU, full-exact-dense-attention assumptions of Modules 1–5 break. The techniques below each **relax one assumption**, and they compose:

| Technique | Relaxes | Keeps |
|---|---|---|
| **Eviction** | *store all* KV | a subset of tokens, exactly |
| **KV quantization** | *store* KV *exactly* | all tokens, imprecisely |
| **Sparse attention** | *attend densely* | a learned/fixed subset of attention |
| **Linear / SSM** | *the attention mechanism* | a constant-size recurrent state |

And when even the reduced footprint exceeds one GPU, **sequence parallelism** (§7.6) splits the context across devices.

---

## 7.2 Eviction — relax "store all KV"

Not all past tokens matter equally; attention in practice is sparse, concentrating on a few tokens. Two findings turned that into policy:

- **StreamingLLM / attention sinks** (Xiao et al., 2023): the *first few tokens* receive large attention weight regardless of content — the model uses them as a "no-op" attention dump. Keep those few **sink** tokens plus a sliding window of recent tokens, evict the middle → effectively infinite streaming at *constant* KV with stable perplexity.
- **H2O / heavy hitters** (Zhang et al., 2023): dynamically keep the "heavy hitter" tokens (highest accumulated attention) plus recent tokens, evict the rest.

Eviction reduces KV from `O(context)` to `O(window)` — but it is **lossy**: an evicted token is gone and can never be attended to again. The accuracy cost is entirely about whether the task needs the evicted tokens (§7.7). This is the long-context extension of Module 5's eviction — a memory-management policy, now over semantic content rather than just blocks.

---

## 7.3 KV quantization — relax "store KV exactly"

Module 3's third quantization target, applied here as the direct attack on Module 2's `dtype_bytes` term: quantize the cached K and V to FP8/INT4/INT2. **KIVI** (2024) quantizes K per-channel and V per-token (their outlier structures differ) down to ~2-bit with little loss; rotation methods (Module 3's QuaRot) also tame KV outliers.

The contrast with eviction matters: KV quantization **keeps every token** (no retrieval failure) but stores each imprecisely, whereas eviction keeps *some* tokens exactly. So KV quantization is **safer for retrieval**, eviction is more aggressive on memory — and they **compose** (quantize the kept window). This pairing is the practical default.

---

## 7.4 Sparse attention — relax "attend densely"

Module 1: attention is `O(S²)` (prefill) / `O(S)` per decode step. **Sparse attention** has each token attend to only a subset, cutting both the compute and the KV that must be read. Patterns range from fixed (sliding-window/local, block-sparse, global+local — the Longformer/BigBird lineage) to the modern **trainable** kind:

- **Native Sparse Attention (NSA, DeepSeek 2025)** and **MoBA (2025)** make sparsity *native to training* — the model **learns which blocks to attend to** rather than wearing a fixed post-hoc mask. Because the pattern is trained, quality is largely retained while compute and KV drop.

The key distinction: **post-hoc eviction (§7.2) is inference-only and lossy; trainable sparse attention bakes the sparsity into the model**, which is why it holds quality better. The tradeoff is still that you give up attending to *everything*; trainable sparsity mitigates it by learning where to look.

---

## 7.5 Linear attention and SSMs — relax the mechanism

The most radical move: replace softmax attention's `O(S²)`/growing-KV with a **recurrence** that is `O(1)` compute and `O(1)` memory per token — *no growing KV cache at all*.

- **Linear attention** approximates `softmax(QKᵀ)V` with a kernel feature map so it computes as a fixed-state recurrence.
- **State-space models (Mamba / Mamba-2)** use a selective SSM — an input-dependent recurrence with a constant-size state, hardware-efficient and competitive in quality.

The catch (Going Deeper): a **fixed-size state must compress all history**, so these models cannot perfectly recall an arbitrary past token — exactly the "recall" task where full-KV attention excels. Hence **hybrids** (e.g. Jamba): interleave a few attention layers (for recall) among many SSM/linear layers (for efficiency) — the current practical sweet spot for long context with bounded memory.

---

## 7.6 When it still doesn't fit: sequence/context parallelism

If a single long sequence's KV/activations exceed one GPU even after quantization and eviction, split the **sequence dimension** across GPUs — the fourth parallelism axis Module 6 teed up. **Ring Attention** gives each GPU a chunk of the sequence and passes KV around a ring as attention is computed, overlapping communication with compute to support near-arbitrary context; **Ulysses** uses an all-to-all over the sequence/head dimension. Module 6's communication-cost lens applies directly — it is another collective whose cost rides the interconnect.

---

## 7.7 Measuring honestly — the retrieval trap

This is the regime's signature methodology error, and it is the analog of Module 5's prefix-caching-on-ShareGPT and Module 3's saturated-benchmark trap. **Eviction and sparse methods often look ~free on perplexity and average long-document QA, while catastrophically failing retrieval** — because the token the task needs was the one evicted or never attended. A method can show 90% KV savings and excellent perplexity and **0% needle retrieval** in the evicted region.

Therefore you must measure on **retrieval-sensitive tasks** — **needle-in-haystack** (a fact planted at a known depth in the full context) and long-document QA — not perplexity alone. KV quantization (§7.3), which keeps all tokens, typically survives these; aggressive eviction does not. Report retrieval, or you will ship a model that silently forgets the middle of every document.

---

## 7.8 Where the frontier is now

The field has shifted from **post-hoc inference tricks toward native architectural solutions**:

- **Trainable sparse attention** (NSA, MoBA, 2025) replacing post-hoc eviction.
- **SSM / attention hybrids** (Jamba-style) and frontier models adding linear/SSM layers for bounded-memory long context.
- **1M+ token native context windows** becoming common, with **MLA** (Module 2) serving as a built-in long-context KV compressor.
- **Cross-instance KV reuse** for long *shared* contexts (Mooncake, Module 5) — a 200-page document attended by many queries is a shared prefix.

The trajectory: StreamingLLM/H2O (post-hoc eviction, 2023) → KV quantization (2024) → **trainable sparse attention and SSM hybrids (2025)** — long context is increasingly solved in the *architecture*, not only at serve time.

---

## 7.9 The picture to carry forward

- Long context is a **distinct regime**: KV memory, KV bandwidth, and attention compute all blow up (Modules 1–2 made manifest).
- Four composable relaxations: **eviction** (store a subset), **KV quantization** (store cheaply — Module 3), **sparse attention** (attend selectively), **linear/SSM** (replace the mechanism).
- **Sequence parallelism** (Module 6) is the fallback when one GPU still can't hold it.
- **Measure retrieval, not perplexity** — eviction's savings are a lie on retrieval; KV quantization is safer.
- The frontier is **native** (trainable sparse, SSM hybrids), not post-hoc.

---

## Going Deeper (appendix) — linear attention, SSMs, and the recall tradeoff

Softmax attention computes, for query `t`, `oₜ = Σ_{i≤t} softmax(qₜ·kᵢ) vᵢ` — `O(t)` work and `O(t)` stored KV. **Linear attention** drops the softmax for a kernel feature map `φ`, so `oₜ = φ(qₜ)ᵀ Sₜ` where `Sₜ = Sₜ₋₁ + φ(kₜ)vₜᵀ` is a **fixed-size state** updated recurrently: `O(1)` per token, `O(1)` memory — no growing KV. **SSMs (Mamba-2)** generalize this to a selective (input-dependent) linear recurrence, trained efficiently with a parallel scan (so training is parallel like attention, while inference is a constant-memory recurrence).

The fundamental tradeoff: a constant-size state `S` must **compress all of history** into fixed dimensions, so it cannot losslessly recall an arbitrary earlier token — whereas attention keeps every token's K/V and can attend to any of them exactly. This is why pure SSMs underperform attention on retrieval/in-context-recall, and why **hybrids** keep a few full-attention layers (the recall backbone) among many SSM layers (the efficiency bulk). The training–inference asymmetry — parallel-scan training, recurrent constant-memory serving — is the practical appeal for long context.

---

## Lab 7 — Push to the wall, recover headroom, and measure retrieval honestly

**Context — what this builds on (checklist).** *Reuse:* `mem_estimate` (M0/M2) to predict the long-context OOM; the `common/traffic` long-doc-QA builder plus a **needle-in-haystack** generator; `common/eval` (M3) for scoring; the M4 open-loop generator; M6 sequence parallelism. *Test the new insight:* the accuracy-vs-context-vs-memory tradeoff and the retrieval-vs-perplexity divergence — the distinctive long-context content. *Exercise prior quantities:* M2 KV law (predict OOM), M1 crossover (attention dominating), M3 KV quant, M5 eviction, M6 sequence parallelism. *Frontier:* a native sparse / SSM-hybrid model. *Consistency:* score **needle-in-haystack**, not perplexity alone.

1. **Hit the wall (M1/M2).** Push context up on long-doc-QA until OOM; confirm the OOM matches `mem_estimate`'s KV prediction (M2), and profile to show **attention compute now dominates** the weight reads (M1's crossover). *Confirms: §7.1 — the regime.*

2. **Recover headroom, three ways.** Apply (a) **eviction** (StreamingLLM sinks + window / H2O), (b) **KV quantization** (FP8→INT4 KV, M3), (c) **sparse attention** if available; record KV recovered and context extension for each. *Confirms: §7.2–7.4.*

3. **The retrieval trap (the honesty centerpiece).** Score eviction on **both** average long-doc-QA/perplexity **and needle-in-haystack with the needle in the evicted region**. Show eviction looking ~free on perplexity but **failing needle retrieval**; contrast with KV quantization, which keeps all tokens and survives. *Confirms: §7.7 — perplexity lies; KV quant is safer for retrieval.*

4. **(Frontier) native vs post-hoc.** Run a native sparse-attention or SSM-hybrid long-context model against post-hoc eviction on the same retrieval task; see whether the trained approach retains retrieval that eviction loses. *Confirms: §7.8.*

5. **(M6 payoff) sequence parallelism.** For a context that exceeds one GPU even after quant+eviction, enable Ring-Attention-style sequence parallelism and show context extending across GPUs at a communication cost (M6 lens). *Confirms: §7.6.*

**Deliverable:** an **accuracy × context × memory** table across {full, eviction, KV-quant, sparse}, with accuracy reported **both** as average long-doc-QA **and** needle-in-haystack retrieval — so eviction's retrieval failure is visible beside its memory win — plus the OOM-prediction check (M2) and the sequence-parallelism context extension (M6). **Mastery test — defend in one sentence each:** *why long context is a distinct regime (which three resources blow up); what each of the four techniques relaxes; why eviction can be ~free on perplexity yet fail needle retrieval, and why KV quantization is safer; when you need sequence parallelism; and the recall weakness of SSMs that motivates hybrids.* *Feeds:* Module 8 (disaggregation, where long prefill especially benefits from separate hardware).

**Reading:** Zhang et al., *H2O* (2023); Xiao et al., *StreamingLLM* (2023); *KIVI* (2024) for KV quantization. Current frontier: *NSA* (DeepSeek, 2025) and *MoBA* (2025) for trainable sparse attention; *Mamba-2* (2024) and a hybrid (e.g. *Jamba*) for SSMs; *Ring Attention* (2023) for sequence parallelism.