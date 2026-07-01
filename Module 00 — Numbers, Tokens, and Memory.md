# Module 0 — Numbers, Tokens, and Memory

*The units the rest of the course is counted in. Short, but everything downstream is denominated in these terms. Read before Module 1.*

---

## In plain English

**Why this matters.** Before you can answer "will this model fit on my GPU?" or "why is it so slow?", you have to know what you're actually counting. Get the units wrong and every cost, speed, and capacity estimate after this is wrong too.

**What this module gives you.** The three things everything else is measured in: *tokens* (the unit of work), the *byte-size of a number* (precision), and what actually fills up GPU memory when you serve.

**How it works (the intuition).** A model is just billions of numbers. How many bytes each number takes, times how many there are, is your memory bill — plus a surprisingly large scratchpad (the "KV cache") that grows with every word of every conversation. This module teaches you to do that arithmetic in your head.

---

## 0.1 The token is the unit of work

An LLM does not see characters or words; it sees **tokens** — subword fragments produced by a tokenizer (typically byte-pair encoding or a variant). A token is usually a few characters. Useful rules of thumb for English:

- ~4 characters per token, ~0.75 words per token → **1,000 tokens ≈ 750 words**.
- Code, non-Latin scripts, and rare strings tokenize less efficiently (more tokens per character).

We count in tokens because the two things this course cares about — **compute (FLOPs)** and **memory (KV cache)** — both scale with token count, and so does every price list and throughput number you will read. Two token populations matter and behave differently (Module 1 explains why):

- **Prompt / input tokens** — consumed during *prefill*, processed in parallel.
- **Generated / output tokens** — produced during *decode*, one at a time.

**Context length** is the maximum number of tokens (prompt + generated) the model can attend over at once. It is the variable that, together with batch size, drives KV-cache memory (§0.4).

---

## 0.2 How a number is stored

A model parameter is a single real number, stored in a fixed-width format. The format determines two things at once: **how many bytes it costs** and **how much numerical fidelity it keeps**. A floating-point format splits its bits into *sign*, *exponent* (dynamic range), and *mantissa* (precision).

| Format | Bits | Bytes | Sign/Exp/Mantissa | Dynamic range | Where it's used |
|---|---|---|---|---|---|
| FP32 | 32 | 4 | 1 / 8 / 23 | ~1e±38 | reference / accumulation; rarely the storage format for LLM weights |
| TF32¹ | 19² | (stored as 4) | 1 / 8 / 10 | ~1e±38 | NVIDIA tensor-core *compute* format, not a storage format |
| FP16 | 16 | 2 | 1 / 5 / 10 | ~±65,504 | classic mixed precision; **narrow range → overflow risk** |
| BF16 | 16 | 2 | 1 / 8 / 7 | ~1e±38 | **today's default** for training and inference |
| FP8 (E4M3) | 8 | 1 | 1 / 4 / 3 | ~±448 | inference (and increasingly training); needs scaling |
| FP8 (E5M2) | 8 | 1 | 1 / 5 / 2 | ~±57,344 | the higher-range FP8 variant (gradients) |
| INT8 | 8 | 1 | integer + scale (+ zero-point) | per-tensor/-group scale | post-training quantization |
| INT4 / NF4 | 4 | 0.5 | integer + group scale | per-group scale | GPTQ / AWQ / QLoRA |

¹ TF32 and FP16 are easy to conflate: TF32 is what the tensor cores *compute* in by default on Ampere+, but tensors are still **stored** in FP16/BF16/FP32. ² TF32 uses 19 bits internally but occupies an FP32 slot in memory.

The single most important design lesson here: **BF16 beat FP16 as the default because dynamic range matters more than mantissa precision for deep learning.** BF16 keeps FP32's 8-bit exponent (so activations and gradients rarely overflow) at the cost of mantissa bits. FP16's wider mantissa but narrow exponent is exactly the wrong trade for training stability — hence loss-scaling hacks. Internalize "range > precision" and the format zoo stops being arbitrary.

---

## 0.3 The core translation: parameters ↔ bytes

The whole reason §0.2 matters operationally:

```
weight_bytes = N_params × bytes_per_param
```

and the rule of thumb you should be able to do in your head:

```
weights (GB) ≈ N_params (billions) × bytes_per_param
            =  ×4  (FP32)
               ×2  (FP16 / BF16)
               ×1  (FP8 / INT8)
               ×0.5 (INT4)
```

Worked, weights only:

| Model | FP16/BF16 | INT8 | INT4 |
|---|---|---|---|
| 7B | 14 GB | 7 GB | 3.5 GB |
| 13B | 26 GB | 13 GB | 6.5 GB |
| 70B | 140 GB | 70 GB | 35 GB |

This is the calculation behind every "does it fit?" question:

- **70B in BF16 = 140 GB** → does not fit one 80 GB GPU; needs ≥ 2 (Module 6).
- **70B in INT4 = 35 GB** → fits a single 48 GB card, with room left for KV.
- **7B in BF16 = 14 GB** → fits a 16 GB card, but barely — almost no room for KV cache or activations, so concurrency will be tiny.

That last bullet is the trap: weights fitting is necessary, not sufficient. Serving needs more.

---

## 0.4 What actually sits in GPU memory at serve time

```
total ≈ weights + KV cache + activations + framework/CUDA overhead
```

- **Weights** — §0.3, fixed.
- **KV cache** — the dominant *variable* term, and the one that surprises people. Per token:
  ```
  KV_bytes_per_token = 2 × n_layers × n_kv_heads × head_dim × kv_dtype_bytes
  ```
  (The leading 2 is for K and V. Note the KV cache has its *own* dtype — often FP16, increasingly FP8.) It grows linearly with **context length × batch size**.

  *Worked (Llama-3-8B-class: 32 layers, 8 KV heads via GQA, head_dim 128, FP16):*
  `2 × 32 × 8 × 128 × 2 = 131,072 bytes/token ≈ 0.125 MB/token`.
  At 8K context that is **1 GB per sequence**; at 32 concurrent sequences, **32 GB of KV** — more than twice the 14 GB of weights. On long contexts and high concurrency, KV cache, not weights, is the binding constraint. (Module 2 develops this; Module 5 is how serving engines manage it.)
- **Activations** — transient working memory for the current forward pass; small relative to KV cache during inference.
- **Overhead** — CUDA context, framework buffers, fragmentation; budget ~1–2 GB.

The practical sizing inequality that falls out, and which you will use in every later module:

```
usable_KV_memory ≈ VRAM − weights − overhead
max_concurrency  ≈ usable_KV_memory ÷ (KV_bytes_per_token × context_length)
```

This is why a 7B model on a 16 GB card can serve almost no concurrent users despite "fitting," and why quantizing weights (freeing KV room) raises throughput, not just capacity.

---

## 0.5 What gets quantized — and what doesn't

Quantization is not monolithic. Three distinct things can be reduced in precision, with different effects:

- **Weights** — the most common target and the biggest *latency* win on a bandwidth-bound workload (Module 1 explains why fewer weight-bytes means faster decode, not merely smaller).
- **Activations** — harder, because activations contain large outliers; weight-activation quantization (e.g. for INT8 GEMMs) must handle them (SmoothQuant-style methods).
- **KV cache** — quantizing KV to FP8/INT8 directly attacks the §0.4 term that bounds concurrency, often the highest-leverage choice for long-context serving.

And it is rarely uniform: precision is assigned **group-wise** (a separate scale per block of weights), with sensitive layers (often the first/last, and some attention projections) kept at higher precision — *mixed precision*. The engineering question is therefore never "quantize or not" but "**quality per byte**," which Module 3 measures rigorously on public benchmarks.

---

## Going Deeper (appendix) — numerical precision in the dot product

*Assumes basic numerical analysis. The main path never depends on this; it justifies why §0.2's formats behave as they do.*

**FP32 accumulation.** A tensor-core matmul multiplies low-precision inputs (BF16/FP16/FP8) but accumulates the running sum in **FP32**. The reason is *swamping*: when a small partial product `p` is added to a large running sum `S`, the sum is rounded to the accumulator's precision; if the accumulator were FP16 (~3 decimal digits), any `p` below `S · 2⁻¹¹` is lost entirely. Keeping `S` in FP32 (~7 digits) preserves those contributions across a long reduction. This is why "compute in FP16, accumulate in FP32" is the universal MMA contract — and why a tensor's *storage* dtype and its *accumulation* dtype are different numbers.

**Dot-product error growth.** Summing `N` products each with unit roundoff `u`, naive left-to-right accumulation has worst-case relative error bounded by `≈ N·u` — it grows with the reduction length. **Pairwise (tree) summation** cuts this to `≈ log₂N · u`. At LLM scale (hidden dims of thousands, attention over long sequences) the difference is real, which is why good kernels reduce in a tree — and why the floating-point *result* of a batched reduction depends on its order, the non-determinism Module 12 §12.5 returns to.

**Stochastic rounding.** Round-to-nearest is *biased* when you accumulate many tiny increments: each increment below half a ULP rounds to zero, so a long stream of them never moves the value — the accumulator stalls. **Stochastic rounding** rounds up or down with probability proportional to the fractional position, so it is *unbiased in expectation* and small increments accumulate correctly on average. It is what makes very-low-precision (FP8 and below) *training* tractable; at inference — frozen weights, no accumulation of updates — it matters far less. *Reading:* Micikevicius et al., *Mixed Precision Training* (2017); the OCP *FP8 Formats* spec.

---

## Lab 0 — Build the memory calculator

**Goal:** turn §0.3–0.4 into a tool you trust, and confront the gap between theory and a real allocator.

1. Implement `mem_estimate(model_config, weight_dtype, kv_dtype, context_len, batch)` returning predicted VRAM, broken into weights / KV / overhead.
2. Produce a params↔bytes table (§0.3) for three open models (e.g. Llama-3.1-8B, Qwen2.5-14B, Mistral-7B) across BF16 / INT8 / INT4.
3. Load one model in vLLM at two precisions; read the actually-reserved VRAM and the KV-cache blocks it reports; compare to your estimate.
4. Explain any discrepancy (allocator rounding, paging block size, CUDA-graph capture, reserved fragmentation).

**Deliverable:** `mem_estimate()` validated against measured VRAM within ~15%, plus the params↔bytes table. *Feeds:* Module 2 (KV budget) and the card-sizing decision in every later module.

**Reading:** Micikevicius et al., *Mixed Precision Training* (2017) — the origin of the FP16/BF16 training regime; the OCP *FP8 Formats* specification — E4M3/E5M2 and why two FP8s exist.