# Module 3 — Quantization: Shrinking the Numerator

*Prerequisites: Module 0 (`bytes = params × bytes_per_param`; the dtype table), Module 1 (decode latency ≳ `weight_bytes / B`; prefill is compute-bound), Module 2 (shrinking the KV footprint by architecture). This module shrinks the **weight** bytes by precision — the most direct lever on Module 1's numerator — and establishes the often-missed fact that quantization speeds up decode and prefill by **two different mechanisms**.*

---

## In plain English

**Why this matters.** The most direct way to make a model faster *and* smaller is to store its numbers in fewer bits. But squeeze too hard and the model gets subtly dumber — on exactly the hard tasks you care about, while easy benchmarks still look perfect and hide the damage.

**What this module gives you.** How low-precision storage (INT4, FP8) speeds things up — including the crucial twist that it speeds up "typing" (decode) and "reading" (prefill) by two *different* mechanisms — and how to measure the quality you actually give up.

**How it works (the intuition).** Think of it like compressing the model the way a JPEG compresses a photo: smaller and faster to load, but lossy. The whole art is compressing the parts that don't matter while protecting the few rare "outlier" values that do.

---

## 3.1 The lever: quantization is the direct attack on Module 1's numerator

Module 1 gave the decode latency floor: `time per token ≳ weight_bytes / B`. The numerator is `N_params × bytes_per_param` (Module 0). Architecture (Module 2) attacked the *KV* term; quantization attacks the *weight* term, and it does so at the most direct point — `bytes_per_param`. Going FP16 → INT8 halves the bytes read per decode step; FP16 → INT4 quarters them. On a bandwidth-bound workload that translates almost directly into decode speedup, and it lifts the entire context-throughput curve you measured in Module 2.

A subtlety that trips people up, and that the rest of the module rests on: **weight-only quantization speeds up decode even when the matmul still computes in FP16.** A typical INT4 weight-only kernel reads 4-bit weights from HBM, dequantizes them to FP16 *on-chip*, and runs an FP16 GEMM. There is no arithmetic speedup — but decode is memory-bound, so the win was never in the arithmetic; it was in moving 4× fewer bytes from HBM. The bottleneck you relieved is exactly the one Module 1 identified.

That same subtlety is why quantization's effect on *prefill* is completely different.

---

## 3.2 The decode-vs-prefill split (two mechanisms)

Module 1 established that prefill and decode sit on opposite sides of the roofline. Quantization therefore helps them by opposite mechanisms — and this determines which scheme you should use.

- **Decode is memory-bound** → the lever is **bytes moved**. Reducing `bytes_per_param` speeds decode *regardless of how the GEMM computes*. **Weight-only** low-bit quantization (INT4 via GPTQ/AWQ) is sufficient: it shrinks the HBM read, computes in FP16, and still wins.
- **Prefill is compute-bound** → the lever is **arithmetic throughput**, and fewer weight-bytes do *nothing* for a kernel that is not bandwidth-limited. To speed prefill you must make the **tensor cores compute in lower precision**, which requires quantizing **both weights and activations** to a format the hardware executes natively (FP8, INT8). Per Appendix A, H100 FP8 MMA runs at ≈ 2× BF16 throughput — that, not byte count, is the prefill speedup.

The practical consequence is a clean decision rule:

| Workload shape | Bound by | Use | Why |
|---|---|---|---|
| Decode-heavy (short prompt, long output) | bandwidth | **weight-only INT4** (GPTQ/AWQ) | relieves the HBM read; FP16 compute is fine |
| Prefill-heavy (long prompt, short output: RAG, classification, reranking) | compute | **FP8 weight+activation** | faster low-precision MMA |
| Mixed | both | FP8 (helps both) or INT4-weights + FP8-compute hybrids | covers both levers |

So "is INT4 faster than FP16?" has no single answer — it is *much* faster for decode and *barely* faster (sometimes slower, from dequant overhead) for prefill. The lab measures exactly this.

**One honest refinement — the real axis is arithmetic intensity, not phase.** Module 1 showed intensity rises with batch, so the split above is cleanest at *low batch*. At high *serving* batch, decode itself drifts toward compute-bound, and the rule softens: weight-only INT4's decode advantage shrinks, and on a now-compute-bound decode kernel the on-chip dequantization overhead can make weight-only INT4 *slower* than FP16. The precise statement is therefore: **weight-only low-bit wins in the memory-bound regime (low-to-moderate intensity), and native low-precision compute (FP8) wins in the compute-bound regime (prefill, or decode at high batch).** Read "decode vs prefill" as a proxy for "low vs high arithmetic intensity," and the rule stays consistent with Module 1. The lab makes you find the batch where it flips.

---

## 3.3 Why inference quantization is not training quantization

Module 0 introduced mixed-precision *training* (FP32 master weights, BF16 forward/backward, FP32 accumulation). Inference quantization is a different problem with a different acceptable aggressiveness:

- **Training** must preserve *optimization dynamics* over millions of steps — gradients and optimizer state are fragile, which is why master weights stay FP32 and you keep range (BF16). The constraint is stability.
- **Inference** has *frozen weights and no gradients*. The only thing to preserve is the *output* of a single forward pass. There is no training loop to destabilize, so you can be far more aggressive — INT4 weights are routine at inference and unthinkable for the training master copy. The goal shifts from "don't break learning" to "keep the output distribution close enough," which is a per-task accuracy question (§3.5), not a stability one.

---

## 3.4 The hard part: outliers, and the responses to them

Quantizing a tensor maps floats to a small integer grid through a **scale** `s` (and possibly a zero-point): `q = round(x/s)`, `x ≈ s·q`. The scale must cover the range, so it is set by the largest magnitude in the group.

The problem unique to LLMs: **activations (and some weights) contain large outliers** — a handful of channels with values 10–100× the rest, which emerge as models scale. If one outlier sets the scale, the grid spacing becomes coarse and the overwhelming majority of normal values collapse onto a few levels — catastrophic error. Naive round-to-nearest at INT4 fails for exactly this reason. Five techniques attack it from different angles:

| Scheme | Quantizes | Key idea | Buys |
|---|---|---|---|
| **Group/block scaling** | weights | a separate scale per small group (e.g. 128 weights), so an outlier only coarsens its own group | the baseline that makes INT4 viable at all (~0.5–1 bit/weight overhead) |
| **AWQ** | weights (activation-*aware*) | not all weights matter equally; protect the weight channels tied to high-magnitude *activation* channels by scaling them before quantizing | strong INT4 weights, calibration-light |
| **SmoothQuant** | weights **+** activations | migrate quantization difficulty from activations to weights via a mathematically-equivalent per-channel transform (divide activations, multiply weights), so activations become smooth enough to quantize | enables INT8 weight+activation (→ faster prefill, §3.2) |
| **GPTQ** | weights | minimize the *layer output error*, not per-weight error: quantize column-by-column and update the remaining columns to compensate, using second-order (Hessian) information | best-in-class INT4 weights (derivation in Going Deeper) |
| **Rotation (QuaRot, SpinQuant)** | weights **+** activations **+** KV | apply an orthogonal rotation (Hadamard or learned) that *spreads* each outlier across many channels — a rotation preserves the layer's output but eliminates the spikes that ruin per-channel scaling | the current path to **4-bit weight+activation+KV** |

The taxonomy is the point: group scaling handles outliers by *locality*, AWQ by *protecting salient weights*, SmoothQuant by *moving the difficulty* to weights, GPTQ by *compensating the error*, and rotation by *destroying the outlier structure itself*. They are not competitors so much as different answers to the same outlier problem, and they compose (rotation is often applied *before* GPTQ/AWQ). The 2022–23 trio (group/AWQ/SmoothQuant/GPTQ) is the foundation; **rotation-based methods (2024) are the current frontier**, because they are what make aggressive 4-bit *activation* and *KV* quantization — not just weights — actually work.

---

## 3.5 The quality-per-byte curve — and why aggregate scores lie

| Precision | Bytes/param | Typical quality | Note |
|---|---|---|---|
| FP16/BF16 | 2 | reference | baseline |
| FP8 | 1 | near-lossless on most models | **the current production default**; + faster prefill (native MMA) |
| INT8 | 1 | near-lossless with SmoothQuant/group scaling | |
| INT4 | 0.5 | small but **non-uniform** loss | the workhorse for decode-bound serving |
| **FP4 (NVFP4 / MXFP4)** | 0.5 | competitive with INT4, improving fast | **the frontier**: hardware-native on Blackwell; microscaling (a shared block scale) is what M0 flagged |
| 2–3 bit | <0.5 | significant loss | research |

This reconciles the thread M0 opened: the **microscaling formats (MXFP8/MXFP4)** previewed there are precisely the current direction — a small *block-shared* scale baked into the format, executed natively by Blackwell-class tensor cores, so 4-bit stops being a software trick and becomes a hardware datatype. FP8 is already the default many labs train and serve in; FP4 via microscaling (plus the rotation methods of §3.4 to tame activation outliers) is where production is heading.

The lesson that matters operationally: **low-bit degradation is not uniform across tasks, and aggregate benchmarks hide it.** Easy, high-redundancy tasks (e.g. broad multiple-choice) look essentially unchanged, while *reasoning*, *code*, *instruction-following*, and *long-context* tasks degrade measurably. A single averaged score, or a benchmark that happens to be easy, will tell you INT4 is "free" when it is silently failing the workloads you most care about — which is why the lab measures **per task**. (Caveat for the suite itself: MMLU/GSM8K/HumanEval are now largely *saturated* on strong models, so on a frontier model use harder discriminators — MMLU-Pro, GPQA, AIME/MATH, LiveCodeBench — or the damage hides in the ceiling.)

A third quantization target completes the picture: alongside weights and activations, the **KV cache** can be quantized (FP8/INT4 KV) — a direct attack on Module 2's binding term, developed in Module 7. So quantization touches all three of the serve-time memory consumers from Module 0: weights (here), activations (here, for prefill), and KV cache (Module 7).

---

## 3.6 The picture to carry forward

- Quantization is the most direct lever on Module 1's decode numerator; it raises the whole Module 2 context-throughput curve.
- **Low and high arithmetic intensity are sped up by different mechanisms** — bandwidth (weight-only suffices, e.g. low-batch decode) vs MMA throughput (needs native low-precision weights+activations, e.g. prefill and high-batch decode). Read "decode vs prefill" as a proxy for intensity, per Module 1.
- The enemy is **outliers**; group scaling, AWQ, SmoothQuant, and GPTQ are the foundational responses, and **rotation methods (QuaRot/SpinQuant) are the current frontier** that unlock 4-bit activations and KV.
- The format frontier is **FP8 (production default) → FP4 via microscaling (NVFP4/MXFP4)**, the direction Module 0 flagged.
- Quality loss is **non-uniform** — measure per task (on un-saturated benchmarks) or be misled.

Module 5 manages these (now smaller) weights and caches in a real engine; Module 7 applies quantization to the KV cache for long context.

---

## Going Deeper (appendix) — GPTQ and the second-order (OBQ) view

GPTQ descends from Optimal Brain Surgeon/Quantization. Frame layer quantization as minimizing the output reconstruction error on calibration inputs `X`:

```
minimize  ‖ W X − Ŵ X ‖²₂      over quantized Ŵ
```

This is quadratic in the weight perturbation, with Hessian

```
H = 2 X Xᵀ
```

— and crucially `H` depends only on the inputs, so it is **shared across all rows** of the weight matrix (computed once per layer). The OBS result gives, for quantizing a single weight `w_q` to its grid value `quant(w_q)`, both the optimal compensating update to the remaining weights and the resulting loss increase:

```
δ_remaining = − (w_q − quant(w_q)) / [H⁻¹]_qq · H⁻¹_{:,q}
Δloss        =   (w_q − quant(w_q))² / [H⁻¹]_qq
```

You greedily pick the weight with smallest `Δloss`, quantize it, apply `δ` to compensate, and repeat. OBQ does exactly this but is too slow at LLM scale. GPTQ makes it tractable with three engineering moves: (1) quantize columns in a **fixed order** (the greedy ordering turns out to matter little), enabling all rows to be processed in lockstep; (2) **lazy batched updates** to keep the GPU busy; (3) a **Cholesky** reformulation of the sequence of `H⁻¹` updates for numerical stability. The result quantizes billions of weights from a few hundred calibration samples and substantially beats round-to-nearest, especially at INT4 — the empirical reason GPTQ became a default. We cite the derivation (Frantar et al.) rather than reproving OBS; the takeaway is that *compensating for quantization error using second-order information* is what separates a good INT4 from a broken one.

---

## Lab 3 — Quantize, and prove the mechanism, not just the speedup

**Context — what this builds on (the checklist).** *Reuse:* the extended `mem_estimate` from Module 2 (quantization changes the `bytes_per_param` term it already models) and **Module 1's prefill/decode phase-isolation** method and roofline; the accuracy scorers in `common/eval`. *Test the new insight:* the centerpiece is the §3.2 **decode-vs-prefill split** — not a generic "measure tok/s." *Exercise prior quantities:* M0 bytes (footprint), M1 roofline (where each speedup lands), M2's context curve (which should lift).

1. **Footprint (reuse the M0/M2 tool).** Serve one model (e.g. Llama-3.1-8B) at FP16, INT8, INT4 (GPTQ/AWQ), and FP8; confirm `mem_estimate` predicts each weight footprint from the `bytes_per_param` term and validate against measured VRAM. *Confirms: quantization shrinks the Module 1 numerator by the predicted factor.*

2. **The mechanism split — the experiment that matters (reuse M1 isolation, and find where it flips).** Using Module 1's phase isolation (long-prompt/1-token for **prefill** throughput; short-prompt/long-output for **decode** throughput), measure prefill tok/s *and* decode tok/s for three configs: FP16 baseline, **weight-only INT4**, and **FP8 weight+activation**. At **low batch**, confirm the clean §3.2 prediction: weight-only INT4 gives a large *decode* speedup but ~no *prefill* speedup; FP8 speeds *both*. Then **sweep decode batch upward** and find the point where weight-only INT4's decode advantage erodes (and may invert as it becomes compute-bound and pays dequant overhead) — this is the §3.2 refinement made visible, and it ties the result back to Module 1's intensity curve. Place each gain on the roofline. *Confirms: the headline claim, including its boundary condition.*

3. **Per-task quality (reuse `common/eval`).** Run FP16/AWQ/GPTQ/FP8 (and FP4 if your hardware supports it) across MMLU, GSM8K, HumanEval, and IFEval — and, on a strong model, a harder discriminator (MMLU-Pro/GPQA/LiveCodeBench), since the easy suite saturates. Tabulate per-task — do **not** average. Show low-bit looking ~free on the easy task while degrading on reasoning/code/instruction-following. *Confirms: §3.5 — degradation is non-uniform and aggregate (or saturated) scores hide it.*

4. **(Optional) See the outliers.** Hook a forward pass and plot per-channel activation magnitude for one layer; identify the outlier channels that motivate group scaling / AWQ / SmoothQuant. *Confirms: §3.4 — the problem is concrete, not abstract.*

**Deliverable:** a **quality × throughput × memory** table whose throughput column is **split into prefill and decode**, and whose quality column is **per task** — so both the mechanism split (step 2) and the non-uniform degradation (step 3) are visible at a glance — plus a one-line workload→scheme recommendation derived from §3.2 *and the batch at which it flips*. **Mastery test — defend in one sentence each:** *why weight-only INT4 speeds decode but not prefill at low batch, and why that can invert at high batch (intensity, per Module 1); why FP8 speeds both; where low-bit's quality cost concentrates and why an averaged or saturated benchmark hides it; and what outlier problem group scaling, GPTQ, and rotation each solve.* *Feeds:* Module 5 (managing the quantized weights/cache) and Module 7 (KV-cache quantization).

**Reading:** Frantar et al., *GPTQ* (2022); Lin et al., *AWQ* (2023); Xiao et al., *SmoothQuant* (2022). Current frontier: Ashkboos et al., *QuaRot* (2024) and *SpinQuant* (2024) for rotation-based 4-bit; the NVFP4/MXFP4 (microscaling, OCP) formats for hardware-native FP4. Background: Dettmers et al., *LLM.int8()* (2022) for outlier emergence.