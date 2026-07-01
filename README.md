# Efficient LLM Inference & Serving Systems

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) ![Engines](https://img.shields.io/badge/engines-vLLM%20%7C%20SGLang-orange.svg) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

A hands-on, first-principles course on how LLMs are actually served in production — from the roofline up to vLLM & SGLang, with GPU labs on open models and every result reproducible.

---

## Why generating text barely uses your GPU

To produce **one** token, a model reads *every* weight it has out of GPU memory — billions of numbers — and does almost no math with each. A GPU crunches numbers far faster than it can pull them from memory, so decoding isn't limited by how fast the chip computes. **It's limited by how fast it can read memory.** One sequence at a time, the expensive math units you paid for sit almost entirely idle, waiting for weights to arrive:

```
   math a GPU can do per byte it reads from memory   ██████████████████████████████
   what generating one token actually uses           ▏  ← about 1 part in 300
```

That single imbalance is the spine of the whole course. **Batching, KV caches, quantization, paging, parallelism, speculative decoding — almost every technique below is a different way to reclaim that idle capacity:**

| The question it forces | The technique that answers it | Module |
|---|---|---|
| Decode wastes FLOPs waiting on memory. How do we use them? | **Batching** — amortize one weight-read across many sequences | 4 |
| Batching demands memory for many KV caches. Why does it run out? | **KV-cache management / PagedAttention** | 2, 5 |
| The numerator (bytes per weight) is the direct lever. Shrink it? | **Quantization** — buys *latency*, not just capacity | 3 |
| One GPU can't hold the largest models. | **Tensor / pipeline / expert parallelism** | 6 |
| The context itself is too long for the KV cache. | **Long-context serving** (eviction, KV quant, sparse attention) | 7 |
| Decode is strictly sequential per sequence. Break that? | **Speculative decoding; chunked prefill; disaggregation** | 8 |
| The output must obey a schema/grammar, not merely read well. | **Structured / constrained decoding** | 9 |
| A single call becomes a multi-turn agent — what keeps it cheap? | **Tool-use / agentic serving** (cross-turn prefix reuse) | 10 |
| How is any of this exposed, kept up, and depended on? | **Frameworks, API, observability, resilience** | 11 |
| How do we *know* a change helped vs moved the bottleneck? | **Evaluation methodology** | 12 |

*Modules 9–12 add the layers a real deployment needs beyond raw efficiency — structured output, agentic serving, the API/gateway, honest benchmarking — non-negotiable for anyone self-hosting.*

<details>
<summary><b>The same idea, precisely — the roofline in numbers</b> (for readers who want the derivation)</summary>

<br>

Generating one token from a dense model with *N* parameters means reading ≈ all *N* weights from HBM (≈ 2*N* bytes in FP16) while doing only ≈ 2*N* FLOPs — an **arithmetic intensity of ≈ 1 FLOP/byte**. That is hundreds of times below the *ridge point* where an accelerator stops being memory-bound: an H100 does ~990 BF16 TFLOP/s over ~3.35 TB/s of bandwidth ≈ **~295 FLOP/byte**, so a lone decode stream reaches only ≈ 1/295 ≈ **0.3%** of peak compute (that's the "1 part in 300" above). Decode lives deep in the bandwidth-bound regime; **prefill** — digesting the prompt — sits on the compute-bound side, which is why the two optimize in opposite directions. The numbers are real silicon; **Appendix A** derives where peak compute *P* and bandwidth *B* come from.

```
 attainable compute (log)
                                            ┌── peak ≈ 990 TFLOP/s (BF16)
                              ridge ──►╱─────┴──────────────  compute-bound
                            ≈ 295     ╱          region  →  PREFILL
                          FLOP/byte  ╱
                                    ╱
                                  ╱
                                ╱     the roofline
                              ╱
                            ╱       bandwidth-bound region  →  DECODE
                          ╱
                        ╱
                      ╱
      decode ●──────╱   ≈ 1 FLOP/byte  —  ~300× below the ridge:
                        the FLOPs you paid for sit idle, waiting on HBM
 └──────────────────────────────────────────────────────────────────►
    arithmetic intensity  (FLOP/byte, log)      1  ····  295  ····  ∞
```

</details>

> **★ Star it** if you want inference reasoned from physics instead of folklore. New labs and real-hardware numbers land as contributors run them — the star keeps this one keystroke away.

---

## Audience and prerequisites

Designed for rusty human individuals in the crazy evolving AI rat race and looking for self-hosting LLM models in their own bunker.

**Assumed for the main path:** comfort with Python and Linux; basic deep-learning literacy; computer-architecture fundamentals — memory hierarchy, bandwidth vs latency, the **roofline model**; basic probability.

**Assumed only for specific "Going Deeper" appendices** (each flags this, and the main path never depends on them): numerical analysis (M0 error analysis), convex/quadratic optimization (M3 OBQ/GPTQ), queueing theory (M4 derivations), and deeper computer architecture (the hardware appendix). The heavy mathematics is deliberately quarantined in appendices so the critical path stays teachable to a mixed cohort.

---

## Course structure: two tiers

Every module has a **Main** thread — the derivations and labs a cohort must follow — and **most also have a Going Deeper** section (or linked appendix) with the heavier theory, proofs, and hardware detail for those who want it. A reader can complete the course on the Main thread alone; the appendices reward the PhD appetite without making the main line unteachable. Throughout, we **go deep on the durable things (the physics, the math, the architecture) and stay deliberately shallow on what rots** (specific SOTA models, framework APIs, leaderboard numbers).

---

## Learning outcomes

By the end, a student can:

1. Derive, from the roofline, whether an inference workload is compute- or memory-bound, and predict which optimization helps — including *when attention, not weights, becomes the bottleneck* at long context.
2. Compute the KV-cache memory budget for a model/context/batch and predict serving limits; explain the MHA→MQA→GQA→MLA progression as responses to it.
3. Quantify the quality–throughput–memory trade-off of quantization on standard benchmarks, and explain *why it helps decode and prefill by different mechanisms*.
4. Model a serving system as a queue (Little's law, the saturation knee, goodput) and reason about the latency–throughput frontier.
5. Derive the communication cost of tensor/pipeline/expert parallelism and choose among them from interconnect characteristics.
6. Explain and apply long-context techniques (KV eviction, KV quantization, sparse/linear attention) and their accuracy costs.
7. State and defend the speculative-decoding correctness argument and its expected-speedup law.
8. Guarantee schema-valid output with constrained decoding, and measure its *quality* cost — not just adherence.
9. Reason about agentic serving as a cross-turn prefix-reuse problem, and diagnose its two dominant cost failures (silent cache invalidation; tool-latency-gated GPU).
10. Implement a correct, observable serving gateway and validate graceful failover.
11. Design a benchmark that does not lie — correct metrics, realistic traffic, no coordinated omission, reported variance.

---

## Format and assessment

Each module pairs a **lecture/derivation** with a **lab** on open models and public data, plus a primary-source **reading**. Labs are short, individually GPU-cheap (most run on a single 24–48 GB GPU), and graded on the *measurement and analysis*, not on getting a "good" number.

A recurring lab discipline (also a transferable skill): GPU time is metered and idle time is wasted money. Labs are written so all setup, scripting, and analysis happen off the accelerator; the GPU is powered on only to execute a prepared sweep, then released. The shared harness in `common/` is built to run exactly this way — it drives a real engine over the OpenAI-compatible API, or, for development, an **offline simulator that needs no GPU**, so a whole lab can be written and dry-run on a laptop before the accelerator is ever powered on.

**Open models used throughout:** Llama-3.1-8B, Qwen2.5-7B/14B, Mistral-7B, and one mid-size MoE (Mixtral-class) for the parallelism module.
**Public benchmarks:** ShareGPT and LMSYS-Chat-1M for traffic; MMLU, GSM8K, HumanEval, IFEval for quality; a constructed long-document-QA set (one fixed public document as a shared prefix, many questions) for prefix-sharing and long-context labs; JSON-Schema-Bench-style tasks for structured output.

---

## The modern serving stack: vLLM and SGLang

The course runs on the two open-source engines that define current practice; students must be fluent in launching, configuring, and benchmarking **both**.

A research script (`transformers.generate()`) is *not* a serving system: one request at a time, padded batches, naive KV allocation, no scheduler. Production serving requires batching independent requests arriving at different times, managing KV memory dynamically, and meeting latency SLOs.

- **vLLM** — the throughput workhorse. Defining contribution **PagedAttention** (Module 5: KV cache managed like OS virtual memory in fixed blocks, near-zero fragmentation) plus **continuous batching** (Module 4). OpenAI-compatible API, tensor parallelism, most quantization formats, speculative decoding. `vllm serve <model> --enable-prefix-caching ...`.
- **SGLang** — prefix reuse and structured generation. Defining contribution **RadixAttention** (Module 5: automatic KV prefix sharing via a radix tree) plus first-class constrained/JSON decoding. Also OpenAI-compatible. `python -m sglang.launch_server --model <model> ...`.

Lineage the module order retraces: **Orca** (iteration-level batching, 2022) → **vLLM** (paged KV, 2023) → **SGLang** (radix sharing + structured, 2023–24). vLLM is the default engine; SGLang is the comparison, foregrounded in the prefix-sharing (M5) and structured-output (M9) labs; both sit behind one OpenAI-compatible interface so the gateway (M11) treats them uniformly. **Full notes:** `Appendix C — The Modern Serving Stack.md`.

---

## Modules

> Each module states **the question** (why it exists, given the previous answer), **key ideas** (the Main thread), **Going deeper** (the opt-in appendix), **reading**, **lab**, and **deliverable**.

### Part I — Inference as a systems workload

**Module 0 — Numbers, tokens, and memory** *(the units; short but load-bearing)*
*The question:* before we reason about speed or cost, what are we counting? *Key ideas:* tokens as the unit; FP32/FP16/BF16/FP8/INT8/INT4 layouts, ranges, byte costs, and why "range > precision" made BF16 the default; `bytes = params × bytes_per_param`; serve-time memory = weights + KV + activations + overhead; **microscaling formats (MXFP8/MXFP4)** as the current frontier. *Going deeper (appendix):* FP32 accumulation and dot-product error analysis; stochastic rounding. *Reading:* Micikevicius et al., *Mixed Precision Training* (2017); OCP *FP8 Formats*. *Lab:* a memory calculator predicting VRAM from (model, dtype, context, batch), validated against a loaded model within ~15%. *Deliverable:* `mem_estimate()` + params↔bytes table. *(Full notes: `Module 00 — Numbers, Tokens, and Memory.md`.)*

**Module 1 — The roofline of a transformer forward pass**
*The question:* why is generation slow at all when the GPU is rated for petaFLOPs? *Key ideas:* the forward pass as GEMMs; arithmetic intensity; **exact FLOP/memory accounting that separates parameter-GEMM cost from attention's `O(S·d)`-per-token term, deriving the long-context crossover where attention — not weights — binds**; prefill (compute-bound) vs decode (memory-bound) and their opposite optimization targets; the `weight_bytes / B` latency bound. *Reading:* Pope et al., *Efficiently Scaling Transformer Inference* (2022). *Lab:* profile prefill and decode separately on an 8B model; place both on a measured roofline; sweep batch and watch intensity climb. *Deliverable:* roofline plot + decode bandwidth-utilization + batch-sweep curve. *(Full notes: `Module 01 — The Roofline of a Transformer Forward Pass.md`.)*

**Module 2 — Attention, the KV cache, and where the memory goes**
*The question:* decode rereads history every step — does it have to? *Key ideas:* the KV cache as the O(n²)-recompute → O(n)-with-memory trade; the KV-size law; **the architecture line that responds to it: MHA → MQA → GQA → MLA (latent-KV compression)**; **FlashAttention as an IO-optimal kernel (tiling, SRAM residency)**. *Going deeper (appendix):* the FlashAttention IO-complexity derivation. *Reading:* Dao et al., *FlashAttention* (2022); the GQA paper (Ainslie et al., 2023); the MLA description (DeepSeek-V2, 2024). *Lab:* measure peak memory vs context and batch; fit the KV law; compare MHA vs GQA memory/throughput. *Deliverable:* `kv_budget()` validated against measured OOM within ~10%.

### Part II — Single-model efficiency

**Module 3 — Quantization: shrinking the numerator**
*The question:* the most direct attack on decode latency is fewer bytes per weight — what does it cost? *Key ideas:* why *inference* quantization targets latency (bandwidth-bound) while training quantization targets memory/compute; **the decode-vs-prefill split — quantization helps decode via bandwidth and prefill via faster low-precision MMA throughput, two different mechanisms**; weight-only vs weight-activation; **activation outliers, group/block scaling**, and the schemes that exploit them (GPTQ, AWQ, SmoothQuant); FP8/INT4 as points on a quality-per-byte curve. *Going deeper (appendix):* the **second-order error view — GPTQ's Hessian/OBQ derivation**. *Reading:* Frantar et al., *GPTQ* (2022); Lin et al., *AWQ* (2023); Xiao et al., *SmoothQuant* (2022). *Lab:* serve one model at FP16/AWQ/GPTQ/FP8; measure tokens/s, VRAM, and **per-benchmark** accuracy; find where reasoning silently degrades. *Deliverable:* quality×throughput×memory table with per-task degradation called out.

### Part III — Serving many requests

**Module 4 — Batching and scheduling** *(Going Deeper assumes queueing theory)*
*The question:* batching fills decode's idle FLOPs, but naive static batching stalls on the longest sequence — how do real systems schedule? *Key ideas:* arithmetic intensity as a function of batch size (and where KV-bandwidth re-binds it); **continuous (iteration-level) batching** as dynamic scheduling; the latency–throughput Pareto front. *Going deeper (appendix):* the **queueing-theoretic view — Little's law, the saturation knee, goodput, open-loop arrivals**. *Reading:* Yu et al., *Orca* (OSDI 2022). *Lab:* replay ShareGPT at rising request rates, static vs continuous batching; plot throughput, p50/p95/p99, and goodput under an SLO. *Deliverable:* latency–throughput curves and the rate at which static batching collapses.

**Module 5 — Serving-time memory management**
*The question:* batching needs many concurrent, growing KV caches; static allocation fragments. How is it managed? *Key ideas:* **PagedAttention** (virtual-memory-style paging; internal vs external fragmentation; copy-on-write); **prefix sharing / RadixAttention**. *Going deeper (appendix):* KV/prefix-cache **eviction policies** (radix LRU, sharing accounting). *Reading:* Kwon et al., *PagedAttention/vLLM* (SOSP 2023); Zheng et al., *SGLang/RadixAttention* (2023). *Lab:* on long-document-QA, prefix caching off vs on; effective batch with vs without paging. *Deliverable:* prefix-cache speedup and paging memory-efficiency gain.

### Part IV — Scaling out and speeding up

**Module 6 — Multi-GPU parallelism** *(Main includes communication-cost derivations)*
*The question:* the largest models exceed one GPU's HBM, and long contexts make it worse — how do we split a model, and what does splitting cost? *Key ideas:* **tensor** (intra-layer), **pipeline** (inter-layer), and **expert** (MoE) parallelism; **the communication cost made explicit — ring vs tree all-reduce, the two all-reduces per transformer block in TP, the pipeline bubble fraction `(p−1)/m`, and why TP demands NVLink (latency-bound small messages) while PP tolerates PCIe**; **the MoE all-to-all expert-routing pattern**. *Reading:* Shoeybi et al., *Megatron-LM* (2019); Fedus et al., *Switch Transformers* (2021). *Lab:* serve a model that won't fit on one GPU at TP=2 and TP=4; measure scaling and interconnect overhead; contrast dense vs MoE. *Deliverable:* parallel-scaling curve + identification of the binding resource (compute vs communication).

**Module 7 — Long-context serving**
*The question:* the KV cache grows with context × batch (Modules 0, 2); at 100K+ tokens it dominates memory and attention dominates compute (Module 1's crossover) — how do we serve long context at all? *Key ideas:* **KV-cache eviction** (H2O heavy-hitters; StreamingLLM attention sinks); **KV-cache quantization** as the direct attack on the binding term; **sparse and linear attention** alternatives and what they give up; the accuracy-vs-memory frontier and how to measure it honestly (long-document QA, needle-in-haystack). *Going deeper (appendix):* linear-attention/state-space formulations and their training–inference trade-offs. *Reading:* Zhang et al., *H2O* (2023); Xiao et al., *StreamingLLM* (2023); a KV-quantization paper (e.g. *KIVI*, 2024). *Lab:* push context to OOM, then recover headroom with eviction and KV quantization; measure the accuracy cost on long-document QA at each setting. *Deliverable:* an accuracy-vs-context-vs-memory table across eviction/quantization settings.

**Module 8 — Attacking the sequential dependency** *(Main includes the correctness proof)*
*The question:* even perfectly batched, a single sequence's decode is strictly serial — can we break the data dependency? *Key ideas:* **speculative decoding** — a draft model proposes *k* tokens, the target verifies in one memory-bound pass; **the correctness argument that modified rejection sampling preserves the target distribution exactly, and the expected-speedup law as a function of acceptance rate α and draft length k**; **chunked prefill**; **prefill–decode disaggregation** (Module 1's opposite profiles on separate hardware). *Reading:* Leviathan et al., *Speculative Decoding* (2023); a disaggregation paper (*DistServe* or *Splitwise*, 2024). *Lab:* measure speculative speedup vs draft acceptance rate; find where speculation stops paying. *Deliverable:* speedup-vs-acceptance curve and the regime where speculation helps.

### Part V — Structured output and agentic serving

**Module 9 — Structured / constrained decoding**
*The question:* applications need guaranteed structure (a JSON schema, a grammar, valid SQL), but a free-sampling model emits *mostly*-valid output that occasionally breaks the consumer — how do we guarantee validity, and what does the guarantee cost in serving? *Key ideas:* a decode-step intervention (the peer of Module 8's speculation, but for *correctness* not speed) — compile the constraint into an **FSM over the token vocabulary** and **mask disallowed logits** every step, so output is valid *by construction*; the **token–vocabulary alignment** problem as the real engineering (Outlines, XGrammar); the cost split — **compile (amortized by caching)** + **per-step mask**; the **batching** (per-sequence masks) and **speculative-decoding** (the draft must be grammar-aware or α collapses) interactions; and the trap that it guarantees **syntax, not semantics** and can *degrade quality* — so prefer **reason-then-constrain** and measure quality, not adherence. *Reading:* Willard & Louf, *Outlines* (2023); *XGrammar* (2024). *Lab:* guarantee schema validity, then measure per-step overhead and the quality cost of constraining too early. *Deliverable:* an adherence × overhead × quality table proving adherence and quality are different axes.

**Module 10 — Tool-use / agentic serving**
*The question:* an agent is a multi-turn generate → emit-tool-call → execute → resume loop that re-processes its entire growing context each turn — what makes that cost 1× instead of 20×? *Key ideas:* the one real mechanism is **cross-turn prefix reuse** (Module 5 applied to turns), which turns cumulative re-prefill from `O(T²)` into `O(T)` (~20× at 40 turns) — *if you get the cache hit*; and the **two realities that decide the bill**: (1) cross-turn cache hits are **fragile and fail silently** (non-deterministic tool output, injected metadata, history rewriting, system-prompt drift) — discovered on the bill, not the latency graph; (2) **tool latency gates GPU occupancy**, not model speed (`busy ≈ T_gen/(T_gen+T_tool)`), so the decision is hold-vs-offload the paused KV. The model is usually *not* the bottleneck — measure end-to-end and decomposed. *Reading:* vLLM/SGLang automatic-prefix-caching docs (read for *what invalidates the cache*); a provider prompt-caching API. *Lab:* break the cache silently and watch cost explode; sweep tool latency and recover the slot with KV offload. *Deliverable:* the `O(T²)`-vs-`O(T)` re-prefill curve, the cache-hit-rate-vs-prefix-stability silent blowup, and the tool-latency capacity curve.

### Part VI — Systems engineering & methodology

**Module 11 — Frameworks, the API layer, and resilience**
*The question:* we have the techniques; how are they packaged, exposed, observed, and made dependable? *Key ideas:* what vLLM and SGLang are and how they differ; the OpenAI-compatible interface as de-facto standard; a **minimal, auditable gateway/router** (failover, auth, metadata-only telemetry, capability parity across backends); load-shedding and graceful degradation. *Going deeper (appendix):* **scheduler internals at principle level** — admission, preemption/swapping, priority. *Reading:* the vLLM and SGLang documentation/papers. *Lab:* stand up both engines behind a ~150-line gateway; instrument it; inject failure (kill a backend mid-stream) and measure the user-visible failover gap. *Deliverable:* the gateway + `resilience.md` quantifying each failure mode.

**Module 12 — How to benchmark without lying to yourself** *(Main includes the statistics)*
*The question:* every prior lab produced numbers — but a number is meaningless without the right metric and a realistic workload. *Key ideas:* precise **TTFT / TPOT / throughput / goodput / tail-latency** definitions and which an application cares about; the **synthetic-vs-realistic-traffic trap**; **open- vs closed-loop load generation and coordinated omission**; **warmup, confidence intervals, reported variance**; the non-determinism of batched inference at temperature 0; MLPerf-style scenarios. *Reading:* MLPerf Inference rules; a serving-benchmark methodology paper. *Lab:* take an earlier result and *break* it by changing only the traffic distribution; document how the conclusion flips. *Deliverable:* a methodology note showing the same system "winning" and "losing" under defensible benchmark choices.

---

## Appendix A — Accelerator hardware *(anchors Part I; referenced by M3 and M6)*

Where the numbers `P` and `B` come from, and the silicon facts the optimizations exploit:
- the on-chip memory hierarchy (register / shared / L2 / HBM) — capacities and bandwidths, and why FlashAttention's SRAM residency matters;
- what a tensor-core MMA actually computes, and **why lower precision is *faster*, not merely smaller** (e.g. H100 FP8 ≈ 2× BF16 dense throughput) — the hardware basis for Module 3's decode-vs-prefill split;
- interconnect topology — NVLink/NVSwitch vs PCIe bandwidth and latency — the hardware basis for Module 6's TP-vs-PP choice.

Self-contained; assumes computer-architecture fundamentals.

---

## Appendix B — Consumer-hardware reality *(the self-hosting substrate; referenced by M1, M3, M6, M7, M11)*

The course is anchored on the H100; most people self-hosting own a 4090, a Mac, or a couple of gaming GPUs on PCIe. This appendix re-grounds the same principles on that substrate — **the physics doesn't change, only the numbers and the binding constraint do**:
- the central bandwidth law restated on consumer silicon (B.1);
- the two consumer walls — **VRAM capacity** and **memory bandwidth** (B.2);
- **quantization as your main lever, and why GGUF is Module 3 in folk clothing** (B.3);
- **multi-GPU without NVLink** — Module 6 over PCIe, and where it collapses (B.4);
- the local framework landscape — **llama.cpp / Ollama / LM Studio vs vLLM / SGLang** (B.5);
- which modules actually bite when you serve locally (B.6).

The point: after this, the everyday local-LLM questions — *"which quant?", "why is my Mac slow?", "will two 3090s help?"* — answer themselves from the modules you already read. The most directly useful appendix for the "self-hosting in your own bunker" reader.

---

## Appendix C — The modern serving stack: vLLM and SGLang *(expanded from the serving-stack section above; drives every lab)*

The operational companion to the **modern serving stack** section above: what vLLM and SGLang are, why a research `generate()` is not a server, their defining ideas (PagedAttention, continuous batching, RadixAttention, constrained decoding), the **Orca → vLLM → SGLang** lineage, and the exact commands to launch, configure, instrument, and benchmark **both** behind one OpenAI-compatible interface — the engines `common/` and every lab talk to. *(File: `Appendix C — The Modern Serving Stack.md`.)*

---

## Final project (capstone)

**Yes — the course ends in a capstone**, weighted 40%, written to workshop standard with a reproducible repo + short report, on **open models and public benchmarks only**. The capstone is where the now-deeper modules pay off: a strong project derives its result from the spine (roofline, KV budget, communication cost, or the speculative-decoding law), not just reports a speedup.

Three archetypes:

- **Measurement study** — characterize a technique across models/workloads and *explain* it from first principles. E.g. *quantify FlashAttention's IO savings against the roofline*; *map the long-context accuracy–memory frontier across eviction and KV-quantization settings*; *measure TP-vs-PP scaling and attribute the cost to the collectives*.
- **Build-and-benchmark** — implement an optimization or serving feature and show, with correct metrics on realistic traffic, where it helps and where it does not. E.g. *a KV-cache eviction policy*; *a draft-model selector for speculative decoding*; *a prefix-aware scheduling policy in the gateway*.
- **Reproduction + stress** — reproduce a paper's headline result on open models, then find where it breaks (a workload or context regime the paper did not test). This trains the Module 12 instinct directly.

Grading rewards the **analysis** — connecting measured numbers back to the spine — over the headline number.

---

## Repository layout

**Built so far:** all 13 module lecture-notes, all three appendices (the serving-stack notes are now Appendix C), the shared lab harness (`common/`, tested — 20 tests), and all 13 labs under `labs/` — two **built & tested** (`m04_batching/`, `m10_agentic/`) and 11 **implemented, awaiting real-hardware runs** (a `run_lab.py` each; run them on your own GPU — we're curious how they behave across environments; see `labs/README.md`). Plus `infra/` (rent→run→teardown: `serve_and_run.sh`, cost table, HF-cache notes) and `project/` (capstone brief + rubric + report template). The course is feature-complete; the open invitation is running the 11 labs on your own hardware.

```
.
├── README.md                                             # this syllabus
│
├── Module 00 — Numbers, Tokens, and Memory.md            # lecture notes (Main + Going Deeper), one file per module
├── Module 01 — The Roofline of a Transformer Forward Pass.md
├── Module 02 — Attention, the KV Cache, and Where the Memory Goes.md
├── Module 03 — Quantization - Shrinking the Numerator.md
├── Module 04 — Batching and Scheduling.md
├── Module 05 — Serving-Time Memory Management.md
├── Module 06 — Multi-GPU Parallelism.md
├── Module 07 — Long-Context Serving.md
├── Module 08 — Attacking the Sequential Dependency.md
├── Module 09 — Structured and Constrained Decoding.md
├── Module 10 — Tool-Use and Agentic Serving.md
├── Module 11 — Frameworks, the API Layer, and Resilience.md
├── Module 12 — How to Benchmark Without Lying to Yourself.md
│
├── Appendix A — Accelerator Hardware.md                  # H100 silicon: memory hierarchy, MMA, interconnect
├── Appendix B — Consumer-Hardware Reality.md             # 4090 / Mac / multi-GPU PCIe — the self-hosting substrate
├── Appendix C — The Modern Serving Stack.md              # vLLM & SGLang: what they are, why, and how to drive them
│
├── common/                                               # shared lab harness — BUILT & TESTED (no GPU needed)
│   ├── README.md                                         #   usage; sim-vs-real endpoint; the 7 invariants → test map
│   ├── requirements.txt
│   ├── mem.py                                            #   mem_estimate / kv_budget — pure math, unit-tested
│   ├── bench/                                            #   client (real OpenAIEndpoint + offline SimEndpoint),
│   │                                                     #   drivers (open/closed/agentic), faults, metrics
│   ├── traffic/                                          #   sharegpt · long_doc_qa · needle · agentic · schedules
│   ├── eval/                                             #   score_task/score_suite · needle · score_json_schema · deltas
│   └── tests/                                            #   test_harness.py — one test per invariant
│
├── labs/                                                 # one lab per module — drive common/ against a real engine
│   ├── README.md                                         #   index + which GPU / provider to rent for each lab
│   ├── m04_batching/   (run_lab.py + README)             #   BUILT — static-vs-continuous · knee · goodput · chunked prefill
│   ├── m10_agentic/    (run_lab.py + README)             #   BUILT — re-prefill O(T²)→O(T) · silent cache collapse · tool latency
│   └── m00…m12  (the other 11, implemented)                #   a run_lab.py each · run on your GPU
│
├── infra/                                                # cheap-GPU logistics: serve_and_run.sh + cost / HF-cache notes
├── project/                                              # capstone: brief · rubric · report template · worked examples
├── translations/                                         # community translations (zh/ started) — help wanted
├── .github/ISSUE_TEMPLATE/                               # lab-report + translation issue forms
├── CONTRIBUTING.md                                       # how to help — run labs on your hardware, translate, fix, improve
├── LICENSE                                               # MIT
└── .gitignore
```

---

## Core reading list

- Vaswani et al. (2017), *Attention Is All You Need*
- Micikevicius et al. (2017), *Mixed Precision Training*; OCP *FP8 Formats*
- Pope et al. (2022), *Efficiently Scaling Transformer Inference*
- Dao et al. (2022), *FlashAttention*; Ainslie et al. (2023), *GQA*; DeepSeek-V2 (2024) for *MLA*
- Frantar et al. (2022), *GPTQ*; Lin et al. (2023), *AWQ*; Xiao et al. (2022), *SmoothQuant*
- Yu et al. (2022), *Orca*; Kwon et al. (2023), *PagedAttention/vLLM*; Zheng et al. (2023), *SGLang/RadixAttention*
- Shoeybi et al. (2019), *Megatron-LM*; Fedus et al. (2021), *Switch Transformers*
- Zhang et al. (2023), *H2O*; Xiao et al. (2023), *StreamingLLM*; *KIVI* (2024)
- Leviathan et al. (2023), *Speculative Decoding*; *DistServe* or *Splitwise* (2024)
- Willard & Louf (2023), *Outlines*; *XGrammar* (2024) — structured / constrained decoding
- vLLM / SGLang automatic-prefix-caching docs; a provider prompt-caching API — cross-turn reuse for agents
- MLPerf Inference rules (benchmark methodology)

---

## Contributing

Contributions are warmly welcome — the most valuable being **running a lab on your own hardware and reporting how it behaves across environments** (a working plot, or a traceback). **Translations** (Chinese 中文 and other languages) are especially welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

*License: **MIT** — see [`LICENSE`](LICENSE). Code and prose are shared under the same permissive terms. Labs assume open-weight models and publicly redistributable benchmark data only.*