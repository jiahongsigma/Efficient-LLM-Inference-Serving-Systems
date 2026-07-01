# Module 1 — The Roofline of a Transformer Forward Pass

*Prerequisite: Module 0 (you must be able to convert parameters to bytes). This module establishes the single fact the rest of the course derives from — that autoregressive decode is memory-bandwidth-bound — and gives you the tool (the roofline) to predict which optimization helps a given workload.*

---

## In plain English

**Why this matters.** Your GPU can do trillions of calculations a second, yet it still produces text slowly. Before you optimize anything, you need to know *what* it's waiting on — otherwise you tune the wrong thing.

**What this module gives you.** One mental model (the "roofline") that tells you, for any job, whether it's stuck waiting on *math* or waiting on *memory*. For text generation, it's almost always memory.

**How it works (the intuition).** Writing one word of output means hauling the model's entire set of weights out of memory but doing very little math with them — like driving a truck across town to deliver a single envelope. The drive dominates, not the delivery. Every later trick either makes that one drive carry more envelopes (batching) or shrinks the truck (quantization).

---

## 1.1 The roofline model, recalled

You have seen this in computer architecture; we use it as the lens for everything. A kernel running on an accelerator is limited by one of two ceilings:

- **Peak compute** `P` (FLOP/s) — how fast it can do arithmetic.
- **Peak memory bandwidth** `B` (byte/s) — how fast it can move operands to/from HBM.

A kernel's position is set by its **arithmetic intensity**:

```
I  =  FLOPs performed  /  bytes moved          [FLOP/byte]
```

Achievable performance is `min(P, I × B)`. The crossover — the **ridge point** — is at `I* = P / B`. Below `I*` the kernel is **memory-bound** (you are limited by `B`, and faster compute does nothing); above it, **compute-bound**.

For an H100-class GPU: `B ≈ 3.35 TB/s`, and dense BF16 `P ≈ 990 TFLOP/s` (the often-quoted ~1979 figure counts 2:1 structured sparsity). So:

```
I* ≈ 990e12 / 3.35e12 ≈ 295 FLOP/byte   (≈ 590 if you count sparsity)
```

Hold that number — a few hundred FLOP/byte — against what decode actually achieves.

---

## 1.2 The forward pass is a stack of GEMMs

Strip away the diagram-level view of a transformer. Mechanically, a forward pass is a sequence of matrix multiplications (GEMMs) — the attention projections (Q, K, V, O) and the MLP up/down projections — interleaved with the attention operation itself. The GEMMs dominate the parameter count and the FLOPs, so to first order:

> Processing one token through a dense model with **N parameters** costs about **2N FLOPs** (one multiply + one add per parameter) and requires reading about **N weights** from HBM.

This approximation — "≈2 FLOPs and one weight-read per parameter per token" — is all we need to locate decode on the roofline.

---

## 1.3 Decode: arithmetic intensity ≈ 1

In autoregressive **decode**, the model emits one token at a time; each step processes a single new token through all N parameters. Using §1.2 and Module 0 (FP16 weights = 2 bytes each):

```
FLOPs per token ≈ 2N
bytes per token ≈ 2N        (read every weight once, FP16)
I_decode ≈ 2N / 2N ≈ 1 FLOP/byte
```

Compare to the ridge `I* ≈ 295`. Decode sits **two-and-a-half orders of magnitude inside the memory-bound regime.** The expensive tensor cores are almost entirely idle; the GPU spends decode waiting for weights to arrive from HBM. This is *the* fact of LLM inference.

Its immediate, quantitative consequence — a latency lower bound that ignores everything except moving the weights:

```
time per token  ≳  weight_bytes / B
```

*Worked (Llama-3.1-8B, BF16 → 16 GB of weights):*
```
16e9 bytes / 3.35e12 byte/s ≈ 4.8 ms/token  →  ~210 tokens/s   (batch 1, ceiling)
```

Real single-stream decode is somewhat slower (you also read the KV cache, run attention, and pay kernel-launch overhead), but no amount of compute optimization can beat this bandwidth ceiling at batch 1. Memorize the shape of the argument: **decode speed at batch 1 is a bandwidth division.** It tells you immediately that (a) a faster-compute GPU with the same bandwidth won't help single-stream decode, and (b) the two levers that *do* help are shrinking `weight_bytes` (Module 3, quantization) or amortizing the weight-read across many tokens (next).

---

## 1.4 Prefill: the opposite regime

**Prefill** processes the entire prompt of `S` tokens *in parallel* through the same weights. The weights are read once and reused across all S tokens, so:

```
FLOPs ≈ 2N·S
bytes ≈ 2N            (weights read once for the whole prompt)
I_prefill ≈ S
```

For any realistic prompt length, `I_prefill ≫ I*`: **prefill is compute-bound.** The same model, the same weights, in the same forward pass — but prefill saturates the tensor cores while decode starves them. This asymmetry is not a footnote; it is the reason several later techniques exist (chunked prefill and prefill–decode disaggregation, Module 8), and the reason TTFT and per-token decode latency must be measured and optimized separately (instrumented in Module 11, measured honestly in Module 12).

---

## 1.5 Batching: climbing the roofline

If decode is memory-bound because one token reuses each weight only once, the fix writes itself: **make many sequences reuse the same weight-read.** Run a batch of `B` independent sequences' decode steps together — the weights are still read once per step, now amortized across B tokens of useful work:

```
FLOPs per step ≈ 2N·B
weight bytes per step ≈ 2N
I ≈ B
```

Batching slides the workload *up the roofline*, from `I ≈ 1` toward the ridge at `I ≈ 295`. This is the single most important throughput lever in serving, and §1.3 is precisely why it exists. It is the entire motivation for continuous batching (Module 4).

Two caveats that the later modules cash out:

1. **KV-cache reads also scale with B (and with context).** The clean `I ≈ B` ignores that each sequence must read its own growing KV cache. Past some batch × context, KV-cache bandwidth re-binds the workload to memory — which is why long-context, high-concurrency serving is hard and why KV management (Module 5) matters.
2. **Batch size is bounded by memory, not willingness.** You can only batch as many sequences as their KV caches fit (Module 0's sizing inequality). Raising the achievable batch is itself a memory-management problem — exactly what PagedAttention solves.

---

## 1.6 The picture to carry forward

| | Prefill | Decode |
|---|---|---|
| Tokens processed per pass | many (the prompt) | one |
| Arithmetic intensity | ≈ S (high) | ≈ 1 (low) |
| Bound by | compute | **memory bandwidth** |
| Helped most by | more FLOP/s, chunking | quantization, batching |
| Latency metric | TTFT | TPOT / inter-token |

Everything after this module is a response to the bottom-right cell. Quantization (M3) shrinks the bytes moved; batching and scheduling (M4) and paging (M5) maximize how much useful work rides each weight-read; parallelism (M6) handles models too large for one device's memory; speculative decoding (M8) attacks the sequential dependency that forces decode to be one-token-at-a-time in the first place.

---

## Lab 1 — Place prefill and decode on a measured roofline

**Goal:** verify §1.3–1.5 on real hardware, not just on paper.

1. Serve an 8B model (vLLM) and instrument GPU telemetry with `nvidia-smi dmon` (or Nsight Compute for a kernel-level view). Track SM utilization and, crucially, **achieved HBM bandwidth**.
2. **Isolate the phases.** Send a long-prompt / 1-token-output request to characterize prefill; send a short-prompt / long-output request to characterize decode. Record FLOP/s and achieved bandwidth for each.
3. **Plot the roofline.** Draw the two ceilings (`P`, `B`) and place your measured prefill and decode points. Decode should land deep in the memory-bound region with high achieved-bandwidth utilization (commonly 60–80%); prefill should sit near the compute ceiling.
4. **Sweep the batch.** Replay decode at batch sizes 1, 4, 16, 64 and watch the operating point climb the roofline toward the ridge; note where throughput gains start to flatten (and reason about whether KV-bandwidth or memory capacity is the cause).
5. **Validate the latency bound.** Compare measured batch-1 per-token latency to the `weight_bytes / B` lower bound from §1.3 and account for the gap.

**Deliverable:** a roofline plot with measured prefill and decode points, the decode achieved-bandwidth-utilization figure, and a batch-sweep curve showing arithmetic intensity rising with batch size. **In your writeup, state in one sentence why a higher-FLOP/s GPU at the same bandwidth would not speed up your batch-1 decode** — if you can defend that sentence, you own the module.

**Reading:** Pope et al., *Efficiently Scaling Transformer Inference* (2022) — the canonical roofline-and-serving analysis this module compresses.