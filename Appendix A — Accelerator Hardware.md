# Appendix A — Accelerator Hardware

*Where the roofline's `P` (peak compute) and `B` (peak bandwidth) come from, and the silicon facts the course's optimizations exploit. Self-contained; assumes computer-architecture fundamentals. Worked numbers use the **H100 SXM5** as the reference architecture (the one cited throughout); the principles generalize, and §A.8 notes where Blackwell moves the boundaries.*

**Referenced by:** Module 1 (the ridge point), Module 2 (SRAM residency for FlashAttention), Module 3 (why low precision is *faster*), Modules 5/7 (the memory levels KV spills through), Module 6 (NVLink vs PCIe), Module 8 (KV transfer over the interconnect).

---

## A.1 The execution model, briefly

A GPU is a throughput machine: ~132 **streaming multiprocessors (SMs)** on an H100, each running many **warps** (groups of 32 threads in lock-step, SIMT) to hide memory latency by switching to a ready warp whenever one stalls on memory. Two kinds of math units matter:

- **CUDA cores** — scalar fused-multiply-add (FMA) units; general-purpose, lower throughput.
- **Tensor cores** — units that perform a small **matrix-multiply-accumulate (MMA)**, `D = A·B + C` over tiles (e.g. 16×16), in one operation. These deliver the overwhelming majority of an LLM's FLOPs, and they are why `P` is what it is (§A.4).

LLM inference is almost entirely tensor-core GEMMs feeding on data moved through a steep memory hierarchy. The whole performance story is the interaction of those two.

---

## A.2 The memory hierarchy — a bandwidth/latency/capacity pyramid

This single table is the foundation of the course. Bandwidth **falls by orders of magnitude** and latency **rises** as you move out from the registers; capacity grows in compensation. (H100 SXM5; approximate.)

| Level | Capacity | Bandwidth | Latency | The course technique that lives here |
|---|---|---|---|---|
| Registers | 256 KB / SM | ~tens of TB/s/SM | ~1 cycle | the GEMM's working set |
| Shared mem / L1 (SRAM) | up to 228 KB / SM | ~tens of TB/s | ~20–30 cycles | **FlashAttention** keeps attention here (Module 2) |
| L2 cache | 50 MB | several TB/s | ~200 cycles | reuse across SMs |
| **HBM3** | **80 GB** | **3.35 TB/s** | ~hundreds of ns | weights + **KV cache** live here (Modules 1, 5) |
| NVLink (intra-node) | a peer GPU's 80 GB | 900 GB/s | ~sub-µs | **tensor parallelism** (Module 6) |
| PCIe (to host) | host RAM | ~64 GB/s | ~µs | **KV offload / swap** (Modules 5, 7) |
| Network (InfiniBand) | other nodes | ~50 GB/s/NIC | ~µs+ | **pipeline parallelism, disaggregation** (Modules 6, 8) |

Read the right-hand column top to bottom and you have re-derived the whole course as *a campaign to keep each piece of data at the cheapest level it can live at*: FlashAttention fights to stay in SRAM, quantization shrinks what must cross the HBM line, paging manages HBM, parallelism pays the NVLink toll, offload spills to PCIe, disaggregation crosses the network.

---

## A.3 Where `B` comes from

`B` in Module 1's roofline is the **HBM bandwidth: 3.35 TB/s** on an H100 (HBM3, the line between the compute and the 80 GB of weights+KV). It is the bottleneck for everything memory-bound — and Module 1 showed decode *is* memory-bound. The latency to first byte from HBM is hundreds of nanoseconds (~500 cycles), which is why the SM keeps dozens of warps in flight to hide it. When a kernel's working set fits in SRAM (228 KB/SM) instead, it pays the SRAM bandwidth (tens of TB/s) and a fraction of the latency — **exactly why FlashAttention's refusal to spill the `S×S` score matrix to HBM (Module 2) is the whole game**.

---

## A.4 The tensor core, `P`, and why low precision is *faster*

`P` in the roofline is the **tensor-core throughput**, and it depends on precision. Dense figures (no 2:1 structured sparsity, which most LLM inference does not use — the often-quoted marketing numbers are 2× these):

| Precision | H100 dense throughput |
|---|---|
| FP16 / BF16 | ~989 TFLOP/s |
| FP8 (E4M3/E5M2) | ~1,979 TFLOP/s |
| INT8 | ~1,979 TOP/s |
| TF32 | ~495 TFLOP/s |

**Why lower precision is faster, not merely smaller** — the hardware fact behind Module 3's decode-vs-prefill split. A tensor core pushes a **fixed number of bits per cycle** through its datapath. At lower precision, *more elements* fit in that same bit-width, so *more MACs* execute per cycle → higher throughput. FP8 packs twice the elements of BF16 into the same silicon → **~2× the FLOP/s**; FP4 (Blackwell) doubles it again. This is a *compute-throughput* lever, entirely separate from the *memory* saving of fewer bytes.

That separation is precisely Module 3's two mechanisms:
- **Decode (memory-bound)** is helped by **fewer bytes moved from HBM** — so weight-only quantization suffices, and the GEMM may still compute in FP16.
- **Prefill (compute-bound)** is helped by **higher MMA throughput** — which requires the tensor cores to *actually compute* in low precision (weights *and* activations in FP8), using the datapath-packing above.

Two different hardware levers, one for each side of the roofline.

---

## A.5 The roofline ridge, derived

Module 1's ridge point is just `I* = P / B`. For the H100, in BF16:

```
I* = 989e12 FLOP/s ÷ 3.35e12 B/s ≈ 295 FLOP/byte   (dense; ≈ 590 if counting 2:1 sparsity)
```

A subtlety worth carrying: the ridge **moves with precision**. In FP8, `P ≈ 1,979` TFLOP/s, so `I* ≈ 590` FLOP/byte — a *higher* arithmetic intensity is needed to be compute-bound. Faster tensor cores push the ridge right, which is part of why low-precision compute changes which regime a workload sits in (Modules 1, 3, 8). Decode at intensity ≈ 1 is far below the ridge at *any* precision — but the batch at which it crosses into compute-bound depends on where the ridge sits.

---

## A.6 Interconnect — the scaling hierarchy

When a model spans GPUs (Module 6), performance is bounded by the link between them, and there is a sharp hierarchy:

| Tier | Link | Bandwidth | Latency | Topology |
|---|---|---|---|---|
| Within a node | NVLink 4 + NVSwitch | 900 GB/s / GPU | ~sub-µs | all-to-all among 8 GPUs |
| Host / fallback | PCIe Gen5 ×16 | ~64 GB/s / direction | ~µs | GPU↔CPU |
| Across nodes | InfiniBand NDR | ~50 GB/s / NIC | ~µs + network | switched fabric |

This table *is* Module 6's strategy choice:
- **TP needs NVLink.** Its all-reduces are frequent, synchronous, on the critical path, latency-sensitive. NVLink's 900 GB/s and sub-µs latency make them tolerable; PCIe (≈14× less bandwidth, higher latency) makes TP collapse — hence TP stays **within an NVLink node** (≤8 GPUs).
- **PP tolerates PCIe.** Its communication is sparse, point-to-point, latency-insensitive, so it survives the slower cross-node fabric — hence PP spans **across nodes**.
- **EP's all-to-all and disaggregation's KV transfer** (Modules 6, 8) ride whichever tier separates the participants — which is why their cost is an interconnect-bandwidth question (`bytes / link-bandwidth`).

The general law: **collective, frequent, latency-sensitive traffic demands the fastest tier; sparse point-to-point traffic tolerates a slower one.**

---

## A.7 The unifying pyramid

Stack §A.2 and §A.6 into one bandwidth ladder, register → network, and every module's technique is a move on it:

- **FlashAttention** (M2): keep attention in **SRAM**, off the HBM line.
- **Quantization** (M3): shrink what crosses the **HBM** line (bandwidth) *and* pack the **tensor-core** datapath (compute).
- **PagedAttention / RadixAttention** (M5): manage and deduplicate the **HBM**-resident KV cache.
- **Tensor parallelism** (M6): pay the **NVLink** toll to split a layer.
- **KV offload / long context** (M5, M7): spill KV down to **PCIe** (host) and beyond.
- **Pipeline parallelism, disaggregation** (M6, M8): cross the **network**, accepting its latency for sparse traffic.

Performance engineering for LLMs *is* deciding, for each byte, the cheapest level of this pyramid it can live at — and paying the next level's bandwidth only when forced. Every roofline argument and communication-cost derivation in the course is an instance of that one principle.

---

## A.8 Where the frontier is (Blackwell)

The reference numbers above are H100; the current generation moves three of the boundaries the modules care about:

- **A new datatype tier.** Blackwell (B200) adds **native FP4** with roughly **2× the FP8 throughput**, extending §A.4's precision ladder one rung lower — the hardware behind Module 3's NVFP4/MXFP4 frontier.
- **Faster interconnect.** **NVLink 5** raises per-GPU bandwidth to **~1.8 TB/s** (≈2× NVLink 4), and **GB200 NVL72** puts **72 GPUs in a single NVLink domain** rather than 8. This dramatically widens the "within-node" tier of §A.6 — **moving Module 6's TP-vs-PP boundary** so far more of a model can stay in cheap-communication tensor parallelism.
- **More and faster HBM** (HBM3e), raising `B` and the KV-capacity ceiling (Modules 1, 2, 7).

The structure is unchanged — a steep bandwidth pyramid feeding precision-tiered tensor cores — but the numbers shift, and with them the exact batch sizes, context lengths, and parallel-config boundaries at which each technique's regime begins. Re-derive the ridge and the interconnect table for your actual hardware before trusting any threshold in these notes.

---

*Reading: the NVIDIA H100 and Blackwell architecture whitepapers for the silicon numbers; any GPU-architecture course for the SIMT execution model and memory hierarchy. The durable content is the pyramid and the two compute levers, not the specific figures — those date with each generation.*