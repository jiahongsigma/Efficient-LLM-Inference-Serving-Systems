# Module 6 — Multi-GPU Parallelism

*Prerequisites: Module 0 (weights in bytes), Module 2 (KV memory), Module 1 (the roofline: compute vs HBM bandwidth), Module 5 (KV memory management). When weights + KV exceed one GPU's HBM, you must split the model across GPUs — and splitting introduces a cost the single-GPU roofline never had: **communication**. This module adds a third ceiling to Module 1's two, and makes the communication cost of each parallelism strategy explicit, because that cost — not FLOPs — is usually what binds.*

---

## In plain English

**Why this matters.** The biggest models don't fit on one GPU, so you split them across several. But the moment you do, the GPUs must talk to each other constantly — and that conversation, not the math, is usually what becomes the bottleneck.

**What this module gives you.** The three ways to split a model (tensor, pipeline, and expert parallelism), what each one costs in communication, and how to choose based on the cables connecting your GPUs.

**How it works (the intuition).** Splitting a job across workers only helps if coordinating them is cheap. Some splits make the workers chat every fraction of a second (so they need a fast link — NVLink); others let them hand work along only occasionally (so a slower link is fine). This module makes that hidden coordination cost visible and measurable.

---

## 6.1 The necessity, and the third ceiling

A 70B model in FP16 is 140 GB (Module 0) — it does not fit on an 80 GB GPU, and long-context KV (Module 2) only makes the overflow worse. So you split the model across GPUs. But the GPUs must now exchange intermediate results, and that communication is a new term in the performance budget.

Module 1 said `time ≈ max(compute_time, memory_time)` — two ceilings, compute throughput and HBM bandwidth. Multi-GPU adds a third:

```
comm_time ≈ message_bytes / interconnect_bandwidth + latency
```

and because the most important collectives are **synchronous and on the critical path** (the next layer cannot start until they finish), this term often *adds* to compute rather than overlapping with it. Performance is now bounded by `min(compute, HBM-bandwidth, interconnect-bandwidth)`, and the interconnect ceiling is frequently the lowest. The whole module is: which strategy you pick determines how much you communicate, over what link, and therefore whether you are compute- or communication-bound.

The three strategies differ in *what* they split — and that determines their communication pattern.

---

## 6.2 Tensor parallelism (TP) — split *within* each layer

**Megatron-LM** (2019). Split each weight matrix across GPUs; every GPU computes a slice of every layer. The clever part is choosing splits that minimize synchronization (full algebra in Going Deeper):

- In the MLP, split the first matrix **column-wise** (the elementwise GeLU then needs no communication) and the second **row-wise** (each GPU produces a partial sum) — requiring **one all-reduce** to sum the partials.
- In attention, split the **heads** across GPUs and row-split the output projection — requiring **one all-reduce**.

So TP costs **two all-reduces per transformer layer in the forward pass** (one for attention, one for the MLP), each over an activation tensor of ~`batch × seq × hidden`, **every layer, every forward step**. These are frequent, synchronous, and on the critical path.

This is why **TP demands NVLink.** The all-reduces are constant and latency-sensitive; on NVLink/NVSwitch (~900 GB/s on H100, Appendix A) they are tolerable, but over PCIe (~64 GB/s) or across nodes they dominate and collapse performance. TP is therefore kept **within a node** (typically ≤ 8 GPUs on one NVLink domain).

For *inference*, TP is the workhorse: it both fits a model that won't fit on one GPU **and reduces single-request latency** (each GPU does less work per layer). It also splits the **KV cache** across ranks (each rank holds the KV for its heads), relieving Module 2's KV pressure per GPU. The price is the per-layer all-reduce.

---

## 6.3 Pipeline parallelism (PP) — split *across* layers

Split the model by **layer ranges**: GPU 0 holds layers 1–8, GPU 1 holds 9–16, and so on. Activations flow GPU0 → GPU1 → … in **point-to-point** sends, only at stage boundaries — far less communication than TP, and not a collective, so it **tolerates slower links** (PCIe, cross-node). PP is how you scale *across* nodes.

The catch is the **pipeline bubble**. Naively, downstream GPUs idle while the first stage works. You fill the pipe by splitting the batch into `m` microbatches; but with `p` stages there is still a fill/drain bubble:

```
bubble fraction = (p − 1) / (m + p − 1)
```

— minimized only by `m ≫ p` (many microbatches in flight). Smarter schedules (1F1B, interleaved) shrink it further. So PP efficiency hinges on having enough concurrent work to hide the bubble.

For inference, PP buys **throughput** (many requests traversing the stages), **not single-request latency** — one request still passes through every stage serially, plus its share of the bubble. The latency-vs-throughput distinction between TP and PP is the practical decision.

---

## 6.4 Expert parallelism (EP) — for mixtures of experts

A Mixture-of-Experts layer (Switch Transformer, 2021) has many expert FFNs; a gate routes each token to its top-`k` experts. **EP places different experts on different GPUs.** The communication is the **all-to-all**: tokens are *dispatched* to the GPUs holding their chosen experts, then results are *combined* back — every GPU sending some tokens to every other GPU, **twice per MoE layer**.

MoE's appeal is sparsity (Module 0): only top-`k` experts activate per token, so FLOPs-per-token stay low even as total parameters explode. The all-to-all is the price, and **at scale (many experts across many GPUs) the all-to-all becomes the dominant bottleneck** — which is why MoE serving is, fundamentally, an all-to-all-communication engineering problem.

*(A fourth axis, for very long context where even one sequence's activations/KV don't fit: **sequence/context parallelism** — Ring Attention, Ulysses — splits the *sequence* dimension across GPUs. Developed in Module 7.)*

---

## 6.5 The communication roofline, and how to choose

For a TP layer, the synchronous all-reduce means `layer_time ≈ compute_time + comm_time` (unless explicitly overlapped, §6.6). So **parallel scaling is sub-linear**: as you add GPUs, per-GPU compute shrinks but communication grows, and past some degree communication dominates and adding GPUs stops helping — or hurts. *The deviation of your measured scaling curve from the ideal linear line is exactly the communication cost.* Identifying which ceiling binds — compute (Module 1) or interconnect — is the core diagnostic, and the lab's deliverable.

The strategy is dictated by three things: **does it fit** (memory, M0/M2), **latency vs throughput** target (M4), and **what interconnect** you have (Appendix A). In practice they **compose** — "3D parallelism": **TP within a node** (on NVLink), **PP across nodes** (point-to-point over the slower link), and **EP for MoE** layers. You pick TP degree to fit + hit latency, PP to span nodes for throughput, EP to place experts.

---

## 6.6 Where the frontier is now

TP/PP/EP (2019–2021) are established; current practice has moved on three axes:

- **Expert parallelism at scale.** Frontier MoE models (e.g. DeepSeek-V3) are served with EP across *many* GPUs, where the all-to-all is the central challenge — answered by communication-computation overlap and custom all-to-all kernels (DeepEP).
- **Larger NVLink domains.** Hardware like GB200 NVL72 puts 72 GPUs in a single NVLink domain, dramatically widening the "within-node" regime and **moving the TP-vs-PP boundary** — more of the model can stay in cheap-communication TP.
- **Communication–computation overlap.** Good implementations now overlap the all-reduce with subsequent compute to hide TP's cost — turning the naive `compute + comm` back toward `max(compute, comm)`.

Teaching only Megatron's 2019 design would miss that the live problem today is *EP all-to-all at scale* and *overlap*, on *much larger NVLink fabrics*.

---

## 6.7 The picture to carry forward

- When a model + KV exceed one GPU, you split — and splitting adds a **third roofline ceiling: interconnect bandwidth**, often the binding one.
- **TP** (intra-layer) costs **two all-reduces per layer**, is synchronous and critical-path, so it **needs NVLink** and stays within a node; for inference it cuts latency and splits the KV cache.
- **PP** (inter-layer) communicates sparsely point-to-point, so it **tolerates PCIe/cross-node**, but pays the **`(p−1)/(m+p−1)` bubble** and buys throughput, not latency.
- **EP** (MoE) costs an **all-to-all** that dominates at scale.
- **Scaling is sub-linear; the gap from linear is the communication cost.** Strategies compose as 3D parallelism.

---

## Going Deeper (appendix) — the TP algebra, the bubble, and ring vs tree

**Why exactly two all-reduces (Megatron).** For an MLP `Y = GeLU(XA)B`, partition `A = [A₁ A₂]` column-wise: `XA = [XA₁, XA₂]`, and since GeLU is elementwise, each GPU computes `GeLU(XAᵢ)` with **no communication** (the `f` operator: identity forward, all-reduce backward). Partition `B = [B₁; B₂]` row-wise: `Y = GeLU(XA₁)B₁ + GeLU(XA₂)B₂`, a sum of per-GPU partials requiring **one all-reduce** (the `g` operator: all-reduce forward). Attention is analogous — heads split across GPUs, output projection row-split, **one all-reduce**. Forward pass: **2 all-reduces/layer**, each of volume ~`2·b·s·h` bytes (FP16).

**The bubble.** With `p` stages and `m` microbatches, the simplest (GPipe) schedule does useful work for `m` steps but the pipe fills and drains over `p−1` steps each, so wall-clock is `m + p − 1` steps → idle fraction `(p−1)/(m+p−1)`. Interleaved/1F1B schedules reduce the constant.

**Ring vs tree all-reduce.** A **ring** all-reduce moves `2·(N−1)/N · M` bytes per GPU (→ `2M` as N grows) — **bandwidth-optimal**, independent of N in the limit, but `N−1` sequential steps so latency grows with N. A **tree** all-reduce finishes in `log N` steps — **latency-optimal** for small messages — but is less bandwidth-efficient. TP's all-reduces are large and frequent → bandwidth-bound → ring on NVLink; small control messages → tree. NCCL picks adaptively. This is why the *same* collective behaves differently depending on message size and interconnect.

---

## Lab 6 — Make the communication cost visible

**Context — what this builds on (checklist).** *Reuse:* `mem_estimate` (M0/M2) to confirm the model overflows one GPU and to pick the TP degree; the `common/bench` **open-loop** generator and Module 4's latency/throughput methodology; Module 1's roofline to classify the binding resource. *Test the new insight:* the **communication cost** of parallelism, not a restated batch sweep. *Exercise prior quantities:* memory overflow (why >1 GPU), the roofline (now a communication ceiling), latency/throughput (M4). *Frontier:* EP all-to-all and comm-compute overlap. *Consistency:* report **sub-linear** scaling and name the binding wall — do not claim linear speedup.

1. **TP scaling and its sub-linearity (the new insight).** Serve a model that won't fit on one GPU at **TP=2 and TP=4** (open-loop load). Measure throughput and **single-request latency**; plot the scaling curve against the ideal linear line. The gap *is* the communication cost. At each TP degree, classify the binding resource — compute (Module 1 roofline) vs interconnect (measure per-layer all-reduce time). *Confirms: §6.2, §6.5 — parallelism trades compute for communication; scaling is sub-linear.*

2. **The interconnect decides (why NVLink).** Measure all-reduce time and relate it to interconnect bandwidth (Appendix A: NVLink ~900 GB/s vs PCIe ~64 GB/s). If you can place TP ranks across NVLink vs across PCIe, show TP throughput **collapsing** on the slow link. *Confirms: §6.2 — communication, not FLOPs, binds; TP needs NVLink.*

3. **TP vs PP character (latency vs throughput).** Compare TP and PP on the *same* model: show TP lowers **single-request latency** while PP needs many in-flight microbatches/requests to amortize the **`(p−1)/(m+p−1)` bubble** and does **not** lower single-request latency. *Confirms: §6.3.*

4. **Dense vs MoE (the all-to-all).** Contrast a dense model under TP with an MoE model (Mixtral-class) under **expert parallelism**; observe the all-to-all dispatch/combine and how its cost grows with EP degree. *Confirms: §6.4.*

5. **(Frontier) overlap.** Enable communication–computation overlap if supported and show TP's overhead shrink; note the EP-at-scale / NVL72 direction. *Confirms: §6.6.*

**Deliverable:** the **parallel-scaling curve** (throughput + latency vs TP degree) with the **binding resource named at each point** (compute vs communication); the all-reduce-time-vs-interconnect relation; the TP-vs-PP latency/throughput contrast; and the dense-vs-MoE all-to-all observation. **Mastery test — defend in one sentence each:** *why multi-GPU adds a third (communication) roofline ceiling; why TP needs NVLink but PP tolerates PCIe; the two all-reduces per layer in TP and the `(p−1)/(m+p−1)` bubble in PP; what the MoE all-to-all does; and why parallel scaling is sub-linear.* *Feeds:* Module 7 (sequence/context parallelism for long context) and Module 8 (disaggregation, which assigns different parallel configs to prefill and decode).

**Reading:** Shoeybi et al., *Megatron-LM* (2019); Fedus et al., *Switch Transformers* (2021). Current frontier: the *DeepSeek-V3* technical report (large-scale EP) and *DeepEP* (2025) for all-to-all kernels; NVLink/NVSwitch and GB200 NVL72 documentation for the interconnect picture.