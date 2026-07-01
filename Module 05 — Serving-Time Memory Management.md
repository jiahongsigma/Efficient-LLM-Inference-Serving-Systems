# Module 5 — Serving-Time Memory Management

*Prerequisites: Module 2 (the KV cache is the dominant variable memory; its size law), Module 4 (continuous batching needs dynamic KV memory, and the achievable batch is capped by KV capacity). Module 4 left a debt: iteration-level batching demands that KV caches be allocated and freed, per step, for sequences of unpredictable length — without that, the throughput it promised is unreachable. This module is how a real engine manages **where the KV lives**: the allocator (PagedAttention) and the reuse/eviction layer (RadixAttention).*

---

## In plain English

**Why this matters.** Continuous batching (Module 4) only pays off if you can hand out and reclaim that growing KV-cache memory cleanly. Do it the old way and you waste most of your GPU memory to fragmentation — capping how many users fit long before the GPU is actually full.

**What this module gives you.** Two ideas: PagedAttention (managing KV memory the way an operating system manages RAM) and prefix sharing / RadixAttention (not recomputing text that many requests have in common).

**How it works (the intuition).** Instead of reserving a giant fixed parking space per request "just in case," hand out memory in small blocks as the request grows — and let requests that begin with the same text (a shared document, a common system prompt) point at one shared copy instead of storing it N times.

---

## 5.1 The problem continuous batching creates

Module 4's fluid batch means, at any instant, a changing set of sequences each holding a **growing** KV cache (Module 2) of **unknown final length**. The pre-2023 way to hold a KV cache was a single **contiguous** buffer per sequence, sized to `max_model_len` — because you don't know the final length, and the attention kernel needed contiguous memory. That choice is catastrophic for serving:

- **Internal fragmentation.** You reserve `max_model_len` but the sequence may emit 50 tokens. The rest of the reserved slot is wasted — often >90% of it.
- **External fragmentation.** Variable-sized contiguous buffers, as sequences come and go, leave unusable gaps between them.
- **Reservation waste.** You cannot admit a new sequence unless a *full contiguous* max-length block is free, even when plenty of total memory is free but scattered.

The consequence is the thing that throttles Module 4: the **achievable** batch is far below what total KV memory should allow — commonly 2–4× worse. Fewer concurrent sequences means a smaller batch, lower arithmetic intensity (Module 1), and a throughput ceiling you hit long before the GPU is actually full. The memory allocator, not the GPU, is the bottleneck.

---

## 5.2 PagedAttention: OS virtual memory for the KV cache

**PagedAttention** (vLLM, SOSP 2023) borrows the operating system's answer to fragmentation: paging. Divide the KV cache into fixed-size **blocks** (e.g. 16 tokens of KV per block). A sequence's cache becomes a **list of blocks** described by a **block table** mapping logical token positions → physical blocks — which need **not** be contiguous. The attention kernel is rewritten to gather KV through the block table.

What this buys, directly against §5.1:

- **Internal fragmentation bounded by one block.** Blocks are allocated on demand as the sequence grows, so the only waste is at most the unfilled tail of the last block (≤ `block_size − 1` tokens), never `max_model_len`.
- **External fragmentation eliminated.** All blocks are identical in size, so *any* free block fits *any* sequence — there are no unusable gaps.
- **The achievable batch rises** toward what total memory allows — several× more concurrent sequences, which raises the batch, the arithmetic intensity, and the throughput ceiling (Modules 1, 4). This is the payoff that makes continuous batching actually deliver.

There is a cost — the block-table indirection and a slightly more complex kernel add minor overhead — but the fragmentation win dominates by a wide margin.

One more capability falls out of the block-table design, and it is the bridge to §5.3: because a sequence's KV is addressed through a table of pages, two sequences can **point their tables at the same physical blocks** — sharing KV — and only **copy-on-write** a block when their contents diverge. (This also makes parallel sampling and beam search cheap: all candidates share the prompt's blocks.)

---

## 5.3 Prefix sharing and RadixAttention: reuse, don't recompute

Many requests share a common **prefix**: a fixed system prompt, few-shot exemplars, or — the canonical case — *one long document followed by many different questions* (RAG, long-document QA). Their prefix produces **identical KV**. Recomputing it from scratch for every request, and storing N copies, is pure waste that §5.2's block sharing makes avoidable.

- **Prefix caching.** Cache the shared prefix's KV blocks; an incoming request whose prefix matches reuses them, so it only prefills its **unique suffix**. The shared prefill is computed **once**.
- **RadixAttention** (SGLang, 2023) generalizes this with a **radix tree** over cached KV keyed by token sequences. The tree captures *all* shared prefixes automatically — not just one fixed system prompt — and handles branching (requests sharing a partial prefix then diverging) as tree structure. When memory is tight, least-recently-used paths are evicted (Going Deeper).

The payoff has two parts, both large on prefix-heavy workloads:

- **Compute saved** — the shared prefix is prefilled once, not N times → a major TTFT and throughput win.
- **Memory saved** — one copy of the prefix's KV instead of N.

This is exactly the long-document-QA workload the course uses, and it is why SGLang foregrounds RadixAttention. **But the benefit is entirely workload-dependent**: on traffic with no shared prefix (e.g. independent ShareGPT chats), prefix caching does essentially nothing. Measuring it on the wrong traffic is the Module 4 / Module 12 realism trap, and the lab confronts it head-on.

---

## 5.4 Two halves of "manage where the KV lives"

Module 4 said the scheduler decides *which* sequences run; this module is the other half — *where their KV lives* — and it has two layers:

- the **allocator** — PagedAttention — eliminates fragmentation so memory is nearly fully usable (raising Module 4's batch-capacity wall);
- the **reuse + eviction layer** — RadixAttention / prefix caching — deduplicates KV across requests and decides what to evict under pressure.

They compose: prefix sharing is *built on* paging's block-sharing. Together they convert Module 2's hard KV-capacity bound from "reserve max-length per sequence" into "use almost every byte, and share what's shareable."

---

## 5.5 Where the frontier is now

PagedAttention and prefix caching are **settled** — vLLM, SGLang, TGI, and TensorRT-LLM all ship them; they are the baseline. The 2024 frontier extends the cache **across the memory hierarchy and across instances**:

- **Tiered / offloaded KV caching.** Spill KV from GPU HBM → CPU host memory → SSD to serve contexts or concurrency beyond GPU memory (e.g. LMCache), trading latency for capacity.
- **KVCache-centric, cross-instance reuse.** Treat the KV cache as a **distributed store** shared across serving instances, so a prefix computed by one instance is reused by another (e.g. Mooncake's disaggregated, cache-centric architecture). The cache becomes the center of the system, not a per-instance afterthought.

The arc: contiguous allocation (broken) → **PagedAttention** (paging within a GPU, 2023) → **RadixAttention** (dedup within an instance, 2023) → **tiered + cross-instance KV stores** (the cache as distributed, hierarchical storage, 2024). Teaching only paging would stop two years short.

---

## 5.6 The picture to carry forward

- Continuous batching (Module 4) demands dynamic, variable-length KV allocation; **contiguous max-length reservation fragments memory** and throttles the achievable batch.
- **PagedAttention** pages the KV cache like OS virtual memory — bounding internal fragmentation to one block and eliminating external fragmentation — which **raises the batch ceiling** toward the roofline.
- **Prefix sharing / RadixAttention** reuses identical-prefix KV across requests (compute + memory saved) — but only on workloads that *have* shared prefixes.
- The frontier is **tiered and cross-instance KV stores** (Module 8's disaggregation builds on this).

---

## Going Deeper (appendix) — eviction and the scheduling↔memory interaction

When KV memory is full and a running or arriving sequence needs a block, something must give. Two contexts:

**Preemption of running sequences (the Module 4 ↔ Module 5 seam).** If an active sequence needs a block and none is free, the scheduler **preempts** a victim sequence by either **swapping** its KV out to CPU host memory (and back later) or **recomputing** it (drop the KV; redo the prefill on resumption). The choice is a cost trade: *recompute* is often cheaper for short sequences (prefill is fast and parallel) and avoids PCIe traffic; *swap* is better for long sequences whose recompute would be expensive. This is a place where scheduling (M4) and memory (M5) are inseparable.

**Prefix-cache eviction (sharing accounting).** Cached prefixes compete for memory and are evicted **LRU over the radix tree**, with **reference counting**: a block shared by *k* sequences carries refcount *k* and is freed only at zero, and a prefix currently in use by an active request cannot be evicted. The subtlety is the accounting — correctly tracking which blocks are shared, by whom, and which are safe to drop — which is what makes prefix sharing safe under concurrency.

---

## Lab 5 — Reclaim the memory, reuse the prefix (on the right workload)

**Context — what this builds on (checklist).** *Reuse:* `mem_estimate` (M0/M2) to predict contiguous-vs-paged capacity, and the `common/bench` open-loop generator + `common/traffic` **long-document-QA** builder (M4) — no new tooling. *Test the new insight:* §5.2 (paging kills fragmentation → higher batch) and §5.3 (prefix reuse), not a restated KV measurement. *Exercise prior quantities:* the higher batch should move throughput toward Module 1's roofline ceiling that Module 4 mapped. *Frontier:* KV offloading. *Methodology consistency:* prefix caching is tested on long-doc-QA **and** on ShareGPT, to prove the benefit is workload-dependent (the M4/M10 realism point).

1. **Paging vs contiguous — the fragmentation win.** At fixed KV memory, compare the **achievable batch** (max concurrent sequences) and resulting throughput under PagedAttention against the contiguous-max-length baseline `mem_estimate` predicts (reserve `max_model_len` per sequence). Show paging yields several× more concurrency, and that the higher batch lifts throughput toward Module 1's roofline ceiling (Module 4's wall, raised). *Confirms: §5.2.*

2. **Prefix sharing — and why the workload decides.** On **long-document-QA** (one shared document prefix, varied questions), run prefix caching **off vs on**; measure TTFT and throughput, and **sweep the prefix-sharing ratio**, showing the gain scale with it. Then run the *same* prefix-cache test on **ShareGPT** (no shared prefix) and show ≈ zero benefit. *Confirms: §5.3 — and that benchmarking prefix caching on ShareGPT would have falsely declared it useless.*

3. **Memory pressure — preemption.** Drive concurrency past KV capacity and observe the scheduler **preempt** (swap or recompute) victim sequences; measure the latency cost. *Confirms: Going Deeper — scheduling and memory are one problem under pressure.*

4. **(Frontier) KV offloading.** If available, enable CPU KV offloading and show context/concurrency extending past GPU memory at a latency cost; note the tiered / cross-instance direction (LMCache, Mooncake). *Confirms: §5.5.*

**Deliverable:** the paging-vs-contiguous achievable-batch and throughput comparison, with roofline proximity noted; the prefix-cache TTFT/throughput **gain-vs-sharing-ratio** curve on long-doc-QA **plus** the ShareGPT null-result beside it; and the preemption latency cost. **Mastery test — defend in one sentence each:** *why contiguous max-length allocation wastes most of KV memory (internal + external fragmentation); how paging fixes both and what it costs; why prefix sharing helps only when the workload has shared prefixes, and why testing it on ShareGPT misleads; and what the scheduler does under memory pressure (swap vs recompute).* *Feeds:* Module 7 (long-context, where this cache is the enemy) and Module 8 (disaggregation, built on cross-instance KV).

**Reading:** Kwon et al., *PagedAttention / vLLM* (SOSP 2023); Zheng et al., *SGLang / RadixAttention* (2023). Current frontier: *Mooncake* (2024) for KVCache-centric disaggregation; *LMCache* (2024) for KV offload/reuse.