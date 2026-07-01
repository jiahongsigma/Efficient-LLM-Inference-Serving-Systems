# Module 2 — Attention, the KV Cache, and Where the Memory Goes

*Prerequisites: Module 0 (bytes ↔ parameters; the serve-time memory breakdown) and Module 1 (decode is bandwidth-bound; the attention `O(S·d)`-per-token term). This module explains the structure that dominates serving memory — the KV cache — derives its size, surveys the architectures invented to shrink it, and shows how FlashAttention removes attention's hidden memory-traffic cost.*

---

## In plain English

**Why this matters.** As a conversation gets longer, your GPU fills up and slows down — even though the model itself hasn't changed. The culprit is a hidden, growing scratchpad called the KV cache, and it's the single biggest limit on how many users you can serve at once.

**What this module gives you.** What the KV cache is, a formula for exactly how big it gets, the model-design tricks (MHA → MQA → GQA → MLA) invented to shrink it, and FlashAttention, which keeps attention from becoming a memory hog.

**How it works (the intuition).** To avoid re-reading the whole conversation for every new word, the model keeps notes on everything said so far. Those notes pile up with every token and every concurrent user. This module shows you how to size that pile — and the engineering used to keep it from overflowing.

---

## 2.1 Why decode needs a cache: trading compute for memory

Self-attention over a sequence of length `S` forms an `S × S` score matrix: every token attends to every earlier token. Written out, with per-token query/key/value vectors `q,k,v` of width `d`:

```
scores = Q Kᵀ / √d        [S × S]
A      = softmax(scores)   (row-wise)
out    = A V               [S × d]
```

In **decode** the model emits one token at a time. At step *t* the new token's query `qₜ` must attend to the keys and values of *all* previous tokens, `K[1:t]`, `V[1:t]`. The keys and values of past tokens **do not change** as generation proceeds — so recomputing them every step is pure waste.

That waste is quantifiable. Without a cache, step *t* re-runs the K/V projections for *t* tokens at `O(t·d)` cost; summed over a generation of length *S* that is **`O(S²·d)` of redundant projection work**. The fix is to **store** `K[1:t]` and `V[1:t]` and append one new `(kₜ, vₜ)` per step — the **KV cache**. It converts `O(S²·d)` redundant compute into `O(S·d)` of stored memory plus `O(S·d)` of (non-redundant) projection work.

One subtlety to carry into Module 7: the cache eliminates redundant *projection*, not the attention itself. Computing `qₜ · K[1:t]ᵀ` still costs `O(t·d)` per step, i.e. **`O(S²·d)` of attention compute over a full generation** regardless of caching. This is exactly Module 1's crossover term — at long context, attention compute (not weight reads) becomes the bottleneck.

So the KV cache is a classic space-for-time trade. The "time" it buys is large; the "space" it costs is the central problem of the rest of this module.

---

## 2.2 The KV-cache size law, and why it re-binds decode

From Module 0, the cache holds, per token, a K and a V vector in every layer:

```
KV_bytes_per_token = 2 · n_layers · n_kv_heads · head_dim · dtype_bytes
KV_total           = KV_bytes_per_token · seq_len · batch_size
```

The leading 2 is K and V; `n_kv_heads · head_dim` is the width stored per token per layer; the cache has its *own* dtype (often FP16, increasingly FP8).

*Worked (Llama-3.1-8B: 32 layers, 8 KV heads via GQA, head_dim 128, FP16):*
```
2 · 32 · (8·128) · 2 = 131,072 bytes/token ≈ 0.125 MB/token
```
At 8K context that is **1 GB per sequence**; at batch 32, **32 GB of KV** — more than twice the model's 14 GB of weights. The §2.3 architectures exist because this number, not the weights, sets the achievable batch size and context length.

**The bandwidth angle (the link back to Module 1).** Module 1's clean result — batching slides decode up the roofline because weights are read once and amortized across the batch — assumed weights were the only bytes moved. They are not: each sequence must also **read its entire KV cache every decode step**. Those reads scale with `batch × context`, so past some point KV-cache bandwidth, not weight bandwidth, bounds throughput. This is why "just batch more" stops working at long context, and why both axes — *footprint* (fits in memory) and *traffic* (fits in bandwidth) — must shrink. Everything below shrinks both.

---

## 2.3 The architecture line: MHA → MQA → GQA → MLA

The number of **KV heads** is the lever. Query heads are cheap to keep; it is the *cached* K and V that hurt. Four designs trade KV size against quality:

| Scheme | n_kv_heads | KV per token (rel. to MHA) | Quality | Typical use |
|---|---|---|---|---|
| **MHA** (multi-head) | H (= n_query_heads) | 1× | reference | original Transformer |
| **MQA** (multi-query) | 1 | **1/H** | degrades; training can destabilize | older/large-batch models |
| **GQA** (grouped-query) | G (1 < G < H) | **G/H** | ≈ MHA | **today's default** (Llama-3, Qwen2.5, …) |
| **MLA** (latent) | — (latent compression) | small; ≈ MQA-level or better | ≈ MHA | DeepSeek-V2/V3 |

- **MHA** gives every query head its own K and V. Full quality, full cost: KV ∝ `H · head_dim = d_model`.
- **MQA** (Shazeer, 2019) forces *all* query heads to share a **single** K/V head: KV drops by a factor of `H` (e.g. 32×). A large memory win, but quality falls and training is less stable — all heads are yoked to one K/V subspace.
- **GQA** (Ainslie et al., 2023) is the interpolation that won. Partition the `H` query heads into `G` groups, each group sharing one K/V head: KV drops by `H/G`. Llama-3.1-8B uses `H=32, G=8` → a **4× KV reduction at near-MHA quality**. The reason GQA dominates is that it sits at the knee of the quality–memory curve: most of MQA's savings, almost none of its quality loss.
- **MLA** (DeepSeek-V2, 2024) changes the mechanism rather than the head count. Instead of fewer KV heads, it projects K and V down to a **shared low-rank latent** `cₜ` (width `d_c ≪ d_model`) and caches *that*; full K and V are reconstructed on the fly during attention, with the up-projection matrices **absorbed** into the query and output projections so they are never explicitly materialized. The cached latent is a fraction of even GQA's footprint while retaining quality close to full MHA — the latent carries more information than a single shared head. *(Going Deeper covers MLA's one wrinkle: it is incompatible with naive RoPE, so DeepSeek splits off a small "decoupled RoPE" dimension to carry position.)*

The throughline: each design is a direct response to §2.2. When you read a model card, `n_kv_heads` tells you its serving memory profile before you load it.

---

## 2.4 FlashAttention: computing attention without paying HBM for it

GQA/MLA shrink the cache's *footprint*. A separate problem is the *traffic of computing attention itself* — and it is, surprisingly, a memory problem, not a compute one.

Naive attention materializes the `S × S` score matrix in HBM: compute `S = QKᵀ`, **write it to HBM**, read it back to softmax, write the result, read it back to multiply by `V`. The `S × S` matrix round-trips through HBM, so attention's cost is dominated by `O(S²)` memory traffic — the tensor cores sit idle waiting on HBM, exactly the Module 1 pathology, now from attention rather than weights. It also costs `O(S²)` *memory* to hold the matrix, which is why naive long-context attention OOMs.

**FlashAttention** (Dao et al., 2022) never materializes the score matrix. Two ideas:

1. **Tiling.** Stream Q, K, V through the kernel in blocks small enough to live in on-chip **SRAM** (fast, ~tens of KB), computing the output block by block and keeping intermediates on-chip.
2. **Online (streaming) softmax.** Maintain a running max and running normalizer so each output block is accumulated correctly without ever seeing the whole row at once (the recurrence is in Going Deeper).

The payoff: HBM accesses for attention fall from `Θ(S² + Sd)` to `Θ(S²d²/M)` (with `M` the SRAM size), proven **IO-optimal** up to constants — a large reduction whenever `M ≫ d²`. Attention memory becomes **linear in `S`, not quadratic**, and attention moves off the HBM-bandwidth wall toward compute-bound. Concretely this is what makes long contexts feasible and gives the multi-× wall-clock speedups you will measure. (A decode-time variant, FlashDecoding, further parallelizes across the KV length so the long-context decode step is not serialized over the cache.)

FlashAttention is on by default in vLLM and SGLang; you rarely invoke it directly, but you must understand it, because it is *why* the KV cache (and not a quadratic score matrix) is the thing you budget for.

---

## 2.5 The picture to carry forward

- The KV cache trades `O(S²·d)` recompute for `O(S·d)` memory; that memory is the dominant *variable* term in serving (§2.2).
- It binds on two axes — **footprint** (fits in VRAM) and **traffic** (fits in bandwidth) — and both scale with `batch × context`.
- `n_kv_heads` is the architectural lever: **GQA** is the current default (4×-ish savings, near-zero quality cost); **MLA** pushes further by caching a latent.
- **FlashAttention** removes attention's `O(S²)` HBM traffic and memory by never materializing the score matrix — making the KV cache, not the attention matrix, the quantity you manage.

Module 5 is *how a serving engine physically manages this cache* (paging, sharing); Module 7 is *what to do when context itself is the enemy* (eviction, KV quantization, sparse attention). Both stand on this module's size law.

---

## Going Deeper (appendix) — the online softmax and FlashAttention's IO bound

**Safe streaming softmax.** Process the score row in blocks. Maintain running max `m`, running denominator `ℓ`, and running output accumulator `O`. For a new block of scores `s` (with values `v`):

```
m' = max(m, max(s))
ℓ' = e^{m−m'}·ℓ + Σ e^{sᵢ−m'}
O' = e^{m−m'}·O + Σ e^{sᵢ−m'}·vᵢ
```

After the last block, `out = O / ℓ`. The `e^{m−m'}` factors rescale the previously-accumulated partial results when a larger max appears, keeping the computation numerically identical to a full-row softmax while never storing the full row. This recurrence is the heart of the kernel.

**The IO bound.** Standard attention requires `Θ(Sd + S²)` HBM accesses (the `S²` from materializing scores). FlashAttention, by tiling so the working set fits in SRAM of size `M`, requires `Θ(S²d²/M)` HBM accesses, which Dao et al. prove is optimal over `d ≤ M ≤ Sd` up to constant factors. For typical `M` (tens of KB) and head dim `d` ≈ 64–128, the reduction factor `~M/d²` is roughly 10–100×, which is why attention stops being HBM-bound. We cite the optimality theorem rather than reproving it; the takeaway is that the materialization is the cost, and avoiding it is provably the best you can do.

**MLA and RoPE.** MLA's absorption trick assumes K can be reconstructed by a fixed linear map from the latent — but RoPE applies a *position-dependent* rotation to K, breaking that linearity. DeepSeek's fix is *decoupled RoPE*: a small extra dimension carries the rotary positional component separately from the compressed content latent, so the bulk of K/V stays compressible while position is still encoded.

---

## Lab 2 — Measure the cache, fit the law, and watch context bind the roofline

**Context — what this builds on.** You are not starting from scratch. From **Module 0** you have `mem_estimate(model, dtype, context, batch)` and the per-dtype byte costs; this lab **extends and stress-tests its KV term** rather than rebuilding it. From **Module 1** you have the roofline and a batch-sweep; here you push that sweep into the **context** dimension to expose the *traffic* axis of §2.2. Each step below validates exactly one claim from the notes — footprint, traffic, the dtype factor, the `H/G` saving, and FlashAttention's linear memory — so you can point to which sentence of §2.1–2.4 you just confirmed.

1. **Footprint — fit the size law (§2.2, extending the M0 tool).** Extend `mem_estimate`'s KV term; serve an 8B model (vLLM) and measure reserved KV memory across context {2K, 8K, 32K} × batch {1, 8, 32}. Fit `KV_total` to the measurements, then use it to predict the (context × batch) that exhausts the pool and confirm by driving the server to OOM. *Confirms: KV footprint scales as `batch × context`.*

2. **Traffic — context slides you back down the roofline (the Module 1 link).** Hold batch fixed and sweep *context* {2K, 8K, 32K, 64K}; measure decode throughput (tokens/s). Throughput falls as context grows, because each decode step now rereads a larger KV cache — the bandwidth term §2.2 warned about. State the contrast explicitly: in Module 1, batching *climbed* the roofline; here, long context *descends* it, at fixed batch. *Confirms: KV traffic, not just footprint, bounds decode — this module's headline claim, and the one the old lab missed.*

3. **The dtype factor (using M0's bytes).** Re-run step 1 with `--kv-cache-dtype fp8`. The per-token KV bytes should halve, exactly as the `dtype_bytes` term predicts; record the footprint change, any throughput gain (less KV traffic to move), and a short accuracy check. *Confirms: the dtype factor in the size law; previews Module 7's KV quantization.*

4. **The `H/G` saving (stated honestly).** Compute KV-bytes-per-token *analytically from the configs* of a full-MHA model (e.g. Llama-2-7B, `H=32`) and a GQA model (Llama-3.1-8B, `G=8`), then validate each number against that model's *own* measured footprint and relate the ratio to `H/G`. Note the confound: the two models differ in more than KV-head count, so this is a config-derived prediction checked against measurement — not a controlled experiment. *Confirms: GQA's `H/G` footprint reduction.*

5. **FlashAttention — linear vs quadratic (standalone kernel, not the server).** Outside vLLM, micro-benchmark two attention kernels on a toy tensor: a naive `softmax(QKᵀ/√d)·V` that materializes the `S×S` matrix, vs `torch.nn.functional.scaled_dot_product_attention` (flash / memory-efficient backend). Plot `torch.cuda.max_memory_allocated()` against sequence length: the naive curve is `O(S²)`, the flash curve `O(S)`. (Doing this as a kernel micro-benchmark is both cleaner and more reproducible than trying to toggle attention backends inside the serving engine.) *Confirms: §2.4 — FlashAttention never materializes the score matrix.*

**Deliverable:** an extended `mem_estimate` whose KV term is validated against measured OOM within ~10%; the **context-throughput curve** showing decode descending the roofline (the Module 1 link); the FP16-vs-FP8 KV footprint comparison; the config-derived MHA-vs-GQA table; and the linear-vs-quadratic attention-memory plot. **Mastery test — defend in one sentence each:** *why the KV cache trades compute for memory; why long context degrades decode throughput even at fixed batch; why GQA buys ~`H/G`× at near-zero quality cost; and what FlashAttention refuses to write to HBM.* *Feeds:* Module 5 (managing this cache) and Module 7 (defeating it at long context).

**Reading:** Dao et al., *FlashAttention* (2022); Ainslie et al., *GQA* (2023); DeepSeek-V2 (2024) for *MLA*. Optional: Shazeer, *Fast Transformer Decoding* (2019) for the original MQA.