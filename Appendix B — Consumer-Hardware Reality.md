# Appendix B — Consumer-Hardware Reality

*The course is anchored on the H100 (Appendix A). Most people serving an LLM "locally" own a 4090, a Mac, or a couple of gaming GPUs on PCIe — a different substrate. This appendix re-grounds the same principles on that hardware. **Nothing in the physics changes**; the central law still holds. What changes is the numbers, the binding constraint, and the levers you actually have. The goal is that after this, the local-LLM questions — "which quant?", "why is my Mac slow?", "will two 3090s help?" — answer themselves from the modules you already read.*

**Referenced by:** Module 1 (the law that explains everything below), Module 3 (GGUF is this module in folk clothing), Module 6 (which collapses without NVLink), Module 7 (KV-OOM is the local user's wall), Module 11 (where llama.cpp/Ollama sit vs vLLM).

---

## B.1 The central law is hardware-independent

Module 1's result — decode is **memory-bandwidth-bound** — is not an H100 fact, it is an arithmetic fact, so it governs your laptop too:

```
single-stream tokens/sec  ≈  memory_bandwidth / bytes_read_per_token
                          ≈  memory_bandwidth / model_size_in_bytes   (weights dominate the read)
```

This one equation explains the entire consumer experience. Take a **70B model quantized to 4-bit (~40 GB of weights)** and read tokens/sec straight off the bandwidth:

| Machine | Mem bandwidth | Fits 40 GB? | ~tokens/sec (40 GB read/token) |
|---|---|---|---|
| H100 SXM5 | 3.35 TB/s | yes (80 GB) | ~84 |
| RTX 4090 | ~1.0 TB/s | **no (24 GB)** → CPU offload | ~2–5 (offload-bound) |
| M3 Max (128 GB) | ~400 GB/s | yes | ~10 |
| M3 Ultra (512 GB) | ~819 GB/s | yes | ~20 |
| CPU (DDR5, ~80 GB/s) | ~0.08 TB/s | yes (system RAM) | ~2 |

(Real numbers run lower — overhead, sub-peak bandwidth — but the *ratios* are exactly what you live.) Everything important is in this table: the **4090 is fast but too small**, the **Mac fits the model but decodes slowly**, the **CPU fits anything and crawls**. All three are the same law with different bandwidth and capacity. "Why is my Mac slow even though the model loaded?" — because it loaded (capacity) but decode reads 40 GB per token at 400 GB/s (bandwidth). The course predicted your lived experience.

---

## B.2 The two consumer constraints: capacity and bandwidth

The H100 has plenty of both; consumer hardware forces a brutal trade, and each substrate sits at a different corner:

- **A single gaming GPU (RTX 4090, Ada).** ~1 TB/s bandwidth (good — a third of an H100), but **24 GB capacity** (the binding constraint) and, critically, **no NVLink** (§B.4). Ada *does* have FP8 tensor cores (the 3090/Ampere does not — a reason the 4090 quantizes faster). It is the **high-bandwidth, low-capacity** corner: small models fly, big models don't fit.
- **Apple Silicon (M-series), unified memory.** The CPU and GPU share one memory pool, so the "GPU memory" is your whole RAM — up to **128 GB (M3 Max)** or **512 GB (M3 Ultra)**. That fits models a 4090 cannot dream of. But bandwidth is **~100 GB/s (M3) to ~819 GB/s (M3 Ultra)** — below a 4090, well below an H100. It is the **high-capacity, lower-bandwidth** corner: it *fits* the 70B (or even a 405B on the big ones) and decodes it slowly. Great for large models at reading pace; not for low latency.
- **CPU / CPU-offload.** System RAM is the largest and cheapest capacity, at **DDR bandwidth (~50–100 GB/s desktop, ~200–400 GB/s 8-channel server)** — an order of magnitude under a GPU. The fallback when the model won't fit in VRAM: llama.cpp offloads the overflow layers to CPU (`n_gpu_layers`), and those layers then decode at RAM speed, which (per §B.1) dominates and tanks the rate. Offload is a capacity bridge you pay for in bandwidth.

The local hardware decision is just *which corner of the capacity-bandwidth trade you want* — and that is a Module 1 decision, not a brand preference.

---

## B.3 Quantization is your main lever — and GGUF is Module 3 in folk clothing

For a single local user, **quantization (Module 3) is the dominant knob**, because it attacks *both* constraints at once: fewer bytes per weight means the model **fits** (capacity) *and* each token **reads less** (bandwidth → speed). The local ecosystem ships this as **GGUF** quant types (llama.cpp), whose cryptic names map cleanly onto Module 3 once decoded:

| GGUF type | ~bits/wt | Module 3 equivalent |
|---|---|---|
| `Q8_0` | ~8.5 | INT8 weight-only — effectively lossless, the safe default |
| `Q6_K` | ~6.6 | near-lossless k-quant |
| `Q5_K_M` | ~5.7 | high-quality 5-bit, group-scaled |
| **`Q4_K_M`** | **~4.8** | **4-bit group-wise, the practical sweet spot — matches Module 3's "4-bit weight-only" finding** |
| `Q3_K_M` | ~3.9 | aggressive, visible quality drop |
| `Q2_K` | ~2.6 | last resort, heavy degradation |
| `IQ4_XS`, `IQ3_XXS`, `IQ2_XXS` | ~2–4 | importance-aware / codebook quants (the AWQ–AQLM family) |

The structure is pure Module 3:
- The **`_K` ("k-quant") super-block** layout is **group-wise quantization** (Module 3's group scaling) — small blocks of weights each with their own scale.
- **`Q4_K_M` keeping a few sensitive tensors (attention `wv`, FFN `down`) at `Q6_K`** is **sensitivity-aware mixed precision** — llama.cpp's empirical version of Module 3's "not all layers tolerate quantization equally."
- The **`imatrix` (importance matrix)** used by the `IQ` quants is computed by running calibration text through the model — **activation-aware calibration**, exactly AWQ/GPTQ's idea (Module 3).
- The **`IQ` codebook quants** are the consumer face of Module 3's **vector-quantization frontier** (QuIP#/AQLM).

So the practical recommendation falls out of Module 3, not folklore: **`Q4_K_M` or `Q5_K_M` for the general sweet spot; `Q8_0` if you have the room and want lossless; an `IQ` quant (with imatrix) only when you must squeeze a larger model into limited VRAM and will accept the quality cost.**

And don't forget the **KV cache** (Modules 2, 7): on 24 GB it OOMs at long context even when the weights fit, so llama.cpp lets you quantize it (`--cache-type-k/v q8_0`/`q4_0`) — which is **Module 7's KV quantization**, the same lever, now the thing standing between you and a long-context crash on a small card.

---

## B.4 Multi-GPU without NVLink: Module 6, crippled

Two 3090s or two 4090s is a common "fit a bigger model" move — but consumer cards (4090: none; 3090: a weak bridge, since removed) **lack NVLink**, so they talk over **PCIe (Gen4 ×16 ≈ 32 GB/s)** — ~30× less than an H100's NVLink. Module 6's conclusions invert accordingly:

- **Tensor parallelism is crippled.** Its frequent, synchronous all-reduces (Module 6) need NVLink's bandwidth and sub-µs latency; over PCIe they dominate and TP barely helps (often hurts). The data-center default is the wrong tool here.
- **Pipeline / layer-split is what works** — llama.cpp's multi-GPU mode splits *layers* across cards (Module 6's pipeline parallelism), whose communication is sparse and PCIe-tolerant. But note *what it buys*: it mostly extends **capacity** (a 70B spread across two 24 GB cards) rather than **speed** — at any moment one card is active and the link is the bottleneck, so you fit the model, you don't accelerate it much.

So "will a second GPU help?" — for **fitting** a bigger model, yes (layer split); for **faster** single-stream decode, mostly no (you're PCIe-bound, no NVLink). Exactly Module 6's interconnect lesson, read on the hardware you own.

---

## B.5 The framework landscape: opposite design centers

The course's engines (vLLM, SGLang) and the local engines optimize **opposite things**, which is why they feel so different:

- **vLLM / SGLang** — data-center, **many-user throughput**: continuous batching (Module 4), PagedAttention (Module 5), tensor parallelism over NVLink (Module 6). Built to keep an expensive GPU busy across many concurrent requests.
- **llama.cpp** — the local engine: C++, runs on **anything** (CUDA, Metal, ROCm, Vulkan, CPU), GGUF quant, and **CPU-offload** when the model doesn't fit. Optimized for **one user's tokens/sec on whatever you have**, not throughput across many. It is the opposite corner from vLLM by design.
- **Ollama** — a friendly distribution layer **on top of llama.cpp** (model pull/run, a REST API, modelfiles). Ollama : llama.cpp ≈ a gateway : an engine (Module 11) — the UX, not the inference.
- **MLX / mlx-lm** — Apple's native array framework for Metal and unified memory; often the fastest path on a Mac, and it does fine-tuning too. The Apple-native alternative to llama.cpp.
- **LM Studio** — a desktop GUI wrapping llama.cpp/MLX: the "I just want a chat window" layer.

Choosing among them is choosing a design center: **single-user-latency on your hardware (llama.cpp/Ollama/MLX)** vs **many-user-throughput on GPUs you rent (vLLM/SGLang)**. If you are the only user, the throughput machinery vLLM is famous for (Modules 4–5) is largely irrelevant to you; if you are serving an app, it is the whole point.

---

## B.6 Which modules actually bite locally

For a **single local user**, the course re-weights:

- **Central (read closely):** Module 0 (memory/precision math — does it fit?), **Module 1** (bandwidth = your tokens/sec), **Module 2 & 7** (the KV cache = your long-context OOM), **Module 3** (quantization = how you fit *and* speed up). These four are your daily reality.
- **Speed even for one user:** **Module 8's speculative decoding** is a *latency* technique, so it helps single-stream — and llama.cpp supports draft-model speculation. Worth knowing.
- **Mostly multi-user (skim unless you're serving an app):** Module 4 (batching), Module 5 (paging), Module 6 (parallelism over NVLink) — these pay off when you have *many* concurrent requests, which a solo user does not.
- **If you do it locally:** Module 9 (constrained decoding — llama.cpp has GBNF grammars) and Module 10 (agentic loops — same prefix-cache and tool-latency realities, smaller scale).

---

## B.7 The picture to carry forward

- The **bandwidth law (Module 1) is hardware-independent** and explains every consumer behavior: `tokens/sec ≈ bandwidth / model_bytes`.
- The local trade is **capacity vs bandwidth**: the **4090** has bandwidth but not capacity; **Apple Silicon** has capacity but less bandwidth; **CPU/offload** has capacity and almost no bandwidth.
- **Quantization (Module 3) is your main lever** — it fixes both at once; **GGUF is Module 3** (group scaling, sensitivity-mixing, imatrix calibration, codebook quants) with `Q4_K_M`/`Q5_K_M` the sweet spot; quantize the **KV cache** (Module 7) to survive long context on small VRAM.
- **No NVLink ⇒ Module 6 inverts**: layer-split for capacity, not tensor-parallel for speed; you are PCIe-bound.
- **llama.cpp/Ollama/MLX optimize single-user latency**; vLLM/SGLang optimize many-user throughput — opposite design centers, and the modules that matter to you depend on which you are.

*The discipline is the same; only the substrate changed. If Appendix A told you where the H100's numbers come from, this one told you how to read your own.*