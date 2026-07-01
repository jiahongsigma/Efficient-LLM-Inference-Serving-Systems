# Appendix C — The Modern Serving Stack: vLLM and SGLang

*Companion notes to the README section. These are the two engines every lab runs on; this page is what they are, why they exist, and how to drive them. The deep dives into their internals are Modules 4–5 (batching, paging, radix) and Module 11 (frameworks and the API layer).*

---

## Why a research script is not a serving system

The natural first thing to reach for is Hugging Face `transformers`:

```python
out = model.generate(input_ids, max_new_tokens=256)
```

This is correct and useful for research, and unfit for serving, for concrete reasons that motivate everything below:

1. **One request at a time.** `generate()` has no notion of independent requests arriving at different times; to "batch" you must pad to a common length and submit together, wasting compute on padding and forcing the batch to finish at the slowest member's length.
2. **No cross-request scheduler.** Real traffic is a stream — requests arrive, finish, and vary wildly in length. Throughput depends on *scheduling* them onto the GPU continuously, which a single `generate()` call cannot do.
3. **Naive KV memory.** The KV cache is pre-allocated to the maximum length per sequence, fragmenting memory and capping how many sequences fit (recall the Module 0 sizing inequality).
4. **No SLO machinery.** No way to bound tail latency, shed load, or trade latency for throughput.

A serving engine exists to fix exactly these. Two open-source engines define current practice.

---

## vLLM — the throughput workhorse

**Defining idea: PagedAttention (Module 5).** vLLM manages the KV cache the way an operating system manages RAM: the cache is split into fixed-size *blocks*, and each sequence holds a *block table* mapping logical positions to physical blocks. Sequences need not be contiguous in memory, so internal/external fragmentation nearly vanishes and far more sequences fit concurrently. Copy-on-write blocks let sequences share identical prefixes cheaply.

**Second idea: continuous (iteration-level) batching (Module 4).** Instead of forming a fixed batch and waiting for it to finish, vLLM makes a scheduling decision *every decode step*: finished sequences leave, waiting ones join. The batch composition is fluid, which keeps the GPU saturated under streaming traffic — the direct remedy to problems (1) and (2) above.

**What you get:** an OpenAI-compatible HTTP server (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`), tensor parallelism, most quantization formats (AWQ, GPTQ, FP8, …), prefix caching, and speculative decoding. It is the de-facto throughput baseline against which other systems are measured.

**Driving it:**
```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-prefix-caching \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.90 \
  --api-key $KEY
# multi-GPU:        --tensor-parallel-size 2
# quantized:        --quantization awq
```
Key knobs you will sweep in the labs: `--max-num-seqs` (max concurrent sequences), `--max-num-batched-tokens` (token budget per step — the prefill/decode balance), `--max-model-len` (context cap, which bounds KV via Module 0), `--gpu-memory-utilization` (how much VRAM to claim for the KV pool).

---

## SGLang — prefix reuse and structured generation

**Defining idea: RadixAttention (Module 5).** vLLM shares prefixes when explicitly told; SGLang shares them *automatically*. It maintains a **radix tree** over the KV cache keyed by token prefixes, so any new request whose prefix already exists in the tree reuses that cached KV instead of recomputing it. For workloads where many requests share a long head — a fixed system prompt, few-shot exemplars, a retrieved document — this is a large, structural win (you compute the shared prefix once and amortize it across all requests that share it).

**Second strength: first-class structured/constrained decoding.** SGLang ships a frontend for forcing outputs to obey a grammar or JSON schema efficiently (compressed finite-state machines over the vocabulary), plus a high-concurrency scheduler. When the application needs the model to reliably emit valid structured output, SGLang's constrained decoding is a primary tool — and the labs measure its throughput cost.

**What you get:** also OpenAI-compatible, with tensor parallelism, quantization, and speculative decoding — so it is a drop-in alternative behind the same API.

**Driving it:**
```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --tp 2 \
  --port 8002 \
  --api-key $KEY
```

---

## The lineage (why the course is ordered the way it is)

These engines are the latest points on a short, legible history, and the module sequence deliberately retraces it:

- **Orca** (OSDI 2022) — introduced *iteration-level batching*, the scheduling insight. → Module 4.
- **vLLM** (SOSP 2023) — introduced *PagedAttention*, the memory-management insight. → Module 5.
- **SGLang** (2023–24) — introduced *RadixAttention* and efficient structured decoding, the prefix-sharing and programmability insights. → Module 5 and the structured-output labs.

Teaching the techniques in the order they were discovered is not nostalgia: each one is the answer to a bottleneck the previous one exposed, which is the same causal spine the README lays out.

---

## How this course uses them

- **vLLM is the default engine** for most labs (Modules 1–4, 6–7): it is the throughput baseline and the most widely documented.
- **SGLang is the comparison engine**, foregrounded where its ideas matter most — the prefix-sharing lab (Module 5) and the structured-output-under-load lab (Module 9).
- **Both sit behind one OpenAI-compatible interface**, so the gateway you build in Module 11 routes to either without the client knowing. Being fluent in launching, configuring, instrumenting, and benchmarking *both* — and explaining when each wins — is an explicit learning outcome.

A note on honesty in comparison (Module 12): these engines evolve fast and leapfrog each other on benchmarks. The course's stance is never "X is faster than Y" as a fact to memorize, but "here is how to measure which wins *for a given workload*, and here is the architectural reason." Run them on your traffic; do not trust a leaderboard screenshot.