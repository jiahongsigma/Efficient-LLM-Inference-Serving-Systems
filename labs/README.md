# Labs

One lab per module. Every lab drives the shared **`common/`** harness against a **real serving
engine** (vLLM / SGLang) over the OpenAI-compatible API — you rent a GPU, run a prepared sweep,
and tear it down (the metered-GPU discipline the course preaches; see the root README).

- **Built & runnable:** `m04_batching/`, `m10_agentic/` — each has a `run_lab.py` + `README.md`.
  *(They also ship a `--endpoint sim` dev mode for wiring the script without a GPU; the real
  numbers come from the server.)*
- **Implemented — first real-hardware run welcome:** the other 11 — each now has a `run_lab.py`
  written against the harness (plus `gateway.py` for M11) and a README with the steps + the
  **server to rent**. These haven't been run on a GPU here yet, so try them on your own hardware
  and tell us how they behave across environments (see the banner atop each `run_lab.py`). The
  shared memory math (`common/mem.py`, used by Labs 0/2/3/5/7) *is* unit-tested.

Every lab is wired the same way: `OpenAIEndpoint(url, model)` → a `common.traffic` builder →
`run_open_loop` → `compute_metrics` / `common.eval`. No simulation — they hit a real engine.

## Which server to rent for each lab

| Lab | GPU you need | Rent from (pick one) |
|---|---|---|
| **M0** numerics | none — the calculator is laptop math; **24 GB** only to *validate* | **RunPod** / **Vast.ai** (or your laptop) |
| **M1** roofline | **1× H100 80 GB** — needs real achieved HBM bandwidth / Nsight | **Lambda** / **RunPod** (H100) |
| **M2** kv-cache | **1× 24–48 GB** (A10 / L40S / 4090) | **RunPod** / **Lambda** |
| **M3** quantization | **1× 24–48 GB Ada/Hopper** (FP8 needs Ada+) | **RunPod** (L40S/4090) / **Vast.ai** |
| **M4** batching ✅ | **1× 24–48 GB** | **RunPod** / **Lambda** |
| **M5** paging | **1× 24–48 GB** | **RunPod** / **Lambda** |
| **M6** parallelism | **2–4× A100/H100 + NVLink** | **Lambda** (NVLink nodes) / **CoreWeave** / **Crusoe** |
| **M7** long-context | **1× H100 80 GB** (128K KV ≈ 16 GB/seq) | **Lambda** / **RunPod** (H100) |
| **M8** speculative | **1× 24–48 GB** (draft + target fit) | **RunPod** / **Lambda** |
| **M9** structured | **1× 24–48 GB** | **RunPod** / **Lambda** (vLLM / SGLang) |
| **M10** agentic ✅ | **1× 24–48 GB** | **RunPod** / **Lambda** |
| **M11** gateway | **2 backends** (2 small pods, or 1 GPU + 2 ports) | **RunPod** (2 pods) / **Lambda** |
| **M12** methodology | **reuse any single-GPU box** | **RunPod** / hosted API (**Together** / **Fireworks**) for trace replay |

### Provider notes

- **Lambda** — on-demand A100/H100 and multi-GPU **NVLink** nodes; the pick when you need an
  80 GB card (M1, M7) or real NVLink (M6).
- **RunPod** — the widest GPU range (RTX 4090 → H100), per-second billing, cheapest convenient
  default for the single-GPU labs.
- **Vast.ai** — a marketplace of consumer GPUs (3090/4090) at the lowest price; ideal for the
  budget single-GPU labs and the consumer-hardware angle of Appendix B.
- **CoreWeave / Crusoe** — enterprise multi-GPU H100/NVLink clusters; M6 at scale / multi-node PP.
- **Hosted OpenAI-compatible** (**Together**, **Fireworks**, **Baseten**) — only give you an
  *endpoint*, not engine control, so they fit M12 trace-replay and quick checks, not the
  serving-internals labs.

> Most labs are **single 24–48 GB GPU, <1 hour** — a few dollars each. Only M1/M7 (one 80 GB H100)
> and M6 (multi-GPU NVLink) cost more. Prepare and dry-run the script locally first, then rent.
