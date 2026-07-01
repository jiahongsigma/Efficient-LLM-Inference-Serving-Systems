# `infra/` — cheap-GPU logistics

The course's standing discipline: **GPU time is metered; idle time is wasted money.** Do all
setup, scripting, and analysis off the accelerator, power the GPU on only to execute a prepared
sweep, then release it. `serve_and_run.sh` automates exactly that loop.

## The one-command loop

```bash
# launch an engine, wait for it, run a lab against it, tear it down — all in one go:
infra/serve_and_run.sh \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --lab labs/m05_paging/run_lab.py \
    -- --label cache_on            # everything after `--` is passed to the lab

# pass engine flags with --serve-args (quote them):
infra/serve_and_run.sh --model <id> --lab labs/m03_quantization/run_lab.py \
    --serve-args "--quantization awq" -- --label int4_awq --params-b 4.0

# keep the server up (e.g. to run several labs against it): add --keep
```

It starts vLLM (or `--engine sglang`), polls `/v1/models` until ready, runs the lab with
`--endpoint http://localhost:PORT --model <id>`, and kills the engine on exit (even on Ctrl-C).

## Which box to rent

Per-lab GPU + provider recommendations live in [`../labs/README.md`](../labs/README.md). The short
version — **most labs are one 24–48 GB GPU for under an hour**:

| You need | Labs | Rent from |
|---|---|---|
| 1× 24–48 GB | M0(validate), M2, M3, M4, M5, M8, M9, M10, M12 | **RunPod** / **Vast.ai** / **Lambda** |
| 1× H100 80 GB | M1 (roofline + Nsight), M7 (long context) | **Lambda** / **RunPod** |
| 2–4× NVLink | M6 (parallelism) | **Lambda** / **CoreWeave** / **Crusoe** |
| 2 backends | M11 (gateway) | **RunPod** (2 pods) / **Lambda** |

**Batch tip:** rent *one* 24–48 GB box and run M2, M3, M5, M8, M9, M12 (+ M0 validate, M4, M10)
back-to-back in a single session — that clears ~9 of the 13 labs for a few dollars.

## Rough cost (on-demand, 2025-ish — verify live)

| GPU | ~$/hr | Good for |
|---|---|---|
| RTX 4090 (24 GB) | $0.30–0.50 | M0/M2/M3/M5/M8/M9/M12, INT4 |
| L40S / A10 (48 GB) | $0.50–0.90 | the same, with headroom |
| A100 80 GB | $1.20–1.90 | M1/M7, larger models |
| H100 80 GB | $2.00–3.50 | M1 (Nsight), M7, FP8 |
| 8× A100/H100 (NVLink) | $12–28 | M6 multi-GPU |

## Persist the weights (don't re-download every rental)

Hugging Face caches to `~/.cache/huggingface`. On a fresh box, point it at a persistent volume so
re-renting doesn't re-pull tens of GB:

```bash
export HF_HOME=/workspace/hf            # a persistent/volume path on the provider
export HF_HUB_ENABLE_HF_TRANSFER=1      # faster downloads
huggingface-cli login                   # for gated models (Llama)
```

## On the rented box

```bash
git clone https://github.com/jiahongsigma/Efficient-LLM-Inference-Serving-Systems
cd Efficient-LLM-Inference-Serving-Systems
pip install -r infra/requirements-gpu.txt     # engine + harness deps

# 1) smoke-test the wiring (30s) BEFORE any lab:
vllm serve <model> --port 8000 &
python infra/smoke.py --endpoint http://localhost:8000 --model <model>
#    -> prints status / TTFT / tokens; exit 0 means the whole harness reaches the engine.

# 2) then run labs (one-command serve+run+teardown):
infra/serve_and_run.sh --model <model> --lab labs/m05_paging/run_lab.py -- --label cache_on
```

Start with **M5** and **M12** — they're the most harness-supported and the most likely to run
clean on the first try.

> **Note:** `serve_and_run.sh` is validated for shell syntax only (no GPU here). Sanity-check it on
> your first rental; engine flag names drift across vLLM/SGLang versions.
