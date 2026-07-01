# Contributing

Thanks for wanting to help. This is an open course on efficient LLM inference & serving — part
lecture notes, part a tested benchmarking harness, part hands-on labs. Contributions of every size
are welcome, from a typo fix to a validated lab to a whole new module.

## ⭐ The most useful thing you can do: run a lab on your own hardware

The labs run against a real engine on your own GPU, and we're genuinely curious how they behave
**across different environments** — GPUs, engine versions, drivers, quantization formats. That
variety is the interesting part. **Running a lab and reporting how it went on your setup** is the
single highest-value contribution. (Two of the thirteen, `m04_batching` and `m10_agentic`, also ship
an offline dev mode; the rest are meant to be exercised on real hardware.)

1. Pick a lab — start with **M5** or **M12** (most likely to run clean). `labs/README.md` says which
   GPU/provider each needs.
2. Confirm the harness reaches the engine first:
   ```bash
   pip install -r infra/requirements-gpu.txt
   vllm serve <model> --port 8000 &
   python infra/smoke.py --model <model>     # exit 0 = wired up
   ```
3. Run the lab (one command, with teardown):
   ```bash
   infra/serve_and_run.sh --model <model> --lab labs/mXX_name/run_lab.py -- <lab args>
   ```
4. **Open an issue (or a PR with a fix)** using the template below.

**Lab report template** (paste into a GitHub issue):
```
Lab:          m05_paging
GPU / engine: RunPod RTX 4090 / vLLM 0.6.x
Command:      infra/serve_and_run.sh --model meta-llama/Llama-3.1-8B-Instruct \
              --lab labs/m05_paging/run_lab.py --serve-args "--enable-prefix-caching" -- --label cache_on
Outcome:      <worked + the numbers/plot, OR the traceback>
Fix (if any): <diff or short description>
```
Ran a lab successfully? A PR adding a short `runs on <GPU> / <engine version>` note to its README
helps the next person know it's been exercised there.

## Other contributions, all welcome

- **Bug fixes** in the harness (`common/`) or the labs.
- **Content fixes** in the modules/appendices — typos, technical errors, clearer wording.
- **New or improved labs**, a missing *Going Deeper* section, an extra appendix.
- **Infra** — provider guides, cost-table updates, better scripts.

## Translations (翻译 / multilingual)

Translations are warmly welcome — **Chinese (中文) especially**, and any other language. You can
**translate** a module or appendix, **review/check** an existing translation for accuracy, or
**report** a translation error. Open a **Translation help** issue (the dropdown when you click
*New issue*) to claim a module so two people don't translate the same one.

Proposed layout: translated files live under `translations/<lang-code>/` mirroring the English file
names — e.g. `translations/zh/Module 00 — Numbers, Tokens, and Memory.md`. The English files are the
**source of truth**; translations track them.

Guidelines:
- **Translate the prose, not the machinery.** Keep code, file/identifier names, math, and the
  terms-of-art (KV cache, roofline, prefill/decode, TTFT/TPOT) in English — gloss them in the target
  language on first use if it helps.
- **Mirror the English structure** (one file per module, same `§N.x` numbering and headings) so
  readers can cross-reference and translations stay easy to keep in sync as the English source moves.
- A partial translation is fine — translate what you can and note where you stopped.

## Setup

**No GPU (harness + content):**
```bash
pip install -r common/requirements.txt
pytest common/tests        # 21 tests, ~5s — must stay green
```
**GPU (labs):** see [`infra/README.md`](infra/README.md).

## Standards

**Code (`common/`, `labs/`):**
- `pytest common/tests` must pass. If you change harness behavior, add or update a test.
- Respect the **seven invariants** (`common/README.md` → the invariant→test map) — they are the point:
  open-loop by default, latency from `intended_send_time`, warmup discarded, prefix caching tested on
  *both* long-doc-QA and ShareGPT, retrieval not perplexity, quality per task, agentic decomposed.
- Match the existing style: type hints, dataclasses, `async`/`await`; no new heavy dependencies.
- Use the harness — don't re-implement metrics or load generation.

**Course content (modules / appendices):**
- Keep the **spine**: techniques are *derived* from "autoregressive decode is memory-bandwidth-bound,"
  not listed as a bag of tricks.
- Match the module shape: italic preamble → `## In plain English` → numbered `§N.x` sections →
  `## Going Deeper` (where it earns it) → `## Lab N` → Reading.
- **Go deep on the durable** (physics, math, architecture); stay shallow on what rots (specific SOTA
  model names, framework APIs, leaderboard numbers).
- Be honest (the Module 12 ethos): no claim without a measurement, no speed number without a quality
  number.

## Workflow

- Fork → branch → pull request. Say what and why.
- **Do not force-push or rewrite `main`.** Its history is intentionally clean and public; only normal
  commits go forward.
- Keep commits focused with clear messages; one logical change per PR where you can.

## License

By contributing, you agree your contributions are licensed under this repository's **MIT** license
(see [`LICENSE`](LICENSE)).
