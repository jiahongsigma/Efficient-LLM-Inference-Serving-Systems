# Capstone rubric

Grade on the **analysis**, not the headline number. A modest speedup *explained from the spine*
beats a big one that's just reported. Total 100.

| # | Criterion | Weight | What "excellent" looks like |
|---|---|---:|---|
| 1 | **Connection to the spine** | 30 | The result is *derived*, not just shown. You predict the number from the roofline / KV budget / communication cost / speculative-decoding law, measure it, and **explain the gap** between prediction and reality. |
| 2 | **Measurement honesty (Module 12)** | 30 | Open-loop, fixed-schedule generation; warmup discarded; **variance/CI over ≥3 runs**; the metric the *workload* cares about, reported on the **tail** (p95/p99), not the average; quality paired with every speed claim on an un-saturated eval. No coordinated omission. |
| 3 | **Reproducibility** | 20 | The `common/` harness is used (not a hand-rolled loop); seeds pinned; exact model/engine/flags and the `serve_and_run.sh` commands committed; a reader reproduces the headline figure from your repo. |
| 4 | **Analysis & clarity** | 15 | The report explains *why*, names the binding resource (compute / KV-bandwidth / KV-capacity / interconnect), and states threats to validity honestly. Figures are correct and legible. |
| 5 | **Scope & difficulty** | 5 | The question is non-trivial and the depth is appropriate for one GPU-day. |

## Automatic deductions (the Module 12 traps)

- Average-only or throughput-only reporting where the workload cares about tail latency: **−10**.
- A speed result for a quality-trading technique (quant / eviction / speculation) with **no quality number**: **−15**.
- Closed-loop latency presented as if it were the true tail (coordinated omission): **−10**.
- A single run, no variance/CI: **−10**.
- Prefix-caching / long-context / quant claims measured on the *wrong* workload or a saturated benchmark: **−10**.
- Not reproducible from the repo (missing seeds/commands/configs): **−15**.

## Self-check before submitting

- [ ] My headline figure is reproducible from a clean clone with one command.
- [ ] Every speed number has the metric defined and is on the tail; quality-trading results have a quality number beside them.
- [ ] I predicted the result from the spine first, then explained the measured gap.
- [ ] I named the binding resource and stated at least one threat to validity.
- [ ] Open models + public data only.
