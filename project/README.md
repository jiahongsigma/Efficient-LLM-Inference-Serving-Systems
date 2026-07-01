# Capstone

The course ends here. The capstone is where the modules pay off: a strong project **derives its
result from the spine** (the roofline, the KV budget, the communication cost, the speculative-decoding
law) rather than just reporting a speedup. On **open models and public benchmarks only.**

Deliverable: a **reproducible repo** + a **short report** (use [`TEMPLATE.md`](TEMPLATE.md)). Graded on
the **analysis** — connecting measured numbers back to the spine — over the headline number
(see [`RUBRIC.md`](RUBRIC.md)).

## Pick an archetype

1. **Measurement study** — characterize a technique across models/workloads and *explain* it from
   first principles. *E.g.* quantify FlashAttention's IO savings against the roofline; map the
   long-context accuracy–memory frontier across eviction and KV-quant settings; measure TP-vs-PP
   scaling and attribute the cost to the collectives.
2. **Build-and-benchmark** — implement an optimization or serving feature and show, **with correct
   metrics on realistic traffic**, where it helps and where it does not. *E.g.* a KV-cache eviction
   policy; a draft-model selector for speculative decoding; a prefix-aware scheduling policy in the
   gateway (Lab 11).
3. **Reproduction + stress** — reproduce a paper's headline result on open models, then **find where
   it breaks** (a workload or context regime the paper didn't test). This trains the Module 12
   instinct directly.

## Requirements

- **Reproducible.** Use the `common/` harness for load generation and metrics; pin seeds; commit the
  exact commands (the `infra/serve_and_run.sh` invocations) and the model/engine/flags. Anyone should
  reproduce your headline figure from your repo.
- **Honest measurement (this is most of the grade).** Open-loop generation, warmup discarded,
  variance/CI over ≥3 runs, the metric the workload actually cares about (on the tail), and — for any
  technique that can trade quality — a **quality number beside every speed number** (Module 12).
- **Tied to the spine.** Your analysis must explain the result through the course's physics, not just
  present a curve. The best projects predict the number *before* measuring it, then explain the gap.
- **Open only.** Open-weight models (Llama-3.1-8B, Qwen2.5, Mistral, a Mixtral-class MoE) and publicly
  redistributable benchmark data.

## Scope

One rented GPU-day is plenty for a strong measurement study or build-and-benchmark (see `infra/`).
Depth of analysis beats breadth of sweeps — a single curve you fully explain from the spine is worth
more than ten you don't.

## Submit

A repo containing: your code/config, the filled [`TEMPLATE.md`](TEMPLATE.md) report (~4–8 pages),
the raw result CSVs, and a one-command reproduce script. Self-assess against [`RUBRIC.md`](RUBRIC.md)
before you call it done.
