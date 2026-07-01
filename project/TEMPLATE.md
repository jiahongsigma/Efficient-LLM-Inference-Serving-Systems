# <Project title>

*Author · date · archetype (measurement study / build-and-benchmark / reproduction+stress)*

## 1. Question
One or two sentences. What are you measuring or building, and why does it matter for serving?

## 2. Spine prediction
**Before measuring**, what does the course's physics predict? Cite the relevant law (decode latency
≳ `weight_bytes/B`; KV size law; ridge point `I*=P/B`; communication cost `bytes/link-bw`; expected
speedup `(1−α^{k+1})/((1−α)(1+kc))`; …) and write down the number you expect.

## 3. Setup
- **Model(s):** open-weight id(s) + dtype.
- **Hardware:** GPU(s), VRAM, interconnect (from `nvidia-smi` / the provider).
- **Engine:** vLLM / SGLang version + the exact launch flags.
- **Reproduce:** the `infra/serve_and_run.sh` command(s).

## 4. Method
- **Workload:** which `common.traffic` builder, lengths, arrival pattern, prefix-sharing ratio.
- **Metrics:** which `common.bench`/`common.eval` quantities, and **why those** for this workload.
- **Protocol:** open-loop; warmup discarded; N≥3 repeats; seeds. Note anything non-default.

## 5. Results
Tables and figures (the CSVs your `run_lab.py` emitted). Report on the **tail**; show mean ± CI.
Pair every speed number with a quality number where the technique can trade quality.

## 6. Analysis — back to the spine
The heart of the grade. Explain the numbers through the physics. **Name the binding resource**
(compute / KV-bandwidth / KV-capacity / interconnect). Reconcile measurement with your §2 prediction
and explain the gap (allocator overhead, kernel efficiency, batch regime, …).

## 7. Threats to validity
The Module 12 honesty pass on *your own* result: is the workload representative? Is the benchmark
saturated/contaminated? Could a defensible different choice flip the conclusion? Did you avoid
coordinated omission?

## 8. Reproduce
The exact commands to regenerate every figure from a clean clone.

## References
Papers + the course modules you relied on.
