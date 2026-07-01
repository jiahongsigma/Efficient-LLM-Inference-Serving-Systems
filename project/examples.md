# Worked capstone outlines

Two example skeletons — not solutions, just the shape a strong submission takes. Each fits one
GPU-day and derives its result from the spine.

## A. Measurement study — "Where does INT4 actually pay?"

*Spine prediction (§2):* weight-only INT4 helps decode (memory-bound: 4× fewer weight-bytes) but
~nothing for prefill (compute-bound), and the decode advantage should **erode at high batch** as
arithmetic intensity rises past the ridge (Module 1/3). Predict the batch where it flips from the
roofline.

*Method:* one 24–48 GB box. Serve FP16 and INT4-AWQ (relaunch per config). Use Module 1's phase
isolation (long-prompt/1-token = prefill; short-prompt/long-output = decode) and sweep decode batch
{1, 4, 16, 64}. Build on `labs/m03_quantization`.

*Result/analysis:* the prefill/decode/batch table; place each speedup on the roofline; report the
measured flip-point and explain the gap to your prediction. Pair with per-task accuracy
(`score_suite`, never averaged) so "faster" isn't just "worse."

*Why it scores:* a clean spine prediction, the honest batch-regime caveat, quality beside speed.

## B. Build-and-benchmark — "A prefix-aware router for the gateway"

*Spine prediction (§2):* on shared-prefix traffic, routing a request to the replica that already
holds its prefix (Modules 5, 7) turns N re-prefills into 1 — predict the TTFT win as a function of
the prefix-sharing ratio.

*Method:* extend `labs/m11_gateway`'s gateway with a prefix-hash → replica routing table; two
backends. Drive `common.traffic.build_long_doc_qa` at swept `prefix_share_ratio` (and ShareGPT as the
null control — INVARIANT 4). Open-loop; measure TTFT tail and `cache_hit_rate` round-robin vs
prefix-aware.

*Result/analysis:* the TTFT-and-hit-rate gain vs sharing ratio, beside the ShareGPT null. Explain
where it helps and where it's pointless; name the regime.

*Why it scores:* a real feature, correct metrics on *both* the positive and null workloads,
reproducible via the harness.
