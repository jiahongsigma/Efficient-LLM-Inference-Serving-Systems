"""Acceptance tests for the `common/` harness — one per invariant, plus the
key behaviours each module lab relies on. All run against `SimEndpoint`, offline,
in well under a second total. Run with `pytest common/tests` or
`python -m common.tests.test_harness`.
"""

from __future__ import annotations

import asyncio
import json

from common.bench import (
    FaultSpec,
    Request,
    Result,
    SimEndpoint,
    StaticBatchEndpoint,
    SLO,
    aggregate_runs,
    compute_metrics,
    compute_session_metrics,
    determinism_check,
    run_agentic_session,
    run_closed_loop,
    run_open_loop,
)
from common.eval import (
    quant_delta,
    score_json_schema,
    score_needle,
    score_suite,
    score_task,
)
from common.mem import kv_bytes_per_token, kv_budget, mem_estimate
from common.traffic import (
    build_agentic_sessions,
    build_long_doc_qa,
    build_needle,
    build_sharegpt,
)


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# INVARIANT 1 — coordinated-omission-correct latency.
# --------------------------------------------------------------------------- #
def test_inv1_latency_from_intended_not_actual():
    r = Result(
        id="x", intended_send_time=1.0, actual_send_time=3.0, first_token_time=3.5,
        end_time=5.0, prompt_tokens=10, completion_tokens=4,
    )
    assert r.e2e == 4.0  # 5.0 - 1.0  (NOT 5.0 - 3.0)
    assert r.ttft == 2.5  # 3.5 - 1.0  (NOT 3.5 - 3.0)


def test_inv1_queue_wait_counts_toward_latency():
    # one server, many requests scheduled almost together -> a backlog forms,
    # and the wait must show up in e2e (measured from intended).
    ep = SimEndpoint(max_concurrency=1, decode_ms_per_tok=1.0, gen_token_cap=8)
    reqs = [Request(id=f"r{i}", messages=[{"role": "user", "content": f"u{i} hello"}],
                    max_tokens=8, intended_send_time=0.0) for i in range(15)]
    results = _run(run_open_loop(reqs, ep))
    assert all(r.e2e == r.end_time - r.intended_send_time for r in results)
    # latest finisher waited behind the queue: its e2e exceeds a single service time
    service_times = [r.end_time - r.actual_send_time for r in results]
    assert max(r.e2e for r in results) > 2 * min(service_times)


# --------------------------------------------------------------------------- #
# INVARIANT 2 — open-loop reveals the knee; closed-loop hides it.
# --------------------------------------------------------------------------- #
def test_inv2_open_loop_reveals_knee_closed_hides():
    def fresh():
        return SimEndpoint(max_concurrency=2, decode_ms_per_tok=1.0, gen_token_cap=8)

    # offer load far above capacity, all arriving in a tight window (open-loop)
    reqs = [Request(id=f"o{i}", messages=[{"role": "user", "content": f"u{i}"}],
                    max_tokens=8, intended_send_time=i * 0.001) for i in range(40)]
    open_m = compute_metrics(_run(run_open_loop(reqs, fresh())))

    closed_reqs = [Request(id=f"c{i}", messages=[{"role": "user", "content": f"u{i}"}],
                           max_tokens=8) for i in range(40)]
    closed_m = compute_metrics(_run(run_closed_loop(closed_reqs, fresh(), concurrency=2)))

    # open-loop p99 includes real queueing; closed-loop self-throttles -> far lower
    assert open_m.e2e_p99 > 3 * closed_m.e2e_p99


# --------------------------------------------------------------------------- #
# INVARIANT 3 — warmup discarded; variance/CI over >=3 repeats.
# --------------------------------------------------------------------------- #
def test_inv3_warmup_discarded():
    results = []
    for i in range(20):
        big = i < 3  # first three are cold/slow
        results.append(Result(id=f"w{i}", intended_send_time=float(i),
                              actual_send_time=float(i), first_token_time=float(i) + 0.01,
                              end_time=float(i) + (5.0 if big else 0.05),
                              prompt_tokens=10, completion_tokens=5))
    cold = compute_metrics(results, warmup=0)
    warm = compute_metrics(results, warmup=3)
    assert warm.e2e_p99 < cold.e2e_p99
    assert warm.n == 17


def test_inv3_aggregate_runs_reports_ci():
    ep = SimEndpoint(gen_token_cap=8)
    runs = []
    for s in range(3):
        reqs = build_sharegpt(20, rate_or_trace=200.0, length_profile="short", seed=s)
        runs.append(compute_metrics(_run(run_open_loop(reqs, ep)), warmup=2))
    agg = aggregate_runs(runs)
    assert agg.n_runs == 3
    assert "e2e_p99" in agg.mean and "e2e_p99" in agg.ci95
    assert agg.ci95["e2e_p99"] >= 0.0


# --------------------------------------------------------------------------- #
# INVARIANT 4 — prefix caching tested on long-doc-QA AND ShareGPT.
# --------------------------------------------------------------------------- #
def test_inv4_prefix_cache_helps_only_with_shared_prefix():
    ep = SimEndpoint(gen_token_cap=8)
    ldq = build_long_doc_qa(20, doc_tokens=400, prefix_share_ratio=1.0, rate_or_trace=200.0, seed=1)
    ldq_m = compute_metrics(_run(run_open_loop(ldq, ep)))

    ep2 = SimEndpoint(gen_token_cap=8)
    sg = build_sharegpt(20, rate_or_trace=200.0, length_profile="short", seed=1)
    sg_m = compute_metrics(_run(run_open_loop(sg, ep2)))

    assert ldq_m.cache_hit_rate > 0.3   # shared doc -> heavy reuse after the first
    assert sg_m.cache_hit_rate < 0.05   # unique prompts -> ~no reuse (the null control)


# --------------------------------------------------------------------------- #
# INVARIANT 5 — long-context accuracy is needle retrieval, not perplexity.
# --------------------------------------------------------------------------- #
def test_inv5_needle_scored_by_depth():
    # simulate eviction: the needle is found at shallow depth, lost deep.
    results = []
    for i in range(20):
        deep = i % 2 == 0
        depth = 0.9 if deep else 0.1
        secret = str(1000 + i)
        out = "nope" if deep else secret  # deep needle evicted
        results.append(Result(id=f"n{i}", intended_send_time=0, actual_send_time=0,
                              first_token_time=0.01, end_time=0.02, prompt_tokens=100,
                              completion_tokens=1, output_text=out,
                              meta={"expected": secret, "needle_depth": depth}))
    score = score_needle(results)
    assert score.by_depth["0.0-0.2"] == 1.0   # shallow retrieved
    assert score.by_depth["0.8-1.0"] == 0.0   # deep lost
    assert 0.0 < score.retrieval_rate < 1.0


def test_inv5_builder_plants_needle():
    reqs = build_needle(5, context_len=500, needle_depth_fraction=0.5, seed=0)
    for r in reqs:
        assert r.meta["expected"] in r.messages[0]["content"]
        assert r.meta["needle_depth"] == 0.5


# --------------------------------------------------------------------------- #
# INVARIANT 6 — quality per task, never one mean; saturation flagged.
# --------------------------------------------------------------------------- #
def test_inv6_per_task_and_saturation_and_delta():
    def mk(task, correct, expected, out, i):
        return Result(id=f"{task}{i}", intended_send_time=0, actual_send_time=0,
                     first_token_time=0.01, end_time=0.02, prompt_tokens=10,
                     completion_tokens=2, output_text=out,
                     meta={"task": task, "expected": expected})

    # mmlu fully correct (saturated); gsm8k half correct
    base = []
    for i in range(10):
        base.append(mk("mmlu", True, "B", "The answer is B", i))
    for i in range(10):
        base.append(mk("gsm8k", i < 5, "42", "42" if i < 5 else "7", i))
    suite = score_suite(base)
    assert set(suite) == {"mmlu", "gsm8k"}          # a dict per task, not a mean
    assert suite["mmlu"].saturated is True
    assert suite["gsm8k"].saturated is False
    assert abs(suite["gsm8k"].score - 0.5) < 1e-9

    # a "quantized" candidate that regresses only gsm8k
    cand = []
    for i in range(10):
        cand.append(mk("mmlu", True, "B", "The answer is B", i))
    for i in range(10):
        cand.append(mk("gsm8k", i < 2, "42", "42" if i < 2 else "9", i))
    delta = quant_delta(suite, score_suite(cand))
    assert abs(delta["mmlu"]) < 1e-9 and delta["gsm8k"] < 0  # regression localized to gsm8k


# --------------------------------------------------------------------------- #
# INVARIANT 7 — agentic measured end-to-end and decomposed, with cache-hit rate.
# --------------------------------------------------------------------------- #
def test_inv7_agentic_decomposition_and_cache_fragility():
    ep = SimEndpoint(gen_token_cap=8, prefill_ms_per_tok=0.05)

    stable = build_agentic_sessions(4, turns=6, increment_tokens=120, system_tokens=200,
                                    tool_latency=0.01, prefix_stable=True, seed=0)
    pert = build_agentic_sessions(4, turns=6, increment_tokens=120, system_tokens=200,
                                  tool_latency=0.01, prefix_stable=False, seed=0)

    stable_sr = [_run(run_agentic_session(s, ep, kv_policy="hold")) for s in stable]
    ep2 = SimEndpoint(gen_token_cap=8, prefill_ms_per_tok=0.05)
    pert_sr = [_run(run_agentic_session(s, ep2, kv_policy="hold")) for s in pert]

    sm_stable = compute_session_metrics(stable_sr)
    sm_pert = compute_session_metrics(pert_sr)

    # decomposed breakdown present and normalized (model/tool/reprefill)
    bd = sm_stable.latency_breakdown
    assert set(bd) == {"model", "tool", "reprefill"}
    assert abs(sum(bd.values()) - 1.0) < 1e-6

    # stable prefixes hit the cross-turn cache; perturbed ones silently miss
    assert sm_stable.cache_hit_rate > 0.4
    assert sm_pert.cache_hit_rate < 0.05
    # silent cost blowup: perturbed re-prefills far more tokens (O(T^2) vs O(T))
    assert sm_pert.reprefill_tokens_total > 2 * sm_stable.reprefill_tokens_total


def test_inv7_kv_policy_offload_frees_the_slot():
    ep = SimEndpoint(gen_token_cap=8)
    sess = build_agentic_sessions(1, turns=5, increment_tokens=100, system_tokens=100,
                                  tool_latency=0.02, prefix_stable=True, seed=0)[0]
    hold = _run(run_agentic_session(sess, ep, kv_policy="hold"))
    ep2 = SimEndpoint(gen_token_cap=8)
    off = _run(run_agentic_session(sess, ep2, kv_policy="offload"))
    # holding KV through the tool wait inflates slot-held time; offload frees it
    assert off.slot_held_time < hold.slot_held_time
    assert compute_session_metrics([off]).slot_utilization > compute_session_metrics([hold]).slot_utilization


# --------------------------------------------------------------------------- #
# Module 9 — structured-output adherence (syntax, not quality).
# --------------------------------------------------------------------------- #
def test_schema_adherence_and_parse_failures():
    schema = {"type": "object", "required": ["name", "age"],
              "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}

    def mk(out, i):
        return Result(id=f"s{i}", intended_send_time=0, actual_send_time=0,
                     first_token_time=0.01, end_time=0.02, prompt_tokens=10,
                     completion_tokens=5, output_text=out)

    valid = [mk('{"name": "a", "age": 3}', i) for i in range(8)]
    invalid_schema = [mk('{"name": "a"}', i) for i in range(1)]      # missing required
    not_json = [mk("here you go: name=a", i) for i in range(1)]      # not parseable
    score = score_json_schema(valid + invalid_schema + not_json, schema)
    assert score.n == 10
    assert abs(score.valid_fraction - 0.8) < 1e-9
    assert abs(score.parse_failure_rate - 0.1) < 1e-9


# --------------------------------------------------------------------------- #
# Module 12 — determinism check.
# --------------------------------------------------------------------------- #
def test_determinism_check_detects_batch_nondeterminism():
    nd = SimEndpoint(nondeterministic=True, gen_token_cap=8)
    rep = _run(determinism_check(nd, "hello", n_runs=6, vary_batch=True))
    assert rep.identical_fraction < 1.0  # batch composition changed the output

    det = SimEndpoint(nondeterministic=False, gen_token_cap=8)
    rep2 = _run(determinism_check(det, "hello", n_runs=6, vary_batch=True))
    assert rep2.identical_fraction == 1.0


# --------------------------------------------------------------------------- #
# Module 11 — fault injection: errors count toward the tail, never dropped.
# --------------------------------------------------------------------------- #
def test_fault_injection_errors_are_counted():
    ep = SimEndpoint(max_concurrency=4, decode_ms_per_tok=0.5, gen_token_cap=8,
                     backends=("only",))
    reqs = [Request(id=f"f{i}", messages=[{"role": "user", "content": f"u{i}"}],
                    max_tokens=8, intended_send_time=i * 0.01) for i in range(30)]
    fault = FaultSpec(at_time=0.05, kind="kill_backend", target="only")
    results = _run(run_open_loop(reqs, ep, fault=fault))
    assert len(results) == 30                 # nothing silently dropped
    assert any(not r.ok for r in results)     # post-kill requests errored
    m = compute_metrics(results)
    assert m.n_error > 0 and m.n == 30        # errors counted in the run


def test_fault_failover_keeps_serving():
    ep = SimEndpoint(max_concurrency=4, decode_ms_per_tok=0.5, gen_token_cap=8,
                     backends=("a", "b"))
    reqs = [Request(id=f"g{i}", messages=[{"role": "user", "content": f"u{i}"}],
                    max_tokens=8, intended_send_time=i * 0.01) for i in range(30)]
    fault = FaultSpec(at_time=0.05, kind="kill_backend", target="a")
    results = _run(run_open_loop(reqs, ep, fault=fault))
    # one backend dies but the other keeps serving -> most requests still succeed
    assert sum(1 for r in results if r.ok) >= 15


# --------------------------------------------------------------------------- #
# goodput honours the SLO.
# --------------------------------------------------------------------------- #
def test_goodput_respects_slo():
    ep = SimEndpoint(max_concurrency=2, decode_ms_per_tok=1.0, gen_token_cap=8)
    reqs = [Request(id=f"q{i}", messages=[{"role": "user", "content": f"u{i}"}],
                    max_tokens=8, intended_send_time=i * 0.001) for i in range(40)]
    results = _run(run_open_loop(reqs, ep))
    tight = compute_metrics(results, slo=SLO(e2e=0.01))
    loose = compute_metrics(results, slo=SLO(e2e=10.0))
    assert tight.goodput_req_s < loose.goodput_req_s  # tighter SLO -> fewer good


# --------------------------------------------------------------------------- #
# Module 4 — static batching collapses where continuous batching holds.
# --------------------------------------------------------------------------- #
def test_static_batching_tail_worse_than_continuous():
    # mostly-short requests with a few long ones, all arriving together
    def reqs():
        return [Request(id=f"r{i}", messages=[{"role": "user", "content": "p"}],
                        max_tokens=(64 if i % 4 == 0 else 4), intended_send_time=0.0)
                for i in range(16)]

    cont = SimEndpoint(max_concurrency=16, decode_ms_per_tok=1.0)
    cont_m = compute_metrics(_run(run_open_loop(reqs(), cont)))
    stat = StaticBatchEndpoint(batch_size=16, decode_ms_per_tok=1.0)
    stat_m = compute_metrics(_run(run_open_loop(reqs(), stat)))

    # head-of-line blocking: every short request waits for the batch's long member,
    # so static's median latency is inflated; continuous serves the shorts fast.
    assert stat_m.e2e_p50 > 3 * cont_m.e2e_p50


# --------------------------------------------------------------------------- #
# Module 4 — chunked prefill flattens the decode TPOT spike from a big prefill.
# --------------------------------------------------------------------------- #
def test_chunked_prefill_flattens_decode_tpot():
    def workload():
        decoders = [Request(id=f"d{i}", messages=[{"role": "user", "content": "short prompt"}],
                           max_tokens=24, intended_send_time=0.0, meta={"role": "decoder"})
                    for i in range(8)]
        big = Request(id="big", messages=[{"role": "user", "content": "x " * 1000}],
                      max_tokens=2, intended_send_time=0.004, meta={"role": "prefill"})
        return decoders + [big]

    def decoder_tpot(chunked):
        ep = SimEndpoint(max_concurrency=16, decode_ms_per_tok=1.0, prefill_ms_per_tok=0.1,
                         base_ms=0.5, model_prefill_interference=True, chunked_prefill=chunked,
                         prefill_chunk_tokens=256, enable_prefix_cache=False)
        res = _run(run_open_loop(workload(), ep))
        decoders = [r for r in res if r.meta.get("role") == "decoder"]
        return compute_metrics(decoders).tpot_p99

    off = decoder_tpot(chunked=False)
    on = decoder_tpot(chunked=True)
    assert off > 2 * on  # a non-chunked big prefill stalls every in-flight decode mid-stream


# --------------------------------------------------------------------------- #
# Module 0 / 2 — the memory math matches the modules' worked examples.
# --------------------------------------------------------------------------- #
def test_mem_estimate_matches_module_examples():
    cfg = {"params_b": 8.03, "n_layers": 32, "n_kv_heads": 8, "head_dim": 128}  # Llama-3.1-8B
    # weights in bf16 ≈ 16 GB (Module 0 §0.3)
    e = mem_estimate(cfg, weight_dtype="bf16", kv_dtype="fp16", context_len=8192, batch=1)
    assert abs(e.weights - 16.06e9) < 0.1e9
    # KV per token = 2·32·8·128·2 = 131072 bytes (Module 2 §2.2 worked example)
    assert kv_bytes_per_token(n_layers=32, n_kv_heads=8, head_dim=128, kv_dtype="fp16") == 131072
    # at 8K context, batch 1 → ~1 GB per sequence
    assert abs(e.kv - 1.07e9) < 0.05e9
    # INT4 quarters the bf16 weight bytes
    assert abs(mem_estimate(cfg, weight_dtype="int4").weights - e.weights / 4) < 1.0
    # FP8 KV halves the KV term
    assert abs(mem_estimate(cfg, kv_dtype="fp8", context_len=8192).kv - e.kv / 2) < 1.0


def test_kv_budget_concurrency():
    cfg = {"params_b": 8.03, "n_layers": 32, "n_kv_heads": 8, "head_dim": 128}
    b = kv_budget(cfg, vram_gb=24, weight_dtype="bf16", context_len=8192)
    # 24 - 16.06 - 1.5 = ~6.4 GB usable; ~1.07 GB/seq -> a handful of sequences
    assert b["max_concurrency"] == 5 or b["max_concurrency"] == 6
    # a bigger card fits many more
    assert kv_budget(cfg, vram_gb=80, context_len=8192)["max_concurrency"] > b["max_concurrency"] * 5


# --------------------------------------------------------------------------- #
# The REAL OpenAIEndpoint HTTP/SSE path — parsed against a mocked vLLM stream.
# (Every lab depends on this; without it, the only coverage is SimEndpoint.)
# --------------------------------------------------------------------------- #
def test_openai_endpoint_parses_vllm_stream():
    import httpx

    from common.bench.client import OpenAIEndpoint, stream_chat

    sse = (
        'data: {"choices":[{"delta":{"role":"assistant"},"index":0}]}\n\n'
        'data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n'
        'data: {"choices":[{"delta":{"content":" world"},"index":0}]}\n\n'
        'data: {"choices":[{"delta":{},"index":0,"finish_reason":"stop"}]}\n\n'
        'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2,'
        '"prompt_tokens_details":{"cached_tokens":4}}}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request):
        # sanity: the client must POST the OpenAI chat schema with stream=true
        body = json.loads(request.content)
        assert body["stream"] is True and body["model"] == "test-model"
        return httpx.Response(200, content=sse.encode(), headers={"content-type": "text/event-stream"})

    ep = OpenAIEndpoint("http://mock", "test-model", transport=httpx.MockTransport(handler))
    res = _run(stream_chat(ep, Request(id="x", messages=[{"role": "user", "content": "hi"}], max_tokens=8)))

    assert res.status == "ok"
    assert res.output_text == "Hello world"      # token deltas concatenated
    assert res.completion_tokens == 2
    assert res.prompt_tokens == 10               # from the usage event
    assert res.cached_prefix_tokens == 4         # prompt_tokens_details.cached_tokens
    assert res.first_token_time is not None       # TTFT was measured
    assert res.tpot is not None                  # inter-token latency computable


if __name__ == "__main__":
    import inspect
    import sys

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and inspect.isfunction(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
