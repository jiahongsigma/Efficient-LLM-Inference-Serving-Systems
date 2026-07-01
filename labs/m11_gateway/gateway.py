"""Minimal, auditable OpenAI-compatible gateway (Module 11) — ~140 lines.

Routes /v1/chat/completions across backends with failover, capability parity,
metadata-only telemetry, and load shedding. Deliberately small: a gateway you can
read end-to-end is the safer choice.

!!! Not yet run against real backends here — try it on yours and see how it does. Built on the tested common/ harness. !!!

Run:
    GATEWAY_BACKENDS="vllm=http://localhost:8000,sglang=http://localhost:8001" \
    GATEWAY_MAX_INFLIGHT=64 uvicorn gateway:app --host 0.0.0.0 --port 8080

Backends may declare capabilities they LACK, comma-separated, after a '#':
    "vllm=http://localhost:8000#tools,  sglang=http://localhost:8001"
(here vLLM is declared unable to serve tool calls -> such requests are rejected, not degraded).
"""

from __future__ import annotations

import asyncio
import json
import os
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


def _parse_backends() -> list[dict]:
    out = []
    for spec in os.environ.get("GATEWAY_BACKENDS", "vllm=http://localhost:8000").split(","):
        spec = spec.strip()
        if not spec:
            continue
        name_url, _, lacks = spec.partition("#")
        name, _, url = name_url.partition("=")
        out.append({"id": name.strip(), "url": url.strip().rstrip("/"),
                    "lacks": {c.strip() for c in lacks.split(",") if c.strip()}, "alive": True})
    return out


BACKENDS = _parse_backends()
API_KEY = os.environ.get("GATEWAY_API_KEY")
MAX_INFLIGHT = int(os.environ.get("GATEWAY_MAX_INFLIGHT", "64"))
TELEMETRY = os.environ.get("GATEWAY_TELEMETRY", "telemetry.jsonl")

app = FastAPI()
_sema = asyncio.Semaphore(MAX_INFLIGHT)
_rr = {"i": 0}


def _live() -> list[dict]:
    return [b for b in BACKENDS if b["alive"]]


def _route(required_caps: set[str]) -> dict | None:
    live = [b for b in _live() if not (required_caps & b["lacks"])]
    if not live:
        return None
    b = live[_rr["i"] % len(live)]
    _rr["i"] += 1
    return b


def _log(rec: dict) -> None:
    rec["t"] = time.time()
    try:
        with open(TELEMETRY, "a") as f:
            f.write(json.dumps(rec) + "\n")  # metadata only — never the payload
    except Exception:
        pass


@app.get("/healthz")
async def healthz():
    return {"backends": [{"id": b["id"], "alive": b["alive"]} for b in BACKENDS]}


@app.post("/admin/kill")
async def admin_kill(backend: str):
    """Fault injection for the lab: mark a backend dead (or 'all'/'revive')."""
    for b in BACKENDS:
        if backend in (b["id"], "all"):
            b["alive"] = False
        if backend == "revive":
            b["alive"] = True
    return await healthz()


@app.post("/v1/chat/completions")
async def chat(req: Request):
    if API_KEY and req.headers.get("authorization") != f"Bearer {API_KEY}":
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if _sema.locked():  # load shedding: protect goodput of admitted requests
        return JSONResponse({"error": "overloaded"}, status_code=503)

    body = await req.body()
    payload = json.loads(body)
    required = set()
    if payload.get("tools") or payload.get("functions"):
        required.add("tools")
    if payload.get("response_format") or payload.get("guided_json"):
        required.add("json")
    stream = bool(payload.get("stream"))

    async with _sema:
        tried = []
        for _ in range(len(BACKENDS)):
            b = _route(required)
            if b is None:
                _log({"event": "reject", "reason": "no_capable_backend", "caps": list(required)})
                return JSONResponse({"error": f"no live backend with caps {sorted(required)}"}, status_code=503)
            if b["id"] in tried:
                continue
            tried.append(b["id"])
            t0 = time.time()
            try:
                if stream:
                    gen = await _open_stream(b, payload, t0)  # connects HERE -> failover works
                    return StreamingResponse(gen, media_type="text/event-stream")
                return await _proxy_json(b, payload, t0)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadError, httpx.HTTPStatusError):
                b["alive"] = False  # circuit-break this backend; fail over to the next
                _log({"event": "failover", "from": b["id"]})
                continue
        return JSONResponse({"error": "all backends failed"}, status_code=502)


async def _proxy_json(b, payload, t0):
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(f"{b['url']}/v1/chat/completions", json=payload)
        r.raise_for_status()
        obj = r.json()
    usage = obj.get("usage", {})
    _log({"event": "ok", "backend": b["id"], "latency": time.time() - t0,
          "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")})
    return JSONResponse(obj)


async def _open_stream(b, payload, t0):
    """Connect EAGERLY so a dead/erroring backend raises here (inside chat()'s try,
    enabling failover + circuit-break) instead of later inside the lazily-run
    response generator. Returns a primed async generator that streams the rest."""
    payload = {**payload, "stream_options": {"include_usage": True}}
    client = httpx.AsyncClient(timeout=120)
    stream_cm = client.stream("POST", f"{b['url']}/v1/chat/completions", json=payload)
    resp = await stream_cm.__aenter__()  # connection + response headers happen here
    try:
        resp.raise_for_status()
    except Exception:
        await stream_cm.__aexit__(None, None, None)
        await client.aclose()
        raise

    async def gen():
        first = None
        usage = {}
        try:
            async for line in resp.aiter_lines():
                if first is None and line.startswith("data:"):
                    first = time.time() - t0
                if line.startswith("data:") and '"usage"' in line:
                    try:
                        usage = json.loads(line[5:]).get("usage") or usage
                    except Exception:
                        pass
                yield (line + "\n").encode()
        finally:
            await stream_cm.__aexit__(None, None, None)
            await client.aclose()
            _log({"event": "ok", "backend": b["id"], "ttft": first, "latency": time.time() - t0,
                  "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")})

    return gen()
