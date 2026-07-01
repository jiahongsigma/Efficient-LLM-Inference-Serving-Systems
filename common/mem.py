"""Memory math — predict serve-time VRAM and the KV budget (Modules 0, 2).

Pure arithmetic: GPU-free, deterministic, and unit-tested. Labs 0/2/3/5/7 import
these instead of re-deriving the byte math. On the GPU box you *validate* the
predictions against the real allocator (Lab 0 step 3) — within ~15% is the goal.
"""

from __future__ import annotations

from dataclasses import dataclass

# bytes per stored parameter, by dtype (Module 0 §0.2)
BYTES_PER_PARAM = {
    "fp32": 4, "tf32": 4, "fp16": 2, "bf16": 2,
    "fp8": 1, "e4m3": 1, "e5m2": 1, "int8": 1,
    "int4": 0.5, "nf4": 0.5,
}


def bytes_per_param(dtype: str) -> float:
    try:
        return BYTES_PER_PARAM[dtype.lower()]
    except KeyError:
        raise ValueError(f"unknown dtype {dtype!r}; known: {sorted(BYTES_PER_PARAM)}")


def kv_bytes_per_token(*, n_layers: int, n_kv_heads: int, head_dim: int, kv_dtype: str = "fp16") -> float:
    """Module 2 §2.2: 2 (K and V) · layers · kv_heads · head_dim · dtype_bytes."""
    return 2 * n_layers * n_kv_heads * head_dim * bytes_per_param(kv_dtype)


@dataclass
class MemEstimate:
    weights: float  # bytes
    kv: float        # bytes
    overhead: float  # bytes

    @property
    def total(self) -> float:
        return self.weights + self.kv + self.overhead

    def as_dict(self) -> dict:
        return {"weights": self.weights, "kv": self.kv, "overhead": self.overhead, "total": self.total}

    def gb(self) -> dict:
        return {k: v / 1e9 for k, v in self.as_dict().items()}


def mem_estimate(
    cfg: dict,
    *,
    weight_dtype: str = "bf16",
    kv_dtype: str = "fp16",
    context_len: int = 4096,
    batch: int = 1,
    overhead_gb: float = 1.5,
) -> MemEstimate:
    """Predicted serve-time VRAM (bytes), split weights / KV / overhead (Module 0 §0.4).

    ``cfg`` keys: ``params_b`` (billions), ``n_layers``, ``n_kv_heads``, ``head_dim``.
    """
    weights = cfg["params_b"] * 1e9 * bytes_per_param(weight_dtype)
    kv = kv_bytes_per_token(
        n_layers=cfg["n_layers"], n_kv_heads=cfg["n_kv_heads"],
        head_dim=cfg["head_dim"], kv_dtype=kv_dtype,
    ) * context_len * batch
    return MemEstimate(weights=weights, kv=kv, overhead=overhead_gb * 1e9)


def kv_budget(
    cfg: dict,
    *,
    vram_gb: float,
    weight_dtype: str = "bf16",
    kv_dtype: str = "fp16",
    context_len: int = 4096,
    overhead_gb: float = 1.5,
) -> dict:
    """Max concurrency that fits (Module 0 §0.4):
    usable_KV = VRAM − weights − overhead; max_conc = usable_KV ÷ (kv/token · context)."""
    vram = vram_gb * 1e9
    weights = cfg["params_b"] * 1e9 * bytes_per_param(weight_dtype)
    usable = vram - weights - overhead_gb * 1e9
    per_seq = kv_bytes_per_token(
        n_layers=cfg["n_layers"], n_kv_heads=cfg["n_kv_heads"],
        head_dim=cfg["head_dim"], kv_dtype=kv_dtype,
    ) * context_len
    max_conc = max(0, int(usable // per_seq)) if per_seq > 0 else 0
    return {"usable_kv_bytes": usable, "kv_bytes_per_seq": per_seq, "max_concurrency": max_conc}


# A few stock configs so the labs can run out of the box (verify head_dim/kv_heads
# against the real model card before trusting a number).
MODELS = {
    "llama-3.1-8b": {"params_b": 8.03, "n_layers": 32, "n_kv_heads": 8, "head_dim": 128},
    "llama-2-7b":   {"params_b": 6.74, "n_layers": 32, "n_kv_heads": 32, "head_dim": 128},
    "qwen2.5-14b":  {"params_b": 14.7, "n_layers": 48, "n_kv_heads": 8, "head_dim": 128},
    "mistral-7b":   {"params_b": 7.24, "n_layers": 32, "n_kv_heads": 8, "head_dim": 128},
}
