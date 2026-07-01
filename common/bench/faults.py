"""Fault injection for the resilience lab (Module 11).

A ``FaultSpec`` is scheduled by ``run_open_loop`` and applied at ``at_time`` via
``inject``. Against a real deployment the lab kills a backend process directly;
against ``SimEndpoint`` the fault is applied in-process so the whole failover
story is testable offline.

INVARIANT 1 corollary: a request in flight on a killed backend gets
``status="error"`` and still counts toward the tail — it must NOT be silently
dropped. ``SimEndpoint`` enforces this by raising ``FaultError`` mid-stream.
"""

from __future__ import annotations

from dataclasses import dataclass


class FaultError(RuntimeError):
    """Raised by a simulated backend that was killed while serving a request."""


@dataclass
class FaultSpec:
    at_time: float  # seconds into the run
    kind: str  # "kill_backend" | "slow_backend"
    target: str  # backend id
    severity: float = 0.0  # for "slow_backend": added latency (seconds)


async def inject(fault: FaultSpec, endpoint) -> None:
    """Apply a fault to an endpoint that supports it.

    ``run_open_loop`` schedules this at ``fault.at_time``. Endpoints that cannot
    self-injure (e.g. a real HTTP endpoint) are expected to be faulted out of
    band by the lab; we surface that rather than failing silently.
    """
    apply = getattr(endpoint, "apply_fault", None)
    if apply is None:
        raise NotImplementedError(
            f"{type(endpoint).__name__} cannot self-inject {fault.kind!r}; "
            "kill/slow the backend out of band (e.g. `kill` the engine process)."
        )
    apply(fault)
