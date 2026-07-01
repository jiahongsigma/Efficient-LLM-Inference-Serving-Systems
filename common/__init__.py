"""``common`` — the shared lab harness for *Efficient LLM Inference & Serving Systems*.

Three sub-packages, imported once and reused by every module lab:

* ``common.bench``  — load generation (open/closed/agentic), the streaming
  client (real ``OpenAIEndpoint`` + offline ``SimEndpoint``), and metrics.
* ``common.traffic`` — workload builders (ShareGPT, long-doc-QA, needle,
  agentic sessions) + arrival schedules.
* ``common.eval``    — per-task accuracy, needle retrieval, JSON-schema
  adherence, and deltas.

The seven invariants listed in this package's ``README.md`` are the acceptance
criteria; the test suite in ``common/tests/`` checks them against ``SimEndpoint``
with no GPU.
"""

from . import bench, eval, mem, traffic
from .mem import MODELS, kv_budget, mem_estimate

__all__ = ["bench", "traffic", "eval", "mem", "mem_estimate", "kv_budget", "MODELS"]
