"""Per-task accuracy scorers — reported per task, NEVER averaged (INVARIANT 6).

``score_suite`` returns a dict keyed by task and flags saturation, because an
easy/saturated benchmark hides exactly the regression a quantization or eviction
change introduces (Module 3 §3.5). The classic suite is pluggable with harder
discriminators (MMLU-Pro, GPQA, LiveCodeBench).
"""

from __future__ import annotations

import re
import signal
from collections import defaultdict
from dataclasses import dataclass

from ..bench.schema import Result

SATURATION_THRESHOLD = 0.95


@dataclass
class TaskScore:
    task: str
    score: float
    n: int
    n_correct: int
    saturated: bool


# --- per-example checkers --------------------------------------------------- #
def _check_mmlu(out: str, expected: str) -> bool:
    # standalone letter only (so the 'A' in "Answer" doesn't count); take the last
    matches = re.findall(r"\b([A-D])\b", out.upper())
    pred = matches[-1] if matches else out.strip()[:1].upper()
    return pred == str(expected).strip().upper()[:1]


def _last_number(s: str):
    nums = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    return nums[-1] if nums else None


def _check_gsm8k(out: str, expected: str) -> bool:
    pred = _last_number(out)
    gold = _last_number(str(expected))
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return False


def _check_ifeval(out: str, instructions: list[dict]) -> bool:
    """A small instruction-following checker. Each instruction is a dict like
    {"type": "max_words", "n": 50} / {"type": "contains", "text": "foo"} /
    {"type": "json"} / {"type": "startswith", "text": "Answer:"}."""
    for ins in instructions or []:
        t = ins.get("type")
        if t == "max_words" and len(out.split()) > ins["n"]:
            return False
        if t == "min_words" and len(out.split()) < ins["n"]:
            return False
        if t == "contains" and ins["text"] not in out:
            return False
        if t == "startswith" and not out.lstrip().startswith(ins["text"]):
            return False
        if t == "json":
            import json

            try:
                json.loads(out)
            except Exception:
                return False
    return True


class _Timeout(Exception):
    pass


def _check_humaneval(out: str, meta: dict) -> bool:
    """pass@1 by executing the completion + the provided test, guarded by a
    wall-clock alarm. Used for our own trusted fixtures; do not run untrusted
    completions in-process in production."""
    code = meta.get("prompt_code", "") + out
    test = meta.get("test", "")
    entry = meta.get("entry_point", "")
    program = f"{code}\n{test}\ncheck({entry})\n"

    def _handler(signum, frame):
        raise _Timeout()

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(meta.get("timeout", 3))
    try:
        ns: dict = {}
        exec(program, ns)  # noqa: S102 — trusted fixtures only
        return True
    except Exception:
        return False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


_CHECKERS = {
    "mmlu": lambda r: _check_mmlu(r.output_text, r.meta.get("expected", "")),
    "gsm8k": lambda r: _check_gsm8k(r.output_text, r.meta.get("expected", "")),
    "ifeval": lambda r: _check_ifeval(r.output_text, r.meta.get("instructions", [])),
    "humaneval": lambda r: _check_humaneval(r.output_text, r.meta),
}


def register_task(name: str, checker) -> None:
    """Plug in a harder discriminator (MMLU-Pro, GPQA, LiveCodeBench, ...)."""
    _CHECKERS[name] = checker


def score_task(results: list[Result], task: str) -> TaskScore:
    if task not in _CHECKERS:
        raise KeyError(f"no checker for task {task!r}; register_task() it first")
    check = _CHECKERS[task]
    scored = [r for r in results if r.ok]
    n_correct = sum(1 for r in scored if check(r))
    n = len(scored)
    score = n_correct / n if n else 0.0
    return TaskScore(task=task, score=score, n=n, n_correct=n_correct,
                     saturated=score >= SATURATION_THRESHOLD)


def score_suite(results: list[Result]) -> dict[str, TaskScore]:
    """Group results by ``meta['task']`` and score each. Never collapses to one
    number (INVARIANT 6)."""
    by_task: dict[str, list[Result]] = defaultdict(list)
    for r in results:
        task = r.meta.get("task")
        if task in _CHECKERS:
            by_task[task].append(r)
    return {task: score_task(rs, task) for task, rs in by_task.items()}
