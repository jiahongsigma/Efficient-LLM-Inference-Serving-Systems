"""Synthetic text sizing.

The harness must size prompts/contexts in *tokens* without shipping a tokenizer
or a dataset. ``make_text(n)`` returns a string of ~``n`` tokens under the same
~4-chars/token rule the rest of the course uses (Module 0); an optional
``prefix`` lets a builder make the *leading* content unique (to defeat prefix
sharing) or shared (to force it).
"""

from __future__ import annotations


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def make_text(n_tokens: int, *, prefix: str = "") -> str:
    """A string whose ``approx_tokens`` ≈ ``n_tokens``.

    ``prefix`` is placed first (so it determines the leading-token identity used
    by a prefix cache) and counts toward the budget.
    """
    n_tokens = max(1, int(n_tokens))
    used = approx_tokens(prefix) if prefix else 0
    filler = max(1, n_tokens - used)
    body = "tok " * filler  # "tok " ≈ 1 token
    return (f"{prefix} {body}").strip() if prefix else body.strip()
