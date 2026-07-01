"""Structured-output adherence (Module 9), shared with tool-calling (Module 10).

Reports the fraction of outputs that validate against a JSON schema and the
parse-failure rate. Note (Module 9 §9.5): adherence is NOT quality — a 100%-valid
JSON can still be semantically wrong. The Module 9 lab pairs this with
``score_suite`` on a reasoning task to show the two axes diverge.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

try:
    import jsonschema

    _HAVE_JSONSCHEMA = True
except Exception:  # pragma: no cover
    _HAVE_JSONSCHEMA = False


@dataclass
class SchemaScore:
    valid_fraction: float  # validates against the schema
    parse_failure_rate: float  # does not even parse as JSON
    n: int
    n_valid: int
    n_parse_fail: int


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str):
    """Best-effort: a fenced block, else the first balanced {...} or [...]."""
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i, j = text.find(open_c), text.rfind(close_c)
        if 0 <= i < j:
            try:
                return json.loads(text[i : j + 1])
            except Exception:
                continue
    return None


def _validate(obj, schema) -> bool:
    if _HAVE_JSONSCHEMA:
        try:
            jsonschema.validate(obj, schema)
            return True
        except Exception:
            return False
    return _minimal_validate(obj, schema)


def _minimal_validate(obj, schema) -> bool:
    """Tiny fallback validator (type/properties/required/enum) if jsonschema is
    unavailable — enough to keep the harness honest without the dependency."""
    t = schema.get("type")
    types = {"object": dict, "array": list, "string": str, "number": (int, float),
             "integer": int, "boolean": bool}
    if t and t in types and not isinstance(obj, types[t]):
        return False
    if "enum" in schema and obj not in schema["enum"]:
        return False
    if t == "object":
        for key in schema.get("required", []):
            if key not in obj:
                return False
        for key, sub in schema.get("properties", {}).items():
            if key in obj and not _minimal_validate(obj[key], sub):
                return False
    if t == "array" and "items" in schema:
        return all(_minimal_validate(x, schema["items"]) for x in obj)
    return True


def score_json_schema(results, schema) -> SchemaScore:
    n = n_valid = n_parse_fail = 0
    for r in results:
        if not getattr(r, "ok", True):
            continue
        n += 1
        obj = extract_json(r.output_text)
        if obj is None:
            n_parse_fail += 1
            continue
        if _validate(obj, schema):
            n_valid += 1
    return SchemaScore(
        valid_fraction=(n_valid / n) if n else 0.0,
        parse_failure_rate=(n_parse_fail / n) if n else 0.0,
        n=n, n_valid=n_valid, n_parse_fail=n_parse_fail,
    )
