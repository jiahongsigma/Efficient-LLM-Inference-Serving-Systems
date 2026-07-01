"""Quality scorers — quality and performance are one measurement (Module 12 §12.6)."""

from .deltas import quant_delta
from .needle import NeedleScore, score_needle
from .schema import SchemaScore, extract_json, score_json_schema
from .tasks import TaskScore, register_task, score_suite, score_task

__all__ = [
    "score_task",
    "score_suite",
    "register_task",
    "TaskScore",
    "score_needle",
    "NeedleScore",
    "score_json_schema",
    "extract_json",
    "SchemaScore",
    "quant_delta",
]
