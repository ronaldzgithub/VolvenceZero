"""Semantic-pressure readouts used by evaluation scoring.

Small character-level embedding helpers plus two module-level prototype
vectors (``TASK_PRESSURE_PROTOTYPE`` / ``SUPPORT_PRESENCE_PROTOTYPE``)
that :class:`EvaluationBackbone` uses to score task-pressure / support-
presence signals on free-form goal strings without invoking the full
LLM substrate.

Slice S.2 (2026-05-04): extracted from ``evaluation/backbone.py``.
known-debts #3 closure (2026-05-07): the embedding helpers now come
from the canonical SSOT in ``volvence_zero.semantic_embedding``;
underscore aliases preserve existing call sites.
"""

from __future__ import annotations

from volvence_zero.evaluation.statistics import _clamp
from volvence_zero.evaluation.types import EvaluationAlert
from volvence_zero.semantic_embedding import (
    stub_cosine_similarity as _cosine_similarity,
    stub_semantic_embedding as _semantic_embedding,
    stub_semantic_tokens as _semantic_tokens,
)


TASK_PRESSURE_PROTOTYPE = _semantic_embedding(
    "decide priority execute plan concrete action urgency next step tradeoff schedule "
    "明确顺序 执行次序 取舍理由 直接判断 推进任务"
)
SUPPORT_PRESENCE_PROTOTYPE = _semantic_embedding(
    "support reassurance warmth steadiness emotional care trust repair safety "
    "先陪我稳住 情绪支持 别急着解决 温暖 安抚 慢一点"
)


def _goal_semantic_pressure(goals: tuple[str, ...], *, prototype: tuple[float, ...]) -> float:
    if not goals:
        return 0.0
    values = [
        _clamp((_cosine_similarity(_semantic_embedding(goal), prototype) + 1.0) / 2.0)
        for goal in goals
    ]
    return sum(values) / len(values)


def _relationship_relevant_alerts(alerts: tuple[EvaluationAlert, ...]) -> tuple[EvaluationAlert, ...]:
    return tuple(
        alert
        for alert in alerts
        if alert.code in {"cross_track_stability_degraded", "rollback_pressure_elevated"}
    )
