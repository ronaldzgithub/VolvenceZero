"""Semantic-pressure readouts used by evaluation scoring.

Small character-level embedding helpers plus two module-level prototype
vectors (``TASK_PRESSURE_PROTOTYPE`` / ``SUPPORT_PRESENCE_PROTOTYPE``)
that :class:`EvaluationBackbone` uses to score task-pressure / support-
presence signals on free-form goal strings without invoking the full
LLM substrate.

Slice S.2 (2026-05-04): extracted from ``evaluation/backbone.py``.
"""

from __future__ import annotations

import math

from volvence_zero.evaluation.statistics import _clamp


def _semantic_embedding(text: str, *, dim: int = 8) -> tuple[float, ...]:
    tokens = _semantic_tokens(text)
    if not tokens:
        return tuple(0.0 for _ in range(dim))
    vector = [0.0 for _ in range(dim)]
    for token in tokens:
        token_scale = max(len(token), 1)
        for index, char in enumerate(token):
            vector[(index + len(token)) % dim] += (ord(char) % 41) / 41.0 / token_scale
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-6:
        return tuple(0.0 for _ in range(dim))
    return tuple(value / norm for value in vector)


def _semantic_tokens(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    ascii_buffer: list[str] = []
    compact = "".join(char for char in text.lower() if not char.isspace())
    for char in text.lower():
        if char.isascii() and char.isalnum():
            ascii_buffer.append(char)
            continue
        if ascii_buffer:
            tokens.append("".join(ascii_buffer))
            ascii_buffer.clear()
        if not char.isspace():
            tokens.append(char)
    if ascii_buffer:
        tokens.append("".join(ascii_buffer))
    tokens.extend(compact[index : index + 2] for index in range(len(compact) - 1))
    return tuple(tokens)


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or not right:
        return 0.0
    return sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))


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


def _relationship_relevant_alerts(alerts: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        alert
        for alert in alerts
        if "cross-track stability" in alert.lower() or "rollback pressure" in alert.lower()
    )
