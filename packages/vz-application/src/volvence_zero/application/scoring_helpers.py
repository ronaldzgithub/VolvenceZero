"""Pure helpers extracted from ``application.runtime`` (W5 of ssot-cleanup-p0-p4).

These utilities are stateless and free of regime / snapshot semantics,
so they live in their own module to keep ``runtime.py`` focused on the
actual scoring orchestration.

Public names re-exported from ``application.runtime`` for backward compat:

* :func:`clamp01` (legacy ``_clamp``)
* :func:`signed_centered`
* :func:`clamp_signed`
* :func:`dedupe`
* :func:`nearest_anchor_value`
* :func:`truncate_text`
* :func:`ranked_labels`
* :func:`semantic_tokens`
* :func:`semantic_embedding`
* :func:`cosine_similarity`
* :func:`semantic_similarity`

Closes ``docs/known-debts.md`` #3: semantic embedding helpers are
re-exported from the canonical SSOT in
``volvence_zero.semantic_embedding``. Existing call sites keep using
the original names from this module unchanged.
"""

from __future__ import annotations

from typing import Any, Mapping

from volvence_zero.semantic_embedding import (
    stub_cosine_similarity as cosine_similarity,
    stub_semantic_embedding as semantic_embedding,
    stub_semantic_tokens as semantic_tokens,
)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def signed_centered(score: float) -> float:
    return (score - 0.5) * 2.0


def clamp_signed(value: float, *, magnitude: float = 0.25) -> float:
    return max(-magnitude, min(magnitude, value))


def dedupe(items: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)


def nearest_anchor_value(score: float, anchors: tuple[tuple[Any, float], ...]) -> Any:
    return min(anchors, key=lambda item: abs(score - item[1]))[0]


def truncate_text(text: str, *, max_chars: int = 96) -> str:
    compact = " ".join(part for part in text.split() if part)
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def ranked_labels(score_map: Mapping[str, float], *, max_count: int) -> tuple[str, ...]:
    ranked = sorted(score_map.items(), key=lambda item: (-item[1], item[0]))
    return tuple(label for label, _ in ranked[:max_count])


def semantic_similarity(text: str, prototype_text: str) -> float:
    embedding = semantic_embedding(text)
    prototype = semantic_embedding(prototype_text)
    return clamp01((cosine_similarity(embedding, prototype) + 1.0) / 2.0)


__all__ = [
    "clamp01",
    "clamp_signed",
    "cosine_similarity",
    "dedupe",
    "nearest_anchor_value",
    "ranked_labels",
    "semantic_embedding",
    "semantic_similarity",
    "semantic_tokens",
    "signed_centered",
    "truncate_text",
]
