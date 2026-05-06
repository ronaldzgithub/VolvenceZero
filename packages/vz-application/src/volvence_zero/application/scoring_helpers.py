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

The semantic-embedding stubs in this module are one of three forks
that ``docs/known-debts.md`` debt #3 tracks; extraction here is the
first step toward unifying them across ``vz-cognition`` and
``vz-application``. The actual unification (replacing the three forks
with one canonical implementation) is a follow-up wave.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


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


# -----------------------------------------------------------------------------
# Semantic embedding stub (one of three forks tracked in known-debts #3).
# -----------------------------------------------------------------------------


def semantic_tokens(text: str) -> tuple[str, ...]:
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


def semantic_embedding(text: str, *, dim: int = 8) -> tuple[float, ...]:
    tokens = semantic_tokens(text)
    if not tokens:
        return tuple(0.0 for _ in range(dim))
    vector = [0.0 for _ in range(dim)]
    for token in tokens:
        token_scale = max(len(token), 1)
        for index, char in enumerate(token):
            vector[(index + len(token)) % dim] += (ord(char) % 37) / 37.0 / token_scale
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-6:
        return tuple(0.0 for _ in range(dim))
    return tuple(value / norm for value in vector)


def cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or not right:
        return 0.0
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    )


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
