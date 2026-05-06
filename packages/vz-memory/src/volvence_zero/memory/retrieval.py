"""Derived retrieval helpers for explicit memory artifacts.

These helpers are rebuildable from artifact state and are not memory truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, TYPE_CHECKING

from volvence_zero.memory.contracts import MemoryEntry

if TYPE_CHECKING:
    from volvence_zero.substrate import FeatureSignal


def _clamp_strength(value: float) -> float:
    return max(0.0, min(1.0, value))


def _entry_in_subject_scope(
    entry: MemoryEntry, active_subject_ids: tuple[str, ...]
) -> bool:
    return any(subject_id in active_subject_ids for subject_id in entry.subject_ids)


def _tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    ascii_buffer: list[str] = []
    compact = "".join(char for char in text.lower() if not char.isspace())
    for char in text.lower():
        if char.isascii() and char.isalnum():
            ascii_buffer.append(char)
            continue
        if ascii_buffer:
            tokens.add("".join(ascii_buffer))
            ascii_buffer.clear()
        if not char.isspace():
            tokens.add(char)
    if ascii_buffer:
        tokens.add("".join(ascii_buffer))
    for index in range(len(compact) - 1):
        tokens.add(compact[index : index + 2])
    return tokens


def _semantic_embedding(*, text: str, tags: tuple[str, ...], dim: int = 6) -> tuple[float, ...]:
    tokens = tuple(sorted(_tokenize(text) | {tag.lower() for tag in tags}))
    if not tokens:
        return tuple(0.0 for _ in range(dim))
    vector = [0.0 for _ in range(dim)]
    for token in tokens:
        token_strength = max(len(token), 1)
        for index, char in enumerate(token):
            vector[(index + len(token)) % dim] += (ord(char) % 31) / 31.0 / token_strength
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-6:
        return tuple(0.0 for _ in range(dim))
    return tuple(value / norm for value in vector)


def _substrate_embedding(
    *,
    feature_surface: tuple["FeatureSignal", ...],
    dim: int,
) -> tuple[float, ...]:
    """Phase 1.C: build a dense embedding from substrate feature signals.

    NL alignment (R8 + retrieval-from-substrate): retrieval should be a
    downstream readout of the substrate's feature_surface, not a parallel
    character-hash space. We concatenate the published per-signal value
    tuples in stable name order, L2-normalize, then truncate / zero-pad
    to ``dim`` so retrieval cosine math is unchanged.

    Returns an empty tuple of length ``dim`` (zero vector) when no signal
    contributes any value, so the caller can detect the "no substrate"
    case and fall back to ``_semantic_embedding``.
    """

    if not feature_surface or dim <= 0:
        return tuple(0.0 for _ in range(dim))
    ordered = sorted(feature_surface, key=lambda signal: signal.name)
    flat: list[float] = []
    for signal in ordered:
        for value in signal.values:
            flat.append(float(value))
    if not flat:
        return tuple(0.0 for _ in range(dim))
    norm = math.sqrt(sum(value * value for value in flat))
    if norm <= 1e-9:
        return tuple(0.0 for _ in range(dim))
    flat = [value / norm for value in flat]
    if len(flat) >= dim:
        return tuple(flat[:dim])
    return tuple(flat + [0.0] * (dim - len(flat)))


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or not right:
        return 0.0
    return sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))


def _mean_abs(values: tuple[float, ...]) -> float:
    if not values:
        return 0.0
    return sum(abs(value) for value in values) / len(values)


def _align_signal(signal: tuple[float, ...], *, dim: int) -> tuple[float, ...]:
    if len(signal) == dim:
        return signal
    if not signal:
        return tuple(0.0 for _ in range(dim))
    return tuple(signal[index % len(signal)] for index in range(dim))


def _blend_signals(
    *,
    dim: int,
    weighted_signals: tuple[tuple[tuple[float, ...], float], ...],
) -> tuple[float, ...]:
    total_weight = sum(weight for _, weight in weighted_signals if weight > 0.0)
    if total_weight <= 1e-6:
        return tuple(0.0 for _ in range(dim))
    blended = [0.0 for _ in range(dim)]
    for signal, weight in weighted_signals:
        if weight <= 0.0:
            continue
        aligned = _align_signal(signal, dim=dim)
        for index in range(dim):
            blended[index] += aligned[index] * weight
    return tuple(_clamp_strength(value / total_weight) for value in blended)


def summarize_entries(entries: Iterable[MemoryEntry], *, fallback: str) -> str:
    collected = tuple(entries)
    if not collected:
        return fallback
    preview = "; ".join(entry.content for entry in collected[:3])
    if len(collected) > 3:
        preview += "; ..."
    return preview


@dataclass(frozen=True)
class LearnedMemoryRecall:
    query_base_signal: tuple[float, ...]
    query_signal: tuple[float, ...]
    core_signal: tuple[float, ...]
    tower_profile_id: str
    tower_depth: int
    retrieval_confidence: float
    tower_alignment: float
    query_only_alignment: float
    composite_alignment: float
    transfer_alignment: float
    artifact_weight: float
    learned_weight: float
    description: str



class DerivedRetrievalIndex:
    """Rebuildable artifact retrieval support.

    This index is intentionally derived from explicit artifacts and can be
    reconstructed from checkpoints. It should not be treated as memory truth.
    """

    def __init__(self) -> None:
        self._artifact_embeddings: dict[str, tuple[float, ...]] = {}

    def index_entry(self, entry: MemoryEntry, *, embedding: tuple[float, ...] | None = None) -> None:
        self._artifact_embeddings[entry.entry_id] = embedding or _semantic_embedding(
            text=entry.content,
            tags=entry.tags,
        )

    def affinity(self, *, entry: MemoryEntry, query_embedding: tuple[float, ...]) -> float:
        return _cosine_similarity(query_embedding, self._artifact_embeddings.get(entry.entry_id, (0.0,) * len(query_embedding)))

    def export_state(self) -> tuple[tuple[str, tuple[float, ...]], ...]:
        return tuple(sorted(self._artifact_embeddings.items()))

    def restore(self, embeddings: tuple[tuple[str, tuple[float, ...]], ...]) -> None:
        self._artifact_embeddings = dict(embeddings)

