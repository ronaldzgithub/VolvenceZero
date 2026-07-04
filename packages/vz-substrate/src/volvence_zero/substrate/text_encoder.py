"""Substrate-LM-backed text embedding backend (closes ``known-debts.md`` #91).

This is the real :class:`~volvence_zero.semantic_embedding.SemanticEmbeddingBackend`
implementation. Instead of the deterministic character-hash stub, it embeds
arbitrary text by reusing the *already loaded* substrate LM: it runs the
runtime's ``capture(source_text=...)`` and projects the returned
``feature_surface`` into a dense, L2-normalized vector of the requested
``dim``.

Why reuse the substrate LM (vs a separate sentence encoder): the model is
already resident, so there is no extra dependency or weight download, and the
resulting embedding lives in the same feature space the rest of the kernel
already reads from the substrate snapshot (``feature_surface``). The
projection intentionally mirrors ``volvence_zero.memory.retrieval._substrate_embedding``
so retrieval and prototype-comparison consumers share one representation.

Boundary: this lives in ``vz-substrate`` (which owns the runtime), not in
``vz-contracts`` (the zero-upstream seam). A higher tier (Brain wiring)
constructs this and injects it via
``volvence_zero.semantic_embedding.set_semantic_embedding_backend``.
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict

from volvence_zero.semantic_embedding import stub_semantic_embedding
from volvence_zero.substrate.adapter import FeatureSignal
from volvence_zero.substrate.residual_interfaces import OpenWeightResidualRuntime


def _project_feature_surface(
    feature_surface: tuple[FeatureSignal, ...], *, dim: int
) -> tuple[float, ...]:
    """Flatten feature signals in stable name order, L2-normalize, fit ``dim``.

    Returns a zero vector when nothing contributes, so the caller can detect
    the empty case and fall back to the stub. Kept byte-identical in shape to
    ``memory.retrieval._substrate_embedding`` so cosine math is unchanged.
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


def _default_cache_size() -> int:
    raw = os.environ.get("VZ_SEMANTIC_EMBED_CACHE", "512")
    try:
        size = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"VZ_SEMANTIC_EMBED_CACHE must be an integer, got {raw!r}"
        ) from exc
    return max(size, 0)


class SubstrateTextEncoderBackend:
    """Embed arbitrary text via the loaded substrate LM's feature surface.

    Implements the structural
    :class:`~volvence_zero.semantic_embedding.SemanticEmbeddingBackend`
    protocol. Prototypes (fixed reference strings) are encoded once and then
    served from an LRU cache; empty text delegates to the stub so callers
    always receive a well-formed vector.
    """

    def __init__(
        self,
        runtime: OpenWeightResidualRuntime,
        *,
        cache_size: int | None = None,
    ) -> None:
        self._runtime = runtime
        self._cache_size = _default_cache_size() if cache_size is None else max(cache_size, 0)
        self._cache: "OrderedDict[tuple[str, int], tuple[float, ...]]" = OrderedDict()

    def embed(self, text: str, *, dim: int) -> tuple[float, ...]:
        if not text.strip():
            return stub_semantic_embedding(text, dim=dim)
        key = (text, dim)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        capture = self._runtime.capture(source_text=text)
        vector = _project_feature_surface(capture.feature_surface, dim=dim)
        norm = math.sqrt(sum(value * value for value in vector))
        # No usable substrate signal (e.g. degenerate capture) -> stub so the
        # consumer still gets a discriminative vector rather than zeros.
        if norm <= 1e-9:
            vector = stub_semantic_embedding(text, dim=dim)
        if self._cache_size > 0:
            self._cache[key] = vector
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return vector


__all__ = ["SubstrateTextEncoderBackend"]
