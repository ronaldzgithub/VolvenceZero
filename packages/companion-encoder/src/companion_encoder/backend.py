# Copyright 2026 Companion Standard Contributors
# Licensed under the Apache License, Version 2.0.

"""SemanticEmbeddingBackend adapter for a trained checkpoint.

This is the seam the standard defines: any consumer that accepts a
``companion_standard.SemanticEmbeddingBackend`` can be handed a trained
relationship encoder through this class, with no knowledge of torch or
checkpoints.

Dimension adaptation: the encoder has a fixed ``embedding_dim``; the
protocol lets callers request any ``dim``. Requests are served by
truncation (dim <= native) or zero-padding (dim > native), followed by
re-normalization — deterministic, structure-preserving for the leading
components, and honest about carrying no extra information beyond the
native width. Empty/whitespace text delegates to the standard's stub
(the protocol makes empty-text robustness the backend's job).
"""

from __future__ import annotations

import math
import pathlib

from companion_standard import stub_semantic_embedding


class EncoderEmbeddingBackend:
    """Wrap a trained checkpoint as a ``SemanticEmbeddingBackend``."""

    def __init__(
        self, checkpoint_path: pathlib.Path | str, *, device: str = "cpu"
    ) -> None:
        from companion_encoder.model import load_checkpoint

        self._model = load_checkpoint(checkpoint_path, device=device)
        self._native_dim = self._model.config.embedding_dim

    def embed(self, text: str, *, dim: int) -> tuple[float, ...]:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if not text.strip():
            return stub_semantic_embedding(text, dim=dim)

        import torch

        with torch.no_grad():
            native = self._model([text])["embedding"][0]
        values = [float(value) for value in native]
        if dim <= self._native_dim:
            resized = values[:dim]
        else:
            resized = values + [0.0] * (dim - self._native_dim)
        norm = math.sqrt(sum(value * value for value in resized))
        if norm <= 1e-9:
            return stub_semantic_embedding(text, dim=dim)
        return tuple(value / norm for value in resized)


__all__ = ["EncoderEmbeddingBackend"]
