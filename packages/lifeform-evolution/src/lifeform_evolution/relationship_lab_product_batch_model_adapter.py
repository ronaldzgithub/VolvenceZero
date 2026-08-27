"""Add-only fixed-batch extension for the frozen public BGE adapter.

The base adapter remains byte-exact for historical admission replay.  This
module owns only the batch execution surface needed by source-v5 table
materialization; it does not own model identity or persisted table schemas.
"""

from __future__ import annotations

import math
import os
import pathlib
from typing import Callable, Protocol

from lifeform_evolution.relationship_lab_product_model_adapters import (
    BGE_M3_MODEL_REVISION,
    BGE_M3_SENTENCE_TRANSFORMERS_VERSION,
    BGE_M3_WEIGHT_BYTES_SHA256,
    RevisionPinnedBgeM3PublicSemanticEmbedder,
    RevisionPinnedProductHistorySemanticEmbedder,
)


class RevisionPinnedBatchProductHistorySemanticEmbedder(
    RevisionPinnedProductHistorySemanticEmbedder,
    Protocol,
):
    """Revision-pinned encoder with one explicit fixed-size batch surface."""

    def embed_many(
        self,
        texts: tuple[str, ...],
        *,
        batch_size: int,
    ) -> tuple[tuple[float, ...], ...]: ...


class RevisionPinnedBatchBgeM3PublicSemanticEmbedder(
    RevisionPinnedBgeM3PublicSemanticEmbedder
):
    """Batch-only extension that preserves the frozen base adapter path."""

    def embed_many(
        self,
        texts: tuple[str, ...],
        *,
        batch_size: int,
    ) -> tuple[tuple[float, ...], ...]:
        """Embed one immutable ordered batch without fallback or resizing."""

        if not isinstance(texts, tuple) or not texts:
            raise ValueError("texts must be a non-empty tuple")
        for index, value in enumerate(texts):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"texts[{index}] must be a non-empty string")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        encoded = self._ensure_model().encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        raw_rows = encoded.tolist()
        if not isinstance(raw_rows, list) or len(raw_rows) != len(texts):
            raise ValueError("BGE batch output row count must match input text count")

        rows: list[tuple[float, ...]] = []
        expected_width: int | None = None
        for row_index, raw_row in enumerate(raw_rows):
            if not isinstance(raw_row, list) or not raw_row:
                raise ValueError(f"BGE batch output row {row_index} must be non-empty")
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in raw_row
            ):
                raise ValueError(
                    f"BGE batch output row {row_index} values must be numeric"
                )
            row = tuple(float(value) for value in raw_row)
            if not all(math.isfinite(value) for value in row):
                raise ValueError(
                    f"BGE batch output row {row_index} values must be finite"
                )
            if expected_width is None:
                expected_width = len(row)
            elif len(row) != expected_width:
                raise ValueError("BGE batch output rows must have one fixed width")
            rows.append(row)
        return tuple(rows)


def bge_m3_batch_public_semantic_embedder(
    *,
    device: str | None = None,
    model_revision: str = BGE_M3_MODEL_REVISION,
    weights_sha256: str = BGE_M3_WEIGHT_BYTES_SHA256,
    sentence_transformers_version: str = BGE_M3_SENTENCE_TRANSFORMERS_VERSION,
    model_factory: Callable[..., object] | None = None,
    snapshot_path: pathlib.Path | None = None,
    snapshot_resolver: Callable[..., str | os.PathLike[str]] | None = None,
    runtime_version_resolver: Callable[[str], str] | None = None,
) -> RevisionPinnedBatchProductHistorySemanticEmbedder:
    """Return the lazy offline BGE adapter with the batch extension enabled."""

    return RevisionPinnedBatchBgeM3PublicSemanticEmbedder(
        model_revision=model_revision,
        weights_sha256=weights_sha256,
        sentence_transformers_version=sentence_transformers_version,
        device=device,
        model_factory=model_factory,
        snapshot_path=snapshot_path,
        snapshot_resolver=snapshot_resolver,
        runtime_version_resolver=runtime_version_resolver,
    )


__all__ = [
    "RevisionPinnedBatchBgeM3PublicSemanticEmbedder",
    "RevisionPinnedBatchProductHistorySemanticEmbedder",
    "bge_m3_batch_public_semantic_embedder",
]
