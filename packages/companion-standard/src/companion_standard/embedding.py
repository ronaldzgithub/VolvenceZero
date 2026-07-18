# Copyright 2026 Companion Standard Contributors
# Licensed under the Apache License, Version 2.0.

"""Semantic embedding seam — the injectable backend protocol and the
deterministic fallback stub.

Part of the Relationship Representation Standard: any encoder that
implements :class:`SemanticEmbeddingBackend` can serve as the embedding
backend for standard consumers (this is the seam an open-weights
relationship encoder plugs into — see
the public RFC's data-pipeline section).

The stub is a deterministic character-level token + hash embedding —
useful for tests and offline conformance runs, explicitly NOT a semantic
model. The process-level backend registry (install / conflict / reset
wiring) is runtime mechanism and stays outside the standard; a conformant
runtime supplies its own injection seam.

Why ``CANONICAL_MODULUS = 65537``:

* It is a Fermat prime (2**16 + 1) and therefore coprime with every
  small embedding ``dim`` in use (4 / 6 / 8 / 16 / ...). Coprimality
  matters because the hash bucket index is computed as
  ``(index + len(token)) % dim``; if ``modulus`` shared a factor with
  ``dim`` the value distribution would collapse onto a coset, biasing
  the resulting vector.
* It is also large enough to spread ``ord(char)`` over its full range
  for Han / Latin code points instead of saturating the modulus.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

CANONICAL_MODULUS = 65537


def stub_semantic_tokens(text: str) -> tuple[str, ...]:
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


def stub_semantic_embedding(text: str, *, dim: int = 8) -> tuple[float, ...]:
    tokens = stub_semantic_tokens(text)
    if not tokens:
        return tuple(0.0 for _ in range(dim))
    vector = [0.0 for _ in range(dim)]
    for token in tokens:
        token_scale = max(len(token), 1)
        for index, char in enumerate(token):
            vector[(index + len(token)) % dim] += (
                (ord(char) % CANONICAL_MODULUS) / CANONICAL_MODULUS / token_scale
            )
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-6:
        return tuple(0.0 for _ in range(dim))
    return tuple(value / norm for value in vector)


def stub_cosine_similarity(
    left: tuple[float, ...], right: tuple[float, ...]
) -> float:
    if not left or not right:
        return 0.0
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    )


@runtime_checkable
class SemanticEmbeddingBackend(Protocol):
    """Injectable real text-embedding backend.

    The stub above is a deterministic character-hash placeholder. A real
    backend (a substrate LM's hidden states, an open-weights relationship
    encoder, ...) implements ``embed`` and is injected at wiring time by
    the consuming runtime. A backend is responsible for its own
    empty/short-text robustness (it may delegate to
    :func:`stub_semantic_embedding` internally).
    """

    def embed(self, text: str, *, dim: int) -> tuple[float, ...]:
        ...


__all__ = [
    "CANONICAL_MODULUS",
    "SemanticEmbeddingBackend",
    "stub_cosine_similarity",
    "stub_semantic_embedding",
    "stub_semantic_tokens",
]
