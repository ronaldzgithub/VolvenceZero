"""Stub semantic embedding SSOT (closes ``known-debts.md`` #3).

This module is the single canonical implementation of the placeholder
character-level token + hash embedding used by ``vz-application``,
``vz-cognition.dual_track`` and ``vz-cognition.evaluation`` while the
real embedding head is being wired. All call sites must reuse this
function; new forks are forbidden (enforced by
``tests/contracts/test_semantic_embedding_ssot.py``).

Why this lives in ``vz-contracts``:

* The three call sites span ``vz-application`` and ``vz-cognition``;
  hosting the SSOT in either one would force the other to add a
  cross-tier dependency. ``vz-contracts`` is the foundation wheel with
  zero upstream deps, so consumers may freely depend on it.

Why ``CANONICAL_MODULUS = 65537``:

* It is a Fermat prime (2**16 + 1) and therefore coprime with every
  small embedding ``dim`` we currently use (4 / 6 / 8 / 16 / ...).
  Coprimality matters because the hash bucket index is computed as
  ``(index + len(token)) % dim``; if ``modulus`` shared a factor with
  ``dim`` the value distribution would collapse onto a coset, biasing
  the resulting vector. The previous fork values (37 / 41) accidentally
  satisfied coprimality for ``dim == 8`` but did not reserve headroom
  for higher-dim experiments.
* It is also large enough to spread ``ord(char)`` over its full range
  for the Han / Latin code points we feed in, instead of saturating
  the modulus the way mod-37 does on 16-bit CJK ranges.

The implementation deliberately mirrors the previous forks line-for-line
so that the closure is a pure single-source consolidation — only the
modulus changes.
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
    """Injectable real text-embedding backend (closes ``known-debts.md`` #91).

    The stub above is a deterministic character-hash placeholder. A real
    backend (e.g. one that reuses the loaded substrate LM's hidden states,
    see ``volvence_zero.substrate.SubstrateTextEncoderBackend``) can be
    injected at initialization via :func:`set_semantic_embedding_backend`.

    Consumers that want the real embedding when available call
    :func:`semantic_embedding`, which routes to the active backend and
    falls back to :func:`stub_semantic_embedding` when none is set.

    Boundary note: this module lives in ``vz-contracts`` (the zero-upstream
    foundation wheel), so it may not import ``substrate``. The backend is a
    process-level seam supplied by a higher tier and injected at wiring
    time. This keeps the SSOT (one embedding entry point) while letting the
    real encoder live where the runtime does.
    """

    def embed(self, text: str, *, dim: int) -> tuple[float, ...]:
        ...


_ACTIVE_BACKEND: SemanticEmbeddingBackend | None = None


def set_semantic_embedding_backend(backend: SemanticEmbeddingBackend | None) -> None:
    """Install (or clear, with ``None``) the process-level embedding backend.

    Wiring time only. Passing ``None`` restores the stub fallback. This is a
    single global seam, appropriate for single-substrate processes; a
    multi-substrate deployment (DLaaS multi-instance) should leave it unset
    to avoid cross-substrate embedding leakage (tracked as #91 follow-up).
    """

    global _ACTIVE_BACKEND
    _ACTIVE_BACKEND = backend


def get_semantic_embedding_backend() -> SemanticEmbeddingBackend | None:
    return _ACTIVE_BACKEND


def reset_semantic_embedding_backend() -> None:
    """Clear the active backend (restores stub fallback). Idempotent."""

    global _ACTIVE_BACKEND
    _ACTIVE_BACKEND = None


def semantic_embedding(text: str, *, dim: int = 8) -> tuple[float, ...]:
    """Route to the active real backend when set, else the stub fallback.

    Errors from an installed backend are NOT swallowed (they surface as
    real substrate failures, per ``no-swallow-errors``). A backend is
    responsible for its own empty/short-text robustness (it may delegate to
    :func:`stub_semantic_embedding` internally).
    """

    backend = _ACTIVE_BACKEND
    if backend is None:
        return stub_semantic_embedding(text, dim=dim)
    return backend.embed(text, dim=dim)


def semantic_cosine(
    left: tuple[float, ...], right: tuple[float, ...]
) -> float:
    """Cosine similarity for embeddings from :func:`semantic_embedding`.

    Backend-agnostic: both stub and real backends return L2-normalized
    vectors, so this is the plain dot product (same as the stub helper).
    """

    return stub_cosine_similarity(left, right)


__all__ = [
    "CANONICAL_MODULUS",
    "SemanticEmbeddingBackend",
    "get_semantic_embedding_backend",
    "reset_semantic_embedding_backend",
    "semantic_cosine",
    "semantic_embedding",
    "set_semantic_embedding_backend",
    "stub_cosine_similarity",
    "stub_semantic_embedding",
    "stub_semantic_tokens",
]
