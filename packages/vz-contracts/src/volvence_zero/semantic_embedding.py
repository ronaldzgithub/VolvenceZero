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
# M1 (#91 follow-up): multi-substrate process isolation. The seam is
# process-global, so two substrates injecting different encoders would
# silently cross-contaminate every consumer's embedding space. We track
# the installing owner (e.g. substrate model_id); a second install from a
# DIFFERENT owner demotes the whole process to the stub (deterministic,
# substrate-independent) and latches a conflict flag that stays queryable.
_ACTIVE_BACKEND_OWNER: str = ""
_BACKEND_CONFLICT: bool = False


def set_semantic_embedding_backend(
    backend: SemanticEmbeddingBackend | None,
    *,
    owner: str = "",
) -> str:
    """Install (or clear, with ``None``) the process-level embedding backend.

    Wiring time only. Passing ``None`` restores the stub fallback (and
    clears the owner/conflict state — an explicit reset is the rollback
    path).

    Multi-substrate isolation (#91 follow-up): when a backend is already
    installed by a different ``owner``, the process demotes to the stub
    for ALL consumers instead of letting two substrates interleave
    incompatible embedding spaces. The demotion is explicit and
    observable: this function returns ``"conflict-stub"`` and
    :func:`semantic_embedding_backend_status` reports the latched
    conflict until a reset.

    Returns one of ``"installed"`` / ``"cleared"`` / ``"conflict-stub"``.
    """

    global _ACTIVE_BACKEND, _ACTIVE_BACKEND_OWNER, _BACKEND_CONFLICT
    if backend is None:
        _ACTIVE_BACKEND = None
        _ACTIVE_BACKEND_OWNER = ""
        _BACKEND_CONFLICT = False
        return "cleared"
    if (
        _ACTIVE_BACKEND is not None
        and _ACTIVE_BACKEND_OWNER
        and owner != _ACTIVE_BACKEND_OWNER
    ):
        _ACTIVE_BACKEND = None
        _ACTIVE_BACKEND_OWNER = ""
        _BACKEND_CONFLICT = True
        return "conflict-stub"
    _ACTIVE_BACKEND = backend
    _ACTIVE_BACKEND_OWNER = owner
    _BACKEND_CONFLICT = False
    return "installed"


def get_semantic_embedding_backend() -> SemanticEmbeddingBackend | None:
    return _ACTIVE_BACKEND


def semantic_embedding_backend_status() -> tuple[str, str, bool]:
    """Return ``(state, owner, conflict)`` for observability.

    ``state`` is ``"backend"`` when a real backend serves embeddings,
    else ``"stub"``. ``conflict`` stays latched after a multi-substrate
    demotion until the next explicit reset.
    """

    state = "backend" if _ACTIVE_BACKEND is not None else "stub"
    return (state, _ACTIVE_BACKEND_OWNER, _BACKEND_CONFLICT)


def reset_semantic_embedding_backend() -> None:
    """Clear the active backend (restores stub fallback). Idempotent."""

    global _ACTIVE_BACKEND, _ACTIVE_BACKEND_OWNER, _BACKEND_CONFLICT
    _ACTIVE_BACKEND = None
    _ACTIVE_BACKEND_OWNER = ""
    _BACKEND_CONFLICT = False


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


def semantic_topic_similarity(left_text: str, right_text: str) -> float:
    """Topic similarity in ``[0, 1]``, backend-aware (M1 / #91 follow-up).

    With a real embedding backend installed, similarity is the clamped
    cosine of the two texts' embeddings (true semantic space). Without a
    backend the historical stub-token Jaccard is preserved byte-for-byte,
    so the synthetic / test surface is unchanged. Consumers that used to
    hand-roll ``Jaccard(stub_semantic_tokens(...))`` should call this
    instead so the upgrade point stays a single seam.
    """

    if not left_text or not right_text:
        return 0.0
    if _ACTIVE_BACKEND is not None:
        cosine = semantic_cosine(
            semantic_embedding(left_text, dim=16),
            semantic_embedding(right_text, dim=16),
        )
        return max(0.0, min(1.0, cosine))
    left = frozenset(stub_semantic_tokens(left_text))
    right = frozenset(stub_semantic_tokens(right_text))
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    if intersection == 0:
        return 0.0
    return intersection / len(left | right)


__all__ = [
    "CANONICAL_MODULUS",
    "SemanticEmbeddingBackend",
    "get_semantic_embedding_backend",
    "reset_semantic_embedding_backend",
    "semantic_cosine",
    "semantic_embedding",
    "semantic_embedding_backend_status",
    "semantic_topic_similarity",
    "set_semantic_embedding_backend",
    "stub_cosine_similarity",
    "stub_semantic_embedding",
    "stub_semantic_tokens",
]
