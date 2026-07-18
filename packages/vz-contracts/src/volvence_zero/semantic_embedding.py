"""Semantic embedding SSOT entry point (closes ``known-debts.md`` #3 / #91).

SSOT split (oss-relationship-representation-standard.md, Phase A1):

* The *seam* — :class:`SemanticEmbeddingBackend` protocol, the
  deterministic character-hash stub, and ``CANONICAL_MODULUS`` — lives in
  ``companion_standard.embedding`` (the public Relationship Representation
  Standard) and is re-exported here so every existing
  ``volvence_zero.semantic_embedding`` import keeps working.
* The *wiring mechanism* — the process-level backend registry
  (install / conflict-demotion / reset) and the routing entry points
  (:func:`semantic_embedding`, :func:`semantic_topic_similarity`) — stays
  private in this module.

All internal call sites must reuse these entry points; new forks are
forbidden (enforced by ``tests/contracts/test_semantic_embedding_ssot.py``).

Boundary note: this module lives in ``vz-contracts`` (the foundation
wheel), so it may not import ``substrate``. A real backend (e.g.
``volvence_zero.substrate.SubstrateTextEncoderBackend``) is a process-level
seam supplied by a higher tier and injected at wiring time.
"""

from __future__ import annotations

from companion_standard.embedding import (  # noqa: F401
    CANONICAL_MODULUS,
    SemanticEmbeddingBackend,
    stub_cosine_similarity,
    stub_semantic_embedding,
    stub_semantic_tokens,
)

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
