# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Embed-retrieval component (H-B).

A standard memory wrapper retrieves the most relevant prior turns (across all of
a user's sessions) and splices them into the current prompt. This module owns:

* :class:`Embedder` — the protocol the harness calls to vectorise text.
* :class:`HashingEmbedder` — a deterministic, dependency-free bag-of-tokens
  embedder. It is the default so smoke runs and the reproducibility contract do
  not depend on a downloaded model. A production deployment may inject a real
  sentence-embedding model with the same protocol; the retrieval math is
  identical.
* :class:`EmbedEntry` — one indexed turn.
* :func:`cosine` / :func:`top_k` — retrieval helpers.

The embedder is deterministic and local on purpose: the value being measured is
"does retrieval of prior turns help", not "how good is a specific embedding
vendor". Swapping in a stronger embedder only strengthens this baseline.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import re
from typing import Iterable, Protocol, runtime_checkable


# Public constants — part of the reproducibility contract.
EMBED_DIM: int = 256
RETRIEVAL_TOP_K: int = 4
RETRIEVAL_MIN_SCORE: float = 0.05

_TOKEN_RE: re.Pattern[str] = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Embedder(Protocol):
    """Vectorise text into a fixed-dimension list of floats."""

    @property
    def dim(self) -> int: ...

    @property
    def name(self) -> str: ...

    def embed(self, text: str) -> tuple[float, ...]: ...


class HashingEmbedder:
    """Deterministic bag-of-tokens hashing embedder (no dependencies).

    Tokens are lowercased ``[a-z0-9]+`` runs; each token is hashed into one of
    :data:`EMBED_DIM` buckets and accumulated, then the vector is L2-normalised
    so :func:`cosine` reduces to a dot product.
    """

    def __init__(self, *, dim: int = EMBED_DIM, name: str = "ref-harness/hashing-embed-v1") -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive; got {dim}")
        self._dim = dim
        self._name = name

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self._name

    def embed(self, text: str) -> tuple[float, ...]:
        vec = [0.0] * self._dim
        for token in _TOKEN_RE.findall((text or "").lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest, "big") % self._dim
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return tuple(vec)
        return tuple(v / norm for v in vec)


class SentenceTransformerEmbedder:
    """Real semantic embedder backed by ``sentence-transformers``.

    Defaults to ``BAAI/bge-m3`` (multilingual, 1024-dim) so retrieval works
    on both the English and Chinese public scenarios. The model is loaded
    lazily on first :meth:`embed` so importing this module stays cheap and
    offline-safe; the heavy dependency + weight download only happens when
    the embedder is actually selected (``--embedder bge-m3``).

    Embeddings are L2-normalised (``normalize_embeddings=True``) so
    :func:`cosine` reduces to a dot product, matching HashingEmbedder.
    """

    def __init__(
        self,
        *,
        model_id: str = "BAAI/bge-m3",
        device: str | None = None,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._model = None  # lazy
        self._dim: int | None = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - guarded at call site
            raise ImportError(
                "SentenceTransformerEmbedder requires the 'embed' extra: "
                "pip install 'companion-ref-harness[embed]' (sentence-transformers)."
            ) from exc
        self._model = SentenceTransformer(self._model_id, device=self._device)
        # sentence-transformers renamed get_sentence_embedding_dimension ->
        # get_embedding_dimension; support both.
        get_dim = getattr(
            self._model,
            "get_embedding_dimension",
            None,
        ) or self._model.get_sentence_embedding_dimension
        self._dim = int(get_dim())

    @property
    def dim(self) -> int:
        self._ensure_model()
        assert self._dim is not None
        return self._dim

    @property
    def name(self) -> str:
        return f"ref-harness/sentence-transformer:{self._model_id}"

    def embed(self, text: str) -> tuple[float, ...]:
        self._ensure_model()
        assert self._model is not None
        vec = self._model.encode(
            text or "",
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return tuple(float(x) for x in vec.tolist())


@dataclasses.dataclass(frozen=True)
class EmbedEntry:
    """One indexed turn in the embed retrieval store."""

    scope_key: str
    turn_id: str
    role: str
    content: str
    embedding: tuple[float, ...]
    ts: str  # ISO-8601 UTC

    def to_prompt_line(self) -> str:
        return f"- ({self.role}) {self.content}"


def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Cosine similarity. Inputs need not be normalised."""

    if len(a) != len(b):
        raise ValueError(f"embedding length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def top_k(
    *,
    query: tuple[float, ...],
    entries: Iterable[EmbedEntry],
    k: int = RETRIEVAL_TOP_K,
    min_score: float = RETRIEVAL_MIN_SCORE,
) -> tuple[EmbedEntry, ...]:
    """Return the ``k`` highest-cosine entries above ``min_score``.

    Ties broken by recency (``ts`` descending) so the freshest relevant turn
    wins. Deterministic for a fixed input order.
    """

    scored = [(cosine(query, e.embedding), e) for e in entries]
    scored = [(s, e) for (s, e) in scored if s >= min_score]
    scored.sort(key=lambda pair: (pair[0], pair[1].ts), reverse=True)
    return tuple(e for _s, e in scored[:k])
