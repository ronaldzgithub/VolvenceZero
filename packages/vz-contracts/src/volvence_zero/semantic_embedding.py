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


__all__ = [
    "CANONICAL_MODULUS",
    "stub_cosine_similarity",
    "stub_semantic_embedding",
    "stub_semantic_tokens",
]
