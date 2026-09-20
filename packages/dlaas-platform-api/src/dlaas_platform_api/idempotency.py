"""Transport-neutral idempotency header validation."""

from __future__ import annotations


def valid_idempotency_key(value: str) -> bool:
    return bool(value and value.strip() and len(value) <= 256 and all(character.isprintable() for character in value))


__all__ = ["valid_idempotency_key"]
