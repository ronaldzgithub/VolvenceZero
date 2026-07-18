# Copyright 2026 Companion Standard Contributors
# Licensed under the Apache License, Version 2.0.

"""Canonical JSON serialisation and stable content hashing.

Every value type in the standard serialises through :func:`to_canonical_json`
so that two semantically identical objects produce byte-identical JSON —
regardless of dict ordering or whitespace — and therefore the same
:func:`stable_hash`. This mirrors the ``scenario_hash`` semantics that
Companion Bench uses for audit trails: the hash can be cited publicly
without revealing the object body.

Rules:

* dataclasses serialise via ``dataclasses.asdict`` (tuples become lists);
* enums serialise to their ``.value``;
* keys are sorted, separators are compact, output is UTF-8 with
  ``ensure_ascii=False`` so non-ASCII text hashes identically across
  platforms;
* ``None`` fields are kept (dropping them would make hashes ambiguous
  between "absent" and "null").
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
from typing import Any


def to_jsonable(value: Any) -> Any:
    """Recursively convert a standard value object into JSON-safe primitives."""
    if isinstance(value, enum.Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(
        f"to_jsonable: unsupported type {type(value).__name__!r}; "
        "standard values must be dataclasses, enums, or JSON primitives"
    )


def to_canonical_json(value: Any) -> str:
    """Serialise to canonical (sorted-key, compact) JSON text."""
    return json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def stable_hash(value: Any) -> str:
    """SHA-256 hex digest of the canonical JSON serialisation."""
    return hashlib.sha256(to_canonical_json(value).encode("utf-8")).hexdigest()
