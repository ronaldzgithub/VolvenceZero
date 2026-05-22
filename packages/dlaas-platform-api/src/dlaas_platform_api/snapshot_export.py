"""Read-only snapshot serialization helpers for DLaaS observability."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from enum import Enum
from typing import Any

_REDACT_KEYS = frozenset(
    {
        "content",
        "detail",
        "raw_text",
        "text",
        "prompt",
        "system_prompt",
    }
)


def snapshot_to_json(value: Any, *, redact: bool = True) -> Any:
    """Convert immutable snapshot values to JSON-safe data.

    This function is intentionally structural and read-only. It does
    not mutate the source, does not call ``copy.deepcopy()``, and does
    not invoke producer-specific methods. Producers remain the SSOT
    for their own descriptions; this helper only serializes public
    dataclass fields.
    """

    return _to_json(value, redact=redact, field_name="")


def _to_json(value: Any, *, redact: bool, field_name: str) -> Any:
    if redact and field_name in _REDACT_KEYS and isinstance(value, str):
        if not value:
            return value
        return {
            "redacted": True,
            "char_count": len(value),
            "preview": value[:80],
        }
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value):
        return {
            field.name: _to_json(
                getattr(value, field.name),
                redact=redact,
                field_name=field.name,
            )
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(k): _to_json(v, redact=redact, field_name=str(k))
            for k, v in value.items()
        }
    if isinstance(value, tuple | list | frozenset | set):
        return [_to_json(v, redact=redact, field_name=field_name) for v in value]
    return repr(value)


__all__ = ["snapshot_to_json"]
