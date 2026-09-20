"""Immutable request-level contracts for expression delivery."""

from __future__ import annotations

from dataclasses import dataclass
import json

from volvence_zero.canonical_json import CanonicalJsonError, canonical_json_bytes


_MAX_SCHEMA_BYTES = 64 * 1024


@dataclass(frozen=True)
class ExpressionOutputContract:
    """Strict output envelope requested for one expression synthesis.

    This contract shapes and validates the face presented by an already
    completed cognitive turn.  It is not an input to cognition, memory, PE,
    or world-state ownership.
    """

    schema_name: str
    schema_json: str
    strict: bool = True

    def __post_init__(self) -> None:
        if not self.schema_name.strip():
            raise ValueError("expression output schema_name must be non-empty")
        if self.schema_name != self.schema_name.strip():
            raise ValueError("expression output schema_name must be trimmed")
        if not self.strict:
            raise ValueError("expression output contract requires strict=true")
        encoded = self.schema_json.encode("utf-8")
        if len(encoded) > _MAX_SCHEMA_BYTES:
            raise ValueError("expression output schema exceeds 64 KiB")
        try:
            parsed = json.loads(self.schema_json)
            canonical = canonical_json_bytes(parsed).decode("utf-8")
        except (CanonicalJsonError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("expression output schema must be valid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("expression output schema must be a JSON object")
        if canonical != self.schema_json:
            raise ValueError("expression output schema_json must be canonical JSON")

    @property
    def schema(self) -> dict[str, object]:
        parsed = json.loads(self.schema_json)
        assert isinstance(parsed, dict)
        return parsed


__all__ = ["ExpressionOutputContract"]
