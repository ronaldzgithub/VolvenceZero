"""Immutable request-level contracts for expression delivery."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json

from volvence_zero.canonical_json import CanonicalJsonError, canonical_json_bytes


_MAX_SCHEMA_BYTES = 64 * 1024


class ExpressionBindingSource(str, Enum):
    """Typed owner readouts permitted as exact expression values."""

    RESPONSE_ACTION_REALIZATION_ACTION_STATEMENT = (
        "response_action_realization.action_statement"
    )


@dataclass(frozen=True)
class ExpressionExactBinding:
    """Bind one JSON pointer to an exact owner-published Unicode string."""

    json_pointer: str
    source: ExpressionBindingSource

    def __post_init__(self) -> None:
        if not isinstance(self.source, ExpressionBindingSource):
            raise TypeError(
                "expression exact binding source must be "
                "ExpressionBindingSource"
            )
        _json_pointer_parts(self.json_pointer)


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
    exact_bindings: tuple[ExpressionExactBinding, ...] = ()

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
        if not isinstance(self.exact_bindings, tuple):
            raise TypeError("expression output exact_bindings must be a tuple")
        if any(
            not isinstance(binding, ExpressionExactBinding)
            for binding in self.exact_bindings
        ):
            raise TypeError(
                "expression output exact_bindings must contain "
                "ExpressionExactBinding values"
            )
        pointers = tuple(binding.json_pointer for binding in self.exact_bindings)
        if len(set(pointers)) != len(pointers):
            raise ValueError("expression output exact binding pointers must be unique")
        sources = tuple(binding.source for binding in self.exact_bindings)
        if len(set(sources)) != len(sources):
            raise ValueError("expression output exact binding sources must be unique")
        for binding in self.exact_bindings:
            _validate_binding_schema(
                schema=parsed,
                json_pointer=binding.json_pointer,
            )

    @property
    def schema(self) -> dict[str, object]:
        parsed = json.loads(self.schema_json)
        assert isinstance(parsed, dict)
        return parsed


def _json_pointer_parts(json_pointer: str) -> tuple[str, ...]:
    if not isinstance(json_pointer, str):
        raise TypeError("expression exact binding json_pointer must be a string")
    if not json_pointer.startswith("/") or json_pointer == "/":
        raise ValueError(
            "expression exact binding json_pointer must identify a field"
        )
    parts: list[str] = []
    for raw_part in json_pointer[1:].split("/"):
        index = 0
        while index < len(raw_part):
            if raw_part[index] == "~":
                if index + 1 >= len(raw_part) or raw_part[index + 1] not in (
                    "0",
                    "1",
                ):
                    raise ValueError(
                        "expression exact binding json_pointer has an invalid "
                        "escape"
                    )
                index += 2
            else:
                index += 1
        parts.append(raw_part.replace("~1", "/").replace("~0", "~"))
    if any(not part for part in parts):
        raise ValueError(
            "expression exact binding json_pointer fields must be non-empty"
        )
    return tuple(parts)


def _validate_binding_schema(
    *,
    schema: dict[str, object],
    json_pointer: str,
) -> None:
    current: object = schema
    traversed: list[str] = []
    for part in _json_pointer_parts(json_pointer):
        if not isinstance(current, dict) or current.get("type") != "object":
            location = "/" + "/".join(traversed) if traversed else "$"
            raise ValueError(
                "expression exact binding schema path must traverse explicit "
                f"object schemas at {location}"
            )
        properties = current.get("properties")
        required = current.get("required")
        if not isinstance(properties, dict) or part not in properties:
            raise ValueError(
                "expression exact binding json_pointer is absent from schema: "
                f"{json_pointer}"
            )
        if not isinstance(required, list) or part not in required:
            raise ValueError(
                "expression exact binding field must be required by schema: "
                f"{json_pointer}"
            )
        current = properties[part]
        traversed.append(part)
    if not isinstance(current, dict) or current.get("type") != "string":
        raise ValueError(
            "expression exact binding target must have JSON Schema type "
            f"string: {json_pointer}"
        )


__all__ = [
    "ExpressionBindingSource",
    "ExpressionExactBinding",
    "ExpressionOutputContract",
]
