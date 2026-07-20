"""JSON Schema export for immutable corpus contracts."""

from __future__ import annotations

import json
import types
import typing
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import get_args, get_origin, get_type_hints

from .contracts import CorpusManifest, ExperienceTrajectory


def build_json_schema() -> dict[str, object]:
    definitions: dict[str, object] = {}
    roots = [
        _schema_for(ExperienceTrajectory, definitions),
        _schema_for(CorpusManifest, definitions),
    ]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://schemas.volvence.ai/synthetic-experience/v1.json",
        "title": "Volvence Unified Synthetic Experience Corpus v1",
        "oneOf": roots,
        "$defs": dict(sorted(definitions.items())),
    }


def export_json_schema(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        build_json_schema(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    path.write_text(f"{payload}\n", encoding="utf-8")


def bundled_schema_path() -> Path:
    return Path(__file__).parent / "schemas" / "synthetic-experience.schema.json"


def _schema_for(expected: object, definitions: dict[str, object]) -> dict[str, object]:
    origin = get_origin(expected)
    arguments = get_args(expected)

    if origin is tuple:
        if len(arguments) != 2 or arguments[1] is not Ellipsis:
            raise TypeError("only homogeneous tuple contracts are supported")
        return {
            "type": "array",
            "items": _schema_for(arguments[0], definitions),
        }

    if origin in {types.UnionType, typing.Union}:
        schemas = [_schema_for(option, definitions) for option in arguments]
        return {"anyOf": schemas}

    if expected is type(None):
        return {"type": "null"}
    if expected is str:
        return {"type": "string"}
    if expected is bool:
        return {"type": "boolean"}
    if expected is int:
        return {"type": "integer"}
    if expected is float:
        return {"type": "number"}

    if isinstance(expected, type) and issubclass(expected, Enum):
        return {
            "type": "string",
            "enum": [member.value for member in expected],
        }

    if isinstance(expected, type) and is_dataclass(expected):
        name = expected.__name__
        if name not in definitions:
            definitions[name] = {}
            hints = get_type_hints(expected)
            properties = {field.name: _schema_for(hints[field.name], definitions) for field in fields(expected)}
            required = [field.name for field in fields(expected)]
            definitions[name] = {
                "type": "object",
                "title": name,
                "additionalProperties": False,
                "properties": properties,
                "required": required,
            }
        return {"$ref": f"#/$defs/{name}"}

    raise TypeError(f"unsupported JSON Schema contract type: {expected!r}")


__all__ = ["build_json_schema", "bundled_schema_path", "export_json_schema"]
