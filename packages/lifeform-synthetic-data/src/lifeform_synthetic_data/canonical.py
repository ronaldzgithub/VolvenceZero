"""Canonical serialization and strict reconstruction for corpus contracts."""

from __future__ import annotations

import hashlib
import json
import types
import typing
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import TypeVar, get_args, get_origin, get_type_hints

from .contracts import CorpusManifest, ExperienceTrajectory

T = TypeVar("T")
JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


def to_primitive(value: object) -> JsonValue:
    """Convert an immutable contract to JSON-compatible primitives."""

    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: to_primitive(object.__getattribute__(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [to_primitive(item) for item in value]
    if isinstance(value, list):
        return [to_primitive(item) for item in value]
    if isinstance(value, dict):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical mappings require string keys")
            result[key] = to_primitive(item)
        return result
    raise TypeError(f"unsupported canonical value type: {type(value).__name__}")


def canonical_json(value: object) -> str:
    """Return stable UTF-8 JSON with sorted keys and no insignificant space."""

    return json.dumps(
        to_primitive(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_hash(value: object) -> str:
    """Return a SHA-256 digest over the canonical UTF-8 representation."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def from_primitive(contract_type: type[T], value: object) -> T:
    """Strictly reconstruct a typed contract, rejecting unknown or missing keys."""

    return typing.cast(T, _coerce(contract_type, value, path="$"))


def trajectory_from_json(payload: str) -> ExperienceTrajectory:
    decoded = _decode_object(payload)
    return from_primitive(ExperienceTrajectory, decoded)


def manifest_from_json(payload: str) -> CorpusManifest:
    decoded = _decode_object(payload)
    return from_primitive(CorpusManifest, decoded)


def write_canonical_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{canonical_json(value)}\n", encoding="utf-8")


def read_trajectory(path: Path) -> ExperienceTrajectory:
    return trajectory_from_json(path.read_text(encoding="utf-8"))


def read_manifest(path: Path) -> CorpusManifest:
    return manifest_from_json(path.read_text(encoding="utf-8"))


def _decode_object(payload: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("invalid JSON document") from error
    if not isinstance(value, dict):
        raise TypeError("contract JSON root must be an object")
    return typing.cast(dict[str, object], value)


def _coerce(expected: object, value: object, *, path: str) -> object:
    origin = get_origin(expected)
    arguments = get_args(expected)

    if origin is tuple:
        if not isinstance(value, list):
            raise TypeError(f"{path} must be an array")
        if len(arguments) != 2 or arguments[1] is not Ellipsis:
            raise TypeError(f"{path}: only homogeneous tuple contracts are supported")
        return tuple(_coerce(arguments[0], item, path=f"{path}[{index}]") for index, item in enumerate(value))

    if origin in {types.UnionType, typing.Union}:
        if type(None) in arguments:
            if value is None:
                return None
            non_none = tuple(item for item in arguments if item is not type(None))
            if len(non_none) != 1:
                raise TypeError(f"{path}: ambiguous optional contract")
            return _coerce(non_none[0], value, path=path)
        errors: list[str] = []
        for option in arguments:
            try:
                return _coerce(option, value, path=path)
            except (TypeError, ValueError) as error:
                errors.append(str(error))
        raise TypeError(f"{path} does not match any union option: {'; '.join(errors)}")

    if isinstance(expected, type) and issubclass(expected, Enum):
        try:
            return expected(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path} has invalid {expected.__name__} value") from error

    if isinstance(expected, type) and is_dataclass(expected):
        if not isinstance(value, dict):
            raise TypeError(f"{path} must be an object")
        hints = get_type_hints(expected)
        expected_fields = {field.name for field in fields(expected)}
        actual_fields = set(value)
        unknown = sorted(actual_fields - expected_fields)
        missing = sorted(expected_fields - actual_fields)
        if unknown:
            raise ValueError(f"{path} contains unknown fields: {unknown}")
        if missing:
            raise ValueError(f"{path} is missing fields: {missing}")
        kwargs = {
            field.name: _coerce(
                hints[field.name],
                value[field.name],
                path=f"{path}.{field.name}",
            )
            for field in fields(expected)
        }
        return expected(**kwargs)

    if expected is bool:
        if type(value) is not bool:
            raise TypeError(f"{path} must be a boolean")
        return value
    if expected is int:
        if type(value) is not int:
            raise TypeError(f"{path} must be an integer")
        return value
    if expected is float:
        if type(value) not in {int, float}:
            raise TypeError(f"{path} must be a number")
        return float(typing.cast(int | float, value))
    if expected is str:
        if not isinstance(value, str):
            raise TypeError(f"{path} must be a string")
        return value
    if expected is type(None):
        if value is not None:
            raise TypeError(f"{path} must be null")
        return None
    raise TypeError(f"{path}: unsupported contract type {expected!r}")


__all__ = [
    "canonical_json",
    "from_primitive",
    "manifest_from_json",
    "read_manifest",
    "read_trajectory",
    "stable_hash",
    "to_primitive",
    "trajectory_from_json",
    "write_canonical_json",
]
