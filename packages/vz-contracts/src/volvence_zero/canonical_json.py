"""Strict canonical JSON helpers for owner-authored persistence payloads.

The typed codec is intentionally schema-driven: an owner supplies the expected
Python type, while consumers only move the resulting JSON value.  No dynamic
class names, object hooks, fallback stringification, or executable codecs are
accepted.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import types
from collections.abc import Mapping, Sequence, Set
from dataclasses import fields, is_dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints


JsonScalar = None | bool | int | float | str
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class CanonicalJsonError(ValueError):
    """A value violates its declared JSON schema or canonical encoding."""


def canonical_json_bytes(value: JsonValue) -> bytes:
    """Return deterministic UTF-8 JSON, rejecting non-finite numbers."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError) as exc:
        raise CanonicalJsonError(f"value is not canonical JSON: {exc}") from exc


def strict_json_loads(data: bytes, *, max_bytes: int) -> JsonValue:
    """Parse UTF-8 JSON with duplicate-key, constant, and size checks."""

    if not isinstance(data, bytes):
        raise TypeError(f"JSON input must be bytes, got {type(data).__name__}")
    if len(data) > max_bytes:
        raise CanonicalJsonError(f"JSON input exceeds size limit: actual={len(data)}, max={max_bytes}")

    def reject_constant(value: str) -> None:
        raise CanonicalJsonError(f"non-finite JSON constant is forbidden: {value}")

    def unique_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {}
        for key, value in pairs:
            if key in result:
                raise CanonicalJsonError(f"duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    try:
        parsed = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (RecursionError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CanonicalJsonError(f"invalid UTF-8 JSON: {exc}") from exc
    try:
        return _validate_json_value(parsed, path="$")
    except RecursionError as exc:
        raise CanonicalJsonError(
            "JSON nesting exceeds the supported depth"
        ) from exc


def typed_to_json(value: object, expected_type: object) -> JsonValue:
    """Encode ``value`` according to a statically supplied owner schema."""

    return _encode_typed(value, expected_type, path="$")


def typed_from_json(payload: JsonValue, expected_type: object) -> object:
    """Reconstruct a typed owner value from strict JSON data."""

    return _decode_typed(payload, expected_type, path="$")


def freeze_json_value(value: JsonValue) -> object:
    """Recursively freeze parsed JSON for immutable snapshot publication."""

    if isinstance(value, dict):
        return MappingProxyType({key: freeze_json_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(freeze_json_value(item) for item in value)
    return value


def _validate_json_value(value: object, *, path: str) -> JsonValue:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalJsonError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, list | tuple):
        return [_validate_json_value(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise CanonicalJsonError(f"{path} contains a non-string object key")
        return {key: _validate_json_value(item, path=f"{path}.{key}") for key, item in value.items()}
    raise CanonicalJsonError(f"{path} contains unsupported JSON value type {type(value).__name__}")


def _encode_typed(value: object, annotation: object, *, path: str) -> JsonValue:
    if annotation is Any:
        return _encode_any(value, path=path)
    if annotation is None or annotation is type(None):
        if value is not None:
            raise CanonicalJsonError(f"{path} must be null")
        return None

    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (types.UnionType, Union):
        return _encode_union(value, args, path=path)
    if origin is Literal:
        if value not in args:
            raise CanonicalJsonError(f"{path} is not one of {args!r}")
        return _encode_any(value, path=path)
    if origin is tuple:
        if not isinstance(value, tuple):
            raise CanonicalJsonError(f"{path} must be a tuple")
        return _encode_tuple(value, args, path=path)
    if origin is list:
        if not isinstance(value, list):
            raise CanonicalJsonError(f"{path} must be a list")
        item_type = args[0] if args else Any
        return [_encode_typed(item, item_type, path=f"{path}[{index}]") for index, item in enumerate(value)]
    if origin in (dict, Mapping):
        if not isinstance(value, Mapping):
            raise CanonicalJsonError(f"{path} must be a mapping")
        key_type, value_type = args if len(args) == 2 else (Any, Any)
        return _encode_mapping(
            value,
            key_type=key_type,
            value_type=value_type,
            path=path,
        )
    if origin in (Sequence, Set, set, frozenset):
        if isinstance(value, str | bytes) or not isinstance(value, origin):
            raise CanonicalJsonError(f"{path} must be {origin.__name__}")
        item_type = args[0] if args else Any
        encoded = [_encode_typed(item, item_type, path=f"{path}[{index}]") for index, item in enumerate(value)]
        if origin in (Set, set, frozenset):
            encoded.sort(key=canonical_json_bytes)
        return encoded

    if isinstance(annotation, type) and issubclass(annotation, Enum):
        if not isinstance(value, annotation):
            raise CanonicalJsonError(f"{path} must be {annotation.__qualname__}, got {type(value).__name__}")
        return _encode_any(value.value, path=path)
    if isinstance(annotation, type) and is_dataclass(annotation):
        if type(value) is not annotation:
            raise CanonicalJsonError(f"{path} must be {annotation.__qualname__}, got {type(value).__name__}")
        hints = get_type_hints(annotation)
        return {
            field.name: _encode_typed(
                getattr(value, field.name),
                hints[field.name],
                path=f"{path}.{field.name}",
            )
            for field in fields(annotation)
        }
    if annotation is bytes:
        if not isinstance(value, bytes):
            raise CanonicalJsonError(f"{path} must be bytes")
        return {"base64": base64.b64encode(value).decode("ascii")}
    if annotation is bool:
        if type(value) is not bool:
            raise CanonicalJsonError(f"{path} must be bool")
        return value
    if annotation is int:
        if type(value) is not int:
            raise CanonicalJsonError(f"{path} must be int")
        return value
    if annotation is float:
        if type(value) not in (int, float):
            raise CanonicalJsonError(f"{path} must be float")
        result = float(value)
        if not math.isfinite(result):
            raise CanonicalJsonError(f"{path} must be finite")
        return result
    if annotation is str:
        if not isinstance(value, str):
            raise CanonicalJsonError(f"{path} must be str")
        return value
    raise CanonicalJsonError(f"{path} uses unsupported schema annotation {annotation!r}")


def _decode_typed(payload: JsonValue, annotation: object, *, path: str) -> object:
    if annotation is Any:
        return _validate_json_value(payload, path=path)
    if annotation is None or annotation is type(None):
        if payload is not None:
            raise CanonicalJsonError(f"{path} must be null")
        return None

    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (types.UnionType, Union):
        return _decode_union(payload, args, path=path)
    if origin is Literal:
        if payload not in args:
            raise CanonicalJsonError(f"{path} is not one of {args!r}")
        return payload
    if origin is tuple:
        if not isinstance(payload, list | tuple):
            raise CanonicalJsonError(f"{path} must be an array")
        return _decode_tuple(payload, args, path=path)
    if origin is list:
        if not isinstance(payload, list | tuple):
            raise CanonicalJsonError(f"{path} must be an array")
        item_type = args[0] if args else Any
        return [_decode_typed(item, item_type, path=f"{path}[{index}]") for index, item in enumerate(payload)]
    if origin in (dict, Mapping):
        key_type, value_type = args if len(args) == 2 else (Any, Any)
        return _decode_mapping(
            payload,
            key_type=key_type,
            value_type=value_type,
            path=path,
        )
    if origin in (Sequence, Set, set, frozenset):
        if not isinstance(payload, list | tuple):
            raise CanonicalJsonError(f"{path} must be an array")
        item_type = args[0] if args else Any
        values = [_decode_typed(item, item_type, path=f"{path}[{index}]") for index, item in enumerate(payload)]
        if origin in (Set, set):
            return set(values)
        if origin is frozenset:
            return frozenset(values)
        return tuple(values)

    if isinstance(annotation, type) and issubclass(annotation, Enum):
        try:
            return annotation(payload)
        except (TypeError, ValueError) as exc:
            raise CanonicalJsonError(f"{path} is not a valid {annotation.__qualname__}: {payload!r}") from exc
    if isinstance(annotation, type) and is_dataclass(annotation):
        if not isinstance(payload, Mapping):
            raise CanonicalJsonError(f"{path} must be an object")
        expected_fields = tuple(field.name for field in fields(annotation))
        if set(payload) != set(expected_fields):
            missing = sorted(set(expected_fields) - set(payload))
            unknown = sorted(set(payload) - set(expected_fields))
            raise CanonicalJsonError(f"{path} field mismatch: missing={missing}, unknown={unknown}")
        hints = get_type_hints(annotation)
        values = {
            field.name: _decode_typed(
                payload[field.name],
                hints[field.name],
                path=f"{path}.{field.name}",
            )
            for field in fields(annotation)
        }
        try:
            return annotation(**values)
        except (TypeError, ValueError) as exc:
            raise CanonicalJsonError(f"{path} failed {annotation.__qualname__} validation: {exc}") from exc
    if annotation is bytes:
        if not isinstance(payload, Mapping) or set(payload) != {"base64"}:
            raise CanonicalJsonError(f"{path} must be a base64 object")
        raw = payload["base64"]
        if not isinstance(raw, str):
            raise CanonicalJsonError(f"{path}.base64 must be str")
        try:
            return base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise CanonicalJsonError(f"{path}.base64 is invalid") from exc
    if annotation is bool:
        if type(payload) is not bool:
            raise CanonicalJsonError(f"{path} must be bool")
        return payload
    if annotation is int:
        if type(payload) is not int:
            raise CanonicalJsonError(f"{path} must be int")
        return payload
    if annotation is float:
        if type(payload) not in (int, float):
            raise CanonicalJsonError(f"{path} must be float")
        result = float(payload)
        if not math.isfinite(result):
            raise CanonicalJsonError(f"{path} must be finite")
        return result
    if annotation is str:
        if not isinstance(payload, str):
            raise CanonicalJsonError(f"{path} must be str")
        return payload
    raise CanonicalJsonError(f"{path} uses unsupported schema annotation {annotation!r}")


def _encode_any(value: object, *, path: str) -> JsonValue:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalJsonError(f"{path} must be finite")
        return value
    if isinstance(value, list | tuple):
        return [_encode_any(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalJsonError(f"{path} mapping keys must be strings")
            result[key] = _encode_any(item, path=f"{path}.{key}")
        return result
    raise CanonicalJsonError(f"{path} contains unsupported untyped value {type(value).__name__}")


def _encode_union(value: object, args: tuple[object, ...], *, path: str) -> JsonValue:
    if value is None and type(None) in args:
        return None
    candidates = tuple(item for item in args if item is not type(None))
    if len(candidates) == 1:
        return _encode_typed(value, candidates[0], path=path)
    for index, candidate in enumerate(candidates):
        if _matches_type(value, candidate):
            return {
                "variant": index,
                "value": _encode_typed(value, candidate, path=f"{path}.value"),
            }
    raise CanonicalJsonError(f"{path} does not match any declared union variant")


def _decode_union(payload: JsonValue, args: tuple[object, ...], *, path: str) -> object:
    if payload is None and type(None) in args:
        return None
    candidates = tuple(item for item in args if item is not type(None))
    if len(candidates) == 1:
        return _decode_typed(payload, candidates[0], path=path)
    if not isinstance(payload, Mapping) or set(payload) != {"variant", "value"}:
        raise CanonicalJsonError(f"{path} must be a tagged union object")
    index = payload["variant"]
    if type(index) is not int or index < 0 or index >= len(candidates):
        raise CanonicalJsonError(f"{path}.variant is out of range")
    return _decode_typed(
        payload["value"],
        candidates[index],
        path=f"{path}.value",
    )


def _matches_type(value: object, annotation: object) -> bool:
    origin = get_origin(annotation)
    if origin is not None:
        return isinstance(value, origin)
    return isinstance(annotation, type) and isinstance(value, annotation)


def _encode_tuple(
    value: tuple[object, ...],
    args: tuple[object, ...],
    *,
    path: str,
) -> list[JsonValue]:
    if len(args) == 2 and args[1] is Ellipsis:
        return [_encode_typed(item, args[0], path=f"{path}[{index}]") for index, item in enumerate(value)]
    if args and len(value) != len(args):
        raise CanonicalJsonError(f"{path} tuple length mismatch: expected={len(args)}, actual={len(value)}")
    item_types = args or tuple(Any for _ in value)
    return [
        _encode_typed(item, item_type, path=f"{path}[{index}]")
        for index, (item, item_type) in enumerate(zip(value, item_types, strict=True))
    ]


def _decode_tuple(
    payload: list[JsonValue] | tuple[object, ...],
    args: tuple[object, ...],
    *,
    path: str,
) -> tuple[object, ...]:
    if len(args) == 2 and args[1] is Ellipsis:
        return tuple(_decode_typed(item, args[0], path=f"{path}[{index}]") for index, item in enumerate(payload))
    if args and len(payload) != len(args):
        raise CanonicalJsonError(f"{path} tuple length mismatch: expected={len(args)}, actual={len(payload)}")
    item_types = args or tuple(Any for _ in payload)
    return tuple(
        _decode_typed(item, item_type, path=f"{path}[{index}]")
        for index, (item, item_type) in enumerate(zip(payload, item_types, strict=True))
    )


def _encode_mapping(
    value: Mapping[object, object],
    *,
    key_type: object,
    value_type: object,
    path: str,
) -> JsonValue:
    if key_type in (str, Any):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalJsonError(f"{path} mapping keys must be strings")
            result[key] = _encode_typed(
                item,
                value_type,
                path=f"{path}.{key}",
            )
        return result
    pairs = [
        [
            _encode_typed(key, key_type, path=f"{path}.key"),
            _encode_typed(item, value_type, path=f"{path}.value"),
        ]
        for key, item in value.items()
    ]
    pairs.sort(key=canonical_json_bytes)
    return {"pairs": pairs}


def _decode_mapping(
    payload: JsonValue,
    *,
    key_type: object,
    value_type: object,
    path: str,
) -> dict[object, object]:
    if key_type in (str, Any):
        if not isinstance(payload, Mapping):
            raise CanonicalJsonError(f"{path} must be an object")
        return {key: _decode_typed(item, value_type, path=f"{path}.{key}") for key, item in payload.items()}
    if not isinstance(payload, Mapping) or set(payload) != {"pairs"}:
        raise CanonicalJsonError(f"{path} must be a mapping-pairs object")
    raw_pairs = payload["pairs"]
    if not isinstance(raw_pairs, list | tuple):
        raise CanonicalJsonError(f"{path}.pairs must be an array")
    result: dict[object, object] = {}
    for index, raw_pair in enumerate(raw_pairs):
        if not isinstance(raw_pair, list | tuple) or len(raw_pair) != 2:
            raise CanonicalJsonError(f"{path}.pairs[{index}] must be a pair")
        key = _decode_typed(raw_pair[0], key_type, path=f"{path}.pairs[{index}][0]")
        if key in result:
            raise CanonicalJsonError(f"{path} contains a duplicate decoded key")
        result[key] = _decode_typed(
            raw_pair[1],
            value_type,
            path=f"{path}.pairs[{index}][1]",
        )
    return result


__all__ = [
    "CanonicalJsonError",
    "JsonValue",
    "canonical_json_bytes",
    "freeze_json_value",
    "strict_json_loads",
    "typed_from_json",
    "typed_to_json",
]
