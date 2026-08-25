"""Safe, canonical archives for owner-authored agent learning state.

Each adaptive owner publishes an ``OwnerPersistenceSnapshot``.  This module
only validates the outer envelope, binds every opaque owner payload to a
digest, and packages one or more agent archives.  It never imports owner
state classes and never executes data-driven object constructors.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from volvence_zero.canonical_json import (
    CanonicalJsonError,
    JsonValue,
    canonical_json_bytes,
    freeze_json_value,
    strict_json_loads,
    typed_to_json,
)
from volvence_zero.owner_hydration import OwnerPersistenceSnapshot


AGENT_LEARNING_ARCHIVE_SCHEMA_VERSION = "agent-learning-archive.v2"
AGENT_LEARNING_COLLECTION_SCHEMA_VERSION = "agent-learning-checkpoint-collection.v1"
_MAX_SINGLE_ARCHIVE_BYTES = 32 * 1024 * 1024
_MAX_COLLECTION_ARCHIVE_BYTES = 128 * 1024 * 1024
_SHA256_HEX_LENGTH = 64


class AgentLearningArchiveError(ValueError):
    """An archive failed schema, integrity, or compatibility checks."""


@dataclass(frozen=True)
class AgentLearningArchiveInfo:
    schema_version: str
    checkpoint_id: str
    state_fingerprint: str
    policy_fingerprint: str
    temporal_fingerprint: str
    memory_fingerprint: str
    owner_versions: tuple[tuple[str, int], ...]
    size_bytes: int


@dataclass(frozen=True)
class AgentLearningArchiveRecord:
    info: AgentLearningArchiveInfo
    owner_snapshots: tuple[OwnerPersistenceSnapshot, ...]


@dataclass(frozen=True)
class AgentLearningArchiveMetadata:
    schema_version: str
    checkpoint_count: int
    checkpoint_fingerprints: tuple[str, ...]
    compatibility: tuple[tuple[str, str], ...]
    archive_sha256: str


@dataclass(frozen=True)
class AgentLearningCheckpointArchive:
    """A colony/package-level container of opaque single-agent archives."""

    metadata: AgentLearningArchiveMetadata
    checkpoint_archives: tuple[bytes, ...]


def encode_agent_learning_archive(
    *,
    checkpoint_id: str,
    owner_snapshots: tuple[OwnerPersistenceSnapshot, ...],
    policy_fingerprint: str,
    temporal_fingerprint: str,
    memory_fingerprint: str,
) -> bytes:
    """Encode one agent's owner-authored parts as canonical UTF-8 JSON."""

    _require_non_empty_string(checkpoint_id, field="checkpoint_id")
    policy_fingerprint = _require_sha256(
        policy_fingerprint,
        field="policy_fingerprint",
    )
    temporal_fingerprint = _require_sha256(
        temporal_fingerprint,
        field="temporal_fingerprint",
    )
    memory_fingerprint = _require_sha256(
        memory_fingerprint,
        field="memory_fingerprint",
    )
    if not owner_snapshots:
        raise AgentLearningArchiveError("agent learning archive requires at least one owner snapshot")
    owner_names = tuple(snapshot.owner_name for snapshot in owner_snapshots)
    if len(set(owner_names)) != len(owner_names):
        raise AgentLearningArchiveError("agent learning archive owner names must be unique")

    parts = tuple(
        sorted(
            (_encode_owner_part(snapshot) for snapshot in owner_snapshots),
            key=lambda item: _require_string(item["owner_name"], field="owner_name"),
        )
    )
    state_fingerprint = _state_fingerprint(parts)
    envelope: dict[str, JsonValue] = {
        "schema_version": AGENT_LEARNING_ARCHIVE_SCHEMA_VERSION,
        "checkpoint_id": checkpoint_id,
        "state_fingerprint": state_fingerprint,
        "policy_fingerprint": policy_fingerprint,
        "temporal_fingerprint": temporal_fingerprint,
        "memory_fingerprint": memory_fingerprint,
        "parts": list(parts),
    }
    archive_sha256 = _sha256(canonical_json_bytes(envelope))
    envelope["archive_sha256"] = archive_sha256
    encoded = canonical_json_bytes(envelope)
    if len(encoded) > _MAX_SINGLE_ARCHIVE_BYTES:
        raise AgentLearningArchiveError(
            f"single agent learning archive exceeds size limit: actual={len(encoded)}, max={_MAX_SINGLE_ARCHIVE_BYTES}"
        )
    return encoded


def decode_agent_learning_archive(
    archive_bytes: bytes,
) -> AgentLearningArchiveRecord:
    """Decode and fully integrity-check one safe JSON agent archive."""

    envelope = _load_canonical_object(
        archive_bytes,
        max_bytes=_MAX_SINGLE_ARCHIVE_BYTES,
    )
    _require_exact_fields(
        envelope,
        {
            "schema_version",
            "checkpoint_id",
            "state_fingerprint",
            "policy_fingerprint",
            "temporal_fingerprint",
            "memory_fingerprint",
            "parts",
            "archive_sha256",
        },
        context="agent learning archive",
    )
    schema_version = _require_string(
        envelope["schema_version"],
        field="schema_version",
    )
    if schema_version != AGENT_LEARNING_ARCHIVE_SCHEMA_VERSION:
        raise AgentLearningArchiveError(f"unsupported agent learning archive schema: {schema_version!r}")
    _verify_envelope_digest(envelope)
    checkpoint_id = _require_non_empty_string(
        envelope["checkpoint_id"],
        field="checkpoint_id",
    )
    policy_fingerprint = _require_sha256(
        envelope["policy_fingerprint"],
        field="policy_fingerprint",
    )
    temporal_fingerprint = _require_sha256(
        envelope["temporal_fingerprint"],
        field="temporal_fingerprint",
    )
    memory_fingerprint = _require_sha256(
        envelope["memory_fingerprint"],
        field="memory_fingerprint",
    )
    raw_parts = envelope["parts"]
    if not isinstance(raw_parts, list) or not raw_parts:
        raise AgentLearningArchiveError("agent learning archive parts must be a non-empty array")
    parsed_parts = tuple(_decode_owner_part(item) for item in raw_parts)
    owner_names = tuple(snapshot.owner_name for snapshot in parsed_parts)
    if owner_names != tuple(sorted(owner_names)):
        raise AgentLearningArchiveError("agent learning archive parts must be sorted by owner_name")
    if len(set(owner_names)) != len(owner_names):
        raise AgentLearningArchiveError("agent learning archive contains duplicate owner_name values")
    expected_state_fingerprint = _require_sha256(
        envelope["state_fingerprint"],
        field="state_fingerprint",
    )
    actual_state_fingerprint = _state_fingerprint(tuple(_encode_owner_part(item) for item in parsed_parts))
    if actual_state_fingerprint != expected_state_fingerprint:
        raise AgentLearningArchiveError("agent learning archive state fingerprint mismatch")
    return AgentLearningArchiveRecord(
        info=AgentLearningArchiveInfo(
            schema_version=schema_version,
            checkpoint_id=checkpoint_id,
            state_fingerprint=actual_state_fingerprint,
            policy_fingerprint=policy_fingerprint,
            temporal_fingerprint=temporal_fingerprint,
            memory_fingerprint=memory_fingerprint,
            owner_versions=tuple((snapshot.owner_name, snapshot.schema_version) for snapshot in parsed_parts),
            size_bytes=len(archive_bytes),
        ),
        owner_snapshots=parsed_parts,
    )


def encode_agent_learning_checkpoint_archive(
    checkpoint_archives: tuple[bytes, ...],
    *,
    compatibility: tuple[tuple[str, str], ...],
) -> bytes:
    """Package safe single-agent archives without inspecting owner payloads."""

    if not checkpoint_archives:
        raise AgentLearningArchiveError(
            "checkpoint collection requires at least one agent archive"
        )
    normalized_compatibility = _validate_compatibility(compatibility)
    records = tuple(
        decode_agent_learning_archive(item)
        for item in checkpoint_archives
    )
    checkpoint_ids = tuple(record.info.checkpoint_id for record in records)
    if len(set(checkpoint_ids)) != len(checkpoint_ids):
        raise AgentLearningArchiveError(
            "checkpoint collection ids must be unique"
        )
    encoded_archives = [
        base64.b64encode(item).decode("ascii")
        for item in checkpoint_archives
    ]
    checkpoint_fingerprints = [record.info.state_fingerprint for record in records]
    envelope: dict[str, JsonValue] = {
        "schema_version": AGENT_LEARNING_COLLECTION_SCHEMA_VERSION,
        "compatibility": [list(item) for item in normalized_compatibility],
        "checkpoint_fingerprints": checkpoint_fingerprints,
        "checkpoint_archives": encoded_archives,
    }
    archive_sha256 = _sha256(canonical_json_bytes(envelope))
    envelope["archive_sha256"] = archive_sha256
    encoded = canonical_json_bytes(envelope)
    if len(encoded) > _MAX_COLLECTION_ARCHIVE_BYTES:
        raise AgentLearningArchiveError(
            f"checkpoint collection exceeds size limit: actual={len(encoded)}, max={_MAX_COLLECTION_ARCHIVE_BYTES}"
        )
    return encoded


def decode_agent_learning_checkpoint_archive(
    archive_bytes: bytes,
    *,
    expected_compatibility: tuple[tuple[str, str], ...],
) -> AgentLearningCheckpointArchive:
    """Decode a collection and validate every nested single-agent archive."""

    envelope = _load_canonical_object(
        archive_bytes,
        max_bytes=_MAX_COLLECTION_ARCHIVE_BYTES,
    )
    _require_exact_fields(
        envelope,
        {
            "schema_version",
            "compatibility",
            "checkpoint_fingerprints",
            "checkpoint_archives",
            "archive_sha256",
        },
        context="agent learning checkpoint collection",
    )
    schema_version = _require_string(
        envelope["schema_version"],
        field="schema_version",
    )
    if schema_version != AGENT_LEARNING_COLLECTION_SCHEMA_VERSION:
        raise AgentLearningArchiveError(f"unsupported checkpoint collection schema: {schema_version!r}")
    _verify_envelope_digest(envelope)
    compatibility = _decode_compatibility(envelope["compatibility"])
    normalized_expected = _validate_compatibility(expected_compatibility)
    if compatibility != normalized_expected:
        raise AgentLearningArchiveError(
            f"checkpoint compatibility mismatch: expected={normalized_expected!r}, actual={compatibility!r}"
        )
    raw_archives = envelope["checkpoint_archives"]
    if not isinstance(raw_archives, list) or not raw_archives:
        raise AgentLearningArchiveError(
            "checkpoint_archives must be a non-empty array"
        )
    checkpoint_archives = tuple(
        _decode_base64_archive(item, index=index)
        for index, item in enumerate(raw_archives)
    )
    records = tuple(
        decode_agent_learning_archive(item)
        for item in checkpoint_archives
    )
    checkpoint_ids = tuple(record.info.checkpoint_id for record in records)
    if len(set(checkpoint_ids)) != len(checkpoint_ids):
        raise AgentLearningArchiveError(
            "checkpoint collection contains duplicate checkpoint ids"
        )
    raw_fingerprints = envelope["checkpoint_fingerprints"]
    if not isinstance(raw_fingerprints, list):
        raise AgentLearningArchiveError("checkpoint_fingerprints must be an array")
    fingerprints = tuple(
        _require_sha256(
            item,
            field=f"checkpoint_fingerprints[{index}]",
        )
        for index, item in enumerate(raw_fingerprints)
    )
    actual_fingerprints = tuple(record.info.state_fingerprint for record in records)
    if fingerprints != actual_fingerprints:
        raise AgentLearningArchiveError("checkpoint fingerprints do not match nested archives")
    return AgentLearningCheckpointArchive(
        metadata=AgentLearningArchiveMetadata(
            schema_version=schema_version,
            checkpoint_count=len(checkpoint_archives),
            checkpoint_fingerprints=fingerprints,
            compatibility=compatibility,
            archive_sha256=_require_sha256(
                envelope["archive_sha256"],
                field="archive_sha256",
            ),
        ),
        checkpoint_archives=checkpoint_archives,
    )


def _encode_owner_part(
    snapshot: OwnerPersistenceSnapshot,
) -> dict[str, JsonValue]:
    if not isinstance(snapshot, OwnerPersistenceSnapshot):
        raise TypeError(
            "owner archive parts must be OwnerPersistenceSnapshot values"
        )
    _require_non_empty_string(snapshot.owner_name, field="owner_name")
    if type(snapshot.schema_version) is not int or snapshot.schema_version < 1:
        raise AgentLearningArchiveError(
            f"owner {snapshot.owner_name!r} schema_version must be a positive integer"
        )
    payload = typed_to_json(snapshot.payload, Mapping[str, Any])
    if not isinstance(payload, dict):
        raise AgentLearningArchiveError(
            f"owner {snapshot.owner_name!r} payload must encode as an object"
        )
    if not isinstance(snapshot.description, str):
        raise AgentLearningArchiveError(
            f"owner {snapshot.owner_name!r} description must be a string"
        )
    payload_sha256 = _sha256(canonical_json_bytes(payload))
    return {
        "owner_name": snapshot.owner_name,
        "schema_version": snapshot.schema_version,
        "payload": payload,
        "description": snapshot.description,
        "payload_sha256": payload_sha256,
    }


def _decode_owner_part(value: JsonValue) -> OwnerPersistenceSnapshot:
    if not isinstance(value, dict):
        raise AgentLearningArchiveError("owner archive part must be an object")
    _require_exact_fields(
        value,
        {
            "owner_name",
            "schema_version",
            "payload",
            "description",
            "payload_sha256",
        },
        context="owner archive part",
    )
    owner_name = _require_non_empty_string(
        value["owner_name"],
        field="owner_name",
    )
    schema_version = value["schema_version"]
    if type(schema_version) is not int or schema_version < 1:
        raise AgentLearningArchiveError(
            f"owner {owner_name!r} schema_version must be a positive integer"
        )
    payload = value["payload"]
    if not isinstance(payload, dict):
        raise AgentLearningArchiveError(f"owner {owner_name!r} payload must be an object")
    expected_payload_sha256 = _require_sha256(
        value["payload_sha256"],
        field=f"{owner_name}.payload_sha256",
    )
    actual_payload_sha256 = _sha256(canonical_json_bytes(payload))
    if actual_payload_sha256 != expected_payload_sha256:
        raise AgentLearningArchiveError(
            f"owner {owner_name!r} payload digest mismatch"
        )
    description = _require_string(
        value["description"],
        field=f"{owner_name}.description",
    )
    frozen_payload = freeze_json_value(payload)
    if not isinstance(frozen_payload, Mapping):
        raise AgentLearningArchiveError(
            f"owner {owner_name!r} frozen payload must remain a mapping"
        )
    return OwnerPersistenceSnapshot(
        owner_name=owner_name,
        schema_version=schema_version,
        payload=frozen_payload,
        description=description,
    )


def _state_fingerprint(
    parts: tuple[dict[str, JsonValue], ...],
) -> str:
    state_parts: list[JsonValue] = []
    for part in parts:
        state_parts.append(
            {
                "owner_name": part["owner_name"],
                "schema_version": part["schema_version"],
                "payload": part["payload"],
            }
        )
    return _sha256(canonical_json_bytes(state_parts))


def _load_canonical_object(
    data: bytes,
    *,
    max_bytes: int,
) -> dict[str, JsonValue]:
    try:
        parsed = strict_json_loads(data, max_bytes=max_bytes)
    except CanonicalJsonError as exc:
        raise AgentLearningArchiveError(str(exc)) from exc
    if not isinstance(parsed, dict):
        raise AgentLearningArchiveError("archive root must be a JSON object")
    if canonical_json_bytes(parsed) != data:
        raise AgentLearningArchiveError("archive bytes are valid JSON but not canonical JSON")
    return parsed


def _verify_envelope_digest(envelope: dict[str, JsonValue]) -> None:
    expected = _require_sha256(
        envelope["archive_sha256"],
        field="archive_sha256",
    )
    unsigned = {key: value for key, value in envelope.items() if key != "archive_sha256"}
    actual = _sha256(canonical_json_bytes(unsigned))
    if actual != expected:
        raise AgentLearningArchiveError("archive envelope digest mismatch")


def _decode_base64_archive(value: JsonValue, *, index: int) -> bytes:
    encoded = _require_string(
        value,
        field=f"checkpoint_archives[{index}]",
    )
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise AgentLearningArchiveError(f"checkpoint_archives[{index}] is not valid base64") from exc


def _validate_compatibility(
    compatibility: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    keys = tuple(key for key, _ in compatibility)
    if len(set(keys)) != len(keys):
        raise AgentLearningArchiveError("archive compatibility keys must be unique")
    normalized: list[tuple[str, str]] = []
    for index, (key, value) in enumerate(compatibility):
        normalized.append(
            (
                _require_non_empty_string(
                    key,
                    field=f"compatibility[{index}].key",
                ),
                _require_string(
                    value,
                    field=f"compatibility[{index}].value",
                ),
            )
        )
    return tuple(normalized)


def _decode_compatibility(
    value: JsonValue,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        raise AgentLearningArchiveError("compatibility must be an array")
    pairs: list[tuple[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, list) or len(item) != 2:
            raise AgentLearningArchiveError(f"compatibility[{index}] must be a key/value pair")
        pairs.append(
            (
                _require_non_empty_string(
                    item[0],
                    field=f"compatibility[{index}].key",
                ),
                _require_string(
                    item[1],
                    field=f"compatibility[{index}].value",
                ),
            )
        )
    return _validate_compatibility(tuple(pairs))


def _require_exact_fields(
    value: dict[str, JsonValue],
    expected: set[str],
    *,
    context: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise AgentLearningArchiveError(
            f"{context} field mismatch: missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )


def _require_string(value: JsonValue, *, field: str) -> str:
    if not isinstance(value, str):
        raise AgentLearningArchiveError(f"{field} must be a string")
    return value


def _require_non_empty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AgentLearningArchiveError(f"{field} must be a non-empty string")
    return value


def _require_sha256(value: JsonValue, *, field: str) -> str:
    digest = _require_string(value, field=field)
    if len(digest) != _SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise AgentLearningArchiveError(f"{field} must be a sha256 digest")
    return digest


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "AGENT_LEARNING_ARCHIVE_SCHEMA_VERSION",
    "AGENT_LEARNING_COLLECTION_SCHEMA_VERSION",
    "AgentLearningArchiveError",
    "AgentLearningArchiveInfo",
    "AgentLearningArchiveMetadata",
    "AgentLearningArchiveRecord",
    "AgentLearningCheckpointArchive",
    "decode_agent_learning_archive",
    "decode_agent_learning_checkpoint_archive",
    "encode_agent_learning_archive",
    "encode_agent_learning_checkpoint_archive",
]
