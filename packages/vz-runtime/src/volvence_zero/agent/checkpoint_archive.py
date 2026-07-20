"""Versioned opaque archives for owner-exported learning checkpoints.

The payload uses Python pickle because ``AgentLearningCheckpoint`` deliberately
contains owner-private value types.  Decoding is therefore restricted to
explicitly trusted local artifacts; no network/API surface may forward
untrusted bytes here.  Integrity and compatibility are verified before
unpickling, and owner fingerprints are verified again after restoration by
``AgentSessionRunner.restore_learning_checkpoint``.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from volvence_zero.agent.session import AgentLearningCheckpoint


AGENT_LEARNING_ARCHIVE_SCHEMA_VERSION = "agent-learning-checkpoint.v1"
_MAGIC = b"VZ-AGENT-LEARNING\x00\x01"
_HEADER_LENGTH_BYTES = 8


class AgentLearningArchiveError(ValueError):
    """A checkpoint archive failed schema, integrity or compatibility checks."""


@dataclass(frozen=True)
class AgentLearningArchiveMetadata:
    schema_version: str
    checkpoint_count: int
    checkpoint_fingerprints: tuple[str, ...]
    compatibility: tuple[tuple[str, str], ...]
    payload_sha256: str


@dataclass(frozen=True)
class AgentLearningCheckpointArchive:
    metadata: AgentLearningArchiveMetadata
    checkpoints: tuple["AgentLearningCheckpoint", ...]


def encode_agent_learning_checkpoint_archive(
    checkpoints: tuple["AgentLearningCheckpoint", ...],
    *,
    compatibility: tuple[tuple[str, str], ...],
) -> bytes:
    """Encode owner-exported values without exposing their internal shape."""

    from volvence_zero.agent.session import AgentLearningCheckpoint

    if not checkpoints:
        raise AgentLearningArchiveError("checkpoint archive requires at least one checkpoint")
    if not all(isinstance(item, AgentLearningCheckpoint) for item in checkpoints):
        raise TypeError("all archive values must be AgentLearningCheckpoint")
    keys = tuple(key for key, _ in compatibility)
    if len(set(keys)) != len(keys) or any(not key for key in keys):
        raise AgentLearningArchiveError("archive compatibility keys must be unique and non-empty")
    normalized_compatibility = tuple((str(key), str(value)) for key, value in compatibility)
    payload = pickle.dumps(checkpoints, protocol=5)
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    header = {
        "schema_version": AGENT_LEARNING_ARCHIVE_SCHEMA_VERSION,
        "checkpoint_count": len(checkpoints),
        "checkpoint_fingerprints": [item.fingerprint for item in checkpoints],
        "compatibility": [list(item) for item in normalized_compatibility],
        "payload_sha256": payload_sha256,
    }
    header_bytes = json.dumps(
        header,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _MAGIC + len(header_bytes).to_bytes(_HEADER_LENGTH_BYTES, "big") + header_bytes + payload


def decode_agent_learning_checkpoint_archive(
    archive_bytes: bytes,
    *,
    trusted_local_artifact: bool,
    expected_compatibility: tuple[tuple[str, str], ...],
) -> AgentLearningCheckpointArchive:
    """Decode an integrity-bound local archive.

    ``trusted_local_artifact`` is intentionally mandatory.  A sha256 detects
    corruption but does not authenticate a malicious pickle.
    """

    from volvence_zero.agent.session import AgentLearningCheckpoint

    if not trusted_local_artifact:
        raise PermissionError("checkpoint pickle decoding is restricted to trusted local artifacts")
    prefix_length = len(_MAGIC) + _HEADER_LENGTH_BYTES
    if len(archive_bytes) < prefix_length or not archive_bytes.startswith(_MAGIC):
        raise AgentLearningArchiveError("invalid checkpoint archive magic")
    header_start = len(_MAGIC) + _HEADER_LENGTH_BYTES
    header_length = int.from_bytes(
        archive_bytes[len(_MAGIC) : header_start],
        "big",
    )
    header_end = header_start + header_length
    if header_length <= 0 or header_end > len(archive_bytes):
        raise AgentLearningArchiveError("invalid checkpoint header length")
    header = json.loads(archive_bytes[header_start:header_end].decode("utf-8"))
    if not isinstance(header, dict):
        raise AgentLearningArchiveError("checkpoint archive header must be an object")
    if header.get("schema_version") != AGENT_LEARNING_ARCHIVE_SCHEMA_VERSION:
        raise AgentLearningArchiveError(f"unsupported checkpoint archive schema: {header.get('schema_version')!r}")
    raw_compatibility = header.get("compatibility")
    if not isinstance(raw_compatibility, list):
        raise AgentLearningArchiveError("checkpoint compatibility metadata must be a list")
    compatibility = tuple(
        (str(item[0]), str(item[1])) for item in raw_compatibility if isinstance(item, list) and len(item) == 2
    )
    if len(compatibility) != len(raw_compatibility):
        raise AgentLearningArchiveError("checkpoint compatibility entries must be key/value pairs")
    if compatibility != expected_compatibility:
        raise AgentLearningArchiveError(
            f"checkpoint compatibility mismatch: expected={expected_compatibility!r}, actual={compatibility!r}"
        )
    payload = archive_bytes[header_end:]
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    expected_sha256 = str(header.get("payload_sha256", ""))
    if actual_sha256 != expected_sha256:
        raise AgentLearningArchiveError("checkpoint archive payload digest mismatch")
    checkpoints = pickle.loads(payload)
    if not isinstance(checkpoints, tuple) or not all(isinstance(item, AgentLearningCheckpoint) for item in checkpoints):
        raise AgentLearningArchiveError("checkpoint payload contains unexpected value types")
    expected_count = int(header.get("checkpoint_count", -1))
    if len(checkpoints) != expected_count:
        raise AgentLearningArchiveError("checkpoint count does not match archive header")
    raw_fingerprints = header.get("checkpoint_fingerprints")
    if not isinstance(raw_fingerprints, list):
        raise AgentLearningArchiveError("checkpoint fingerprints must be a list")
    fingerprints = tuple(str(item) for item in raw_fingerprints)
    actual_fingerprints = tuple(item.fingerprint for item in checkpoints)
    if fingerprints != actual_fingerprints:
        raise AgentLearningArchiveError("checkpoint fingerprints do not match archive payload")
    metadata = AgentLearningArchiveMetadata(
        schema_version=AGENT_LEARNING_ARCHIVE_SCHEMA_VERSION,
        checkpoint_count=expected_count,
        checkpoint_fingerprints=fingerprints,
        compatibility=compatibility,
        payload_sha256=actual_sha256,
    )
    return AgentLearningCheckpointArchive(
        metadata=metadata,
        checkpoints=checkpoints,
    )


__all__ = [
    "AGENT_LEARNING_ARCHIVE_SCHEMA_VERSION",
    "AgentLearningArchiveError",
    "AgentLearningArchiveMetadata",
    "AgentLearningCheckpointArchive",
    "decode_agent_learning_checkpoint_archive",
    "encode_agent_learning_checkpoint_archive",
]
