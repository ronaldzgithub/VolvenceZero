"""Canonical agent learning archive contract tests."""

from __future__ import annotations

import hashlib

import pytest

from volvence_zero.agent.checkpoint_archive import (
    AgentLearningArchiveError,
    decode_agent_learning_archive,
    decode_agent_learning_checkpoint_archive,
    encode_agent_learning_archive,
    encode_agent_learning_checkpoint_archive,
)
from volvence_zero.owner_hydration import OwnerPersistenceSnapshot


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _single_archive(*, checkpoint_id: str = "checkpoint-1") -> bytes:
    return encode_agent_learning_archive(
        checkpoint_id=checkpoint_id,
        owner_snapshots=(
            OwnerPersistenceSnapshot(
                owner_name="owner.beta",
                schema_version=2,
                payload={"weights": [1.25, -0.5], "count": 3},
                description="beta",
            ),
            OwnerPersistenceSnapshot(
                owner_name="owner.alpha",
                schema_version=1,
                payload={"enabled": True},
                description="alpha",
            ),
        ),
        policy_fingerprint=_digest("policy"),
        temporal_fingerprint=_digest("temporal"),
        memory_fingerprint=_digest("memory"),
    )


def test_single_archive_is_deterministic_canonical_json() -> None:
    first = _single_archive()
    second = _single_archive()

    assert first == second
    assert first.startswith(b'{"archive_sha256":')
    assert b"pickle" not in first

    decoded = decode_agent_learning_archive(first)
    assert decoded.info.checkpoint_id == "checkpoint-1"
    assert decoded.info.owner_versions == (
        ("owner.alpha", 1),
        ("owner.beta", 2),
    )
    assert tuple(item.owner_name for item in decoded.owner_snapshots) == (
        "owner.alpha",
        "owner.beta",
    )
    with pytest.raises(TypeError):
        decoded.owner_snapshots[0].payload["enabled"] = False  # type: ignore[index]


def test_collection_round_trip_binds_compatibility_and_nested_archives() -> None:
    compatibility = (
        ("sense_schema", "ant-sense.ecology-v2"),
        ("latent_dim", "16"),
        ("n_ants", "2"),
    )
    archives = (
        _single_archive(checkpoint_id="body-0"),
        _single_archive(checkpoint_id="body-1"),
    )
    encoded = encode_agent_learning_checkpoint_archive(
        archives,
        compatibility=compatibility,
    )
    decoded = decode_agent_learning_checkpoint_archive(
        encoded,
        expected_compatibility=compatibility,
    )

    assert decoded.checkpoint_archives == archives
    assert decoded.metadata.checkpoint_count == 2
    assert decoded.metadata.checkpoint_fingerprints == tuple(
        decode_agent_learning_archive(item).info.state_fingerprint
        for item in archives
    )
    with pytest.raises(AgentLearningArchiveError, match="ids must be unique"):
        encode_agent_learning_checkpoint_archive(
            (archives[0], archives[0]),
            compatibility=compatibility,
        )


def test_archive_rejects_mismatch_tampering_and_noncanonical_json() -> None:
    compatibility = (("sense_schema", "ant-sense.ecology-v2"),)
    encoded = encode_agent_learning_checkpoint_archive(
        (_single_archive(),),
        compatibility=compatibility,
    )
    with pytest.raises(AgentLearningArchiveError, match="compatibility"):
        decode_agent_learning_checkpoint_archive(
            encoded,
            expected_compatibility=(("sense_schema", "ant-sense.v1"),),
        )

    damaged = encoded.replace(
        b"ant-sense.ecology-v2",
        b"ant-sense.ecology-x2",
        1,
    )
    with pytest.raises(AgentLearningArchiveError, match="digest"):
        decode_agent_learning_checkpoint_archive(
            damaged,
            expected_compatibility=compatibility,
        )

    with pytest.raises(AgentLearningArchiveError, match="canonical"):
        decode_agent_learning_archive(b" " + _single_archive())
