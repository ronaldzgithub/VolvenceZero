"""Opaque agent learning checkpoint archive tests."""

from __future__ import annotations

import pytest

from volvence_zero.agent.checkpoint_archive import (
    AgentLearningArchiveError,
    decode_agent_learning_checkpoint_archive,
    encode_agent_learning_checkpoint_archive,
)
from volvence_zero.agent.session import AgentLearningCheckpoint


def _checkpoint() -> AgentLearningCheckpoint:
    return AgentLearningCheckpoint(
        checkpoint_id="checkpoint-1",
        joint_loop_state=("joint", 1),
        prediction_state=("prediction", 2),
        credit_state=("credit", 3),
        regime_state=("regime", 4),
        dual_track_gate_state=("dual", 5),
        reflection_state=("reflection", 6),
        policy_fingerprint="policy-fingerprint",
        temporal_fingerprint="temporal-fingerprint",
        memory_fingerprint="memory-fingerprint",
        fingerprint="aggregate-fingerprint",
    )


def test_checkpoint_archive_round_trip_is_opaque_and_integrity_bound() -> None:
    compatibility = (
        ("sense_schema", "ant-sense.ecology-v2"),
        ("latent_dim", "16"),
        ("n_ants", "1"),
    )
    encoded = encode_agent_learning_checkpoint_archive(
        (_checkpoint(),),
        compatibility=compatibility,
    )
    decoded = decode_agent_learning_checkpoint_archive(
        encoded,
        trusted_local_artifact=True,
        expected_compatibility=compatibility,
    )

    assert decoded.checkpoints == (_checkpoint(),)
    assert decoded.metadata.checkpoint_fingerprints == ("aggregate-fingerprint",)


def test_checkpoint_archive_rejects_untrusted_or_mismatched_bytes() -> None:
    compatibility = (("sense_schema", "ant-sense.ecology-v2"),)
    encoded = encode_agent_learning_checkpoint_archive(
        (_checkpoint(),),
        compatibility=compatibility,
    )
    with pytest.raises(PermissionError):
        decode_agent_learning_checkpoint_archive(
            encoded,
            trusted_local_artifact=False,
            expected_compatibility=compatibility,
        )
    with pytest.raises(AgentLearningArchiveError, match="compatibility"):
        decode_agent_learning_checkpoint_archive(
            encoded,
            trusted_local_artifact=True,
            expected_compatibility=(("sense_schema", "ant-sense.v1"),),
        )
    damaged = encoded[:-1] + bytes([encoded[-1] ^ 0x01])
    with pytest.raises(AgentLearningArchiveError, match="digest"):
        decode_agent_learning_checkpoint_archive(
            damaged,
            trusted_local_artifact=True,
            expected_compatibility=compatibility,
        )
