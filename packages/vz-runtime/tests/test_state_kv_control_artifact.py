from __future__ import annotations

import pytest

from volvence_zero.credit import (
    GateDecision,
    ModificationGate,
    ModificationProposal,
)
from volvence_zero.evaluation import EvaluationSnapshot
from volvence_zero.state_kv_control_artifact import (
    CONTROL_BASIS_TARGET,
    active_control_basis_hash,
    install_control_basis_artifact,
)
from volvence_zero.substrate import (
    FULL_CODE_SINUSOID_CONTROL_BASIS_MODE,
    ControlBasisArtifact,
    build_builtin_transformers_runtime,
    build_sinusoid_control_basis,
    control_basis_fingerprint,
)


def _artifact(runtime) -> ControlBasisArtifact:
    basis = build_sinusoid_control_basis(
        hidden_size=runtime.hidden_size,
        rank=runtime.control_basis_rank,
    )
    return ControlBasisArtifact(
        model_id=runtime.model_id,
        hidden_size=runtime.hidden_size,
        basis=basis,
        layer_indices=runtime.hook_layer_indices,
        layer_gains=tuple(1.0 for _ in runtime.hook_layer_indices),
        training_mode=FULL_CODE_SINUSOID_CONTROL_BASIS_MODE,
        source_fingerprint=control_basis_fingerprint(basis),
        sample_count=8,
        description="Matched full-code diagnostic artifact.",
    )


def _snapshot() -> EvaluationSnapshot:
    return EvaluationSnapshot(
        turn_scores=(),
        session_scores=(),
        alerts=(),
        description="clean offline validation snapshot",
    )


def _proposal(runtime, artifact, *, validation_delta: float = 0.06):
    return ModificationProposal(
        target=CONTROL_BASIS_TARGET,
        desired_gate=ModificationGate.OFFLINE,
        old_value_hash=active_control_basis_hash(runtime),
        new_value_hash=artifact.artifact_id,
        justification="Matched full-code evidence passed its preregistered gate.",
        validation_delta=validation_delta,
        capacity_cost=0.0,
        rollback_evidence="Restore previous basis provenance from freeze manifest.",
    )


def test_offline_gate_installs_matching_control_basis_artifact() -> None:
    runtime = build_builtin_transformers_runtime(activation_width=48)
    artifact = _artifact(runtime)

    result = install_control_basis_artifact(
        runtime=runtime,
        artifact=artifact,
        proposal=_proposal(runtime, artifact),
        evaluation_snapshot=_snapshot(),
    )

    assert result.decision is GateDecision.ALLOW
    assert result.installed is True
    assert runtime.control_basis_provenance == artifact.artifact_id


def test_blocked_gate_does_not_mutate_runtime() -> None:
    runtime = build_builtin_transformers_runtime(activation_width=48)
    artifact = _artifact(runtime)
    before = runtime.control_basis_provenance

    result = install_control_basis_artifact(
        runtime=runtime,
        artifact=artifact,
        proposal=_proposal(runtime, artifact, validation_delta=0.0),
        evaluation_snapshot=_snapshot(),
    )

    assert result.decision is GateDecision.BLOCK
    assert result.installed is False
    assert runtime.control_basis_provenance == before


def test_artifact_hash_mismatch_fails_loudly() -> None:
    runtime = build_builtin_transformers_runtime(activation_width=48)
    artifact = _artifact(runtime)
    proposal = _proposal(runtime, artifact)
    broken = ModificationProposal(
        target=proposal.target,
        desired_gate=proposal.desired_gate,
        old_value_hash=proposal.old_value_hash,
        new_value_hash="wrong",
        justification=proposal.justification,
        validation_delta=proposal.validation_delta,
        rollback_evidence=proposal.rollback_evidence,
    )

    with pytest.raises(ValueError, match="new_value_hash"):
        install_control_basis_artifact(
            runtime=runtime,
            artifact=artifact,
            proposal=broken,
            evaluation_snapshot=_snapshot(),
        )


def test_rank_expansion_cannot_bypass_human_review() -> None:
    runtime = build_builtin_transformers_runtime(activation_width=48)
    basis = build_sinusoid_control_basis(
        hidden_size=runtime.hidden_size,
        rank=8,
    )
    artifact = ControlBasisArtifact(
        model_id=runtime.model_id,
        hidden_size=runtime.hidden_size,
        basis=basis,
        layer_indices=runtime.hook_layer_indices,
        layer_gains=tuple(1.0 for _ in runtime.hook_layer_indices),
        training_mode=FULL_CODE_SINUSOID_CONTROL_BASIS_MODE,
        source_fingerprint=control_basis_fingerprint(basis),
        sample_count=8,
        description="Capacity expansion candidate.",
    )

    with pytest.raises(ValueError, match="human-review"):
        install_control_basis_artifact(
            runtime=runtime,
            artifact=artifact,
            proposal=_proposal(runtime, artifact),
            evaluation_snapshot=_snapshot(),
        )
