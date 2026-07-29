"""ModificationGate-protected installation of substrate control artifacts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from volvence_zero.credit import (
    GateDecision,
    ModificationGate,
    ModificationProposal,
    evaluate_gate,
)
from volvence_zero.evaluation import EvaluationSnapshot
from volvence_zero.substrate import (
    ControlBasisArtifact,
    OpenWeightResidualRuntime,
)

CONTROL_BASIS_TARGET = "substrate.rare_heavy.control_basis"


def active_control_basis_hash(runtime: OpenWeightResidualRuntime) -> str:
    provenance = runtime.control_basis_provenance
    return hashlib.sha256(provenance.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ControlBasisInstallResult:
    decision: GateDecision
    artifact_id: str
    previous_provenance: str
    active_provenance: str
    installed: bool


def install_control_basis_artifact(
    *,
    runtime: OpenWeightResidualRuntime,
    artifact: ControlBasisArtifact,
    proposal: ModificationProposal,
    evaluation_snapshot: EvaluationSnapshot,
) -> ControlBasisInstallResult:
    """Validate and install one rare-heavy artifact after an OFFLINE gate."""

    previous_provenance = runtime.control_basis_provenance
    if proposal.target != CONTROL_BASIS_TARGET:
        raise ValueError(
            f"control-basis proposal target must be {CONTROL_BASIS_TARGET!r}"
        )
    if proposal.desired_gate is not ModificationGate.OFFLINE:
        raise ValueError(
            "control-basis installation requires ModificationGate.OFFLINE"
        )
    if proposal.old_value_hash != active_control_basis_hash(runtime):
        raise ValueError(
            "control-basis proposal old_value_hash does not match active basis"
        )
    if proposal.new_value_hash != artifact.artifact_id:
        raise ValueError(
            "control-basis proposal new_value_hash does not match artifact"
        )
    if artifact.model_id != runtime.model_id:
        raise ValueError(
            "control-basis artifact model_id does not match substrate runtime"
        )
    if artifact.hidden_size != runtime.hidden_size:
        raise ValueError(
            "control-basis artifact hidden_size does not match substrate runtime"
        )
    if artifact.rank != runtime.control_basis_rank:
        raise ValueError(
            "control-basis rank expansion is a substrate.capacity change "
            "and requires the human-review import path; OFFLINE auto-install "
            f"cannot change rank {runtime.control_basis_rank} to {artifact.rank}"
        )
    unavailable = sorted(
        set(artifact.layer_indices) - set(runtime.hook_layer_indices)
    )
    if unavailable:
        raise ValueError(
            f"control-basis artifact targets unavailable layers: {unavailable}"
        )

    decision = evaluate_gate(
        proposal=proposal,
        evaluation_snapshot=evaluation_snapshot,
    )
    if decision is GateDecision.ALLOW:
        runtime.install_control_basis(
            basis=artifact.basis,
            provenance=artifact.artifact_id,
            layer_indices=artifact.layer_indices,
            layer_gains=artifact.layer_gains,
        )
    return ControlBasisInstallResult(
        decision=decision,
        artifact_id=artifact.artifact_id,
        previous_provenance=previous_provenance,
        active_provenance=runtime.control_basis_provenance,
        installed=decision is GateDecision.ALLOW,
    )


__all__ = [
    "CONTROL_BASIS_TARGET",
    "ControlBasisInstallResult",
    "active_control_basis_hash",
    "install_control_basis_artifact",
]
