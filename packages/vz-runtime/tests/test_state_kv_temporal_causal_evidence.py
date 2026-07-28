from __future__ import annotations

from dataclasses import replace

from volvence_zero.conditioning_bank_contracts import ConditioningLineageRef
from volvence_zero.state_kv_temporal_causal_evidence import (
    TemporalCausalArm,
    build_temporal_causal_verdict,
)
from volvence_zero.substrate import (
    ResidualActivation,
    ResidualSequenceStep,
    SubstrateSnapshot,
    SurfaceKind,
)
from volvence_zero.temporal import (
    ControllerState,
    TemporalAbstractionSnapshot,
)


def _lineage(*, fingerprint: str) -> ConditioningLineageRef:
    return ConditioningLineageRef(
        session_scope="session",
        selected_bank_set=("personal",),
        bank_fingerprints=(("personal", fingerprint),),
        carrier="prefix_kv",
        delivery_phase="substrate-capture",
    )


def _arm(
    *,
    label: str,
    value: float,
    beta: float,
    lineage: ConditioningLineageRef | None,
    applied: bool,
) -> TemporalCausalArm:
    activation = ResidualActivation(
        layer_index=0,
        activation=(value, value + 0.1),
        step=0,
    )
    step = ResidualSequenceStep(
        step=0,
        token="x",
        feature_surface=(),
        residual_activations=(activation,),
        description="test",
        conditioning_lineage=lineage,
    )
    substrate = SubstrateSnapshot(
        model_id="test",
        is_frozen=True,
        surface_kind=SurfaceKind.RESIDUAL_STREAM,
        token_logits=(),
        feature_surface=(),
        residual_activations=(activation,),
        residual_sequence=(step,),
        unavailable_fields=(),
        description="test",
        conditioning_lineage=lineage,
        personal_conditioning_applied=applied,
    )
    temporal = TemporalAbstractionSnapshot(
        controller_state=ControllerState(
            code=(value, value + 0.1, value + 0.2),
            code_dim=3,
            switch_gate=beta,
            is_switching=False,
            steps_since_switch=1,
        ),
        active_abstract_action="test",
        controller_params_hash="test",
        description="test",
        conditioning_lineage_refs=(lineage,) if lineage is not None else (),
    )
    return TemporalCausalArm(label=label, substrate=substrate, temporal=temporal)


def _passing_arms() -> tuple[TemporalCausalArm, ...]:
    return (
        _arm(
            label="baseline",
            value=0.1,
            beta=0.2,
            lineage=None,
            applied=False,
        ),
        _arm(
            label="correct-state",
            value=0.4,
            beta=0.5,
            lineage=_lineage(fingerprint="a" * 64),
            applied=True,
        ),
        _arm(
            label="wrong-user",
            value=0.7,
            beta=0.8,
            lineage=_lineage(fingerprint="b" * 64),
            applied=True,
        ),
        _arm(
            label="revoked",
            value=0.1,
            beta=0.2,
            lineage=None,
            applied=False,
        ),
    )


def test_temporal_causal_gate_passes_complete_matched_ablation() -> None:
    baseline, correct, wrong, revoked = _passing_arms()

    verdict = build_temporal_causal_verdict(
        baseline=baseline,
        correct_state=correct,
        wrong_user=wrong,
        revoked=revoked,
        artifact_id="c" * 64,
        substrate_fingerprint="model@revision",
        source_text="same prompt",
    )

    assert verdict.gate_state == "pass"
    assert all(claim.state == "pass" for claim in verdict.claims)
    assert verdict.as_json_dict()["schema_version"] == (
        "state-kv-temporal-causal.v1"
    )


def test_temporal_causal_gate_fails_lineage_only_noop() -> None:
    baseline, correct, wrong, revoked = _passing_arms()
    correct = replace(
        correct,
        substrate=replace(
            correct.substrate,
            residual_activations=baseline.substrate.residual_activations,
            residual_sequence=baseline.substrate.residual_sequence,
        ),
        temporal=replace(
            correct.temporal,
            controller_state=baseline.temporal.controller_state,
        ),
    )

    verdict = build_temporal_causal_verdict(
        baseline=baseline,
        correct_state=correct,
        wrong_user=wrong,
        revoked=revoked,
        artifact_id="c" * 64,
        substrate_fingerprint="model@revision",
        source_text="same prompt",
    )

    assert verdict.gate_state == "fail"
    failed = {
        claim.claim for claim in verdict.claims if claim.state == "fail"
    }
    assert "claim_residual_causality" in failed
    assert "claim_temporal_code_causality" in failed


def test_temporal_causal_gate_fails_when_revocation_does_not_restore_baseline() -> None:
    baseline, correct, wrong, revoked = _passing_arms()
    leaked_lineage = _lineage(fingerprint="d" * 64)
    revoked = replace(
        revoked,
        substrate=replace(
            revoked.substrate,
            personal_conditioning_applied=True,
            conditioning_lineage=leaked_lineage,
        ),
    )

    verdict = build_temporal_causal_verdict(
        baseline=baseline,
        correct_state=correct,
        wrong_user=wrong,
        revoked=revoked,
        artifact_id="c" * 64,
        substrate_fingerprint="model@revision",
        source_text="same prompt",
    )

    assert verdict.gate_state == "fail"
    failed = {
        claim.claim for claim in verdict.claims if claim.state == "fail"
    }
    assert "claim_capture_conditioning_attested" in failed
    assert "claim_conditioning_lineage_alignment" in failed
