from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import math

import pytest

from lifeform_domain_emogpt import relationship_action_gate_v2_geometry as geometry
from lifeform_domain_emogpt.relationship_action_contracts import RelationshipAction
from lifeform_domain_emogpt import relationship_action_gate as legacy_gate
from lifeform_domain_emogpt.relationship_action_gate import RelationshipGateAction
from lifeform_domain_emogpt.relationship_action_gate_v2 import (
    RelationshipActionGateV2,
    RelationshipActionGateV2Artifact,
)
from lifeform_domain_emogpt.relationship_action_gate_v2_geometry import (
    RELATIONSHIP_ACTION_GATE_V2_GEOMETRIC_WITNESS_SCOPE,
    RelationshipActionGateV2GeometricDose,
    _DirectionalBoxProblem,
    _solve_l2,
    _solve_linf,
    analyze_relationship_action_gate_v2_geometric_reachability,
    replay_relationship_action_gate_v2_geometric_reachability_receipt,
)
from volvence_zero.dialogue_trace import DialogueExternalOutcomeKind
from volvence_zero.social_cognition import (
    PreferenceActionForecast,
    SocialActionCandidatePrediction,
    SocialActionOutcomeProbability,
    preference_action_forecast_to_payload,
)


_OUTCOMES = (
    DialogueExternalOutcomeKind.HELPED.value,
    DialogueExternalOutcomeKind.FELT_HEARD.value,
    DialogueExternalOutcomeKind.MISSED.value,
    DialogueExternalOutcomeKind.OVER_DIRECTIVE.value,
)


def _candidate(
    action_id: str,
    probabilities: tuple[float, float, float, float],
) -> SocialActionCandidatePrediction:
    return SocialActionCandidatePrediction(
        action_id=action_id,
        outcomes=tuple(
            SocialActionOutcomeProbability(outcome_id, probability)
            for outcome_id, probability in zip(
                _OUTCOMES,
                probabilities,
                strict=True,
            )
        ),
    )


def _forecast(
    suffix: str,
    *,
    confidence: float = 0.8,
    recommended_action_id: str = RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value,
) -> PreferenceActionForecast:
    return PreferenceActionForecast(
        forecast_id=f"relationship-v2-geometry-forecast-{suffix}",
        decision_id=f"relationship-v2-geometry-decision-{suffix}",
        interlocutor_id="primary",
        candidate_predictions=(
            _candidate("stay_present_without_probe", (0.15, 0.55, 0.2, 0.1)),
            _candidate("respect_space_with_return_option", (0.2, 0.2, 0.2, 0.4)),
            _candidate("neutral_noop", (0.25, 0.25, 0.25, 0.25)),
        ),
        recommended_action_id=recommended_action_id,
        confidence=confidence,
        source_record_ids=("geometry-preference-record-1",),
        issued_turn=4,
        evidence=("runtime:bounded-owner-reader",),
        session_scope=f"geometry-user-{suffix}",
    )


def _policy():
    artifact = RelationshipActionGateV2Artifact.create_bootstrap_seed(
        bootstrap_learning_rate=0.25,
        online_learning_rate=0.125,
        max_abs_parameter=4.0,
        bootstrap_source_artifact_id=(f"relationship-product-source-sha256:{'a' * 64}"),
    )
    return RelationshipActionGateV2(artifact=artifact).freeze_for_evaluation()


def _dose(
    *,
    coordinate_caps: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0),
    clearance: float = 2.0**-30,
) -> RelationshipActionGateV2GeometricDose:
    return RelationshipActionGateV2GeometricDose.create(
        absolute_parameter_cap=4.0,
        coordinate_delta_cap=coordinate_caps,
        robust_logit_clearance=clearance,
    )


def test_box_solvers_return_known_minimum_norm_witnesses() -> None:
    problem = _DirectionalBoxProblem(
        direction=(1.0, 2.0, 0.0, 0.0),
        lower_delta=(-10.0, -10.0, -10.0, -10.0),
        upper_delta=(10.0, 10.0, 10.0, 10.0),
    )

    l2 = _solve_l2(problem, 5.0)
    linf = _solve_linf(problem, 5.0)

    assert l2 is not None
    assert linf is not None
    assert l2 == pytest.approx((1.0, 2.0, 0.0, 0.0))
    assert math.sqrt(math.fsum(value * value for value in l2)) == pytest.approx(math.sqrt(5.0))
    assert linf == pytest.approx((5.0 / 3.0, 5.0 / 3.0, 0.0, 0.0))
    assert max(abs(value) for value in linf) == pytest.approx(5.0 / 3.0)


def test_box_solvers_respect_saturation_and_report_infeasible_dose() -> None:
    problem = _DirectionalBoxProblem(
        direction=(1.0, 2.0, 0.0, 0.0),
        lower_delta=(-0.5, -10.0, 0.0, 0.0),
        upper_delta=(0.5, 10.0, 0.0, 0.0),
    )

    l2 = _solve_l2(problem, 4.5)
    linf = _solve_linf(problem, 4.5)

    assert l2 is not None
    assert linf is not None
    assert l2 == pytest.approx((0.5, 2.0, 0.0, 0.0))
    assert linf == pytest.approx((0.5, 2.0, 0.0, 0.0))
    assert _solve_l2(problem, 20.5000000001) is None
    assert _solve_linf(problem, 20.5000000001) is None


def test_box_solvers_scale_tiny_features_without_squaring_underflow() -> None:
    tiny = float.fromhex("0x1p-600")
    problem = _DirectionalBoxProblem(
        direction=(tiny, 0.0, 0.0, 0.0),
        lower_delta=(-1.0, -1.0, -1.0, -1.0),
        upper_delta=(1.0, 1.0, 1.0, 1.0),
    )

    assert _solve_l2(problem, float.fromhex("0x1p-601")) == (0.5, 0.0, 0.0, 0.0)
    assert _solve_linf(problem, float.fromhex("0x1p-601")) == (0.5, 0.0, 0.0, 0.0)


def test_box_solvers_scale_only_movable_coordinates() -> None:
    tiny = float.fromhex("0x1p-600")
    problem = _DirectionalBoxProblem(
        direction=(1.0, tiny, 0.0, 0.0),
        lower_delta=(0.0, -1.0, -1.0, -1.0),
        upper_delta=(0.0, 1.0, 1.0, 1.0),
    )

    assert _solve_l2(problem, float.fromhex("0x1p-601")) == (0.0, 0.5, 0.0, 0.0)
    assert _solve_linf(problem, float.fromhex("0x1p-601")) == (0.0, 0.5, 0.0, 0.0)


def test_box_solvers_do_not_normalize_immovable_coordinate_by_min_subnormal() -> None:
    min_subnormal = float.fromhex("0x0.0000000000001p-1022")
    problem = _DirectionalBoxProblem(
        direction=(1.0, min_subnormal, 0.0, 0.0),
        lower_delta=(0.0, -1.0, -1.0, -1.0),
        upper_delta=(0.0, 1.0, 1.0, 1.0),
    )

    assert _solve_l2(problem, min_subnormal) == (0.0, 1.0, 0.0, 0.0)
    assert _solve_linf(problem, min_subnormal) == (0.0, 1.0, 0.0, 0.0)


def test_exact_endpoint_can_establish_robust_witness_when_continuous_solver_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_solve_l2 = geometry._solve_l2

    def _boundary_only_l2(
        problem: _DirectionalBoxProblem,
        required_progress: float,
    ) -> tuple[float, ...] | None:
        if required_progress > 0.0:
            return None
        return original_solve_l2(problem, required_progress)

    monkeypatch.setattr(geometry, "_solve_l2", _boundary_only_l2)
    receipt = analyze_relationship_action_gate_v2_geometric_reachability(
        frozen_policy=_policy(),
        forecast=_forecast("endpoint-fallback"),
        dose=_dose(),
    )

    assert receipt.l2_result.continuous_boundary_feasible is True
    assert receipt.l2_result.representable_robust_witness_exists is True
    assert receipt.l2_result.robust_witness_gate_action is RelationshipGateAction.STEER


def test_outcome_free_geometry_closes_open_boundary_with_exact_gate_replay() -> None:
    policy = _policy()
    forecast = _forecast("open-boundary")

    receipt = analyze_relationship_action_gate_v2_geometric_reachability(
        frozen_policy=policy,
        forecast=forecast,
        dose=_dose(),
    )

    assert receipt.baseline_logit_hex == (0.0).hex()
    assert receipt.baseline_probability_hex == (0.5).hex()
    assert receipt.baseline_gate_action is RelationshipGateAction.NOOP
    assert receipt.target_gate_action is RelationshipGateAction.STEER
    assert receipt.l2_result.boundary_distance_kind == "infimum"
    assert receipt.linf_result.boundary_distance_kind == "infimum"
    assert receipt.l2_result.continuous_boundary_distance_hex == (0.0).hex()
    assert receipt.linf_result.continuous_boundary_distance_hex == (0.0).hex()
    assert receipt.analytic_boundary_feasible is True
    assert receipt.operational_gate_flip_reachable is True
    assert receipt.robust_gate_flip_witness_exists is True
    assert receipt.l2_result.robust_witness_gate_action is RelationshipGateAction.STEER
    assert receipt.linf_result.robust_witness_gate_action is RelationshipGateAction.STEER
    assert receipt.operational_delivered_action_flip_reachable is True
    assert receipt.robust_delivered_action_flip_witness_exists is True
    assert receipt.logit_sign_action_consistent is True
    assert receipt.floating_threshold_band_detected is False
    assert receipt.to_payload()["witness_scope"] == RELATIONSHIP_ACTION_GATE_V2_GEOMETRIC_WITNESS_SCOPE


def test_gate_flip_is_not_mislabeled_as_delivered_treatment_when_recommendation_is_noop() -> None:
    receipt = analyze_relationship_action_gate_v2_geometric_reachability(
        frozen_policy=_policy(),
        forecast=_forecast(
            "recommended-noop",
            recommended_action_id=RelationshipAction.NEUTRAL_NOOP.value,
        ),
        dose=_dose(),
    )

    assert receipt.robust_gate_flip_witness_exists is True
    assert receipt.baseline_delivered_action_id == RelationshipAction.NEUTRAL_NOOP.value
    assert receipt.target_delivered_action_id == RelationshipAction.NEUTRAL_NOOP.value
    assert receipt.operational_delivered_action_flip_reachable is False
    assert receipt.robust_delivered_action_flip_witness_exists is False
    assert "delivered-action:recommended-noop-no-treatment-divergence" in receipt.reason_codes


def test_boundary_can_be_geometrically_present_while_robust_treatment_is_unreachable() -> None:
    receipt = analyze_relationship_action_gate_v2_geometric_reachability(
        frozen_policy=_policy(),
        forecast=_forecast("zero-direction-dose", confidence=0.5),
        dose=_dose(coordinate_caps=(1.0, 0.0, 0.0, 0.0)),
    )

    assert receipt.baseline_features_hex[0] == (0.0).hex()
    assert receipt.analytic_boundary_feasible is True
    assert receipt.l2_result.continuous_boundary_distance_hex == (0.0).hex()
    assert receipt.linf_result.continuous_boundary_distance_hex == (0.0).hex()
    assert receipt.robust_gate_flip_witness_exists is False
    assert receipt.operational_gate_flip_reachable is False
    assert receipt.operational_delivered_action_flip_reachable is False
    assert receipt.robust_delivered_action_flip_witness_exists is False
    assert receipt.l2_result.robust_witness_actual_delta_hex is None
    assert receipt.linf_result.robust_witness_actual_delta_hex is None


def test_operational_flip_and_protocol_clearance_are_distinct_quantities() -> None:
    receipt = analyze_relationship_action_gate_v2_geometric_reachability(
        frozen_policy=_policy(),
        forecast=_forecast("operational-without-clearance"),
        dose=_dose(
            coordinate_caps=(0.000001, 0.000001, 0.000001, 0.000001),
            clearance=0.001,
        ),
    )

    assert receipt.directional_endpoint_gate_action is RelationshipGateAction.STEER
    assert receipt.operational_gate_flip_reachable is True
    assert receipt.robust_gate_flip_witness_exists is False
    assert receipt.l2_result.representable_robust_witness_exists is False
    assert receipt.linf_result.representable_robust_witness_exists is False


def test_receipt_is_content_addressed_replayable_and_has_zero_learning_authority() -> None:
    policy = _policy()
    forecast = _forecast("receipt")
    receipt = analyze_relationship_action_gate_v2_geometric_reachability(
        frozen_policy=policy,
        forecast=forecast,
        dose=_dose(),
    )
    payload = receipt.to_payload()

    assert receipt.receipt_id == (
        "relationship-action-gate-v2-geometric-receipt-sha256:"
        "f3b8dcd281b4033e70ed199b9e5d533ef1de98462097c3ed595f4307173f3df3"
    )
    assert (
        replay_relationship_action_gate_v2_geometric_reachability_receipt(
            payload=payload,
            frozen_policy=policy,
            forecast=forecast,
        )
        == receipt
    )
    assert payload["receipt_id"] == receipt.receipt_id
    assert receipt.forecast_sha256 == legacy_gate._canonical_sha256(preference_action_forecast_to_payload(forecast))
    for field_name in (
        "outcome_read_count",
        "prediction_error_read_count",
        "credit_read_count",
        "evaluation_read_count",
        "gate_update_count",
        "policy_apply_count",
        "model_forward_count",
        "cuda_call_count",
    ):
        assert payload[field_name] == 0
    for field_name in (
        "credit_achievability_established",
        "effect_authorized",
        "learnable_authorized",
        "steerable_authorized",
        "campaign_execution_authorized",
    ):
        assert payload[field_name] is False

    tampered = {**payload, "outcome_read_count": 1}
    with pytest.raises(ValueError, match="receipt replay drifted"):
        replay_relationship_action_gate_v2_geometric_reachability_receipt(
            payload=tampered,
            frozen_policy=policy,
            forecast=forecast,
        )


def test_geometric_contract_rejects_noncanonical_values_and_incoherent_receipts() -> None:
    with pytest.raises(TypeError, match="absolute_parameter_cap"):
        RelationshipActionGateV2GeometricDose.create(
            absolute_parameter_cap=True,
            coordinate_delta_cap=(1.0, 1.0, 1.0, 1.0),
            robust_logit_clearance=0.01,
        )
    with pytest.raises(TypeError, match="coordinate_delta_cap"):
        RelationshipActionGateV2GeometricDose.create(
            absolute_parameter_cap=4.0,
            coordinate_delta_cap=(1.0, 1.0, 1.0, 1),
            robust_logit_clearance=0.01,
        )
    with pytest.raises(ValueError, match="finite floats"):
        RelationshipActionGateV2GeometricDose.create(
            absolute_parameter_cap=4.0,
            coordinate_delta_cap=(1.0, 1.0, 1.0, 1.0),
            robust_logit_clearance=float("nan"),
        )
    with pytest.raises(ValueError, match="floating threshold band"):
        RelationshipActionGateV2GeometricDose.create(
            absolute_parameter_cap=4.0,
            coordinate_delta_cap=(1.0, 1.0, 1.0, 1.0),
            robust_logit_clearance=float.fromhex("0x0.0000000000001p-1022"),
        )
    with pytest.raises(ValueError, match="canonical finite float hex"):
        RelationshipActionGateV2GeometricDose.from_payload(
            {
                **_dose().to_payload(),
                "robust_logit_clearance_hex": "-0x0.0p+0",
            }
        )

    receipt = analyze_relationship_action_gate_v2_geometric_reachability(
        frozen_policy=_policy(),
        forecast=_forecast("coherence"),
        dose=_dose(),
    )
    with pytest.raises(ValueError, match="analytic boundary closure"):
        replace(receipt, analytic_boundary_feasible=False)
    with pytest.raises(ValueError, match="distance differs from its actual delta"):
        replace(
            receipt.l2_result,
            robust_witness_distance_upper_bound_hex=(1.0).hex(),
        )
    forged_l2 = replace(
        receipt.l2_result,
        robust_witness_logit_hex=(0.0).hex(),
    )
    with pytest.raises(ValueError, match="exact clearance replay"):
        replace(receipt, l2_result=forged_l2)
    forged_cap_hits = replace(
        receipt.l2_result,
        robust_witness_cap_hit_coordinate_indices=(0,),
    )
    with pytest.raises(ValueError, match="cap-hit indices"):
        replace(receipt, l2_result=forged_cap_hits)
    with pytest.raises(ValueError, match="endpoint probability differs"):
        replace(receipt, directional_endpoint_probability_hex=(0.5).hex())
    with pytest.raises(FrozenInstanceError):
        receipt.reason_codes = ()  # type: ignore[misc]
