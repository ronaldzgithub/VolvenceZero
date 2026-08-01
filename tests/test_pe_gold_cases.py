from __future__ import annotations

from dataclasses import asdict

import pytest

from volvence_zero.agent.gate1_pe_mechanism_evidence import (
    evaluate_evaluation_decoupling,
)
from volvence_zero.prediction import (
    ActualOutcome,
    PredictedOutcome,
    PredictionErrorModule,
)
from volvence_zero.prediction.torch_lss import (
    bridge_typed_runtime_pe_to_lss,
)
from volvence_zero.runtime import WiringLevel


@pytest.mark.parametrize(
    ("surface_kind", "predicted", "actual", "expected_runtime_pe"),
    (
        ("numeric", (0.25,), 0.75, (0.5,)),
        ("probability", (0.2,), 1.0, (0.8,)),
        ("enum", (0.1, 0.7, 0.2), 2, (-0.1, -0.7, 0.8)),
        (
            "vector",
            (0.1, 0.4, 0.9),
            (0.2, 0.0, 0.6),
            (0.1, -0.4, -0.3),
        ),
        (
            "distribution",
            (0.2, 0.5, 0.3),
            (0.1, 0.2, 0.7),
            (-0.1, -0.3, 0.4),
        ),
    ),
)
def test_gate1_typed_gold_cases_match_true_lss(
    surface_kind,
    predicted,
    actual,
    expected_runtime_pe,
) -> None:
    first = bridge_typed_runtime_pe_to_lss(
        surface_kind=surface_kind,
        predicted=predicted,
        actual=actual,
    )
    second = bridge_typed_runtime_pe_to_lss(
        surface_kind=surface_kind,
        predicted=predicted,
        actual=actual,
    )

    assert asdict(first) == asdict(second)
    assert first.runtime_signed_pe == pytest.approx(
        expected_runtime_pe,
        abs=1e-12,
    )
    assert tuple(
        runtime_value + lss_value
        for runtime_value, lss_value in zip(
            first.runtime_signed_pe,
            first.lss,
            strict=True,
        )
    ) == pytest.approx(
        (0.0,) * len(expected_runtime_pe),
        abs=1e-9,
    )
    assert first.bridge_passed
    assert first.max_abs_bridge_error <= 1e-9
    assert len(first.component_decomposition) == len(expected_runtime_pe)


def test_gate1_owner_numeric_gold_case_is_exact() -> None:
    module = PredictionErrorModule(wiring_level=WiringLevel.ACTIVE)
    predicted = PredictedOutcome(
        source_turn_index=1,
        target_turn_index=2,
        predicted_task_progress=0.5,
        predicted_relationship_delta=0.5,
        predicted_regime_stability=0.5,
        predicted_action_payoff=0.5,
        confidence=0.0,
        description="Gate 1 owner gold prediction",
    )
    actual = ActualOutcome(
        observed_turn_index=2,
        task_progress=0.7,
        relationship_delta=0.3,
        regime_stability=0.9,
        action_payoff=0.1,
        description="Gate 1 owner gold actual",
    )

    error = module.compute_prediction_error(
        predicted=predicted,
        actual_outcome=actual,
    )

    assert error.task_error == pytest.approx(0.2)
    assert error.relationship_error == pytest.approx(-0.2)
    assert error.regime_error == pytest.approx(0.4)
    assert error.action_error == pytest.approx(-0.4)
    assert error.magnitude == pytest.approx(1.2)
    assert error.signed_reward == pytest.approx(0.0)
    assert "weighted_axes[" in error.description


@pytest.mark.parametrize(
    ("surface_kind", "predicted", "actual", "message"),
    (
        (
            "probability",
            (1.0,),
            1.0,
            "prediction in \\(0, 1\\)",
        ),
        (
            "enum",
            (0.2, 0.8),
            2,
            "outside the prediction category range",
        ),
        (
            "vector",
            (0.1, 0.2),
            (0.1,),
            "dimensions must match",
        ),
        (
            "distribution",
            (0.2, 0.8),
            (0.2, 0.2),
            "normalized probability simplex",
        ),
    ),
)
def test_gate1_typed_gold_cases_fail_loudly(
    surface_kind,
    predicted,
    actual,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        bridge_typed_runtime_pe_to_lss(
            surface_kind=surface_kind,
            predicted=predicted,
            actual=actual,
        )


def test_gate1_evaluation_decoupled_trace_is_byte_invariant(
    monkeypatch,
) -> None:
    monkeypatch.delenv("VZ_PE_EVALUATION_DECOUPLED", raising=False)

    report = evaluate_evaluation_decoupling()

    assert report.passed
    assert report.byte_invariant
    assert (
        report.active_payload_sha256_a
        == report.active_payload_sha256_b
    )
    assert report.shadow_evaluation_sensitive
    assert report.active_shadow_different
    assert report.pe_load_bearing
    assert "VZ_PE_EVALUATION_DECOUPLED" not in __import__("os").environ
