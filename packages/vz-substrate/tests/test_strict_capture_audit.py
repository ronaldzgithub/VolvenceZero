"""Contract tests for the substrate-owned strict capture audit summary."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from volvence_zero.substrate.adapter import (
    FeatureSignal,
    ResidualActivation,
    ResidualSequenceStep,
)
from volvence_zero.substrate.residual_contracts import OpenWeightRuntimeCapture
from volvence_zero.substrate.strict_capture_audit import (
    STRICT_CAPTURE_AUDIT_SCHEMA_VERSION,
    StrictCaptureAuditSummary,
    audit_strict_capture,
)


_EXPECTED_FEATURE_VALUES = (
    ("hook_layer_coverage", 1.0),
    ("hook_fire_rate", 1.0),
    ("token_step_coverage", 1.0),
    ("residual_sequence_present", 1.0),
    ("fallback_active", 0.0),
)


def _feature_surface() -> tuple[FeatureSignal, ...]:
    return tuple(
        FeatureSignal(
            name=name,
            values=(value,),
            source="strict-capture-audit-test",
        )
        for name, value in _EXPECTED_FEATURE_VALUES
    )


def _step(
    step: int,
    values: tuple[float, ...],
    *,
    layer_index: int = 20,
    activation_step: int | None = None,
) -> ResidualSequenceStep:
    resolved_activation_step = step if activation_step is None else activation_step
    return ResidualSequenceStep(
        step=step,
        token=f"token-{step}",
        feature_surface=(),
        residual_activations=(
            ResidualActivation(
                layer_index=layer_index,
                activation=values,
                step=resolved_activation_step,
            ),
        ),
        description="strict capture step",
    )


def _capture(
    *,
    sequence: tuple[ResidualSequenceStep, ...] | None = None,
    token_logits: tuple[float, ...] = (0.125, -1.5),
    feature_surface: tuple[FeatureSignal, ...] | None = None,
    residual_activations: tuple[ResidualActivation, ...] | None = None,
) -> OpenWeightRuntimeCapture:
    resolved_sequence = sequence or (
        _step(0, (1.0, -2.0, 0.5)),
        _step(1, (3.25, 4.5, -0.0)),
    )
    return OpenWeightRuntimeCapture(
        token_logits=token_logits,
        feature_surface=(_feature_surface() if feature_surface is None else feature_surface),
        residual_activations=(
            resolved_sequence[-1].residual_activations if residual_activations is None else residual_activations
        ),
        residual_sequence=resolved_sequence,
        description="strict owner capture",
    )


def _audit(capture: OpenWeightRuntimeCapture) -> StrictCaptureAuditSummary:
    return audit_strict_capture(
        capture,
        expected_layer_index=20,
        expected_activation_width=3,
    )


def test_happy_summary_is_bounded_and_hash_deterministic() -> None:
    capture = _capture()

    first = _audit(capture)
    second = _audit(capture)

    assert first == second
    assert first.schema_version == STRICT_CAPTURE_AUDIT_SCHEMA_VERSION
    assert first.residual_sequence_length == 2
    assert first.residual_step_continuity_exact is True
    assert first.capture_layer_exact is True
    assert first.capture_width_exact is True
    assert first.residual_activation_value_count == 6
    assert first.finite_residual_activation_value_count == 6
    assert first.capture_values_all_finite is True
    assert first.residual_sequence_sha256 == "b179ba070aeadcc59330601abbabe2664a285cfb4abc783c99243cd134dfb9cb"
    assert first.latest_activation_width == 3
    assert first.latest_activation_sha256 == "d3a11970929fca2ab11673c7f57ab9539b0953d424e3cf253f1c3e6a8fb06863"
    assert first.latest_matches_sequence_exact is True
    assert first.top_logit_count == 2
    assert first.top_logits_finite_nonempty is True
    assert first.top_logits_sha256 == "6538eff31ec923e39c8e4ae4e7b0a6af10182c17dc50ddfe3e91fc9e964235e2"
    assert first.selected_feature_values == _EXPECTED_FEATURE_VALUES
    assert first.description_sha256 == "46109cff6dd61bfbc41de20da382613f87087ae14109183238f8734930a36c9f"

    payload = first.to_payload()
    assert set(payload) == {
        "schema_version",
        "residual_sequence_length",
        "residual_step_continuity_exact",
        "capture_layer_exact",
        "capture_width_exact",
        "residual_activation_value_count",
        "finite_residual_activation_value_count",
        "capture_values_all_finite",
        "residual_sequence_sha256",
        "latest_activation_width",
        "latest_activation_sha256",
        "latest_matches_sequence_exact",
        "top_logit_count",
        "top_logits_finite_nonempty",
        "top_logits_sha256",
        "selected_feature_values",
        "description_sha256",
    }
    assert "residual_sequence" not in payload
    assert "residual_activations" not in payload
    assert payload["selected_feature_values"] == dict(_EXPECTED_FEATURE_VALUES)

    changed_sequence = (
        capture.residual_sequence[0],
        _step(1, (3.25, 4.5, 0.25)),
    )
    changed = _audit(_capture(sequence=changed_sequence))
    assert changed.residual_sequence_sha256 != first.residual_sequence_sha256
    assert changed.latest_activation_sha256 != first.latest_activation_sha256

    mismatched_latest = _audit(
        _capture(
            residual_activations=capture.residual_sequence[0].residual_activations,
        )
    )
    assert mismatched_latest.latest_matches_sequence_exact is False
    assert mismatched_latest.latest_activation_sha256 != first.latest_activation_sha256


def test_summary_is_frozen_and_rejects_mutable_or_drifted_feature_shape() -> None:
    summary = _audit(_capture())

    with pytest.raises(FrozenInstanceError):
        summary.capture_width_exact = False
    with pytest.raises(TypeError, match="exact frozen pairs"):
        replace(summary, selected_feature_values=list(summary.selected_feature_values))
    with pytest.raises(ValueError, match="feature set/order drift"):
        replace(
            summary,
            selected_feature_values=tuple(reversed(summary.selected_feature_values)),
        )
    with pytest.raises(ValueError, match="finite residual count exceeds"):
        replace(summary, finite_residual_activation_value_count=7)
    with pytest.raises(ValueError, match="finite residual flag/count drift"):
        replace(summary, capture_values_all_finite=False)
    with pytest.raises(ValueError, match="empty logits"):
        replace(summary, top_logit_count=0)

    payload = summary.to_payload()
    selected = payload["selected_feature_values"]
    assert type(selected) is dict
    selected["hook_layer_coverage"] = 0.0
    assert dict(summary.selected_feature_values)["hook_layer_coverage"] == 1.0


def test_duplicate_missing_malformed_and_nonfinite_features_publish_none() -> None:
    feature_surface = (
        FeatureSignal(
            name="hook_layer_coverage",
            values=(1.0,),
            source="duplicate-a",
        ),
        FeatureSignal(
            name="hook_layer_coverage",
            values=(0.5,),
            source="duplicate-b",
        ),
        FeatureSignal(
            name="hook_fire_rate",
            values=(0.75,),
            source="valid",
        ),
        FeatureSignal(
            name="token_step_coverage",
            values=(1.0, 1.0),
            source="wrong-cardinality",
        ),
        FeatureSignal(
            name="residual_sequence_present",
            values=(float("nan"),),
            source="nonfinite",
        ),
        FeatureSignal(
            name="unrelated_owner_signal",
            values=(42.0,),
            source="ignored",
        ),
    )

    summary = _audit(_capture(feature_surface=feature_surface))

    assert summary.selected_feature_values == (
        ("hook_layer_coverage", None),
        ("hook_fire_rate", 0.75),
        ("token_step_coverage", None),
        ("residual_sequence_present", None),
        ("fallback_active", None),
    )


def test_nonfinite_residuals_and_logits_are_counted_without_becoming_passes() -> None:
    sequence = (_step(0, (float("nan"), float("inf"), 3.0)),)

    summary = _audit(
        _capture(
            sequence=sequence,
            token_logits=(0.0, float("-inf")),
        )
    )

    assert summary.residual_activation_value_count == 3
    assert summary.finite_residual_activation_value_count == 1
    assert summary.capture_values_all_finite is False
    assert summary.top_logit_count == 2
    assert summary.top_logits_finite_nonempty is False
    assert len(summary.residual_sequence_sha256) == 64
    assert len(summary.top_logits_sha256) == 64


def test_geometry_flags_separate_step_layer_and_width_drift() -> None:
    baseline = _capture()
    first_step, second_step = baseline.residual_sequence

    discontinuous = _audit(_capture(sequence=(first_step, replace(second_step, step=7))))
    assert discontinuous.residual_step_continuity_exact is False
    assert discontinuous.capture_layer_exact is True
    assert discontinuous.capture_width_exact is True

    wrong_activation_step = _audit(_capture(sequence=(first_step, _step(1, (3.25, 4.5, -0.0), activation_step=7))))
    assert wrong_activation_step.residual_step_continuity_exact is True
    assert wrong_activation_step.capture_layer_exact is False
    assert wrong_activation_step.capture_width_exact is True
    assert wrong_activation_step.residual_sequence_sha256 != _audit(baseline).residual_sequence_sha256

    wrong_layer = _audit(_capture(sequence=(first_step, _step(1, (3.25, 4.5, -0.0), layer_index=21))))
    assert wrong_layer.capture_layer_exact is False
    assert wrong_layer.capture_width_exact is True

    wrong_width = _audit(_capture(sequence=(first_step, _step(1, (3.25, 4.5)))))
    assert wrong_width.capture_layer_exact is True
    assert wrong_width.capture_width_exact is False
    assert wrong_width.latest_activation_width == 2

    duplicate_activation = replace(
        second_step,
        residual_activations=(
            second_step.residual_activations[0],
            second_step.residual_activations[0],
        ),
    )
    wrong_multiplicity = _audit(_capture(sequence=(first_step, duplicate_activation)))
    assert wrong_multiplicity.capture_layer_exact is False
    assert wrong_multiplicity.capture_width_exact is False


def test_nested_collection_and_numeric_type_drift_fails_loudly() -> None:
    baseline = _capture()
    first_step, second_step = baseline.residual_sequence

    with pytest.raises(TypeError, match="residual_sequence must be an exact tuple"):
        _audit(replace(baseline, residual_sequence=list(baseline.residual_sequence)))

    with pytest.raises(TypeError, match="exact ResidualSequenceStep"):
        _audit(
            replace(
                baseline,
                residual_sequence=(
                    SimpleNamespace(
                        step=0,
                        token="fake",
                        feature_surface=(),
                        residual_activations=first_step.residual_activations,
                        description="fake",
                    ),
                    second_step,
                ),
            )
        )

    fake_activation = SimpleNamespace(
        layer_index=20,
        activation=(1.0, -2.0, 0.5),
        step=0,
    )
    with pytest.raises(TypeError, match="exact ResidualActivation"):
        _audit(
            _capture(
                sequence=(
                    replace(first_step, residual_activations=(fake_activation,)),
                    second_step,
                )
            )
        )

    mutable_activations = replace(
        first_step,
        residual_activations=list(first_step.residual_activations),
    )
    with pytest.raises(TypeError, match="residual_activations must be an exact tuple"):
        _audit(_capture(sequence=(mutable_activations, second_step)))

    mutable_values = replace(
        first_step.residual_activations[0],
        activation=list(first_step.residual_activations[0].activation),
    )
    with pytest.raises(TypeError, match="activation must be an exact tuple"):
        _audit(
            _capture(
                sequence=(
                    replace(first_step, residual_activations=(mutable_values,)),
                    second_step,
                )
            )
        )

    nonnumeric_values = replace(
        first_step.residual_activations[0],
        activation=(True, -2.0, 0.5),
    )
    with pytest.raises(TypeError, match="residual activation value"):
        _audit(
            _capture(
                sequence=(
                    replace(first_step, residual_activations=(nonnumeric_values,)),
                    second_step,
                )
            )
        )

    mutable_feature = replace(_feature_surface()[0], values=[1.0])
    with pytest.raises(TypeError, match="feature_surface.*values must be an exact tuple"):
        _audit(_capture(feature_surface=(mutable_feature,)))

    fake_feature = SimpleNamespace(
        name="hook_layer_coverage",
        values=(1.0,),
        source="fake",
        layer_hint=None,
    )
    with pytest.raises(TypeError, match="exact FeatureSignal"):
        _audit(_capture(feature_surface=(fake_feature,)))


def test_top_level_contract_and_expected_geometry_are_exact() -> None:
    with pytest.raises(TypeError, match="requires OpenWeightRuntimeCapture"):
        audit_strict_capture(
            object(),
            expected_layer_index=20,
            expected_activation_width=3,
        )
    with pytest.raises(ValueError, match="expected_layer_index"):
        audit_strict_capture(
            _capture(),
            expected_layer_index=-1,
            expected_activation_width=3,
        )
    with pytest.raises(ValueError, match="expected_activation_width"):
        audit_strict_capture(
            _capture(),
            expected_layer_index=20,
            expected_activation_width=True,
        )
