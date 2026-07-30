from __future__ import annotations

import json

import pytest

from volvence_zero.internal_rl.counterfactual_selector import (
    CounterfactualActionExample,
    fit_kernel_residual_action_selector,
    fit_residual_action_selector,
    grouped_cross_validate_kernel_residual_action_selector,
    grouped_cross_validate_residual_action_selector,
    residual_action_state_sketch,
    residual_action_state_vector,
    residual_action_state_with_committed_control_summary,
    selector_artifact_from_payload,
    selector_artifact_to_payload,
    select_counterfactual_actions,
    summarize_action_selections,
)
from volvence_zero.substrate import (
    ResidualActivation,
    ResidualSequenceStep,
    SubstrateSnapshot,
    SurfaceKind,
)


pytest.importorskip("torch")


def _example(
    *,
    example_id: str,
    group_id: str,
    first: float,
    second: float,
    audit_deltas: tuple[float, ...] = (),
) -> CounterfactualActionExample:
    best_index = 1 if first > second else 2
    deltas = [0.0, -0.2, -0.2]
    deltas[best_index] = 0.4
    return CounterfactualActionExample(
        example_id=example_id,
        group_id=group_id,
        split="train",
        state_features=(first, second, first - second, 1.0),
        candidate_raw_deltas=tuple(deltas),
        audit_candidate_raw_deltas=audit_deltas,
    )


def _snapshot(*, token: str) -> SubstrateSnapshot:
    steps = tuple(
        ResidualSequenceStep(
            step=step_index,
            token=f"{token}-{step_index}",
            feature_surface=(),
            residual_activations=(
                ResidualActivation(
                    layer_index=2,
                    activation=(
                        0.5 + step_index,
                        -0.25,
                        0.75,
                        -1.0,
                    ),
                    step=step_index,
                ),
                ResidualActivation(
                    layer_index=5,
                    activation=(
                        -0.5,
                        0.25 + step_index,
                        -0.75,
                        1.0,
                    ),
                    step=step_index,
                ),
            ),
            description="selector fixture",
        )
        for step_index in range(3)
    )
    return SubstrateSnapshot(
        model_id="selector-fixture",
        is_frozen=True,
        surface_kind=SurfaceKind.RESIDUAL_STREAM,
        token_logits=(),
        feature_surface=(),
        residual_activations=steps[-1].residual_activations,
        residual_sequence=steps,
        unavailable_fields=(),
        description="selector fixture",
    )


def test_residual_action_state_sketch_is_label_free_and_stable() -> None:
    first = residual_action_state_sketch(
        _snapshot(token="alpha"),
        bucket_width=8,
    )
    second = residual_action_state_sketch(
        _snapshot(token="unrelated-label"),
        bucket_width=8,
    )

    assert len(first) == 36
    assert first == second
    assert any(value != 0.0 for value in first)


def test_full_residual_action_state_preserves_layer_coordinates() -> None:
    first = residual_action_state_vector(_snapshot(token="alpha"))
    second = residual_action_state_vector(
        _snapshot(token="unrelated-label")
    )

    assert len(first) == 36
    assert first == second
    assert any(value != 0.0 for value in first)


def test_committed_control_summary_is_bounded_and_preserves_base_state() -> None:
    snapshot = _snapshot(token="alpha")
    base = residual_action_state_vector(snapshot)
    empty = residual_action_state_with_committed_control_summary(
        snapshot,
        committed_controls=(),
        committed_control_window=2,
        expected_control_dim=3,
    )
    active = residual_action_state_with_committed_control_summary(
        snapshot,
        committed_controls=(
            (0.25, -0.5, 0.75),
            (0.5, 0.25, -0.25),
            (0.75, 0.5, -0.5),
        ),
        committed_control_window=2,
        expected_control_dim=3,
    )

    assert empty[: len(base)] == base
    assert empty[len(base) :] == (0.0,) * 10
    assert active[: len(base)] == base
    assert len(active) == len(base) + 10
    assert active[-1] == 1.0
    assert all(-1.0 <= value <= 1.0 for value in active[len(base) :])


@pytest.mark.parametrize(
    ("controls", "window", "control_dim"),
    (
        (((0.0, 0.0),), 2, 3),
        (((0.0, float("nan"), 0.0),), 2, 3),
        ((), 0, 3),
        ((), 2, 0),
    ),
)
def test_committed_control_summary_fails_loudly_on_invalid_contract(
    controls: tuple[tuple[float, ...], ...],
    window: int,
    control_dim: int,
) -> None:
    with pytest.raises(ValueError):
        residual_action_state_with_committed_control_summary(
            _snapshot(token="alpha"),
            committed_controls=controls,
            committed_control_window=window,
            expected_control_dim=control_dim,
        )


def test_selector_fits_train_only_action_values_and_is_deterministic() -> None:
    examples = tuple(
        _example(
            example_id=f"example-{index}",
            group_id=f"route-{index // 2}",
            first=1.0 if index % 2 == 0 else 0.0,
            second=0.0 if index % 2 == 0 else 1.0,
        )
        for index in range(8)
    )

    first = fit_residual_action_selector(
        examples,
        latent_dim=3,
        ridge_strength=0.1,
    )
    second = fit_residual_action_selector(
        examples,
        latent_dim=3,
        ridge_strength=0.1,
    )
    selections = select_counterfactual_actions(
        first,
        examples,
        prediction_source="train-fit",
    )

    assert first.model_fingerprint == second.model_fingerprint
    assert first.latent_dim == 3
    assert all(selection.top1_match for selection in selections)
    assert dict(summarize_action_selections(selections))[
        "mean_oracle_regret"
    ] == 0.0


def test_selector_grouped_cv_never_trains_on_validation_group() -> None:
    examples = tuple(
        _example(
            example_id=f"example-{group}-{index}",
            group_id=f"route-{group}",
            first=1.0 if index == 0 else 0.0,
            second=0.0 if index == 0 else 1.0,
        )
        for group in range(4)
        for index in range(2)
    )

    selections = grouped_cross_validate_residual_action_selector(
        examples,
        fold_count=4,
        latent_dim=3,
        ridge_strength=0.1,
    )

    assert len(selections) == len(examples)
    assert {
        selection.prediction_source for selection in selections
    } == {
        "train-grouped-cv-fold-0",
        "train-grouped-cv-fold-1",
        "train-grouped-cv-fold-2",
        "train-grouped-cv-fold-3",
    }
    assert all(selection.top1_match for selection in selections)


def test_kernel_selector_uses_full_state_and_grouped_cv() -> None:
    examples = tuple(
        _example(
            example_id=f"kernel-{group}-{index}",
            group_id=f"route-{group}",
            first=1.0 if index == 0 else 0.0,
            second=0.0 if index == 0 else 1.0,
        )
        for group in range(4)
        for index in range(2)
    )

    artifact = fit_kernel_residual_action_selector(
        examples,
        ridge_strength=0.1,
    )
    selections = (
        grouped_cross_validate_kernel_residual_action_selector(
            examples,
            fold_count=4,
            ridge_strength=0.1,
        )
    )

    assert artifact.input_dim == 4
    assert artifact.action_count == 3
    assert len(selections) == len(examples)
    assert all(selection.top1_match for selection in selections)


@pytest.mark.parametrize("model_kind", ("linear", "kernel"))
def test_selector_artifact_json_round_trip_is_exact(model_kind: str) -> None:
    examples = tuple(
        _example(
            example_id=f"roundtrip-{index}",
            group_id=f"route-{index // 2}",
            first=1.0 if index % 2 == 0 else 0.0,
            second=0.0 if index % 2 == 0 else 1.0,
        )
        for index in range(8)
    )
    artifact = (
        fit_residual_action_selector(
            examples,
            latent_dim=3,
            ridge_strength=0.1,
        )
        if model_kind == "linear"
        else fit_kernel_residual_action_selector(
            examples,
            ridge_strength=0.1,
        )
    )

    payload = selector_artifact_to_payload(artifact)
    restored = selector_artifact_from_payload(
        json.loads(json.dumps(payload, sort_keys=True))
    )

    assert restored == artifact
    assert restored.model_fingerprint == artifact.model_fingerprint
    assert restored.predict_action_values(
        examples[0].state_features
    ) == artifact.predict_action_values(examples[0].state_features)


def test_selector_artifact_rejects_dimension_and_fingerprint_drift() -> None:
    examples = tuple(
        _example(
            example_id=f"tamper-{index}",
            group_id=f"route-{index // 2}",
            first=1.0 if index % 2 == 0 else 0.0,
            second=0.0 if index % 2 == 0 else 1.0,
        )
        for index in range(8)
    )
    artifact = fit_kernel_residual_action_selector(
        examples,
        ridge_strength=0.1,
    )
    payload = selector_artifact_to_payload(artifact)

    with pytest.raises(ValueError, match="input_mean dimension"):
        selector_artifact_from_payload(
            {
                **payload,
                "input_mean": payload["input_mean"][:-1],
            }
        )
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        selector_artifact_from_payload(
            {
                **payload,
                "ridge_strength": 10.0,
            }
        )


def test_selector_reports_independent_audit_values_without_fitting_them() -> None:
    examples = tuple(
        _example(
            example_id=f"audit-{group}-{index}",
            group_id=f"route-{group}",
            first=1.0 if index == 0 else 0.0,
            second=0.0 if index == 0 else 1.0,
            audit_deltas=(0.0, -0.3, -0.4),
        )
        for group in range(4)
        for index in range(2)
    )

    selections = grouped_cross_validate_kernel_residual_action_selector(
        examples,
        fold_count=4,
        ridge_strength=0.1,
    )
    summary = dict(summarize_action_selections(selections))

    assert summary["audit_available_rate"] == 1.0
    assert summary["mean_selected_raw_delta"] > 0.0
    assert summary["mean_audit_selected_raw_delta"] < 0.0
    assert all(
        selection.audit_selected_raw_delta is not None
        for selection in selections
    )


def test_selector_fails_loudly_on_schema_drift() -> None:
    examples = (
        _example(
            example_id="first",
            group_id="route-a",
            first=1.0,
            second=0.0,
        ),
        CounterfactualActionExample(
            example_id="second",
            group_id="route-b",
            split="train",
            state_features=(0.0, 1.0),
            candidate_raw_deltas=(0.0, -0.2, 0.4),
        ),
    )

    with pytest.raises(
        ValueError,
        match="one positive input dim",
    ):
        fit_residual_action_selector(examples)


def test_selector_rejects_audit_action_count_drift() -> None:
    examples = (
        _example(
            example_id="first",
            group_id="route-a",
            first=1.0,
            second=0.0,
            audit_deltas=(0.0, -0.1),
        ),
        _example(
            example_id="second",
            group_id="route-b",
            first=0.0,
            second=1.0,
            audit_deltas=(0.0, -0.1),
        ),
    )

    with pytest.raises(ValueError, match="audit action count"):
        fit_kernel_residual_action_selector(examples)
