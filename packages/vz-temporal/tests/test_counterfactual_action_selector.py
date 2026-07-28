from __future__ import annotations

import pytest

from volvence_zero.internal_rl.counterfactual_selector import (
    CounterfactualActionExample,
    fit_kernel_residual_action_selector,
    fit_residual_action_selector,
    grouped_cross_validate_kernel_residual_action_selector,
    grouped_cross_validate_residual_action_selector,
    residual_action_state_sketch,
    residual_action_state_vector,
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
