from __future__ import annotations

from volvence_zero.agent.eta_conditional_steering_screen import (
    ConditionalSteeringAggregate,
    ConditionalSteeringThresholds,
    assess_conditional_steering,
)


def _aggregate(
    *,
    gap: float = 2.5,
    advantage: float = 1.0,
    specificity: float = 1.0,
    fraction: float = 0.9,
) -> ConditionalSteeringAggregate:
    return ConditionalSteeringAggregate(
        seed_count=3,
        heldout_noop_nll_mean=2.8,
        heldout_conditional_nll_mean=2.8 - gap,
        heldout_unconditional_nll_mean=2.8 - gap + advantage,
        heldout_random_condition_nll_mean=2.8 - gap + specificity,
        subgoal_revealed_ceiling_nll_mean=0.2,
        gap_closed_nll_mean=gap,
        conditional_advantage_nll_mean=advantage,
        condition_specificity_nll_mean=specificity,
        gap_closed_fraction_mean=fraction,
        gap_closed_nll_min=gap - 0.05,
        conditional_advantage_nll_min=advantage - 0.05,
    )


def test_admits_when_all_gates_pass():
    admission = assess_conditional_steering(
        aggregate=_aggregate(),
        thresholds=ConditionalSteeringThresholds(),
        free_bias_present=False,
        zero_code_strict_noop=True,
        substrate_trainable_parameter_count=0,
        conditional_parameters_changed=True,
    )
    assert admission.admitted
    assert admission.failed_conditions == ()


def test_blocks_when_conditional_no_better_than_unconditional():
    admission = assess_conditional_steering(
        aggregate=_aggregate(advantage=0.0),
        thresholds=ConditionalSteeringThresholds(),
        free_bias_present=False,
        zero_code_strict_noop=True,
        substrate_trainable_parameter_count=0,
        conditional_parameters_changed=True,
    )
    assert not admission.admitted
    assert "conditional-advantage" in admission.failed_conditions


def test_blocks_when_gap_not_closed():
    admission = assess_conditional_steering(
        aggregate=_aggregate(gap=0.1, fraction=0.04),
        thresholds=ConditionalSteeringThresholds(),
        free_bias_present=False,
        zero_code_strict_noop=True,
        substrate_trainable_parameter_count=0,
        conditional_parameters_changed=True,
    )
    assert not admission.admitted
    assert "gap-closed" in admission.failed_conditions
    assert "gap-closed-fraction" in admission.failed_conditions


def test_structural_gate_catches_free_bias_or_leaky_noop():
    admission = assess_conditional_steering(
        aggregate=_aggregate(),
        thresholds=ConditionalSteeringThresholds(),
        free_bias_present=True,
        zero_code_strict_noop=False,
        substrate_trainable_parameter_count=5,
        conditional_parameters_changed=False,
    )
    assert not admission.admitted
    assert "structural-integrity" in admission.failed_conditions


def test_specificity_gate_requires_beating_random_condition():
    admission = assess_conditional_steering(
        aggregate=_aggregate(specificity=0.0),
        thresholds=ConditionalSteeringThresholds(),
        free_bias_present=False,
        zero_code_strict_noop=True,
        substrate_trainable_parameter_count=0,
        conditional_parameters_changed=True,
    )
    assert not admission.admitted
    assert "condition-specificity" in admission.failed_conditions
