from __future__ import annotations

from dataclasses import replace
import json

import pytest

from volvence_zero.agent.eta_proof_benchmark import ETAOpenWeightRuntimeConfig
from volvence_zero.agent.eta_segment_credit_evidence import (
    SegmentCreditEvent,
    _prediction_metrics,
    export_eta_segment_credit_evidence,
    run_eta_segment_credit_evidence,
)
from volvence_zero.runtime import WiringLevel
from volvence_zero.substrate import (
    ResidualSequenceStep,
    SubstrateSnapshot,
    SurfaceKind,
    build_training_trace,
)
from volvence_zero.temporal import (
    FullLearnedTemporalPolicy,
    MetacontrollerParameterStore,
    MetacontrollerSSLTrainer,
    build_training_trace_from_substrate_snapshots,
)


def _prefix_snapshots() -> tuple[SubstrateSnapshot, ...]:
    source = build_training_trace(
        trace_id="real-prefix-source",
        source_text="repair inspect continue",
    )
    snapshots: list[SubstrateSnapshot] = []
    for index, step in enumerate(source.steps):
        prefix = tuple(
            ResidualSequenceStep(
                step=prefix_step.step,
                token=prefix_step.token,
                feature_surface=prefix_step.feature_surface,
                residual_activations=prefix_step.residual_activations,
                description=f"prefix step {prefix_step.step}",
            )
            for prefix_step in source.steps[: index + 1]
        )
        snapshots.append(
            SubstrateSnapshot(
                model_id="frozen-prefix-runtime",
                is_frozen=True,
                surface_kind=SurfaceKind.RESIDUAL_STREAM,
                token_logits=(0.1,),
                feature_surface=step.feature_surface,
                residual_activations=step.residual_activations,
                residual_sequence=prefix,
                unavailable_fields=(),
                description=f"causal prefix {index}",
            )
        )
    return tuple(snapshots)


def _prediction_event(
    *,
    event_id: str,
    outcome_value: float,
    true_family_id: str,
    eta_family_id: str,
    turn_family_id: str,
) -> SegmentCreditEvent:
    return SegmentCreditEvent(
        run_seed=0,
        case_id="prediction-metric",
        split="train",
        event_id=event_id,
        assignment_reason="subgoal-complete",
        subgoal_id="goal",
        causal_start_step=0,
        causal_end_step=0,
        observed_outcome_step=1,
        observation_lag=1,
        eta_credit_start_step=0,
        eta_credit_end_step=0,
        eta_credit_steps=(0,),
        turn_credit_steps=(1,),
        true_credit_steps=(0,),
        true_family_id=true_family_id,
        eta_family_id=eta_family_id,
        turn_family_id=turn_family_id,
        outcome_value=outcome_value,
        eta_precision=1.0,
        eta_recall=1.0,
        eta_f1=1.0,
        eta_false_credit_rate=0.0,
        turn_precision=0.0,
        turn_recall=0.0,
        turn_f1=0.0,
        turn_false_credit_rate=1.0,
    )


def test_prediction_metrics_use_each_arms_assigned_family() -> None:
    train = (
        _prediction_event(
            event_id="positive",
            outcome_value=1.0,
            true_family_id="truth-positive",
            eta_family_id="eta-positive",
            turn_family_id="turn-mixed",
        ),
        _prediction_event(
            event_id="negative",
            outcome_value=-1.0,
            true_family_id="truth-negative",
            eta_family_id="eta-negative",
            turn_family_id="turn-mixed",
        ),
    )
    heldout = (
        replace(
            train[0],
            event_id="heldout",
            split="heldout",
        ),
    )

    eta_error, eta_reduction = _prediction_metrics(
        train_events=train,
        heldout_events=heldout,
        family_field="eta_family_id",
    )
    turn_error, turn_reduction = _prediction_metrics(
        train_events=train,
        heldout_events=heldout,
        family_field="turn_family_id",
    )

    assert eta_error == 0.0
    assert eta_reduction == 1.0
    assert turn_error == 1.0
    assert turn_reduction == 0.0


def test_residual_trajectory_ssl_uses_each_prefix_latest_real_step() -> None:
    snapshots = _prefix_snapshots()
    trace = build_training_trace_from_substrate_snapshots(
        trace_id="real-prefix-ssl",
        source_text="repair inspect continue",
        snapshots=snapshots,
    )

    assert len(trace.steps) == len(snapshots)
    for index, step in enumerate(trace.steps):
        latest = snapshots[index].residual_sequence[-1]
        assert step.step == index
        assert step.token == latest.token
        assert step.feature_surface == latest.feature_surface
        assert step.residual_activations == latest.residual_activations

    policy = FullLearnedTemporalPolicy(
        parameter_store=MetacontrollerParameterStore(n_z=8)
    )
    policy.parameter_store.set_learning_phase("ssl", structure_frozen=False)
    before_update_state = (
        policy.parameter_store.export_parameter_snapshot().learned_update_rule_state
    )
    report = MetacontrollerSSLTrainer(
        n_z=8,
        ssl_backend=WiringLevel.ACTIVE,
    ).optimize_residual_trajectory(
        policy=policy,
        trace_id="real-prefix-active",
        source_text="repair inspect continue",
        snapshots=snapshots,
    )

    assert report.trained_steps == len(snapshots) - 1
    assert report.torch_parameters_changed > 0
    assert report.torch_wrote_back is True
    after_update_state = (
        policy.parameter_store.export_parameter_snapshot().learned_update_rule_state
    )
    assert before_update_state is not None
    assert after_update_state is not None
    assert after_update_state.update_count == before_update_state.update_count


def test_residual_trajectory_ssl_rejects_mutable_snapshot() -> None:
    snapshots = _prefix_snapshots()
    mutable = (replace(snapshots[0], is_frozen=False), *snapshots[1:])

    with pytest.raises(ValueError, match="frozen substrate"):
        build_training_trace_from_substrate_snapshots(
            trace_id="mutable-prefix",
            source_text="repair inspect continue",
            snapshots=mutable,
        )


def test_eta_segment_credit_evidence_compares_same_delayed_events(
    tmp_path,
) -> None:
    report = run_eta_segment_credit_evidence(
        seed_schedule=(0,),
        backend_label="trace",
        open_weight_config=ETAOpenWeightRuntimeConfig(device="cpu"),
        max_observation_lag=2,
        training_mode="rl-only",
        controller_dim=3,
    )

    assert report.backend_label == "trace"
    assert report.events
    assert report.run_metrics[0].event_count == len(report.events)
    assert all(event.observation_lag >= 1 for event in report.events)
    assert all(event.eta_credit_steps for event in report.events)
    assert all(len(event.turn_credit_steps) == 1 for event in report.events)
    assert {
        interval.metric_name for interval in report.metric_intervals
    } >= {
        "credit_f1_delta",
        "false_credit_reduction",
        "pe_reduction_rate_delta",
        "segment_boundary_f1",
    }

    paths = export_eta_segment_credit_evidence(report, output_dir=tmp_path)
    assert {path.name for path in paths} == {
        "report.json",
        "report.md",
        "predictions.jsonl",
        "outcomes.jsonl",
        "segments.jsonl",
        "credit.jsonl",
    }
    payload = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "eta-segment-credit-evidence.v12"
    assert payload["controller_initialization_seed"] == 42
    assert payload["outcome_target"] == (
        "observed-alignment-minus-nominal-completion-threshold"
    )
    assert payload["ssl_supervision_target"] == "none"
    assert payload["expert_action_supervision"] is False
    assert payload["events"]
    prediction = json.loads(
        (tmp_path / "predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert "eta_predicted_outcome" in prediction
    assert "turn_predicted_outcome" in prediction


def test_eta_segment_credit_evidence_alternates_real_snapshot_ssl_and_rl() -> None:
    report = run_eta_segment_credit_evidence(
        seed_schedule=(0,),
        backend_label="trace",
        open_weight_config=ETAOpenWeightRuntimeConfig(device="cpu"),
        max_observation_lag=2,
        training_mode="ssl-rl-alternating",
        training_cycles=1,
        ssl_updates_per_cycle=2,
        controller_dim=8,
    )

    metrics = report.run_metrics[0]
    assert report.training_mode == "ssl-rl-alternating"
    assert report.training_cycles == 1
    assert report.ssl_updates_per_cycle == 2
    assert report.rollout_replacement_mode == "causal"
    assert report.temporal_fast_prior_enabled is False
    assert report.episode_recurrent_state_isolated is True
    assert report.ssl_supervision_target == "expert-action-vector"
    assert report.expert_action_supervision is True
    assert metrics.training_cycle_count == 1
    assert metrics.ssl_trajectory_count == 6
    assert metrics.ssl_trained_step_count > 0
    assert metrics.ssl_prediction_loss_mean >= 0.0
    assert metrics.ssl_kl_loss_mean >= 0.0
    assert metrics.ssl_parameter_change_count > 0
    assert metrics.ssl_writeback_count == 2
    assert metrics.ssl_optimizer_final_step == 2
    assert metrics.ssl_optimizer_reuse_count == 1
    assert metrics.ssl_supervision_targets == ("expert-action-vector",)
    assert metrics.ssl_expert_action_supervision is True
    assert 0.0 <= metrics.ssl_switch_probability_mean <= 1.0
    assert 0.0 <= metrics.ssl_switch_probability_final <= 1.0
    assert 0.0 <= metrics.ssl_switch_frequency_mean <= 1.0
    assert 0.0 <= metrics.ssl_switch_frequency_final <= 1.0
    assert 0.05 <= metrics.ssl_switch_threshold_final <= 0.95
    assert 0.05 <= metrics.runtime_switch_threshold_final <= 0.95
    assert metrics.ssl_switch_rate_loss_mean >= 0.0
    assert 0.0 <= metrics.ssl_action_boundary_f1_final <= 1.0
    assert (
        0.0
        <= metrics.ssl_boundary_switch_probability_final
        <= 1.0
    )
    assert (
        0.0
        <= metrics.ssl_continuation_switch_probability_final
        <= 1.0
    )
    assert any(
        gate_name == "heldout-delayed-events-observed"
        for gate_name, _passed, _value in report.retain_gates
    )
    assert (
        "ssl-uses-expert-action-targets",
        True,
        1.0,
    ) in report.retain_gates
    if metrics.heldout_event_count == 0:
        assert metrics.eta_heldout_pe == 0.0
        assert metrics.eta_family_assignment_accuracy == 0.0
        assert metrics.eta_credit_f1 == 0.0


def test_eta_segment_credit_evidence_rejects_empty_seed_schedule() -> None:
    try:
        run_eta_segment_credit_evidence(
            seed_schedule=(),
            backend_label="trace",
        )
    except ValueError as exc:
        assert "seed_schedule" in str(exc)
    else:
        raise AssertionError("empty seed schedule must fail loudly")
