from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from volvence_zero.agent.dialogue_steering_evidence import (
    DialogueSteeringEffect,
    DialogueSteeringThresholds,
    DialogueSteeringTraceDataset,
    DialogueSteeringTraceRow,
    run_dialogue_steering_evidence,
)
from volvence_zero.agent.steering_promotion_gate import (
    STEERING_PROMOTION_SCHEMA_VERSION,
    SteeringComponent,
    SteeringPromotionEvidence,
    SteeringValidationAxis,
    build_steering_modification_gate_review,
    evaluate_steering_promotion,
)
from volvence_zero.credit.gate import GateDecision
from volvence_zero.agent import steering_artifact_training
from volvence_zero.steering_contracts import (
    STEERING_GATE_ARTIFACT_SCHEMA_VERSION,
    SteeringGateArtifact,
    SteeringTerminalPredictionError,
)


_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64


def test_steering_artifact_matrix_requires_an_explicit_array_type() -> None:
    import torch

    expected = ((1.0, 2.0),)
    assert steering_artifact_training._matrix(
        np.asarray(((1.0, 2.0),), dtype=np.float64)
    ) == expected
    assert steering_artifact_training._matrix(
        torch.tensor(((1.0, 2.0),), dtype=torch.float32)
    ) == expected
    with pytest.raises(TypeError, match="numpy.ndarray or torch.Tensor"):
        steering_artifact_training._matrix([[1.0, 2.0]])


def _terminal_pe(
    *, sample_id: str, positive: bool
) -> SteeringTerminalPredictionError:
    action_mse, noop_mse = ((0.1, 1.0) if positive else (1.0, 0.1))
    action_cosine, noop_cosine = ((0.9, 0.1) if positive else (0.1, 0.9))
    return SteeringTerminalPredictionError(
        episode_id=f"episode:{sample_id}",
        decision_ids=(f"source-decision:{sample_id}",),
        action_batch_id=f"action:{sample_id}",
        noop_batch_id=f"noop:{sample_id}",
        sample_ids=(sample_id,),
        prediction_head_fingerprint=_SHA_B,
        target_lineage_fingerprint=_SHA_C,
        target_model_id="frozen-dialogue-model",
        target_model_weights_sha256=_SHA_A,
        action_mean_squared_error=action_mse,
        noop_mean_squared_error=noop_mse,
        relative_mse_improvement=(
            (noop_mse - action_mse) / max(action_mse, noop_mse)
        ),
        action_mean_cosine_similarity=action_cosine,
        noop_mean_cosine_similarity=noop_cosine,
        cosine_error_improvement=action_cosine - noop_cosine,
        terminal=True,
        description="Synthetic PE-owned matched terminal settlement.",
    )


def _rows(
    *, split: str, count: int, signal_sensitive: bool = True
) -> tuple[DialogueSteeringTraceRow, ...]:
    rows = []
    for index in range(count):
        positive = index % 2 == 0
        sample_id = f"{split}-sample-{index}"
        settlement = _terminal_pe(sample_id=sample_id, positive=positive)
        if not signal_sensitive:
            settlement = replace(
                settlement,
                action_mean_squared_error=0.5,
                noop_mean_squared_error=0.5,
                relative_mse_improvement=0.0,
                action_mean_cosine_similarity=0.5,
                noop_mean_cosine_similarity=0.5,
                cosine_error_improvement=0.0,
            )
        rows.append(
            DialogueSteeringTraceRow(
                sample_id=sample_id,
                split=split,
                episode_id=settlement.episode_id,
                cluster_id=f"{split}-cluster-{index // 4}",
                session_index=(index % 4) + 1,
                observations=(("belief_margin", 1.0 if positive else 0.0),),
                terminal_prediction_error=settlement,
                reader_artifact_id="reader-v1",
                executor_artifact_id="executor-v1",
                source_model_id="frozen-dialogue-model",
                source_model_weights_sha256=_SHA_A,
                shadow_hook_latency_ms=1.0,
                end_to_end_latency_ms=10.0,
                shadow_owner_chain_complete=True,
                shadow_hook_executed=True,
                free_bias_present=False,
                zero_code_strict_noop=True,
                raw_text_retained=False,
                evaluation_writeback_allowed=False,
                sensor_off_executor_artifact_id="executor-unconditional-v1",
                control_norm=0.1,
                control_norm_cap=0.25,
                sensor_off_control_norm=0.1,
                sensor_off_mean_squared_error=1.5,
                sensor_off_cosine_similarity=0.0,
            )
        )
    return tuple(rows)


def _dataset(*, signal_sensitive: bool = True) -> DialogueSteeringTraceDataset:
    return DialogueSteeringTraceDataset(
        schema_version="dialogue-steering-trace-dataset.v1",
        bundle_id="bundle-v1",
        prediction_head_fingerprint=_SHA_B,
        train_rows=_rows(
            split="train", count=80, signal_sensitive=signal_sensitive
        ),
        validation_rows=_rows(
            split="validation", count=80, signal_sensitive=signal_sensitive
        ),
        raw_text_retained=False,
        evaluation_writeback_allowed=False,
        description="Text-free synthetic steering evidence fixture.",
    )


def _thresholds() -> DialogueSteeringThresholds:
    return DialogueSteeringThresholds(
        min_real_trace_turns=40,
        action_sensitivity_abs_credit=0.1,
        min_action_sensitive_fraction=0.5,
        min_convergence_improvement=-1.0,
        min_gain_vs_noop=0.05,
        min_gain_vs_always_on=0.05,
        min_gain_vs_random_gate=0.05,
        min_gate_selectivity=0.5,
        require_clustered_ci_lower_positive=True,
    )


def test_c3_learns_selective_policy_only_through_pe_credit_owner() -> None:
    dataset = _dataset()

    report = run_dialogue_steering_evidence(
        train_rows=dataset.train_rows,
        validation_rows=dataset.validation_rows,
        preregistration_sha256=_SHA_D,
        seed_schedule=(3, 5),
        policy_restarts=3,
        max_online_episodes=400,
        eval_every=40,
        learning_rate=0.1,
        bootstrap_resamples=200,
        thresholds=_thresholds(),
        artifact_fit_prerequisite_passed=True,
    )

    assert report.admission.admitted is True
    assert report.aggregate.gain_vs_noop_ci_lower_worst_seed > 0.0
    assert report.aggregate.gain_vs_always_on_ci_lower_worst_seed > 0.0
    assert report.aggregate.gain_vs_random_gate_ci_lower_worst_seed > 0.0
    assert report.aggregate.gate_selectivity_worst_seed >= 0.5
    assert report.policy_parameters_changed is True
    assert report.reader_parameters_changed is False
    assert report.executor_parameters_changed is False
    assert report.substrate_trainable_parameter_count == 0
    assert report.terminal_credit_source == (
        "substrate_n_plus_one_representation_pe->credit->steering_gate"
    )
    assert report.raw_text_retained is False
    assert report.evaluation_writeback_allowed is False
    assert type(report).from_json(report.to_json()) == report


def test_c3_exits_honestly_when_n_plus_one_signal_is_action_insensitive() -> None:
    dataset = _dataset(signal_sensitive=False)

    report = run_dialogue_steering_evidence(
        train_rows=dataset.train_rows,
        validation_rows=dataset.validation_rows,
        preregistration_sha256=_SHA_D,
        seed_schedule=(7,),
        policy_restarts=2,
        max_online_episodes=80,
        eval_every=20,
        learning_rate=0.1,
        bootstrap_resamples=100,
        thresholds=replace(_thresholds(), require_clustered_ci_lower_positive=False),
        artifact_fit_prerequisite_passed=True,
    )

    assert report.admission.admitted is False
    assert report.admission.condition_action_sensitivity is False
    assert report.admission.exit_reason == (
        "dialogue-n-plus-one-signal-insensitive-to-steering"
    )


def test_trace_dataset_rejects_cluster_leakage_and_round_trips() -> None:
    dataset = _dataset()
    assert DialogueSteeringTraceDataset.from_json(dataset.to_json()) == dataset
    leaked = replace(
        dataset.validation_rows[0],
        cluster_id=dataset.train_rows[0].cluster_id,
    )

    with pytest.raises(ValueError, match="clusters overlap"):
        replace(dataset, validation_rows=(leaked, *dataset.validation_rows[1:]))


def test_trace_row_rejects_partial_sensor_off_lineage_and_budget_overflow() -> None:
    row = _rows(split="validation", count=1)[0]

    with pytest.raises(ValueError, match="sensor-off PE is invalid"):
        replace(row, sensor_off_executor_artifact_id="")
    with pytest.raises(ValueError, match="control exceeds the shared cap"):
        replace(row, sensor_off_control_norm=row.control_norm_cap + 0.01)


def _effect(*, passed: bool) -> DialogueSteeringEffect:
    return DialogueSteeringEffect(
        mean=0.2 if passed else 0.0,
        ci_lower=0.1 if passed else -0.1,
        ci_upper=0.3 if passed else 0.1,
        cluster_count=20,
        row_count=500,
    )


def _candidate_gate() -> SteeringGateArtifact:
    return SteeringGateArtifact(
        schema_version=STEERING_GATE_ARTIFACT_SCHEMA_VERSION,
        artifact_id="candidate-gate",
        source_preregistration_sha256=_SHA_D,
        feature_names=("belief_margin",),
        weights=((0.0, 1.0),),
        bias=(0.0, 0.0),
        policy_version=2,
        description="Frozen promotion candidate.",
    )


def _promotion_evidence(
    *, sensor_ok: bool = True, executor_ok: bool = True, gate_ok: bool = False
) -> SteeringPromotionEvidence:
    axes = tuple(
        SteeringValidationAxis(
            name=name,
            baseline_arm="noop",
            baseline_error=1.0,
            learned_error=0.8,
            target_std=0.2,
            relative_improvement=0.2,
            absolute_improvement=0.2,
            informative=True,
            passed=True,
        )
        for name in ("normalized_n_plus_one_mse", "n_plus_one_cosine_error")
    )
    return SteeringPromotionEvidence(
        schema_version=STEERING_PROMOTION_SCHEMA_VERSION,
        preregistration_sha256=_SHA_D,
        c3_preregistration_sha256=_SHA_C,
        c3_report_sha256=_SHA_A,
        trace_sha256=_SHA_B,
        bundle_sha256=_SHA_D,
        real_trace_turns=500,
        validation_axes=axes,
        gate_off_vs_noop=_effect(passed=gate_ok),
        gate_off_vs_always_on=_effect(passed=gate_ok),
        executor_on_vs_noop=_effect(passed=executor_ok),
        sensor_off_conditional_advantage=_effect(passed=sensor_ok),
        checkpoint_round_trips_verified=1,
        checkpoint_json_round_trip_verified=True,
        p95_shadow_overhead_ratio=0.1,
        p95_end_to_end_latency_ms=100.0,
        safety_gate_ok=True,
        runtime_acceptance_all_passed=True,
        c3_admitted=gate_ok,
        reader_artifact_id="reader-v1",
        executor_artifact_id="executor-v1",
        sensor_off_executor_artifact_id="executor-unconditional-v1",
        candidate_gate_artifact=_candidate_gate(),
        free_bias_present=False,
        zero_code_strict_noop=True,
        raw_text_retained=False,
        evaluation_writeback_allowed=False,
        production_default_changed=False,
        description="Synthetic B3 evidence.",
    )


def test_b3_allows_sensor_executor_prefix_while_gate_remains_shadow() -> None:
    evidence = _promotion_evidence()
    review = build_steering_modification_gate_review(
        evidence=evidence,
        candidate_bundle_sha256=_SHA_A,
    )
    verdict = evaluate_steering_promotion(
        evidence,
        modification_gate_review=review,
    )

    assert review.decision is GateDecision.ALLOW
    assert verdict.eligible_prefix == (
        SteeringComponent.SENSOR,
        SteeringComponent.EXECUTOR,
    )
    assert verdict.sensor_executor_active_authorized is True
    assert verdict.gate_active_authorized is False
    assert any("gate_off" in reason for reason in verdict.blocking_reasons)


def test_b3_never_skips_a_failed_predecessor() -> None:
    evidence = _promotion_evidence(
        sensor_ok=False,
        executor_ok=True,
        gate_ok=True,
    )
    review = build_steering_modification_gate_review(
        evidence=evidence,
        candidate_bundle_sha256=_SHA_A,
    )
    verdict = evaluate_steering_promotion(
        evidence,
        modification_gate_review=review,
    )

    assert verdict.eligible_prefix == ()
    executor = verdict.component_verdicts[1]
    gate = verdict.component_verdicts[2]
    assert "prior_sensor_active" in executor.missing_gates
    assert "prior_executor_active" in gate.missing_gates


def test_b3_modification_gate_blocks_every_active_prefix() -> None:
    evidence = _promotion_evidence(gate_ok=True)
    weak_axes = tuple(
        replace(axis, relative_improvement=0.01)
        for axis in evidence.validation_axes
    )
    evidence = replace(evidence, validation_axes=weak_axes)
    review = build_steering_modification_gate_review(
        evidence=evidence,
        candidate_bundle_sha256=_SHA_A,
    )

    verdict = evaluate_steering_promotion(
        evidence,
        modification_gate_review=review,
    )

    assert review.decision is GateDecision.BLOCK
    assert review.blocking_reasons == (
        "validation_delta 0.010 below required margin 0.050",
    )
    assert verdict.eligible_prefix == ()
    assert all(
        "modification_gate_offline" in component.missing_gates
        for component in verdict.component_verdicts
    )


def test_artifact_fit_places_sensor_off_control_on_bundle_not_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    examples = (
        SimpleNamespace(
            action_residual=(1.0, 0.0),
            context_residual=(1.0, 0.0),
            subgoal_index=0,
            action_index=0,
            observation_text="fixture-a",
        ),
        SimpleNamespace(
            action_residual=(0.0, 1.0),
            context_residual=(0.0, 1.0),
            subgoal_index=1,
            action_index=1,
            observation_text="fixture-b",
        ),
    )

    class _Reader:
        weights = np.eye(2)
        feature_mean = np.zeros(2)
        feature_scale = np.ones(2)

        def predict(self, values):
            return np.asarray([0 if row[0] >= row[1] else 1 for row in values])

    class _Operator:
        def __init__(
            self,
            *,
            torch,
            width,
            rank,
            class_count,
            conditional,
            seed,
        ) -> None:
            del conditional, seed
            self._U = torch.eye(width, rank)
            self._V = torch.eye(width, rank)
            self._Z = torch.full((class_count, rank), 0.5)

        def deltas(self, *, residuals, subgoal_indices):
            return residuals * self._Z[subgoal_indices, 0].unsqueeze(1)

        def parameters(self):
            return self._U, self._V, self._Z

    monkeypatch.setattr(
        steering_artifact_training, "_subgoal_vocabulary", lambda _corpus: ("a", "b")
    )
    monkeypatch.setattr(
        steering_artifact_training,
        "build_conflict_junction_rows",
        lambda _corpus, *, split: (split,),
    )
    monkeypatch.setattr(
        steering_artifact_training, "_labelled_rows", lambda rows: rows
    )
    monkeypatch.setattr(
        steering_artifact_training,
        "_capture_examples",
        lambda *args, **kwargs: examples,
    )
    monkeypatch.setattr(
        steering_artifact_training,
        "fit_condition_reader",
        lambda *args, **kwargs: _Reader(),
    )
    monkeypatch.setattr(
        steering_artifact_training, "_ConditionalOperator", _Operator
    )
    monkeypatch.setattr(
        steering_artifact_training, "_train_operator", lambda **kwargs: None
    )
    monkeypatch.setattr(
        steering_artifact_training,
        "_per_row_baseline_nll",
        lambda **kwargs: (1.0, 1.0),
    )
    controlled_calls = iter(((0.5, 0.5), (0.8, 0.8)))
    monkeypatch.setattr(
        steering_artifact_training,
        "_per_row_controlled_nll",
        lambda **kwargs: next(controlled_calls),
    )
    monkeypatch.setattr(
        steering_artifact_training, "_reader_accuracy", lambda *args: 1.0
    )
    result = steering_artifact_training.fit_steering_artifact_bundle(
        corpus=object(),
        runtime=SimpleNamespace(model_id="frozen-dialogue-model"),
        scorer=SimpleNamespace(trainable_parameters=lambda: ()),
        model_weights_sha256=_SHA_A,
        source_preregistration_sha256=_SHA_D,
        injection_layer_index=0,
        residual_width=2,
        steering_rank=2,
        executor_updates=1,
        batch_size=2,
    )

    assert result.report.prerequisite_passed is True
    assert result.bundle.sensor_off_executor is not None
    assert len(set(result.bundle.sensor_off_executor.condition_codes)) == 1
    assert not hasattr(result.bundle.gate, "sensor_off_executor")
