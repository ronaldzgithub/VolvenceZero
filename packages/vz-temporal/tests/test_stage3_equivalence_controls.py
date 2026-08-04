"""P1 controls for attributing the sealed Stage-3 negative result."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from volvence_zero.substrate import (  # noqa: E402
    ExpertActionTarget,
    TraceStep,
    TrainingTrace,
    build_training_trace,
)
from volvence_zero.temporal.interface import (  # noqa: E402
    MetacontrollerParameterStore,
)
from volvence_zero.temporal.torch_store_ssl import (  # noqa: E402
    CURRENT_OBSERVATION_LEARNED_PROJECTION,
    STEERED_CONTROL_CYCLIC_PERMUTED_Z,
    STEERED_CONTROL_ZERO_Z,
    STEERED_TRAINING_BIAS_ONLY,
    STEERING_PARAMETERIZATION_LOW_RANK_MULTIPLICATIVE,
    StoreSSLTrainingSession,
    fold_trace_inputs_for_metacontroller,
)

_N_Z = 8


class _TargetOffsetScorer:
    """Frozen differentiable scorer whose optimum needs a non-zero delta."""

    hidden_size = 8

    def action_index(self, action_id: str) -> int:
        return {"move:left": 0, "move:right": 1}[action_id]

    def action_nll(
        self,
        *,
        source_texts: tuple[str, ...],
        control_deltas,
        action_indices: tuple[int, ...],
    ):
        del source_texts, action_indices
        target = torch.full_like(control_deltas, 0.4)
        return (control_deltas - target).pow(2).mean(dim=-1) + 0.5

    def trainable_parameters(self) -> tuple:
        return ()


def _trace(trace_id: str = "stage3-p1") -> TrainingTrace:
    base = build_training_trace(
        trace_id=trace_id,
        source_text="steady harbor observations reveal a changing route plan",
    )
    split = len(base.steps) // 2
    return TrainingTrace(
        trace_id=trace_id,
        source_text=base.source_text,
        steps=tuple(
            TraceStep(
                step=step.step,
                token=step.token,
                feature_surface=step.feature_surface,
                residual_activations=step.residual_activations,
                expert_action_target=ExpertActionTarget(
                    action_id="move:left" if index < split else "move:right",
                    values=(1.0, 0.0) if index < split else (0.0, 1.0),
                    source="stage3-p1-test",
                ),
                observation_text=f"step {index}; exits left and right",
            )
            for index, step in enumerate(base.steps)
        ),
    )


def _session(*, training_mode: str = "full") -> StoreSSLTrainingSession:
    return StoreSSLTrainingSession(
        n_z=_N_Z,
        alpha=0.1,
        learning_rate=0.05,
        switch_rate_weight=0.0,
        switch_binary_weight=0.0,
        switch_group_weight=0.0,
        proposal_prediction_weight=0.0,
        gate_choice_weight=0.0,
        action_scorer=_TargetOffsetScorer(),
        steered_training_mode=training_mode,
    )


def test_bias_only_freezes_controller_and_trains_only_the_free_bias() -> None:
    trace = _trace()
    store = MetacontrollerParameterStore(n_z=_N_Z, initialization_seed=11)
    session = _session(training_mode=STEERED_TRAINING_BIAS_ONLY)

    before = session.evaluate_batch(
        store=store,
        traces=(trace,),
        batch_id="bias-only:before",
    )
    report = session.train_batch(
        store=store,
        traces=(trace,),
        batch_id="bias-only:train",
        write_back=False,
    )
    after = session.evaluate_batch(
        store=store,
        traces=(trace,),
        batch_id="bias-only:after",
    )

    assert len(session._all_parameters()) == 1  # noqa: SLF001
    assert report.parameters_changed == _TargetOffsetScorer.hidden_size
    assert after.distortion < before.distortion
    assert store.ndim_encoder_parameters == report.candidate_encoder_parameters
    assert store.ndim_switch_parameters == report.candidate_switch_parameters
    assert store.ndim_decoder_parameters == report.candidate_decoder_parameters


def test_zero_and_cyclic_permuted_z_are_explicit_no_grad_readouts() -> None:
    trace = _trace()
    store = MetacontrollerParameterStore(n_z=_N_Z, initialization_seed=5)
    session = _session()
    session.train_batch(
        store=store,
        traces=(trace,),
        batch_id="full:train",
        write_back=False,
    )

    zero = session.evaluate_batch(
        store=store,
        traces=(trace,),
        batch_id="full:zero",
        control_ablation=STEERED_CONTROL_ZERO_Z,
    )
    permuted = session.evaluate_batch(
        store=store,
        traces=(trace,),
        batch_id="full:permuted",
        control_ablation=STEERED_CONTROL_CYCLIC_PERMUTED_Z,
    )

    assert zero.control_ablation == STEERED_CONTROL_ZERO_Z
    assert permuted.control_ablation == STEERED_CONTROL_CYCLIC_PERMUTED_Z
    assert zero.step_count == permuted.step_count == len(trace.steps)
    assert session.optimizer_step == 1


def test_external_oracle_boundaries_are_readout_only_and_fail_loudly() -> None:
    trace = _trace()
    store = MetacontrollerParameterStore(n_z=_N_Z, initialization_seed=7)
    session = _session()
    labels = {trace.trace_id: (1.0,) * (len(trace.steps) - 1)}

    report = session.evaluate_batch(
        store=store,
        traces=(trace,),
        batch_id="oracle",
        boundary_labels=labels,
    )

    assert report.boundary_label_source == "external-oracle"
    assert 0.0 <= report.boundary_f1 <= 1.0
    assert session.optimizer_step == 0

    with pytest.raises(ValueError, match="transition count"):
        session.evaluate_batch(
            store=store,
            traces=(trace,),
            batch_id="bad-oracle",
            boundary_labels={trace.trace_id: (1.0,)},
        )


def test_exact_entry_readout_publishes_the_training_fold_surface() -> None:
    trace = _trace()

    vectors = fold_trace_inputs_for_metacontroller(
        trace=trace,
        n_input=_N_Z,
    )

    assert len(vectors) == len(trace.steps)
    assert all(len(vector) == _N_Z for vector in vectors)
    assert vectors == fold_trace_inputs_for_metacontroller(
        trace=trace,
        n_input=_N_Z,
    )


def test_unknown_diagnostic_modes_fail_loudly() -> None:
    with pytest.raises(ValueError, match="steered_training_mode"):
        _session(training_mode="unknown")

    trace = _trace()
    store = MetacontrollerParameterStore(n_z=_N_Z)
    session = _session()
    with pytest.raises(ValueError, match="control_ablation"):
        session.evaluate_batch(
            store=store,
            traces=(trace,),
            batch_id="unknown-control",
            control_ablation="shuffle-ish",
        )


def test_faithful_low_rank_path_learns_projection_without_free_bias() -> None:
    trace = _trace("faithful-low-rank")
    store = MetacontrollerParameterStore(
        n_z=_N_Z,
        n_input=_TargetOffsetScorer.hidden_size,
        initialization_seed=13,
    )
    session = StoreSSLTrainingSession(
        n_z=_N_Z,
        alpha=0.1,
        learning_rate=0.05,
        switch_rate_weight=0.0,
        switch_binary_weight=0.0,
        switch_group_weight=0.0,
        proposal_prediction_weight=0.0,
        gate_choice_weight=0.0,
        action_scorer=_TargetOffsetScorer(),
        steering_parameterization=(
            STEERING_PARAMETERIZATION_LOW_RANK_MULTIPLICATIVE
        ),
        steering_rank=4,
        current_observation_mode=CURRENT_OBSERVATION_LEARNED_PROJECTION,
    )

    session.train_batch(
        store=store,
        traces=(trace,),
        batch_id="faithful:train",
        write_back=False,
    )
    attestation = session.faithful_steering_attestation()
    zero = session.evaluate_batch(
        store=store,
        traces=(trace,),
        batch_id="faithful:zero-z",
        control_ablation=STEERED_CONTROL_ZERO_Z,
    )

    assert attestation.controller_input_width == _TargetOffsetScorer.hidden_size
    assert attestation.residual_width == _TargetOffsetScorer.hidden_size
    assert attestation.steering_rank == 4
    assert attestation.free_bias_present is False
    assert attestation.zero_code_strict_noop is True
    assert attestation.input_projection_parameters_changed > 0
    assert attestation.low_rank_parameters_changed > 0
    assert zero.distortion == pytest.approx(0.66)


def test_faithful_low_rank_path_rejects_input_residual_width_mismatch() -> None:
    trace = _trace("faithful-width-mismatch")
    store = MetacontrollerParameterStore(
        n_z=_N_Z,
        n_input=_TargetOffsetScorer.hidden_size - 1,
        initialization_seed=17,
    )
    session = StoreSSLTrainingSession(
        n_z=_N_Z,
        action_scorer=_TargetOffsetScorer(),
        proposal_prediction_weight=0.0,
        steering_parameterization=(
            STEERING_PARAMETERIZATION_LOW_RANK_MULTIPLICATIVE
        ),
        steering_rank=4,
        current_observation_mode=CURRENT_OBSERVATION_LEARNED_PROJECTION,
    )

    with pytest.raises(ValueError, match="controller input width"):
        session.evaluate_batch(
            store=store,
            traces=(trace,),
            batch_id="faithful:mismatch",
        )
