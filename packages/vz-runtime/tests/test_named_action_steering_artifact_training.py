from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from volvence_zero.agent.named_action_steering_artifact_training import (
    NamedActionSteeringCorpus,
    NamedActionSteeringRow,
    fit_named_action_steering_artifact_bundle,
    named_action_fit_lineage_sha256,
)


_ACTIONS = ("stay_present_without_probe", "respect_space_with_return_option")


def _row(row_id: str, scope: str, action_id: str) -> NamedActionSteeringRow:
    condition = "owner condition A" if action_id == _ACTIONS[0] else "owner condition B"
    return NamedActionSteeringRow(
        row_id=row_id,
        subject_scope=scope,
        action_text="same ambiguous public relationship action surface",
        condition_text=condition,
        condition_label=action_id,
        target_action_id=action_id,
        source_condition_lineage_sha256=("a" if action_id == _ACTIONS[0] else "b")
        * 64,
    )


def _corpus() -> NamedActionSteeringCorpus:
    return NamedActionSteeringCorpus(
        source_protocol_sha256="c" * 64,
        action_ids=_ACTIONS,
        class_labels=_ACTIONS,
        train_rows=(
            _row("train-a-1", "train-scope-a-1", _ACTIONS[0]),
            _row("train-a-2", "train-scope-a-2", _ACTIONS[0]),
            _row("train-b-1", "train-scope-b-1", _ACTIONS[1]),
            _row("train-b-2", "train-scope-b-2", _ACTIONS[1]),
        ),
        heldout_rows=(
            _row("heldout-a", "heldout-scope-a", _ACTIONS[0]),
            _row("heldout-b", "heldout-scope-b", _ACTIONS[1]),
        ),
        description="typed pre-action owner fixture",
    )


@dataclass(frozen=True)
class _Activation:
    layer_index: int
    activation: tuple[float, ...]


@dataclass(frozen=True)
class _Capture:
    residual_activations: tuple[_Activation, ...]


class _Runtime:
    model_id = "frozen-named-action-fixture"

    def capture(self, *, source_text: str) -> _Capture:
        if source_text == "owner condition A":
            values = (1.0, 0.0, 0.5, 0.0)
        elif source_text == "owner condition B":
            values = (0.0, 1.0, 0.0, 0.5)
        else:
            values = (1.0, 1.0, 0.5, -0.5)
        return _Capture((_Activation(layer_index=0, activation=values),))


class _Scorer:
    action_option_ids = _ACTIONS
    probe_hidden_norm = 4.0
    control_norm_cap = 1.0

    @staticmethod
    def trainable_parameters() -> tuple[object, ...]:
        return ()

    @staticmethod
    def action_index(action_id: str) -> int:
        return _ACTIONS.index(action_id)

    @staticmethod
    def _loss(control_deltas, action_indices):
        import torch

        norms = torch.linalg.vector_norm(control_deltas, dim=1, keepdim=True)
        scales = torch.clamp(1.0 / torch.clamp(norms, min=1e-12), max=1.0)
        capped = control_deltas * scales
        targets = torch.tensor(
            [0.75 if index == 0 else -0.75 for index in action_indices],
            dtype=capped.dtype,
            device=capped.device,
        )
        return (capped[:, 0] - targets) ** 2 + 0.05

    def action_nll(self, *, source_texts, control_deltas, action_indices):
        del source_texts
        return self._loss(control_deltas, action_indices)

    def baseline_action_nll(self, *, source_texts, action_indices):
        import torch

        zeros = torch.zeros((len(source_texts), 4), dtype=torch.float32)
        return tuple(float(value) for value in self._loss(zeros, action_indices))

    def controlled_action_nll(
        self,
        *,
        source_texts,
        control_deltas,
        action_indices,
    ):
        del source_texts
        return tuple(
            float(value)
            for value in self._loss(control_deltas, action_indices).detach()
        )


def test_named_action_corpus_round_trip_and_group_firewall() -> None:
    corpus = _corpus()

    assert NamedActionSteeringCorpus.from_payload(corpus.to_payload()) == corpus
    assert NamedActionSteeringCorpus.from_payload(corpus.to_payload()).corpus_id == (
        corpus.corpus_id
    )

    with pytest.raises(ValueError, match="subject scopes must be disjoint"):
        replace(
            corpus,
            heldout_rows=(
                replace(corpus.heldout_rows[0], subject_scope="train-scope-a-1"),
                corpus.heldout_rows[1],
            ),
        )


def test_named_action_fit_reuses_bounded_operator_and_matched_sensor_off() -> None:
    result = fit_named_action_steering_artifact_bundle(
        corpus=_corpus(),
        runtime=_Runtime(),
        scorer=_Scorer(),
        model_weights_sha256="d" * 64,
        source_preregistration_sha256=named_action_fit_lineage_sha256(_corpus()),
        injection_layer_index=0,
        residual_width=4,
        steering_rank=2,
        executor_updates=40,
        executor_learning_rate=0.05,
        reader_ridge_lambda=1.0,
        batch_size=4,
        seed=7,
        control_norm_cap_ratio=0.25,
    )

    bundle = result.bundle
    assert bundle.reader.class_labels == _ACTIONS
    assert bundle.executor.class_labels == _ACTIONS
    assert bundle.executor.free_bias_present is False
    assert bundle.executor.zero_code_strict_noop is True
    assert bundle.executor.control_norm_cap_ratio == 0.25
    assert bundle.sensor_off_executor is not None
    assert len(set(bundle.sensor_off_executor.condition_codes)) == 1
    assert result.report.substrate_trainable_parameter_count == 0
    assert result.report.reader_executor_frozen_for_dialogue is True
    assert result.report.reader_heldout_accuracy == 1.0


def test_named_action_fit_rejects_scorer_surface_drift() -> None:
    scorer = _Scorer()
    scorer.action_option_ids = tuple(reversed(_ACTIONS))
    corpus = _corpus()

    with pytest.raises(ValueError, match="action surface differs"):
        fit_named_action_steering_artifact_bundle(
            corpus=corpus,
            runtime=_Runtime(),
            scorer=scorer,
            model_weights_sha256="d" * 64,
            source_preregistration_sha256=named_action_fit_lineage_sha256(corpus),
            injection_layer_index=0,
            residual_width=4,
            steering_rank=2,
            executor_updates=1,
            executor_learning_rate=0.05,
            reader_ridge_lambda=1.0,
            batch_size=4,
            seed=7,
            control_norm_cap_ratio=0.25,
        )


def test_named_action_fit_rejects_protocol_only_or_arbitrary_lineage() -> None:
    corpus = _corpus()

    with pytest.raises(ValueError, match="exact protocol and corpus"):
        fit_named_action_steering_artifact_bundle(
            corpus=corpus,
            runtime=_Runtime(),
            scorer=_Scorer(),
            model_weights_sha256="d" * 64,
            source_preregistration_sha256=corpus.source_protocol_sha256,
            injection_layer_index=0,
            residual_width=4,
            steering_rank=2,
            executor_updates=1,
            executor_learning_rate=0.05,
            reader_ridge_lambda=1.0,
            batch_size=4,
            seed=7,
            control_norm_cap_ratio=0.25,
        )
