from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json

import pytest

from lifeform_domain_emogpt.lab.relationship_residual_fit_corpus import (
    load_relationship_residual_fit_protocol,
)
from lifeform_evolution import relationship_lab_relationship_residual_fit as lane
from volvence_zero.agent.named_action_steering_artifact_training import (
    NamedActionSteeringCorpus,
    NamedActionSteeringRow,
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


def _protocol():
    return replace(
        load_relationship_residual_fit_protocol(),
        model_id="frozen-named-action-fixture",
        model_revision="fixture-revision",
        model_weights_sha256="d" * 64,
        execution_assets_sha256="f" * 64,
        injection_layer_index=0,
        residual_width=4,
        steering_rank=2,
        conditional_executor_updates=30,
        sensor_off_executor_updates=30,
        executor_learning_rate=0.05,
        reader_ridge_lambda=1.0,
        batch_size=4,
        seed=7,
    )


def _corpus(protocol) -> NamedActionSteeringCorpus:
    return NamedActionSteeringCorpus(
        source_protocol_sha256=protocol.protocol_sha256,
        action_ids=_ACTIONS,
        class_labels=_ACTIONS,
        train_rows=(
            _row("train-a-1", "train-a-1", _ACTIONS[0]),
            _row("train-a-2", "train-a-2", _ACTIONS[0]),
            _row("train-b-1", "train-b-1", _ACTIONS[1]),
            _row("train-b-2", "train-b-2", _ACTIONS[1]),
        ),
        heldout_rows=(
            _row("heldout-a", "heldout-a", _ACTIONS[0]),
            _row("heldout-b", "heldout-b", _ACTIONS[1]),
        ),
        description="relationship fit wrapper fixture",
    )


@dataclass(frozen=True)
class _Activation:
    layer_index: int
    activation: tuple[float, ...]


@dataclass(frozen=True)
class _Capture:
    residual_activations: tuple[_Activation, ...]


class _Attestation:
    def __init__(self, protocol, *, weights_sha256: str | None = None) -> None:
        self._payload = {
            "schema_version": "transformers-execution-attestation.v1",
            "model_id": protocol.model_id,
            "model_revision": protocol.model_revision,
            "model_weights_sha256": weights_sha256 or protocol.model_weights_sha256,
            "execution_assets_sha256": protocol.execution_assets_sha256,
            "platform_system": "Windows",
            "local_files_only": True,
            "fallback_mode": "deny",
            "fail_on_truncation": True,
            "model_dtype": "bfloat16",
            "hidden_size": protocol.residual_width,
            "hook_layer_indices": [protocol.injection_layer_index],
            "device": "cuda:0",
        }
        self.attestation_id = hashlib.sha256(
            json.dumps(
                self._payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def to_payload(self):
        return dict(self._payload)


class _Runtime:
    def __init__(self, protocol, *, weights_sha256: str | None = None) -> None:
        self.model_id = protocol.model_id
        self.execution_attestation = _Attestation(
            protocol,
            weights_sha256=weights_sha256,
        )

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
    def trainable_parameters():
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


def _write_corpus(path, corpus: NamedActionSteeringCorpus) -> None:
    path.write_text(
        json.dumps(
            corpus.to_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


def test_create_only_run_and_offline_validation_bind_runtime_and_corpus(
    tmp_path,
    monkeypatch,
) -> None:
    protocol = _protocol()
    corpus = _corpus(protocol)
    corpus_path = tmp_path / "corpus.json"
    output = tmp_path / "artifact"
    _write_corpus(corpus_path, corpus)
    monkeypatch.setattr(
        lane,
        "_build_runtime_and_scorer",
        lambda **_kwargs: (_Runtime(protocol), _Scorer()),
    )

    created = lane.run_relationship_residual_fit(
        corpus_path=corpus_path,
        output_dir=output,
        protocol=protocol,
    )
    monkeypatch.setattr(
        lane,
        "_build_runtime_and_scorer",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("offline validation constructed a runtime")
        ),
    )
    validated = lane.validate_relationship_residual_fit(
        output_dir=output,
        protocol=protocol,
    )

    assert validated == created
    assert validated.corpus_id == corpus.corpus_id
    report = json.loads((output / "relationship_residual_fit_report.json").read_text())
    assert report["claim_flags"]["raw_model_strict_json_generation_proven"] is False
    assert report["claim_flags"]["user_visible_relationship_reply_changed"] is False
    assert report["claim_flags"]["four_able_complete"] is False


def test_run_rejects_runtime_weight_attestation_drift_before_fit(
    tmp_path,
    monkeypatch,
) -> None:
    protocol = _protocol()
    corpus_path = tmp_path / "corpus.json"
    output = tmp_path / "artifact"
    _write_corpus(corpus_path, _corpus(protocol))
    monkeypatch.setattr(
        lane,
        "_build_runtime_and_scorer",
        lambda **_kwargs: (_Runtime(protocol, weights_sha256="0" * 64), _Scorer()),
    )

    with pytest.raises(ValueError, match="model_weights_sha256 drift"):
        lane.run_relationship_residual_fit(
            corpus_path=corpus_path,
            output_dir=output,
            protocol=protocol,
        )
    assert not output.exists()
