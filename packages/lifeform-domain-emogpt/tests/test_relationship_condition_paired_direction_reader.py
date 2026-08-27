from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import math

import pytest

from lifeform_domain_emogpt.lab import (
    load_relationship_p2_development_evaluator_bundle,
    load_relationship_p2_development_view,
    load_relationship_transfer_dataset,
    run_p2_development_episode,
)
from lifeform_domain_emogpt.relationship_condition_paired_direction_reader import (
    FrozenPairedDirectionRelationshipConditionReaderArtifact,
    FrozenPairedDirectionRelationshipConditionReaderRuntime,
    FrozenPairedDirectionRelationshipPreferenceForecastRuntime,
    MatchedRelationshipConditionEmbeddingPair,
    build_frozen_paired_direction_relationship_condition_reader_artifact,
)
from lifeform_domain_emogpt.relationship_condition_reader import (
    LabeledRelationshipConditionEmbeddingRow,
)


_MODEL_ID = "fixture-frozen-condition-encoder"
_MODEL_REVISION = "fixture-revision-1"
_WEIGHTS_SHA256 = "b" * 64
_RUNTIME_VERSION = "fixture-embedding-runtime.v1"
_LABELS = ("agency_pressure", "belonging_uncertainty")
_CORPUS_ARTIFACT_ID = "c" * 64
_CORPUS_RAW_SHA256 = "d" * 64
_GROUP_SPLIT_ARTIFACT_ID = "e" * 64
_SELECTION_RECEIPT_ARTIFACT_ID = "f" * 64


class _FrozenFixtureEmbedder:
    def __init__(
        self,
        vectors: dict[str, tuple[float, ...]],
        *,
        model_revision: str = _MODEL_REVISION,
    ) -> None:
        self.model_source = _MODEL_ID
        self.model_revision = model_revision
        self.weights_sha256 = _WEIGHTS_SHA256
        self.sentence_transformers_version = _RUNTIME_VERSION
        self._vectors = vectors
        self.calls: list[str] = []

    def embed(self, text: str) -> tuple[float, ...]:
        self.calls.append(text)
        return self._vectors[text]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _embedding_hex(*values: float) -> tuple[str, ...]:
    return tuple(float(value).hex() for value in values)


def _pairs() -> tuple[MatchedRelationshipConditionEmbeddingPair, ...]:
    return (
        MatchedRelationshipConditionEmbeddingPair(
            pair_id=_digest("pair-1"),
            semantic_group_id=_digest("group-1"),
            positive=LabeledRelationshipConditionEmbeddingRow(
                example_id=_digest("agency-1"),
                condition_label=_LABELS[0],
                embedding_hex=_embedding_hex(1.0, 0.0),
            ),
            negative=LabeledRelationshipConditionEmbeddingRow(
                example_id=_digest("belonging-1"),
                condition_label=_LABELS[1],
                embedding_hex=_embedding_hex(0.0, 1.0),
            ),
        ),
        MatchedRelationshipConditionEmbeddingPair(
            pair_id=_digest("pair-2"),
            semantic_group_id=_digest("group-2"),
            positive=LabeledRelationshipConditionEmbeddingRow(
                example_id=_digest("agency-2"),
                condition_label=_LABELS[0],
                embedding_hex=_embedding_hex(0.8, -0.6),
            ),
            negative=LabeledRelationshipConditionEmbeddingRow(
                example_id=_digest("belonging-2"),
                condition_label=_LABELS[1],
                embedding_hex=_embedding_hex(-0.6, 0.8),
            ),
        ),
    )


def _artifact(
    *,
    pairs: tuple[MatchedRelationshipConditionEmbeddingPair, ...] | None = None,
) -> FrozenPairedDirectionRelationshipConditionReaderArtifact:
    return build_frozen_paired_direction_relationship_condition_reader_artifact(
        embedding_model_id=_MODEL_ID,
        embedding_model_revision=_MODEL_REVISION,
        embedding_weights_sha256=_WEIGHTS_SHA256,
        embedding_runtime_version=_RUNTIME_VERSION,
        embedding_width=2,
        labels=_LABELS,
        condition_training_corpus_artifact_id=_CORPUS_ARTIFACT_ID,
        condition_training_corpus_raw_sha256=_CORPUS_RAW_SHA256,
        training_group_split_artifact_id=_GROUP_SPLIT_ARTIFACT_ID,
        training_selection_receipt_artifact_id=_SELECTION_RECEIPT_ARTIFACT_ID,
        pairs=_pairs() if pairs is None else pairs,
    )


def test_builder_is_deterministic_fixed_scale_and_training_only() -> None:
    artifact = _artifact()
    reversed_artifact = _artifact(pairs=tuple(reversed(_pairs())))
    direction = tuple(float.fromhex(value) for value in artifact.direction_hex)

    assert artifact == reversed_artifact
    assert artifact.artifact_id == (
        "c6e55625e578d2a34304f80d2fba46f4cec439a82513c1f3247d6691eaf9723d"
    )
    assert math.hypot(*direction) == pytest.approx(1.0, abs=1e-12)
    assert direction == pytest.approx(
        (1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0))
    )
    assert float.fromhex(artifact.threshold_hex) == pytest.approx(0.0, abs=1e-15)
    assert artifact.pair_count == artifact.semantic_group_count == 2
    parameters = set(
        inspect.signature(
            build_frozen_paired_direction_relationship_condition_reader_artifact
        ).parameters
    )
    assert not parameters & {
        "challenge",
        "action",
        "outcome",
        "prediction_error",
        "credit",
        "evaluation",
        "judge",
        "scale",
        "temperature",
    }


def test_owner_codec_roundtrips_and_rejects_stale_id_tamper() -> None:
    artifact = _artifact()

    assert FrozenPairedDirectionRelationshipConditionReaderArtifact.from_payload(
        artifact.to_payload()
    ) == artifact
    assert FrozenPairedDirectionRelationshipConditionReaderArtifact.from_json(
        artifact.to_json_bytes()
    ) == artifact
    tampered = artifact.to_payload()
    tampered["threshold_hex"] = 0.25.hex()
    with pytest.raises(ValueError, match="artifact_id mismatch"):
        FrozenPairedDirectionRelationshipConditionReaderArtifact.from_payload(
            tampered
        )
    missing = artifact.to_payload()
    missing.pop("training_selection_receipt_artifact_id")
    with pytest.raises(ValueError, match="keys mismatch"):
        FrozenPairedDirectionRelationshipConditionReaderArtifact.from_payload(missing)
    duplicate = artifact.to_json().replace(
        f'"artifact_id":"{artifact.artifact_id}"',
        (
            f'"artifact_id":"{artifact.artifact_id}",'
            f'"artifact_id":"{artifact.artifact_id}"'
        ),
        1,
    )
    with pytest.raises(ValueError, match="duplicate JSON key: artifact_id"):
        FrozenPairedDirectionRelationshipConditionReaderArtifact.from_json(duplicate)
    with pytest.raises(ValueError, match="canonical UTF-8 bytes"):
        FrozenPairedDirectionRelationshipConditionReaderArtifact.from_json(
            artifact.to_json()[:-1]
        )


def test_runtime_uses_geometric_margin_without_free_scale() -> None:
    embedder = _FrozenFixtureEmbedder(
        {
            "agency": (1.0, 0.0),
            "belonging": (0.0, 1.0),
            "boundary": (1.0, 1.0),
        }
    )
    runtime = FrozenPairedDirectionRelationshipConditionReaderRuntime(
        artifact=_artifact(),
        embedder=embedder,
    )

    agency = runtime.read_condition("agency")
    belonging = runtime.read_condition("belonging")
    boundary = runtime.read_condition("boundary")
    expected_score = 1.0 / (2.0 * math.sqrt(2.0))
    assert tuple(label for label, _ in agency.candidate_scores) == _LABELS
    assert tuple(score for _, score in agency.candidate_scores) == pytest.approx(
        (expected_score, -expected_score)
    )
    assert agency.condition_label == _LABELS[0]
    assert agency.normalized_margin == pytest.approx(expected_score)
    assert belonging.condition_label == _LABELS[1]
    assert belonging.normalized_margin == pytest.approx(expected_score)
    assert boundary.condition_label == _LABELS[0]
    assert boundary.normalized_margin == pytest.approx(0.0, abs=1e-15)
    assert "fit" not in type(runtime).__dict__
    assert runtime.read_condition("agency") is agency
    assert embedder.calls.count("agency") == 1


def test_builder_rejects_unmatched_or_ambiguous_training_shape() -> None:
    pairs = _pairs()
    with pytest.raises(ValueError, match="condition labels must differ"):
        replace(
            pairs[0],
            negative=replace(
                pairs[0].negative,
                condition_label=pairs[0].positive.condition_label,
            ),
        )
    with pytest.raises(ValueError, match="pair ids must be unique"):
        _artifact(pairs=(pairs[0], replace(pairs[1], pair_id=pairs[0].pair_id)))
    with pytest.raises(ValueError, match="globally unique"):
        _artifact(
            pairs=(
                pairs[0],
                replace(
                    pairs[1],
                    positive=replace(
                        pairs[1].positive,
                        example_id=pairs[0].positive.example_id,
                    ),
                ),
            )
        )
    with pytest.raises(ValueError, match="at least two semantic groups"):
        _artifact(
            pairs=(
                pairs[0],
                replace(
                    pairs[1],
                    semantic_group_id=pairs[0].semantic_group_id,
                ),
            )
        )
    with pytest.raises(ValueError, match="ordered artifact labels"):
        _artifact(
            pairs=(
                replace(
                    pairs[0],
                    positive=pairs[0].negative,
                    negative=pairs[0].positive,
                ),
                pairs[1],
            )
        )


def test_runtime_rejects_embedder_identity_drift() -> None:
    embedder = _FrozenFixtureEmbedder(
        {"unused": (1.0, 0.0)},
        model_revision="wrong-revision",
    )
    with pytest.raises(ValueError, match="identity does not match"):
        FrozenPairedDirectionRelationshipConditionReaderRuntime(
            artifact=_artifact(),
            embedder=embedder,
        )
    assert embedder.calls == []


async def test_adapter_publishes_named_readout_through_preference_owner() -> None:
    dataset = load_relationship_transfer_dataset(
        package_name="relationship_transfer_v3"
    )
    condition_vectors = {
        "latent_condition_agency_pressure_v3": (1.0, 0.0),
        "latent_condition_belonging_uncertainty_v3": (0.0, 1.0),
    }
    history_bindings = dict(dataset.history_condition_bindings)
    vectors: dict[str, tuple[float, ...]] = {}
    for observation in dataset.observations:
        for history in observation.histories:
            vectors[history.user_utterance] = condition_vectors[
                history_bindings[history.event_id]
            ]
        dynamic = dataset.dynamic_for_scene(observation.scene_id)
        assert dynamic.probe_condition_id is not None
        vectors[observation.current_input] = condition_vectors[
            dynamic.probe_condition_id
        ]
    reader = FrozenPairedDirectionRelationshipConditionReaderRuntime(
        artifact=_artifact(),
        embedder=_FrozenFixtureEmbedder(vectors),
    )
    runtime = FrozenPairedDirectionRelationshipPreferenceForecastRuntime(
        reader=reader
    )
    view = load_relationship_p2_development_view()
    evaluator = load_relationship_p2_development_evaluator_bundle()
    expected_actions = {
        item.episode_id: item.preferred_action_id for item in evaluator.truths
    }

    runs = tuple(
        [
            await run_p2_development_episode(
                episode,
                forecast_runtime=runtime,
            )
            for episode in view.episodes
        ]
    )
    assert all(
        run.forecast.recommended_action_id == expected_actions[run.episode_id]
        for run in runs
    )
    assert all(run.forecast.condition_readout is not None for run in runs)
    assert all(
        run.forecast.condition_readout.reader_artifact_id
        == runtime.artifact.artifact_id
        for run in runs
        if run.forecast.condition_readout is not None
    )
    assert all(
        f"runtime:{runtime.runtime_id}" in run.forecast.evidence for run in runs
    )
    with pytest.raises(
        TypeError,
        match="FrozenPairedDirectionRelationshipConditionReaderRuntime",
    ):
        FrozenPairedDirectionRelationshipPreferenceForecastRuntime(reader=object())
