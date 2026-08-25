from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import inspect
import json
import math

import pytest

from lifeform_domain_emogpt.lab import (
    load_relationship_p2_development_evaluator_bundle,
    load_relationship_p2_development_view,
    load_relationship_transfer_dataset,
    run_p2_development_episode,
)
from lifeform_domain_emogpt.relationship_condition_reader import (
    FrozenLinearRelationshipConditionReaderArtifact,
    FrozenLinearRelationshipConditionReaderRuntime,
    LabeledRelationshipConditionEmbeddingRow,
    PrototypeRelationshipPreferenceForecastRuntime,
    RelationshipConditionPrototype,
    RelationshipConditionLinearClassParameters,
    RelationshipConditionReaderArtifact,
    build_frozen_linear_relationship_condition_reader_artifact,
)
from volvence_zero.runtime import WiringLevel
from volvence_zero.owner_hydration import (
    HydrationPayloadInvalidError,
    OwnerPersistenceSnapshot,
)
from volvence_zero.social import (
    PreferenceAboutOtherModule,
    PreferenceActionForecastProposal,
    SocialRecordStore,
)
from volvence_zero.social_cognition import (
    SocialActionCandidatePrediction,
    SocialActionOutcomeProbability,
)


_WEIGHTS_SHA256 = "a" * 64
_LINEAR_WEIGHTS_SHA256 = "b" * 64
_CONDITION_CORPUS_ARTIFACT_ID = "c" * 64
_CONDITION_CORPUS_RAW_SHA256 = "d" * 64
_GROUP_SPLIT_ARTIFACT_ID = "e" * 64
_LINEAR_MODEL_ID = "fixture-frozen-condition-encoder"
_LINEAR_MODEL_REVISION = "fixture-revision-1"
_LINEAR_RUNTIME_VERSION = "fixture-embedding-runtime.v1"
_LINEAR_LABELS = ("agency_pressure", "belonging_uncertainty")
_CONDITION_LABELS = {
    "latent_condition_agency_pressure_v3": "agency_pressure",
    "latent_condition_belonging_uncertainty_v3": "belonging_uncertainty",
}


class _FixtureEmbedder:
    def __init__(self, vectors: dict[str, tuple[float, ...]]) -> None:
        self._vectors = vectors

    def embed(self, text: str) -> tuple[float, ...]:
        return self._vectors[text]


class _FrozenFixtureEmbedder:
    def __init__(
        self,
        vectors: dict[str, tuple[float, ...]],
        *,
        model_source: str = _LINEAR_MODEL_ID,
        model_revision: str = _LINEAR_MODEL_REVISION,
        weights_sha256: str = _LINEAR_WEIGHTS_SHA256,
        sentence_transformers_version: str = _LINEAR_RUNTIME_VERSION,
    ) -> None:
        self.model_source = model_source
        self.model_revision = model_revision
        self.weights_sha256 = weights_sha256
        self.sentence_transformers_version = sentence_transformers_version
        self._vectors = vectors
        self.calls: list[str] = []

    def embed(self, text: str) -> tuple[float, ...]:
        self.calls.append(text)
        return self._vectors[text]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _embedding_hex(*values: float) -> tuple[str, ...]:
    return tuple(float(value).hex() for value in values)


def _linear_rows() -> tuple[LabeledRelationshipConditionEmbeddingRow, ...]:
    return (
        LabeledRelationshipConditionEmbeddingRow(
            example_id=_digest("agency-example-1"),
            condition_label="agency_pressure",
            embedding_hex=_embedding_hex(1.0, 0.0),
        ),
        LabeledRelationshipConditionEmbeddingRow(
            example_id=_digest("agency-example-2"),
            condition_label="agency_pressure",
            embedding_hex=_embedding_hex(2.0, 0.0),
        ),
        LabeledRelationshipConditionEmbeddingRow(
            example_id=_digest("belonging-example-1"),
            condition_label="belonging_uncertainty",
            embedding_hex=_embedding_hex(0.0, 1.0),
        ),
        LabeledRelationshipConditionEmbeddingRow(
            example_id=_digest("belonging-example-2"),
            condition_label="belonging_uncertainty",
            embedding_hex=_embedding_hex(0.0, 2.0),
        ),
    )


def _linear_artifact(
    *,
    rows: tuple[LabeledRelationshipConditionEmbeddingRow, ...] | None = None,
    embedding_weights_sha256: str = _LINEAR_WEIGHTS_SHA256,
) -> FrozenLinearRelationshipConditionReaderArtifact:
    return build_frozen_linear_relationship_condition_reader_artifact(
        embedding_model_id=_LINEAR_MODEL_ID,
        embedding_model_revision=_LINEAR_MODEL_REVISION,
        embedding_weights_sha256=embedding_weights_sha256,
        embedding_runtime_version=_LINEAR_RUNTIME_VERSION,
        embedding_width=2,
        labels=_LINEAR_LABELS,
        condition_training_corpus_artifact_id=_CONDITION_CORPUS_ARTIFACT_ID,
        condition_training_corpus_raw_sha256=_CONDITION_CORPUS_RAW_SHA256,
        group_split_artifact_id=_GROUP_SPLIT_ARTIFACT_ID,
        rows=_linear_rows() if rows is None else rows,
    )


def _fixture_runtime() -> PrototypeRelationshipPreferenceForecastRuntime:
    dataset = load_relationship_transfer_dataset(
        package_name="relationship_transfer_v3"
    )
    conditions = {item.condition_id: item for item in dataset.abstract_conditions}
    condition_vectors = {
        "latent_condition_agency_pressure_v3": (1.0, 0.0),
        "latent_condition_belonging_uncertainty_v3": (0.0, 1.0),
    }
    vectors = {
        conditions[condition_id].hidden_summary: vector
        for condition_id, vector in condition_vectors.items()
    }
    history_bindings = dict(dataset.history_condition_bindings)
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
    artifact = RelationshipConditionReaderArtifact(
        embedding_model_id="fixture-semantic-encoder",
        embedding_weights_sha256=_WEIGHTS_SHA256,
        prototypes=tuple(
            RelationshipConditionPrototype(
                label=_CONDITION_LABELS[condition_id],
                summary=conditions[condition_id].hidden_summary,
            )
            for condition_id in sorted(conditions)
        ),
    )
    return PrototypeRelationshipPreferenceForecastRuntime(
        artifact=artifact,
        embedder=_FixtureEmbedder(vectors),
    )


def test_frozen_linear_reader_builder_is_deterministic_and_content_addressed() -> None:
    artifact = _linear_artifact()
    reversed_artifact = _linear_artifact(rows=tuple(reversed(_linear_rows())))

    assert artifact == reversed_artifact
    assert artifact.artifact_id == reversed_artifact.artifact_id
    assert artifact.artifact_id == (
        "904529084332d60b99742a16c91458d9d946f41ab7f227a8bb66ca2910720659"
    )
    assert artifact.to_payload()["artifact_id"] == artifact.artifact_id
    assert artifact.embedding_model_revision == _LINEAR_MODEL_REVISION
    assert artifact.embedding_runtime_version == _LINEAR_RUNTIME_VERSION
    assert artifact.labels == _LINEAR_LABELS
    assert tuple(item.example_count for item in artifact.class_parameters) == (2, 2)
    assert all(
        item.centroid_hex == item.coefficient_hex
        and item.bias_hex == 0.0.hex()
        and all(float.fromhex(value).hex() == value for value in item.centroid_hex)
        for item in artifact.class_parameters
    )
    assert _linear_artifact(
        embedding_weights_sha256="f" * 64
    ).artifact_id != artifact.artifact_id
    builder_parameters = set(
        inspect.signature(
            build_frozen_linear_relationship_condition_reader_artifact
        ).parameters
    )
    assert builder_parameters == {
        "embedding_model_id",
        "embedding_model_revision",
        "embedding_weights_sha256",
        "embedding_runtime_version",
        "embedding_width",
        "labels",
        "condition_training_corpus_artifact_id",
        "condition_training_corpus_raw_sha256",
        "group_split_artifact_id",
        "rows",
    }
    assert not builder_parameters & {
        "action",
        "outcome",
        "prediction_error",
        "credit",
        "evaluation",
        "judge",
    }


def test_frozen_linear_reader_owner_serialization_roundtrips_canonical_bytes() -> None:
    artifact = _linear_artifact()
    class_parameters = artifact.class_parameters[0]

    assert RelationshipConditionLinearClassParameters.from_payload(
        class_parameters.to_payload()
    ) == class_parameters
    assert FrozenLinearRelationshipConditionReaderArtifact.from_payload(
        artifact.to_payload()
    ) == artifact
    assert artifact.to_json().endswith("\n")
    assert artifact.to_json_bytes() == artifact.to_json().encode("utf-8")
    assert FrozenLinearRelationshipConditionReaderArtifact.from_json(
        artifact.to_json()
    ) == artifact
    assert FrozenLinearRelationshipConditionReaderArtifact.from_json(
        artifact.to_json_bytes()
    ) == artifact
    assert artifact.artifact_id == (
        "904529084332d60b99742a16c91458d9d946f41ab7f227a8bb66ca2910720659"
    )


def test_frozen_linear_reader_owner_loader_rejects_tamper_and_schema_drift() -> None:
    artifact = _linear_artifact()
    payload = artifact.to_payload()

    tampered = artifact.to_payload()
    tampered["embedding_model_revision"] = "tampered-revision"
    with pytest.raises(ValueError, match="artifact_id mismatch"):
        FrozenLinearRelationshipConditionReaderArtifact.from_payload(tampered)

    missing = dict(payload)
    missing.pop("labels")
    with pytest.raises(ValueError, match="keys mismatch"):
        FrozenLinearRelationshipConditionReaderArtifact.from_payload(missing)
    with pytest.raises(ValueError, match="keys mismatch"):
        FrozenLinearRelationshipConditionReaderArtifact.from_payload(
            {**payload, "unexpected": True}
        )
    with pytest.raises(ValueError, match="JSON array"):
        FrozenLinearRelationshipConditionReaderArtifact.from_payload(
            {**payload, "labels": artifact.labels}
        )

    class_payload = artifact.class_parameters[0].to_payload()
    missing_class = dict(class_payload)
    missing_class.pop("bias_hex")
    with pytest.raises(ValueError, match="keys mismatch"):
        RelationshipConditionLinearClassParameters.from_payload(missing_class)
    with pytest.raises(ValueError, match="keys mismatch"):
        RelationshipConditionLinearClassParameters.from_payload(
            {**class_payload, "unexpected": True}
        )


def test_frozen_linear_reader_owner_loader_rejects_noncanonical_and_duplicate_json() -> None:
    artifact = _linear_artifact()
    canonical = artifact.to_json()
    duplicate = canonical.replace(
        f'"artifact_id":"{artifact.artifact_id}"',
        (
            f'"artifact_id":"{artifact.artifact_id}",'
            f'"artifact_id":"{artifact.artifact_id}"'
        ),
        1,
    )

    with pytest.raises(ValueError, match="duplicate JSON key: artifact_id"):
        FrozenLinearRelationshipConditionReaderArtifact.from_json(duplicate)
    with pytest.raises(ValueError, match="canonical UTF-8 bytes"):
        FrozenLinearRelationshipConditionReaderArtifact.from_json(canonical[:-1])
    with pytest.raises(ValueError, match="canonical UTF-8 bytes"):
        FrozenLinearRelationshipConditionReaderArtifact.from_json(
            json.dumps(artifact.to_payload(), ensure_ascii=False, indent=2) + "\n"
        )
    with pytest.raises(ValueError, match="exact UTF-8"):
        FrozenLinearRelationshipConditionReaderArtifact.from_json(b"\xff")
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        FrozenLinearRelationshipConditionReaderArtifact.from_json(
            canonical.replace('"embedding_width":2', '"embedding_width":NaN')
        )


def test_frozen_linear_reader_runtime_reads_all_scores_and_caches_frozen_output() -> None:
    vectors = {
        "clear agency": (1.0, 0.0),
        "clear belonging": (0.0, 1.0),
        "agency text with wrong embedding": (0.0, 1.0),
        "unknown condition": (1.0, 1.0),
    }
    embedder = _FrozenFixtureEmbedder(vectors)
    artifact = _linear_artifact()
    runtime = FrozenLinearRelationshipConditionReaderRuntime(
        artifact=artifact,
        embedder=embedder,
    )

    assert embedder.calls == []
    assert runtime.runtime_id == (
        "relationship-condition-linear-reader.v2:"
        "904529084332d60b99742a16c91458d9d946f41ab7f227a8bb66ca2910720659"
    )
    assert "fit" not in type(runtime).__dict__
    assert set(inspect.signature(runtime.read_condition).parameters) == {"text"}
    agency = runtime.read_condition("clear agency")
    belonging = runtime.read_condition("clear belonging")
    wrong = runtime.read_condition("agency text with wrong embedding")
    unknown = runtime.read_condition("unknown condition")

    assert agency.condition_label == "agency_pressure"
    assert agency.candidate_scores == (
        ("agency_pressure", 1.0),
        ("belonging_uncertainty", 0.0),
    )
    assert agency.normalized_margin == pytest.approx(0.5)
    assert agency.confidence == pytest.approx(math.e / (math.e + 1.0))
    assert belonging.condition_label == "belonging_uncertainty"
    assert wrong.condition_label == "belonging_uncertainty"
    assert wrong.condition_label != "agency_pressure"
    assert unknown.condition_label == "agency_pressure"
    assert tuple(label for label, _ in unknown.candidate_scores) == _LINEAR_LABELS
    assert unknown.candidate_scores[0][1] == pytest.approx(
        unknown.candidate_scores[1][1]
    )
    assert unknown.normalized_margin == pytest.approx(0.0)
    assert unknown.confidence == pytest.approx(0.5)
    assert agency.reader_artifact_id == artifact.artifact_id
    assert agency.source_observation_sha256 == _digest("clear agency")
    assert runtime.read_condition("clear agency") is agency
    assert embedder.calls.count("clear agency") == 1
    with pytest.raises(FrozenInstanceError):
        agency.confidence = 0.0


def test_frozen_linear_reader_rejects_invalid_offline_rows_and_pins() -> None:
    with pytest.raises(ValueError, match="canonical float hex"):
        LabeledRelationshipConditionEmbeddingRow(
            example_id=_digest("noncanonical"),
            condition_label="agency_pressure",
            embedding_hex=("1.0", 0.0.hex()),
        )
    with pytest.raises(ValueError, match="finite canonical float hex"):
        LabeledRelationshipConditionEmbeddingRow(
            example_id=_digest("nonfinite"),
            condition_label="agency_pressure",
            embedding_hex=(float("nan").hex(), 0.0.hex()),
        )
    with pytest.raises(ValueError, match="norm must be positive"):
        LabeledRelationshipConditionEmbeddingRow(
            example_id=_digest("zero"),
            condition_label="agency_pressure",
            embedding_hex=_embedding_hex(0.0, 0.0),
        )

    rows = _linear_rows()
    duplicate_rows = (
        rows[0],
        replace(rows[1], example_id=rows[0].example_id),
        *rows[2:],
    )
    with pytest.raises(ValueError, match="example ids must be unique"):
        _linear_artifact(rows=duplicate_rows)
    with pytest.raises(ValueError, match="labels must match exactly"):
        _linear_artifact(rows=rows[:2])
    with pytest.raises(ValueError, match="embedding width drift"):
        _linear_artifact(
            rows=(
                replace(rows[0], embedding_hex=_embedding_hex(1.0, 0.0, 0.0)),
                *rows[1:],
            )
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _linear_artifact(embedding_weights_sha256="F" * 64)

    artifact = _linear_artifact()
    with pytest.raises(ValueError, match="strict UTF-8 byte order"):
        replace(artifact, labels=tuple(reversed(artifact.labels)))
    with pytest.raises(ValueError, match="parameter labels must match"):
        replace(
            artifact,
            class_parameters=tuple(reversed(artifact.class_parameters)),
        )
    with pytest.raises(ValueError, match="parameter width drift"):
        replace(artifact, embedding_width=3)
    nonfinite_parameters = (
        float("nan").hex(),
        artifact.class_parameters[0].centroid_hex[1],
    )
    with pytest.raises(ValueError, match="finite canonical float hex"):
        replace(
            artifact.class_parameters[0],
            centroid_hex=nonfinite_parameters,
            coefficient_hex=nonfinite_parameters,
        )
    with pytest.raises(ValueError, match="finite canonical float hex"):
        replace(
            artifact.class_parameters[0],
            bias_hex=(-0.0).hex(),
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        replace(artifact, group_split_artifact_id="not-a-pin")


@pytest.mark.parametrize(
    ("vector", "message"),
    [
        ((1.0,), "width drift"),
        ((float("nan"), 0.0), "finite numeric"),
        ((True, 0.0), "finite numeric"),
        ((0.0, 0.0), "norm must be positive"),
    ],
)
def test_frozen_linear_reader_runtime_rejects_invalid_embedder_output(
    vector: tuple[float, ...],
    message: str,
) -> None:
    embedder = _FrozenFixtureEmbedder({"invalid": vector})
    runtime = FrozenLinearRelationshipConditionReaderRuntime(
        artifact=_linear_artifact(),
        embedder=embedder,
    )

    with pytest.raises(ValueError, match=message):
        runtime.read_condition("invalid")


def test_frozen_linear_reader_runtime_rejects_embedder_identity_drift_without_fit() -> None:
    embedder = _FrozenFixtureEmbedder(
        {"unused": (1.0, 0.0)},
        model_revision="different-revision",
    )

    with pytest.raises(ValueError, match="identity does not match"):
        FrozenLinearRelationshipConditionReaderRuntime(
            artifact=_linear_artifact(),
            embedder=embedder,
        )
    assert embedder.calls == []


def test_prototype_v1_artifact_and_runtime_identity_remain_pinned() -> None:
    artifact = RelationshipConditionReaderArtifact(
        embedding_model_id="fixture-semantic-encoder",
        embedding_weights_sha256=_WEIGHTS_SHA256,
        prototypes=(
            RelationshipConditionPrototype(
                label="agency_pressure",
                summary="agency summary",
            ),
            RelationshipConditionPrototype(
                label="belonging_uncertainty",
                summary="belonging summary",
            ),
        ),
    )
    runtime = PrototypeRelationshipPreferenceForecastRuntime(
        artifact=artifact,
        embedder=_FixtureEmbedder(
            {
                "agency summary": (1.0, 0.0),
                "belonging summary": (0.0, 1.0),
            }
        ),
    )

    assert artifact.artifact_id == (
        "d4995138ff1ef1cd0d4ad742ae8e15825f9293a3a30e0b4dddf899a92f46827c"
    )
    assert runtime.runtime_id == (
        "relationship-p2-prototype-condition-forecast.v1:"
        "d4995138ff1ef1cd0d4ad742ae8e15825f9293a3a30e0b4dddf899a92f46827c"
    )


async def test_prototype_reader_names_condition_and_transfers_across_surfaces() -> None:
    view = load_relationship_p2_development_view()
    evaluator = load_relationship_p2_development_evaluator_bundle()
    expected_actions = {
        item.episode_id: item.preferred_action_id for item in evaluator.truths
    }
    runtime = _fixture_runtime()

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
    assert {
        run.forecast.condition_readout.condition_label
        for run in runs
        if run.forecast.condition_readout is not None
    } == {"agency_pressure", "belonging_uncertainty"}
    assert all(
        run.forecast.condition_readout.reader_artifact_id
        == runtime.artifact.artifact_id
        for run in runs
        if run.forecast.condition_readout is not None
    )
    predictions = tuple(run.forecast.recommended_action_id for run in runs)
    assert all(
        predictions[index] != predictions[index + 1]
        for index in range(0, len(predictions), 2)
    )
    store = SocialRecordStore()
    store.set_preference_action_forecasts((runs[0].forecast,))
    persisted = store.export_persistence_snapshot()
    assert persisted.schema_version == 4
    restored = SocialRecordStore()
    restored.hydrate_from_persistence(persisted)
    assert restored.preference_action_forecasts == (runs[0].forecast,)

    broken_forecast = dict(persisted.payload["preference_action_forecasts"][0])
    del broken_forecast["condition_readout"]
    broken_payload = {
        **persisted.payload,
        "preference_action_forecasts": [broken_forecast],
    }
    with pytest.raises(
        HydrationPayloadInvalidError,
        match="condition_readout",
    ):
        SocialRecordStore().hydrate_from_persistence(
            OwnerPersistenceSnapshot(
                owner_name="social_record_store",
                schema_version=4,
                payload=broken_payload,
            )
        )


async def test_preference_owner_rejects_condition_readout_for_other_observation() -> None:
    episode = load_relationship_p2_development_view().episodes[0]
    runtime = _fixture_runtime()
    request = episode.probe_session.to_request()
    wrong_readout = replace(
        runtime.read_condition(request.current_observation),
        source_observation_sha256="b" * 64,
    )

    class _WrongLineageRuntime:
        runtime_id = "wrong-condition-lineage"

        def propose(self, **kwargs) -> PreferenceActionForecastProposal:
            supplied_request = kwargs["request"]
            uniform = 1.0 / len(supplied_request.outcome_ids)
            return PreferenceActionForecastProposal(
                candidate_predictions=tuple(
                    SocialActionCandidatePrediction(
                        action_id=action_id,
                        outcomes=tuple(
                            SocialActionOutcomeProbability(
                                outcome_id=outcome_id,
                                probability=uniform,
                            )
                            for outcome_id in supplied_request.outcome_ids
                        ),
                    )
                    for action_id in supplied_request.candidate_action_ids
                ),
                recommended_action_id=supplied_request.candidate_action_ids[0],
                confidence=0.5,
                source_record_ids=(),
                evidence=("fixture:wrong-lineage",),
                condition_readout=wrong_readout,
            )

    store = SocialRecordStore()
    with pytest.raises(ValueError, match="different current observation"):
        await PreferenceAboutOtherModule(
            turn_index=request.turn_index,
            wiring_level=WiringLevel.SHADOW,
            record_store=store,
            action_forecast_runtime=_WrongLineageRuntime(),
            action_forecast_request=request,
        ).process({})
