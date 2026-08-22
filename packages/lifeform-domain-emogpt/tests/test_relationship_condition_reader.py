from __future__ import annotations

from dataclasses import replace

import pytest

from lifeform_domain_emogpt.lab import (
    load_relationship_p2_development_evaluator_bundle,
    load_relationship_p2_development_view,
    load_relationship_transfer_dataset,
    run_p2_development_episode,
)
from lifeform_domain_emogpt.relationship_condition_reader import (
    PrototypeRelationshipPreferenceForecastRuntime,
    RelationshipConditionPrototype,
    RelationshipConditionReaderArtifact,
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
_CONDITION_LABELS = {
    "latent_condition_agency_pressure_v3": "agency_pressure",
    "latent_condition_belonging_uncertainty_v3": "belonging_uncertainty",
}


class _FixtureEmbedder:
    def __init__(self, vectors: dict[str, tuple[float, ...]]) -> None:
        self._vectors = vectors

    def embed(self, text: str) -> tuple[float, ...]:
        return self._vectors[text]


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
