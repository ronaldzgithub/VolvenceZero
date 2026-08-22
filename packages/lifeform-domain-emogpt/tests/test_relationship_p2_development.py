from __future__ import annotations

import dataclasses
import json

from lifeform_domain_emogpt.lab import (
    BoundedRelationshipPreferenceForecastRuntime,
    P2_DEVELOPMENT_SPLIT_ID,
    load_relationship_p2_development_evaluator_bundle,
    load_relationship_p2_development_view,
    run_p2_development_episode,
)


_FORBIDDEN_PUBLIC_KEYS = {
    "scene_id",
    "preferred_action",
    "policy_id",
    "condition_id",
    "probe_condition_id",
    "dynamic_id",
    "generator_truth",
    "future_outcome",
    "expected_action",
}


def test_p2_development_view_is_v3_only_incremental_and_truth_free() -> None:
    view = load_relationship_p2_development_view()

    assert view.contract.split_id == P2_DEVELOPMENT_SPLIT_ID
    assert view.contract.training_package_name == "relationship_transfer_v3"
    assert len(view.episodes) == 12
    assert all(len(episode.to_sut_sequence()) == 5 for episode in view.episodes)
    assert all(
        len(episode.history_sessions) == 4
        and episode.probe_session.session_index == 4
        for episode in view.episodes
    )
    public_payload = [
        episode.to_sut_sequence() for episode in view.episodes
    ]
    serialized = json.dumps(public_payload, ensure_ascii=False, sort_keys=True)
    assert not any(f'"{key}"' in serialized for key in _FORBIDDEN_PUBLIC_KEYS)
    assert "relationship_transfer_v4" not in serialized
    assert "histories" not in view.episodes[0].probe_session.to_sut_payload()


def test_p2_training_label_exists_only_in_separate_evaluator_type() -> None:
    view = load_relationship_p2_development_view()
    evaluator = load_relationship_p2_development_evaluator_bundle()

    assert not any(
        field.name == "preferred_action_id"
        for field in dataclasses.fields(type(view.episodes[0]))
    )
    assert len(evaluator.truths) == len(view.episodes)
    assert evaluator.truths[0].episode_id == view.episodes[0].episode_id
    assert evaluator.truths[0].preferred_action_id == "stay_present_without_probe"


async def test_p2_episode_hydrates_four_sessions_and_forecasts_without_replay() -> None:
    episode = load_relationship_p2_development_view().episodes[0]
    semantic_vectors = {
        episode.probe_session.current_observation: (1.0, 0.0),
        episode.history_sessions[0].observation_summary: (1.0, 0.0),
        episode.history_sessions[1].observation_summary: (0.0, 1.0),
        episode.history_sessions[2].observation_summary: (1.0, 0.0),
        episode.history_sessions[3].observation_summary: (0.0, 1.0),
    }

    def semantic_similarity(left: str, right: str) -> float:
        left_vector = semantic_vectors[left]
        right_vector = semantic_vectors[right]
        return sum(
            left_item * right_item
            for left_item, right_item in zip(
                left_vector,
                right_vector,
                strict=True,
            )
        )

    result = await run_p2_development_episode(
        episode,
        forecast_runtime=BoundedRelationshipPreferenceForecastRuntime(
            similarity=semantic_similarity,
        ),
    )

    assert result.forecast.recommended_action_id == "stay_present_without_probe"
    assert result.forecast.decision_id == episode.probe_session.decision_id
    assert result.persisted_record_count == 4
    assert result.persisted_action_outcome_count == 4
    assert len(result.persistence_payload_sha256) == 4
    assert len(set(result.persistence_payload_sha256)) == 4
    assert not result.raw_history_replayed_at_probe
    assert result.forecast.source_record_ids == tuple(
        session.event_id
        for session in (
            episode.history_sessions[0],
            episode.history_sessions[2],
            episode.history_sessions[1],
            episode.history_sessions[3],
        )
    )
