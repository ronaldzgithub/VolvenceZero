"""P2-development multi-session SHADOW lane for relationship forecasts.

Only the v3 consumer-training surface is materialized here. Four public
history events are admitted one at a time through the preference owner, with
an export/hydrate boundary after every event. The fifth session receives only
the current probe plus the hydrated owner state. Sealed labels live in the
separate evaluator bundle at the bottom of this module.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from volvence_zero.runtime import WiringLevel
from volvence_zero.semantic_state import (
    SemanticProposal,
    SemanticProposalBatch,
    SemanticProposalOperation,
    SemanticProposalRuntime,
)
from volvence_zero.social import (
    PreferenceAboutOtherModule,
    PreferenceActionForecastRequest,
    PreferenceActionForecastRuntime,
    SocialRecordStore,
)
from volvence_zero.social_cognition import (
    PreferenceActionForecast,
    PreferenceActionOutcomeEvidence,
    PreferenceAboutOtherSnapshot,
)

from lifeform_domain_emogpt.lab.consumer_split import (
    RelationshipConsumerTrainingView,
    load_relationship_consumer_training_view,
)
from lifeform_domain_emogpt.lab.contracts import (
    RELATIONSHIP_ACTIONS,
    RELATIONSHIP_OUTCOMES,
    sha256_json,
)
from lifeform_domain_emogpt.relationship_forecast import (
    BoundedRelationshipPreferenceForecastRuntime,
    RELATIONSHIP_PREFERENCE_FORECAST_RUNTIME_ID,
)


P2_DEVELOPMENT_SCHEMA_VERSION = "relationship-p2-development-contract.v1"
P2_DEVELOPMENT_SPLIT_ID = "P2-development"
P2_DEVELOPMENT_RUNTIME_ID = RELATIONSHIP_PREFERENCE_FORECAST_RUNTIME_ID
_PREFERENCE_SLOT = "preference_about_other"
_INTERLOCUTOR_ID = "primary"
_EXPECTED_EPISODE_COUNT = 12
_EVIDENCE_SESSIONS_PER_EPISODE = 4
_PROBE_SESSIONS_PER_EPISODE = 1
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
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
)

def relationship_p2_development_contract_path() -> pathlib.Path:
    return (
        pathlib.Path(__file__).resolve().parents[1]
        / "lab_protocols"
        / "relationship_p2_development_v1.json"
    )


@dataclass(frozen=True)
class RelationshipP2DevelopmentContract:
    contract_sha256: str
    consumer_split_contract_id: str
    training_package_name: str
    training_dataset_fingerprint: str
    split_id: str
    expected_episode_count: int
    evidence_sessions_per_episode: int
    probe_sessions_per_episode: int
    candidate_action_ids: tuple[str, ...]
    outcome_ids: tuple[str, ...]
    claim_boundary: str
    schema_version: str = P2_DEVELOPMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != P2_DEVELOPMENT_SCHEMA_VERSION:
            raise ValueError("P2-development schema version mismatch")
        for field_name, value in (
            ("contract_sha256", self.contract_sha256),
            ("consumer_split_contract_id", self.consumer_split_contract_id),
            ("training_dataset_fingerprint", self.training_dataset_fingerprint),
        ):
            _require_sha256(value, field_name)
        if self.training_package_name != "relationship_transfer_v3":
            raise ValueError("P2-development may only consume relationship_transfer_v3")
        if self.split_id != P2_DEVELOPMENT_SPLIT_ID:
            raise ValueError("P2-development split id is not frozen")
        if self.expected_episode_count != _EXPECTED_EPISODE_COUNT:
            raise ValueError("P2-development must contain exactly 12 episodes")
        if self.evidence_sessions_per_episode != _EVIDENCE_SESSIONS_PER_EPISODE:
            raise ValueError("P2-development requires four evidence sessions")
        if self.probe_sessions_per_episode != _PROBE_SESSIONS_PER_EPISODE:
            raise ValueError("P2-development requires one probe session")
        if self.candidate_action_ids != tuple(
            action.value for action in RELATIONSHIP_ACTIONS
        ):
            raise ValueError("P2-development candidate action surface drifted")
        if self.outcome_ids != tuple(outcome.value for outcome in RELATIONSHIP_OUTCOMES):
            raise ValueError("P2-development typed outcome surface drifted")
        _require_text(self.claim_boundary, "claim_boundary")


@dataclass(frozen=True)
class P2DevelopmentHistorySession:
    episode_id: str
    session_id: str
    session_index: int
    event_id: str
    observation_summary: str
    action_id: str
    observed_outcome_id: str
    reaction_summary: str
    observation_ref: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("episode_id", self.episode_id),
            ("session_id", self.session_id),
            ("event_id", self.event_id),
            ("observation_summary", self.observation_summary),
            ("action_id", self.action_id),
            ("observed_outcome_id", self.observed_outcome_id),
            ("reaction_summary", self.reaction_summary),
            ("observation_ref", self.observation_ref),
        ):
            _require_text(value, field_name)
        if self.session_index < 0:
            raise ValueError("history session_index must be >= 0")
        if self.action_id not in {action.value for action in RELATIONSHIP_ACTIONS}:
            raise ValueError("history action_id is outside the frozen surface")
        if self.observed_outcome_id not in {
            outcome.value for outcome in RELATIONSHIP_OUTCOMES
        }:
            raise ValueError("history observed_outcome_id is outside the frozen surface")

    def to_owner_evidence(self) -> PreferenceActionOutcomeEvidence:
        return PreferenceActionOutcomeEvidence(
            evidence_id=self.event_id,
            interlocutor_id=_INTERLOCUTOR_ID,
            observation_summary=self.observation_summary,
            action_id=self.action_id,
            observed_outcome_id=self.observed_outcome_id,
            reaction_summary=self.reaction_summary,
            source_turn=self.session_index,
            evidence_refs=(self.observation_ref,),
        )

    def to_sut_payload(self) -> dict[str, object]:
        return {
            "schema_version": "relationship-p2-history-session.v1",
            "session_id": self.session_id,
            "session_index": self.session_index,
            "event_id": self.event_id,
            "observation_summary": self.observation_summary,
            "action_id": self.action_id,
            "observed_outcome_id": self.observed_outcome_id,
            "reaction_summary": self.reaction_summary,
            "observation_ref": self.observation_ref,
        }


@dataclass(frozen=True)
class P2DevelopmentProbeSession:
    episode_id: str
    session_id: str
    session_index: int
    decision_id: str
    current_observation: str
    observation_ref: str
    candidate_action_ids: tuple[str, ...]
    outcome_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name, value in (
            ("episode_id", self.episode_id),
            ("session_id", self.session_id),
            ("decision_id", self.decision_id),
            ("current_observation", self.current_observation),
            ("observation_ref", self.observation_ref),
        ):
            _require_text(value, field_name)
        if self.session_index != _EVIDENCE_SESSIONS_PER_EPISODE:
            raise ValueError("P2-development probe must be the fifth session")
        if self.candidate_action_ids != tuple(
            action.value for action in RELATIONSHIP_ACTIONS
        ):
            raise ValueError("probe candidate action surface drifted")
        if self.outcome_ids != tuple(outcome.value for outcome in RELATIONSHIP_OUTCOMES):
            raise ValueError("probe typed outcome surface drifted")

    def to_request(self) -> PreferenceActionForecastRequest:
        return PreferenceActionForecastRequest(
            decision_id=self.decision_id,
            interlocutor_id=_INTERLOCUTOR_ID,
            current_observation=self.current_observation,
            observation_ref=self.observation_ref,
            candidate_action_ids=self.candidate_action_ids,
            outcome_ids=self.outcome_ids,
            turn_index=self.session_index,
            session_scope=self.episode_id,
        )

    def to_sut_payload(self) -> dict[str, object]:
        return {
            "schema_version": "relationship-p2-probe-session.v1",
            "session_id": self.session_id,
            "session_index": self.session_index,
            "decision_id": self.decision_id,
            "current_observation": self.current_observation,
            "observation_ref": self.observation_ref,
            "candidate_action_ids": list(self.candidate_action_ids),
            "outcome_ids": list(self.outcome_ids),
        }


@dataclass(frozen=True)
class P2DevelopmentEpisode:
    episode_id: str
    history_sessions: tuple[P2DevelopmentHistorySession, ...]
    probe_session: P2DevelopmentProbeSession

    def __post_init__(self) -> None:
        if len(self.history_sessions) != _EVIDENCE_SESSIONS_PER_EPISODE:
            raise ValueError("P2-development episode requires four histories")
        if tuple(item.session_index for item in self.history_sessions) != tuple(
            range(_EVIDENCE_SESSIONS_PER_EPISODE)
        ):
            raise ValueError("P2-development history sessions must be contiguous")
        if any(item.episode_id != self.episode_id for item in self.history_sessions):
            raise ValueError("history session episode lineage mismatch")
        if self.probe_session.episode_id != self.episode_id:
            raise ValueError("probe session episode lineage mismatch")

    def to_sut_sequence(self) -> tuple[dict[str, object], ...]:
        return (
            *(item.to_sut_payload() for item in self.history_sessions),
            self.probe_session.to_sut_payload(),
        )


@dataclass(frozen=True)
class RelationshipP2DevelopmentView:
    contract: RelationshipP2DevelopmentContract
    episodes: tuple[P2DevelopmentEpisode, ...]

    def __post_init__(self) -> None:
        if len(self.episodes) != self.contract.expected_episode_count:
            raise ValueError("P2-development episode count does not match contract")
        episode_ids = tuple(item.episode_id for item in self.episodes)
        if len(set(episode_ids)) != len(episode_ids):
            raise ValueError("P2-development episode ids must be unique")
        _assert_no_public_truth_leakage(self)


class RelationshipHistoryPreferenceProposalRuntime(SemanticProposalRuntime):
    """Admit one public history event through the preference owner."""

    def __init__(self, session: P2DevelopmentHistorySession) -> None:
        self._session = session
        self.runtime_id = f"relationship-p2-history:{session.event_id}"

    def propose(
        self,
        *,
        target_slot: str,
        user_input: str | None,
        substrate_snapshot: object | None,
        memory_snapshot: object | None,
        previous_snapshot: object | None,
        turn_index: int,
    ) -> SemanticProposalBatch:
        del substrate_snapshot, memory_snapshot, previous_snapshot
        if target_slot != _PREFERENCE_SLOT:
            raise ValueError("P2 history runtime only serves preference_about_other")
        if turn_index != self._session.session_index:
            raise ValueError("P2 history runtime turn lineage mismatch")
        if user_input != self._session.observation_summary:
            raise ValueError("P2 history runtime input differs from typed session")
        return SemanticProposalBatch(
            proposals=(
                SemanticProposal(
                    proposal_id=self._session.event_id,
                    target_slot=_PREFERENCE_SLOT,
                    operation=SemanticProposalOperation.OBSERVE,
                    summary=self._session.observation_summary,
                    detail=self._session.reaction_summary,
                    confidence=0.80,
                    evidence=self._session.observation_ref,
                    control_signal=0.0,
                ),
            ),
            runtime_id=self.runtime_id,
            schema_version=1,
            description="One typed P2 history event proposed to its owner.",
        )


@dataclass(frozen=True)
class P2DevelopmentEpisodeRun:
    episode_id: str
    forecast: PreferenceActionForecast
    persistence_payload_sha256: tuple[str, ...]
    persisted_record_count: int
    persisted_action_outcome_count: int
    raw_history_replayed_at_probe: bool = False
    wiring_level: WiringLevel = WiringLevel.SHADOW

    def __post_init__(self) -> None:
        if self.wiring_level is not WiringLevel.SHADOW:
            raise ValueError("P2-development run must remain SHADOW")
        if self.raw_history_replayed_at_probe:
            raise ValueError("P2-development probe cannot replay raw history")
        if len(self.persistence_payload_sha256) != _EVIDENCE_SESSIONS_PER_EPISODE:
            raise ValueError("P2-development must persist after every evidence session")
        for digest in self.persistence_payload_sha256:
            _require_sha256(digest, "persistence_payload_sha256")


async def run_p2_development_episode(
    episode: P2DevelopmentEpisode,
    *,
    forecast_runtime: PreferenceActionForecastRuntime | None = None,
) -> P2DevelopmentEpisodeRun:
    """Run four process-restart boundaries and one probe-only session."""

    persistence_hashes: list[str] = []
    persistence_snapshot = None
    for session in episode.history_sessions:
        store = SocialRecordStore()
        if persistence_snapshot is not None:
            store.hydrate_from_persistence(persistence_snapshot)
        owner = PreferenceAboutOtherModule(
            proposal_runtime=RelationshipHistoryPreferenceProposalRuntime(session),
            user_input=session.observation_summary,
            turn_index=session.session_index,
            wiring_level=WiringLevel.SHADOW,
            record_store=store,
            action_outcome_evidence=session.to_owner_evidence(),
        )
        snapshot = (await owner.process({})).value
        if not isinstance(snapshot, PreferenceAboutOtherSnapshot):
            raise TypeError("preference owner published an unexpected snapshot type")
        persistence_snapshot = store.export_persistence_snapshot()
        persistence_hashes.append(sha256_json(persistence_snapshot.payload))

    if persistence_snapshot is None:
        raise RuntimeError("P2-development episode contained no evidence sessions")
    probe_store = SocialRecordStore()
    probe_store.hydrate_from_persistence(persistence_snapshot)
    probe_owner = PreferenceAboutOtherModule(
        turn_index=episode.probe_session.session_index,
        wiring_level=WiringLevel.SHADOW,
        record_store=probe_store,
        action_forecast_runtime=(
            forecast_runtime or BoundedRelationshipPreferenceForecastRuntime()
        ),
        action_forecast_request=episode.probe_session.to_request(),
    )
    probe_snapshot = (await probe_owner.process({})).value
    if not isinstance(probe_snapshot, PreferenceAboutOtherSnapshot):
        raise TypeError("preference owner published an unexpected probe snapshot")
    if len(probe_snapshot.action_forecasts) != 1:
        raise RuntimeError("P2-development probe must publish exactly one forecast")
    return P2DevelopmentEpisodeRun(
        episode_id=episode.episode_id,
        forecast=probe_snapshot.action_forecasts[0],
        persistence_payload_sha256=tuple(persistence_hashes),
        persisted_record_count=len(probe_snapshot.records),
        persisted_action_outcome_count=len(probe_snapshot.action_outcome_evidence),
    )


@dataclass(frozen=True)
class P2DevelopmentEvaluatorTruth:
    """Sealed evaluator-only label, physically absent from the public view."""

    episode_id: str
    decision_id: str
    preferred_action_id: str


@dataclass(frozen=True)
class RelationshipP2DevelopmentEvaluatorBundle:
    contract_sha256: str
    truths: tuple[P2DevelopmentEvaluatorTruth, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.contract_sha256, "contract_sha256")
        decision_ids = tuple(item.decision_id for item in self.truths)
        if len(set(decision_ids)) != len(decision_ids):
            raise ValueError("P2-development evaluator decision ids must be unique")


def load_relationship_p2_development_view(
    contract_path: pathlib.Path | None = None,
) -> RelationshipP2DevelopmentView:
    contract = load_relationship_p2_development_contract(contract_path)
    training_view = load_relationship_consumer_training_view()
    _validate_training_lineage(contract, training_view)
    episodes = tuple(
        _public_episode(observation)
        for observation in training_view.training_dataset.observations
    )
    return RelationshipP2DevelopmentView(contract=contract, episodes=episodes)


def load_relationship_p2_development_evaluator_bundle(
    contract_path: pathlib.Path | None = None,
) -> RelationshipP2DevelopmentEvaluatorBundle:
    """Load v3 labels into a separate evaluator-only object."""

    public_view = load_relationship_p2_development_view(contract_path)
    training_view = load_relationship_consumer_training_view()
    _validate_training_lineage(public_view.contract, training_view)
    truths = tuple(
        P2DevelopmentEvaluatorTruth(
            episode_id=episode.episode_id,
            decision_id=episode.probe_session.decision_id,
            preferred_action_id=training_view.training_dataset.dynamic_for_scene(
                observation.scene_id
            ).preferred_action.value,
        )
        for episode, observation in zip(
            public_view.episodes,
            training_view.training_dataset.observations,
            strict=True,
        )
    )
    return RelationshipP2DevelopmentEvaluatorBundle(
        contract_sha256=public_view.contract.contract_sha256,
        truths=truths,
    )


def load_relationship_p2_development_contract(
    contract_path: pathlib.Path | None = None,
) -> RelationshipP2DevelopmentContract:
    path = pathlib.Path(contract_path or relationship_p2_development_contract_path())
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("P2-development contract must contain a JSON object")
    _require_exact_keys(
        raw,
        {
            "schema_version",
            "source",
            "development_split",
            "owner_boundary",
            "firewall",
            "claim_boundary",
        },
        source="P2-development contract",
    )
    source = _require_mapping(raw["source"], "source")
    split = _require_mapping(raw["development_split"], "development_split")
    owner = _require_mapping(raw["owner_boundary"], "owner_boundary")
    firewall = _require_mapping(raw["firewall"], "firewall")
    _require_exact_keys(
        source,
        {
            "consumer_split_contract_id",
            "training_package_name",
            "training_dataset_fingerprint",
            "training_role",
            "p1k_diagnostic_status",
        },
        source="P2-development source",
    )
    _require_exact_keys(
        split,
        {
            "split_id",
            "episode_derivation",
            "expected_episode_count",
            "evidence_sessions_per_episode",
            "probe_sessions_per_episode",
            "process_restart_between_sessions",
            "incremental_context_only",
            "raw_history_replayed_at_probe",
        },
        source="P2-development split",
    )
    _require_exact_keys(
        owner,
        {
            "slot",
            "owner",
            "wiring_level",
            "intent_about_other_enabled",
            "forecast_runtime_role",
            "candidate_actions",
            "typed_outcomes",
        },
        source="P2-development owner boundary",
    )
    required_firewall = {
        "qualification_package_name": "relationship_transfer_v4",
        "qualification_observations_loaded": False,
        "qualification_truth_loaded": False,
        "future_probe_label_visible_to_sut": False,
        "training_label_visible_to_sut": False,
        "evaluation_feedback_to_owner": False,
        "writes_prediction_error": False,
        "writes_credit": False,
        "writes_steering": False,
        "affects_expression": False,
        "formal_hidden_test_opened": False,
    }
    _require_exact_keys(
        firewall,
        set(required_firewall),
        source="P2-development firewall",
    )
    if dict(firewall) != required_firewall:
        raise ValueError("P2-development firewall is not closed")
    required_values = {
        "training_role": "consumer_training_only",
        "p1k_diagnostic_status": "zero_output_not_used_for_owner_selection",
    }
    for field_name, expected in required_values.items():
        if source[field_name] != expected:
            raise ValueError(f"P2-development source {field_name} drifted")
    if split["episode_derivation"] != (
        "four_incremental_history_sessions_then_unseen_surface_probe"
    ):
        raise ValueError("P2-development episode derivation drifted")
    for field_name, expected in (
        ("process_restart_between_sessions", True),
        ("incremental_context_only", True),
        ("raw_history_replayed_at_probe", False),
    ):
        if split[field_name] is not expected:
            raise ValueError(f"P2-development {field_name} drifted")
    required_owner_values = {
        "slot": _PREFERENCE_SLOT,
        "owner": "PreferenceAboutOtherModule",
        "wiring_level": "SHADOW",
        "intent_about_other_enabled": False,
        "forecast_runtime_role": "non_owning_bounded_adapter",
    }
    for field_name, expected in required_owner_values.items():
        if owner[field_name] != expected:
            raise ValueError(f"P2-development owner {field_name} drifted")
    return RelationshipP2DevelopmentContract(
        contract_sha256=sha256_json(raw),
        consumer_split_contract_id=_require_text(
            source["consumer_split_contract_id"],
            "consumer_split_contract_id",
        ),
        training_package_name=_require_text(
            source["training_package_name"],
            "training_package_name",
        ),
        training_dataset_fingerprint=_require_text(
            source["training_dataset_fingerprint"],
            "training_dataset_fingerprint",
        ),
        split_id=_require_text(split["split_id"], "split_id"),
        expected_episode_count=_require_int(
            split["expected_episode_count"],
            "expected_episode_count",
        ),
        evidence_sessions_per_episode=_require_int(
            split["evidence_sessions_per_episode"],
            "evidence_sessions_per_episode",
        ),
        probe_sessions_per_episode=_require_int(
            split["probe_sessions_per_episode"],
            "probe_sessions_per_episode",
        ),
        candidate_action_ids=_require_text_tuple(
            owner["candidate_actions"],
            "candidate_actions",
        ),
        outcome_ids=_require_text_tuple(owner["typed_outcomes"], "typed_outcomes"),
        claim_boundary=_require_text(raw["claim_boundary"], "claim_boundary"),
        schema_version=_require_text(raw["schema_version"], "schema_version"),
    )


def _public_episode(observation: Any) -> P2DevelopmentEpisode:
    episode_id = f"p2-development:{observation.trajectory_sha256}"
    histories = tuple(
        P2DevelopmentHistorySession(
            episode_id=episode_id,
            session_id=f"{episode_id}:session:{index}",
            session_index=index,
            event_id=history.event_id,
            observation_summary=history.user_utterance,
            action_id=history.assistant_action.value,
            observed_outcome_id=history.typed_outcome.value,
            reaction_summary=history.user_reaction,
            observation_ref=f"{episode_id}:history:{index}",
        )
        for index, history in enumerate(observation.histories)
    )
    probe = P2DevelopmentProbeSession(
        episode_id=episode_id,
        session_id=f"{episode_id}:session:{len(histories)}",
        session_index=len(histories),
        decision_id=f"{episode_id}:decision",
        current_observation=observation.current_input,
        observation_ref=f"{episode_id}:probe",
        candidate_action_ids=tuple(
            action.value for action in observation.candidate_action_ids
        ),
        outcome_ids=tuple(outcome.value for outcome in RELATIONSHIP_OUTCOMES),
    )
    return P2DevelopmentEpisode(
        episode_id=episode_id,
        history_sessions=histories,
        probe_session=probe,
    )


def _validate_training_lineage(
    contract: RelationshipP2DevelopmentContract,
    training_view: RelationshipConsumerTrainingView,
) -> None:
    if training_view.contract.contract_sha256 != contract.consumer_split_contract_id:
        raise ValueError("P2-development consumer split contract lineage mismatch")
    if training_view.training_dataset.package_name != contract.training_package_name:
        raise ValueError("P2-development training package lineage mismatch")
    if (
        training_view.training_dataset.dataset_fingerprint
        != contract.training_dataset_fingerprint
    ):
        raise ValueError("P2-development training fingerprint lineage mismatch")


def _assert_no_public_truth_leakage(view: RelationshipP2DevelopmentView) -> None:
    payload = {
        "contract": {
            "contract_sha256": view.contract.contract_sha256,
            "split_id": view.contract.split_id,
        },
        "episodes": [
            {
                "episode_id": episode.episode_id,
                "sessions": episode.to_sut_sequence(),
            }
            for episode in view.episodes
        ],
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    leaked = sorted(key for key in _FORBIDDEN_PUBLIC_KEYS if f'"{key}"' in serialized)
    if leaked:
        raise ValueError(f"P2-development public view leaks evaluator keys: {leaked!r}")


def _require_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} keys must be strings")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    source: str,
) -> None:
    missing = sorted(expected.difference(value))
    extra = sorted(set(value).difference(expected))
    if missing or extra:
        raise ValueError(f"{source} fields drifted; missing={missing}, extra={extra}")


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_text_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty list")
    items = tuple(_require_text(item, field_name) for item in value)
    if len(set(items)) != len(items):
        raise ValueError(f"{field_name} entries must be unique")
    return items


def _require_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _require_sha256(value: object, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase sha256 digest")


__all__ = [
    "P2_DEVELOPMENT_RUNTIME_ID",
    "P2_DEVELOPMENT_SCHEMA_VERSION",
    "P2_DEVELOPMENT_SPLIT_ID",
    "BoundedRelationshipPreferenceForecastRuntime",
    "P2DevelopmentEpisode",
    "P2DevelopmentEpisodeRun",
    "P2DevelopmentEvaluatorTruth",
    "P2DevelopmentHistorySession",
    "P2DevelopmentProbeSession",
    "RelationshipHistoryPreferenceProposalRuntime",
    "RelationshipP2DevelopmentContract",
    "RelationshipP2DevelopmentEvaluatorBundle",
    "RelationshipP2DevelopmentView",
    "load_relationship_p2_development_contract",
    "load_relationship_p2_development_evaluator_bundle",
    "load_relationship_p2_development_view",
    "relationship_p2_development_contract_path",
    "run_p2_development_episode",
]
