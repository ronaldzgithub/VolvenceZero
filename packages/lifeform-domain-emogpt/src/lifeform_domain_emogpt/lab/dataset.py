"""Three-layer dataset loader for versioned relationship-transfer packages.

The rendered observation file is the only data surface exposed to a system
under test.  Generator truth is loaded into separate frozen records and is
reachable only through explicit environment/evaluation methods.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass
from typing import Any

from volvence_zero.dialogue_trace import DialogueExternalOutcomeKind

from lifeform_domain_emogpt.lab.contracts import (
    CandidateOutcomePrediction,
    OutcomeProbability,
    RELATIONSHIP_ACTIONS,
    RELATIONSHIP_OUTCOMES,
    RelationshipAction,
    RelationshipDatasetSplit,
    canonical_json,
)


RELATIONSHIP_TRANSFER_V1_DATASET_SCHEMA_VERSION = "relationship-transfer-dataset.v1"
RELATIONSHIP_TRANSFER_V1_TRUTH_SCHEMA_VERSION = "relationship-transfer-truth.v1"
RELATIONSHIP_TRANSFER_V1_PACKAGE_NAME = "relationship_transfer_v1"
RELATIONSHIP_TRANSFER_V2_DATASET_SCHEMA_VERSION = "relationship-transfer-dataset.v2"
RELATIONSHIP_TRANSFER_V2_TRUTH_SCHEMA_VERSION = "relationship-transfer-truth.v2"
RELATIONSHIP_TRANSFER_V2_PACKAGE_NAME = "relationship_transfer_v2"

# Compatibility aliases keep every frozen v1 consumer on the same default package.
RELATIONSHIP_TRANSFER_DATASET_SCHEMA_VERSION = (
    RELATIONSHIP_TRANSFER_V1_DATASET_SCHEMA_VERSION
)
RELATIONSHIP_TRANSFER_TRUTH_SCHEMA_VERSION = RELATIONSHIP_TRANSFER_V1_TRUTH_SCHEMA_VERSION
RELATIONSHIP_TRANSFER_PACKAGE_NAME = RELATIONSHIP_TRANSFER_V1_PACKAGE_NAME

_PACKAGE_SCHEMAS = {
    RELATIONSHIP_TRANSFER_V1_PACKAGE_NAME: (
        RELATIONSHIP_TRANSFER_V1_DATASET_SCHEMA_VERSION,
        RELATIONSHIP_TRANSFER_V1_TRUTH_SCHEMA_VERSION,
    ),
    RELATIONSHIP_TRANSFER_V2_PACKAGE_NAME: (
        RELATIONSHIP_TRANSFER_V2_DATASET_SCHEMA_VERSION,
        RELATIONSHIP_TRANSFER_V2_TRUTH_SCHEMA_VERSION,
    ),
}

_FORBIDDEN_SUT_KEYS = frozenset(
    {
        "sealed_latent_dynamic_id",
        "latent_dynamic_id",
        "preferred_action",
        "outcome_profile_id",
        "hidden_summary",
        "mirror_pair_id",
        "abstract_condition_id",
        "condition_id",
        "history_condition_bindings",
        "policy_id",
        "policy_profiles",
        "probe_condition_id",
        "future_outcome",
        "generator_truth",
    }
)


def relationship_transfer_package_dir(
    package_name: str = RELATIONSHIP_TRANSFER_PACKAGE_NAME,
) -> pathlib.Path:
    if package_name not in _PACKAGE_SCHEMAS:
        raise ValueError(f"unsupported relationship-transfer package {package_name!r}")
    return (
        pathlib.Path(__file__).resolve().parents[1]
        / "scenario_packages"
        / package_name
    )


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return raw


def _require_exact_keys(
    payload: dict[str, Any],
    expected: set[str],
    *,
    source: str,
) -> None:
    missing = sorted(expected - set(payload))
    extra = sorted(set(payload) - expected)
    if missing or extra:
        raise ValueError(
            f"{source} fields do not match schema; missing={missing}, extra={extra}"
        )


def _require_text(value: object, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source} must be a non-empty string")
    return value


@dataclass(frozen=True)
class RelationshipHistoryEvent:
    event_id: str
    surface_family: str
    user_utterance: str
    assistant_action: RelationshipAction
    typed_outcome: DialogueExternalOutcomeKind
    user_reaction: str

    def __post_init__(self) -> None:
        for name, value in (
            ("event_id", self.event_id),
            ("surface_family", self.surface_family),
            ("user_utterance", self.user_utterance),
            ("user_reaction", self.user_reaction),
        ):
            _require_text(value, name)
        if self.typed_outcome not in RELATIONSHIP_OUTCOMES:
            raise ValueError("history typed_outcome is outside the lab vocabulary")

    def to_payload(self) -> dict[str, str]:
        return {
            "event_id": self.event_id,
            "surface_family": self.surface_family,
            "user_utterance": self.user_utterance,
            "assistant_action": self.assistant_action.value,
            "typed_outcome": self.typed_outcome.value,
            "user_reaction": self.user_reaction,
        }


@dataclass(frozen=True)
class RelationshipObservation:
    """The complete and only system-under-test input for one probe."""

    scene_id: str
    user_scope_hash: str
    probe_surface_family: str
    histories: tuple[RelationshipHistoryEvent, ...]
    current_input: str
    dataset_schema_version: str = RELATIONSHIP_TRANSFER_DATASET_SCHEMA_VERSION
    candidate_action_ids: tuple[RelationshipAction, ...] = RELATIONSHIP_ACTIONS

    def __post_init__(self) -> None:
        for name, value in (
            ("scene_id", self.scene_id),
            ("user_scope_hash", self.user_scope_hash),
            ("probe_surface_family", self.probe_surface_family),
            ("current_input", self.current_input),
        ):
            _require_text(value, name)
        if len(self.user_scope_hash) != 64 or any(
            char not in "0123456789abcdef" for char in self.user_scope_hash
        ):
            raise ValueError("user_scope_hash must be a lowercase sha256 digest")
        if not self.histories:
            raise ValueError("relationship observation requires at least one history event")
        if self.dataset_schema_version not in {
            schema_pair[0] for schema_pair in _PACKAGE_SCHEMAS.values()
        }:
            raise ValueError("relationship observation schema_version is unsupported")
        if self.candidate_action_ids != RELATIONSHIP_ACTIONS:
            raise ValueError("observation must expose the canonical closed action surface")

    def to_sut_payload(self) -> dict[str, object]:
        """Serialize without split, pair identity, hidden truth, or future outcome."""

        return {
            "schema_version": self.dataset_schema_version,
            "user_scope_hash": self.user_scope_hash,
            "probe_surface_family": self.probe_surface_family,
            "histories": [history.to_payload() for history in self.histories],
            "current_input": self.current_input,
            "candidate_action_ids": [action.value for action in self.candidate_action_ids],
        }

    @property
    def trajectory_sha256(self) -> str:
        return hashlib.sha256(
            canonical_json(self.to_sut_payload()).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class LatentRelationshipDynamic:
    """Sealed generator truth; never serialized by ``to_sut_payload``."""

    dynamic_id: str
    mirror_pair_id: str
    split: RelationshipDatasetSplit
    preferred_action: RelationshipAction
    outcome_profile_id: str
    hidden_summary: str
    policy_id: str | None = None
    probe_condition_id: str | None = None


@dataclass(frozen=True)
class AbstractRelationshipCondition:
    """Sealed condition identity used only by environment/evaluation."""

    condition_id: str
    hidden_summary: str

    def __post_init__(self) -> None:
        _require_text(self.condition_id, "condition_id")
        _require_text(self.hidden_summary, "condition hidden_summary")


@dataclass(frozen=True)
class RelationshipPolicyProfile:
    """Sealed user-specific mapping from abstract condition to relationship action."""

    policy_id: str
    condition_actions: tuple[tuple[str, RelationshipAction], ...]

    def __post_init__(self) -> None:
        _require_text(self.policy_id, "policy_id")
        condition_ids = tuple(condition_id for condition_id, _ in self.condition_actions)
        if not condition_ids or len(set(condition_ids)) != len(condition_ids):
            raise ValueError("policy condition ids must be non-empty and unique")
        if self.condition_actions != tuple(
            sorted(self.condition_actions, key=lambda item: item[0])
        ):
            raise ValueError("policy condition actions must use canonical condition order")
        if any(action is RelationshipAction.NEUTRAL_NOOP for _, action in self.condition_actions):
            raise ValueError("neutral_noop cannot be a learned relationship policy action")

    def action_for(self, condition_id: str) -> RelationshipAction:
        for candidate_id, action in self.condition_actions:
            if candidate_id == condition_id:
                return action
        raise KeyError(condition_id)


@dataclass(frozen=True)
class RelationshipTransferDataset:
    observations: tuple[RelationshipObservation, ...]
    dynamics: tuple[LatentRelationshipDynamic, ...]
    outcome_profiles: tuple[tuple[str, tuple[CandidateOutcomePrediction, ...]], ...]
    scene_bindings: tuple[tuple[str, str], ...]
    positive_outcomes: tuple[DialogueExternalOutcomeKind, ...]
    dataset_fingerprint: str
    package_name: str = RELATIONSHIP_TRANSFER_PACKAGE_NAME
    dataset_schema_version: str = RELATIONSHIP_TRANSFER_DATASET_SCHEMA_VERSION
    truth_schema_version: str = RELATIONSHIP_TRANSFER_TRUTH_SCHEMA_VERSION
    abstract_conditions: tuple[AbstractRelationshipCondition, ...] = ()
    policy_profiles: tuple[RelationshipPolicyProfile, ...] = ()
    history_condition_bindings: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        expected_schemas = _PACKAGE_SCHEMAS.get(self.package_name)
        if expected_schemas is None:
            raise ValueError(f"unsupported relationship-transfer package {self.package_name!r}")
        if expected_schemas != (
            self.dataset_schema_version,
            self.truth_schema_version,
        ):
            raise ValueError("relationship-transfer package/schema pairing is invalid")
        observation_ids = tuple(item.scene_id for item in self.observations)
        dynamic_ids = tuple(item.dynamic_id for item in self.dynamics)
        if len(set(observation_ids)) != len(observation_ids):
            raise ValueError("rendered observation scene_ids must be unique")
        if len(set(dynamic_ids)) != len(dynamic_ids):
            raise ValueError("latent dynamic_ids must be unique")
        bindings = dict(self.scene_bindings)
        if set(bindings) != set(observation_ids):
            raise ValueError("scene bindings must cover every rendered observation exactly")
        if not set(bindings.values()) <= set(dynamic_ids):
            raise ValueError("scene binding references an unknown latent dynamic")
        profile_ids = {profile_id for profile_id, _ in self.outcome_profiles}
        if any(item.outcome_profile_id not in profile_ids for item in self.dynamics):
            raise ValueError("latent dynamic references an unknown outcome profile")
        if not self.positive_outcomes or not set(self.positive_outcomes) <= set(
            RELATIONSHIP_OUTCOMES
        ):
            raise ValueError("positive_outcomes must be a non-empty relationship subset")
        self._validate_mirrored_pairs()
        if self.package_name == RELATIONSHIP_TRANSFER_V2_PACKAGE_NAME:
            self._validate_compositional_transfer_v2()
        elif any(
            (
                self.abstract_conditions,
                self.policy_profiles,
                self.history_condition_bindings,
            )
        ):
            raise ValueError("relationship_transfer_v1 cannot carry v2 sealed policy metadata")
        self.assert_no_sut_truth_leakage()

    def _validate_mirrored_pairs(self) -> None:
        dynamics_by_pair: dict[str, list[LatentRelationshipDynamic]] = {}
        observations_by_pair: dict[str, list[RelationshipObservation]] = {}
        dynamic_by_id = {item.dynamic_id: item for item in self.dynamics}
        observation_by_id = {item.scene_id: item for item in self.observations}
        for dynamic in self.dynamics:
            dynamics_by_pair.setdefault(dynamic.mirror_pair_id, []).append(dynamic)
        for scene_id, dynamic_id in self.scene_bindings:
            pair_id = dynamic_by_id[dynamic_id].mirror_pair_id
            observations_by_pair.setdefault(pair_id, []).append(observation_by_id[scene_id])
        if len(dynamics_by_pair) < 6:
            raise ValueError(f"{self.package_name} requires at least six mirrored pairs")
        for pair_id, pair_dynamics in dynamics_by_pair.items():
            if len(pair_dynamics) != 2:
                raise ValueError(f"{pair_id} must contain exactly two latent siblings")
            preferred = {item.preferred_action for item in pair_dynamics}
            if preferred != {
                RelationshipAction.STAY_PRESENT_WITHOUT_PROBE,
                RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION,
            }:
                raise ValueError(f"{pair_id} must require opposite non-noop actions")
            if len({item.split for item in pair_dynamics}) != 1:
                raise ValueError(f"{pair_id} latent siblings must stay in one split")
            if self.package_name == RELATIONSHIP_TRANSFER_V2_PACKAGE_NAME:
                if len({item.probe_condition_id for item in pair_dynamics}) != 1:
                    raise ValueError(
                        f"{pair_id} v2 siblings must share one probe condition"
                    )
                if len({item.policy_id for item in pair_dynamics}) != 2:
                    raise ValueError(
                        f"{pair_id} v2 siblings must use opposite policy profiles"
                    )
            pair_observations = observations_by_pair.get(pair_id, [])
            if len(pair_observations) != 2:
                raise ValueError(f"{pair_id} must bind exactly two rendered observations")
            current_bytes = {
                item.current_input.encode("utf-8") for item in pair_observations
            }
            if len(current_bytes) != 1:
                raise ValueError(f"{pair_id} mirrored current inputs must be byte-identical")
        families = {item.probe_surface_family for item in self.observations}
        minimum_families = (
            6 if self.package_name == RELATIONSHIP_TRANSFER_V2_PACKAGE_NAME else 4
        )
        if len(families) < minimum_families:
            raise ValueError(
                f"{self.package_name} must cover at least {minimum_families} surface families"
            )

    def _validate_compositional_transfer_v2(self) -> None:
        """Reject v2 data that can be solved by global action/outcome tallying."""

        conditions = {item.condition_id: item for item in self.abstract_conditions}
        policies = {item.policy_id: item for item in self.policy_profiles}
        if len(conditions) != 2:
            raise ValueError("relationship_transfer_v2 requires exactly two abstract conditions")
        if len(policies) != 2:
            raise ValueError("relationship_transfer_v2 requires exactly two policy profiles")
        expected_condition_ids = set(conditions)
        for policy in policies.values():
            if {condition_id for condition_id, _ in policy.condition_actions} != expected_condition_ids:
                raise ValueError("every v2 policy must cover every abstract condition")
        for condition_id in expected_condition_ids:
            actions = {policy.action_for(condition_id) for policy in policies.values()}
            if actions != {
                RelationshipAction.STAY_PRESENT_WITHOUT_PROBE,
                RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION,
            }:
                raise ValueError("v2 policy profiles must be complementary per condition")

        all_history_ids = {
            history.event_id
            for observation in self.observations
            for history in observation.histories
        }
        if len(all_history_ids) != sum(
            len(observation.histories) for observation in self.observations
        ):
            raise ValueError("v2 history event ids must be globally unique")
        bindings = dict(self.history_condition_bindings)
        if len(bindings) != len(self.history_condition_bindings):
            raise ValueError("v2 history condition bindings must be unique")
        if set(bindings) != all_history_ids:
            raise ValueError("v2 history condition bindings must cover every history exactly")
        if not set(bindings.values()) <= expected_condition_ids:
            raise ValueError("v2 history binding references an unknown abstract condition")

        positive_outcomes = set(self.positive_outcomes)
        for observation in self.observations:
            if len(observation.histories) != 4:
                raise ValueError("relationship_transfer_v2 requires four histories per user")
            history_families = {item.surface_family for item in observation.histories}
            if len(history_families) != 4:
                raise ValueError("v2 histories must span four distinct surface families")
            if observation.probe_surface_family in history_families:
                raise ValueError("v2 probe family must be unseen in that user's histories")

            dynamic = self.dynamic_for_scene(observation.scene_id)
            if dynamic.policy_id is None or dynamic.probe_condition_id is None:
                raise ValueError("v2 dynamic requires policy_id and probe_condition_id")
            try:
                policy = policies[dynamic.policy_id]
            except KeyError as exc:
                raise ValueError("v2 dynamic references an unknown policy") from exc
            if dynamic.probe_condition_id not in expected_condition_ids:
                raise ValueError("v2 dynamic references an unknown probe condition")
            if policy.action_for(dynamic.probe_condition_id) is not dynamic.preferred_action:
                raise ValueError("v2 preferred action must follow the sealed policy mapping")

            histories_by_condition: dict[str, list[RelationshipHistoryEvent]] = {}
            action_outcome_polarities: dict[RelationshipAction, set[bool]] = {}
            action_counts: dict[RelationshipAction, int] = {}
            for history in observation.histories:
                if history.assistant_action is RelationshipAction.NEUTRAL_NOOP:
                    raise ValueError("v2 histories cannot use neutral_noop as policy evidence")
                condition_id = bindings[history.event_id]
                histories_by_condition.setdefault(condition_id, []).append(history)
                correct_action = policy.action_for(condition_id)
                positive = history.typed_outcome in positive_outcomes
                if (history.assistant_action is correct_action) != positive:
                    raise ValueError(
                        "v2 history outcome polarity must agree with its sealed policy"
                    )
                action_counts[history.assistant_action] = (
                    action_counts.get(history.assistant_action, 0) + 1
                )
                action_outcome_polarities.setdefault(
                    history.assistant_action,
                    set(),
                ).add(positive)

            if set(histories_by_condition) != expected_condition_ids or any(
                len(items) != 2 for items in histories_by_condition.values()
            ):
                raise ValueError("v2 histories must provide two examples per condition")
            for condition_id, histories in histories_by_condition.items():
                if {item.assistant_action for item in histories} != {
                    RelationshipAction.STAY_PRESENT_WITHOUT_PROBE,
                    RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION,
                }:
                    raise ValueError(
                        f"v2 condition {condition_id} must contrast both actions"
                    )
            if action_counts != {
                RelationshipAction.STAY_PRESENT_WITHOUT_PROBE: 2,
                RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION: 2,
            }:
                raise ValueError("v2 histories must balance both non-noop actions")
            if any(polarities != {False, True} for polarities in action_outcome_polarities.values()):
                raise ValueError(
                    "v2 each action must have one positive and one negative outcome"
                )

    def observation(self, scene_id: str) -> RelationshipObservation:
        for observation in self.observations:
            if observation.scene_id == scene_id:
                return observation
        raise KeyError(scene_id)

    def dynamic_for_scene(self, scene_id: str) -> LatentRelationshipDynamic:
        binding = dict(self.scene_bindings)
        try:
            dynamic_id = binding[scene_id]
        except KeyError as exc:
            raise KeyError(scene_id) from exc
        for dynamic in self.dynamics:
            if dynamic.dynamic_id == dynamic_id:
                return dynamic
        raise RuntimeError(f"validated binding lost dynamic {dynamic_id!r}")

    def profile_for_dynamic(
        self,
        dynamic: LatentRelationshipDynamic,
    ) -> tuple[CandidateOutcomePrediction, ...]:
        for profile_id, predictions in self.outcome_profiles:
            if profile_id == dynamic.outcome_profile_id:
                return predictions
        raise RuntimeError(f"validated dataset lost profile {dynamic.outcome_profile_id!r}")

    def distribution(
        self,
        scene_id: str,
        action: RelationshipAction,
    ) -> CandidateOutcomePrediction:
        dynamic = self.dynamic_for_scene(scene_id)
        for prediction in self.profile_for_dynamic(dynamic):
            if prediction.action_id is action:
                return prediction
        raise RuntimeError(f"validated profile lost action {action.value!r}")

    def mirrored_pairs(
        self,
    ) -> tuple[
        tuple[
            str,
            tuple[tuple[RelationshipObservation, LatentRelationshipDynamic], ...],
        ],
        ...,
    ]:
        grouped: dict[
            str,
            list[tuple[RelationshipObservation, LatentRelationshipDynamic]],
        ] = {}
        for observation in self.observations:
            dynamic = self.dynamic_for_scene(observation.scene_id)
            grouped.setdefault(dynamic.mirror_pair_id, []).append((observation, dynamic))
        return tuple(
            (pair_id, tuple(sorted(items, key=lambda item: item[0].scene_id)))
            for pair_id, items in sorted(grouped.items())
        )

    def assert_no_sut_truth_leakage(self) -> None:
        """Fail if sealed identifiers/keys enter any SUT payload."""

        sealed_tokens = {
            item.dynamic_id for item in self.dynamics
        } | {
            item.outcome_profile_id for item in self.dynamics
        } | {
            item.mirror_pair_id for item in self.dynamics
        } | {
            item.condition_id for item in self.abstract_conditions
        } | {
            item.policy_id for item in self.policy_profiles
        }
        for observation in self.observations:
            payload = observation.to_sut_payload()
            keys = _nested_keys(payload)
            leaked_keys = sorted(keys & _FORBIDDEN_SUT_KEYS)
            if leaked_keys:
                raise ValueError(
                    f"SUT payload {observation.scene_id} leaks truth keys {leaked_keys}"
                )
            encoded = canonical_json(payload)
            leaked_tokens = sorted(token for token in sealed_tokens if token in encoded)
            if leaked_tokens:
                raise ValueError(
                    f"SUT payload {observation.scene_id} leaks sealed ids {leaked_tokens}"
                )


def _nested_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        keys = {str(key) for key in value}
        for child in value.values():
            keys.update(_nested_keys(child))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for child in value:
            keys.update(_nested_keys(child))
        return keys
    return set()


def _parse_history(raw: object, source: str) -> RelationshipHistoryEvent:
    if not isinstance(raw, dict):
        raise ValueError(f"{source} must be an object")
    expected = {
        "event_id",
        "surface_family",
        "user_utterance",
        "assistant_action",
        "typed_outcome",
        "user_reaction",
    }
    _require_exact_keys(raw, expected, source=source)
    return RelationshipHistoryEvent(
        event_id=_require_text(raw["event_id"], f"{source}.event_id"),
        surface_family=_require_text(
            raw["surface_family"], f"{source}.surface_family"
        ),
        user_utterance=_require_text(
            raw["user_utterance"], f"{source}.user_utterance"
        ),
        assistant_action=RelationshipAction(
            _require_text(raw["assistant_action"], f"{source}.assistant_action")
        ),
        typed_outcome=DialogueExternalOutcomeKind(
            _require_text(raw["typed_outcome"], f"{source}.typed_outcome")
        ),
        user_reaction=_require_text(
            raw["user_reaction"], f"{source}.user_reaction"
        ),
    )


def _parse_observations(
    raw: dict[str, Any],
    *,
    package_name: str,
    dataset_schema_version: str,
) -> tuple[RelationshipObservation, ...]:
    _require_exact_keys(raw, {"schema_version", "scenes"}, source="rendered")
    if raw["schema_version"] != dataset_schema_version:
        raise ValueError("rendered observations schema_version mismatch")
    raw_scenes = raw["scenes"]
    if not isinstance(raw_scenes, list):
        raise ValueError("rendered.scenes must be an array")
    observations: list[RelationshipObservation] = []
    for index, scene in enumerate(raw_scenes):
        source = f"rendered.scenes[{index}]"
        if not isinstance(scene, dict):
            raise ValueError(f"{source} must be an object")
        expected = {
            "scene_id",
            "probe_surface_family",
            "histories",
            "current_input",
        }
        _require_exact_keys(scene, expected, source=source)
        scene_id = _require_text(scene["scene_id"], f"{source}.scene_id")
        raw_histories = scene["histories"]
        if not isinstance(raw_histories, list):
            raise ValueError(f"{source}.histories must be an array")
        scope_namespace = (
            "relationship-transfer-v1"
            if package_name == RELATIONSHIP_TRANSFER_V1_PACKAGE_NAME
            else package_name
        )
        user_scope_hash = hashlib.sha256(
            f"{scope_namespace}:{scene_id}".encode("utf-8")
        ).hexdigest()
        observations.append(
            RelationshipObservation(
                scene_id=scene_id,
                user_scope_hash=user_scope_hash,
                probe_surface_family=_require_text(
                    scene["probe_surface_family"],
                    f"{source}.probe_surface_family",
                ),
                histories=tuple(
                    _parse_history(item, f"{source}.histories[{history_index}]")
                    for history_index, item in enumerate(raw_histories)
                ),
                current_input=_require_text(
                    scene["current_input"], f"{source}.current_input"
                ),
                dataset_schema_version=dataset_schema_version,
            )
        )
    return tuple(observations)


def _parse_prediction(
    action: RelationshipAction,
    raw_distribution: object,
    source: str,
) -> CandidateOutcomePrediction:
    if not isinstance(raw_distribution, dict):
        raise ValueError(f"{source} must be an object")
    expected = {kind.value for kind in RELATIONSHIP_OUTCOMES}
    _require_exact_keys(raw_distribution, expected, source=source)
    return CandidateOutcomePrediction(
        action_id=action,
        outcomes=tuple(
            OutcomeProbability(kind, float(raw_distribution[kind.value]))
            for kind in RELATIONSHIP_OUTCOMES
        ),
    )


def _parse_truth(
    raw: dict[str, Any],
    *,
    truth_schema_version: str,
) -> tuple[
    tuple[LatentRelationshipDynamic, ...],
    tuple[tuple[str, tuple[CandidateOutcomePrediction, ...]], ...],
    tuple[tuple[str, str], ...],
    tuple[DialogueExternalOutcomeKind, ...],
    tuple[AbstractRelationshipCondition, ...],
    tuple[RelationshipPolicyProfile, ...],
    tuple[tuple[str, str], ...],
]:
    expected = {
        "schema_version",
        "positive_outcomes",
        "outcome_profiles",
        "dynamics",
        "scene_bindings",
    }
    if truth_schema_version == RELATIONSHIP_TRANSFER_V2_TRUTH_SCHEMA_VERSION:
        expected.update(
            {
                "abstract_conditions",
                "policy_profiles",
                "history_condition_bindings",
            }
        )
    _require_exact_keys(raw, expected, source="truth")
    if raw["schema_version"] != truth_schema_version:
        raise ValueError("generator truth schema_version mismatch")

    conditions: list[AbstractRelationshipCondition] = []
    policies: list[RelationshipPolicyProfile] = []
    history_condition_bindings: list[tuple[str, str]] = []
    if truth_schema_version == RELATIONSHIP_TRANSFER_V2_TRUTH_SCHEMA_VERSION:
        raw_conditions = raw["abstract_conditions"]
        if not isinstance(raw_conditions, list):
            raise ValueError("truth.abstract_conditions must be an array")
        for index, item in enumerate(raw_conditions):
            source = f"truth.abstract_conditions[{index}]"
            if not isinstance(item, dict):
                raise ValueError(f"{source} must be an object")
            _require_exact_keys(
                item,
                {"condition_id", "hidden_summary"},
                source=source,
            )
            conditions.append(
                AbstractRelationshipCondition(
                    condition_id=_require_text(
                        item["condition_id"],
                        f"{source}.condition_id",
                    ),
                    hidden_summary=_require_text(
                        item["hidden_summary"],
                        f"{source}.hidden_summary",
                    ),
                )
            )

        raw_policies = raw["policy_profiles"]
        if not isinstance(raw_policies, dict):
            raise ValueError("truth.policy_profiles must be an object")
        for policy_id, raw_mapping in sorted(raw_policies.items()):
            source = f"truth.policy_profiles.{policy_id}"
            if not isinstance(raw_mapping, dict) or not raw_mapping:
                raise ValueError(f"{source} must be a non-empty object")
            policies.append(
                RelationshipPolicyProfile(
                    policy_id=_require_text(policy_id, f"{source}.policy_id"),
                    condition_actions=tuple(
                        sorted(
                            (
                                _require_text(
                                    condition_id,
                                    f"{source}.condition_id",
                                ),
                                RelationshipAction(
                                    _require_text(action, f"{source}.{condition_id}")
                                ),
                            )
                            for condition_id, action in raw_mapping.items()
                        )
                    ),
                )
            )

        raw_history_bindings = raw["history_condition_bindings"]
        if not isinstance(raw_history_bindings, list):
            raise ValueError("truth.history_condition_bindings must be an array")
        for index, item in enumerate(raw_history_bindings):
            source = f"truth.history_condition_bindings[{index}]"
            if not isinstance(item, dict):
                raise ValueError(f"{source} must be an object")
            _require_exact_keys(
                item,
                {"event_id", "condition_id"},
                source=source,
            )
            history_condition_bindings.append(
                (
                    _require_text(item["event_id"], f"{source}.event_id"),
                    _require_text(
                        item["condition_id"],
                        f"{source}.condition_id",
                    ),
                )
            )

    raw_profiles = raw["outcome_profiles"]
    if not isinstance(raw_profiles, dict):
        raise ValueError("truth.outcome_profiles must be an object")
    profiles: list[tuple[str, tuple[CandidateOutcomePrediction, ...]]] = []
    for profile_id, raw_profile in sorted(raw_profiles.items()):
        if not isinstance(raw_profile, dict):
            raise ValueError(f"truth.outcome_profiles.{profile_id} must be an object")
        _require_exact_keys(
            raw_profile,
            {action.value for action in RELATIONSHIP_ACTIONS},
            source=f"truth.outcome_profiles.{profile_id}",
        )
        predictions = tuple(
            _parse_prediction(
                action,
                raw_profile[action.value],
                f"truth.outcome_profiles.{profile_id}.{action.value}",
            )
            for action in RELATIONSHIP_ACTIONS
        )
        profiles.append((str(profile_id), predictions))

    raw_dynamics = raw["dynamics"]
    if not isinstance(raw_dynamics, list):
        raise ValueError("truth.dynamics must be an array")
    dynamics: list[LatentRelationshipDynamic] = []
    dynamic_fields = {
        "dynamic_id",
        "mirror_pair_id",
        "split",
        "preferred_action",
        "outcome_profile_id",
        "hidden_summary",
    }
    if truth_schema_version == RELATIONSHIP_TRANSFER_V2_TRUTH_SCHEMA_VERSION:
        dynamic_fields.update({"policy_id", "probe_condition_id"})
    for index, item in enumerate(raw_dynamics):
        source = f"truth.dynamics[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{source} must be an object")
        _require_exact_keys(item, dynamic_fields, source=source)
        preferred = RelationshipAction(
            _require_text(item["preferred_action"], f"{source}.preferred_action")
        )
        if preferred is RelationshipAction.NEUTRAL_NOOP:
            raise ValueError("neutral_noop cannot be a latent preferred action")
        dynamics.append(
            LatentRelationshipDynamic(
                dynamic_id=_require_text(item["dynamic_id"], f"{source}.dynamic_id"),
                mirror_pair_id=_require_text(
                    item["mirror_pair_id"], f"{source}.mirror_pair_id"
                ),
                split=RelationshipDatasetSplit(
                    _require_text(item["split"], f"{source}.split")
                ),
                preferred_action=preferred,
                outcome_profile_id=_require_text(
                    item["outcome_profile_id"], f"{source}.outcome_profile_id"
                ),
                hidden_summary=_require_text(
                    item["hidden_summary"], f"{source}.hidden_summary"
                ),
                policy_id=(
                    _require_text(item["policy_id"], f"{source}.policy_id")
                    if truth_schema_version
                    == RELATIONSHIP_TRANSFER_V2_TRUTH_SCHEMA_VERSION
                    else None
                ),
                probe_condition_id=(
                    _require_text(
                        item["probe_condition_id"],
                        f"{source}.probe_condition_id",
                    )
                    if truth_schema_version
                    == RELATIONSHIP_TRANSFER_V2_TRUTH_SCHEMA_VERSION
                    else None
                ),
            )
        )

    raw_bindings = raw["scene_bindings"]
    if not isinstance(raw_bindings, list):
        raise ValueError("truth.scene_bindings must be an array")
    bindings: list[tuple[str, str]] = []
    for index, item in enumerate(raw_bindings):
        source = f"truth.scene_bindings[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{source} must be an object")
        _require_exact_keys(
            item,
            {"scene_id", "latent_dynamic_id"},
            source=source,
        )
        bindings.append(
            (
                _require_text(item["scene_id"], f"{source}.scene_id"),
                _require_text(
                    item["latent_dynamic_id"], f"{source}.latent_dynamic_id"
                ),
            )
        )

    raw_positive = raw["positive_outcomes"]
    if not isinstance(raw_positive, list):
        raise ValueError("truth.positive_outcomes must be an array")
    positive = tuple(DialogueExternalOutcomeKind(str(item)) for item in raw_positive)
    return (
        tuple(dynamics),
        tuple(profiles),
        tuple(bindings),
        positive,
        tuple(conditions),
        tuple(policies),
        tuple(history_condition_bindings),
    )


def load_relationship_transfer_dataset(
    package_root: pathlib.Path | None = None,
    *,
    package_name: str | None = None,
) -> RelationshipTransferDataset:
    if package_root is None:
        requested_package = package_name or RELATIONSHIP_TRANSFER_PACKAGE_NAME
        root = relationship_transfer_package_dir(requested_package)
    else:
        requested_package = package_name
        root = pathlib.Path(package_root)
    public_path = root / "rendered_observations.json"
    truth_path = root / "generator_truth.json"
    if not public_path.is_file() or not truth_path.is_file():
        raise FileNotFoundError(
            "relationship-transfer package requires rendered_observations.json and "
            f"generator_truth.json under {root}"
        )
    public_raw = _load_json(public_path)
    truth_raw = _load_json(truth_path)
    if "schema_version" not in public_raw or "schema_version" not in truth_raw:
        raise ValueError("relationship-transfer files must declare schema_version")
    schema_pair = (public_raw["schema_version"], truth_raw["schema_version"])
    matching_packages = tuple(
        candidate
        for candidate, expected_pair in _PACKAGE_SCHEMAS.items()
        if schema_pair == expected_pair
    )
    if len(matching_packages) != 1:
        raise ValueError("unsupported relationship-transfer schema pairing")
    inferred_package = matching_packages[0]
    if requested_package is not None and requested_package != inferred_package:
        raise ValueError("requested package does not match file schema versions")
    observations = _parse_observations(
        public_raw,
        package_name=inferred_package,
        dataset_schema_version=str(schema_pair[0]),
    )
    (
        dynamics,
        profiles,
        bindings,
        positive,
        conditions,
        policies,
        history_condition_bindings,
    ) = _parse_truth(
        truth_raw,
        truth_schema_version=str(schema_pair[1]),
    )
    fingerprint = hashlib.sha256(
        (
            canonical_json(public_raw)
            + "\n"
            + canonical_json(truth_raw)
        ).encode("utf-8")
    ).hexdigest()
    return RelationshipTransferDataset(
        observations=observations,
        dynamics=dynamics,
        outcome_profiles=profiles,
        scene_bindings=bindings,
        positive_outcomes=positive,
        dataset_fingerprint=fingerprint,
        package_name=inferred_package,
        dataset_schema_version=str(schema_pair[0]),
        truth_schema_version=str(schema_pair[1]),
        abstract_conditions=conditions,
        policy_profiles=policies,
        history_condition_bindings=history_condition_bindings,
    )


__all__ = [
    "AbstractRelationshipCondition",
    "LatentRelationshipDynamic",
    "RELATIONSHIP_TRANSFER_DATASET_SCHEMA_VERSION",
    "RELATIONSHIP_TRANSFER_PACKAGE_NAME",
    "RELATIONSHIP_TRANSFER_TRUTH_SCHEMA_VERSION",
    "RELATIONSHIP_TRANSFER_V1_DATASET_SCHEMA_VERSION",
    "RELATIONSHIP_TRANSFER_V1_PACKAGE_NAME",
    "RELATIONSHIP_TRANSFER_V1_TRUTH_SCHEMA_VERSION",
    "RELATIONSHIP_TRANSFER_V2_DATASET_SCHEMA_VERSION",
    "RELATIONSHIP_TRANSFER_V2_PACKAGE_NAME",
    "RELATIONSHIP_TRANSFER_V2_TRUTH_SCHEMA_VERSION",
    "RelationshipHistoryEvent",
    "RelationshipObservation",
    "RelationshipPolicyProfile",
    "RelationshipTransferDataset",
    "load_relationship_transfer_dataset",
    "relationship_transfer_package_dir",
]
