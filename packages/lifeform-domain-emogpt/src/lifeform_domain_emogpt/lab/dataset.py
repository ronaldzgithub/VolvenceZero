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
RELATIONSHIP_TRANSFER_V3_DATASET_SCHEMA_VERSION = "relationship-transfer-dataset.v3"
RELATIONSHIP_TRANSFER_V3_TRUTH_SCHEMA_VERSION = "relationship-transfer-truth.v3"
RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME = "relationship_transfer_v3"
RELATIONSHIP_PUBLIC_EVIDENCE_CONTRACT_SCHEMA_VERSION = (
    "relationship-public-evidence-contract.v1"
)

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
    RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME: (
        RELATIONSHIP_TRANSFER_V3_DATASET_SCHEMA_VERSION,
        RELATIONSHIP_TRANSFER_V3_TRUTH_SCHEMA_VERSION,
    ),
}
_COMPOSITIONAL_PACKAGE_NAMES = frozenset(
    {
        RELATIONSHIP_TRANSFER_V2_PACKAGE_NAME,
        RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME,
    }
)
_COMPOSITIONAL_TRUTH_SCHEMA_VERSIONS = frozenset(
    {
        RELATIONSHIP_TRANSFER_V2_TRUTH_SCHEMA_VERSION,
        RELATIONSHIP_TRANSFER_V3_TRUTH_SCHEMA_VERSION,
    }
)
_PUBLIC_EVIDENCE_CLAIM_BOUNDARY = (
    "This development contract only establishes that every public history and "
    "probe has a frozen semantically separable relation-loss witness under one "
    "audited embedding model. It does not prove human readability, Qwen transfer, "
    "Volvence advantage, any of the four capability axes, formal held-out "
    "superiority, or product value."
)

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


def _require_sha256(value: object, source: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{source} must be a lowercase sha256 digest")
    return value


@dataclass(frozen=True)
class RelationshipPublicEvidenceContract:
    """Frozen development-only contract for public semantic legibility."""

    package_name: str
    source_p1e_report_artifact_id: str
    source_required_verdict: str
    history_text_fields: tuple[str, ...]
    history_text_joiner: str
    probe_text_fields: tuple[str, ...]
    semantic_auditor_version: str
    semantic_audit_method: str
    semantic_similarity: str
    semantic_audit_embedder: str
    semantic_audit_model_source: str
    semantic_audit_weights_sha256: str
    condition_anchor_source: str
    score_precision_decimals: int
    top1_tie_policy: str
    required_evidence_units: int
    required_top1_accuracy: float
    minimum_correct_anchor_margin: float
    minimum_mean_correct_anchor_margin: float
    human_anchor_status: str
    claim_boundary: str
    contract_sha256: str
    schema_version: str = RELATIONSHIP_PUBLIC_EVIDENCE_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_PUBLIC_EVIDENCE_CONTRACT_SCHEMA_VERSION:
            raise ValueError("public evidence contract schema_version mismatch")
        if self.package_name != RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME:
            raise ValueError("public evidence contract must belong to relationship_transfer_v3")
        _require_sha256(
            self.source_p1e_report_artifact_id,
            "public evidence trigger artifact id",
        )
        _require_sha256(
            self.semantic_audit_weights_sha256,
            "public evidence semantic audit weights",
        )
        _require_sha256(self.contract_sha256, "public evidence contract_sha256")
        if self.source_required_verdict != "rewrite_public_evidence_contract":
            raise ValueError("public evidence contract has an unsupported trigger verdict")
        if self.history_text_fields != ("user_utterance", "user_reaction"):
            raise ValueError("public evidence history text surface is not frozen")
        if self.history_text_joiner != "\n":
            raise ValueError("public evidence history text joiner is not frozen")
        if self.probe_text_fields != ("current_input",):
            raise ValueError("public evidence probe text surface is not frozen")
        for field_name, value in (
            ("semantic_audit_method", self.semantic_audit_method),
            ("semantic_auditor_version", self.semantic_auditor_version),
            ("semantic_similarity", self.semantic_similarity),
            ("semantic_audit_embedder", self.semantic_audit_embedder),
            ("semantic_audit_model_source", self.semantic_audit_model_source),
            ("condition_anchor_source", self.condition_anchor_source),
            ("human_anchor_status", self.human_anchor_status),
            ("claim_boundary", self.claim_boundary),
        ):
            _require_text(value, field_name)
        if self.condition_anchor_source != "sealed_generator_truth_hidden_summary":
            raise ValueError("public evidence condition anchor source is not sealed")
        if self.score_precision_decimals != 12:
            raise ValueError("public evidence score precision is not frozen")
        if self.top1_tie_policy != "fail_expected_anchor":
            raise ValueError("public evidence top-1 tie policy is not frozen")
        if (
            self.semantic_auditor_version
            != "relationship-public-evidence-auditor.v1"
            or self.semantic_similarity != "cosine"
            or self.semantic_audit_method
            != "frozen_multilingual_embedding_contrast_against_sealed_condition_summaries"
            or self.semantic_audit_embedder != "bge-m3"
            or self.semantic_audit_model_source != "BAAI/bge-m3"
        ):
            raise ValueError("public evidence semantic auditor is not frozen")
        if self.required_evidence_units != 60:
            raise ValueError("public evidence contract requires exactly 60 evidence units")
        if self.required_top1_accuracy != 1.0:
            raise ValueError("public evidence contract requires perfect development top-1")
        for field_name, value in (
            ("minimum_correct_anchor_margin", self.minimum_correct_anchor_margin),
            (
                "minimum_mean_correct_anchor_margin",
                self.minimum_mean_correct_anchor_margin,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 < value < 1.0
            ):
                raise ValueError(f"{field_name} must be inside (0, 1)")
        if self.human_anchor_status != "pending_before_formal":
            raise ValueError("P1f cannot claim a completed human anchor")
        if self.claim_boundary != _PUBLIC_EVIDENCE_CLAIM_BOUNDARY:
            raise ValueError("public evidence claim_boundary is not frozen")


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
    public_evidence_contract: RelationshipPublicEvidenceContract | None = None

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
        if self.package_name in _COMPOSITIONAL_PACKAGE_NAMES:
            self._validate_compositional_transfer()
        elif any(
            (
                self.abstract_conditions,
                self.policy_profiles,
                self.history_condition_bindings,
                self.public_evidence_contract,
            )
        ):
            raise ValueError(
                "relationship_transfer_v1 cannot carry compositional sealed metadata"
            )
        if self.package_name == RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME:
            if self.public_evidence_contract is None:
                raise ValueError("relationship_transfer_v3 requires a public evidence contract")
            if self.public_evidence_contract.package_name != self.package_name:
                raise ValueError("public evidence contract package does not match dataset")
            evidence_units = sum(
                len(observation.histories) + 1 for observation in self.observations
            )
            if self.public_evidence_contract.required_evidence_units != evidence_units:
                raise ValueError(
                    "public evidence contract unit count does not cover the dataset"
                )
        elif self.public_evidence_contract is not None:
            raise ValueError("only relationship_transfer_v3 can carry public evidence metadata")
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
            if self.package_name in _COMPOSITIONAL_PACKAGE_NAMES:
                if len({item.probe_condition_id for item in pair_dynamics}) != 1:
                    raise ValueError(
                        f"{pair_id} compositional siblings must share one probe condition"
                    )
                if len({item.policy_id for item in pair_dynamics}) != 2:
                    raise ValueError(
                        f"{pair_id} compositional siblings must use opposite policy profiles"
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
        minimum_families = 6 if self.package_name in _COMPOSITIONAL_PACKAGE_NAMES else 4
        if len(families) < minimum_families:
            raise ValueError(
                f"{self.package_name} must cover at least {minimum_families} surface families"
            )

    def _validate_compositional_transfer(self) -> None:
        """Reject compositional data solvable by global action/outcome tallying."""

        conditions = {item.condition_id: item for item in self.abstract_conditions}
        policies = {item.policy_id: item for item in self.policy_profiles}
        if len(conditions) != 2:
            raise ValueError(
                f"{self.package_name} requires exactly two abstract conditions"
            )
        if len(policies) != 2:
            raise ValueError(f"{self.package_name} requires exactly two policy profiles")
        expected_condition_ids = set(conditions)
        for policy in policies.values():
            if {condition_id for condition_id, _ in policy.condition_actions} != expected_condition_ids:
                raise ValueError(
                    "every compositional policy must cover every abstract condition"
                )
        for condition_id in expected_condition_ids:
            actions = {policy.action_for(condition_id) for policy in policies.values()}
            if actions != {
                RelationshipAction.STAY_PRESENT_WITHOUT_PROBE,
                RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION,
            }:
                raise ValueError(
                    "compositional policy profiles must be complementary per condition"
                )

        all_history_ids = {
            history.event_id
            for observation in self.observations
            for history in observation.histories
        }
        if len(all_history_ids) != sum(
            len(observation.histories) for observation in self.observations
        ):
            raise ValueError("compositional history event ids must be globally unique")
        bindings = dict(self.history_condition_bindings)
        if len(bindings) != len(self.history_condition_bindings):
            raise ValueError("compositional history condition bindings must be unique")
        if set(bindings) != all_history_ids:
            raise ValueError(
                "compositional history condition bindings must cover every history exactly"
            )
        if not set(bindings.values()) <= expected_condition_ids:
            raise ValueError(
                "compositional history binding references an unknown abstract condition"
            )

        positive_outcomes = set(self.positive_outcomes)
        for observation in self.observations:
            if len(observation.histories) != 4:
                raise ValueError(f"{self.package_name} requires four histories per user")
            history_families = {item.surface_family for item in observation.histories}
            if len(history_families) != 4:
                raise ValueError(
                    "compositional histories must span four distinct surface families"
                )
            if observation.probe_surface_family in history_families:
                raise ValueError(
                    "compositional probe family must be unseen in that user's histories"
                )

            dynamic = self.dynamic_for_scene(observation.scene_id)
            if dynamic.policy_id is None or dynamic.probe_condition_id is None:
                raise ValueError(
                    "compositional dynamic requires policy_id and probe_condition_id"
                )
            try:
                policy = policies[dynamic.policy_id]
            except KeyError as exc:
                raise ValueError(
                    "compositional dynamic references an unknown policy"
                ) from exc
            if dynamic.probe_condition_id not in expected_condition_ids:
                raise ValueError(
                    "compositional dynamic references an unknown probe condition"
                )
            if policy.action_for(dynamic.probe_condition_id) is not dynamic.preferred_action:
                raise ValueError(
                    "compositional preferred action must follow the sealed policy mapping"
                )

            histories_by_condition: dict[str, list[RelationshipHistoryEvent]] = {}
            action_outcome_polarities: dict[RelationshipAction, set[bool]] = {}
            action_counts: dict[RelationshipAction, int] = {}
            for history in observation.histories:
                if history.assistant_action is RelationshipAction.NEUTRAL_NOOP:
                    raise ValueError(
                        "compositional histories cannot use neutral_noop as policy evidence"
                    )
                condition_id = bindings[history.event_id]
                histories_by_condition.setdefault(condition_id, []).append(history)
                correct_action = policy.action_for(condition_id)
                positive = history.typed_outcome in positive_outcomes
                if (history.assistant_action is correct_action) != positive:
                    raise ValueError(
                        "compositional history outcome polarity must agree with its sealed policy"
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
                raise ValueError(
                    "compositional histories must provide two examples per condition"
                )
            for condition_id, histories in histories_by_condition.items():
                if {item.assistant_action for item in histories} != {
                    RelationshipAction.STAY_PRESENT_WITHOUT_PROBE,
                    RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION,
                }:
                    raise ValueError(
                        f"compositional condition {condition_id} must contrast both actions"
                    )
            if action_counts != {
                RelationshipAction.STAY_PRESENT_WITHOUT_PROBE: 2,
                RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION: 2,
            }:
                raise ValueError(
                    "compositional histories must balance both non-noop actions"
                )
            if any(polarities != {False, True} for polarities in action_outcome_polarities.values()):
                raise ValueError(
                    "compositional each action must have one positive and one negative outcome"
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


def _parse_public_evidence_contract(
    raw: dict[str, Any],
) -> RelationshipPublicEvidenceContract:
    _require_exact_keys(
        raw,
        {
            "schema_version",
            "package_name",
            "trigger",
            "public_language_contract",
            "semantic_legibility_audit",
            "human_anchor",
            "claim_boundary",
        },
        source="public_evidence_contract",
    )
    if raw["schema_version"] != RELATIONSHIP_PUBLIC_EVIDENCE_CONTRACT_SCHEMA_VERSION:
        raise ValueError("public evidence contract schema_version mismatch")
    trigger = raw["trigger"]
    language = raw["public_language_contract"]
    semantic = raw["semantic_legibility_audit"]
    human = raw["human_anchor"]
    for source, value in (
        ("public_evidence_contract.trigger", trigger),
        ("public_evidence_contract.public_language_contract", language),
        ("public_evidence_contract.semantic_legibility_audit", semantic),
        ("public_evidence_contract.human_anchor", human),
    ):
        if not isinstance(value, dict):
            raise ValueError(f"{source} must be an object")
    _require_exact_keys(
        trigger,
        {"p1e_report_artifact_id", "required_verdict"},
        source="public_evidence_contract.trigger",
    )
    _require_exact_keys(
        language,
        {
            "history_text_fields",
            "history_text_joiner",
            "probe_text_fields",
            "incident_and_experienced_relational_loss_both_present",
            "direct_action_request_in_probe",
            "condition_name_visible_to_sut",
            "condition_id_visible_to_sut",
            "preferred_action_visible_to_sut",
            "global_action_outcome_balance_preserved",
        },
        source="public_evidence_contract.public_language_contract",
    )
    expected_language_flags = {
        "incident_and_experienced_relational_loss_both_present": True,
        "direct_action_request_in_probe": False,
        "condition_name_visible_to_sut": False,
        "condition_id_visible_to_sut": False,
        "preferred_action_visible_to_sut": False,
        "global_action_outcome_balance_preserved": True,
    }
    for field_name, expected in expected_language_flags.items():
        if language[field_name] is not expected:
            raise ValueError(f"public evidence language flag {field_name} is invalid")
    _require_exact_keys(
        semantic,
        {
            "method",
            "auditor_version",
            "similarity",
            "embedder",
            "model_source",
            "weights_sha256",
            "condition_anchor_source",
            "score_precision_decimals",
            "top1_tie_policy",
            "required_evidence_units",
            "required_top1_accuracy",
            "minimum_correct_anchor_margin",
            "minimum_mean_correct_anchor_margin",
            "evaluation_feedback_to_sut",
        },
        source="public_evidence_contract.semantic_legibility_audit",
    )
    if semantic["evaluation_feedback_to_sut"] is not False:
        raise ValueError("semantic legibility evaluation cannot feed the SUT")
    for field_name in (
        "required_top1_accuracy",
        "minimum_correct_anchor_margin",
        "minimum_mean_correct_anchor_margin",
    ):
        value = semantic[field_name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"public evidence {field_name} must be numeric")
    if type(semantic["required_evidence_units"]) is not int:
        raise ValueError("public evidence required_evidence_units must be an integer")
    if type(semantic["score_precision_decimals"]) is not int:
        raise ValueError("public evidence score_precision_decimals must be an integer")
    _require_exact_keys(
        human,
        {
            "status",
            "minimum_independent_raters",
            "minimum_majority_agreement",
            "labels_available_to_raters",
            "may_feed_learning_or_steering",
        },
        source="public_evidence_contract.human_anchor",
    )
    if (
        human["status"] != "pending_before_formal"
        or type(human["minimum_independent_raters"]) is not int
        or human["minimum_independent_raters"] < 3
        or isinstance(human["minimum_majority_agreement"], bool)
        or not isinstance(human["minimum_majority_agreement"], (int, float))
        or not 0.5 < float(human["minimum_majority_agreement"]) <= 1.0
        or human["labels_available_to_raters"] is not False
        or human["may_feed_learning_or_steering"] is not False
    ):
        raise ValueError("public evidence human anchor contract is invalid")
    history_fields = language["history_text_fields"]
    history_joiner = language["history_text_joiner"]
    probe_fields = language["probe_text_fields"]
    if not isinstance(history_fields, list) or not all(
        isinstance(item, str) for item in history_fields
    ):
        raise ValueError("public evidence history_text_fields must be strings")
    if not isinstance(probe_fields, list) or not all(
        isinstance(item, str) for item in probe_fields
    ):
        raise ValueError("public evidence probe_text_fields must be strings")
    if not isinstance(history_joiner, str):
        raise ValueError("public evidence history_text_joiner must be a string")
    return RelationshipPublicEvidenceContract(
        package_name=_require_text(raw["package_name"], "public evidence package_name"),
        source_p1e_report_artifact_id=_require_sha256(
            trigger["p1e_report_artifact_id"],
            "public evidence P1e artifact id",
        ),
        source_required_verdict=_require_text(
            trigger["required_verdict"],
            "public evidence trigger verdict",
        ),
        history_text_fields=tuple(history_fields),
        history_text_joiner=history_joiner,
        probe_text_fields=tuple(probe_fields),
        semantic_auditor_version=_require_text(
            semantic["auditor_version"],
            "public evidence auditor version",
        ),
        semantic_audit_method=_require_text(
            semantic["method"],
            "public evidence semantic method",
        ),
        semantic_similarity=_require_text(
            semantic["similarity"],
            "public evidence similarity",
        ),
        semantic_audit_embedder=_require_text(
            semantic["embedder"],
            "public evidence embedder",
        ),
        semantic_audit_model_source=_require_text(
            semantic["model_source"],
            "public evidence model source",
        ),
        semantic_audit_weights_sha256=_require_sha256(
            semantic["weights_sha256"],
            "public evidence semantic weights",
        ),
        condition_anchor_source=_require_text(
            semantic["condition_anchor_source"],
            "public evidence condition anchor source",
        ),
        score_precision_decimals=semantic["score_precision_decimals"],
        top1_tie_policy=_require_text(
            semantic["top1_tie_policy"],
            "public evidence top1 tie policy",
        ),
        required_evidence_units=int(semantic["required_evidence_units"]),
        required_top1_accuracy=float(semantic["required_top1_accuracy"]),
        minimum_correct_anchor_margin=float(
            semantic["minimum_correct_anchor_margin"]
        ),
        minimum_mean_correct_anchor_margin=float(
            semantic["minimum_mean_correct_anchor_margin"]
        ),
        human_anchor_status=_require_text(
            human["status"],
            "public evidence human anchor status",
        ),
        claim_boundary=_require_text(
            raw["claim_boundary"],
            "public evidence claim_boundary",
        ),
        contract_sha256=hashlib.sha256(
            canonical_json(raw).encode("utf-8")
        ).hexdigest(),
    )


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
    if truth_schema_version in _COMPOSITIONAL_TRUTH_SCHEMA_VERSIONS:
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
    if truth_schema_version in _COMPOSITIONAL_TRUTH_SCHEMA_VERSIONS:
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
    if truth_schema_version in _COMPOSITIONAL_TRUTH_SCHEMA_VERSIONS:
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
                    if truth_schema_version in _COMPOSITIONAL_TRUTH_SCHEMA_VERSIONS
                    else None
                ),
                probe_condition_id=(
                    _require_text(
                        item["probe_condition_id"],
                        f"{source}.probe_condition_id",
                    )
                    if truth_schema_version in _COMPOSITIONAL_TRUTH_SCHEMA_VERSIONS
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
    public_evidence_contract: RelationshipPublicEvidenceContract | None = None
    fingerprint_parts = [canonical_json(public_raw), canonical_json(truth_raw)]
    if inferred_package == RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME:
        evidence_contract_path = root / "public_evidence_contract.json"
        if not evidence_contract_path.is_file():
            raise FileNotFoundError(
                "relationship_transfer_v3 requires public_evidence_contract.json "
                f"under {root}"
            )
        evidence_contract_raw = _load_json(evidence_contract_path)
        public_evidence_contract = _parse_public_evidence_contract(
            evidence_contract_raw
        )
        fingerprint_parts.append(canonical_json(evidence_contract_raw))
    fingerprint = hashlib.sha256(
        "\n".join(fingerprint_parts).encode("utf-8")
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
        public_evidence_contract=public_evidence_contract,
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
    "RELATIONSHIP_TRANSFER_V3_DATASET_SCHEMA_VERSION",
    "RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME",
    "RELATIONSHIP_TRANSFER_V3_TRUTH_SCHEMA_VERSION",
    "RELATIONSHIP_PUBLIC_EVIDENCE_CONTRACT_SCHEMA_VERSION",
    "RelationshipHistoryEvent",
    "RelationshipObservation",
    "RelationshipPolicyProfile",
    "RelationshipPublicEvidenceContract",
    "RelationshipTransferDataset",
    "load_relationship_transfer_dataset",
    "relationship_transfer_package_dir",
]
