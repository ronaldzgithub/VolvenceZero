"""P1h consumer-training and unseen-qualification split contract.

This module owns an offline, content-addressed experiment boundary.  It does
not train a consumer and never exposes qualification truth to a system under
test.  The already-observed v3 package is training-only; the new v4 package is
qualification-only and may be consumed only after a later consumer protocol
has been frozen.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass
from typing import Any

from lifeform_domain_emogpt.lab.contracts import (
    RelationshipDatasetSplit,
    canonical_json,
)
from lifeform_domain_emogpt.lab.dataset import (
    RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME,
    RELATIONSHIP_TRANSFER_V4_PACKAGE_NAME,
    RelationshipTransferDataset,
    load_relationship_transfer_dataset,
    relationship_transfer_package_dir,
)


RELATIONSHIP_CONSUMER_SPLIT_CONTRACT_SCHEMA_VERSION = (
    "relationship-consumer-split-contract.v1"
)
RELATIONSHIP_CONSUMER_SPLIT_SELECTION_METHOD = (
    "leave_one_surface_family_out_training_only"
)
RELATIONSHIP_CONSUMER_SPLIT_NEXT_ACTION = "consumer_training_freeze_candidate"

_CLAIM_BOUNDARY = (
    "P1h freezes a development-only consumer-training/unseen-qualification "
    "split before any v4 Qwen output. It does not train or qualify a consumer, "
    "open formal hidden test or P2, prove Volvence advantage, human readability, "
    "product value, or any complete Appendable/Readable/Learnable/Steerable claim."
)


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


def _require_int(value: object, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{source} must be an integer")
    return value


def _require_float(value: object, source: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{source} must be numeric")
    return float(value)


@dataclass(frozen=True)
class RelationshipConsumerSplitContract:
    """Frozen P1h boundary between seen calibration data and one-shot data."""

    source_p1g_report_artifact_id: str
    source_required_verdict: str
    training_package_name: str
    training_dataset_fingerprint: str
    training_role: str
    training_contains_seen_qwen_outputs: bool
    qualification_package_name: str
    qualification_dataset_fingerprint: str
    qualification_role: str
    qualification_mirrored_pairs: int
    qualification_histories_per_user: int
    qualification_qwen_outputs_observed_before_freeze: int
    maximum_consumer_revision_rounds: int
    selection_method: str
    preserve_every_candidate: bool
    allowed_feedback: str
    required_valid_rate: float
    minimum_accuracy: float
    maximum_accuracy: float
    minimum_pair_flip_rate: float
    primary_qualification_arms: tuple[str, ...]
    structured_state_minimum_pair_flip_rate: float
    exact_surface_families_disjoint: bool
    exact_scene_ids_disjoint: bool
    exact_event_ids_disjoint: bool
    exact_public_text_disjoint: bool
    qualification_public_inputs_visible_during_training: bool
    qualification_truth_visible_during_training: bool
    qualification_feedback_to_consumer: bool
    evaluation_feedback_to_pe_credit_reward_or_steering: bool
    formal_hidden_test_opened: bool
    p2_enabled: bool
    next_action: str
    claim_boundary: str
    contract_sha256: str
    schema_version: str = RELATIONSHIP_CONSUMER_SPLIT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_CONSUMER_SPLIT_CONTRACT_SCHEMA_VERSION:
            raise ValueError("consumer split contract schema_version mismatch")
        _require_sha256(self.source_p1g_report_artifact_id, "P1h source P1g report")
        _require_sha256(self.training_dataset_fingerprint, "P1h training fingerprint")
        _require_sha256(
            self.qualification_dataset_fingerprint,
            "P1h qualification fingerprint",
        )
        _require_sha256(self.contract_sha256, "P1h contract_sha256")
        if self.source_required_verdict != "consumer_still_underqualified":
            raise ValueError("P1h requires the frozen P1g underqualification verdict")
        if self.training_package_name != RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME:
            raise ValueError("P1h must demote relationship_transfer_v3 to training-only")
        if self.training_role != "consumer_training_only":
            raise ValueError("P1h training role is not frozen")
        if not self.training_contains_seen_qwen_outputs:
            raise ValueError("P1h must record that v3 Qwen outputs have been observed")
        if self.qualification_package_name != RELATIONSHIP_TRANSFER_V4_PACKAGE_NAME:
            raise ValueError("P1h qualification package must be relationship_transfer_v4")
        if self.qualification_role != "unseen_qualification_only":
            raise ValueError("P1h qualification role is not frozen")
        if self.qualification_mirrored_pairs != 12:
            raise ValueError("P1h requires exactly twelve qualification mirrored pairs")
        if self.qualification_histories_per_user != 4:
            raise ValueError("P1h requires four histories per qualification user")
        if self.qualification_qwen_outputs_observed_before_freeze != 0:
            raise ValueError("P1h must freeze before the first v4 Qwen output")
        if self.maximum_consumer_revision_rounds != 3:
            raise ValueError("P1h consumer search budget must remain three rounds")
        if self.selection_method != RELATIONSHIP_CONSUMER_SPLIT_SELECTION_METHOD:
            raise ValueError("P1h selection method is not frozen")
        if not self.preserve_every_candidate:
            raise ValueError("P1h must preserve every consumer candidate")
        if self.allowed_feedback != "training_labels_only_external_baseline_calibration":
            raise ValueError("P1h feedback boundary is not frozen")
        if (
            self.required_valid_rate != 1.0
            or self.minimum_accuracy != 0.625
            or self.maximum_accuracy != 0.875
            or self.minimum_pair_flip_rate != 0.5
            or self.structured_state_minimum_pair_flip_rate != 0.5
        ):
            raise ValueError("P1h qualification gate diverges from P1g")
        if self.primary_qualification_arms != (
            "prompt-steelman",
            "rag-steelman",
        ):
            raise ValueError("P1h must qualify both prompt and RAG steelmen")
        if not all(
            (
                self.exact_surface_families_disjoint,
                self.exact_scene_ids_disjoint,
                self.exact_event_ids_disjoint,
                self.exact_public_text_disjoint,
            )
        ):
            raise ValueError("P1h split isolation guards cannot be disabled")
        if any(
            (
                self.qualification_public_inputs_visible_during_training,
                self.qualification_truth_visible_during_training,
                self.qualification_feedback_to_consumer,
                self.evaluation_feedback_to_pe_credit_reward_or_steering,
                self.formal_hidden_test_opened,
                self.p2_enabled,
            )
        ):
            raise ValueError("P1h cannot open qualification feedback, formal, or P2")
        if self.next_action != RELATIONSHIP_CONSUMER_SPLIT_NEXT_ACTION:
            raise ValueError("P1h next action is not frozen")
        if self.claim_boundary != _CLAIM_BOUNDARY:
            raise ValueError("P1h claim boundary is not frozen")


@dataclass(frozen=True)
class RelationshipConsumerSplitBundle:
    """Validated training and qualification datasets bound by one P1h contract."""

    contract: RelationshipConsumerSplitContract
    training_dataset: RelationshipTransferDataset
    qualification_dataset: RelationshipTransferDataset

    def __post_init__(self) -> None:
        if self.training_dataset.package_name != self.contract.training_package_name:
            raise ValueError("P1h training package lineage mismatch")
        if (
            self.training_dataset.dataset_fingerprint
            != self.contract.training_dataset_fingerprint
        ):
            raise ValueError("P1h training dataset fingerprint mismatch")
        if (
            self.qualification_dataset.package_name
            != self.contract.qualification_package_name
        ):
            raise ValueError("P1h qualification package lineage mismatch")
        if (
            self.qualification_dataset.dataset_fingerprint
            != self.contract.qualification_dataset_fingerprint
        ):
            raise ValueError("P1h qualification dataset fingerprint mismatch")
        if (
            len(self.qualification_dataset.mirrored_pairs())
            != self.contract.qualification_mirrored_pairs
        ):
            raise ValueError("P1h qualification mirrored-pair count mismatch")
        if any(
            dynamic.split is not RelationshipDatasetSplit.HELDOUT
            for dynamic in self.qualification_dataset.dynamics
        ):
            raise ValueError("P1h qualification dynamics must all be heldout")
        if any(
            len(observation.histories)
            != self.contract.qualification_histories_per_user
            for observation in self.qualification_dataset.observations
        ):
            raise ValueError("P1h qualification history count mismatch")
        self._validate_cross_split_isolation()

    @property
    def artifact_id(self) -> str:
        return self.contract.contract_sha256

    def _validate_cross_split_isolation(self) -> None:
        training = self.training_dataset
        qualification = self.qualification_dataset
        training_scene_ids = {item.scene_id for item in training.observations}
        qualification_scene_ids = {item.scene_id for item in qualification.observations}
        if training_scene_ids & qualification_scene_ids:
            raise ValueError("P1h training and qualification scene ids overlap")
        training_scopes = {item.user_scope_hash for item in training.observations}
        qualification_scopes = {
            item.user_scope_hash for item in qualification.observations
        }
        if training_scopes & qualification_scopes:
            raise ValueError("P1h training and qualification user scopes overlap")

        training_events = {
            history.event_id
            for observation in training.observations
            for history in observation.histories
        }
        qualification_events = {
            history.event_id
            for observation in qualification.observations
            for history in observation.histories
        }
        if training_events & qualification_events:
            raise ValueError("P1h training and qualification event ids overlap")

        def surface_families(dataset: RelationshipTransferDataset) -> set[str]:
            return {
                family
                for observation in dataset.observations
                for family in (
                    observation.probe_surface_family,
                    *(history.surface_family for history in observation.histories),
                )
            }

        if surface_families(training) & surface_families(qualification):
            raise ValueError("P1h training and qualification surface families overlap")

        def public_texts(dataset: RelationshipTransferDataset) -> set[str]:
            return {
                text
                for observation in dataset.observations
                for text in (
                    observation.current_input,
                    *(
                        item
                        for history in observation.histories
                        for item in (history.user_utterance, history.user_reaction)
                    ),
                )
            }

        if public_texts(training) & public_texts(qualification):
            raise ValueError("P1h training and qualification public text overlaps")


@dataclass(frozen=True)
class RelationshipConsumerTrainingView:
    """The only P1i-facing view; qualification observations are absent by type."""

    contract: RelationshipConsumerSplitContract
    training_dataset: RelationshipTransferDataset

    def __post_init__(self) -> None:
        if self.training_dataset.package_name != self.contract.training_package_name:
            raise ValueError("P1h training-view package lineage mismatch")
        if (
            self.training_dataset.dataset_fingerprint
            != self.contract.training_dataset_fingerprint
        ):
            raise ValueError("P1h training-view dataset fingerprint mismatch")


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _parse_contract(raw: dict[str, Any]) -> RelationshipConsumerSplitContract:
    _require_exact_keys(
        raw,
        {
            "schema_version",
            "trigger",
            "training_split",
            "qualification_split",
            "consumer_search_budget",
            "frozen_qualification_gate",
            "isolation_guards",
            "experiment_guards",
            "next_action",
            "claim_boundary",
        },
        source="consumer_split_contract",
    )
    trigger = raw["trigger"]
    training = raw["training_split"]
    qualification = raw["qualification_split"]
    budget = raw["consumer_search_budget"]
    gate = raw["frozen_qualification_gate"]
    isolation = raw["isolation_guards"]
    guards = raw["experiment_guards"]
    for source, value in (
        ("consumer_split_contract.trigger", trigger),
        ("consumer_split_contract.training_split", training),
        ("consumer_split_contract.qualification_split", qualification),
        ("consumer_split_contract.consumer_search_budget", budget),
        ("consumer_split_contract.frozen_qualification_gate", gate),
        ("consumer_split_contract.isolation_guards", isolation),
        ("consumer_split_contract.experiment_guards", guards),
    ):
        if not isinstance(value, dict):
            raise ValueError(f"{source} must be an object")
    _require_exact_keys(
        trigger,
        {"p1g_report_artifact_id", "required_verdict"},
        source="consumer_split_contract.trigger",
    )
    _require_exact_keys(
        training,
        {
            "package_name",
            "dataset_fingerprint",
            "role",
            "contains_seen_qwen_outputs",
        },
        source="consumer_split_contract.training_split",
    )
    _require_exact_keys(
        qualification,
        {
            "package_name",
            "dataset_fingerprint",
            "role",
            "mirrored_pairs",
            "histories_per_user",
            "qwen_outputs_observed_before_freeze",
        },
        source="consumer_split_contract.qualification_split",
    )
    _require_exact_keys(
        budget,
        {
            "maximum_revision_rounds",
            "selection_method",
            "preserve_every_candidate",
            "allowed_feedback",
        },
        source="consumer_split_contract.consumer_search_budget",
    )
    _require_exact_keys(
        gate,
        {
            "required_valid_rate",
            "minimum_accuracy",
            "maximum_accuracy",
            "minimum_pair_flip_rate",
            "primary_qualification_arms",
            "structured_state_minimum_pair_flip_rate",
        },
        source="consumer_split_contract.frozen_qualification_gate",
    )
    _require_exact_keys(
        isolation,
        {
            "exact_surface_families_disjoint",
            "exact_scene_ids_disjoint",
            "exact_event_ids_disjoint",
            "exact_public_text_disjoint",
        },
        source="consumer_split_contract.isolation_guards",
    )
    _require_exact_keys(
        guards,
        {
            "qualification_public_inputs_visible_during_training",
            "qualification_truth_visible_during_training",
            "qualification_feedback_to_consumer",
            "evaluation_feedback_to_pe_credit_reward_or_steering",
            "formal_hidden_test_opened",
            "p2_enabled",
        },
        source="consumer_split_contract.experiment_guards",
    )
    primary_arms = gate["primary_qualification_arms"]
    if not isinstance(primary_arms, list) or not all(
        isinstance(item, str) for item in primary_arms
    ):
        raise ValueError("P1h primary_qualification_arms must be strings")
    boolean_fields = (
        ("training_contains_seen_qwen_outputs", training["contains_seen_qwen_outputs"]),
        ("preserve_every_candidate", budget["preserve_every_candidate"]),
        *tuple((field_name, isolation[field_name]) for field_name in isolation),
        *tuple((field_name, guards[field_name]) for field_name in guards),
    )
    for field_name, value in boolean_fields:
        if not isinstance(value, bool):
            raise ValueError(f"P1h {field_name} must be boolean")
    return RelationshipConsumerSplitContract(
        schema_version=_require_text(raw["schema_version"], "P1h schema_version"),
        source_p1g_report_artifact_id=_require_sha256(
            trigger["p1g_report_artifact_id"],
            "P1h P1g artifact",
        ),
        source_required_verdict=_require_text(
            trigger["required_verdict"],
            "P1h trigger verdict",
        ),
        training_package_name=_require_text(
            training["package_name"],
            "P1h training package",
        ),
        training_dataset_fingerprint=_require_sha256(
            training["dataset_fingerprint"],
            "P1h training fingerprint",
        ),
        training_role=_require_text(training["role"], "P1h training role"),
        training_contains_seen_qwen_outputs=training["contains_seen_qwen_outputs"],
        qualification_package_name=_require_text(
            qualification["package_name"],
            "P1h qualification package",
        ),
        qualification_dataset_fingerprint=_require_sha256(
            qualification["dataset_fingerprint"],
            "P1h qualification fingerprint",
        ),
        qualification_role=_require_text(
            qualification["role"],
            "P1h qualification role",
        ),
        qualification_mirrored_pairs=_require_int(
            qualification["mirrored_pairs"],
            "P1h mirrored_pairs",
        ),
        qualification_histories_per_user=_require_int(
            qualification["histories_per_user"],
            "P1h histories_per_user",
        ),
        qualification_qwen_outputs_observed_before_freeze=_require_int(
            qualification["qwen_outputs_observed_before_freeze"],
            "P1h qwen output count",
        ),
        maximum_consumer_revision_rounds=_require_int(
            budget["maximum_revision_rounds"],
            "P1h revision budget",
        ),
        selection_method=_require_text(
            budget["selection_method"],
            "P1h selection method",
        ),
        preserve_every_candidate=budget["preserve_every_candidate"],
        allowed_feedback=_require_text(
            budget["allowed_feedback"],
            "P1h allowed feedback",
        ),
        required_valid_rate=_require_float(
            gate["required_valid_rate"],
            "P1h required_valid_rate",
        ),
        minimum_accuracy=_require_float(
            gate["minimum_accuracy"],
            "P1h minimum_accuracy",
        ),
        maximum_accuracy=_require_float(
            gate["maximum_accuracy"],
            "P1h maximum_accuracy",
        ),
        minimum_pair_flip_rate=_require_float(
            gate["minimum_pair_flip_rate"],
            "P1h minimum_pair_flip_rate",
        ),
        primary_qualification_arms=tuple(primary_arms),
        structured_state_minimum_pair_flip_rate=_require_float(
            gate["structured_state_minimum_pair_flip_rate"],
            "P1h structured-state pair flip",
        ),
        exact_surface_families_disjoint=isolation[
            "exact_surface_families_disjoint"
        ],
        exact_scene_ids_disjoint=isolation["exact_scene_ids_disjoint"],
        exact_event_ids_disjoint=isolation["exact_event_ids_disjoint"],
        exact_public_text_disjoint=isolation["exact_public_text_disjoint"],
        qualification_public_inputs_visible_during_training=guards[
            "qualification_public_inputs_visible_during_training"
        ],
        qualification_truth_visible_during_training=guards[
            "qualification_truth_visible_during_training"
        ],
        qualification_feedback_to_consumer=guards[
            "qualification_feedback_to_consumer"
        ],
        evaluation_feedback_to_pe_credit_reward_or_steering=guards[
            "evaluation_feedback_to_pe_credit_reward_or_steering"
        ],
        formal_hidden_test_opened=guards["formal_hidden_test_opened"],
        p2_enabled=guards["p2_enabled"],
        next_action=_require_text(raw["next_action"], "P1h next_action"),
        claim_boundary=_require_text(raw["claim_boundary"], "P1h claim_boundary"),
        contract_sha256=hashlib.sha256(
            canonical_json(raw).encode("utf-8")
        ).hexdigest(),
    )


def load_relationship_consumer_split_bundle(
    contract_path: pathlib.Path | None = None,
) -> RelationshipConsumerSplitBundle:
    path = pathlib.Path(
        contract_path
        or relationship_transfer_package_dir(RELATIONSHIP_TRANSFER_V4_PACKAGE_NAME)
        / "consumer_split_contract.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"P1h requires consumer_split_contract.json at {path}")
    contract = _parse_contract(_load_json(path))
    training_dataset = load_relationship_transfer_dataset(
        package_name=contract.training_package_name
    )
    qualification_dataset = load_relationship_transfer_dataset(
        path.parent,
        package_name=contract.qualification_package_name,
    )
    return RelationshipConsumerSplitBundle(
        contract=contract,
        training_dataset=training_dataset,
        qualification_dataset=qualification_dataset,
    )


def load_relationship_consumer_training_view(
    contract_path: pathlib.Path | None = None,
) -> RelationshipConsumerTrainingView:
    """Load P1i calibration data without materializing v4 public inputs or truth."""

    path = pathlib.Path(
        contract_path
        or relationship_transfer_package_dir(RELATIONSHIP_TRANSFER_V4_PACKAGE_NAME)
        / "consumer_split_contract.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"P1h requires consumer_split_contract.json at {path}")
    contract = _parse_contract(_load_json(path))
    return RelationshipConsumerTrainingView(
        contract=contract,
        training_dataset=load_relationship_transfer_dataset(
            package_name=contract.training_package_name
        ),
    )


__all__ = [
    "RELATIONSHIP_CONSUMER_SPLIT_CONTRACT_SCHEMA_VERSION",
    "RELATIONSHIP_CONSUMER_SPLIT_NEXT_ACTION",
    "RELATIONSHIP_CONSUMER_SPLIT_SELECTION_METHOD",
    "RelationshipConsumerSplitBundle",
    "RelationshipConsumerSplitContract",
    "RelationshipConsumerTrainingView",
    "load_relationship_consumer_split_bundle",
    "load_relationship_consumer_training_view",
]
