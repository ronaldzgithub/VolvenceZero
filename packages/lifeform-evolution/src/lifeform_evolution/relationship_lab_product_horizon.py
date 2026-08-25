"""Cross-process Relationship Lab product-horizon development campaign.

This owner deliberately stays outside the product runtime.  It runs the five
typed-control arms from the frozen product source and may additionally invoke
the two public-context baselines through an injected, resident
``RelationshipProductBaselineSuite``.  Every Volvence logical session starts
one fresh OS child.  Decision children use a two-message handshake: they first
publish and fsync an outcome-free pre-action receipt; only then may the parent
join the sealed environment and return typed settlement evidence on stdin.

The artifact can only support a scoped typed-control-effect claim.  It does
not contain a relationship residual executor or user-visible generation and
therefore hard-codes the stronger claim flags to ``False``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import pathlib
import platform
import queue
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Mapping, Sequence
from uuid import uuid4

from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidence,
    DialogueExternalOutcomeEvidenceSource,
    DialogueExternalOutcomeKind,
)
from volvence_zero.memory.persistence import FileSystemPersistenceBackend
from volvence_zero.owner_hydration_store import OwnerHydrationStore
from volvence_zero.runtime import WiringLevel
from volvence_zero.social import (
    PreferenceActionForecastProposal,
    PreferenceActionForecastRequest,
    PreferenceActionForecastRuntime,
    SocialRecordStore,
)
from volvence_zero.social_cognition import (
    PreferenceActionForecast,
    PreferenceActionOutcomeEvidence,
    preference_action_forecast_from_payload,
    preference_action_forecast_to_payload,
)
from volvence_zero.substrate import SubstrateSnapshot, SurfaceKind

from lifeform_domain_emogpt.lab.contracts import canonical_json, sha256_json
from lifeform_domain_emogpt.lab import relationship_product_pilot_source as product_source_owner
from lifeform_domain_emogpt.lab.relationship_product_pilot_source import (
    ProductPilotEvaluatorDecisionSession,
    ProductPilotPublicDecisionSession,
    ProductPilotPublicOnboardingSession,
    ProductPilotPublicSubject,
    RelationshipProductPilotEvaluatorBundle,
    RelationshipProductPilotPublicView,
    build_relationship_product_pilot_environment,
    build_relationship_product_pilot_evaluator_bundle,
    build_relationship_product_pilot_public_view,
    load_relationship_product_pilot_source_protocol,
)
from lifeform_domain_emogpt.lab.relationship_product_pulse import (
    RelationshipProductOnboardingInput,
    RelationshipProductPreActionRequest,
    RelationshipProductPulseAuthorization,
    RelationshipProductSettlementInput,
    append_relationship_product_onboarding,
    prepare_relationship_product_preaction,
    settle_relationship_product_pulse,
)
from lifeform_domain_emogpt.relationship_action_contracts import (
    RELATIONSHIP_ACTIONS,
    RELATIONSHIP_OUTCOMES,
    RelationshipAction,
)
from lifeform_domain_emogpt.relationship_action_gate import (
    RelationshipActionGateCheckpoint,
    RelationshipActionGateDecision,
    RelationshipActionGateMode,
)
from lifeform_domain_emogpt.relationship_forecast import (
    BoundedRelationshipPreferenceForecastRuntime,
)
from lifeform_domain_emogpt.relationship_condition_reader import (
    PrototypeRelationshipPreferenceForecastRuntime,
    RelationshipConditionPrototype,
    RelationshipConditionReaderArtifact,
)
from lifeform_evolution.relationship_lab_product_baselines import (
    ProductBaselineArm,
    ProductBaselineInput,
    ProductBaselineTokenBudget,
    ProductCurrentObservation,
    ProductPublicHistoryBlock,
    RelationshipProductBaselineSuite,
)
from lifeform_evolution import (
    relationship_lab_product_baseline_dispatcher as baseline_dispatcher_owner,
)
from lifeform_evolution import (
    relationship_lab_product_baselines as product_baselines_owner,
)
from lifeform_evolution import relationship_lab_baseline as baseline_policy_owner
from lifeform_evolution import (
    relationship_lab_product_model_adapters as product_model_adapters_owner,
)
from lifeform_evolution.relationship_lab_product_baseline_dispatcher import (
    ProductBaselineCurrentObservationLineage,
    ProductBaselineDecisionBoundary,
    ProductBaselineDispatcherFatalResponse,
    ProductBaselineDispatcherReceivedResponse,
    ProductBaselineDispatcherRequest,
    ProductBaselineHistoryBlockLineage,
    parse_product_baseline_dispatcher_response_line,
)
from lifeform_evolution.relationship_lab_product_model_adapters import (
    BGE_M3_MODEL_ID,
    BGE_M3_SENTENCE_TRANSFORMERS_VERSION,
    BGE_M3_WEIGHT_BYTES_SHA256,
    PrecomputedPublicEmbeddingTable,
    PrecomputedPublicSemanticEmbedder,
    bge_m3_public_semantic_embedder,
    bge_m3_weight_pinned_embedder_identity,
    build_precomputed_public_embedding_table,
    load_precomputed_public_embedding_table,
)


RELATIONSHIP_PRODUCT_HORIZON_SCHEMA_VERSION_V1 = "relationship-product-horizon-campaign.v1"
RELATIONSHIP_PRODUCT_HORIZON_SCHEMA_VERSION_V2 = "relationship-product-horizon-campaign.v2"
# The product-facing default advances to v2.  The v1 constant and packaged
# protocol remain registered so every already-published v1 campaign can still
# be validated byte-for-byte.
RELATIONSHIP_PRODUCT_HORIZON_SCHEMA_VERSION = RELATIONSHIP_PRODUCT_HORIZON_SCHEMA_VERSION_V2
RELATIONSHIP_PRODUCT_WORKER_REQUEST_SCHEMA_VERSION = "relationship-product-worker-request.v1"
RELATIONSHIP_PRODUCT_PREACTION_RECEIPT_SCHEMA_VERSION = "relationship-product-preaction-receipt.v1"
RELATIONSHIP_PRODUCT_PREACTION_RECEIPT_SCHEMA_VERSION_V2 = "relationship-product-preaction-receipt.v2"
PREFERENCE_ACTION_FORECAST_SNAPSHOT_SCHEMA_VERSION = "preference-action-forecast-snapshot.v1"
RELATIONSHIP_PRODUCT_POSTACTION_RECEIPT_SCHEMA_VERSION = "relationship-product-postaction-receipt.v1"
RELATIONSHIP_PRODUCT_POSTACTION_RECEIPT_SCHEMA_VERSION_V2 = (
    "relationship-product-postaction-receipt.v2"
)
RELATIONSHIP_PRODUCT_ONBOARDING_RECEIPT_SCHEMA_VERSION = "relationship-product-onboarding-receipt.v1"
RELATIONSHIP_PRODUCT_ONBOARDING_RECEIPT_SCHEMA_VERSION_V2 = (
    "relationship-product-onboarding-receipt.v2"
)
RELATIONSHIP_PRODUCT_WORKER_SOURCE_LINEAGE_SCHEMA_VERSION = (
    "relationship-product-worker-source-lineage.v1"
)
RELATIONSHIP_PRODUCT_REPORT_SCHEMA_VERSION = "relationship-product-horizon-report.v1"
RELATIONSHIP_PRODUCT_MANIFEST_SCHEMA_VERSION = "relationship-product-horizon-manifest.v1"
_BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"

_OWNER_NAME = "social_record_store"
_INTERLOCUTOR_ID = "primary"
_GATE_CHECKPOINT_FILENAME = "relationship_action_gate_checkpoint.json"
_POSITIVE_OUTCOMES = frozenset({DialogueExternalOutcomeKind.HELPED.value, DialogueExternalOutcomeKind.FELT_HEARD.value})
_SAFETY_NEGATIVE_OUTCOMES = frozenset(
    {
        DialogueExternalOutcomeKind.MISSED.value,
        DialogueExternalOutcomeKind.OVER_DIRECTIVE.value,
    }
)
_VOLVENCE_ARMS_V1 = (
    "volvence_full",
    "appendable_frozen_onboarding",
    "readable_permuted",
    "credit_withheld",
    "strict_noop",
)
_VOLVENCE_ARMS_V2 = (
    "volvence_full",
    "appendable_frozen_onboarding",
    "readable_unnamed_legacy",
    "credit_withheld",
    "strict_noop",
)
# The published v1 runner and validator still use this historical name. Keep
# it pinned to v1 until the incomplete v2 registry reaches every call site and
# receives its own qualification.
_VOLVENCE_ARMS = _VOLVENCE_ARMS_V1
_BASELINE_ARMS = ("native_full_history", "selective_rag")
_ALL_ARMS_V1 = (*_VOLVENCE_ARMS_V1, *_BASELINE_ARMS)
_ALL_ARMS_V2 = (*_VOLVENCE_ARMS_V2, *_BASELINE_ARMS)
_PRODUCT_CONDITION_PROTOTYPES = (
    RelationshipConditionPrototype(
        label="agency_displacement",
        summary=(
            "别人越过当事人的确认，替其表达、选择、承诺或作决定，使当事人失去发言权、"
            "决定权以及按自己节奏回应的空间。"
        ),
    ),
    RelationshipConditionPrototype(
        label="belonging_erasure",
        summary=(
            "当事人被共同经历、邀请、名单、记忆或关系网络遗漏和排除，因而感觉自己在关系中的"
            "位置消失、没有被算在其中。"
        ),
    ),
)
_BGE_M3_WEIGHTS_SHA256 = BGE_M3_WEIGHT_BYTES_SHA256
_SUBPROCESS_ENVIRONMENT_CONTRACT = {
    "HF_HUB_OFFLINE": "1",
    "PYTHONIOENCODING": "utf-8:strict",
    "PYTHONNOUSERSITE": "1",
    "PYTHONUTF8": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
_SUBPROCESS_ENVIRONMENT_CONTRACT_V2 = {
    **_SUBPROCESS_ENVIRONMENT_CONTRACT,
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHOME": None,
    "PYTHONPATH": None,
    "PYTHONSAFEPATH": "1",
}
_CRITICAL_LOCAL_MODULE_NAMES = (
    "lifeform_domain_emogpt.lab.relationship_product_pilot_source",
    "lifeform_domain_emogpt.lab.relationship_product_pulse",
    "lifeform_domain_emogpt.relationship_action_gate",
    "lifeform_domain_emogpt.relationship_condition_reader",
    "lifeform_domain_emogpt.relationship_forecast",
    "lifeform_evolution.relationship_lab_product_horizon",
    "volvence_zero.owner_hydration_store",
    "volvence_zero.social.record_store",
    "volvence_zero.social_cognition",
)
_EXECUTION_SOURCE_KEYS = (
    "baseline_dispatcher_cli",
    "baseline_dispatcher_implementation",
    "baseline_implementation",
    "baseline_policy_implementation",
    "campaign_cli",
    "campaign_implementation",
    "model_adapters_implementation",
)
_LOCAL_EXECUTION_SOURCE_TREE_SCHEMA_VERSION = (
    "relationship-product-local-execution-source-tree.v1"
)
_LOCAL_EXECUTION_SOURCE_TREE_SELECTOR = "packages/*/src/**/*.py"
_NON_AUTHORIZING_TEST_PROTOCOL_ENV = (
    "VOLVENCE_RELATIONSHIP_PRODUCT_HORIZON_NON_AUTHORIZING_TEST_PROTOCOL"
)
_LOCAL_EXECUTION_SOURCE_ENTRYPOINTS = (
    "scripts/run_relationship_lab_product_baseline_dispatcher.py",
    "scripts/run_relationship_lab_product_horizon.py",
)
_LOCAL_EXECUTION_RESOURCE_PATHS = (
    (
        "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/"
        "lab_protocols/relationship_product_pilot_source_v1.json"
    ),
    (
        "packages/lifeform-evolution/src/lifeform_evolution/"
        "prompts/relationship_lab_full_history_steelman_v2.txt"
    ),
    (
        "packages/lifeform-evolution/src/lifeform_evolution/"
        "prompts/relationship_lab_rag_steelman_v2.txt"
    ),
    (
        "packages/lifeform-evolution/src/lifeform_evolution/"
        "prompts/relationship_lab_stateless_v1.txt"
    ),
    (
        "packages/lifeform-evolution/src/lifeform_evolution/"
        "protocols/relationship_product_horizon_campaign_v1.json"
    ),
    (
        "packages/lifeform-evolution/src/lifeform_evolution/"
        "schemas/relationship_action_choice.schema.json"
    ),
)
_LOCAL_EXECUTION_ACTIVE_PROTOCOL_PATH = (
    "packages/lifeform-evolution/src/lifeform_evolution/"
    "protocols/relationship_product_horizon_campaign_v2.json"
)
_LOCAL_EXECUTION_SOURCE_CANONICALIZATION = (
    "utf8-strict-newlines-to-lf-preserve-eof.v1"
)
_ACTIVE_PROTOCOL_RESOURCE_SCHEMA_VERSION = (
    "relationship-product-active-protocol-resource.v1"
)
_HORIZON_SEGMENT_WINDOWS = (
    ("post_reversal", (12, 15)),
    ("correction", (16, 17)),
    ("post_correction", (18, 19)),
    ("return_after_gap", (20, 21)),
    ("mixed_stress", (22, 23)),
)
_MINIMUM_NONNEGATIVE_HORIZON_SEGMENTS_PER_COMPARATOR = 4
_MINIMUM_POSITIVE_HORIZON_SEGMENTS_PER_COMPARATOR = 2
_SEALED_KEYS = frozenset(
    {
        "active_policy_mode",
        "condition_id",
        "dynamic_id",
        "environment_seed",
        "phase_id",
        "policy_id",
        "preferred_action_id",
        "scene_id",
        "stage_id",
        "subject_id",
        "subject_seed",
    }
)


class RelationshipProductArm(str, Enum):
    VOLVENCE_FULL = "volvence_full"
    APPENDABLE_FROZEN_ONBOARDING = "appendable_frozen_onboarding"
    READABLE_PERMUTED = "readable_permuted"
    READABLE_UNNAMED_LEGACY = "readable_unnamed_legacy"
    CREDIT_WITHHELD = "credit_withheld"
    STRICT_NOOP = "strict_noop"
    NATIVE_FULL_HISTORY = "native_full_history"
    SELECTIVE_RAG = "selective_rag"


@dataclass(frozen=True)
class RelationshipProductHorizonProtocol:
    protocol_id: str
    raw_sha256: str
    cohort_id: str
    source_protocol_id: str
    source_protocol_raw_sha256: str
    source_implementation_sha256: str
    public_plan_sha256: str
    sealed_evaluator_bundle_sha256: str
    subject_count: int
    onboarding_sessions_per_subject: int
    decision_sessions_per_subject: int
    baseline_model_source: str
    baseline_model_id: str
    baseline_model_revision: str
    baseline_model_weights_sha256: str
    baseline_tokenizer_id: str
    baseline_action_schema_sha256: str
    baseline_generation_config_sha256: str
    baseline_native_prompt_sha256: str
    baseline_rag_prompt_sha256: str
    baseline_cuda_device: str
    context_window_tokens: int
    generation_token_reserve: int
    generation_seed_namespace: str
    generation_prefill_chunk_size: int
    generation_use_cache: bool
    generation_torch_version: str
    generation_transformers_version: str
    semantic_model_source: str
    semantic_model_revision: str
    semantic_device: str
    semantic_table_artifact_id: str
    semantic_table_raw_sha256: str
    semantic_table_record_count: int
    rag_top_k: int
    primary_window: tuple[int, int]
    development_directional_effect_floor: float
    minimum_subjects_with_positive_paired_effect: int
    maximum_full_safety_rate_increase_vs_any_comparator: float
    execution_source_sha256s: tuple[tuple[str, str], ...]
    local_execution_source_tree_sha256: str | None = None
    local_execution_source_file_count: int = 0
    local_execution_source_canonical_bytes: int = 0
    semantic_model_weights_sha256: str | None = None
    semantic_sentence_transformers_version: str | None = None
    condition_reader_artifact_id: str | None = None
    baseline_constrained_action_choice: bool = False
    schema_version: str = RELATIONSHIP_PRODUCT_HORIZON_SCHEMA_VERSION_V1

    @property
    def is_v2(self) -> bool:
        return self.schema_version == RELATIONSHIP_PRODUCT_HORIZON_SCHEMA_VERSION_V2

    @property
    def volvence_arms(self) -> tuple[str, ...]:
        return _VOLVENCE_ARMS_V2 if self.is_v2 else _VOLVENCE_ARMS_V1

    @property
    def all_arms(self) -> tuple[str, ...]:
        return (*self.volvence_arms, *_BASELINE_ARMS)

    @property
    def persists_full_forecast(self) -> bool:
        return self.is_v2

    def __post_init__(self) -> None:
        _require_sha256(self.protocol_id, "protocol_id")
        _require_sha256(self.raw_sha256, "raw_sha256")
        for field_name, value in (
            ("source_protocol_id", self.source_protocol_id),
            ("source_protocol_raw_sha256", self.source_protocol_raw_sha256),
            ("source_implementation_sha256", self.source_implementation_sha256),
            ("public_plan_sha256", self.public_plan_sha256),
            ("sealed_evaluator_bundle_sha256", self.sealed_evaluator_bundle_sha256),
            ("baseline_model_weights_sha256", self.baseline_model_weights_sha256),
            ("baseline_action_schema_sha256", self.baseline_action_schema_sha256),
            (
                "baseline_generation_config_sha256",
                self.baseline_generation_config_sha256,
            ),
            ("baseline_native_prompt_sha256", self.baseline_native_prompt_sha256),
            ("baseline_rag_prompt_sha256", self.baseline_rag_prompt_sha256),
            ("semantic_table_artifact_id", self.semantic_table_artifact_id),
            ("semantic_table_raw_sha256", self.semantic_table_raw_sha256),
        ):
            _require_sha256(value, field_name)
        if not self.baseline_tokenizer_id:
            raise ValueError("baseline_tokenizer_id must be non-empty")
        if self.execution_source_sha256s:
            if tuple(key for key, _digest_value in self.execution_source_sha256s) != _EXECUTION_SOURCE_KEYS:
                raise ValueError("execution source pin keys/order drifted")
            for key, digest_value in self.execution_source_sha256s:
                _require_sha256(digest_value, f"execution source {key}")
        if self.schema_version not in {
            RELATIONSHIP_PRODUCT_HORIZON_SCHEMA_VERSION_V1,
            RELATIONSHIP_PRODUCT_HORIZON_SCHEMA_VERSION_V2,
        }:
            raise ValueError("relationship product campaign schema mismatch")
        if self.subject_count != 8 or self.onboarding_sessions_per_subject != 4:
            raise ValueError("product campaign requires eight subjects and four onboarding sessions")
        if self.decision_sessions_per_subject != 24:
            raise ValueError("product campaign requires twenty-four decision sessions")
        if self.generation_prefill_chunk_size != 2048:
            raise ValueError("product campaign prefill chunk size must be 2048")
        if self.generation_use_cache is not True:
            raise ValueError("product campaign chunked prefill requires cache")
        if self.generation_torch_version != "2.12.0+cu126":
            raise ValueError("product campaign torch runtime version drifted")
        if self.generation_transformers_version != "5.9.0":
            raise ValueError("product campaign transformers runtime version drifted")
        if self.primary_window != (12, 23):
            raise ValueError("product campaign primary window must be decision indices 12..23")
        if not 0.0 < self.development_directional_effect_floor <= 1.0:
            raise ValueError("product campaign directional effect floor must be in (0, 1]")
        if not 1 <= self.minimum_subjects_with_positive_paired_effect <= self.subject_count:
            raise ValueError("minimum positive paired subjects must be within the cohort")
        if self.rag_top_k < 1:
            raise ValueError("product campaign RAG top_k must be positive")
        expected_record_count = 30 if self.is_v2 else 28
        if self.semantic_table_record_count != expected_record_count:
            raise ValueError(
                "product campaign public semantic table must have "
                f"{expected_record_count} records"
            )
        if self.is_v2:
            _require_sha256(
                self.condition_reader_artifact_id,
                "condition_reader_artifact_id",
            )
            frozen_v2_digests = (
                self.baseline_generation_config_sha256,
                self.baseline_native_prompt_sha256,
                self.baseline_rag_prompt_sha256,
                self.semantic_table_artifact_id,
                self.semantic_table_raw_sha256,
                self.semantic_model_weights_sha256,
                self.condition_reader_artifact_id,
            )
            if self.execution_source_sha256s:
                raise ValueError(
                    "v2 product campaign uses the canonical source tree, not legacy raw pins"
                )
            if any(digest == "0" * 64 for digest in frozen_v2_digests):
                raise ValueError("v2 product campaign contains an unfrozen digest pin")
            _require_sha256(
                self.local_execution_source_tree_sha256,
                "local_execution_source_tree_sha256",
            )
            _require_sha256(
                self.semantic_model_weights_sha256,
                "semantic_model_weights_sha256",
            )
            if self.semantic_model_weights_sha256 != _BGE_M3_WEIGHTS_SHA256:
                raise ValueError("v2 semantic model weight pin drifted")
            if (
                self.semantic_sentence_transformers_version
                != BGE_M3_SENTENCE_TRANSFORMERS_VERSION
            ):
                raise ValueError("v2 sentence-transformers runtime pin drifted")
            if self.local_execution_source_tree_sha256 == "0" * 64:
                raise ValueError("v2 product campaign source-tree pin is unfrozen")
            if self.local_execution_source_file_count < 1:
                raise ValueError("v2 product campaign source-tree file count must be positive")
            if self.local_execution_source_canonical_bytes < 1:
                raise ValueError("v2 product campaign source-tree byte count must be positive")
            expected_reader = relationship_product_condition_reader_artifact()
            if self.condition_reader_artifact_id != expected_reader.artifact_id:
                raise ValueError("product condition-reader artifact pin drifted")
            if self.baseline_constrained_action_choice is not True:
                raise ValueError("v2 product campaign requires constrained baseline action choice")
        elif (
            self.condition_reader_artifact_id is not None
            or self.baseline_constrained_action_choice
            or self.local_execution_source_tree_sha256 is not None
            or self.local_execution_source_file_count != 0
            or self.local_execution_source_canonical_bytes != 0
            or self.semantic_model_weights_sha256 is not None
            or self.semantic_sentence_transformers_version is not None
        ):
            raise ValueError("v1 product campaign cannot claim v2 reader/decoder controls")
        if not 0.0 <= self.maximum_full_safety_rate_increase_vs_any_comparator <= 1.0:
            raise ValueError("product campaign safety noninferiority margin must be in [0, 1]")


@dataclass(frozen=True)
class RelationshipProductCampaignSelection:
    """Test-only cohort prefix; the default always executes the full protocol."""

    subject_count: int = 8
    onboarding_session_count: int = 4
    decision_session_count: int = 24

    def __post_init__(self) -> None:
        for name, value, maximum in (
            ("subject_count", self.subject_count, 8),
            ("onboarding_session_count", self.onboarding_session_count, 4),
            ("decision_session_count", self.decision_session_count, 24),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be in [1, {maximum}]")

    @property
    def is_full(self) -> bool:
        return (self.subject_count, self.onboarding_session_count, self.decision_session_count) == (8, 4, 24)


def relationship_product_condition_reader_artifact() -> RelationshipConditionReaderArtifact:
    """Return the frozen public-only named reader used by product campaign v2."""

    return RelationshipConditionReaderArtifact(
        embedding_model_id=f"{BGE_M3_MODEL_ID}@revision:{_BGE_M3_REVISION}",
        embedding_weights_sha256=_BGE_M3_WEIGHTS_SHA256,
        prototypes=_PRODUCT_CONDITION_PROTOTYPES,
        softmax_temperature=0.05,
        semantic_similarity="cosine",
    )


def relationship_product_horizon_protocol_path(
    version: str = "v2",
) -> pathlib.Path:
    if version not in {"v1", "v2"}:
        raise ValueError(f"unknown relationship product protocol version: {version}")
    return (
        pathlib.Path(__file__).resolve().parent
        / "protocols"
        / f"relationship_product_horizon_campaign_{version}.json"
    )


def relationship_product_horizon_protocol_paths() -> tuple[pathlib.Path, ...]:
    """Registered immutable preregistrations, newest first."""

    return (
        relationship_product_horizon_protocol_path("v2"),
        relationship_product_horizon_protocol_path("v1"),
    )


def _non_authorizing_test_protocol_path() -> pathlib.Path | None:
    """Return the explicit pytest-only protocol override, never a production fallback."""

    raw_path = os.environ.get(_NON_AUTHORIZING_TEST_PROTOCOL_ENV)
    if raw_path is None:
        return None
    if "PYTEST_CURRENT_TEST" not in os.environ:
        raise RuntimeError("non-authorizing product protocol override is pytest-only")
    path = pathlib.Path(raw_path).resolve()
    if not path.name.endswith(".non_authorizing_test.json") or not path.is_file():
        raise ValueError("non-authorizing product test protocol path is invalid")
    protocol = load_relationship_product_horizon_protocol(path)
    if not protocol.is_v2:
        raise ValueError("non-authorizing product test protocol must use the v2 contract")
    registered = {item.resolve() for item in relationship_product_horizon_protocol_paths()}
    if path in registered:
        raise ValueError("non-authorizing product test protocol must have a distinct identity")
    return path


def _admitted_product_protocol_paths() -> tuple[pathlib.Path, ...]:
    test_path = _non_authorizing_test_protocol_path()
    registered_paths = relationship_product_horizon_protocol_paths()
    if test_path is None:
        return registered_paths
    test_bytes = test_path.read_bytes()
    if any(
        path.is_file() and path.read_bytes() == test_bytes
        for path in registered_paths
    ):
        return registered_paths
    return (*registered_paths, test_path)


def _registered_product_protocol_path_for_bytes(raw_bytes: bytes) -> pathlib.Path:
    matches = tuple(
        path
        for path in _admitted_product_protocol_paths()
        if path.is_file() and path.read_bytes() == raw_bytes
    )
    if len(matches) != 1:
        raise ValueError(
            "campaign protocol differs from every packaged preregistration SSOT"
        )
    return matches[0]


def _registered_product_protocol_for_id(
    protocol_id: object,
) -> RelationshipProductHorizonProtocol:
    expected = _digest(protocol_id, "protocol_id")
    identified_paths: list[pathlib.Path] = []
    for path in _admitted_product_protocol_paths():
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"registered product protocol is invalid JSON: {path}") from exc
        if isinstance(payload, Mapping) and sha256_json(payload) == expected:
            identified_paths.append(path)
    if len(identified_paths) != 1:
        raise ValueError("worker protocol_id is not one registered product preregistration")
    return load_relationship_product_horizon_protocol(identified_paths[0])


def load_relationship_product_horizon_protocol(
    path: pathlib.Path | None = None,
) -> RelationshipProductHorizonProtocol:
    source = pathlib.Path(path or relationship_product_horizon_protocol_path())
    raw_bytes = source.read_bytes()
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid relationship product campaign protocol: {source}") from exc
    schema_version = payload.get("schema_version") if isinstance(payload, Mapping) else None
    if schema_version == RELATIONSHIP_PRODUCT_HORIZON_SCHEMA_VERSION_V2:
        return _load_relationship_product_horizon_protocol_v2(
            payload=payload,
            raw_bytes=raw_bytes,
            source=source,
        )
    if schema_version != RELATIONSHIP_PRODUCT_HORIZON_SCHEMA_VERSION_V1:
        raise ValueError("relationship product campaign schema mismatch")
    _require_exact_keys(
        payload,
        {"schema_version", "owner", "source", "arms", "execution", "analysis", "claim_boundary"},
        "campaign protocol",
    )
    owner = _mapping(payload["owner"], "owner")
    source_spec = _mapping(payload["source"], "source")
    execution = _mapping(payload["execution"], "execution")
    analysis = _mapping(payload["analysis"], "analysis")
    claims = _mapping(payload["claim_boundary"], "claim_boundary")
    arms = payload["arms"]
    if not isinstance(arms, list) or tuple(_mapping(item, "arm").get("arm_id") for item in arms) != _ALL_ARMS_V1:
        raise ValueError("campaign arm order/identity drifted")
    expected_arms = [
        {
            "arm_id": "volvence_full",
            "system_family": "volvence",
            "prior_owner_state": "hydrate_exact_prior",
            "named_reader": "identity",
            "pe_credit": "apply",
            "typed_control": "learned",
        },
        {
            "arm_id": "appendable_frozen_onboarding",
            "system_family": "volvence",
            "prior_owner_state": "restore_same_frozen_post_onboarding_boundary_each_decision",
            "named_reader": "identity",
            "pe_credit": "apply",
            "typed_control": "learned_with_gate_checkpoint_persisted_across_decisions",
        },
        {
            "arm_id": "readable_permuted",
            "system_family": "volvence",
            "prior_owner_state": "hydrate_exact_prior",
            "named_reader": "deterministic_stay_space_permutation_reader_intervention",
            "pe_credit": "apply",
            "typed_control": "learned",
        },
        {
            "arm_id": "credit_withheld",
            "system_family": "volvence",
            "prior_owner_state": "hydrate_exact_prior",
            "named_reader": "identity",
            "pe_credit": "derive_but_do_not_apply",
            "typed_control": "learned",
        },
        {
            "arm_id": "strict_noop",
            "system_family": "volvence",
            "prior_owner_state": "hydrate_exact_prior",
            "named_reader": "identity",
            "pe_credit": "derive_but_do_not_apply_counterfactual_credit",
            "typed_control": "force_neutral_noop_after_reader",
        },
        {
            "arm_id": "native_full_history",
            "system_family": "frozen_qwen_baseline",
            "context_selection": "chronological_public_history",
            "online_parameter_update": False,
        },
        {
            "arm_id": "selective_rag",
            "system_family": "frozen_qwen_baseline",
            "context_selection": "semantic_top_k_public_history",
            "online_parameter_update": False,
        },
    ]
    if arms != expected_arms:
        raise ValueError("campaign arm contracts drifted")
    expected_owner = {
        "module": "lifeform_evolution.relationship_lab_product_horizon",
        "evidence_role": "development_product_horizon_pilot_only",
        "production_runtime_modified": False,
        "formal_evidence_authorized": False,
    }
    if owner != expected_owner:
        raise ValueError("campaign owner/firewall drifted")
    if source_spec.get("schema_version") != "relationship-product-pilot-source.v1":
        raise ValueError("campaign source schema drifted")
    _require_exact_keys(
        source_spec,
        {
            "schema_version",
            "cohort_id",
            "source_protocol_id",
            "source_protocol_raw_sha256",
            "source_implementation_sha256",
            "public_plan_sha256",
            "sealed_evaluator_bundle_sha256",
            "subject_count",
            "onboarding_sessions_per_subject",
            "decision_sessions_per_subject",
            "arm_identity_affects_exogenous_source_or_environment_seed",
        },
        "source",
    )
    if source_spec.get("arm_identity_affects_exogenous_source_or_environment_seed") is not False:
        raise ValueError("campaign arms must share an exogenous world clone")
    expected_claims = {
        "typed_control_effect_may_be_reported": True,
        "residual_steerable": False,
        "user_visible_generation": False,
        "four_able_complete": False,
        "human_product_validation": False,
        "production_active": False,
        "thesis_validated": False,
    }
    if claims != expected_claims:
        raise ValueError("campaign claim boundary drifted")
    if execution.get("volvence_fresh_process_per_logical_session") is not True:
        raise ValueError("campaign must use fresh Volvence session processes")
    if execution.get("validate_existing_requires_model_or_cuda") is not False:
        raise ValueError("offline campaign validation must remain model/GPU-free")
    expected_execution_keys = {
        "volvence_fresh_process_per_logical_session",
        "baseline_single_resident_model_with_stateless_calls",
        "baseline_model_source",
        "baseline_model_id",
        "baseline_model_revision",
        "baseline_model_weights_sha256",
        "baseline_tokenizer_id",
        "baseline_action_schema_sha256",
        "baseline_generation_config_sha256",
        "baseline_arm_prompt_sha256s",
        "baseline_cuda_device",
        "baseline_context_window_tokens",
        "baseline_generation_token_reserve",
        "baseline_generation",
        "semantic_embedder",
        "rag_top_k",
        "output_commit_rule",
        "validate_existing_requires_model_or_cuda",
    }
    if "execution_source_sha256s" in execution:
        expected_execution_keys.add("execution_source_sha256s")
    _require_exact_keys(
        execution,
        expected_execution_keys,
        "execution",
    )
    if execution.get("baseline_single_resident_model_with_stateless_calls") is not True:
        raise ValueError("baseline must use one resident model with stateless calls")
    if execution.get("output_commit_rule") != "create_temporary_root_then_atomic_rename_and_write_manifest_last":
        raise ValueError("campaign output commit rule drifted")
    prompt_sha256s = _mapping(
        execution.get("baseline_arm_prompt_sha256s"),
        "execution.baseline_arm_prompt_sha256s",
    )
    _require_exact_keys(
        prompt_sha256s,
        {"native_full_history", "selective_rag"},
        "execution.baseline_arm_prompt_sha256s",
    )
    execution_source_sha256s: tuple[tuple[str, str], ...] = ()
    if "execution_source_sha256s" in execution:
        raw_execution_sources = _mapping(
            execution.get("execution_source_sha256s"),
            "execution.execution_source_sha256s",
        )
        _require_exact_keys(
            raw_execution_sources,
            set(_EXECUTION_SOURCE_KEYS),
            "execution.execution_source_sha256s",
        )
        execution_source_sha256s = tuple(
            (
                key,
                _digest(
                    raw_execution_sources.get(key),
                    f"execution.execution_source_sha256s.{key}",
                ),
            )
            for key in _EXECUTION_SOURCE_KEYS
        )
    baseline_generation = _mapping(execution["baseline_generation"], "execution.baseline_generation")
    expected_generation = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": 64,
        "do_sample": False,
        "prefill_chunk_size": 2048,
        "generation_use_cache": True,
        "torch_version": "2.12.0+cu126",
        "transformers_version": "5.9.0",
        "seed_namespace": "relationship-product-horizon-qwen-generation-v1",
        "arm_identity_affects_seed": False,
        "seed_derivation": "sha256_u64(namespace,cohort_id,subject_scope,decision_index)",
    }
    if baseline_generation != expected_generation:
        raise ValueError("baseline generation contract drifted")
    semantic = _mapping(execution["semantic_embedder"], "execution.semantic_embedder")
    expected_semantic = {
        "model_source": BGE_M3_MODEL_ID,
        "model_revision": _BGE_M3_REVISION,
        "device": "cuda",
        "stub_fallback_allowed": False,
        "public_ledger_provenance_required": True,
        "public_table_artifact_id": ("5199276e8857aedfe32abceef8b61b82e1621f473abe733494d68afd5ccc348c"),
        "public_table_raw_sha256": ("0369a9068f517303c76a8e5e6b5a990d013e4fc633b9fdf94b6a9f74279a821b"),
        "public_table_record_count": 28,
        "fresh_process_reobservation_required": True,
        "fresh_process_reobservation_comparison": ("exact_table_artifact_id_and_vector_payload"),
    }
    if semantic != expected_semantic:
        raise ValueError("semantic embedder contract drifted")
    _require_exact_keys(
        analysis,
        {
            "primary_window_decision_indices",
            "primary_endpoint",
            "safety_endpoint",
            "secondary_endpoints",
            "analysis_unit",
            "development_directional_effect_floor",
            "minimum_subjects_with_positive_paired_effect",
            "maximum_full_safety_rate_increase_vs_any_comparator",
            "stage_two_admission_requires_all_four_targeted_axis_interventions",
            "stage_two_admission_requires_full_vs_each_strong_baseline",
            "stage_two_admission_requires_safety_noninferiority",
            "no_interim_source_or_threshold_tuning_from_results",
            "single_axis_contrast_claim_authorized",
            "intervention_scope",
        },
        "analysis",
    )
    if analysis.get("stage_two_admission_requires_all_four_targeted_axis_interventions") is not True:
        raise ValueError("stage-two admission must require all targeted interventions")
    if analysis.get("stage_two_admission_requires_full_vs_each_strong_baseline") is not True:
        raise ValueError("stage-two admission must require both strong baselines")
    if analysis.get("stage_two_admission_requires_safety_noninferiority") is not True:
        raise ValueError("stage-two admission must require safety noninferiority")
    if analysis.get("safety_endpoint") != "missed_or_over_directive_typed_outcome_rate":
        raise ValueError("campaign safety endpoint drifted")
    if analysis.get("no_interim_source_or_threshold_tuning_from_results") is not True:
        raise ValueError("campaign must forbid interim tuning from results")
    if analysis.get("single_axis_contrast_claim_authorized") is not False:
        raise ValueError("single-axis causal claims are not authorized")
    expected_interventions = {
        "appendable_frozen_onboarding": (
            "holds the post-onboarding owner boundary fixed while retaining the learned gate checkpoint"
        ),
        "readable_permuted": "replaces the named forecast reader with a deterministic stay-space permutation",
        "credit_withheld": "derives PE credit but withholds its gate update",
        "strict_noop": (
            "closed-loop typed-executor ablation; PE credit is derived but no counterfactual gate update is fabricated"
        ),
    }
    if analysis.get("intervention_scope") != expected_interventions:
        raise ValueError("campaign intervention scope drifted")
    primary = analysis.get("primary_window_decision_indices")
    if not isinstance(primary, list) or len(primary) != 2:
        raise ValueError("campaign primary window must have two endpoints")
    protocol_id = sha256_json(payload)
    return RelationshipProductHorizonProtocol(
        protocol_id=protocol_id,
        raw_sha256=_sha256_bytes(raw_bytes),
        cohort_id=_text(source_spec.get("cohort_id"), "source.cohort_id"),
        source_protocol_id=_digest(source_spec.get("source_protocol_id"), "source.source_protocol_id"),
        source_protocol_raw_sha256=_digest(
            source_spec.get("source_protocol_raw_sha256"),
            "source.source_protocol_raw_sha256",
        ),
        source_implementation_sha256=_digest(
            source_spec.get("source_implementation_sha256"),
            "source.source_implementation_sha256",
        ),
        public_plan_sha256=_digest(source_spec.get("public_plan_sha256"), "source.public_plan_sha256"),
        sealed_evaluator_bundle_sha256=_digest(
            source_spec.get("sealed_evaluator_bundle_sha256"),
            "source.sealed_evaluator_bundle_sha256",
        ),
        subject_count=_integer(source_spec.get("subject_count"), "source.subject_count"),
        onboarding_sessions_per_subject=_integer(
            source_spec.get("onboarding_sessions_per_subject"), "source.onboarding_sessions_per_subject"
        ),
        decision_sessions_per_subject=_integer(
            source_spec.get("decision_sessions_per_subject"), "source.decision_sessions_per_subject"
        ),
        baseline_model_source=_text(execution.get("baseline_model_source"), "execution.baseline_model_source"),
        baseline_model_id=_text(
            execution.get("baseline_model_id"),
            "execution.baseline_model_id",
        ),
        baseline_model_revision=_text(execution.get("baseline_model_revision"), "execution.baseline_model_revision"),
        baseline_model_weights_sha256=_digest(
            execution.get("baseline_model_weights_sha256"),
            "execution.baseline_model_weights_sha256",
        ),
        baseline_tokenizer_id=_text(
            execution.get("baseline_tokenizer_id"),
            "execution.baseline_tokenizer_id",
        ),
        baseline_action_schema_sha256=_digest(
            execution.get("baseline_action_schema_sha256"),
            "execution.baseline_action_schema_sha256",
        ),
        baseline_generation_config_sha256=_digest(
            execution.get("baseline_generation_config_sha256"),
            "execution.baseline_generation_config_sha256",
        ),
        baseline_native_prompt_sha256=_digest(
            prompt_sha256s.get("native_full_history"),
            "execution.baseline_arm_prompt_sha256s.native_full_history",
        ),
        baseline_rag_prompt_sha256=_digest(
            prompt_sha256s.get("selective_rag"),
            "execution.baseline_arm_prompt_sha256s.selective_rag",
        ),
        baseline_cuda_device=_text(execution.get("baseline_cuda_device"), "execution.baseline_cuda_device"),
        context_window_tokens=_integer(
            execution.get("baseline_context_window_tokens"), "execution.baseline_context_window_tokens"
        ),
        generation_token_reserve=_integer(
            execution.get("baseline_generation_token_reserve"), "execution.baseline_generation_token_reserve"
        ),
        generation_seed_namespace=_text(
            baseline_generation.get("seed_namespace"), "execution.baseline_generation.seed_namespace"
        ),
        generation_prefill_chunk_size=_integer(
            baseline_generation.get("prefill_chunk_size"),
            "execution.baseline_generation.prefill_chunk_size",
        ),
        generation_use_cache=_boolean(
            baseline_generation.get("generation_use_cache"),
            "execution.baseline_generation.generation_use_cache",
        ),
        generation_torch_version=_text(
            baseline_generation.get("torch_version"),
            "execution.baseline_generation.torch_version",
        ),
        generation_transformers_version=_text(
            baseline_generation.get("transformers_version"),
            "execution.baseline_generation.transformers_version",
        ),
        semantic_model_source=_text(semantic.get("model_source"), "execution.semantic_embedder.model_source"),
        semantic_model_revision=_text(semantic.get("model_revision"), "execution.semantic_embedder.model_revision"),
        semantic_device=_text(semantic.get("device"), "execution.semantic_embedder.device"),
        semantic_table_artifact_id=_digest(
            semantic.get("public_table_artifact_id"),
            "execution.semantic_embedder.public_table_artifact_id",
        ),
        semantic_table_raw_sha256=_digest(
            semantic.get("public_table_raw_sha256"),
            "execution.semantic_embedder.public_table_raw_sha256",
        ),
        semantic_table_record_count=_integer(
            semantic.get("public_table_record_count"),
            "execution.semantic_embedder.public_table_record_count",
        ),
        rag_top_k=_integer(execution.get("rag_top_k"), "execution.rag_top_k"),
        primary_window=(_integer(primary[0], "primary[0]"), _integer(primary[1], "primary[1]")),
        development_directional_effect_floor=_number(
            analysis.get("development_directional_effect_floor"), "analysis.development_directional_effect_floor"
        ),
        minimum_subjects_with_positive_paired_effect=_integer(
            analysis.get("minimum_subjects_with_positive_paired_effect"),
            "analysis.minimum_subjects_with_positive_paired_effect",
        ),
        maximum_full_safety_rate_increase_vs_any_comparator=_number(
            analysis.get("maximum_full_safety_rate_increase_vs_any_comparator"),
            "analysis.maximum_full_safety_rate_increase_vs_any_comparator",
        ),
        execution_source_sha256s=execution_source_sha256s,
        schema_version=_text(payload.get("schema_version"), "schema_version"),
    )


def _load_relationship_product_horizon_protocol_v2(
    *,
    payload: Mapping[str, object],
    raw_bytes: bytes,
    source: pathlib.Path,
) -> RelationshipProductHorizonProtocol:
    """Load the v2 named-reader/output-contract convergence protocol.

    v2 deliberately keeps the v1 source, endpoints, cohort, horizon and
    thresholds.  Only invalid instrument contracts discovered by the frozen
    v1 result are changed: the Readable contrast removes the named readout
    instead of permuting actions, pre-action receipts retain the complete
    forecast, and the frozen baseline decoder is restricted to the exact
    action-schema surface.
    """

    del source  # retained in the signature for symmetric loader diagnostics
    _require_exact_keys(
        payload,
        {"schema_version", "owner", "source", "arms", "execution", "analysis", "claim_boundary"},
        "campaign protocol v2",
    )
    owner = _mapping(payload["owner"], "owner")
    source_spec = _mapping(payload["source"], "source")
    execution = _mapping(payload["execution"], "execution")
    analysis = _mapping(payload["analysis"], "analysis")
    claims = _mapping(payload["claim_boundary"], "claim_boundary")
    expected_owner = {
        "module": "lifeform_evolution.relationship_lab_product_horizon",
        "evidence_role": "development_product_horizon_pilot_only",
        "production_runtime_modified": False,
        "formal_evidence_authorized": False,
    }
    if owner != expected_owner:
        raise ValueError("campaign v2 owner/firewall drifted")
    expected_source_keys = {
        "schema_version",
        "cohort_id",
        "source_protocol_id",
        "source_protocol_raw_sha256",
        "source_implementation_sha256",
        "public_plan_sha256",
        "sealed_evaluator_bundle_sha256",
        "subject_count",
        "onboarding_sessions_per_subject",
        "decision_sessions_per_subject",
        "arm_identity_affects_exogenous_source_or_environment_seed",
    }
    _require_exact_keys(source_spec, expected_source_keys, "source")
    if (
        source_spec.get("schema_version") != "relationship-product-pilot-source.v1"
        or source_spec.get("arm_identity_affects_exogenous_source_or_environment_seed") is not False
    ):
        raise ValueError("campaign v2 source/world-clone contract drifted")

    arms = payload["arms"]
    if not isinstance(arms, list) or tuple(
        _mapping(item, "arm").get("arm_id") for item in arms
    ) != _ALL_ARMS_V2:
        raise ValueError("campaign v2 arm order/identity drifted")
    expected_arms = [
        {
            "arm_id": "volvence_full",
            "system_family": "volvence",
            "prior_owner_state": "hydrate_exact_prior",
            "named_reader": "prototype_named_condition_readout",
            "pe_credit": "apply",
            "typed_control": "learned",
        },
        {
            "arm_id": "appendable_frozen_onboarding",
            "system_family": "volvence",
            "prior_owner_state": "restore_same_frozen_post_onboarding_boundary_each_decision",
            "named_reader": "prototype_named_condition_readout",
            "pe_credit": "apply",
            "typed_control": "learned_with_gate_checkpoint_persisted_across_decisions",
        },
        {
            "arm_id": "readable_unnamed_legacy",
            "system_family": "volvence",
            "prior_owner_state": "hydrate_exact_prior",
            "named_reader": "legacy_unnamed_semantic_similarity",
            "pe_credit": "apply",
            "typed_control": "learned",
        },
        {
            "arm_id": "credit_withheld",
            "system_family": "volvence",
            "prior_owner_state": "hydrate_exact_prior",
            "named_reader": "prototype_named_condition_readout",
            "pe_credit": "derive_but_do_not_apply",
            "typed_control": "learned",
        },
        {
            "arm_id": "strict_noop",
            "system_family": "volvence",
            "prior_owner_state": "hydrate_exact_prior",
            "named_reader": "prototype_named_condition_readout",
            "pe_credit": "derive_but_do_not_apply_counterfactual_credit",
            "typed_control": "force_neutral_noop_after_reader",
        },
        {
            "arm_id": "native_full_history",
            "system_family": "frozen_qwen_baseline",
            "context_selection": "chronological_public_history",
            "online_parameter_update": False,
        },
        {
            "arm_id": "selective_rag",
            "system_family": "frozen_qwen_baseline",
            "context_selection": "semantic_top_k_public_history",
            "online_parameter_update": False,
        },
    ]
    if arms != expected_arms:
        raise ValueError("campaign v2 arm contracts drifted")

    expected_claims = {
        "typed_control_effect_may_be_reported": True,
        "residual_steerable": False,
        "user_visible_generation": False,
        "four_able_complete": False,
        "human_product_validation": False,
        "production_active": False,
        "thesis_validated": False,
    }
    if claims != expected_claims:
        raise ValueError("campaign v2 claim boundary drifted")
    expected_execution_keys = {
        "volvence_fresh_process_per_logical_session",
        "baseline_single_resident_model_with_stateless_calls",
        "baseline_model_source",
        "baseline_model_id",
        "baseline_model_revision",
        "baseline_model_weights_sha256",
        "baseline_tokenizer_id",
        "baseline_action_schema_sha256",
        "baseline_generation_config_sha256",
        "baseline_arm_prompt_sha256s",
        "baseline_cuda_device",
        "baseline_context_window_tokens",
        "baseline_generation_token_reserve",
        "baseline_generation",
        "semantic_embedder",
        "named_condition_reader",
        "rag_top_k",
        "output_commit_rule",
        "validate_existing_requires_model_or_cuda",
        "local_execution_source_tree",
    }
    _require_exact_keys(execution, expected_execution_keys, "execution")
    if (
        execution.get("volvence_fresh_process_per_logical_session") is not True
        or execution.get("baseline_single_resident_model_with_stateless_calls") is not True
        or execution.get("validate_existing_requires_model_or_cuda") is not False
        or execution.get("output_commit_rule")
        != "create_temporary_root_then_atomic_rename_and_write_manifest_last"
    ):
        raise ValueError("campaign v2 execution boundary drifted")
    prompt_sha256s = _mapping(
        execution.get("baseline_arm_prompt_sha256s"),
        "execution.baseline_arm_prompt_sha256s",
    )
    _require_exact_keys(
        prompt_sha256s,
        {"native_full_history", "selective_rag"},
        "execution.baseline_arm_prompt_sha256s",
    )
    generation = _mapping(execution["baseline_generation"], "execution.baseline_generation")
    exact_action_surface = [
        canonical_json({"action_id": action.value}) for action in RELATIONSHIP_ACTIONS
    ]
    expected_generation = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": 64,
        "do_sample": False,
        "prefill_chunk_size": 2048,
        "generation_use_cache": True,
        "torch_version": "2.12.0+cu126",
        "transformers_version": "5.9.0",
        "seed_namespace": "relationship-product-horizon-qwen-generation-v1",
        "arm_identity_affects_seed": False,
        "seed_derivation": "sha256_u64(namespace,cohort_id,subject_scope,decision_index)",
        "constrained_action_choice": True,
        "constrained_action_surface": exact_action_surface,
    }
    if generation != expected_generation:
        raise ValueError("campaign v2 baseline generation contract drifted")

    semantic = _mapping(execution["semantic_embedder"], "execution.semantic_embedder")
    _require_exact_keys(
        semantic,
        {
            "model_source",
            "model_revision",
            "embedding_weights_sha256",
            "sentence_transformers_version",
            "device",
            "stub_fallback_allowed",
            "public_ledger_provenance_required",
            "public_table_artifact_id",
            "public_table_raw_sha256",
            "public_table_record_count",
            "fresh_process_reobservation_required",
            "fresh_process_reobservation_comparison",
        },
        "execution.semantic_embedder",
    )
    if {
        "model_source": semantic.get("model_source"),
        "model_revision": semantic.get("model_revision"),
        "embedding_weights_sha256": semantic.get("embedding_weights_sha256"),
        "sentence_transformers_version": semantic.get(
            "sentence_transformers_version"
        ),
        "device": semantic.get("device"),
        "stub_fallback_allowed": semantic.get("stub_fallback_allowed"),
        "public_ledger_provenance_required": semantic.get("public_ledger_provenance_required"),
        "public_table_record_count": semantic.get("public_table_record_count"),
        "fresh_process_reobservation_required": semantic.get("fresh_process_reobservation_required"),
        "fresh_process_reobservation_comparison": semantic.get(
            "fresh_process_reobservation_comparison"
        ),
    } != {
        "model_source": BGE_M3_MODEL_ID,
        "model_revision": _BGE_M3_REVISION,
        "embedding_weights_sha256": _BGE_M3_WEIGHTS_SHA256,
        "sentence_transformers_version": BGE_M3_SENTENCE_TRANSFORMERS_VERSION,
        "device": "cuda",
        "stub_fallback_allowed": False,
        "public_ledger_provenance_required": True,
        "public_table_record_count": 30,
        "fresh_process_reobservation_required": True,
        "fresh_process_reobservation_comparison": "exact_table_artifact_id_and_vector_payload",
    }:
        raise ValueError("campaign v2 semantic embedder contract drifted")
    _digest(
        semantic.get("public_table_artifact_id"),
        "execution.semantic_embedder.public_table_artifact_id",
    )
    _digest(
        semantic.get("public_table_raw_sha256"),
        "execution.semantic_embedder.public_table_raw_sha256",
    )

    reader_artifact = relationship_product_condition_reader_artifact()
    expected_reader = {
        "runtime": "PrototypeRelationshipPreferenceForecastRuntime",
        "artifact": {**reader_artifact.to_payload(), "artifact_id": reader_artifact.artifact_id},
        "prior_count": 1.0,
        "evidence_weight": 4.0,
        "public_only": True,
        "evaluator_labels_or_actions_received": False,
    }
    if _mapping(execution["named_condition_reader"], "execution.named_condition_reader") != expected_reader:
        raise ValueError("campaign v2 named condition reader contract drifted")

    execution_source_sha256s: tuple[tuple[str, str], ...] = ()
    local_source_tree = _mapping(
        execution.get("local_execution_source_tree"),
        "execution.local_execution_source_tree",
    )
    _require_exact_keys(
        local_source_tree,
        {
            "schema_version",
            "selector",
            "entrypoints",
            "resource_paths",
            "active_protocol_path",
            "canonicalization",
            "tree_sha256",
            "file_count",
            "canonical_bytes",
        },
        "execution.local_execution_source_tree",
    )
    if (
        local_source_tree.get("schema_version")
        != _LOCAL_EXECUTION_SOURCE_TREE_SCHEMA_VERSION
        or local_source_tree.get("selector")
        != _LOCAL_EXECUTION_SOURCE_TREE_SELECTOR
        or local_source_tree.get("entrypoints")
        != list(_LOCAL_EXECUTION_SOURCE_ENTRYPOINTS)
        or local_source_tree.get("resource_paths")
        != list(_LOCAL_EXECUTION_RESOURCE_PATHS)
        or local_source_tree.get("active_protocol_path")
        != _LOCAL_EXECUTION_ACTIVE_PROTOCOL_PATH
        or local_source_tree.get("canonicalization")
        != _LOCAL_EXECUTION_SOURCE_CANONICALIZATION
    ):
        raise ValueError("campaign v2 local Python source-tree contract drifted")
    local_source_tree_sha256 = _digest(
        local_source_tree.get("tree_sha256"),
        "execution.local_execution_source_tree.tree_sha256",
    )
    local_source_file_count = _integer(
        local_source_tree.get("file_count"),
        "execution.local_execution_source_tree.file_count",
    )
    local_source_canonical_bytes = _integer(
        local_source_tree.get("canonical_bytes"),
        "execution.local_execution_source_tree.canonical_bytes",
    )

    _require_exact_keys(
        analysis,
        {
            "primary_window_decision_indices",
            "primary_endpoint",
            "safety_endpoint",
            "secondary_endpoints",
            "analysis_unit",
            "development_directional_effect_floor",
            "minimum_subjects_with_positive_paired_effect",
            "maximum_full_safety_rate_increase_vs_any_comparator",
            "stage_two_admission_requires_all_targeted_interventions",
            "stage_two_admission_requires_full_vs_each_strong_baseline",
            "stage_two_admission_requires_safety_noninferiority",
            "stage_two_admission_requires_horizon_durability",
            "horizon_segment_windows",
            "minimum_nonnegative_horizon_segments_per_comparator",
            "minimum_positive_horizon_segments_per_comparator",
            "no_interim_source_or_threshold_tuning_from_results",
            "single_axis_contrast_claim_authorized",
            "intervention_scope",
        },
        "analysis",
    )
    if (
        analysis.get("stage_two_admission_requires_all_targeted_interventions")
        is not True
        or analysis.get("stage_two_admission_requires_full_vs_each_strong_baseline") is not True
        or analysis.get("stage_two_admission_requires_safety_noninferiority") is not True
        or analysis.get("stage_two_admission_requires_horizon_durability")
        is not True
        or analysis.get("no_interim_source_or_threshold_tuning_from_results") is not True
        or analysis.get("single_axis_contrast_claim_authorized") is not False
        or analysis.get("safety_endpoint") != "missed_or_over_directive_typed_outcome_rate"
    ):
        raise ValueError("campaign v2 analysis admission contract drifted")
    expected_horizon_windows = {
        name: [bounds[0], bounds[1]]
        for name, bounds in _HORIZON_SEGMENT_WINDOWS
    }
    if (
        analysis.get("horizon_segment_windows") != expected_horizon_windows
        or analysis.get("minimum_nonnegative_horizon_segments_per_comparator")
        != _MINIMUM_NONNEGATIVE_HORIZON_SEGMENTS_PER_COMPARATOR
        or analysis.get("minimum_positive_horizon_segments_per_comparator")
        != _MINIMUM_POSITIVE_HORIZON_SEGMENTS_PER_COMPARATOR
    ):
        raise ValueError("campaign v2 horizon durability contract drifted")
    expected_interventions = {
        "appendable_frozen_onboarding": (
            "holds the post-onboarding owner boundary fixed while retaining the learned gate checkpoint"
        ),
        "readable_unnamed_legacy": (
            "removes the frozen named condition readout while retaining public semantic similarity; "
            "never permutes action identities"
        ),
        "credit_withheld": "derives PE credit but withholds its gate update",
        "strict_noop": (
            "closed-loop typed-executor ablation; PE credit is derived but no counterfactual gate update is fabricated"
        ),
    }
    if analysis.get("intervention_scope") != expected_interventions:
        raise ValueError("campaign v2 intervention scope drifted")
    primary = analysis.get("primary_window_decision_indices")
    if not isinstance(primary, list) or len(primary) != 2:
        raise ValueError("campaign v2 primary window must have two endpoints")

    return RelationshipProductHorizonProtocol(
        protocol_id=sha256_json(payload),
        raw_sha256=_sha256_bytes(raw_bytes),
        cohort_id=_text(source_spec.get("cohort_id"), "source.cohort_id"),
        source_protocol_id=_digest(source_spec.get("source_protocol_id"), "source.source_protocol_id"),
        source_protocol_raw_sha256=_digest(
            source_spec.get("source_protocol_raw_sha256"),
            "source.source_protocol_raw_sha256",
        ),
        source_implementation_sha256=_digest(
            source_spec.get("source_implementation_sha256"),
            "source.source_implementation_sha256",
        ),
        public_plan_sha256=_digest(source_spec.get("public_plan_sha256"), "source.public_plan_sha256"),
        sealed_evaluator_bundle_sha256=_digest(
            source_spec.get("sealed_evaluator_bundle_sha256"),
            "source.sealed_evaluator_bundle_sha256",
        ),
        subject_count=_integer(source_spec.get("subject_count"), "source.subject_count"),
        onboarding_sessions_per_subject=_integer(
            source_spec.get("onboarding_sessions_per_subject"),
            "source.onboarding_sessions_per_subject",
        ),
        decision_sessions_per_subject=_integer(
            source_spec.get("decision_sessions_per_subject"),
            "source.decision_sessions_per_subject",
        ),
        baseline_model_source=_text(execution.get("baseline_model_source"), "execution.baseline_model_source"),
        baseline_model_id=_text(execution.get("baseline_model_id"), "execution.baseline_model_id"),
        baseline_model_revision=_text(
            execution.get("baseline_model_revision"),
            "execution.baseline_model_revision",
        ),
        baseline_model_weights_sha256=_digest(
            execution.get("baseline_model_weights_sha256"),
            "execution.baseline_model_weights_sha256",
        ),
        baseline_tokenizer_id=_text(execution.get("baseline_tokenizer_id"), "execution.baseline_tokenizer_id"),
        baseline_action_schema_sha256=_digest(
            execution.get("baseline_action_schema_sha256"),
            "execution.baseline_action_schema_sha256",
        ),
        baseline_generation_config_sha256=_digest(
            execution.get("baseline_generation_config_sha256"),
            "execution.baseline_generation_config_sha256",
        ),
        baseline_native_prompt_sha256=_digest(
            prompt_sha256s.get("native_full_history"),
            "execution.baseline_arm_prompt_sha256s.native_full_history",
        ),
        baseline_rag_prompt_sha256=_digest(
            prompt_sha256s.get("selective_rag"),
            "execution.baseline_arm_prompt_sha256s.selective_rag",
        ),
        baseline_cuda_device=_text(execution.get("baseline_cuda_device"), "execution.baseline_cuda_device"),
        context_window_tokens=_integer(
            execution.get("baseline_context_window_tokens"),
            "execution.baseline_context_window_tokens",
        ),
        generation_token_reserve=_integer(
            execution.get("baseline_generation_token_reserve"),
            "execution.baseline_generation_token_reserve",
        ),
        generation_seed_namespace=_text(generation.get("seed_namespace"), "baseline_generation.seed_namespace"),
        generation_prefill_chunk_size=_integer(
            generation.get("prefill_chunk_size"),
            "baseline_generation.prefill_chunk_size",
        ),
        generation_use_cache=_boolean(
            generation.get("generation_use_cache"),
            "baseline_generation.generation_use_cache",
        ),
        generation_torch_version=_text(generation.get("torch_version"), "baseline_generation.torch_version"),
        generation_transformers_version=_text(
            generation.get("transformers_version"),
            "baseline_generation.transformers_version",
        ),
        semantic_model_source=_text(semantic.get("model_source"), "semantic.model_source"),
        semantic_model_revision=_text(semantic.get("model_revision"), "semantic.model_revision"),
        semantic_device=_text(semantic.get("device"), "semantic.device"),
        semantic_table_artifact_id=_digest(
            semantic.get("public_table_artifact_id"),
            "semantic.public_table_artifact_id",
        ),
        semantic_table_raw_sha256=_digest(
            semantic.get("public_table_raw_sha256"),
            "semantic.public_table_raw_sha256",
        ),
        semantic_table_record_count=_integer(
            semantic.get("public_table_record_count"),
            "semantic.public_table_record_count",
        ),
        rag_top_k=_integer(execution.get("rag_top_k"), "execution.rag_top_k"),
        primary_window=(
            _integer(primary[0], "primary[0]"),
            _integer(primary[1], "primary[1]"),
        ),
        development_directional_effect_floor=_number(
            analysis.get("development_directional_effect_floor"),
            "analysis.development_directional_effect_floor",
        ),
        minimum_subjects_with_positive_paired_effect=_integer(
            analysis.get("minimum_subjects_with_positive_paired_effect"),
            "analysis.minimum_subjects_with_positive_paired_effect",
        ),
        maximum_full_safety_rate_increase_vs_any_comparator=_number(
            analysis.get("maximum_full_safety_rate_increase_vs_any_comparator"),
            "analysis.maximum_full_safety_rate_increase_vs_any_comparator",
        ),
        execution_source_sha256s=execution_source_sha256s,
        local_execution_source_tree_sha256=local_source_tree_sha256,
        local_execution_source_file_count=local_source_file_count,
        local_execution_source_canonical_bytes=local_source_canonical_bytes,
        semantic_model_weights_sha256=_digest(
            semantic.get("embedding_weights_sha256"),
            "semantic.embedding_weights_sha256",
        ),
        semantic_sentence_transformers_version=_text(
            semantic.get("sentence_transformers_version"),
            "semantic.sentence_transformers_version",
        ),
        condition_reader_artifact_id=reader_artifact.artifact_id,
        baseline_constrained_action_choice=True,
        schema_version=RELATIONSHIP_PRODUCT_HORIZON_SCHEMA_VERSION_V2,
    )


def relationship_product_required_semantic_texts(
    public_view: RelationshipProductPilotPublicView,
    *,
    selection: RelationshipProductCampaignSelection | None = None,
    protocol: RelationshipProductHorizonProtocol | None = None,
) -> tuple[str, ...]:
    """Return every exact public string queried by the frozen named reader."""

    chosen = selection or RelationshipProductCampaignSelection()
    texts: set[str] = set()
    for subject in public_view.subjects[: chosen.subject_count]:
        texts.update(
            session.user_utterance for session in subject.onboarding_sessions[: chosen.onboarding_session_count]
        )
        texts.update(session.current_input for session in subject.decision_sessions[: chosen.decision_session_count])
    if protocol is not None and protocol.is_v2:
        texts.update(item.summary for item in relationship_product_condition_reader_artifact().prototypes)
    return tuple(sorted(texts, key=lambda value: (_sha256_text(value), value)))


def relationship_product_public_embedding_inputs(
    public_view: RelationshipProductPilotPublicView,
    *,
    selection: RelationshipProductCampaignSelection | None = None,
    protocol: RelationshipProductHorizonProtocol | None = None,
) -> tuple[ProductBaselineInput, ...]:
    """Represent every exact forecast-reader query through the public table API.

    Each item has empty history deliberately: the table builder sees only the
    exact public current observation that a fresh Volvence child may query.
    The resident selective-RAG baseline owns embeddings for its growing,
    action-dependent visible history separately.
    """

    return tuple(
        ProductBaselineInput(
            history=(),
            current_observation=ProductCurrentObservation(content=text_value),
        )
        for text_value in relationship_product_required_semantic_texts(
            public_view,
            selection=selection,
            protocol=protocol,
        )
    )


def verify_relationship_product_public_embedding_table(
    *,
    table_path: pathlib.Path,
    output_attestation_path: pathlib.Path,
    protocol_path: pathlib.Path | None = None,
) -> Mapping[str, object]:
    """Fresh-process exact reobservation of the protocol-pinned public table."""

    protocol = load_relationship_product_horizon_protocol(protocol_path)
    environment_contract = _subprocess_environment_contract(protocol)
    if (
        sys.flags.no_user_site != 1
        or {key: os.environ.get(key) for key in environment_contract}
        != environment_contract
    ):
        raise RuntimeError("embedding reobservation requires python -s and the offline environment")
    if protocol.is_v2:
        observed_source_tree, _source_files = _local_execution_source_tree(
            repository_root=pathlib.Path(__file__).resolve().parents[4],
        )
        expected_source_tree = {
            "schema_version": _LOCAL_EXECUTION_SOURCE_TREE_SCHEMA_VERSION,
            "selector": _LOCAL_EXECUTION_SOURCE_TREE_SELECTOR,
            "entrypoints": list(_LOCAL_EXECUTION_SOURCE_ENTRYPOINTS),
            "resource_paths": list(_LOCAL_EXECUTION_RESOURCE_PATHS),
            "active_protocol_path": _LOCAL_EXECUTION_ACTIVE_PROTOCOL_PATH,
            "canonicalization": _LOCAL_EXECUTION_SOURCE_CANONICALIZATION,
            "tree_sha256": protocol.local_execution_source_tree_sha256,
            "file_count": protocol.local_execution_source_file_count,
            "canonical_bytes": protocol.local_execution_source_canonical_bytes,
        }
        if observed_source_tree != expected_source_tree:
            raise ValueError(
                "embedding reobservation local source tree differs from protocol"
            )
    source = load_relationship_product_pilot_source_protocol()
    public = build_relationship_product_pilot_public_view(source)
    table_source = pathlib.Path(table_path).resolve()
    table = load_precomputed_public_embedding_table(table_source)
    _validate_protocol_pinned_embedding_table(
        protocol=protocol,
        table=table,
        table_path=table_source,
    )
    if not protocol.is_v2:
        raise ValueError(
            "legacy v1 embedding reobservation is immutable; validate its stored attestation"
        )
    embedder = bge_m3_public_semantic_embedder(
        device=protocol.semantic_device,
        model_revision=protocol.semantic_model_revision,
        weights_sha256=_digest(
            protocol.semantic_model_weights_sha256,
            "semantic model weights_sha256",
        ),
        sentence_transformers_version=_text(
            protocol.semantic_sentence_transformers_version,
            "semantic sentence-transformers version",
        ),
    )
    recomputed = build_precomputed_public_embedding_table(
        embedder=embedder,
        public_inputs=relationship_product_public_embedding_inputs(
            public,
            protocol=protocol,
        ),
    )
    exact_match = recomputed.to_payload() == table.to_payload()
    if not exact_match:
        raise ValueError("fresh BGE reobservation differs from the protocol-pinned table")
    core = {
        "schema_version": ("relationship-product-public-embedding-reobservation.v1"),
        "protocol_id": protocol.protocol_id,
        "public_plan_sha256": public.public_plan_sha256,
        "table_artifact_id": table.artifact_id,
        "table_raw_sha256": _sha256_file(table_source),
        "table_record_count": len(table.records),
        "model_source": protocol.semantic_model_source,
        "model_revision": protocol.semantic_model_revision,
        "device": protocol.semantic_device,
        "comparison": "exact_table_artifact_id_and_vector_payload",
        "recomputed_table_artifact_id": recomputed.artifact_id,
        "exact_vector_payload_match": exact_match,
        "child_pid": os.getpid(),
        "parent_pid": os.getppid(),
        "python_executable": sys.executable,
        "python_no_user_site": sys.flags.no_user_site == 1,
        "subprocess_environment_contract": dict(environment_contract),
    }
    if protocol.is_v2:
        core["model_weights_sha256"] = protocol.semantic_model_weights_sha256
        core["sentence_transformers_version"] = (
            protocol.semantic_sentence_transformers_version
        )
        core["local_execution_source_tree_sha256"] = (
            protocol.local_execution_source_tree_sha256
        )
    attestation = _with_artifact_id(core)
    _write_json_create_only(
        pathlib.Path(output_attestation_path),
        attestation,
    )
    return attestation


class _PermutedForecastRuntime:
    """Single intervention on the producer proposal, never on raw owner state."""

    runtime_id = "relationship-product-readable-stay-space-permutation.v1"

    def __init__(self, delegate: PreferenceActionForecastRuntime) -> None:
        self._delegate = delegate

    def propose(self, **kwargs: Any) -> PreferenceActionForecastProposal | None:
        proposal = self._delegate.propose(**kwargs)
        if proposal is None:
            return None
        mapping = {
            RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value: (
                RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION.value
            ),
            RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION.value: (
                RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value
            ),
            RelationshipAction.NEUTRAL_NOOP.value: RelationshipAction.NEUTRAL_NOOP.value,
        }
        return replace(
            proposal,
            recommended_action_id=mapping[proposal.recommended_action_id],
            evidence=(*proposal.evidence, f"reader-intervention:{self.runtime_id}"),
        )


class _ResidentBaselineDispatcher:
    """One bounded JSONL client; the child owns one resident Qwen/BGE pair."""

    def __init__(
        self,
        *,
        process: subprocess.Popen[str],
        timeout_seconds: float,
        stderr_lines: list[str],
        startup_attestation: Mapping[str, object],
    ) -> None:
        self._process = process
        self._timeout_seconds = timeout_seconds
        self._stderr_lines = stderr_lines
        self._startup_attestation = startup_attestation
        self._failed = False

    @classmethod
    def start(
        cls,
        *,
        command: Sequence[str],
        timeout_seconds: float,
        execution_source_bundle_artifact_id: str,
        protocol: RelationshipProductHorizonProtocol,
        working_directory: pathlib.Path | None,
    ) -> _ResidentBaselineDispatcher:
        if (
            not isinstance(command, Sequence)
            or isinstance(command, str | bytes)
            or not command
            or any(not isinstance(part, str) or not part for part in command)
        ):
            raise ValueError("baseline dispatcher command must be non-empty argv")
        if timeout_seconds <= 0:
            raise ValueError("baseline dispatcher timeout must be positive")
        if len(command) < 3 or command[1] != "-s":
            raise ValueError("baseline dispatcher must use python -s and a script")
        _require_sha256(
            execution_source_bundle_artifact_id,
            "execution_source_bundle_artifact_id",
        )
        process = subprocess.Popen(
            list(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=_child_environment(),
            cwd=working_directory,
        )
        assert process.stderr is not None
        stderr_lines: list[str] = []

        def drain_stderr() -> None:
            for line in process.stderr:
                stderr_lines.append(line)
                if len(stderr_lines) > 200:
                    del stderr_lines[:100]

        threading.Thread(target=drain_stderr, daemon=True).start()
        python_path = pathlib.Path(command[0]).resolve()
        raw_dispatcher_path = pathlib.Path(command[2])
        dispatcher_path = (
            (pathlib.Path(working_directory) / raw_dispatcher_path).resolve()
            if not raw_dispatcher_path.is_absolute()
            and working_directory is not None
            else raw_dispatcher_path.resolve()
        )
        startup_core: dict[str, object] = {
            "schema_version": (
                "relationship-product-baseline-dispatcher-startup.v2"
                if protocol.is_v2
                else "relationship-product-baseline-dispatcher-startup.v1"
            ),
            "command": list(command),
            "process_pid": process.pid,
            "python_executable_resolved": str(python_path),
            "python_executable_sha256": _sha256_file(python_path),
            "execution_source_bundle_artifact_id": (
                execution_source_bundle_artifact_id
            ),
            "subprocess_environment_contract": dict(
                _subprocess_environment_contract(protocol)
            ),
        }
        if protocol.is_v2:
            startup_core.update(
                {
                    "dispatcher_script_repository_path": (
                        _LOCAL_EXECUTION_SOURCE_ENTRYPOINTS[0]
                    ),
                    "dispatcher_script_raw_sha256": _sha256_file(
                        dispatcher_path
                    ),
                }
            )
        else:
            startup_core.update(
                {
                    "dispatcher_script_resolved": str(dispatcher_path),
                    "dispatcher_script_sha256": _sha256_file(dispatcher_path),
                }
            )
        startup_attestation = _with_artifact_id(startup_core)
        return cls(
            process=process,
            timeout_seconds=timeout_seconds,
            stderr_lines=stderr_lines,
            startup_attestation=startup_attestation,
        )

    @property
    def startup_attestation(self) -> Mapping[str, object]:
        return self._startup_attestation

    def call(
        self,
        request: ProductBaselineDispatcherRequest,
    ) -> ProductBaselineDispatcherReceivedResponse:
        if self._failed:
            raise RuntimeError("resident baseline dispatcher is already failed")
        process = self._process
        if process.poll() is not None:
            raise RuntimeError(
                f"resident baseline dispatcher exited before request; stderr={''.join(self._stderr_lines)[-2000:]}"
            )
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(canonical_json(request.to_payload()) + "\n")
        process.stdin.flush()
        raw_response = _read_text_line(
            process.stdout,
            timeout_seconds=self._timeout_seconds,
            process=process,
        )
        parsed = parse_product_baseline_dispatcher_response_line(raw_response)
        if isinstance(parsed, ProductBaselineDispatcherFatalResponse):
            self._failed = True
            raise RuntimeError(f"resident baseline dispatcher failed: {parsed.error_type}: {parsed.error_message}")
        if parsed.nonce != request.nonce:
            self._failed = True
            raise ValueError("baseline dispatcher response nonce mismatch")
        return parsed

    def close(self) -> None:
        process = self._process
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        try:
            return_code = process.wait(timeout=30)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.wait(timeout=10)
            raise TimeoutError("resident baseline dispatcher did not stop at EOF") from exc
        if return_code != 0 and not self._failed:
            raise RuntimeError(
                f"resident baseline dispatcher exited {return_code}; stderr={''.join(self._stderr_lines)[-2000:]}"
            )


def run_relationship_product_horizon_campaign(
    *,
    output_dir: pathlib.Path,
    public_embedding_table_path: pathlib.Path,
    public_embedding_attestation_path: pathlib.Path | None = None,
    worker_script: pathlib.Path,
    python_executable: str = sys.executable,
    protocol_path: pathlib.Path | None = None,
    baseline_suite: RelationshipProductBaselineSuite | None = None,
    baseline_dispatcher_command: Sequence[str] | None = None,
    selection: RelationshipProductCampaignSelection | None = None,
    allow_test_semantic_backend: bool = False,
    max_workers: int = 4,
    worker_timeout_seconds: float = 60.0,
    baseline_timeout_seconds: float = 600.0,
) -> Mapping[str, object]:
    """Run one create-only typed campaign and publish its manifest last."""

    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("relationship product campaign requires PYTHONNOUSERSITE=1")
    target = pathlib.Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError(f"relationship product output already exists: {target}")
    chosen = selection or RelationshipProductCampaignSelection()
    if not chosen.is_full and not allow_test_semantic_backend:
        raise ValueError("a reduced cohort is test-only and must be explicitly authorized")
    protocol_source = pathlib.Path(
        protocol_path or relationship_product_horizon_protocol_path()
    ).resolve()
    registered_sources = {
        path.resolve()
        for path in _admitted_product_protocol_paths()
        if path.is_file()
    }
    if protocol_source not in registered_sources:
        raise ValueError(
            "product-horizon campaign only admits a packaged registered preregistration"
        )
    if chosen.is_full and sys.flags.no_user_site != 1:
        raise RuntimeError("formal product-horizon campaign must launch the parent with python -s")
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    protocol = load_relationship_product_horizon_protocol(protocol_source)
    test_protocol_path = _non_authorizing_test_protocol_path()
    if test_protocol_path is not None and protocol_source == test_protocol_path:
        if chosen.is_full or not allow_test_semantic_backend:
            raise ValueError(
                "non-authorizing product test protocol requires a reduced fake-backend run"
            )
    if chosen.is_full and not protocol.is_v2 and not protocol.execution_source_sha256s:
        raise ValueError("formal campaign protocol lacks execution source pins")
    expected_campaign_cli = (
        pathlib.Path(__file__).resolve().parents[4] / "scripts" / "run_relationship_lab_product_horizon.py"
    ).resolve()
    if pathlib.Path(worker_script).resolve() != expected_campaign_cli:
        raise ValueError("campaign worker script differs from the packaged CLI")
    if pathlib.Path(python_executable).resolve() != pathlib.Path(sys.executable).resolve():
        raise ValueError("campaign children must use the verified parent Python")
    if baseline_dispatcher_command is not None and tuple(
        baseline_dispatcher_command
    ) != _expected_baseline_dispatcher_command(
        protocol=protocol,
        python_executable=python_executable,
    ):
        raise ValueError("baseline dispatcher command differs from the exact protocol argv")
    if baseline_suite is not None and baseline_dispatcher_command is not None:
        raise ValueError("inject either a baseline suite or a resident dispatcher, not both")
    if baseline_suite is not None:
        _validate_baseline_suite_contract(
            suite=baseline_suite,
            protocol=protocol,
            allow_test_backend=allow_test_semantic_backend,
        )
    source_protocol = load_relationship_product_pilot_source_protocol()
    public_view = build_relationship_product_pilot_public_view(source_protocol)
    evaluator = build_relationship_product_pilot_evaluator_bundle(source_protocol)
    _validate_campaign_source_binding(protocol, source_protocol, public_view, evaluator)

    table = load_precomputed_public_embedding_table(pathlib.Path(public_embedding_table_path))
    semantic_backend = _semantic_backend_label(table.source_embedder_name)
    if semantic_backend == "fake_test_only" and not allow_test_semantic_backend:
        raise ValueError("formal development campaign rejects fake semantic embeddings")
    embedding_attestation: Mapping[str, object] | None = None
    if semantic_backend == "bge_m3_precomputed_public_table":
        _validate_protocol_pinned_embedding_table(
            protocol=protocol,
            table=table,
            table_path=pathlib.Path(public_embedding_table_path),
        )
        if public_embedding_attestation_path is None:
            raise ValueError("formal semantic table requires fresh-process reobservation attestation")
        embedding_attestation = _load_json(pathlib.Path(public_embedding_attestation_path))
        _validate_embedding_reobservation_attestation(
            embedding_attestation,
            protocol=protocol,
            table=table,
            public_plan_sha256=public_view.public_plan_sha256,
        )
    elif public_embedding_attestation_path is not None:
        raise ValueError("fake semantic backend cannot carry formal reobservation")
    embedder = PrecomputedPublicSemanticEmbedder(table)
    for text_value in relationship_product_required_semantic_texts(
        public_view,
        selection=chosen,
        protocol=protocol,
    ):
        embedder.embed(text_value)

    target.parent.mkdir(parents=True, exist_ok=True)
    temp_root = pathlib.Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent)))
    try:
        (temp_root / "source" / "public").mkdir(parents=True)
        (temp_root / "inputs").mkdir(parents=True)
        _write_bytes_create_only(
            temp_root / "protocol.json",
            protocol_source.read_bytes(),
        )
        public_payload = public_view.to_sut_payload()
        _write_json_create_only(temp_root / "source" / "public" / "public_plan.json", public_payload)
        table_target = temp_root / "inputs" / "public_embedding_table.json"
        _write_text_create_only(table_target, table.to_json())
        if embedding_attestation is not None:
            _write_bytes_create_only(
                temp_root / "inputs" / "public_embedding_reobservation.json",
                pathlib.Path(public_embedding_attestation_path).read_bytes(),
            )
        execution_source_bundle = _publish_execution_source_bundle(
            root=temp_root,
            protocol=protocol,
            campaign_cli=pathlib.Path(worker_script).resolve(),
            protocol_source=protocol_source,
        )
        if (
            _validate_execution_source_bundle(
                root=temp_root,
                protocol=protocol,
            )
            != execution_source_bundle
        ):
            raise RuntimeError("published execution source bundle changed during preflight")
        bundled_worker_script = pathlib.Path(worker_script).resolve()
        bundled_baseline_dispatcher_command = baseline_dispatcher_command
        if protocol.is_v2:
            bundled_worker_script = _resolve_relative(
                temp_root,
                (
                    "inputs/execution_sources/tree/scripts/"
                    "run_relationship_lab_product_horizon.py"
                ),
            )
            if baseline_dispatcher_command is not None:
                _resolve_relative(
                    temp_root,
                    (
                        "inputs/execution_sources/tree/scripts/"
                        "run_relationship_lab_product_baseline_dispatcher.py"
                    ),
                )
                bundled_baseline_dispatcher_command = (
                    *baseline_dispatcher_command[:2],
                    _LOCAL_EXECUTION_SOURCE_ENTRYPOINTS[0],
                    *baseline_dispatcher_command[3:],
                )

        subjects = public_view.subjects[: chosen.subject_count]
        typed_tasks = tuple(
            (subject, RelationshipProductArm(arm))
            for subject in subjects
            for arm in protocol.volvence_arms
        )
        chain_args = {
            "root": temp_root,
            "protocol": protocol,
            "evaluator": evaluator,
            "selection": chosen,
            "worker_script": bundled_worker_script,
            "python_executable": python_executable,
            "semantic_table_relpath": _relative_posix(temp_root, table_target),
            "semantic_table_artifact_id": table.artifact_id,
            "semantic_backend": semantic_backend,
            "worker_timeout_seconds": worker_timeout_seconds,
            "execution_source_bundle_artifact_id": _digest(
                execution_source_bundle.get("artifact_id"),
                "execution source bundle artifact_id",
            ),
        }
        with ThreadPoolExecutor(max_workers=min(max_workers, len(typed_tasks))) as executor:
            futures = [
                executor.submit(_run_typed_chain, subject=subject, arm=arm, **chain_args)
                for subject, arm in typed_tasks
            ]
            typed_chains = tuple(future.result() for future in futures)

        baseline_chains: tuple[Mapping[str, object], ...] = ()
        dispatcher: _ResidentBaselineDispatcher | None = None
        try:
            if bundled_baseline_dispatcher_command is not None:
                dispatcher = _ResidentBaselineDispatcher.start(
                    command=bundled_baseline_dispatcher_command,
                    timeout_seconds=baseline_timeout_seconds,
                    execution_source_bundle_artifact_id=_digest(
                        execution_source_bundle.get("artifact_id"),
                        "execution source bundle artifact_id",
                    ),
                    protocol=protocol,
                    working_directory=(
                        temp_root / "inputs" / "execution_sources" / "tree"
                        if protocol.is_v2
                        else None
                    ),
                )
            if baseline_suite is not None or dispatcher is not None:
                baseline_chains = tuple(
                    _run_baseline_chain(
                        root=temp_root,
                        protocol=protocol,
                        evaluator=evaluator,
                        subject=subject,
                        arm=RelationshipProductArm(arm),
                        selection=chosen,
                        suite=baseline_suite,
                        dispatcher=dispatcher,
                        public_plan_artifact_id=public_view.public_plan_sha256,
                    )
                    for subject in subjects
                    for arm in _BASELINE_ARMS
                )
        finally:
            if dispatcher is not None:
                dispatcher.close()

        if protocol.is_v2:
            final_source_tree, _source_files = _local_execution_source_tree(
                repository_root=pathlib.Path(__file__).resolve().parents[4],
            )
            expected_source_tree = {
                "schema_version": _LOCAL_EXECUTION_SOURCE_TREE_SCHEMA_VERSION,
                "selector": _LOCAL_EXECUTION_SOURCE_TREE_SELECTOR,
                "entrypoints": list(_LOCAL_EXECUTION_SOURCE_ENTRYPOINTS),
                "resource_paths": list(_LOCAL_EXECUTION_RESOURCE_PATHS),
                "active_protocol_path": _LOCAL_EXECUTION_ACTIVE_PROTOCOL_PATH,
                "canonicalization": _LOCAL_EXECUTION_SOURCE_CANONICALIZATION,
                "tree_sha256": protocol.local_execution_source_tree_sha256,
                "file_count": protocol.local_execution_source_file_count,
                "canonical_bytes": protocol.local_execution_source_canonical_bytes,
            }
            if final_source_tree != expected_source_tree:
                raise RuntimeError("local execution source tree changed during campaign")

        # No child/model action remains live beyond this point.  Sealed source
        # and evaluator-derived sidecars are published only after every action
        # has been frozen in parent memory.
        (temp_root / "source" / "sealed").mkdir(parents=True)
        _write_json_create_only(
            temp_root / "source" / "sealed" / "evaluator_bundle.json",
            _evaluator_payload(evaluator),
        )
        typed_chains = tuple(
            _publish_completed_chain(root=temp_root, evaluator=evaluator, chain=chain) for chain in typed_chains
        )
        baseline_chains = tuple(
            _publish_completed_chain(root=temp_root, evaluator=evaluator, chain=chain) for chain in baseline_chains
        )

        report = _build_report(
            root=temp_root,
            protocol=protocol,
            source_protocol_id=source_protocol.protocol_sha256,
            public_plan_sha256=public_view.public_plan_sha256,
            sealed_bundle_sha256=evaluator.sealed_bundle_sha256,
            embedding_table_artifact_id=table.artifact_id,
            semantic_backend=semantic_backend,
            embedding_table_fresh_process_reobserved=(embedding_attestation is not None),
            execution_source_bundle_artifact_id=_digest(
                execution_source_bundle.get("artifact_id"),
                "execution source bundle artifact_id",
            ),
            selection=chosen,
            typed_chains=typed_chains,
            baseline_chains=baseline_chains,
        )
        _write_json_create_only(temp_root / "report.json", report)
        files = _manifest_file_entries(temp_root)
        manifest_core = {
            "schema_version": RELATIONSHIP_PRODUCT_MANIFEST_SCHEMA_VERSION,
            "protocol_id": protocol.protocol_id,
            "report_artifact_id": report["artifact_id"],
            "file_count": len(files),
            "files": files,
            "manifest_written_last": True,
        }
        manifest = _with_artifact_id(manifest_core)
        os.replace(temp_root, target)
        _write_json_create_only(target / "manifest.json", manifest)
        validate_relationship_product_horizon_campaign(
            output_dir=target,
            expected_protocol_id=protocol.protocol_id,
        )
        return report
    except BaseException:
        if temp_root.exists():
            shutil.rmtree(temp_root)
        raise


def _run_typed_chain(
    *,
    root: pathlib.Path,
    protocol: RelationshipProductHorizonProtocol,
    evaluator: RelationshipProductPilotEvaluatorBundle,
    subject: ProductPilotPublicSubject,
    arm: RelationshipProductArm,
    selection: RelationshipProductCampaignSelection,
    worker_script: pathlib.Path,
    python_executable: str,
    semantic_table_relpath: str,
    semantic_table_artifact_id: str,
    semantic_backend: str,
    worker_timeout_seconds: float,
    execution_source_bundle_artifact_id: str,
) -> Mapping[str, object]:
    chain_root = root / "chains" / subject.subject_scope / arm.value
    chain_root.mkdir(parents=True)
    persistent_state = chain_root / "state"
    persistent_state.mkdir()
    persistent_owner_state = persistent_state / "owner"
    persistent_gate_state = persistent_state / "gate"
    persistent_owner_state.mkdir()
    persistent_gate_state.mkdir()
    onboarding_receipts: list[Mapping[str, object]] = []
    for session in subject.onboarding_sessions[: selection.onboarding_session_count]:
        receipt = _launch_onboarding_worker(
            root=root,
            chain_root=chain_root,
            state_root=persistent_owner_state,
            protocol=protocol,
            subject=subject,
            arm=arm,
            session=session,
            worker_script=worker_script,
            python_executable=python_executable,
            timeout_seconds=worker_timeout_seconds,
            execution_source_bundle_artifact_id=(
                execution_source_bundle_artifact_id
            ),
        )
        onboarding_receipts.append(receipt)
    onboarding_boundary = _state_directory_sha256(persistent_owner_state)

    decision_records: list[Mapping[str, object]] = []
    for session in subject.decision_sessions[: selection.decision_session_count]:
        if arm is RelationshipProductArm.APPENDABLE_FROZEN_ONBOARDING:
            state_root = chain_root / "frozen_owner_sessions" / f"decision-{session.decision_index:02d}"
            shutil.copytree(persistent_owner_state, state_root)
        else:
            state_root = persistent_owner_state
        record = _launch_decision_worker(
            root=root,
            chain_root=chain_root,
            state_root=state_root,
            gate_state_root=persistent_gate_state,
            protocol=protocol,
            evaluator=evaluator,
            subject=subject,
            arm=arm,
            session=session,
            worker_script=worker_script,
            python_executable=python_executable,
            semantic_table_relpath=semantic_table_relpath,
            semantic_table_artifact_id=semantic_table_artifact_id,
            semantic_backend=semantic_backend,
            timeout_seconds=worker_timeout_seconds,
            execution_source_bundle_artifact_id=(
                execution_source_bundle_artifact_id
            ),
        )
        decision_records.append(record)
    core = {
        "schema_version": "relationship-product-typed-chain.v1",
        "subject_scope": subject.subject_scope,
        "world_clone_id": subject.world_clone_id,
        "arm_id": arm.value,
        "onboarding_boundary_sha256": onboarding_boundary,
        "appendable_reset_basis": (
            "same_frozen_post_onboarding_boundary_each_decision"
            if arm is RelationshipProductArm.APPENDABLE_FROZEN_ONBOARDING
            else "hydrate_exact_prior_decision_boundary"
        ),
        "onboarding_receipts": onboarding_receipts,
        "decisions": decision_records,
    }
    return core


def _launch_onboarding_worker(
    *,
    root: pathlib.Path,
    chain_root: pathlib.Path,
    state_root: pathlib.Path,
    protocol: RelationshipProductHorizonProtocol,
    subject: ProductPilotPublicSubject,
    arm: RelationshipProductArm,
    session: ProductPilotPublicOnboardingSession,
    worker_script: pathlib.Path,
    python_executable: str,
    timeout_seconds: float,
    execution_source_bundle_artifact_id: str,
) -> Mapping[str, object]:
    request_dir = chain_root / "requests"
    receipt_dir = chain_root / "receipts"
    request_dir.mkdir(exist_ok=True)
    receipt_dir.mkdir(exist_ok=True)
    request_path = request_dir / f"onboarding-{session.session_index:02d}.json"
    receipt_path = receipt_dir / f"onboarding-{session.session_index:02d}.json"
    nonce = uuid4().hex
    core = {
        "schema_version": RELATIONSHIP_PRODUCT_WORKER_REQUEST_SCHEMA_VERSION,
        "operation": "onboarding",
        "protocol_id": protocol.protocol_id,
        "arm_id": arm.value,
        "subject_scope": subject.subject_scope,
        "world_clone_id": subject.world_clone_id,
        "state_root": _relative_posix(root, state_root),
        "session": session.to_sut_payload(),
        "subprocess_environment_contract": dict(
            _subprocess_environment_contract(protocol)
        ),
        "invocation_nonce": nonce,
        "parent_pid": os.getpid(),
    }
    if protocol.is_v2:
        core["execution_source_bundle_artifact_id"] = _digest(
            execution_source_bundle_artifact_id,
            "execution_source_bundle_artifact_id",
        )
        core["local_execution_source_tree_sha256"] = _digest(
            protocol.local_execution_source_tree_sha256,
            "local_execution_source_tree_sha256",
        )
    request = _with_artifact_id(core)
    _assert_truth_firewall(request)
    _write_json_create_only(request_path, request)
    command = [
        python_executable,
        "-s",
        str(worker_script),
        "worker-onboarding",
        "--request",
        str(request_path),
        "--receipt",
        str(receipt_path),
        "--run-root",
        str(root),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout_seconds,
        check=False,
        env=_child_environment(),
        cwd=(
            root / "inputs" / "execution_sources" / "tree"
            if protocol.is_v2
            else None
        ),
    )
    if completed.returncode != 0:
        raise RuntimeError(f"onboarding child failed ({completed.returncode}): {completed.stderr[-2000:]}")
    receipt = _load_json(receipt_path)
    _validate_onboarding_receipt(receipt)
    if (
        receipt["request_artifact_id"] != request["artifact_id"]
        or receipt["invocation_nonce"] != request["invocation_nonce"]
        or receipt["parent_pid"] != request["parent_pid"]
    ):
        raise ValueError("onboarding request/receipt lineage mismatch")
    return {
        "request_path": _relative_posix(root, request_path),
        "request_sha256": _sha256_file(request_path),
        "request_artifact_id": request["artifact_id"],
        "receipt_path": _relative_posix(root, receipt_path),
        "receipt_sha256": _sha256_file(receipt_path),
        "receipt_artifact_id": receipt["artifact_id"],
        "child_pid": receipt["child_pid"],
        "launch_identity_sha256": receipt["launch_identity_sha256"],
        "owner_loaded": receipt["owner_loaded"],
        "owner_snapshot_sha256": receipt["owner_snapshot_sha256"],
    }


def _launch_decision_worker(
    *,
    root: pathlib.Path,
    chain_root: pathlib.Path,
    state_root: pathlib.Path,
    gate_state_root: pathlib.Path,
    protocol: RelationshipProductHorizonProtocol,
    evaluator: RelationshipProductPilotEvaluatorBundle,
    subject: ProductPilotPublicSubject,
    arm: RelationshipProductArm,
    session: ProductPilotPublicDecisionSession,
    worker_script: pathlib.Path,
    python_executable: str,
    semantic_table_relpath: str,
    semantic_table_artifact_id: str,
    semantic_backend: str,
    timeout_seconds: float,
    execution_source_bundle_artifact_id: str,
) -> Mapping[str, object]:
    request_dir = chain_root / "requests"
    receipt_dir = chain_root / "receipts"
    request_dir.mkdir(exist_ok=True)
    receipt_dir.mkdir(exist_ok=True)
    index = session.decision_index
    request_path = request_dir / f"decision-{index:02d}.json"
    pre_path = receipt_dir / f"decision-{index:02d}.preaction.json"
    post_path = receipt_dir / f"decision-{index:02d}.postaction.json"
    gate_mode = (
        RelationshipActionGateMode.NOOP
        if arm is RelationshipProductArm.STRICT_NOOP
        else RelationshipActionGateMode.LEARNED
    )
    if protocol.is_v2:
        named_reader = (
            "legacy_unnamed_semantic_similarity"
            if arm is RelationshipProductArm.READABLE_UNNAMED_LEGACY
            else "prototype_named_condition_readout"
        )
    else:
        named_reader = (
            "permuted_stay_space"
            if arm is RelationshipProductArm.READABLE_PERMUTED
            else "identity"
        )
    apply_credit = arm not in {RelationshipProductArm.CREDIT_WITHHELD, RelationshipProductArm.STRICT_NOOP}
    nonce = uuid4().hex
    request_core: dict[str, object] = {
            "schema_version": RELATIONSHIP_PRODUCT_WORKER_REQUEST_SCHEMA_VERSION,
            "operation": "decision_handshake",
            "protocol_id": protocol.protocol_id,
            "arm_id": arm.value,
            "subject_scope": subject.subject_scope,
            "world_clone_id": subject.world_clone_id,
            "state_root": _relative_posix(root, state_root),
            "gate_state_root": _relative_posix(root, gate_state_root),
            "session": session.to_sut_payload(),
            "subprocess_environment_contract": dict(
                _subprocess_environment_contract(protocol)
            ),
            "semantic_table_path": semantic_table_relpath,
            "semantic_table_artifact_id": semantic_table_artifact_id,
            "semantic_backend": semantic_backend,
            "named_reader": named_reader,
            "gate_mode": gate_mode.value,
            "apply_credit_to_gate": apply_credit,
            "authorization_id": f"relationship-product-horizon:{protocol.protocol_id}",
            "invocation_nonce": nonce,
            "parent_pid": os.getpid(),
    }
    if protocol.is_v2:
        request_core["execution_source_bundle_artifact_id"] = _digest(
            execution_source_bundle_artifact_id,
            "execution_source_bundle_artifact_id",
        )
        request_core["local_execution_source_tree_sha256"] = _digest(
            protocol.local_execution_source_tree_sha256,
            "local_execution_source_tree_sha256",
        )
    request = _with_artifact_id(request_core)
    _assert_truth_firewall(request)
    _write_json_create_only(request_path, request)
    command = [
        python_executable,
        "-s",
        str(worker_script),
        "worker-decision",
        "--request",
        str(request_path),
        "--preaction-receipt",
        str(pre_path),
        "--postaction-receipt",
        str(post_path),
        "--run-root",
        str(root),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=_child_environment(),
        cwd=(
            root / "inputs" / "execution_sources" / "tree"
            if protocol.is_v2
            else None
        ),
    )
    assert process.stdout is not None and process.stdin is not None and process.stderr is not None
    try:
        pre_ack = _read_json_line(process.stdout, timeout_seconds=timeout_seconds, process=process)
        _require_exact_keys(
            pre_ack,
            {
                "phase",
                "request_artifact_id",
                "preaction_artifact_id",
                "child_pid",
            },
            "decision preaction ack",
        )
        if (
            pre_ack.get("phase") != "preaction_fsynced"
            or pre_ack.get("request_artifact_id") != request["artifact_id"]
            or _integer(pre_ack.get("child_pid"), "preaction ack child_pid") != process.pid
        ):
            raise RuntimeError(f"decision child emitted invalid preaction ack: {pre_ack!r}")
        _require_sha256(
            pre_ack.get("preaction_artifact_id"),
            "preaction ack artifact_id",
        )
        pre_receipt = _load_json(pre_path)
        _validate_preaction_receipt(pre_receipt)
        if (
            pre_receipt.get("request_artifact_id") != request["artifact_id"]
            or pre_receipt.get("invocation_nonce") != request["invocation_nonce"]
            or pre_receipt.get("parent_pid") != request["parent_pid"]
            or pre_receipt.get("artifact_id") != pre_ack.get("preaction_artifact_id")
            or pre_receipt.get("child_pid") != pre_ack.get("child_pid")
            or pre_receipt.get("child_pid") != process.pid
        ):
            raise ValueError("preaction request/receipt lineage mismatch")
        evaluator_session = evaluator.session(session.session_id)
        environment = build_relationship_product_pilot_environment(evaluator, subject_id=evaluator_session.subject_id)
        action = RelationshipAction(_text(pre_receipt.get("selected_action_id"), "selected_action_id"))
        outcome = environment.settle(
            scene_id=evaluator_session.scene_id,
            decision_id=session.decision_id,
            action=action,
            seed=evaluator_session.environment_seed,
        )
        settlement = _settlement_payload(
            session=session,
            subject_scope=subject.subject_scope,
            pre_receipt=pre_receipt,
            outcome=outcome,
            apply_credit_to_gate=apply_credit,
        )
        _assert_settlement_firewall(settlement)
        process.stdin.write(canonical_json(settlement) + "\n")
        process.stdin.flush()
        process.stdin.close()
        process.stdin = None
        post_ack = _read_json_line(process.stdout, timeout_seconds=timeout_seconds, process=process)
        _require_exact_keys(
            post_ack,
            {
                "phase",
                "request_artifact_id",
                "postaction_artifact_id",
                "child_pid",
            },
            "decision settlement ack",
        )
        if (
            post_ack.get("phase") != "settlement_fsynced"
            or post_ack.get("request_artifact_id") != request["artifact_id"]
            or _integer(post_ack.get("child_pid"), "postaction ack child_pid") != process.pid
        ):
            raise RuntimeError(f"decision child emitted invalid settlement ack: {post_ack!r}")
        _require_sha256(
            post_ack.get("postaction_artifact_id"),
            "postaction ack artifact_id",
        )
        return_code = process.wait(timeout=timeout_seconds)
        stderr = process.stderr.read()
        remaining_stdout = process.stdout.read()
        if return_code != 0 or remaining_stdout.strip():
            raise RuntimeError(
                f"decision child failed ({return_code}); stdout={remaining_stdout!r}; stderr={stderr[-2000:]}"
            )
    except BaseException:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        raise
    post_receipt = _load_json(post_path)
    _validate_postaction_receipt(post_receipt)
    if protocol.is_v2:
        _validate_v2_worker_source_lineage(
            _mapping(
                post_receipt.get("execution_source_lineage"),
                "postaction execution source lineage",
            ),
            root=root,
            protocol=protocol,
            execution_source_bundle_artifact_id=execution_source_bundle_artifact_id,
        )
    if post_receipt["preaction_artifact_id"] != pre_receipt["artifact_id"]:
        raise ValueError("postaction receipt is not bound to the flushed preaction")
    if (
        post_receipt.get("request_artifact_id") != request["artifact_id"]
        or post_receipt.get("child_pid") != pre_receipt["child_pid"]
        or post_receipt.get("artifact_id") != post_ack.get("postaction_artifact_id")
        or post_receipt.get("child_pid") != post_ack.get("child_pid")
        or post_receipt.get("child_pid") != process.pid
    ):
        raise ValueError("postaction request/process lineage mismatch")
    return {
        "schema_version": "relationship-product-decision-record.v1",
        "decision_index": index,
        "session_id": session.session_id,
        "decision_id": session.decision_id,
        "request_path": _relative_posix(root, request_path),
        "request_sha256": _sha256_file(request_path),
        "request_artifact_id": request["artifact_id"],
        "preaction_receipt_path": _relative_posix(root, pre_path),
        "preaction_receipt_sha256": _sha256_file(pre_path),
        "preaction_artifact_id": pre_receipt["artifact_id"],
        "postaction_receipt_path": _relative_posix(root, post_path),
        "postaction_receipt_sha256": _sha256_file(post_path),
        "postaction_artifact_id": post_receipt["artifact_id"],
        "child_pid": pre_receipt["child_pid"],
        "launch_identity_sha256": pre_receipt["launch_identity_sha256"],
        "handshake_order": [
            "preaction_fsynced",
            "parent_environment_settled",
            "typed_settlement_sent",
            "settlement_fsynced",
        ],
        "selected_action_id": action.value,
        "recommended_action_id": pre_receipt["recommended_action_id"],
        "typed_outcome_id": outcome.typed_outcome.value,
        "rendered_user_reaction": outcome.rendered_user_reaction,
        "environment_evidence_ref": outcome.environment_evidence_ref,
        "positive_outcome": outcome.typed_outcome.value in _POSITIVE_OUTCOMES,
        "preferred_action_match": action.value == evaluator_session.preferred_action_id,
        "owner_loaded": pre_receipt["owner_loaded"],
        "pre_owner_snapshot_sha256": pre_receipt["pre_owner_snapshot_sha256"],
        "post_owner_snapshot_sha256": post_receipt["post_owner_snapshot_sha256"],
        "gate_update_count_before": pre_receipt["gate_update_count_before"],
        "gate_update_count_after": post_receipt["gate_update_count_after"],
        "credit_value_hex": post_receipt["credit_value_hex"],
        "credit_applied_to_gate": post_receipt["credit_applied_to_gate"],
        "world_clone_id": subject.world_clone_id,
    }


def _run_baseline_chain(
    *,
    root: pathlib.Path,
    protocol: RelationshipProductHorizonProtocol,
    evaluator: RelationshipProductPilotEvaluatorBundle,
    subject: ProductPilotPublicSubject,
    arm: RelationshipProductArm,
    selection: RelationshipProductCampaignSelection,
    suite: RelationshipProductBaselineSuite | None,
    dispatcher: _ResidentBaselineDispatcher | None,
    public_plan_artifact_id: str,
) -> Mapping[str, object]:
    chain_root = root / "chains" / subject.subject_scope / arm.value
    chain_root.mkdir(parents=True)
    ledger_root = chain_root / "public_ledger"
    ledger_root.mkdir()
    history = _initial_baseline_history(subject, selection.onboarding_session_count)
    prior_source_session_ids = [
        session.session_id for session in subject.onboarding_sessions[: selection.onboarding_session_count]
    ]
    records: list[Mapping[str, object]] = []
    for session in subject.decision_sessions[: selection.decision_session_count]:
        current = ProductCurrentObservation(content=f"{session.public_context_chunk}\n\n{session.current_input}")
        public_input = ProductBaselineInput(history=tuple(history), current_observation=current)
        ledger = _public_baseline_ledger(
            subject=subject,
            session=session,
            public_input=public_input,
            public_plan_artifact_id=public_plan_artifact_id,
            ordered_source_session_ids=tuple((*prior_source_session_ids, session.session_id)),
        )
        ledger_path = ledger_root / f"decision-{session.decision_index:02d}.json"
        _write_json_create_only(ledger_path, ledger)
        seed = _sha256_u64(
            protocol.generation_seed_namespace,
            protocol.cohort_id,
            subject.subject_scope,
            str(session.decision_index),
        )
        baseline_arm = {
            RelationshipProductArm.NATIVE_FULL_HISTORY: ProductBaselineArm.NATIVE_CHRONOLOGICAL_FULL_HISTORY,
            RelationshipProductArm.SELECTIVE_RAG: ProductBaselineArm.SELECTIVE_SEMANTIC_RAG,
        }[arm]
        top_k = None
        if arm is RelationshipProductArm.SELECTIVE_RAG:
            top_k = protocol.rag_top_k
        if dispatcher is not None:
            history_entries = tuple(
                _mapping(item, "public ledger history entry")
                for item in _list(
                    ledger.get("history_entries"),
                    "public ledger history entries",
                )
            )
            current_entry = _mapping(
                ledger.get("current_entry"),
                "public ledger current entry",
            )
            dispatcher_request = ProductBaselineDispatcherRequest(
                nonce=uuid4().hex,
                arm=baseline_arm,
                public_plan_artifact_id=public_plan_artifact_id,
                subject_scope=subject.subject_scope,
                decision_boundary=ProductBaselineDecisionBoundary(
                    current_session_id=session.session_id,
                    decision_id=session.decision_id,
                    decision_index=session.decision_index,
                ),
                ordered_source_session_ids=tuple((*prior_source_session_ids, session.session_id)),
                ordered_source_block_artifact_ids=tuple(block.artifact_id for block in public_input.history),
                public_ledger_artifact_id=_digest(
                    ledger.get("artifact_id"),
                    "public ledger artifact_id",
                ),
                public_input=public_input,
                history_block_lineage=tuple(
                    ProductBaselineHistoryBlockLineage(
                        ordinal=block.ordinal,
                        block_artifact_id=block.artifact_id,
                        public_ledger_entry_artifact_id=_digest(
                            entry.get("artifact_id"),
                            "public ledger entry artifact_id",
                        ),
                    )
                    for block, entry in zip(
                        public_input.history,
                        history_entries,
                        strict=True,
                    )
                ),
                current_observation_lineage=ProductBaselineCurrentObservationLineage(
                    observation_artifact_id=public_input.current_observation.artifact_id,
                    public_ledger_entry_artifact_id=_digest(
                        current_entry.get("artifact_id"),
                        "public ledger current entry artifact_id",
                    ),
                ),
                seed=seed,
                top_k=top_k,
            )
            response = dispatcher.call(dispatcher_request)
            dispatcher_request_payload: Mapping[str, object] | None = dispatcher_request.to_payload()
            dispatcher_response_payload: Mapping[str, object] | None = _mapping(
                json.loads(response.canonical_response_json),
                "baseline dispatcher response",
            )
            result_payload = response.result_payload
            selected = response.action_id or RelationshipAction.NEUTRAL_NOOP
            valid_completion = response.valid
            baseline_execution_backend = "resident_jsonl_dispatcher"
        else:
            assert suite is not None
            dispatcher_request_payload = None
            dispatcher_response_payload = None
            if baseline_arm is ProductBaselineArm.NATIVE_CHRONOLOGICAL_FULL_HISTORY:
                result = suite.run_native_chronological_full_history(
                    public_input=public_input,
                    seed=seed,
                )
            else:
                assert top_k is not None
                result = suite.run_selective_semantic_rag(
                    public_input=public_input,
                    seed=seed,
                    top_k=top_k,
                )
            result_payload = result.to_payload()
            selected = result.action_completion.chosen_action_id or RelationshipAction.NEUTRAL_NOOP
            valid_completion = result.action_completion.valid
            baseline_execution_backend = "injected_resident_suite"
        if result_payload.get("input_artifact_id") != public_input.artifact_id:
            raise ValueError("baseline result drifted from the published public ledger input")
        evaluator_session = evaluator.session(session.session_id)
        environment = build_relationship_product_pilot_environment(evaluator, subject_id=evaluator_session.subject_id)
        outcome = environment.settle(
            scene_id=evaluator_session.scene_id,
            decision_id=session.decision_id,
            action=selected,
            seed=evaluator_session.environment_seed,
        )
        record = {
            "schema_version": "relationship-product-baseline-decision-record.v1",
            "decision_index": session.decision_index,
            "session_id": session.session_id,
            "decision_id": session.decision_id,
            "public_ledger_path": _relative_posix(root, ledger_path),
            "public_ledger_sha256": _sha256_file(ledger_path),
            "public_ledger_artifact_id": ledger["artifact_id"],
            "public_input_artifact_id": public_input.artifact_id,
            "history_block_artifact_ids": [block.artifact_id for block in public_input.history],
            "current_observation_artifact_id": public_input.current_observation.artifact_id,
            "baseline_execution_backend": baseline_execution_backend,
            "baseline_dispatcher_request": dispatcher_request_payload,
            "baseline_dispatcher_response": dispatcher_response_payload,
            "baseline_result": result_payload,
            "selected_action_id": selected.value,
            "typed_outcome_id": outcome.typed_outcome.value,
            "rendered_user_reaction": outcome.rendered_user_reaction,
            "environment_evidence_ref": outcome.environment_evidence_ref,
            "positive_outcome": outcome.typed_outcome.value in _POSITIVE_OUTCOMES,
            "preferred_action_match": selected.value == evaluator_session.preferred_action_id,
            "valid_completion": valid_completion,
            "invalid_completion_mapped_to": (None if valid_completion else RelationshipAction.NEUTRAL_NOOP.value),
            "world_clone_id": subject.world_clone_id,
        }
        records.append(record)
        history.extend(
            _decision_public_history_blocks(
                history, session, selected, outcome.typed_outcome.value, outcome.rendered_user_reaction
            )
        )
        prior_source_session_ids.append(session.session_id)
    return {
        "schema_version": "relationship-product-baseline-chain.v1",
        "subject_scope": subject.subject_scope,
        "world_clone_id": subject.world_clone_id,
        "arm_id": arm.value,
        "dispatcher_startup_attestation": (dispatcher.startup_attestation if dispatcher is not None else None),
        "decisions": records,
    }


def _publish_completed_chain(
    *,
    root: pathlib.Path,
    evaluator: RelationshipProductPilotEvaluatorBundle,
    chain: Mapping[str, object],
) -> Mapping[str, object]:
    """Publish evaluator sidecars only after every SUT/model action has ended."""

    subject_scope = _text(chain.get("subject_scope"), "chain.subject_scope")
    arm_id = _text(chain.get("arm_id"), "chain.arm_id")
    chain_root = root / "chains" / subject_scope / arm_id
    sealed_root = chain_root / "sealed"
    sealed_root.mkdir()
    enriched_decisions: list[Mapping[str, object]] = []
    for raw_decision in _list(chain.get("decisions"), "chain.decisions"):
        decision = _mapping(raw_decision, "chain decision")
        session_id = _text(decision.get("session_id"), "decision.session_id")
        evaluator_session = evaluator.session(session_id)
        action = RelationshipAction(_text(decision.get("selected_action_id"), "decision.selected_action_id"))
        environment = build_relationship_product_pilot_environment(
            evaluator,
            subject_id=evaluator_session.subject_id,
        )
        outcome = environment.settle(
            scene_id=evaluator_session.scene_id,
            decision_id=evaluator_session.decision_id,
            action=action,
            seed=evaluator_session.environment_seed,
        )
        if (
            decision.get("typed_outcome_id") != outcome.typed_outcome.value
            or decision.get("rendered_user_reaction") != outcome.rendered_user_reaction
            or decision.get("environment_evidence_ref") != outcome.environment_evidence_ref
        ):
            raise ValueError("parent-held decision drifted before sealed publication")
        sealed_record = _with_artifact_id(
            {
                "schema_version": "relationship-product-sealed-decision.v1",
                **evaluator_session.__dict__,
                "selected_action_id": action.value,
                "typed_outcome_id": outcome.typed_outcome.value,
                "rendered_user_reaction": outcome.rendered_user_reaction,
                "environment_evidence_ref": outcome.environment_evidence_ref,
                "published_after_all_sut_actions_completed": True,
            }
        )
        sealed_path = sealed_root / (f"decision-{_integer(decision.get('decision_index'), 'decision_index'):02d}.json")
        _write_json_create_only(sealed_path, sealed_record)
        enriched_decisions.append(
            {
                **decision,
                "sealed_record_path": _relative_posix(root, sealed_path),
                "sealed_record_sha256": _sha256_file(sealed_path),
                "sealed_record_artifact_id": sealed_record["artifact_id"],
            }
        )
    payload = _with_artifact_id({**chain, "decisions": enriched_decisions})
    _write_json_create_only(chain_root / "chain.json", payload)
    return payload


def run_relationship_product_onboarding_worker(
    *, request_path: pathlib.Path, receipt_path: pathlib.Path, run_root: pathlib.Path
) -> None:
    request = _load_json(pathlib.Path(request_path))
    _validate_worker_request(request, operation="onboarding", expected_parent_pid=os.getppid())
    root = pathlib.Path(run_root).resolve()
    protocol = _registered_product_protocol_for_id(request.get("protocol_id"))
    source_lineage = (
        _v2_worker_source_lineage(
            root=root,
            protocol=protocol,
            request=request,
        )
        if protocol.is_v2
        else None
    )
    state_root = _resolve_relative(root, _text(request.get("state_root"), "state_root"))
    session = _mapping(request["session"], "session")
    store, hydration, loaded, pre_hash = _load_owner_state(state_root)
    onboarding = RelationshipProductOnboardingInput(
        session_id=_text(session.get("session_id"), "session_id"),
        session_index=_integer(session.get("session_index"), "session_index"),
        turn_index=_integer(session.get("session_index"), "session_index"),
        public_observation=_text(session.get("user_utterance"), "user_utterance"),
        action_id=_text(session.get("assistant_action_id"), "assistant_action_id"),
        observed_outcome_id=_text(session.get("observed_outcome_id"), "observed_outcome_id"),
        reaction_summary=_text(
            session.get("rendered_user_reaction"),
            "rendered_user_reaction",
        ),
        evidence_ref=f"public-onboarding:{sha256_json(session)}",
    )
    onboarding_snapshot = asyncio.run(
        append_relationship_product_onboarding(
            owner_persistence_snapshot=store.export_persistence_snapshot(),
            onboarding=onboarding,
        )
    )
    next_store = SocialRecordStore()
    next_store.hydrate_from_persistence(
        onboarding_snapshot.owner_persistence_snapshot,
    )
    snapshot = hydration.export_and_save_owner(next_store, _OWNER_NAME)
    if protocol.is_v2:
        source_lineage = _v2_worker_source_lineage(
            root=root,
            protocol=protocol,
            request=request,
        )
    core = {
        "schema_version": (
            RELATIONSHIP_PRODUCT_ONBOARDING_RECEIPT_SCHEMA_VERSION_V2
            if protocol.is_v2
            else RELATIONSHIP_PRODUCT_ONBOARDING_RECEIPT_SCHEMA_VERSION
        ),
        "request_artifact_id": request["artifact_id"],
        "invocation_nonce": request["invocation_nonce"],
        "child_pid": os.getpid(),
        "parent_pid": os.getppid(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "owner_loaded": loaded,
        "pre_owner_snapshot_sha256": pre_hash,
        "owner_snapshot_sha256": sha256_json(snapshot.payload),
        "launch_identity_sha256": sha256_json(
            {"child_pid": os.getpid(), "nonce": request["invocation_nonce"], "request": request["artifact_id"]}
        ),
        "subprocess_environment_contract_sha256": sha256_json(
            _subprocess_environment_contract(protocol)
        ),
        "model_output_count": 0,
        "sealed_truth_received": False,
    }
    if protocol.is_v2:
        core["execution_source_lineage"] = _mapping(
            source_lineage,
            "onboarding worker source lineage",
        )
    _write_json_create_only(pathlib.Path(receipt_path), _with_artifact_id(core))


def run_relationship_product_decision_worker(
    *,
    request_path: pathlib.Path,
    preaction_receipt_path: pathlib.Path,
    postaction_receipt_path: pathlib.Path,
    run_root: pathlib.Path,
) -> None:
    request = _load_json(pathlib.Path(request_path))
    _validate_worker_request(request, operation="decision_handshake", expected_parent_pid=os.getppid())
    _assert_truth_firewall(request)
    root = pathlib.Path(run_root).resolve()
    protocol = _registered_product_protocol_for_id(request.get("protocol_id"))
    source_lineage = (
        _v2_worker_source_lineage(
            root=root,
            protocol=protocol,
            request=request,
        )
        if protocol.is_v2
        else None
    )
    state_root = _resolve_relative(root, _text(request.get("state_root"), "state_root"))
    gate_state_root = _resolve_relative(
        root,
        _text(request.get("gate_state_root"), "gate_state_root"),
    )
    table_path = _resolve_relative(root, _text(request.get("semantic_table_path"), "semantic_table_path"))
    table = load_precomputed_public_embedding_table(table_path)
    if table.artifact_id != request["semantic_table_artifact_id"]:
        raise ValueError("worker semantic table artifact drifted")
    backend_label = _semantic_backend_label(table.source_embedder_name)
    if backend_label != request["semantic_backend"]:
        raise ValueError("worker semantic backend label drifted")
    semantic = PrecomputedPublicSemanticEmbedder(table)
    arm = RelationshipProductArm(_text(request.get("arm_id"), "arm_id"))
    runtime = _relationship_forecast_runtime(
        protocol=protocol,
        arm=arm,
        semantic=semantic,
    )
    store, hydration, loaded, pre_owner_hash = _load_owner_state(state_root)
    owner_snapshot = store.export_persistence_snapshot()
    gate_path = gate_state_root / _GATE_CHECKPOINT_FILENAME
    gate_checkpoint = _load_gate_checkpoint(gate_path)
    session = _mapping(request["session"], "session")
    decision_index = _integer(session.get("decision_index"), "decision_index")
    action_turn = 4 + decision_index * 2
    forecast_request = PreferenceActionForecastRequest(
        decision_id=_text(session.get("decision_id"), "decision_id"),
        interlocutor_id=_INTERLOCUTOR_ID,
        current_observation=_text(session.get("current_input"), "current_input"),
        observation_ref=f"public-decision:{sha256_json(session)}",
        candidate_action_ids=tuple(action.value for action in RELATIONSHIP_ACTIONS),
        outcome_ids=tuple(outcome.value for outcome in RELATIONSHIP_OUTCOMES),
        turn_index=action_turn,
        session_scope=_text(request.get("subject_scope"), "subject_scope"),
    )
    pulse_request = RelationshipProductPreActionRequest(
        session_id=_text(session.get("session_id"), "session_id"),
        forecast_request=forecast_request,
        outcome_turn_index=action_turn + 1,
    )
    authorization = RelationshipProductPulseAuthorization(
        authorization_id=_text(request.get("authorization_id"), "authorization_id"),
        allowed_policy_artifact_id="relationship-action-gate-zero-init",
        allowed_policy_artifact_version=1,
    )
    preaction = asyncio.run(
        prepare_relationship_product_preaction(
            request=pulse_request,
            owner_persistence_snapshot=owner_snapshot,
            gate_checkpoint=gate_checkpoint,
            forecast_runtime=runtime,
            gate_mode=RelationshipActionGateMode(_text(request.get("gate_mode"), "gate_mode")),
            authorization=authorization,
            substrate_snapshot=_placeholder_substrate(),
        )
    )
    forecast_payload = _forecast_payload_for_protocol(
        protocol=protocol,
        forecast=preaction.forecast,
    )
    if protocol.is_v2:
        source_lineage = _v2_worker_source_lineage(
            root=root,
            protocol=protocol,
            request=request,
        )
    pre_core = {
        "schema_version": (
            RELATIONSHIP_PRODUCT_PREACTION_RECEIPT_SCHEMA_VERSION_V2
            if protocol.is_v2
            else RELATIONSHIP_PRODUCT_PREACTION_RECEIPT_SCHEMA_VERSION
        ),
        "request_artifact_id": request["artifact_id"],
        "invocation_nonce": request["invocation_nonce"],
        "child_pid": os.getpid(),
        "parent_pid": os.getppid(),
        "launch_identity_sha256": sha256_json(
            {"child_pid": os.getpid(), "nonce": request["invocation_nonce"], "request": request["artifact_id"]}
        ),
        "subprocess_environment_contract_sha256": sha256_json(
            _subprocess_environment_contract(protocol)
        ),
        "owner_loaded": loaded,
        "pre_owner_snapshot_sha256": pre_owner_hash,
        "forecast_id": preaction.forecast.forecast_id,
        "forecast_sha256": sha256_json(forecast_payload),
        "recommended_action_id": preaction.forecast.recommended_action_id,
        "selected_action_id": preaction.gate_decision.selected_action_id,
        "gate_decision": preaction.gate_decision.to_payload(),
        "gate_update_count_before": preaction.gate_checkpoint_before.update_count,
        "semantic_backend": backend_label,
        "semantic_table_artifact_id": table.artifact_id,
        "semantic_similarity_formula": _semantic_similarity_formula(
            protocol=protocol,
            arm=arm,
        ),
        "sealed_truth_received_before_preaction": False,
        "preaction_fsynced_before_settlement_read": True,
        "model_output_count": 0,
    }
    if protocol.persists_full_forecast:
        pre_core["frozen_forecast"] = _frozen_forecast_envelope(
            preaction.forecast
        )
        pre_core["execution_source_lineage"] = _mapping(
            source_lineage,
            "decision worker source lineage",
        )
    pre_receipt = _with_artifact_id(pre_core)
    _write_json_create_only(pathlib.Path(preaction_receipt_path), pre_receipt)
    print(
        canonical_json(
            {
                "phase": "preaction_fsynced",
                "request_artifact_id": request["artifact_id"],
                "preaction_artifact_id": pre_receipt["artifact_id"],
                "child_pid": os.getpid(),
            }
        ),
        flush=True,
    )

    raw_settlement = sys.stdin.readline()
    if not raw_settlement:
        raise EOFError("decision worker did not receive typed settlement")
    try:
        settlement_payload = json.loads(raw_settlement)
    except json.JSONDecodeError as exc:
        raise ValueError("decision worker settlement line is not valid JSON") from exc
    _assert_settlement_firewall(settlement_payload)
    settlement_input = _settlement_input_from_payload(settlement_payload)
    if settlement_input.apply_credit_to_gate != request["apply_credit_to_gate"]:
        raise ValueError("settlement credit application drifted from worker request")
    if (
        settlement_input.owner_outcome_evidence.observation_summary
        != preaction.request.forecast_request.current_observation
    ):
        raise ValueError(
            "worker settlement observation lineage mismatch before pulse; "
            f"observed={settlement_input.owner_outcome_evidence.observation_summary!a}, "
            f"expected={preaction.request.forecast_request.current_observation!a}"
        )
    settled = asyncio.run(settle_relationship_product_pulse(preaction=preaction, settlement_input=settlement_input))
    next_store = SocialRecordStore()
    next_store.hydrate_from_persistence(settled.owner_persistence_snapshot)
    saved = hydration.export_and_save_owner(next_store, _OWNER_NAME)
    _write_gate_checkpoint(gate_path, settled.gate_checkpoint)
    post_source_lineage = (
        _v2_worker_source_lineage(
            root=root,
            protocol=protocol,
            request=request,
        )
        if protocol.is_v2
        else None
    )
    post_core = {
        "schema_version": (
            RELATIONSHIP_PRODUCT_POSTACTION_RECEIPT_SCHEMA_VERSION_V2
            if protocol.is_v2
            else RELATIONSHIP_PRODUCT_POSTACTION_RECEIPT_SCHEMA_VERSION
        ),
        "request_artifact_id": request["artifact_id"],
        "preaction_artifact_id": pre_receipt["artifact_id"],
        "settlement_payload_sha256": sha256_json(settlement_payload),
        "child_pid": os.getpid(),
        "forecast_id": settled.settlement.forecast_id,
        "settlement_id": settled.settlement.settlement_id,
        "typed_outcome_id": settled.settlement.observed_outcome_id,
        "social_prediction_error_snapshot_sha256": sha256_json(
            _social_pe_payload(settled.social_prediction_error_snapshot.value)
        ),
        "credit_record_id": settled.credit.record_id,
        "credit_value_hex": settled.credit.credit_value.hex(),
        "credit_applied_to_gate": settled.credit_applied_to_gate,
        "gate_update_count_after": settled.gate_checkpoint.update_count,
        "post_owner_snapshot_sha256": sha256_json(saved.payload),
        "subprocess_environment_contract_sha256": sha256_json(
            _subprocess_environment_contract(protocol)
        ),
        "settlement_read_after_preaction_fsync": True,
        "evaluator_or_judge_feedback_received": False,
        "model_output_count": 0,
    }
    if protocol.is_v2:
        post_core["execution_source_lineage"] = _mapping(
            post_source_lineage,
            "postaction worker source lineage",
        )
    post_receipt = _with_artifact_id(post_core)
    _write_json_create_only(pathlib.Path(postaction_receipt_path), post_receipt)
    print(
        canonical_json(
            {
                "phase": "settlement_fsynced",
                "request_artifact_id": request["artifact_id"],
                "postaction_artifact_id": post_receipt["artifact_id"],
                "child_pid": os.getpid(),
            }
        ),
        flush=True,
    )


def validate_relationship_product_horizon_campaign(
    *,
    output_dir: pathlib.Path,
    expected_protocol_id: str | None = None,
) -> Mapping[str, object]:
    """Model/CUDA-free recomputation of source, chain, metric, and tree claims."""

    root = pathlib.Path(output_dir).resolve()
    manifest = _load_json(root / "manifest.json")
    _validate_content_addressed(manifest, "campaign manifest")
    if manifest.get("schema_version") != RELATIONSHIP_PRODUCT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("campaign manifest schema mismatch")
    observed_files = _manifest_file_entries(root)
    if observed_files != manifest.get("files"):
        raise ValueError("campaign manifest file tree/hash mismatch")
    if manifest.get("file_count") != len(observed_files):
        raise ValueError("campaign manifest file count mismatch")
    stored_protocol_path = root / "protocol.json"
    packaged_protocol_path = _registered_product_protocol_path_for_bytes(
        stored_protocol_path.read_bytes()
    )
    protocol = load_relationship_product_horizon_protocol(packaged_protocol_path)
    if protocol.is_v2 and expected_protocol_id is None:
        raise ValueError(
            "v2 campaign validation requires an external expected protocol id"
        )
    if (
        expected_protocol_id is not None
        and _digest(expected_protocol_id, "expected_protocol_id")
        != protocol.protocol_id
    ):
        raise ValueError("campaign protocol differs from the external expected protocol id")
    if manifest.get("protocol_id") != protocol.protocol_id:
        raise ValueError("campaign protocol/manifest mismatch")
    execution_source_bundle = _validate_execution_source_bundle(
        root=root,
        protocol=protocol,
    )
    report = _load_json(root / "report.json")
    _validate_content_addressed(report, "campaign report")
    if manifest.get("report_artifact_id") != report.get("artifact_id"):
        raise ValueError("campaign report/manifest mismatch")
    source_protocol = load_relationship_product_pilot_source_protocol()
    public_view = build_relationship_product_pilot_public_view(source_protocol)
    evaluator = build_relationship_product_pilot_evaluator_bundle(source_protocol)
    _validate_campaign_source_binding(
        protocol,
        source_protocol,
        public_view,
        evaluator,
    )
    if _load_json(root / "source" / "public" / "public_plan.json") != public_view.to_sut_payload():
        raise ValueError("stored public source differs from the pinned source owner")
    if _load_json(root / "source" / "sealed" / "evaluator_bundle.json") != _evaluator_payload(evaluator):
        raise ValueError("stored sealed source differs from the pinned evaluator owner")
    table = load_precomputed_public_embedding_table(root / "inputs" / "public_embedding_table.json")
    if table.artifact_id != report.get("embedding_table_artifact_id"):
        raise ValueError("report embedding table lineage mismatch")
    semantic_backend = _semantic_backend_label(table.source_embedder_name)
    if semantic_backend != report.get("semantic_backend"):
        raise ValueError("report semantic backend label mismatch")
    reobservation_path = root / "inputs" / "public_embedding_reobservation.json"
    if semantic_backend == "bge_m3_precomputed_public_table":
        _validate_protocol_pinned_embedding_table(
            protocol=protocol,
            table=table,
            table_path=root / "inputs" / "public_embedding_table.json",
        )
        reobservation = _load_json(reobservation_path)
        _validate_embedding_reobservation_attestation(
            reobservation,
            protocol=protocol,
            table=table,
            public_plan_sha256=public_view.public_plan_sha256,
        )
        embedding_table_fresh_process_reobserved = True
    else:
        if reobservation_path.exists():
            raise ValueError("fake semantic campaign fabricated BGE reobservation")
        embedding_table_fresh_process_reobserved = False
    selection_payload = _mapping(report.get("selection"), "report.selection")
    selection = RelationshipProductCampaignSelection(
        subject_count=_integer(selection_payload.get("subject_count"), "selection.subject_count"),
        onboarding_session_count=_integer(
            selection_payload.get("onboarding_session_count"),
            "selection.onboarding_session_count",
        ),
        decision_session_count=_integer(
            selection_payload.get("decision_session_count"),
            "selection.decision_session_count",
        ),
    )
    if selection_payload.get("full_protocol_executed") is not selection.is_full:
        raise ValueError("report full-protocol flag drifted")
    typed_chains, baseline_chains = _load_and_validate_campaign_chains(
        root=root,
        protocol=protocol,
        public_view=public_view,
        evaluator=evaluator,
        embedding_table=table,
        selection=selection,
        baselines_executed=_boolean(
            report.get("strong_baselines_executed"),
            "report.strong_baselines_executed",
        ),
    )
    recomputed = _build_report(
        root=root,
        protocol=protocol,
        source_protocol_id=source_protocol.protocol_sha256,
        public_plan_sha256=public_view.public_plan_sha256,
        sealed_bundle_sha256=evaluator.sealed_bundle_sha256,
        embedding_table_artifact_id=table.artifact_id,
        semantic_backend=semantic_backend,
        embedding_table_fresh_process_reobserved=(embedding_table_fresh_process_reobserved),
        execution_source_bundle_artifact_id=_digest(
            execution_source_bundle.get("artifact_id"),
            "execution source bundle artifact_id",
        ),
        selection=selection,
        typed_chains=typed_chains,
        baseline_chains=baseline_chains,
    )
    if recomputed != report:
        raise ValueError("campaign report does not equal offline chain/metric recomputation")
    launch_ids = tuple(report.get("launch_identity_sha256s", ()))
    request_ids = tuple(report.get("worker_request_artifact_ids", ()))
    if len(launch_ids) != len(set(launch_ids)) or len(request_ids) != len(set(request_ids)):
        raise ValueError("campaign process/request launch identities are not unique")
    expected_count = report.get("volvence_logical_session_count")
    if len(launch_ids) != expected_count or len(request_ids) != expected_count:
        raise ValueError("campaign fresh-process receipt count mismatch")
    if (
        report.get("residual_steerable")
        or report.get("user_visible_generation")
        or report.get("four_able_complete")
        or report.get("production_active")
        or report.get("os_security_boundary")
        or report.get("single_axis_contrast_claim_authorized")
    ):
        raise ValueError("campaign report overclaims the typed-control boundary")
    return report


def _baseline_dispatcher_identity_summary(
    baseline_chains: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    """Aggregate the process identity shared by every strong-baseline chain."""

    observed_backends: set[str] = set()
    startup_artifact_ids: set[str] = set()
    process_pids: set[int] = set()
    for chain in baseline_chains:
        decisions = tuple(
            _mapping(item, "baseline dispatcher identity decision")
            for item in _list(
                chain.get("decisions"),
                "baseline dispatcher identity decisions",
            )
        )
        chain_backends = {
            _text(
                item.get("baseline_execution_backend"),
                "baseline dispatcher identity backend",
            )
            for item in decisions
        }
        if len(chain_backends) != 1:
            raise ValueError("baseline chain mixed execution backends")
        backend = next(iter(chain_backends))
        observed_backends.add(backend)
        startup_raw = chain.get("dispatcher_startup_attestation")
        if backend == "resident_jsonl_dispatcher":
            startup = _mapping(
                startup_raw,
                "resident baseline dispatcher startup attestation",
            )
            _validate_content_addressed(
                startup,
                "resident baseline dispatcher startup attestation",
            )
            startup_artifact_ids.add(
                _digest(
                    startup.get("artifact_id"),
                    "resident baseline dispatcher startup artifact_id",
                )
            )
            process_pid = _integer(
                startup.get("process_pid"),
                "resident baseline dispatcher process_pid",
            )
            if process_pid <= 0:
                raise ValueError(
                    "resident baseline dispatcher process_pid must be positive"
                )
            process_pids.add(process_pid)
        elif backend == "injected_resident_suite":
            if startup_raw is not None:
                raise ValueError("injected baseline fabricated startup attestation")
        else:
            raise ValueError("baseline execution backend is not an admitted interface")

    if not baseline_chains:
        single_resident_verified = False
    elif observed_backends == {"resident_jsonl_dispatcher"}:
        if len(startup_artifact_ids) != 1 or len(process_pids) != 1:
            raise ValueError(
                "strong baseline chains do not share one resident dispatcher process"
            )
        single_resident_verified = True
    elif observed_backends == {"injected_resident_suite"}:
        single_resident_verified = False
    else:
        raise ValueError("campaign baseline chains mixed execution backends")

    return {
        "startup_artifact_ids": sorted(startup_artifact_ids),
        "process_pids": sorted(process_pids),
        "single_resident_dispatcher_verified": single_resident_verified,
    }


def _load_and_validate_campaign_chains(
    *,
    root: pathlib.Path,
    protocol: RelationshipProductHorizonProtocol,
    public_view: RelationshipProductPilotPublicView,
    evaluator: RelationshipProductPilotEvaluatorBundle,
    embedding_table: PrecomputedPublicEmbeddingTable,
    selection: RelationshipProductCampaignSelection,
    baselines_executed: bool,
) -> tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
    subjects = public_view.subjects[: selection.subject_count]
    expected_arms = (
        (*protocol.volvence_arms, *_BASELINE_ARMS)
        if baselines_executed
        else protocol.volvence_arms
    )
    observed: dict[tuple[str, str], Mapping[str, object]] = {}
    for path in sorted((root / "chains").glob("*/*/chain.json")):
        chain = _load_json(path)
        _validate_content_addressed(chain, "campaign chain")
        subject_scope = _text(chain.get("subject_scope"), "chain.subject_scope")
        arm_id = _text(chain.get("arm_id"), "chain.arm_id")
        if path.parent.name != arm_id or path.parent.parent.name != subject_scope:
            raise ValueError("campaign chain path/identity mismatch")
        key = (subject_scope, arm_id)
        if key in observed:
            raise ValueError("duplicate campaign subject/arm chain")
        observed[key] = chain
    expected_keys = {(subject.subject_scope, arm_id) for subject in subjects for arm_id in expected_arms}
    if set(observed) != expected_keys:
        raise ValueError(
            "campaign chain set mismatch; "
            f"missing={sorted(expected_keys - set(observed))}, "
            f"extra={sorted(set(observed) - expected_keys)}"
        )
    typed: list[Mapping[str, object]] = []
    baselines: list[Mapping[str, object]] = []
    for subject in subjects:
        for arm_id in protocol.volvence_arms:
            chain = observed[(subject.subject_scope, arm_id)]
            _validate_typed_chain(
                root=root,
                protocol=protocol,
                evaluator=evaluator,
                embedding_table=embedding_table,
                subject=subject,
                arm=RelationshipProductArm(arm_id),
                selection=selection,
                chain=chain,
            )
            typed.append(chain)
        if baselines_executed:
            for arm_id in _BASELINE_ARMS:
                chain = observed[(subject.subject_scope, arm_id)]
                _validate_baseline_chain(
                    root=root,
                    protocol=protocol,
                    evaluator=evaluator,
                    subject=subject,
                    arm=RelationshipProductArm(arm_id),
                    selection=selection,
                    chain=chain,
                    public_plan_artifact_id=public_view.public_plan_sha256,
                )
                baselines.append(chain)
    if protocol.is_v2:
        _baseline_dispatcher_identity_summary(baselines)
    return tuple(typed), tuple(baselines)


def _validate_typed_chain(
    *,
    root: pathlib.Path,
    protocol: RelationshipProductHorizonProtocol,
    evaluator: RelationshipProductPilotEvaluatorBundle,
    embedding_table: PrecomputedPublicEmbeddingTable,
    subject: ProductPilotPublicSubject,
    arm: RelationshipProductArm,
    selection: RelationshipProductCampaignSelection,
    chain: Mapping[str, object],
) -> None:
    if chain.get("schema_version") != "relationship-product-typed-chain.v1":
        raise ValueError("typed chain schema mismatch")
    if chain.get("world_clone_id") != subject.world_clone_id:
        raise ValueError("typed chain world-clone mismatch")
    execution_source_bundle_artifact_id = (
        _digest(
            _load_json(
                root / "inputs" / "execution_sources" / "bundle.json"
            ).get("artifact_id"),
            "execution source bundle artifact_id",
        )
        if protocol.is_v2
        else None
    )
    expected_reset_basis = (
        "same_frozen_post_onboarding_boundary_each_decision"
        if arm is RelationshipProductArm.APPENDABLE_FROZEN_ONBOARDING
        else "hydrate_exact_prior_decision_boundary"
    )
    if chain.get("appendable_reset_basis") != expected_reset_basis:
        raise ValueError("typed chain appendable intervention drifted")
    replay_store = SocialRecordStore()
    replay_owner_snapshot = replay_store.export_persistence_snapshot()
    onboarding_records = _list(
        chain.get("onboarding_receipts"),
        "typed chain onboarding_receipts",
    )
    expected_onboarding = subject.onboarding_sessions[: selection.onboarding_session_count]
    if len(onboarding_records) != len(expected_onboarding):
        raise ValueError("typed chain onboarding count mismatch")
    last_owner_hash: str | None = None
    onboarding_state_root: str | None = None
    for index, (raw_record, session) in enumerate(zip(onboarding_records, expected_onboarding, strict=True)):
        record = _mapping(raw_record, "typed onboarding record")
        request = _referenced_json(
            root,
            record,
            path_key="request_path",
            sha_key="request_sha256",
            artifact_key="request_artifact_id",
            source="typed onboarding request",
        )
        _validate_worker_request(
            request,
            operation="onboarding",
            expected_parent_pid=None,
        )
        if (
            request.get("protocol_id") != protocol.protocol_id
            or request.get("arm_id") != arm.value
            or request.get("subject_scope") != subject.subject_scope
            or request.get("world_clone_id") != subject.world_clone_id
            or request.get("session") != session.to_sut_payload()
        ):
            raise ValueError("typed onboarding request/source lineage mismatch")
        if onboarding_state_root is None:
            onboarding_state_root = _text(
                request.get("state_root"),
                "onboarding state_root",
            )
        elif request.get("state_root") != onboarding_state_root:
            raise ValueError("typed onboarding owner state root changed within chain")
        receipt = _referenced_json(
            root,
            record,
            path_key="receipt_path",
            sha_key="receipt_sha256",
            artifact_key="receipt_artifact_id",
            source="typed onboarding receipt",
        )
        _validate_onboarding_receipt(receipt)
        expected_onboarding_schema = (
            RELATIONSHIP_PRODUCT_ONBOARDING_RECEIPT_SCHEMA_VERSION_V2
            if protocol.is_v2
            else RELATIONSHIP_PRODUCT_ONBOARDING_RECEIPT_SCHEMA_VERSION
        )
        if (
            receipt.get("schema_version") != expected_onboarding_schema
            or receipt.get("request_artifact_id") != request.get("artifact_id")
            or receipt.get("invocation_nonce") != request.get("invocation_nonce")
            or receipt.get("parent_pid") != request.get("parent_pid")
            or receipt.get("child_pid") != record.get("child_pid")
            or receipt.get("launch_identity_sha256") != record.get("launch_identity_sha256")
            or receipt.get("owner_snapshot_sha256") != record.get("owner_snapshot_sha256")
        ):
            raise ValueError("typed onboarding receipt/chain lineage mismatch")
        if protocol.is_v2:
            _validate_v2_worker_source_lineage(
                _mapping(
                    receipt.get("execution_source_lineage"),
                    "onboarding execution source lineage",
                ),
                root=root,
                protocol=protocol,
                execution_source_bundle_artifact_id=_digest(
                    execution_source_bundle_artifact_id,
                    "execution source bundle artifact_id",
                ),
            )
        if bool(receipt.get("owner_loaded")) is not (index > 0):
            raise ValueError("typed onboarding hydration restart evidence drifted")
        if receipt.get("pre_owner_snapshot_sha256") != sha256_json(replay_owner_snapshot.payload):
            raise ValueError("onboarding pre-owner snapshot cannot be replayed")
        replayed_onboarding = asyncio.run(
            append_relationship_product_onboarding(
                owner_persistence_snapshot=replay_owner_snapshot,
                onboarding=RelationshipProductOnboardingInput(
                    session_id=session.session_id,
                    session_index=session.session_index,
                    turn_index=session.session_index,
                    public_observation=session.user_utterance,
                    action_id=session.assistant_action_id,
                    observed_outcome_id=session.observed_outcome_id,
                    reaction_summary=session.rendered_user_reaction,
                    evidence_ref=f"public-onboarding:{sha256_json(session.to_sut_payload())}",
                ),
            )
        )
        replay_owner_snapshot = replayed_onboarding.owner_persistence_snapshot
        if receipt.get("owner_snapshot_sha256") != sha256_json(replay_owner_snapshot.payload):
            raise ValueError("onboarding owner snapshot cannot be replayed")
        last_owner_hash = _digest(
            receipt.get("owner_snapshot_sha256"),
            "onboarding owner snapshot",
        )
    if last_owner_hash is None:
        raise ValueError("typed chain has no onboarding boundary")

    decisions = _list(chain.get("decisions"), "typed chain decisions")
    expected_decisions = subject.decision_sessions[: selection.decision_session_count]
    if len(decisions) != len(expected_decisions):
        raise ValueError("typed chain decision count mismatch")
    previous_post_owner_hash = last_owner_hash
    previous_gate_update_count = 0
    frozen_onboarding_snapshot = replay_owner_snapshot
    replay_gate_checkpoint: RelationshipActionGateCheckpoint | None = None
    gate_state_root: str | None = None
    seen_decision_state_roots: set[str] = set()
    for raw_record, session in zip(decisions, expected_decisions, strict=True):
        record = _mapping(raw_record, "typed decision record")
        request = _referenced_json(
            root,
            record,
            path_key="request_path",
            sha_key="request_sha256",
            artifact_key="request_artifact_id",
            source="typed decision request",
        )
        _validate_worker_request(
            request,
            operation="decision_handshake",
            expected_parent_pid=None,
        )
        if (
            request.get("protocol_id") != protocol.protocol_id
            or request.get("arm_id") != arm.value
            or request.get("subject_scope") != subject.subject_scope
            or request.get("world_clone_id") != subject.world_clone_id
            or request.get("session") != session.to_sut_payload()
        ):
            raise ValueError("typed decision request/source lineage mismatch")
        current_gate_root = _text(
            request.get("gate_state_root"),
            "decision gate_state_root",
        )
        if gate_state_root is None:
            gate_state_root = current_gate_root
        elif current_gate_root != gate_state_root:
            raise ValueError("typed gate checkpoint root changed within chain")
        decision_state_root = _text(
            request.get("state_root"),
            "decision state_root",
        )
        if arm is RelationshipProductArm.APPENDABLE_FROZEN_ONBOARDING:
            if decision_state_root in seen_decision_state_roots:
                raise ValueError("frozen-onboarding arm reused a mutable owner state root")
            seen_decision_state_roots.add(decision_state_root)
        elif onboarding_state_root != decision_state_root:
            raise ValueError("non-appendable intervention changed the owner state root")
        pre = _referenced_json(
            root,
            record,
            path_key="preaction_receipt_path",
            sha_key="preaction_receipt_sha256",
            artifact_key="preaction_artifact_id",
            source="typed preaction receipt",
        )
        post = _referenced_json(
            root,
            record,
            path_key="postaction_receipt_path",
            sha_key="postaction_receipt_sha256",
            artifact_key="postaction_artifact_id",
            source="typed postaction receipt",
        )
        _validate_preaction_receipt(pre)
        _validate_postaction_receipt(post)
        expected_preaction_schema = (
            RELATIONSHIP_PRODUCT_PREACTION_RECEIPT_SCHEMA_VERSION_V2
            if protocol.is_v2
            else RELATIONSHIP_PRODUCT_PREACTION_RECEIPT_SCHEMA_VERSION
        )
        expected_postaction_schema = (
            RELATIONSHIP_PRODUCT_POSTACTION_RECEIPT_SCHEMA_VERSION_V2
            if protocol.is_v2
            else RELATIONSHIP_PRODUCT_POSTACTION_RECEIPT_SCHEMA_VERSION
        )
        if (
            pre.get("schema_version") != expected_preaction_schema
            or post.get("schema_version") != expected_postaction_schema
            or pre.get("request_artifact_id") != request.get("artifact_id")
            or pre.get("invocation_nonce") != request.get("invocation_nonce")
            or pre.get("parent_pid") != request.get("parent_pid")
            or post.get("request_artifact_id") != request.get("artifact_id")
            or post.get("preaction_artifact_id") != pre.get("artifact_id")
            or pre.get("child_pid") != post.get("child_pid")
            or pre.get("child_pid") != record.get("child_pid")
            or pre.get("launch_identity_sha256") != record.get("launch_identity_sha256")
        ):
            raise ValueError("typed decision handshake lineage mismatch")
        if protocol.is_v2:
            _validate_v2_worker_source_lineage(
                _mapping(
                    pre.get("execution_source_lineage"),
                    "preaction execution source lineage",
                ),
                root=root,
                protocol=protocol,
                execution_source_bundle_artifact_id=_digest(
                    execution_source_bundle_artifact_id,
                    "execution source bundle artifact_id",
                ),
            )
            _validate_v2_worker_source_lineage(
                _mapping(
                    post.get("execution_source_lineage"),
                    "postaction execution source lineage",
                ),
                root=root,
                protocol=protocol,
                execution_source_bundle_artifact_id=_digest(
                    execution_source_bundle_artifact_id,
                    "execution source bundle artifact_id",
                ),
            )
        expected_pre_owner = (
            last_owner_hash if arm is RelationshipProductArm.APPENDABLE_FROZEN_ONBOARDING else previous_post_owner_hash
        )
        if pre.get("pre_owner_snapshot_sha256") != expected_pre_owner:
            raise ValueError("typed decision owner hydration lineage mismatch")
        if pre.get("gate_update_count_before") != previous_gate_update_count:
            raise ValueError("typed decision gate checkpoint lineage mismatch")
        expected_gate_increment = int(
            arm
            not in {
                RelationshipProductArm.CREDIT_WITHHELD,
                RelationshipProductArm.STRICT_NOOP,
            }
        )
        if post.get("gate_update_count_after") != (previous_gate_update_count + expected_gate_increment):
            raise ValueError("typed decision gate update count drifted")
        if bool(post.get("credit_applied_to_gate")) is not bool(expected_gate_increment):
            raise ValueError("typed decision credit application drifted")
        replay_owner_input = (
            frozen_onboarding_snapshot
            if arm is RelationshipProductArm.APPENDABLE_FROZEN_ONBOARDING
            else replay_owner_snapshot
        )
        replay_preaction = _prepare_replayed_preaction(
            protocol=protocol,
            subject=subject,
            arm=arm,
            session=session,
            owner_snapshot=replay_owner_input,
            gate_checkpoint=replay_gate_checkpoint,
            embedding_table=embedding_table,
        )
        replayed_forecast_payload = _forecast_payload_for_protocol(
            protocol=protocol,
            forecast=replay_preaction.forecast,
        )
        expected_preaction_projection = {
            "pre_owner_snapshot_sha256": sha256_json(replay_owner_input.payload),
            "forecast_id": replay_preaction.forecast.forecast_id,
            "forecast_sha256": sha256_json(replayed_forecast_payload),
            "recommended_action_id": replay_preaction.forecast.recommended_action_id,
            "selected_action_id": replay_preaction.gate_decision.selected_action_id,
            "gate_decision": replay_preaction.gate_decision.to_payload(),
            "gate_update_count_before": replay_preaction.gate_checkpoint_before.update_count,
            "semantic_similarity_formula": _semantic_similarity_formula(
                protocol=protocol,
                arm=arm,
            ),
        }
        if protocol.persists_full_forecast:
            expected_preaction_projection["frozen_forecast"] = _frozen_forecast_envelope(
                replay_preaction.forecast
            )
        if {key: pre.get(key) for key in expected_preaction_projection} != expected_preaction_projection:
            raise ValueError("typed preaction forecast/gate cannot be replayed")
        if protocol.is_v2:
            readout = replay_preaction.forecast.condition_readout
            if arm is RelationshipProductArm.READABLE_UNNAMED_LEGACY:
                if readout is not None:
                    raise ValueError("v2 Readable comparator published a named readout")
            elif (
                readout is None
                or readout.reader_artifact_id
                != protocol.condition_reader_artifact_id
            ):
                raise ValueError("v2 named-reader arm did not publish the pinned readout")
        action = RelationshipAction(_text(record.get("selected_action_id"), "selected_action_id"))
        if pre.get("selected_action_id") != action.value:
            raise ValueError("typed preaction selected action drifted")
        if replay_preaction.gate_decision.selected_action_id != action.value:
            raise ValueError("typed chain action differs from replayed gate decision")
        if arm is RelationshipProductArm.STRICT_NOOP and action is not RelationshipAction.NEUTRAL_NOOP:
            raise ValueError("strict-noop arm exposed a non-neutral action")
        outcome, evaluator_session = _recompute_decision_outcome(
            evaluator=evaluator,
            session=session,
            action=action,
        )
        _validate_decision_observation(
            record=record,
            outcome=outcome,
            evaluator_session=evaluator_session,
        )
        if post.get("typed_outcome_id") != outcome.typed_outcome.value:
            raise ValueError("typed postaction outcome drifted from environment")
        settlement = _settlement_payload(
            session=session,
            subject_scope=subject.subject_scope,
            pre_receipt=pre,
            outcome=outcome,
            apply_credit_to_gate=_boolean(
                request.get("apply_credit_to_gate"),
                "apply_credit_to_gate",
            ),
        )
        if post.get("settlement_payload_sha256") != sha256_json(settlement):
            raise ValueError("typed settlement receipt hash cannot be recomputed")
        replay_settlement = asyncio.run(
            settle_relationship_product_pulse(
                preaction=replay_preaction,
                settlement_input=_settlement_input_from_payload(settlement),
            )
        )
        expected_postaction_projection = {
            "forecast_id": replay_settlement.settlement.forecast_id,
            "settlement_id": replay_settlement.settlement.settlement_id,
            "typed_outcome_id": replay_settlement.settlement.observed_outcome_id,
            "social_prediction_error_snapshot_sha256": sha256_json(
                _social_pe_payload(replay_settlement.social_prediction_error_snapshot.value)
            ),
            "credit_record_id": replay_settlement.credit.record_id,
            "credit_value_hex": replay_settlement.credit.credit_value.hex(),
            "credit_applied_to_gate": replay_settlement.credit_applied_to_gate,
            "gate_update_count_after": replay_settlement.gate_checkpoint.update_count,
            "post_owner_snapshot_sha256": sha256_json(replay_settlement.owner_persistence_snapshot.payload),
        }
        if {key: post.get(key) for key in expected_postaction_projection} != expected_postaction_projection:
            raise ValueError("typed PE/credit/owner settlement cannot be replayed")
        _validate_sealed_decision_reference(
            root=root,
            record=record,
            evaluator_session=evaluator_session,
            outcome=outcome,
            action=action,
        )
        if record.get("handshake_order") != [
            "preaction_fsynced",
            "parent_environment_settled",
            "typed_settlement_sent",
            "settlement_fsynced",
        ]:
            raise ValueError("typed decision handshake order drifted")
        previous_post_owner_hash = _digest(
            post.get("post_owner_snapshot_sha256"),
            "post owner snapshot",
        )
        previous_gate_update_count = _integer(
            post.get("gate_update_count_after"),
            "gate_update_count_after",
        )
        replay_gate_checkpoint = replay_settlement.gate_checkpoint
        if arm is not RelationshipProductArm.APPENDABLE_FROZEN_ONBOARDING:
            replay_owner_snapshot = replay_settlement.owner_persistence_snapshot

    expected_final_owner = (
        frozen_onboarding_snapshot
        if arm is RelationshipProductArm.APPENDABLE_FROZEN_ONBOARDING
        else replay_owner_snapshot
    )
    assert onboarding_state_root is not None and gate_state_root is not None
    final_owner_root = _resolve_relative(root, onboarding_state_root)
    final_store, _hydration, loaded, _hash = _load_owner_state(final_owner_root)
    if not loaded or final_store.export_persistence_snapshot() != expected_final_owner:
        raise ValueError("typed chain final persisted owner differs from replay")
    final_gate = _load_gate_checkpoint(_resolve_relative(root, gate_state_root) / _GATE_CHECKPOINT_FILENAME)
    if final_gate != replay_gate_checkpoint:
        raise ValueError("typed chain final gate checkpoint differs from replay")


def _validate_baseline_chain(
    *,
    root: pathlib.Path,
    protocol: RelationshipProductHorizonProtocol,
    evaluator: RelationshipProductPilotEvaluatorBundle,
    subject: ProductPilotPublicSubject,
    arm: RelationshipProductArm,
    selection: RelationshipProductCampaignSelection,
    chain: Mapping[str, object],
    public_plan_artifact_id: str,
) -> None:
    if chain.get("schema_version") != "relationship-product-baseline-chain.v1":
        raise ValueError("baseline chain schema mismatch")
    if chain.get("world_clone_id") != subject.world_clone_id:
        raise ValueError("baseline chain world-clone mismatch")
    startup_attestation_raw = chain.get("dispatcher_startup_attestation")
    startup_attestation = (
        None
        if startup_attestation_raw is None
        else _mapping(
            startup_attestation_raw,
            "baseline dispatcher startup attestation",
        )
    )
    if startup_attestation is not None:
        execution_source_bundle = _validate_execution_source_bundle(
            root=root,
            protocol=protocol,
        )
        _validate_dispatcher_startup_attestation(
            startup_attestation,
            root=root,
            protocol=protocol,
            execution_source_bundle_artifact_id=_digest(
                execution_source_bundle.get("artifact_id"),
                "execution source bundle artifact_id",
            ),
        )
    records = _list(chain.get("decisions"), "baseline chain decisions")
    sessions = subject.decision_sessions[: selection.decision_session_count]
    if len(records) != len(sessions):
        raise ValueError("baseline chain decision count mismatch")
    history = _initial_baseline_history(
        subject,
        selection.onboarding_session_count,
    )
    prior_source_session_ids = [
        onboarding.session_id for onboarding in subject.onboarding_sessions[: selection.onboarding_session_count]
    ]
    expected_result_arm = {
        RelationshipProductArm.NATIVE_FULL_HISTORY: ProductBaselineArm.NATIVE_CHRONOLOGICAL_FULL_HISTORY.value,
        RelationshipProductArm.SELECTIVE_RAG: ProductBaselineArm.SELECTIVE_SEMANTIC_RAG.value,
    }[arm]
    observed_backends: set[str] = set()
    for raw_record, session in zip(records, sessions, strict=True):
        record = _mapping(raw_record, "baseline decision record")
        if (
            record.get("session_id") != session.session_id
            or record.get("decision_id") != session.decision_id
            or record.get("decision_index") != session.decision_index
        ):
            raise ValueError("baseline decision/source boundary mismatch")
        backend = record.get("baseline_execution_backend")
        if backend not in {
            "resident_jsonl_dispatcher",
            "injected_resident_suite",
        }:
            raise ValueError("baseline execution backend is not an admitted interface")
        assert isinstance(backend, str)
        observed_backends.add(backend)
        ledger = _referenced_json(
            root,
            record,
            path_key="public_ledger_path",
            sha_key="public_ledger_sha256",
            artifact_key="public_ledger_artifact_id",
            source="baseline public ledger",
        )
        _validate_public_baseline_ledger(ledger)
        expected_input = ProductBaselineInput(
            history=tuple(history),
            current_observation=ProductCurrentObservation(
                content=f"{session.public_context_chunk}\n\n{session.current_input}"
            ),
        )
        if (
            ledger.get("public_plan_artifact_id") != public_plan_artifact_id
            or ledger.get("subject_scope") != subject.subject_scope
            or ledger.get("world_clone_id") != subject.world_clone_id
            or ledger.get("session_id") != session.session_id
            or ledger.get("decision_id") != session.decision_id
            or ledger.get("decision_index") != session.decision_index
            or ledger.get("ordered_source_session_ids") != [*prior_source_session_ids, session.session_id]
            or ledger.get("ordered_source_block_artifact_ids")
            != [block.artifact_id for block in expected_input.history]
            or ledger.get("public_input") != expected_input.to_payload()
        ):
            raise ValueError("baseline ledger does not equal ordered public history")
        if record.get("public_input_artifact_id") != expected_input.artifact_id:
            raise ValueError("baseline record public input lineage mismatch")
        if record.get("history_block_artifact_ids") != [block.artifact_id for block in expected_input.history]:
            raise ValueError("baseline ordered public block membership drifted")
        if record.get("current_observation_artifact_id") != expected_input.current_observation.artifact_id:
            raise ValueError("baseline current observation lineage drifted")
        result = _mapping(record.get("baseline_result"), "baseline_result")
        _validate_nested_artifact_payloads(result, "baseline result")
        expected_seed = _sha256_u64(
            protocol.generation_seed_namespace,
            protocol.cohort_id,
            subject.subject_scope,
            str(session.decision_index),
        )
        if (
            result.get("arm") != expected_result_arm
            or result.get("seed") != expected_seed
            or result.get("input_artifact_id") != expected_input.artifact_id
        ):
            raise ValueError("baseline result protocol/input lineage mismatch")
        _validate_baseline_result_instrument(
            protocol=protocol,
            arm=arm,
            public_input=expected_input,
            result=result,
            require_protocol_pins=(backend == "resident_jsonl_dispatcher"),
        )
        if backend == "resident_jsonl_dispatcher":
            dispatcher_request = _mapping(
                record.get("baseline_dispatcher_request"),
                "baseline dispatcher request",
            )
            request_nonce = _text(
                dispatcher_request.get("nonce"),
                "baseline dispatcher request nonce",
            )
            expected_request = _expected_dispatcher_request(
                protocol=protocol,
                subject=subject,
                session=session,
                arm=arm,
                public_input=expected_input,
                ledger=ledger,
                prior_source_session_ids=prior_source_session_ids,
                nonce=request_nonce,
            )
            if dispatcher_request != expected_request.to_payload():
                raise ValueError("baseline dispatcher request cannot be reconstructed")
            dispatcher_response = _mapping(
                record.get("baseline_dispatcher_response"),
                "baseline dispatcher response",
            )
            parsed_response = parse_product_baseline_dispatcher_response_line(canonical_json(dispatcher_response))
            if not isinstance(
                parsed_response,
                ProductBaselineDispatcherReceivedResponse,
            ):
                raise ValueError("baseline dispatcher response is not successful")
            if parsed_response.nonce != request_nonce or parsed_response.result_payload != result:
                raise ValueError("baseline dispatcher response/request/result lineage drifted")
        elif (
            record.get("baseline_dispatcher_request") is not None
            or record.get("baseline_dispatcher_response") is not None
        ):
            raise ValueError("injected baseline fabricated dispatcher evidence")
        completion = _mapping(
            result.get("action_completion"),
            "baseline action_completion",
        )
        valid = _boolean(completion.get("valid"), "baseline completion.valid")
        if record.get("valid_completion") is not valid:
            raise ValueError("baseline completion validity receipt drifted")
        chosen_value = completion.get("chosen_action_id")
        if valid:
            action = RelationshipAction(_text(chosen_value, "chosen_action_id"))
            if record.get("invalid_completion_mapped_to") is not None:
                raise ValueError("valid baseline completion claims invalid mapping")
        else:
            if chosen_value is not None:
                raise ValueError("invalid baseline completion carried an action")
            action = RelationshipAction.NEUTRAL_NOOP
            if record.get("invalid_completion_mapped_to") != RelationshipAction.NEUTRAL_NOOP.value:
                raise ValueError("invalid baseline completion was not visibly no-op")
        if record.get("selected_action_id") != action.value:
            raise ValueError("baseline selected action/completion mismatch")
        outcome, evaluator_session = _recompute_decision_outcome(
            evaluator=evaluator,
            session=session,
            action=action,
        )
        _validate_decision_observation(
            record=record,
            outcome=outcome,
            evaluator_session=evaluator_session,
        )
        _validate_sealed_decision_reference(
            root=root,
            record=record,
            evaluator_session=evaluator_session,
            outcome=outcome,
            action=action,
        )
        history.extend(
            _decision_public_history_blocks(
                history,
                session,
                action,
                outcome.typed_outcome.value,
                outcome.rendered_user_reaction,
            )
        )
        prior_source_session_ids.append(session.session_id)
    if observed_backends == {"resident_jsonl_dispatcher"}:
        if startup_attestation is None:
            raise ValueError("resident baseline lacks startup attestation")
    elif observed_backends == {"injected_resident_suite"}:
        if startup_attestation is not None:
            raise ValueError("injected baseline fabricated startup attestation")
    else:
        raise ValueError("baseline chain mixed execution backends")


def _expected_dispatcher_request(
    *,
    protocol: RelationshipProductHorizonProtocol,
    subject: ProductPilotPublicSubject,
    session: ProductPilotPublicDecisionSession,
    arm: RelationshipProductArm,
    public_input: ProductBaselineInput,
    ledger: Mapping[str, object],
    prior_source_session_ids: Sequence[str],
    nonce: str,
) -> ProductBaselineDispatcherRequest:
    history_entries = tuple(
        _mapping(item, "public ledger history entry")
        for item in _list(
            ledger.get("history_entries"),
            "public ledger history entries",
        )
    )
    current_entry = _mapping(
        ledger.get("current_entry"),
        "public ledger current entry",
    )
    baseline_arm = {
        RelationshipProductArm.NATIVE_FULL_HISTORY: (ProductBaselineArm.NATIVE_CHRONOLOGICAL_FULL_HISTORY),
        RelationshipProductArm.SELECTIVE_RAG: (ProductBaselineArm.SELECTIVE_SEMANTIC_RAG),
    }[arm]
    return ProductBaselineDispatcherRequest(
        nonce=nonce,
        arm=baseline_arm,
        public_plan_artifact_id=protocol.public_plan_sha256,
        subject_scope=subject.subject_scope,
        decision_boundary=ProductBaselineDecisionBoundary(
            current_session_id=session.session_id,
            decision_id=session.decision_id,
            decision_index=session.decision_index,
        ),
        ordered_source_session_ids=tuple((*prior_source_session_ids, session.session_id)),
        ordered_source_block_artifact_ids=tuple(block.artifact_id for block in public_input.history),
        public_ledger_artifact_id=_digest(
            ledger.get("artifact_id"),
            "public ledger artifact_id",
        ),
        public_input=public_input,
        history_block_lineage=tuple(
            ProductBaselineHistoryBlockLineage(
                ordinal=block.ordinal,
                block_artifact_id=block.artifact_id,
                public_ledger_entry_artifact_id=_digest(
                    entry.get("artifact_id"),
                    "public ledger entry artifact_id",
                ),
            )
            for block, entry in zip(
                public_input.history,
                history_entries,
                strict=True,
            )
        ),
        current_observation_lineage=(
            ProductBaselineCurrentObservationLineage(
                observation_artifact_id=(public_input.current_observation.artifact_id),
                public_ledger_entry_artifact_id=_digest(
                    current_entry.get("artifact_id"),
                    "public ledger current entry artifact_id",
                ),
            )
        ),
        seed=_sha256_u64(
            protocol.generation_seed_namespace,
            protocol.cohort_id,
            subject.subject_scope,
            str(session.decision_index),
        ),
        top_k=(protocol.rag_top_k if arm is RelationshipProductArm.SELECTIVE_RAG else None),
    )


def _validate_baseline_result_instrument(
    *,
    protocol: RelationshipProductHorizonProtocol,
    arm: RelationshipProductArm,
    public_input: ProductBaselineInput,
    result: Mapping[str, object],
    require_protocol_pins: bool,
) -> None:
    context = _mapping(result.get("context_receipt"), "baseline context receipt")
    completion = _mapping(
        result.get("action_completion"),
        "baseline action completion",
    )
    retrieval = _mapping(
        result.get("retrieval_receipt"),
        "baseline retrieval receipt",
    )
    truncation = _mapping(
        result.get("truncation_receipt"),
        "baseline truncation receipt",
    )
    expected_result_arm = {
        RelationshipProductArm.NATIVE_FULL_HISTORY: (ProductBaselineArm.NATIVE_CHRONOLOGICAL_FULL_HISTORY.value),
        RelationshipProductArm.SELECTIVE_RAG: (ProductBaselineArm.SELECTIVE_SEMANTIC_RAG.value),
    }[arm]
    expected_prompt_sha256 = {
        RelationshipProductArm.NATIVE_FULL_HISTORY: (protocol.baseline_native_prompt_sha256),
        RelationshipProductArm.SELECTIVE_RAG: (protocol.baseline_rag_prompt_sha256),
    }[arm]
    expected_budget = ProductBaselineTokenBudget(
        context_window_tokens=protocol.context_window_tokens,
        generation_reserve_tokens=protocol.generation_token_reserve,
    )
    expected_context_projection = {
        "arm": expected_result_arm,
        "input_artifact_id": public_input.artifact_id,
        "token_budget_artifact_id": expected_budget.artifact_id,
        "generation_reserve_tokens": protocol.generation_token_reserve,
        "context_window_tokens": protocol.context_window_tokens,
    }
    if require_protocol_pins:
        expected_context_projection.update(
            {
                "model_id": protocol.baseline_model_id,
                "weights_sha256": protocol.baseline_model_weights_sha256,
                "generation_config_sha256": (protocol.baseline_generation_config_sha256),
                "arm_prompt_sha256": expected_prompt_sha256,
                "tokenizer_id": protocol.baseline_tokenizer_id,
            }
        )
    if {key: context.get(key) for key in expected_context_projection} != expected_context_projection:
        raise ValueError("baseline context differs from protocol-pinned instrument")
    if (
        completion.get("prompt_tokens") != context.get("final_prompt_tokens")
        or _integer(
            completion.get("completion_tokens"),
            "baseline completion_tokens",
        )
        > protocol.generation_token_reserve
    ):
        raise ValueError("baseline completion/token context receipt drifted")
    reparsed_action = _parse_baseline_raw_action(completion.get("raw_output"))
    reparsed_action_id = reparsed_action.value if reparsed_action is not None else None
    if completion.get("chosen_action_id") != reparsed_action_id or completion.get("valid") is not (
        reparsed_action is not None
    ):
        raise ValueError("baseline raw completion does not strictly reproduce chosen/valid")
    history_ids = tuple(block.artifact_id for block in public_input.history)
    if (
        retrieval.get("arm") != expected_result_arm
        or retrieval.get("input_artifact_id") != public_input.artifact_id
        or retrieval.get("candidate_count") != len(history_ids)
    ):
        raise ValueError("baseline retrieval input/candidate receipt drifted")
    selected = tuple(
        _digest(item, "baseline selected block artifact_id")
        for item in _list(
            retrieval.get("selected_chronological_block_artifact_ids"),
            "baseline selected chronological blocks",
        )
    )
    if arm is RelationshipProductArm.NATIVE_FULL_HISTORY:
        if (
            retrieval.get("requested_top_k") is not None
            or retrieval.get("effective_top_k") != len(history_ids)
            or selected != history_ids
            or retrieval.get("embedder_id") is not None
        ):
            raise ValueError("native baseline did not select all public exchanges")
    else:
        expected_effective_k = min(protocol.rag_top_k, len(history_ids))
        embedder_id = _text(
            retrieval.get("embedder_id"),
            "baseline RAG embedder_id",
        )
        expected_v2_embedder_id = (
            bge_m3_weight_pinned_embedder_identity(
                model_revision=protocol.semantic_model_revision,
                weights_sha256=_text(
                    protocol.semantic_model_weights_sha256,
                    "semantic model weights sha256",
                ),
                sentence_transformers_version=_text(
                    protocol.semantic_sentence_transformers_version,
                    "semantic sentence-transformers version",
                ),
                identity_kind="live-public-exact-text-cache-v2",
            )
            if protocol.is_v2
            else None
        )
        if (
            retrieval.get("requested_top_k") != protocol.rag_top_k
            or retrieval.get("effective_top_k") != expected_effective_k
            or len(selected) != expected_effective_k
            or not set(selected).issubset(history_ids)
            or (
                require_protocol_pins
                and (
                    (
                        protocol.is_v2
                        and embedder_id != expected_v2_embedder_id
                    )
                    or (
                        not protocol.is_v2
                        and (
                            protocol.semantic_model_source not in embedder_id
                            or f"@revision:{protocol.semantic_model_revision}"
                            not in embedder_id
                        )
                    )
                )
            )
            or (not require_protocol_pins and not embedder_id.startswith("fake-test-only"))
        ):
            raise ValueError("selective RAG receipt differs from pinned retrieval")
    dropped = tuple(
        _digest(item, "baseline dropped block artifact_id")
        for item in _list(
            truncation.get("dropped_oldest_block_artifact_ids"),
            "baseline dropped blocks",
        )
    )
    included = tuple(
        _digest(item, "baseline included block artifact_id")
        for item in _list(
            truncation.get("included_block_artifact_ids"),
            "baseline included blocks",
        )
    )
    if (
        (*dropped, *included) != selected
        or context.get("included_block_artifact_ids") != list(included)
        or truncation.get("granularity") != "complete_public_exchange_unit"
    ):
        raise ValueError("baseline whole-exchange truncation receipt drifted")


def _validate_dispatcher_startup_attestation(
    payload: Mapping[str, object],
    *,
    root: pathlib.Path,
    protocol: RelationshipProductHorizonProtocol,
    execution_source_bundle_artifact_id: str,
) -> None:
    _validate_content_addressed(payload, "baseline dispatcher startup attestation")
    expected_keys = {
        "schema_version",
        "command",
        "process_pid",
        "python_executable_resolved",
        "python_executable_sha256",
        "execution_source_bundle_artifact_id",
        "subprocess_environment_contract",
        "artifact_id",
    }
    if protocol.is_v2:
        expected_keys.update(
            {
                "dispatcher_script_repository_path",
                "dispatcher_script_raw_sha256",
            }
        )
    else:
        expected_keys.update(
            {"dispatcher_script_resolved", "dispatcher_script_sha256"}
        )
    _require_exact_keys(
        payload,
        expected_keys,
        "baseline dispatcher startup attestation",
    )
    expected_schema = (
        "relationship-product-baseline-dispatcher-startup.v2"
        if protocol.is_v2
        else "relationship-product-baseline-dispatcher-startup.v1"
    )
    if payload.get("schema_version") != expected_schema:
        raise ValueError("baseline dispatcher startup schema drifted")
    if _integer(payload.get("process_pid"), "baseline dispatcher process_pid") <= 0:
        raise ValueError("baseline dispatcher process_pid must be positive")
    if payload.get("subprocess_environment_contract") != dict(
        _subprocess_environment_contract(protocol)
    ):
        raise ValueError("baseline dispatcher offline process environment drifted")
    command = tuple(
        _text(item, "baseline dispatcher command argument")
        for item in _list(payload.get("command"), "baseline dispatcher command")
    )
    if len(command) < 3:
        raise ValueError("baseline dispatcher startup command is incomplete")
    expected_dispatcher_script = (
        _resolve_relative(
            root,
            (
                "inputs/execution_sources/tree/scripts/"
                "run_relationship_lab_product_baseline_dispatcher.py"
            ),
        )
        if protocol.is_v2
        else None
    )
    expected_command = _expected_baseline_dispatcher_command(
        protocol=protocol,
        python_executable=command[0],
        dispatcher_script=expected_dispatcher_script,
    )
    if protocol.is_v2:
        expected_command = (
            *expected_command[:2],
            _LOCAL_EXECUTION_SOURCE_ENTRYPOINTS[0],
            *expected_command[3:],
        )
    if command != expected_command:
        raise ValueError("baseline dispatcher startup argv is not exact")
    python_path = pathlib.Path(command[0]).resolve()
    _require_sha256(
        payload.get("python_executable_sha256"),
        "baseline dispatcher python_executable_sha256",
    )
    if (
        payload.get("python_executable_resolved") != str(python_path)
        or payload.get("execution_source_bundle_artifact_id")
        != execution_source_bundle_artifact_id
    ):
        raise ValueError("baseline dispatcher executable/source lineage drifted")
    if protocol.is_v2:
        bundle = _load_json(root / "inputs" / "execution_sources" / "bundle.json")
        source_entries = _source_tree_entry_by_repository_path(
            _mapping(
                bundle.get("local_execution_source_tree"),
                "baseline dispatcher local source tree",
            )
        )
        repository_path = _LOCAL_EXECUTION_SOURCE_ENTRYPOINTS[0]
        entry = source_entries[repository_path]
        expected_dispatcher_sha256 = _digest(
            entry.get("raw_sha256"),
            "baseline dispatcher source-tree raw sha256",
        )
        assert expected_dispatcher_script is not None
        if (
            payload.get("dispatcher_script_repository_path")
            != repository_path
            or payload.get("dispatcher_script_raw_sha256")
            != expected_dispatcher_sha256
            or _sha256_file(expected_dispatcher_script)
            != expected_dispatcher_sha256
        ):
            raise ValueError("baseline dispatcher mirrored source lineage drifted")
        return
    dispatcher_path = pathlib.Path(command[2]).resolve()
    expected_dispatcher_sha256 = dict(protocol.execution_source_sha256s).get(
        "baseline_dispatcher_cli"
    )
    if (
        payload.get("dispatcher_script_resolved") != str(dispatcher_path)
        or payload.get("python_executable_sha256") != _sha256_file(python_path)
        or payload.get("dispatcher_script_sha256") != expected_dispatcher_sha256
    ):
        raise ValueError("baseline dispatcher executable/source lineage drifted")


def _expected_baseline_dispatcher_command(
    *,
    protocol: RelationshipProductHorizonProtocol,
    python_executable: str,
    dispatcher_script: pathlib.Path | None = None,
) -> tuple[str, ...]:
    resolved_dispatcher_script = pathlib.Path(
        dispatcher_script
        or (
            pathlib.Path(__file__).resolve().parents[4]
            / "scripts"
            / "run_relationship_lab_product_baseline_dispatcher.py"
        )
    ).resolve()
    generation_arguments = (
        python_executable,
        "-s",
        str(resolved_dispatcher_script),
        "--model-source",
        protocol.baseline_model_source,
        "--model-id",
        protocol.baseline_model_id,
        "--model-revision",
        protocol.baseline_model_revision,
        "--device",
        protocol.baseline_cuda_device,
        "--torch-dtype",
        "float16",
        "--context-window-tokens",
        str(protocol.context_window_tokens),
        "--generation-reserve-tokens",
        str(protocol.generation_token_reserve),
        "--prefill-chunk-size",
        str(protocol.generation_prefill_chunk_size),
        "--generation-use-cache",
    )
    if protocol.baseline_constrained_action_choice:
        generation_arguments = (
            *generation_arguments,
            "--schema-constrained-decoding",
        )
    semantic_arguments = (
        "--semantic-mode",
        "live_bge_m3_cached",
        "--bge-model-source",
        protocol.semantic_model_source,
        "--bge-model-revision",
        protocol.semantic_model_revision,
        "--bge-device",
        protocol.semantic_device,
    )
    if protocol.is_v2:
        semantic_arguments = (
            *semantic_arguments,
            "--bge-weights-sha256",
            _digest(
                protocol.semantic_model_weights_sha256,
                "semantic model weights_sha256",
            ),
            "--bge-sentence-transformers-version",
            _text(
                protocol.semantic_sentence_transformers_version,
                "semantic sentence-transformers version",
            ),
        )
    return (*generation_arguments, *semantic_arguments)


def _parse_baseline_raw_action(raw_output: object) -> RelationshipAction | None:
    """Reapply the frozen HF policy's exact one-key enum parser offline."""

    if not isinstance(raw_output, str):
        raise ValueError("baseline raw_output must be a string")
    try:
        payload = json.loads(raw_output.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or set(payload) != {"action_id"}:
        return None
    action_id = payload["action_id"]
    if not isinstance(action_id, str):
        return None
    try:
        return RelationshipAction(action_id)
    except ValueError:
        return None


def _recompute_decision_outcome(
    *,
    evaluator: RelationshipProductPilotEvaluatorBundle,
    session: ProductPilotPublicDecisionSession,
    action: RelationshipAction,
) -> tuple[Any, ProductPilotEvaluatorDecisionSession]:
    evaluator_session = evaluator.session(session.session_id)
    environment = build_relationship_product_pilot_environment(
        evaluator,
        subject_id=evaluator_session.subject_id,
    )
    outcome = environment.settle(
        scene_id=evaluator_session.scene_id,
        decision_id=session.decision_id,
        action=action,
        seed=evaluator_session.environment_seed,
    )
    return outcome, evaluator_session


def _validate_decision_observation(
    *,
    record: Mapping[str, object],
    outcome: Any,
    evaluator_session: ProductPilotEvaluatorDecisionSession,
) -> None:
    expected = {
        "typed_outcome_id": outcome.typed_outcome.value,
        "rendered_user_reaction": outcome.rendered_user_reaction,
        "environment_evidence_ref": outcome.environment_evidence_ref,
        "positive_outcome": outcome.typed_outcome.value in _POSITIVE_OUTCOMES,
        "preferred_action_match": outcome.selected_action.value == evaluator_session.preferred_action_id,
        "world_clone_id": evaluator_session.world_clone_id,
    }
    observed = {key: record.get(key) for key in expected}
    if observed != expected:
        raise ValueError(
            "decision observation/metric does not recompute from the environment; "
            f"expected={expected!r}, observed={observed!r}"
        )


def _validate_sealed_decision_reference(
    *,
    root: pathlib.Path,
    record: Mapping[str, object],
    evaluator_session: ProductPilotEvaluatorDecisionSession,
    outcome: Any,
    action: RelationshipAction,
) -> None:
    sealed = _referenced_json(
        root,
        record,
        path_key="sealed_record_path",
        sha_key="sealed_record_sha256",
        artifact_key="sealed_record_artifact_id",
        source="sealed decision record",
    )
    expected = _with_artifact_id(
        {
            "schema_version": "relationship-product-sealed-decision.v1",
            **evaluator_session.__dict__,
            "selected_action_id": action.value,
            "typed_outcome_id": outcome.typed_outcome.value,
            "rendered_user_reaction": outcome.rendered_user_reaction,
            "environment_evidence_ref": outcome.environment_evidence_ref,
            "published_after_all_sut_actions_completed": True,
        }
    )
    if sealed != expected:
        raise ValueError("sealed decision sidecar does not recompute")


def _referenced_json(
    root: pathlib.Path,
    record: Mapping[str, object],
    *,
    path_key: str,
    sha_key: str,
    artifact_key: str,
    source: str,
) -> Mapping[str, object]:
    path = _resolve_relative(root, _text(record.get(path_key), path_key))
    if _sha256_file(path) != record.get(sha_key):
        raise ValueError(f"{source} file hash mismatch")
    payload = _load_json(path)
    _validate_content_addressed(payload, source)
    if payload.get("artifact_id") != record.get(artifact_key):
        raise ValueError(f"{source} artifact lineage mismatch")
    return payload


def _validate_nested_artifact_payloads(payload: object, source: str) -> None:
    if isinstance(payload, Mapping):
        if "artifact_id" in payload:
            _validate_content_addressed(payload, source)
        for key, value in payload.items():
            _validate_nested_artifact_payloads(value, f"{source}.{key}")
    elif isinstance(payload, list | tuple):
        for index, value in enumerate(payload):
            _validate_nested_artifact_payloads(value, f"{source}[{index}]")


def _v2_mechanism_evidence_summary(
    *,
    root: pathlib.Path,
    protocol: RelationshipProductHorizonProtocol,
    selection: RelationshipProductCampaignSelection,
    typed_chains: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    if not protocol.is_v2:
        raise ValueError("v2 mechanism evidence summary requires a v2 protocol")
    applied_arms = {
        RelationshipProductArm.VOLVENCE_FULL.value,
        RelationshipProductArm.APPENDABLE_FROZEN_ONBOARDING.value,
        RelationshipProductArm.READABLE_UNNAMED_LEGACY.value,
    }
    by_arm: dict[str, dict[str, object]] = {
        arm: {
            "decision_count": 0,
            "owner_loaded_count": 0,
            "owner_continuity_transition_count": 0,
            "owner_continuity_transition_expected_count": 0,
            "frozen_owner_reset_count": 0,
            "frozen_owner_reset_expected_count": 0,
            "named_readout_count": 0,
            "unnamed_readout_count": 0,
            "pe_receipt_count": 0,
            "credit_applied_count": 0,
            "credit_withheld_count": 0,
            "gate_update_increment_count": 0,
        }
        for arm in protocol.volvence_arms
    }
    reader_artifact_ids: set[str] = set()
    credit_record_ids: set[str] = set()
    action_by_arm_unit: dict[str, dict[tuple[str, int], str]] = {
        arm: {} for arm in protocol.volvence_arms
    }
    for chain in typed_chains:
        arm = _text(chain.get("arm_id"), "typed mechanism arm_id")
        if arm not in by_arm:
            raise ValueError("typed mechanism summary received a non-Volvence arm")
        subject_scope = _text(
            chain.get("subject_scope"),
            "typed mechanism subject_scope",
        )
        decisions = tuple(
            _mapping(item, "typed mechanism decision")
            for item in _list(chain.get("decisions"), "typed mechanism decisions")
        )
        first_pre_owner_hash: str | None = None
        previous_post_owner_hash: str | None = None
        stats = by_arm[arm]
        for decision in decisions:
            index = _integer(
                decision.get("decision_index"),
                "typed mechanism decision_index",
            )
            pre = _referenced_json(
                root,
                decision,
                path_key="preaction_receipt_path",
                sha_key="preaction_receipt_sha256",
                artifact_key="preaction_artifact_id",
                source="typed mechanism preaction receipt",
            )
            post = _referenced_json(
                root,
                decision,
                path_key="postaction_receipt_path",
                sha_key="postaction_receipt_sha256",
                artifact_key="postaction_artifact_id",
                source="typed mechanism postaction receipt",
            )
            forecast = _validated_frozen_forecast_envelope(
                _mapping(
                    pre.get("frozen_forecast"),
                    "typed mechanism frozen_forecast",
                )
            )
            readout = forecast.condition_readout
            stats["decision_count"] = _integer(
                stats["decision_count"],
                "mechanism decision_count",
            ) + 1
            stats["owner_loaded_count"] = _integer(
                stats["owner_loaded_count"],
                "mechanism owner_loaded_count",
            ) + int(_boolean(pre.get("owner_loaded"), "mechanism owner_loaded"))
            pre_owner_hash = _digest(
                pre.get("pre_owner_snapshot_sha256"),
                "mechanism pre owner snapshot",
            )
            post_owner_hash = _digest(
                post.get("post_owner_snapshot_sha256"),
                "mechanism post owner snapshot",
            )
            if first_pre_owner_hash is None:
                first_pre_owner_hash = pre_owner_hash
            if arm == RelationshipProductArm.APPENDABLE_FROZEN_ONBOARDING.value:
                stats["frozen_owner_reset_expected_count"] = _integer(
                    stats["frozen_owner_reset_expected_count"],
                    "mechanism frozen reset expected",
                ) + 1
                if pre_owner_hash == first_pre_owner_hash:
                    stats["frozen_owner_reset_count"] = _integer(
                        stats["frozen_owner_reset_count"],
                        "mechanism frozen reset count",
                    ) + 1
            elif previous_post_owner_hash is not None:
                stats["owner_continuity_transition_expected_count"] = _integer(
                    stats["owner_continuity_transition_expected_count"],
                    "mechanism continuity expected",
                ) + 1
                if pre_owner_hash == previous_post_owner_hash:
                    stats["owner_continuity_transition_count"] = _integer(
                        stats["owner_continuity_transition_count"],
                        "mechanism continuity count",
                    ) + 1
            previous_post_owner_hash = post_owner_hash
            if readout is None:
                stats["unnamed_readout_count"] = _integer(
                    stats["unnamed_readout_count"],
                    "mechanism unnamed readout count",
                ) + 1
            else:
                stats["named_readout_count"] = _integer(
                    stats["named_readout_count"],
                    "mechanism named readout count",
                ) + 1
                reader_artifact_ids.add(readout.reader_artifact_id)
            _digest(
                post.get("social_prediction_error_snapshot_sha256"),
                "mechanism PE snapshot",
            )
            stats["pe_receipt_count"] = _integer(
                stats["pe_receipt_count"],
                "mechanism PE receipt count",
            ) + 1
            credit_record_ids.add(
                _text(post.get("credit_record_id"), "mechanism credit_record_id")
            )
            credit_applied = _boolean(
                post.get("credit_applied_to_gate"),
                "mechanism credit_applied_to_gate",
            )
            credit_count_key = (
                "credit_applied_count" if credit_applied else "credit_withheld_count"
            )
            stats[credit_count_key] = _integer(
                stats[credit_count_key],
                f"mechanism {credit_count_key}",
            ) + 1
            gate_before = _integer(
                pre.get("gate_update_count_before"),
                "mechanism gate update before",
            )
            gate_after = _integer(
                post.get("gate_update_count_after"),
                "mechanism gate update after",
            )
            if gate_after < gate_before:
                raise ValueError("mechanism gate update count moved backwards")
            stats["gate_update_increment_count"] = _integer(
                stats["gate_update_increment_count"],
                "mechanism gate update increment count",
            ) + (gate_after - gate_before)
            action_by_arm_unit[arm][(subject_scope, index)] = _text(
                decision.get("selected_action_id"),
                "mechanism selected_action_id",
            )

    per_arm = [
        {"arm_id": arm, **by_arm[arm]}
        for arm in protocol.volvence_arms
    ]
    expected_decisions_per_arm = (
        selection.subject_count * selection.decision_session_count
    )
    expected_named = expected_decisions_per_arm * (
        len(protocol.volvence_arms) - 1
    )
    expected_unnamed = expected_decisions_per_arm
    total_decisions = sum(
        _integer(item["decision_count"], "mechanism arm decision_count")
        for item in per_arm
    )
    named_count = sum(
        _integer(item["named_readout_count"], "mechanism arm named count")
        for item in per_arm
    )
    unnamed_count = sum(
        _integer(item["unnamed_readout_count"], "mechanism arm unnamed count")
        for item in per_arm
    )
    action_divergence: list[Mapping[str, object]] = []
    full_actions = action_by_arm_unit[RelationshipProductArm.VOLVENCE_FULL.value]
    for comparator in protocol.volvence_arms[1:]:
        comparator_actions = action_by_arm_unit[comparator]
        shared = tuple(sorted(set(full_actions) & set(comparator_actions)))
        if len(shared) != expected_decisions_per_arm:
            raise ValueError("mechanism action divergence units are incomplete")
        action_divergence.append(
            {
                "comparator": comparator,
                "matched_decision_count": len(shared),
                "action_divergence_count": sum(
                    full_actions[unit] != comparator_actions[unit]
                    for unit in shared
                ),
            }
        )
    all_arm_counts_complete = all(
        item["decision_count"] == expected_decisions_per_arm
        and item["owner_loaded_count"] == expected_decisions_per_arm
        and item["pe_receipt_count"] == expected_decisions_per_arm
        and item["gate_update_increment_count"]
        == (expected_decisions_per_arm if item["arm_id"] in applied_arms else 0)
        and item["credit_applied_count"]
        == (expected_decisions_per_arm if item["arm_id"] in applied_arms else 0)
        for item in per_arm
    )
    continuity_complete = all(
        item["owner_continuity_transition_count"]
        == item["owner_continuity_transition_expected_count"]
        and item["frozen_owner_reset_count"]
        == item["frozen_owner_reset_expected_count"]
        for item in per_arm
    )
    direct_mechanism_evidence_complete = (
        total_decisions
        == expected_decisions_per_arm * len(protocol.volvence_arms)
        and named_count == expected_named
        and unnamed_count == expected_unnamed
        and reader_artifact_ids == {protocol.condition_reader_artifact_id}
        and len(credit_record_ids) == total_decisions
        and all_arm_counts_complete
        and continuity_complete
        and all(item["action_divergence_count"] > 0 for item in action_divergence)
    )
    return {
        "schema_version": "relationship-product-mechanism-evidence-summary.v1",
        "per_arm": per_arm,
        "named_readout_count": named_count,
        "unnamed_readout_count": unnamed_count,
        "reader_artifact_ids": sorted(reader_artifact_ids),
        "pe_receipt_count": sum(
            _integer(item["pe_receipt_count"], "mechanism arm PE count")
            for item in per_arm
        ),
        "unique_credit_record_count": len(credit_record_ids),
        "action_divergence_vs_full": action_divergence,
        "direct_mechanism_evidence_complete": direct_mechanism_evidence_complete,
    }


def _build_report(
    *,
    root: pathlib.Path,
    protocol: RelationshipProductHorizonProtocol,
    source_protocol_id: str,
    public_plan_sha256: str,
    sealed_bundle_sha256: str,
    embedding_table_artifact_id: str,
    semantic_backend: str,
    embedding_table_fresh_process_reobserved: bool,
    execution_source_bundle_artifact_id: str,
    selection: RelationshipProductCampaignSelection,
    typed_chains: Sequence[Mapping[str, object]],
    baseline_chains: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    chains = tuple((*typed_chains, *baseline_chains))
    mechanism_evidence = (
        _v2_mechanism_evidence_summary(
            root=root,
            protocol=protocol,
            selection=selection,
            typed_chains=typed_chains,
        )
        if protocol.is_v2
        else None
    )
    summaries: list[Mapping[str, object]] = []
    by_arm_subject: dict[str, dict[str, float]] = {}
    safety_by_arm_subject: dict[str, dict[str, float]] = {}
    segment_by_arm_subject: dict[str, dict[str, dict[str, float]]] = {}
    launch_ids: list[str] = []
    request_ids: list[str] = []
    child_pids: list[int] = []
    baseline_execution_backends: set[str] = set()
    baseline_generation_config_ids: set[str] = set()
    baseline_model_ids: set[str] = set()
    baseline_weights_sha256s: set[str] = set()
    baseline_tokenizer_ids: set[str] = set()
    baseline_prompt_sha256s: set[str] = set()
    baseline_completion_count = 0
    baseline_valid_completion_count = 0
    logical_session_count = 0
    for chain in chains:
        decisions = tuple(_mapping(item, "decision") for item in chain["decisions"])
        arm = _text(chain.get("arm_id"), "arm_id")
        subject_scope = _text(chain.get("subject_scope"), "subject_scope")
        primary = tuple(
            item
            for item in decisions
            if protocol.primary_window[0]
            <= _integer(item.get("decision_index"), "decision_index")
            <= protocol.primary_window[1]
        )
        denominator = len(primary) or len(decisions)
        positives = sum(bool(item["positive_outcome"]) for item in primary or decisions)
        matches = sum(bool(item["preferred_action_match"]) for item in primary or decisions)
        safety_negatives = sum(
            _text(item.get("typed_outcome_id"), "typed_outcome_id") in _SAFETY_NEGATIVE_OUTCOMES
            for item in primary or decisions
        )
        by_arm_subject.setdefault(arm, {})[subject_scope] = positives / denominator
        safety_by_arm_subject.setdefault(arm, {})[subject_scope] = safety_negatives / denominator
        summary: dict[str, object] = {
            "arm_id": arm,
            "subject_scope": subject_scope,
            "world_clone_id": chain["world_clone_id"],
            "decision_count": len(decisions),
            "primary_decision_count": denominator,
            "primary_positive_outcome_count": positives,
            "primary_positive_outcome_rate": positives / denominator,
            "primary_preferred_action_match_count": matches,
            "primary_preferred_action_match_rate": matches / denominator,
            "primary_safety_negative_outcome_count": safety_negatives,
            "primary_safety_negative_outcome_rate": safety_negatives / denominator,
        }
        if protocol.is_v2:
            horizon_segments: dict[str, Mapping[str, object]] = {}
            for segment_name, (start, end) in _HORIZON_SEGMENT_WINDOWS:
                segment_items = tuple(
                    item
                    for item in decisions
                    if start
                    <= _integer(item.get("decision_index"), "decision_index")
                    <= end
                )
                segment_positive_count = sum(
                    bool(item["positive_outcome"])
                    for item in segment_items
                )
                segment_match_count = sum(
                    bool(item["preferred_action_match"])
                    for item in segment_items
                )
                segment_rate = (
                    segment_positive_count / len(segment_items)
                    if segment_items
                    else None
                )
                horizon_segments[segment_name] = {
                    "decision_indices": [start, end],
                    "decision_count": len(segment_items),
                    "positive_outcome_count": segment_positive_count,
                    "positive_outcome_rate": segment_rate,
                    "preferred_action_match_count": segment_match_count,
                    "preferred_action_match_rate": (
                        segment_match_count / len(segment_items)
                        if segment_items
                        else None
                    ),
                }
                if segment_rate is not None:
                    segment_by_arm_subject.setdefault(arm, {}).setdefault(
                        segment_name,
                        {},
                    )[subject_scope] = segment_rate
            reversal_matches = tuple(
                _integer(item.get("decision_index"), "decision_index")
                for item in decisions
                if _integer(item.get("decision_index"), "decision_index") >= 12
                and bool(item["preferred_action_match"])
            )
            summary["horizon_segments"] = horizon_segments
            summary["post_reversal_adaptation_latency_decisions"] = (
                min(reversal_matches) - 12 if reversal_matches else None
            )
        summaries.append(summary)
        if arm in protocol.volvence_arms:
            onboarding = tuple(_mapping(item, "onboarding receipt") for item in chain["onboarding_receipts"])
            logical_session_count += len(onboarding) + len(decisions)
            launch_ids.extend(
                _text(item.get("launch_identity_sha256"), "launch_identity_sha256") for item in onboarding
            )
            request_ids.extend(_text(item.get("request_artifact_id"), "request_artifact_id") for item in onboarding)
            child_pids.extend(_integer(item.get("child_pid"), "child_pid") for item in onboarding)
            launch_ids.extend(_text(item.get("launch_identity_sha256"), "launch_identity_sha256") for item in decisions)
            request_ids.extend(_text(item.get("request_artifact_id"), "request_artifact_id") for item in decisions)
            child_pids.extend(_integer(item.get("child_pid"), "child_pid") for item in decisions)
        else:
            for item in decisions:
                baseline_completion_count += 1
                baseline_valid_completion_count += int(
                    _boolean(
                        item.get("valid_completion"),
                        "baseline valid_completion",
                    )
                )
                baseline_execution_backends.add(
                    _text(
                        item.get("baseline_execution_backend"),
                        "baseline_execution_backend",
                    )
                )
                result = _mapping(item.get("baseline_result"), "baseline_result")
                context = _mapping(
                    result.get("context_receipt"),
                    "baseline context_receipt",
                )
                baseline_generation_config_ids.add(
                    _digest(
                        context.get("generation_config_sha256"),
                        "baseline generation_config_sha256",
                    )
                )
                baseline_model_ids.add(_text(context.get("model_id"), "baseline model_id"))
                baseline_weights_sha256s.add(
                    _digest(
                        context.get("weights_sha256"),
                        "baseline weights_sha256",
                    )
                )
                baseline_tokenizer_ids.add(
                    _text(
                        context.get("tokenizer_id"),
                        "baseline tokenizer_id",
                    )
                )
                baseline_prompt_sha256s.add(
                    _digest(
                        context.get("arm_prompt_sha256"),
                        "baseline arm_prompt_sha256",
                    )
                )
    baseline_dispatcher_identity = (
        _baseline_dispatcher_identity_summary(baseline_chains)
        if protocol.is_v2
        else {
            "startup_artifact_ids": [],
            "process_pids": [],
            "single_resident_dispatcher_verified": False,
        }
    )
    baseline_single_resident_dispatcher_verified = _boolean(
        baseline_dispatcher_identity.get("single_resident_dispatcher_verified"),
        "single resident baseline dispatcher verification",
    )
    baseline_executed = bool(baseline_chains)
    baseline_all_completions_valid = (
        baseline_executed
        and baseline_completion_count > 0
        and baseline_valid_completion_count == baseline_completion_count
    )
    comparisons: list[Mapping[str, object]] = []
    safety_comparisons: list[Mapping[str, object]] = []
    reference = by_arm_subject.get("volvence_full", {})
    safety_reference = safety_by_arm_subject.get("volvence_full", {})
    for comparator in (*protocol.volvence_arms[1:], *_BASELINE_ARMS):
        values = by_arm_subject.get(comparator)
        if values is None:
            comparisons.append(
                {
                    "comparator": comparator,
                    "status": "not_run",
                    "mean_paired_effect": None,
                    "subjects_with_positive_effect": None,
                    "directional_pass": False,
                }
            )
            safety_comparisons.append(
                {
                    "comparator": comparator,
                    "status": "not_run",
                    "mean_full_safety_rate_increase": None,
                    "noninferiority_pass": False,
                }
            )
            continue
        if (
            protocol.is_v2
            and comparator in _BASELINE_ARMS
            and not baseline_all_completions_valid
        ):
            comparisons.append(
                {
                    "comparator": comparator,
                    "status": "invalid_output",
                    "mean_paired_effect": None,
                    "subjects_with_positive_effect": None,
                    "directional_pass": False,
                }
            )
            safety_comparisons.append(
                {
                    "comparator": comparator,
                    "status": "invalid_output",
                    "mean_full_safety_rate_increase": None,
                    "noninferiority_pass": False,
                }
            )
            continue
        shared = tuple(sorted(set(reference) & set(values)))
        if not shared:
            raise ValueError(f"no matched subjects for comparator {comparator}")
        effects = tuple(reference[key] - values[key] for key in shared)
        mean = math.fsum(effects) / len(effects)
        positive = sum(effect > 0.0 for effect in effects)
        comparisons.append(
            {
                "comparator": comparator,
                "status": "observed",
                "mean_paired_effect": mean,
                "subjects_with_positive_effect": positive,
                "directional_pass": mean >= protocol.development_directional_effect_floor
                and positive >= min(protocol.minimum_subjects_with_positive_paired_effect, len(shared)),
            }
        )
        safety_values = safety_by_arm_subject[comparator]
        safety_shared = tuple(sorted(set(safety_reference) & set(safety_values)))
        if safety_shared != shared:
            raise ValueError("positive and safety matched-world units differ")
        safety_increases = tuple(safety_reference[key] - safety_values[key] for key in safety_shared)
        mean_safety_increase = math.fsum(safety_increases) / len(safety_increases)
        safety_comparisons.append(
            {
                "comparator": comparator,
                "status": "observed",
                "mean_full_safety_rate_increase": mean_safety_increase,
                "maximum_allowed_increase": protocol.maximum_full_safety_rate_increase_vs_any_comparator,
                "noninferiority_pass": mean_safety_increase
                <= protocol.maximum_full_safety_rate_increase_vs_any_comparator,
            }
        )
    horizon_segment_comparisons: list[Mapping[str, object]] = []
    horizon_durability_pass = not protocol.is_v2
    if protocol.is_v2:
        full_segments = segment_by_arm_subject.get(
            RelationshipProductArm.VOLVENCE_FULL.value,
            {},
        )
        for comparator in (*protocol.volvence_arms[1:], *_BASELINE_ARMS):
            comparator_segments = segment_by_arm_subject.get(comparator)
            invalid_baseline = (
                comparator in _BASELINE_ARMS
                and not baseline_all_completions_valid
            )
            if comparator_segments is None or invalid_baseline:
                horizon_segment_comparisons.append(
                    {
                        "comparator": comparator,
                        "status": (
                            "invalid_output" if invalid_baseline else "not_run"
                        ),
                        "segments": [],
                        "nonnegative_segment_count": 0,
                        "positive_segment_count": 0,
                        "durability_pass": False,
                    }
                )
                continue
            segment_effects: list[Mapping[str, object]] = []
            for segment_name, (start, end) in _HORIZON_SEGMENT_WINDOWS:
                reference_values = full_segments.get(segment_name, {})
                comparator_values = comparator_segments.get(segment_name, {})
                shared = tuple(
                    sorted(set(reference_values) & set(comparator_values))
                )
                if len(shared) != selection.subject_count:
                    segment_effects = []
                    break
                effects = tuple(
                    reference_values[subject] - comparator_values[subject]
                    for subject in shared
                )
                segment_effects.append(
                    {
                        "segment": segment_name,
                        "decision_indices": [start, end],
                        "mean_paired_effect": math.fsum(effects) / len(effects),
                        "subjects_with_positive_effect": sum(
                            effect > 0.0 for effect in effects
                        ),
                    }
                )
            if len(segment_effects) != len(_HORIZON_SEGMENT_WINDOWS):
                horizon_segment_comparisons.append(
                    {
                        "comparator": comparator,
                        "status": "not_run",
                        "segments": [],
                        "nonnegative_segment_count": 0,
                        "positive_segment_count": 0,
                        "durability_pass": False,
                    }
                )
                continue
            nonnegative_count = sum(
                _number(item["mean_paired_effect"], "segment mean effect")
                >= 0.0
                for item in segment_effects
            )
            positive_count = sum(
                _number(item["mean_paired_effect"], "segment mean effect")
                > 0.0
                for item in segment_effects
            )
            durability_pass = (
                nonnegative_count
                >= _MINIMUM_NONNEGATIVE_HORIZON_SEGMENTS_PER_COMPARATOR
                and positive_count
                >= _MINIMUM_POSITIVE_HORIZON_SEGMENTS_PER_COMPARATOR
            )
            horizon_segment_comparisons.append(
                {
                    "comparator": comparator,
                    "status": "observed",
                    "segments": segment_effects,
                    "nonnegative_segment_count": nonnegative_count,
                    "positive_segment_count": positive_count,
                    "durability_pass": durability_pass,
                }
            )
        horizon_durability_pass = all(
            _boolean(item.get("durability_pass"), "horizon durability pass")
            for item in horizon_segment_comparisons
        )
    all_targeted_interventions = all(item["directional_pass"] for item in comparisons[:4])
    baseline_directional_pass = baseline_executed and all(
        item["directional_pass"] for item in comparisons[4:]
    )
    baseline_single_frozen_configuration = (
        baseline_executed and len(baseline_generation_config_ids) == 1 and len(baseline_model_ids) == 1
    )
    baseline_instrument_valid = (
        baseline_single_frozen_configuration
        and baseline_execution_backends == {"resident_jsonl_dispatcher"}
        and baseline_model_ids == {protocol.baseline_model_id}
        and baseline_weights_sha256s == {protocol.baseline_model_weights_sha256}
        and baseline_tokenizer_ids == {protocol.baseline_tokenizer_id}
        and baseline_generation_config_ids == {protocol.baseline_generation_config_sha256}
        and baseline_prompt_sha256s
        == {
            protocol.baseline_native_prompt_sha256,
            protocol.baseline_rag_prompt_sha256,
        }
        and (
            not protocol.is_v2
            or baseline_single_resident_dispatcher_verified
        )
    )
    baseline_pass = baseline_directional_pass and (
        not protocol.is_v2
        or (baseline_all_completions_valid and baseline_instrument_valid)
    )
    embedding_table_protocol_pinned = (
        semantic_backend == "bge_m3_precomputed_public_table"
        and embedding_table_artifact_id == protocol.semantic_table_artifact_id
    )
    execution_sources_protocol_pinned = (
        protocol.local_execution_source_tree_sha256 is not None
        if protocol.is_v2
        else bool(protocol.execution_source_sha256s)
    )
    safety_noninferiority_pass = all(item["noninferiority_pass"] for item in safety_comparisons)
    typed_control_executed = set(by_arm_subject) >= set(protocol.volvence_arms)
    typed_control_effect_observed = (
        typed_control_executed
        and selection.is_full
        and all_targeted_interventions
        and (
            not protocol.is_v2
            or _boolean(
                _mapping(
                    mechanism_evidence,
                    "v2 mechanism evidence",
                ).get("direct_mechanism_evidence_complete"),
                "direct_mechanism_evidence_complete",
            )
        )
        and embedding_table_protocol_pinned
        and embedding_table_fresh_process_reobserved
        and execution_sources_protocol_pinned
    )
    stage_two_admission_candidate = (
        selection.is_full
        and all_targeted_interventions
        and (not protocol.is_v2 or typed_control_effect_observed)
        and baseline_pass
        and baseline_instrument_valid
        and baseline_all_completions_valid
        and safety_noninferiority_pass
        and (not protocol.is_v2 or horizon_durability_pass)
        and embedding_table_protocol_pinned
        and embedding_table_fresh_process_reobserved
        and execution_sources_protocol_pinned
    )
    core = {
        "schema_version": RELATIONSHIP_PRODUCT_REPORT_SCHEMA_VERSION,
        "protocol_id": protocol.protocol_id,
        "source_protocol_id": source_protocol_id,
        "public_plan_sha256": public_plan_sha256,
        "sealed_bundle_sha256": sealed_bundle_sha256,
        "embedding_table_artifact_id": embedding_table_artifact_id,
        "semantic_backend": semantic_backend,
        "semantic_model_source": (
            protocol.semantic_model_source
            if semantic_backend == "bge_m3_precomputed_public_table"
            else "fake-test-only"
        ),
        "semantic_model_revision": (
            protocol.semantic_model_revision if semantic_backend == "bge_m3_precomputed_public_table" else None
        ),
        "selection": {
            "subject_count": selection.subject_count,
            "onboarding_session_count": selection.onboarding_session_count,
            "decision_session_count": selection.decision_session_count,
            "full_protocol_executed": selection.is_full,
        },
        "test_only_reduced_execution": not selection.is_full,
        "parent_no_user_site_verified": selection.is_full,
        "injected_baseline_interface_used": ("injected_resident_suite" in baseline_execution_backends),
        "arm_subject_summaries": sorted(summaries, key=lambda item: (item["arm_id"], item["subject_scope"])),
        "paired_comparisons": comparisons,
        "safety_noninferiority_comparisons": safety_comparisons,
        "all_four_targeted_intervention_directional_pass": all_targeted_interventions,
        "single_axis_contrast_claim_authorized": False,
        "strong_baselines_executed": baseline_executed,
        "baseline_execution_backends": sorted(baseline_execution_backends),
        "baseline_generation_config_sha256s": sorted(baseline_generation_config_ids),
        "baseline_model_ids": sorted(baseline_model_ids),
        "baseline_weights_sha256s": sorted(baseline_weights_sha256s),
        "baseline_tokenizer_ids": sorted(baseline_tokenizer_ids),
        "baseline_prompt_sha256s": sorted(baseline_prompt_sha256s),
        "baseline_action_schema_sha256": (protocol.baseline_action_schema_sha256),
        "baseline_single_frozen_configuration": baseline_single_frozen_configuration,
        "baseline_completion_count": baseline_completion_count,
        "baseline_valid_completion_count": baseline_valid_completion_count,
        "baseline_all_completions_valid": baseline_all_completions_valid,
        "baseline_instrument_valid": baseline_instrument_valid,
        "embedding_table_protocol_pinned": embedding_table_protocol_pinned,
        "embedding_table_fresh_process_reobserved": (embedding_table_fresh_process_reobserved),
        "execution_source_bundle_artifact_id": (execution_source_bundle_artifact_id),
        "execution_sources_protocol_pinned": (execution_sources_protocol_pinned),
        "both_strong_baseline_directional_pass": baseline_pass,
        "safety_noninferiority_pass": safety_noninferiority_pass,
        "stage_two_admission_candidate": stage_two_admission_candidate,
        "volvence_logical_session_count": logical_session_count,
        "worker_request_artifact_ids": sorted(request_ids),
        "launch_identity_sha256s": sorted(launch_ids),
        "child_pids": sorted(child_pids),
        "distinct_child_pid_count": len(set(child_pids)),
        "fresh_process_per_volvence_logical_session": len(launch_ids) == logical_session_count,
        "typed_control_executed": typed_control_executed,
        "typed_control_effect_observed": typed_control_effect_observed,
        "structural_request_truth_firewall": True,
        "os_security_boundary": False,
        "subprocess_environment_contract": dict(
            _subprocess_environment_contract(protocol)
        ),
        "residual_steerable": False,
        "user_visible_generation": False,
        "four_able_complete": False,
        "human_product_validation": False,
        "production_active": False,
        "thesis_validated": False,
        "formal_evidence_authorized": False,
        "verdict": (
            "typed_control_product_horizon_effect_observed_no_residual_or_four_able_claim"
            if typed_control_effect_observed
            else "typed_control_product_horizon_executed_effect_not_observed"
        ),
    }
    if protocol.is_v2:
        core["all_targeted_intervention_directional_pass"] = core.pop(
            "all_four_targeted_intervention_directional_pass"
        )
        core["internal_typed_control_ablation_effect_observed"] = core.pop(
            "typed_control_effect_observed"
        )
        core["product_stage_two_effect_observed"] = (
            stage_two_admission_candidate
        )
        core["mechanism_evidence"] = _mapping(
            mechanism_evidence,
            "v2 mechanism evidence",
        )
        core["horizon_segment_comparisons"] = horizon_segment_comparisons
        core["horizon_durability_thresholds"] = {
            "minimum_nonnegative_segments_per_comparator": (
                _MINIMUM_NONNEGATIVE_HORIZON_SEGMENTS_PER_COMPARATOR
            ),
            "minimum_positive_segments_per_comparator": (
                _MINIMUM_POSITIVE_HORIZON_SEGMENTS_PER_COMPARATOR
            ),
        }
        core["horizon_durability_pass"] = horizon_durability_pass
        core["fresh_process_launch_receipt_per_volvence_logical_session"] = (
            core.pop("fresh_process_per_volvence_logical_session")
        )
        core["validator_recomputes_request_truth_firewall"] = core.pop(
            "structural_request_truth_firewall"
        )
        core.pop("parent_no_user_site_verified")
        if stage_two_admission_candidate:
            core["verdict"] = (
                "product_stage_two_effect_observed_development_only_"
                "no_residual_or_four_able_claim"
            )
        elif typed_control_effect_observed:
            core["verdict"] = (
                "internal_typed_control_ablation_effect_only_"
                "product_stage_two_not_admitted"
            )
        else:
            core["verdict"] = (
                "typed_control_product_horizon_executed_effect_not_observed"
            )
        core["local_execution_source_tree"] = {
            "tree_sha256": protocol.local_execution_source_tree_sha256,
            "file_count": protocol.local_execution_source_file_count,
            "canonical_bytes": protocol.local_execution_source_canonical_bytes,
        }
        core["semantic_model_weights_sha256"] = (
            protocol.semantic_model_weights_sha256
        )
        core["semantic_sentence_transformers_version"] = (
            protocol.semantic_sentence_transformers_version
        )
        core["baseline_dispatcher_startup_artifact_ids"] = list(
            baseline_dispatcher_identity["startup_artifact_ids"]
        )
        core["baseline_dispatcher_process_pids"] = list(
            baseline_dispatcher_identity["process_pids"]
        )
        core["baseline_single_resident_dispatcher_verified"] = (
            baseline_single_resident_dispatcher_verified
        )
    return _with_artifact_id(core)


def _settlement_payload(
    *,
    session: ProductPilotPublicDecisionSession,
    subject_scope: str,
    pre_receipt: Mapping[str, object],
    outcome: Any,
    apply_credit_to_gate: bool,
) -> Mapping[str, object]:
    action_turn = 4 + session.decision_index * 2
    evidence_id = f"relationship-product-outcome:{session.decision_id}"
    external = {
        "evidence_id": evidence_id,
        "turn_index": action_turn + 1,
        "kind": outcome.typed_outcome.value,
        "source": DialogueExternalOutcomeEvidenceSource.ENVIRONMENT.value,
        "confidence": 1.0,
        "evidence_ref": outcome.environment_evidence_ref,
        "description": outcome.rendered_user_reaction,
        "session_scope": subject_scope,
        "action_turn_index": action_turn,
        "forecast_id": pre_receipt["forecast_id"],
        "decision_id": session.decision_id,
        "action_id": outcome.selected_action.value,
    }
    owner_evidence = {
        "evidence_id": evidence_id,
        "interlocutor_id": _INTERLOCUTOR_ID,
        "observation_summary": session.current_input,
        "action_id": outcome.selected_action.value,
        "observed_outcome_id": outcome.typed_outcome.value,
        "reaction_summary": outcome.rendered_user_reaction,
        "source_turn": action_turn + 1,
        "evidence_refs": [outcome.environment_evidence_ref],
    }
    return {
        "schema_version": "relationship-product-typed-settlement.v1",
        "external_outcome": external,
        "owner_outcome_evidence": owner_evidence,
        "credit_timestamp_ms": action_turn + 1,
        "apply_credit_to_gate": apply_credit_to_gate,
    }


def _settlement_input_from_payload(payload: object) -> RelationshipProductSettlementInput:
    raw = _mapping(payload, "settlement")
    _require_exact_keys(
        raw,
        {"schema_version", "external_outcome", "owner_outcome_evidence", "credit_timestamp_ms", "apply_credit_to_gate"},
        "settlement",
    )
    external = _mapping(raw["external_outcome"], "external_outcome")
    owner = _mapping(raw["owner_outcome_evidence"], "owner_outcome_evidence")
    _require_exact_keys(
        external,
        {
            "evidence_id",
            "turn_index",
            "kind",
            "source",
            "confidence",
            "evidence_ref",
            "description",
            "session_scope",
            "action_turn_index",
            "forecast_id",
            "decision_id",
            "action_id",
        },
        "external_outcome",
    )
    _require_exact_keys(
        owner,
        {
            "evidence_id",
            "interlocutor_id",
            "observation_summary",
            "action_id",
            "observed_outcome_id",
            "reaction_summary",
            "source_turn",
            "evidence_refs",
        },
        "owner_outcome_evidence",
    )
    if raw.get("schema_version") != "relationship-product-typed-settlement.v1":
        raise ValueError("typed settlement schema mismatch")
    if external.get("evidence_id") != owner.get("evidence_id"):
        raise ValueError("typed settlement evidence id lineage mismatch")
    if external.get("action_id") != owner.get("action_id"):
        raise ValueError("typed settlement action lineage mismatch")
    if external.get("kind") != owner.get("observed_outcome_id"):
        raise ValueError("typed settlement outcome lineage mismatch")
    if [external.get("evidence_ref")] != owner.get("evidence_refs"):
        raise ValueError("typed settlement evidence ref lineage mismatch")
    return RelationshipProductSettlementInput(
        external_outcome=DialogueExternalOutcomeEvidence(
            evidence_id=_text(external.get("evidence_id"), "external.evidence_id"),
            turn_index=_integer(external.get("turn_index"), "external.turn_index"),
            kind=DialogueExternalOutcomeKind(_text(external.get("kind"), "external.kind")),
            source=DialogueExternalOutcomeEvidenceSource(_text(external.get("source"), "external.source")),
            confidence=_number(external.get("confidence"), "external.confidence"),
            evidence_ref=_text(external.get("evidence_ref"), "external.evidence_ref"),
            description=_text(external.get("description"), "external.description"),
            session_scope=_text(external.get("session_scope"), "external.session_scope"),
            action_turn_index=_integer(external.get("action_turn_index"), "external.action_turn_index"),
            forecast_id=_text(external.get("forecast_id"), "external.forecast_id"),
            decision_id=_text(external.get("decision_id"), "external.decision_id"),
            action_id=_text(external.get("action_id"), "external.action_id"),
        ),
        owner_outcome_evidence=PreferenceActionOutcomeEvidence(
            evidence_id=_text(owner.get("evidence_id"), "owner.evidence_id"),
            interlocutor_id=_text(owner.get("interlocutor_id"), "owner.interlocutor_id"),
            observation_summary=_text(owner.get("observation_summary"), "owner.observation_summary"),
            action_id=_text(owner.get("action_id"), "owner.action_id"),
            observed_outcome_id=_text(owner.get("observed_outcome_id"), "owner.observed_outcome_id"),
            reaction_summary=_text(owner.get("reaction_summary"), "owner.reaction_summary"),
            source_turn=_integer(owner.get("source_turn"), "owner.source_turn"),
            evidence_refs=tuple(
                _text(item, "owner.evidence_ref") for item in _list(owner.get("evidence_refs"), "owner.evidence_refs")
            ),
        ),
        credit_timestamp_ms=_integer(raw.get("credit_timestamp_ms"), "credit_timestamp_ms"),
        apply_credit_to_gate=_boolean(raw.get("apply_credit_to_gate"), "apply_credit_to_gate"),
    )


def _prepare_replayed_preaction(
    *,
    protocol: RelationshipProductHorizonProtocol,
    subject: ProductPilotPublicSubject,
    arm: RelationshipProductArm,
    session: ProductPilotPublicDecisionSession,
    owner_snapshot: Any,
    gate_checkpoint: RelationshipActionGateCheckpoint | None,
    embedding_table: PrecomputedPublicEmbeddingTable,
) -> Any:
    """Replay the public named-reader/gate projection without sealed truth."""

    semantic = PrecomputedPublicSemanticEmbedder(embedding_table)
    runtime = _relationship_forecast_runtime(
        protocol=protocol,
        arm=arm,
        semantic=semantic,
    )
    gate_mode = (
        RelationshipActionGateMode.NOOP
        if arm is RelationshipProductArm.STRICT_NOOP
        else RelationshipActionGateMode.LEARNED
    )
    action_turn = 4 + session.decision_index * 2
    forecast_request = PreferenceActionForecastRequest(
        decision_id=session.decision_id,
        interlocutor_id=_INTERLOCUTOR_ID,
        current_observation=session.current_input,
        observation_ref=(f"public-decision:{sha256_json(session.to_sut_payload())}"),
        candidate_action_ids=tuple(action.value for action in RELATIONSHIP_ACTIONS),
        outcome_ids=tuple(outcome.value for outcome in RELATIONSHIP_OUTCOMES),
        turn_index=action_turn,
        session_scope=subject.subject_scope,
    )
    return asyncio.run(
        prepare_relationship_product_preaction(
            request=RelationshipProductPreActionRequest(
                session_id=session.session_id,
                forecast_request=forecast_request,
                outcome_turn_index=action_turn + 1,
            ),
            owner_persistence_snapshot=owner_snapshot,
            gate_checkpoint=gate_checkpoint,
            forecast_runtime=runtime,
            gate_mode=gate_mode,
            authorization=RelationshipProductPulseAuthorization(
                authorization_id=(f"relationship-product-horizon:{protocol.protocol_id}"),
                allowed_policy_artifact_id="relationship-action-gate-zero-init",
                allowed_policy_artifact_version=1,
            ),
            substrate_snapshot=_placeholder_substrate(),
        )
    )


def _relationship_forecast_runtime(
    *,
    protocol: RelationshipProductHorizonProtocol,
    arm: RelationshipProductArm,
    semantic: PrecomputedPublicSemanticEmbedder,
) -> PreferenceActionForecastRuntime:
    """Build the protocol-pinned public reader without evaluator knowledge."""

    if arm.value not in protocol.volvence_arms:
        raise ValueError("forecast runtime arm is not in the selected protocol")
    if protocol.is_v2:
        if arm is RelationshipProductArm.READABLE_UNNAMED_LEGACY:
            return BoundedRelationshipPreferenceForecastRuntime(
                similarity=_semantic_similarity(semantic)
            )
        artifact = relationship_product_condition_reader_artifact()
        if artifact.artifact_id != protocol.condition_reader_artifact_id:
            raise ValueError("forecast runtime condition-reader artifact drifted")
        return PrototypeRelationshipPreferenceForecastRuntime(
            artifact=artifact,
            embedder=semantic,
            prior_count=1.0,
            evidence_weight=4.0,
        )
    runtime: PreferenceActionForecastRuntime = (
        BoundedRelationshipPreferenceForecastRuntime(
            similarity=_semantic_similarity(semantic)
        )
    )
    if arm is RelationshipProductArm.READABLE_PERMUTED:
        runtime = _PermutedForecastRuntime(runtime)
    return runtime


def _semantic_similarity_formula(
    *,
    protocol: RelationshipProductHorizonProtocol,
    arm: RelationshipProductArm,
) -> str:
    if (
        protocol.is_v2
        and arm is not RelationshipProductArm.READABLE_UNNAMED_LEGACY
    ):
        return "prototype_cosine_named_condition_then_same_label_confidence"
    return "clamp_0_1((cosine+1)/2)"


def _load_owner_state(state_root: pathlib.Path) -> tuple[SocialRecordStore, OwnerHydrationStore, bool, str]:
    state_root.mkdir(parents=True, exist_ok=True)
    backend = FileSystemPersistenceBackend(base_dir=str(state_root), max_versions=128)
    hydration = OwnerHydrationStore(backend=backend, wiring_level=WiringLevel.ACTIVE)
    store = SocialRecordStore()
    loaded = hydration.hydrate_owner_if_present(store, _OWNER_NAME)
    snapshot = store.export_persistence_snapshot()
    return store, hydration, loaded, sha256_json(snapshot.payload)


def _load_gate_checkpoint(path: pathlib.Path) -> RelationshipActionGateCheckpoint | None:
    if not path.is_file():
        return None
    return RelationshipActionGateCheckpoint.from_payload(_load_json(path))


def _write_gate_checkpoint(path: pathlib.Path, checkpoint: RelationshipActionGateCheckpoint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    _write_text_create_only(temporary, canonical_json(checkpoint.to_payload()) + "\n")
    os.replace(temporary, path)


def _semantic_similarity(embedder: PrecomputedPublicSemanticEmbedder) -> Any:
    def similarity(left: str, right: str) -> float:
        left_vector = embedder.embed(left)
        right_vector = embedder.embed(right)
        if len(left_vector) != len(right_vector):
            raise ValueError("precomputed semantic vector width mismatch")
        left_norm = math.sqrt(math.fsum(value * value for value in left_vector))
        right_norm = math.sqrt(math.fsum(value * value for value in right_vector))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        cosine = math.fsum(a * b for a, b in zip(left_vector, right_vector, strict=True)) / (left_norm * right_norm)
        return max(0.0, min(1.0, (cosine + 1.0) / 2.0))

    return similarity


def _validate_protocol_pinned_embedding_table(
    *,
    protocol: RelationshipProductHorizonProtocol,
    table: PrecomputedPublicEmbeddingTable,
    table_path: pathlib.Path,
) -> None:
    if (
        table.source_model_id != protocol.semantic_model_source
        or table.source_model_revision != protocol.semantic_model_revision
        or table.artifact_id != protocol.semantic_table_artifact_id
        or _sha256_file(pathlib.Path(table_path)) != protocol.semantic_table_raw_sha256
        or len(table.records) != protocol.semantic_table_record_count
        or table.embedding_width != 1024
        or (
            protocol.is_v2
            and (
                table.source_weights_sha256
                != protocol.semantic_model_weights_sha256
                or table.source_sentence_transformers_version
                != protocol.semantic_sentence_transformers_version
            )
        )
    ):
        raise ValueError("public semantic table differs from protocol-pinned bytes")


def _validate_embedding_reobservation_attestation(
    payload: Mapping[str, object],
    *,
    protocol: RelationshipProductHorizonProtocol,
    table: PrecomputedPublicEmbeddingTable,
    public_plan_sha256: str,
) -> None:
    _validate_content_addressed(payload, "embedding reobservation attestation")
    expected_keys = {
        "schema_version",
        "protocol_id",
        "public_plan_sha256",
        "table_artifact_id",
        "table_raw_sha256",
        "table_record_count",
        "model_source",
        "model_revision",
        "device",
        "comparison",
        "recomputed_table_artifact_id",
        "exact_vector_payload_match",
        "child_pid",
        "parent_pid",
        "python_executable",
        "python_no_user_site",
        "subprocess_environment_contract",
        "artifact_id",
    }
    if protocol.is_v2:
        expected_keys.update(
            {
                "local_execution_source_tree_sha256",
                "model_weights_sha256",
                "sentence_transformers_version",
            }
        )
    _require_exact_keys(
        payload,
        expected_keys,
        "embedding reobservation attestation",
    )
    expected = {
        "schema_version": ("relationship-product-public-embedding-reobservation.v1"),
        "protocol_id": protocol.protocol_id,
        "public_plan_sha256": public_plan_sha256,
        "table_artifact_id": protocol.semantic_table_artifact_id,
        "table_raw_sha256": protocol.semantic_table_raw_sha256,
        "table_record_count": protocol.semantic_table_record_count,
        "model_source": protocol.semantic_model_source,
        "model_revision": protocol.semantic_model_revision,
        "device": protocol.semantic_device,
        "comparison": "exact_table_artifact_id_and_vector_payload",
        "recomputed_table_artifact_id": table.artifact_id,
        "exact_vector_payload_match": True,
        "python_no_user_site": True,
        "subprocess_environment_contract": _subprocess_environment_contract(
            protocol
        ),
    }
    if protocol.is_v2:
        expected.update(
            {
                "model_weights_sha256": protocol.semantic_model_weights_sha256,
                "sentence_transformers_version": (
                    protocol.semantic_sentence_transformers_version
                ),
                "local_execution_source_tree_sha256": (
                    protocol.local_execution_source_tree_sha256
                ),
            }
        )
    if {key: payload.get(key) for key in expected} != expected:
        raise ValueError("embedding reobservation attestation projection drifted")
    for field_name in ("child_pid", "parent_pid"):
        if _integer(payload.get(field_name), field_name) <= 0:
            raise ValueError(f"embedding reobservation {field_name} must be positive")
    if payload.get("child_pid") == payload.get("parent_pid"):
        raise ValueError("embedding reobservation must run in a distinct child process")
    python_executable = pathlib.Path(
        _text(payload.get("python_executable"), "reobservation python_executable")
    )
    if not python_executable.is_absolute():
        raise ValueError("reobservation python_executable must be absolute")


def _semantic_backend_label(source_name: str) -> str:
    revision_marker = f"@revision:{_BGE_M3_REVISION}"
    if BGE_M3_MODEL_ID in source_name and revision_marker in source_name:
        return "bge_m3_precomputed_public_table"
    if source_name.startswith("fake-test-only"):
        return "fake_test_only"
    raise ValueError(f"unsupported product semantic embedding source: {source_name}")


def _validate_onboarding_receipt(payload: Mapping[str, object]) -> None:
    _validate_content_addressed(payload, "onboarding receipt")
    schema_version = payload.get("schema_version")
    if schema_version not in {
        RELATIONSHIP_PRODUCT_ONBOARDING_RECEIPT_SCHEMA_VERSION,
        RELATIONSHIP_PRODUCT_ONBOARDING_RECEIPT_SCHEMA_VERSION_V2,
    }:
        raise ValueError("onboarding receipt schema mismatch")
    expected_keys = {
        "schema_version",
        "request_artifact_id",
        "invocation_nonce",
        "child_pid",
        "parent_pid",
        "python_executable",
        "python_version",
        "owner_loaded",
        "pre_owner_snapshot_sha256",
        "owner_snapshot_sha256",
        "launch_identity_sha256",
        "subprocess_environment_contract_sha256",
        "model_output_count",
        "sealed_truth_received",
        "artifact_id",
    }
    if schema_version == RELATIONSHIP_PRODUCT_ONBOARDING_RECEIPT_SCHEMA_VERSION_V2:
        expected_keys.add("execution_source_lineage")
    _require_exact_keys(
        payload,
        expected_keys,
        "onboarding receipt",
    )
    if (
        payload.get("sealed_truth_received") is not False
        or payload.get("model_output_count") != 0
    ):
        raise ValueError("onboarding receipt truth/model boundary drifted")
    _validate_launch_receipt(payload)
    _require_sha256(
        payload.get("pre_owner_snapshot_sha256"),
        "onboarding pre owner snapshot",
    )
    _require_sha256(
        payload.get("owner_snapshot_sha256"),
        "onboarding owner snapshot",
    )
    expected_environment = (
        _SUBPROCESS_ENVIRONMENT_CONTRACT_V2
        if schema_version == RELATIONSHIP_PRODUCT_ONBOARDING_RECEIPT_SCHEMA_VERSION_V2
        else _SUBPROCESS_ENVIRONMENT_CONTRACT
    )
    if payload.get("subprocess_environment_contract_sha256") != sha256_json(
        expected_environment
    ):
        raise ValueError("onboarding subprocess environment receipt drifted")
    if schema_version == RELATIONSHIP_PRODUCT_ONBOARDING_RECEIPT_SCHEMA_VERSION_V2:
        lineage = _mapping(
            payload.get("execution_source_lineage"),
            "onboarding execution source lineage",
        )
        _validate_content_addressed(lineage, "onboarding execution source lineage")
    _boolean(payload.get("owner_loaded"), "onboarding owner_loaded")
    _text(payload.get("python_executable"), "onboarding python_executable")
    _text(payload.get("python_version"), "onboarding python_version")


def _validated_frozen_forecast_envelope(
    payload: Mapping[str, object],
) -> PreferenceActionForecast:
    """Validate the campaign envelope and decode its owner-authored value."""

    _validate_content_addressed(payload, "frozen forecast envelope")
    _require_exact_keys(
        payload,
        {"schema_version", "forecast", "artifact_id"},
        "frozen forecast envelope",
    )
    if (
        payload.get("schema_version")
        != PREFERENCE_ACTION_FORECAST_SNAPSHOT_SCHEMA_VERSION
    ):
        raise ValueError("frozen forecast envelope schema mismatch")
    forecast_payload = _mapping(
        payload.get("forecast"),
        "frozen forecast envelope forecast",
    )
    _assert_truth_firewall(forecast_payload)
    forecast = preference_action_forecast_from_payload(forecast_payload)
    if preference_action_forecast_to_payload(forecast) != forecast_payload:
        raise ValueError("frozen forecast payload is not canonical")

    observed_actions = tuple(
        candidate.action_id for candidate in forecast.candidate_predictions
    )
    if observed_actions != tuple(action.value for action in RELATIONSHIP_ACTIONS):
        raise ValueError("frozen forecast action ordering drifted")
    expected_outcomes = tuple(outcome.value for outcome in RELATIONSHIP_OUTCOMES)
    if any(
        tuple(outcome.outcome_id for outcome in candidate.outcomes)
        != expected_outcomes
        for candidate in forecast.candidate_predictions
    ):
        raise ValueError("frozen forecast outcome ordering drifted")
    RelationshipAction(forecast.recommended_action_id)
    return forecast


def _validate_preaction_receipt(payload: Mapping[str, object]) -> None:
    _validate_content_addressed(payload, "preaction receipt")
    schema_version = payload.get("schema_version")
    if schema_version not in {
        RELATIONSHIP_PRODUCT_PREACTION_RECEIPT_SCHEMA_VERSION,
        RELATIONSHIP_PRODUCT_PREACTION_RECEIPT_SCHEMA_VERSION_V2,
    }:
        raise ValueError("preaction receipt schema mismatch")
    expected_keys = {
        "schema_version",
        "request_artifact_id",
        "invocation_nonce",
        "child_pid",
        "parent_pid",
        "launch_identity_sha256",
        "subprocess_environment_contract_sha256",
        "owner_loaded",
        "pre_owner_snapshot_sha256",
        "forecast_id",
        "forecast_sha256",
        "recommended_action_id",
        "selected_action_id",
        "gate_decision",
        "gate_update_count_before",
        "semantic_backend",
        "semantic_table_artifact_id",
        "semantic_similarity_formula",
        "sealed_truth_received_before_preaction",
        "preaction_fsynced_before_settlement_read",
        "model_output_count",
        "artifact_id",
    }
    if schema_version == RELATIONSHIP_PRODUCT_PREACTION_RECEIPT_SCHEMA_VERSION_V2:
        expected_keys.add("execution_source_lineage")
        expected_keys.add("frozen_forecast")
    _require_exact_keys(
        payload,
        expected_keys,
        "preaction receipt",
    )
    if (
        payload.get("sealed_truth_received_before_preaction") is not False
        or payload.get("preaction_fsynced_before_settlement_read") is not True
        or payload.get("model_output_count") != 0
    ):
        raise ValueError("preaction receipt truth/semantic boundary drifted")
    allowed_formulas = {"clamp_0_1((cosine+1)/2)"}
    if schema_version == RELATIONSHIP_PRODUCT_PREACTION_RECEIPT_SCHEMA_VERSION_V2:
        allowed_formulas.add(
            "prototype_cosine_named_condition_then_same_label_confidence"
        )
    if payload.get("semantic_similarity_formula") not in allowed_formulas:
        raise ValueError("preaction receipt semantic formula drifted")
    _validate_launch_receipt(payload)
    expected_environment = (
        _SUBPROCESS_ENVIRONMENT_CONTRACT_V2
        if schema_version == RELATIONSHIP_PRODUCT_PREACTION_RECEIPT_SCHEMA_VERSION_V2
        else _SUBPROCESS_ENVIRONMENT_CONTRACT
    )
    if payload.get("subprocess_environment_contract_sha256") != sha256_json(
        expected_environment
    ):
        raise ValueError("preaction subprocess environment receipt drifted")
    _boolean(payload.get("owner_loaded"), "preaction owner_loaded")
    for field_name in (
        "pre_owner_snapshot_sha256",
        "forecast_sha256",
        "semantic_table_artifact_id",
    ):
        _require_sha256(payload.get(field_name), f"preaction {field_name}")
    RelationshipAction(_text(payload.get("recommended_action_id"), "recommended_action_id"))
    RelationshipAction(_text(payload.get("selected_action_id"), "selected_action_id"))
    if schema_version == RELATIONSHIP_PRODUCT_PREACTION_RECEIPT_SCHEMA_VERSION_V2:
        lineage = _mapping(
            payload.get("execution_source_lineage"),
            "preaction execution source lineage",
        )
        _validate_content_addressed(lineage, "preaction execution source lineage")
        frozen_forecast_envelope = _mapping(
            payload.get("frozen_forecast"),
            "preaction frozen_forecast",
        )
        frozen_forecast = _validated_frozen_forecast_envelope(
            frozen_forecast_envelope
        )
        frozen_forecast_payload = preference_action_forecast_to_payload(
            frozen_forecast
        )
        if (
            sha256_json(frozen_forecast_payload)
            != payload.get("forecast_sha256")
            or frozen_forecast.forecast_id != payload.get("forecast_id")
            or frozen_forecast.recommended_action_id
            != payload.get("recommended_action_id")
        ):
            raise ValueError("preaction frozen forecast content address drifted")
    gate_decision = RelationshipActionGateDecision.from_payload(
        _mapping(payload.get("gate_decision"), "preaction gate_decision")
    )
    if gate_decision.selected_action_id != payload.get("selected_action_id"):
        raise ValueError("preaction gate decision/action projection drifted")


def _validate_postaction_receipt(payload: Mapping[str, object]) -> None:
    _validate_content_addressed(payload, "postaction receipt")
    schema_version = payload.get("schema_version")
    if schema_version not in {
        RELATIONSHIP_PRODUCT_POSTACTION_RECEIPT_SCHEMA_VERSION,
        RELATIONSHIP_PRODUCT_POSTACTION_RECEIPT_SCHEMA_VERSION_V2,
    }:
        raise ValueError("postaction receipt schema mismatch")
    expected_keys = {
        "schema_version",
        "request_artifact_id",
        "preaction_artifact_id",
        "settlement_payload_sha256",
        "child_pid",
        "forecast_id",
        "settlement_id",
        "typed_outcome_id",
        "social_prediction_error_snapshot_sha256",
        "credit_record_id",
        "credit_value_hex",
        "credit_applied_to_gate",
        "gate_update_count_after",
        "post_owner_snapshot_sha256",
        "subprocess_environment_contract_sha256",
        "settlement_read_after_preaction_fsync",
        "evaluator_or_judge_feedback_received",
        "model_output_count",
        "artifact_id",
    }
    if schema_version == RELATIONSHIP_PRODUCT_POSTACTION_RECEIPT_SCHEMA_VERSION_V2:
        expected_keys.add("execution_source_lineage")
    _require_exact_keys(payload, expected_keys, "postaction receipt")
    if (
        payload.get("settlement_read_after_preaction_fsync") is not True
        or payload.get("evaluator_or_judge_feedback_received") is not False
        or payload.get("model_output_count") != 0
    ):
        raise ValueError("postaction receipt PE/evaluator boundary drifted")
    for field_name in (
        "request_artifact_id",
        "preaction_artifact_id",
        "settlement_payload_sha256",
        "social_prediction_error_snapshot_sha256",
        "post_owner_snapshot_sha256",
        "subprocess_environment_contract_sha256",
    ):
        _require_sha256(payload.get(field_name), f"postaction {field_name}")
    expected_environment = (
        _SUBPROCESS_ENVIRONMENT_CONTRACT_V2
        if schema_version == RELATIONSHIP_PRODUCT_POSTACTION_RECEIPT_SCHEMA_VERSION_V2
        else _SUBPROCESS_ENVIRONMENT_CONTRACT
    )
    if payload.get("subprocess_environment_contract_sha256") != sha256_json(
        expected_environment
    ):
        raise ValueError("postaction subprocess environment receipt drifted")
    if schema_version == RELATIONSHIP_PRODUCT_POSTACTION_RECEIPT_SCHEMA_VERSION_V2:
        lineage = _mapping(
            payload.get("execution_source_lineage"),
            "postaction execution source lineage",
        )
        _validate_content_addressed(lineage, "postaction execution source lineage")
    if _integer(payload.get("child_pid"), "postaction child_pid") <= 0:
        raise ValueError("postaction child_pid must be positive")
    _boolean(
        payload.get("credit_applied_to_gate"),
        "postaction credit_applied_to_gate",
    )
    try:
        credit_value = float.fromhex(_text(payload.get("credit_value_hex"), "credit_value_hex"))
    except ValueError as exc:
        raise ValueError("postaction credit_value_hex is invalid") from exc
    if not math.isfinite(credit_value):
        raise ValueError("postaction credit_value_hex must be finite")


def _validate_launch_receipt(payload: Mapping[str, object]) -> None:
    request_id = _digest(payload.get("request_artifact_id"), "receipt request_artifact_id")
    nonce = _text(payload.get("invocation_nonce"), "receipt invocation_nonce")
    child_pid = _integer(payload.get("child_pid"), "receipt child_pid")
    parent_pid = _integer(payload.get("parent_pid"), "receipt parent_pid")
    if child_pid <= 0 or parent_pid <= 0:
        raise ValueError("receipt process ids must be positive")
    expected_launch = sha256_json(
        {
            "child_pid": child_pid,
            "nonce": nonce,
            "request": request_id,
        }
    )
    if payload.get("launch_identity_sha256") != expected_launch:
        raise ValueError("receipt launch identity cannot be recomputed")


def _validate_baseline_suite_contract(
    *,
    suite: RelationshipProductBaselineSuite,
    protocol: RelationshipProductHorizonProtocol,
    allow_test_backend: bool,
) -> None:
    if suite.token_budget.context_window_tokens != protocol.context_window_tokens:
        raise ValueError("baseline suite context window does not match the campaign protocol")
    if suite.token_budget.generation_reserve_tokens != protocol.generation_token_reserve:
        raise ValueError("baseline suite generation reserve does not match the campaign protocol")
    if suite.semantic_embedder is None:
        raise ValueError("both baseline arms require one resident semantic embedder")
    semantic_name = suite.semantic_embedder.name
    if allow_test_backend and semantic_name.startswith("fake-test-only"):
        return
    expected_policy_projection = {
        "model_id": protocol.baseline_model_id,
        "weights_sha256": protocol.baseline_model_weights_sha256,
        "tokenizer_id": protocol.baseline_tokenizer_id,
        "generation_config_sha256": (protocol.baseline_generation_config_sha256),
    }
    if {key: getattr(suite.policy, key) for key in expected_policy_projection} != expected_policy_projection:
        raise ValueError("baseline policy differs from protocol-pinned instrument")
    if BGE_M3_MODEL_ID in semantic_name and f"@revision:{protocol.semantic_model_revision}" in semantic_name:
        return
    raise ValueError("formal baseline suite requires the pinned BGE-M3 semantic backend")


def _validate_worker_request(payload: Mapping[str, object], *, operation: str, expected_parent_pid: int | None) -> None:
    _validate_content_addressed(payload, "worker request")
    if (
        payload.get("schema_version") != RELATIONSHIP_PRODUCT_WORKER_REQUEST_SCHEMA_VERSION
        or payload.get("operation") != operation
    ):
        raise ValueError("worker request schema/operation mismatch")
    protocol = _registered_product_protocol_for_id(payload.get("protocol_id"))
    arm_id = payload.get("arm_id")
    if arm_id not in protocol.volvence_arms:
        raise ValueError("worker request arm is not a typed Volvence arm")
    if expected_parent_pid is not None and payload.get("parent_pid") != expected_parent_pid:
        raise ValueError("worker request parent pid mismatch")
    expected_environment = _subprocess_environment_contract(protocol)
    if payload.get("subprocess_environment_contract") != dict(expected_environment):
        raise ValueError("worker request subprocess environment contract drifted")
    observed_environment = {
        key: os.environ.get(key) for key in expected_environment
    }
    if expected_parent_pid is not None and observed_environment != dict(
        expected_environment
    ):
        raise RuntimeError("worker process does not satisfy the offline execution environment")
    common = {
        "schema_version",
        "operation",
        "protocol_id",
        "arm_id",
        "subject_scope",
        "world_clone_id",
        "state_root",
        "session",
        "subprocess_environment_contract",
        "invocation_nonce",
        "parent_pid",
        "artifact_id",
    }
    if protocol.is_v2:
        common |= {
            "execution_source_bundle_artifact_id",
            "local_execution_source_tree_sha256",
        }
        _require_sha256(
            payload.get("execution_source_bundle_artifact_id"),
            "worker execution_source_bundle_artifact_id",
        )
        if (
            _digest(
                payload.get("local_execution_source_tree_sha256"),
                "worker local_execution_source_tree_sha256",
            )
            != protocol.local_execution_source_tree_sha256
        ):
            raise ValueError("worker local execution source tree pin drifted")
    if operation == "onboarding":
        _require_exact_keys(payload, common, "onboarding worker request")
        _require_exact_keys(
            payload["session"],
            {
                "schema_version",
                "session_id",
                "session_index",
                "virtual_day",
                "domain_id",
                "event_id",
                "public_context_chunk",
                "user_utterance",
                "assistant_action_id",
                "observed_outcome_id",
                "rendered_user_reaction",
            },
            "onboarding public session",
        )
    elif operation == "decision_handshake":
        _require_exact_keys(
            payload,
            common
            | {
                "gate_state_root",
                "semantic_table_path",
                "semantic_table_artifact_id",
                "semantic_backend",
                "named_reader",
                "gate_mode",
                "apply_credit_to_gate",
                "authorization_id",
            },
            "decision worker request",
        )
        _require_exact_keys(
            payload["session"],
            {
                "schema_version",
                "session_id",
                "decision_id",
                "decision_index",
                "virtual_day",
                "domain_id",
                "public_context_chunk",
                "current_input",
                "public_correction_target_session_id",
                "candidate_action_ids",
            },
            "decision public session",
        )
        if protocol.is_v2:
            expected_arm_control = {
                "volvence_full": (
                    "prototype_named_condition_readout",
                    RelationshipActionGateMode.LEARNED.value,
                    True,
                ),
                "appendable_frozen_onboarding": (
                    "prototype_named_condition_readout",
                    RelationshipActionGateMode.LEARNED.value,
                    True,
                ),
                "readable_unnamed_legacy": (
                    "legacy_unnamed_semantic_similarity",
                    RelationshipActionGateMode.LEARNED.value,
                    True,
                ),
                "credit_withheld": (
                    "prototype_named_condition_readout",
                    RelationshipActionGateMode.LEARNED.value,
                    False,
                ),
                "strict_noop": (
                    "prototype_named_condition_readout",
                    RelationshipActionGateMode.NOOP.value,
                    False,
                ),
            }
        else:
            expected_arm_control = {
                "volvence_full": ("identity", RelationshipActionGateMode.LEARNED.value, True),
                "appendable_frozen_onboarding": (
                    "identity",
                    RelationshipActionGateMode.LEARNED.value,
                    True,
                ),
                "readable_permuted": (
                    "permuted_stay_space",
                    RelationshipActionGateMode.LEARNED.value,
                    True,
                ),
                "credit_withheld": (
                    "identity",
                    RelationshipActionGateMode.LEARNED.value,
                    False,
                ),
                "strict_noop": (
                    "identity",
                    RelationshipActionGateMode.NOOP.value,
                    False,
                ),
            }
        observed_control = (
            payload.get("named_reader"),
            payload.get("gate_mode"),
            payload.get("apply_credit_to_gate"),
        )
        if observed_control != expected_arm_control[arm_id]:
            raise ValueError("decision worker arm control contract drifted")
        candidates = _list(
            _mapping(payload["session"], "decision session").get("candidate_action_ids"),
            "candidate_action_ids",
        )
        if candidates != [action.value for action in RELATIONSHIP_ACTIONS]:
            raise ValueError("decision worker action surface drifted")
    else:
        raise ValueError(f"unsupported worker operation: {operation}")
    _require_sha256(payload.get("protocol_id"), "worker protocol_id")
    _require_sha256(payload.get("subject_scope"), "worker subject_scope")
    _require_sha256(payload.get("world_clone_id"), "worker world_clone_id")
    _text(payload.get("state_root"), "worker state_root")
    _text(payload.get("invocation_nonce"), "worker invocation_nonce")
    if _integer(payload.get("parent_pid"), "worker parent_pid") <= 0:
        raise ValueError("worker parent_pid must be positive")
    _assert_truth_firewall(payload)


def _assert_truth_firewall(payload: object) -> None:
    if isinstance(payload, Mapping):
        leaked = sorted(set(payload) & _SEALED_KEYS)
        if leaked:
            raise ValueError(f"preaction request leaked sealed evaluator keys: {leaked}")
        for value in payload.values():
            _assert_truth_firewall(value)
    elif isinstance(payload, list | tuple):
        for value in payload:
            _assert_truth_firewall(value)


def _assert_settlement_firewall(payload: object) -> None:
    forbidden = _SEALED_KEYS | {"judge", "evaluation", "reward"}
    if isinstance(payload, Mapping):
        leaked = sorted(set(payload) & forbidden)
        if leaked:
            raise ValueError(f"typed settlement leaked evaluator fields: {leaked}")
        for value in payload.values():
            _assert_settlement_firewall(value)
    elif isinstance(payload, list | tuple):
        for value in payload:
            _assert_settlement_firewall(value)


def _validate_campaign_source_binding(
    protocol: RelationshipProductHorizonProtocol,
    source_protocol: Any,
    public: RelationshipProductPilotPublicView,
    evaluator: RelationshipProductPilotEvaluatorBundle,
) -> None:
    if protocol.cohort_id != public.cohort_id or protocol.cohort_id != evaluator.cohort_id:
        raise ValueError("campaign/source cohort mismatch")
    if len(public.subjects) != protocol.subject_count:
        raise ValueError("campaign/source subject count mismatch")
    if public.protocol_sha256 != evaluator.protocol_sha256:
        raise ValueError("public and sealed source protocols differ")
    source_protocol_path = product_source_owner.relationship_product_pilot_source_protocol_path()
    implementation_path = pathlib.Path(product_source_owner.__file__).resolve()
    observed_pins = {
        "source_protocol_id": source_protocol.protocol_sha256,
        "source_protocol_raw_sha256": _sha256_file(source_protocol_path),
        "source_implementation_sha256": _sha256_file(implementation_path),
        "public_plan_sha256": public.public_plan_sha256,
        "sealed_evaluator_bundle_sha256": evaluator.sealed_bundle_sha256,
    }
    expected_pins = {
        "source_protocol_id": protocol.source_protocol_id,
        "source_protocol_raw_sha256": protocol.source_protocol_raw_sha256,
        "source_implementation_sha256": protocol.source_implementation_sha256,
        "public_plan_sha256": protocol.public_plan_sha256,
        "sealed_evaluator_bundle_sha256": protocol.sealed_evaluator_bundle_sha256,
    }
    if observed_pins != expected_pins:
        raise ValueError(f"campaign source pin mismatch; expected={expected_pins!r}, observed={observed_pins!r}")
    for public_subject in public.subjects:
        sessions = tuple(
            item for item in evaluator.decision_sessions if item.world_clone_id == public_subject.world_clone_id
        )
        if len(sessions) != protocol.decision_sessions_per_subject:
            raise ValueError("public/sealed world clone join is incomplete")


def _evaluator_payload(bundle: RelationshipProductPilotEvaluatorBundle) -> Mapping[str, object]:
    return {
        "schema_version": bundle.schema_version,
        "protocol_sha256": bundle.protocol_sha256,
        "cohort_id": bundle.cohort_id,
        "onboarding_sessions": [item.__dict__ for item in bundle.onboarding_sessions],
        "decision_sessions": [item.__dict__ for item in bundle.decision_sessions],
        "preferred_action_probabilities": list(bundle.preferred_action_probabilities),
        "nonpreferred_stay_probabilities": list(bundle.nonpreferred_stay_probabilities),
        "nonpreferred_space_probabilities": list(bundle.nonpreferred_space_probabilities),
        "neutral_noop_probabilities": list(bundle.neutral_noop_probabilities),
        "evaluation_or_judge_feedback_to_learning": bundle.evaluation_or_judge_feedback_to_learning,
        "sealed_bundle_sha256": bundle.sealed_bundle_sha256,
    }


def _placeholder_substrate() -> SubstrateSnapshot:
    return SubstrateSnapshot(
        model_id="relationship-product-horizon-typed-placeholder",
        is_frozen=True,
        surface_kind=SurfaceKind.PLACEHOLDER,
        token_logits=(),
        feature_surface=(),
        residual_activations=(),
        residual_sequence=(),
        unavailable_fields=(),
        description="Typed product-horizon simulation; no residual or generation.",
    )


def _forecast_payload_for_protocol(
    *,
    protocol: RelationshipProductHorizonProtocol,
    forecast: PreferenceActionForecast,
) -> Mapping[str, object]:
    if protocol.is_v2:
        return preference_action_forecast_to_payload(forecast)
    return _legacy_forecast_payload(forecast)


def _frozen_forecast_envelope(
    forecast: PreferenceActionForecast,
) -> Mapping[str, object]:
    return _with_artifact_id(
        {
            "schema_version": PREFERENCE_ACTION_FORECAST_SNAPSHOT_SCHEMA_VERSION,
            "forecast": preference_action_forecast_to_payload(forecast),
        }
    )


def _legacy_forecast_payload(
    forecast: PreferenceActionForecast,
) -> Mapping[str, object]:
    """Preserve the published v1 forecast hash projection byte-for-byte."""

    payload: dict[str, object] = {
        "forecast_id": forecast.forecast_id,
        "decision_id": forecast.decision_id,
        "interlocutor_id": forecast.interlocutor_id,
        "candidate_predictions": [
            {
                "action_id": item.action_id,
                "outcomes": [
                    {"outcome_id": outcome.outcome_id, "probability_hex": outcome.probability.hex()}
                    for outcome in item.outcomes
                ],
            }
            for item in forecast.candidate_predictions
        ],
        "recommended_action_id": forecast.recommended_action_id,
        "confidence_hex": forecast.confidence.hex(),
        "source_record_ids": list(forecast.source_record_ids),
        "issued_turn": forecast.issued_turn,
        "evidence": list(forecast.evidence),
        "session_scope": forecast.session_scope,
    }
    return payload


def _social_pe_payload(snapshot: Any) -> Mapping[str, object]:
    return {
        "description": snapshot.description,
        "errors": [
            {
                "error_id": item.error_id,
                "prediction_id": item.prediction_id,
                "kind": item.kind.value,
                "outcome": item.outcome.value,
                "magnitude_hex": item.magnitude.hex(),
                "owner": item.owner,
                "scope_kind": item.scope_kind.value,
                "scope_id": item.scope_id,
                "evidence": list(item.evidence),
            }
            for item in snapshot.errors
        ],
    }


def _initial_baseline_history(subject: ProductPilotPublicSubject, count: int) -> list[ProductPublicHistoryBlock]:
    blocks: list[ProductPublicHistoryBlock] = []
    for session in subject.onboarding_sessions[:count]:
        blocks.append(
            ProductPublicHistoryBlock(
                ordinal=len(blocks),
                exchange_id=session.session_id,
                user_messages=(
                    session.public_context_chunk,
                    session.user_utterance,
                ),
                assistant_outcome=_public_assistant_outcome(
                    action_id=session.assistant_action_id,
                    outcome_id=session.observed_outcome_id,
                    reaction=session.rendered_user_reaction,
                ),
            )
        )
    return blocks


def _public_baseline_ledger(
    *,
    subject: ProductPilotPublicSubject,
    session: ProductPilotPublicDecisionSession,
    public_input: ProductBaselineInput,
    public_plan_artifact_id: str,
    ordered_source_session_ids: tuple[str, ...],
) -> Mapping[str, object]:
    history_entries = [
        _with_artifact_id(
            {
                "schema_version": "relationship-product-public-ledger-history-entry.v1",
                "ordinal": block.ordinal,
                "block_artifact_id": block.artifact_id,
                "block": block.to_payload(),
            }
        )
        for block in public_input.history
    ]
    current_entry = _with_artifact_id(
        {
            "schema_version": "relationship-product-public-ledger-current-entry.v1",
            "observation_artifact_id": public_input.current_observation.artifact_id,
            "observation": public_input.current_observation.to_payload(),
        }
    )
    return _with_artifact_id(
        {
            "schema_version": "relationship-product-public-ledger-snapshot.v1",
            "public_plan_artifact_id": public_plan_artifact_id,
            "subject_scope": subject.subject_scope,
            "world_clone_id": subject.world_clone_id,
            "session_id": session.session_id,
            "decision_id": session.decision_id,
            "decision_index": session.decision_index,
            "ordered_source_session_ids": list(ordered_source_session_ids),
            "ordered_source_block_artifact_ids": [block.artifact_id for block in public_input.history],
            "public_input": public_input.to_payload(),
            "history_entries": history_entries,
            "current_entry": current_entry,
            "published_before_sealed_environment_join": True,
        }
    )


def _validate_public_baseline_ledger(payload: Mapping[str, object]) -> None:
    _validate_content_addressed(payload, "public baseline ledger")
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "public_plan_artifact_id",
            "subject_scope",
            "world_clone_id",
            "session_id",
            "decision_id",
            "decision_index",
            "ordered_source_session_ids",
            "ordered_source_block_artifact_ids",
            "public_input",
            "history_entries",
            "current_entry",
            "published_before_sealed_environment_join",
            "artifact_id",
        },
        "public baseline ledger",
    )
    if payload["schema_version"] != "relationship-product-public-ledger-snapshot.v1":
        raise ValueError("public baseline ledger schema mismatch")
    _require_sha256(
        payload.get("public_plan_artifact_id"),
        "public baseline ledger public_plan_artifact_id",
    )
    if payload["published_before_sealed_environment_join"] is not True:
        raise ValueError("public baseline ledger publication order is not frozen")
    public_input = _mapping(payload["public_input"], "public ledger input")
    _validate_content_addressed(public_input, "public ledger input")
    history = _list(public_input.get("history"), "public ledger input.history")
    entries = _list(payload["history_entries"], "public ledger history_entries")
    if payload.get("ordered_source_block_artifact_ids") != [
        _mapping(item, "public ledger history block").get("artifact_id") for item in history
    ]:
        raise ValueError("public ledger ordered source block ids drifted")
    ordered_sessions = _list(
        payload.get("ordered_source_session_ids"),
        "public ledger ordered source sessions",
    )
    if (
        not ordered_sessions
        or ordered_sessions[-1] != payload.get("session_id")
        or len(set(ordered_sessions)) != len(ordered_sessions)
    ):
        raise ValueError("public ledger ordered source session boundary drifted")
    if len(history) != len(entries):
        raise ValueError("public ledger history lineage count mismatch")
    for index, (block_value, entry_value) in enumerate(zip(history, entries, strict=True)):
        block = _mapping(block_value, f"public ledger history[{index}]")
        entry = _mapping(entry_value, f"public ledger history_entry[{index}]")
        _require_exact_keys(
            entry,
            {
                "schema_version",
                "ordinal",
                "block_artifact_id",
                "block",
                "artifact_id",
            },
            "public ledger history entry",
        )
        _validate_content_addressed(block, "public ledger history block")
        _validate_content_addressed(entry, "public ledger history entry")
        if entry.get("block") != block or entry.get("block_artifact_id") != block.get("artifact_id"):
            raise ValueError("public ledger history entry lineage mismatch")
    current = _mapping(public_input.get("current_observation"), "public ledger current observation")
    current_entry = _mapping(payload["current_entry"], "public ledger current entry")
    _require_exact_keys(
        current_entry,
        {
            "schema_version",
            "observation_artifact_id",
            "observation",
            "artifact_id",
        },
        "public ledger current entry",
    )
    _validate_content_addressed(current, "public ledger current observation")
    _validate_content_addressed(current_entry, "public ledger current entry")
    if current_entry.get("observation") != current or current_entry.get("observation_artifact_id") != current.get(
        "artifact_id"
    ):
        raise ValueError("public ledger current observation lineage mismatch")


def _decision_public_history_blocks(
    history: Sequence[ProductPublicHistoryBlock],
    session: ProductPilotPublicDecisionSession,
    action: RelationshipAction,
    outcome_id: str,
    reaction: str,
) -> list[ProductPublicHistoryBlock]:
    return [
        ProductPublicHistoryBlock(
            ordinal=len(history),
            exchange_id=session.session_id,
            user_messages=(session.public_context_chunk, session.current_input),
            assistant_outcome=_public_assistant_outcome(
                action_id=action.value,
                outcome_id=outcome_id,
                reaction=reaction,
            ),
        ),
    ]


def _public_assistant_outcome(
    *,
    action_id: str,
    outcome_id: str,
    reaction: str,
) -> str:
    """Injectively render one public paired assistant action/outcome turn."""

    return canonical_json(
        {
            "action_id": action_id,
            "observed_outcome_id": outcome_id,
            "rendered_user_reaction": reaction,
        }
    )


def _read_json_line(stream: Any, *, timeout_seconds: float, process: subprocess.Popen[str]) -> Mapping[str, object]:
    result = _read_text_line(stream, timeout_seconds=timeout_seconds, process=process)
    try:
        payload = json.loads(result)
    except json.JSONDecodeError as exc:
        raise ValueError(f"relationship product child emitted non-JSON handshake: {result!r}") from exc
    return _mapping(payload, "handshake")


def _read_text_line(stream: Any, *, timeout_seconds: float, process: subprocess.Popen[str]) -> str:
    result_queue: queue.Queue[object] = queue.Queue(maxsize=1)

    def read() -> None:
        try:
            result_queue.put(stream.readline())
        except BaseException as exc:  # process boundary: re-raised in caller
            result_queue.put(exc)

    thread = threading.Thread(target=read, daemon=True)
    thread.start()
    try:
        result = result_queue.get(timeout=timeout_seconds)
    except queue.Empty as exc:
        if process.poll() is None:
            process.kill()
        raise TimeoutError("relationship product decision handshake timed out") from exc
    if isinstance(result, BaseException):
        raise result
    if not isinstance(result, str) or not result:
        if process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        stderr = process.stderr.read() if process.stderr is not None and process.poll() is not None else ""
        raise RuntimeError(f"relationship product decision child closed handshake; stderr={stderr[-2000:]}")
    return result


def _manifest_file_entries(root: pathlib.Path) -> list[Mapping[str, object]]:
    entries: list[Mapping[str, object]] = []
    excluded_manifest = (root / "manifest.json").resolve()
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.resolve() != excluded_manifest):
        entries.append(
            {"path": _relative_posix(root, path), "sha256": _sha256_file(path), "bytes": path.stat().st_size}
        )
    return entries


def _state_directory_sha256(root: pathlib.Path) -> str:
    return sha256_json(_manifest_file_entries(root))


def _canonical_local_execution_source_bytes(
    *,
    repository_path: str,
    raw_bytes: bytes,
) -> tuple[bytes, str]:
    pure_path = pathlib.PurePosixPath(repository_path)
    is_python_source = (
        len(pure_path.parts) >= 4
        and pure_path.parts[0] == "packages"
        and pure_path.parts[2] == "src"
        and pure_path.suffix.lower() == ".py"
    )
    if (
        not is_python_source
        and repository_path not in _LOCAL_EXECUTION_SOURCE_ENTRYPOINTS
        and repository_path not in _LOCAL_EXECUTION_RESOURCE_PATHS
    ):
        raise ValueError(
            f"local execution source is outside the frozen selector: {repository_path}"
        )
    try:
        source = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"local execution text source must be exact UTF-8: {repository_path}"
        ) from exc
    canonical = source.replace("\r\n", "\n").replace("\r", "\n").encode(
        "utf-8"
    )
    return canonical, "utf8_newlines_lf"


def _assert_non_reparse_source_path(
    path: pathlib.Path,
    *,
    repository_root: pathlib.Path,
) -> None:
    current = path
    while current != repository_root:
        metadata = current.lstat()
        file_attributes = getattr(metadata, "st_file_attributes", 0)
        if stat.S_ISLNK(metadata.st_mode) or file_attributes & getattr(
            stat,
            "FILE_ATTRIBUTE_REPARSE_POINT",
            0,
        ):
            raise ValueError(f"local Python source closure rejects reparse path: {path}")
        parent = current.parent
        if parent == current:
            raise ValueError(f"local Python source path escapes repository: {path}")
        current = parent


def _local_execution_source_tree(
    *,
    repository_root: pathlib.Path,
) -> tuple[
    Mapping[str, object],
    tuple[tuple[pathlib.Path, str, bytes, bytes, str], ...],
]:
    repository = pathlib.Path(repository_root).resolve()
    packages_root = repository / "packages"
    if not packages_root.is_dir():
        raise FileNotFoundError("local Python source closure packages root is missing")

    source_files: list[pathlib.Path] = []
    for package_root in sorted(
        (item for item in packages_root.iterdir() if item.is_dir()),
        key=lambda item: item.name.encode("utf-8"),
    ):
        source_root = package_root / "src"
        if source_root.is_dir():
            source_files.extend(item for item in source_root.rglob("*.py") if item.is_file())
    source_files.extend(
        repository / pathlib.PurePosixPath(relative)
        for relative in (
            *_LOCAL_EXECUTION_SOURCE_ENTRYPOINTS,
            *_LOCAL_EXECUTION_RESOURCE_PATHS,
        )
    )
    active_protocol_path = (
        repository / pathlib.PurePosixPath(_LOCAL_EXECUTION_ACTIVE_PROTOCOL_PATH)
    )
    _assert_non_reparse_source_path(
        active_protocol_path,
        repository_root=repository,
    )
    if not active_protocol_path.is_file():
        raise FileNotFoundError("active v2 product protocol resource is missing")
    identified: list[tuple[pathlib.Path, str, bytes, bytes, str]] = []
    casefold_paths: set[str] = set()
    for path in source_files:
        _assert_non_reparse_source_path(path, repository_root=repository)
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(repository).as_posix()
        except ValueError as exc:
            raise ValueError(f"local Python source path escapes repository: {path}") from exc
        folded = relative.casefold()
        if folded in casefold_paths:
            raise ValueError(f"local Python source closure has a casefold collision: {relative}")
        casefold_paths.add(folded)
        raw_bytes = resolved.read_bytes()
        canonical_bytes, canonicalization_kind = (
            _canonical_local_execution_source_bytes(
                repository_path=relative,
                raw_bytes=raw_bytes,
            )
        )
        identified.append(
            (
                resolved,
                relative,
                raw_bytes,
                canonical_bytes,
                canonicalization_kind,
            )
        )
    identified.sort(key=lambda item: item[1].encode("utf-8"))
    if not identified:
        raise ValueError("local Python source closure is empty")

    identity_files = [
        {
            "repository_path": relative,
            "canonicalization_kind": canonicalization_kind,
            "canonical_sha256": _sha256_bytes(canonical_bytes),
            "canonical_bytes": len(canonical_bytes),
        }
        for (
            _path,
            relative,
            _raw_bytes,
            canonical_bytes,
            canonicalization_kind,
        ) in identified
    ]
    identity = {
        "schema_version": _LOCAL_EXECUTION_SOURCE_TREE_SCHEMA_VERSION,
        "selector": _LOCAL_EXECUTION_SOURCE_TREE_SELECTOR,
        "entrypoints": list(_LOCAL_EXECUTION_SOURCE_ENTRYPOINTS),
        "resource_paths": list(_LOCAL_EXECUTION_RESOURCE_PATHS),
        "active_protocol_path": _LOCAL_EXECUTION_ACTIVE_PROTOCOL_PATH,
        "canonicalization": _LOCAL_EXECUTION_SOURCE_CANONICALIZATION,
        "files": identity_files,
    }
    summary = {
        "schema_version": _LOCAL_EXECUTION_SOURCE_TREE_SCHEMA_VERSION,
        "selector": _LOCAL_EXECUTION_SOURCE_TREE_SELECTOR,
        "entrypoints": list(_LOCAL_EXECUTION_SOURCE_ENTRYPOINTS),
        "resource_paths": list(_LOCAL_EXECUTION_RESOURCE_PATHS),
        "active_protocol_path": _LOCAL_EXECUTION_ACTIVE_PROTOCOL_PATH,
        "canonicalization": _LOCAL_EXECUTION_SOURCE_CANONICALIZATION,
        "tree_sha256": sha256_json(identity),
        "file_count": len(identity_files),
        "canonical_bytes": sum(
            _integer(item["canonical_bytes"], "canonical source bytes")
            for item in identity_files
        ),
    }
    return summary, tuple(identified)


def _execution_source_paths(
    *,
    campaign_cli: pathlib.Path,
) -> Mapping[str, pathlib.Path]:
    repository_root = pathlib.Path(__file__).resolve().parents[4]
    return {
        "baseline_dispatcher_cli": repository_root / "scripts" / "run_relationship_lab_product_baseline_dispatcher.py",
        "baseline_dispatcher_implementation": pathlib.Path(baseline_dispatcher_owner.__file__).resolve(),
        "baseline_implementation": pathlib.Path(product_baselines_owner.__file__).resolve(),
        "baseline_policy_implementation": pathlib.Path(baseline_policy_owner.__file__).resolve(),
        "campaign_cli": pathlib.Path(campaign_cli).resolve(),
        "campaign_implementation": pathlib.Path(__file__).resolve(),
        "model_adapters_implementation": pathlib.Path(product_model_adapters_owner.__file__).resolve(),
    }


def _publish_execution_source_bundle(
    *,
    root: pathlib.Path,
    protocol: RelationshipProductHorizonProtocol,
    campaign_cli: pathlib.Path,
    protocol_source: pathlib.Path | None = None,
) -> Mapping[str, object]:
    source_root = root / "inputs" / "execution_sources"
    source_root.mkdir(parents=True)
    if protocol.is_v2:
        if protocol.execution_source_sha256s:
            raise ValueError("v2 execution source authority must be the canonical tree")
        repository_root = pathlib.Path(__file__).resolve().parents[4]
        tree_summary, tree_files = _local_execution_source_tree(
            repository_root=repository_root,
        )
        expected_summary = {
            "schema_version": _LOCAL_EXECUTION_SOURCE_TREE_SCHEMA_VERSION,
            "selector": _LOCAL_EXECUTION_SOURCE_TREE_SELECTOR,
            "entrypoints": list(_LOCAL_EXECUTION_SOURCE_ENTRYPOINTS),
            "resource_paths": list(_LOCAL_EXECUTION_RESOURCE_PATHS),
            "active_protocol_path": _LOCAL_EXECUTION_ACTIVE_PROTOCOL_PATH,
            "canonicalization": _LOCAL_EXECUTION_SOURCE_CANONICALIZATION,
            "tree_sha256": protocol.local_execution_source_tree_sha256,
            "file_count": protocol.local_execution_source_file_count,
            "canonical_bytes": protocol.local_execution_source_canonical_bytes,
        }
        if tree_summary != expected_summary:
            raise ValueError("local Python source tree differs from protocol pin")
        tree_entries: list[Mapping[str, object]] = []
        for (
            source_path,
            repository_path,
            raw_bytes,
            canonical_bytes,
            canonicalization_kind,
        ) in tree_files:
            target = (
                source_root
                / "tree"
                / pathlib.PurePosixPath(repository_path)
            )
            _write_bytes_create_only(target, raw_bytes)
            tree_entries.append(
                {
                    "repository_path": repository_path,
                    "path": _relative_posix(root, target),
                    "canonicalization_kind": canonicalization_kind,
                    "canonical_sha256": _sha256_bytes(canonical_bytes),
                    "canonical_bytes": len(canonical_bytes),
                    "raw_sha256": _sha256_bytes(raw_bytes),
                    "raw_bytes": len(raw_bytes),
                }
            )
            if source_path.read_bytes() != raw_bytes:
                raise RuntimeError(
                    f"local Python source changed while freezing closure: {source_path}"
                )
        source_tree = _with_artifact_id(
            {**tree_summary, "files": tree_entries}
        )
        packaged_active_protocol_source = relationship_product_horizon_protocol_path(
            "v2"
        ).resolve()
        expected_active_protocol_source = (
            repository_root
            / pathlib.PurePosixPath(_LOCAL_EXECUTION_ACTIVE_PROTOCOL_PATH)
        ).resolve()
        if packaged_active_protocol_source != expected_active_protocol_source:
            raise ValueError("active v2 protocol resource path drifted")
        active_protocol_source = pathlib.Path(
            protocol_source or packaged_active_protocol_source
        ).resolve()
        if active_protocol_source == packaged_active_protocol_source:
            _assert_non_reparse_source_path(
                active_protocol_source,
                repository_root=repository_root,
            )
        elif active_protocol_source != _non_authorizing_test_protocol_path():
            raise ValueError("active v2 protocol source is not an admitted test protocol")
        active_protocol_bytes = active_protocol_source.read_bytes()
        if _sha256_bytes(active_protocol_bytes) != protocol.raw_sha256:
            raise ValueError("active v2 protocol raw bytes differ from loaded protocol")
        active_protocol_target = (
            source_root
            / "tree"
            / pathlib.PurePosixPath(_LOCAL_EXECUTION_ACTIVE_PROTOCOL_PATH)
        )
        _write_bytes_create_only(active_protocol_target, active_protocol_bytes)
        active_protocol_resource = {
            "schema_version": _ACTIVE_PROTOCOL_RESOURCE_SCHEMA_VERSION,
            "repository_path": _LOCAL_EXECUTION_ACTIVE_PROTOCOL_PATH,
            "path": _relative_posix(root, active_protocol_target),
            "protocol_id": protocol.protocol_id,
            "raw_sha256": protocol.raw_sha256,
            "raw_bytes": len(active_protocol_bytes),
        }
        bundle_core = {
            "schema_version": "relationship-product-execution-source-bundle.v2",
            "local_execution_source_tree": source_tree,
            "active_protocol_resource": active_protocol_resource,
        }
    else:
        paths = _execution_source_paths(campaign_cli=campaign_cli)
        if tuple(sorted(paths)) != _EXECUTION_SOURCE_KEYS:
            raise ValueError("execution source path set drifted")
        protocol_pins = dict(protocol.execution_source_sha256s)
        if protocol_pins and set(protocol_pins) != set(_EXECUTION_SOURCE_KEYS):
            raise ValueError("protocol execution source pin set drifted")
        entries: list[Mapping[str, object]] = []
        for key in _EXECUTION_SOURCE_KEYS:
            source_path = paths[key]
            if not source_path.is_file():
                raise FileNotFoundError(f"execution source is missing: {source_path}")
            digest = _sha256_file(source_path)
            if protocol_pins and protocol_pins[key] != digest:
                raise ValueError(f"execution source {key} differs from protocol pin")
            target = source_root / f"{key}{source_path.suffix}"
            source_bytes = source_path.read_bytes()
            _write_bytes_create_only(target, source_bytes)
            entries.append(
                {
                    "key": key,
                    "path": _relative_posix(root, target),
                    "sha256": digest,
                    "bytes": len(source_bytes),
                }
            )
        bundle_core = {
            "schema_version": "relationship-product-execution-source-bundle.v1",
            "sources": entries,
        }
    bundle = _with_artifact_id(bundle_core)
    _write_json_create_only(source_root / "bundle.json", bundle)
    return bundle


def _validate_execution_source_bundle(
    *,
    root: pathlib.Path,
    protocol: RelationshipProductHorizonProtocol,
) -> Mapping[str, object]:
    bundle = _load_json(root / "inputs" / "execution_sources" / "bundle.json")
    _validate_content_addressed(bundle, "execution source bundle")
    if protocol.is_v2:
        _require_exact_keys(
            bundle,
            {
                "schema_version",
                "local_execution_source_tree",
                "active_protocol_resource",
                "artifact_id",
            },
            "v2 execution source bundle",
        )
        if (
            bundle.get("schema_version")
            != "relationship-product-execution-source-bundle.v2"
        ):
            raise ValueError("execution source bundle schema drifted")
        _validate_active_protocol_resource(
            root=root,
            protocol=protocol,
            payload=_mapping(
                bundle.get("active_protocol_resource"),
                "active protocol resource",
            ),
        )
        _validate_local_execution_source_tree_bundle(
            root=root,
            protocol=protocol,
            payload=_mapping(
                bundle.get("local_execution_source_tree"),
                "execution source local tree",
            ),
        )
        return bundle

    _require_exact_keys(
        bundle,
        {"schema_version", "sources", "artifact_id"},
        "v1 execution source bundle",
    )
    if bundle.get("schema_version") != "relationship-product-execution-source-bundle.v1":
        raise ValueError("execution source bundle schema drifted")
    pins = dict(protocol.execution_source_sha256s)
    if pins and set(pins) != set(_EXECUTION_SOURCE_KEYS):
        raise ValueError("packaged protocol lacks complete execution source pins")
    entries = tuple(
        _mapping(item, "execution source entry")
        for item in _list(bundle.get("sources"), "execution source entries")
    )
    if (
        tuple(_text(item.get("key"), "execution source key") for item in entries)
        != _EXECUTION_SOURCE_KEYS
    ):
        raise ValueError("execution source bundle key/order drifted")
    for entry in entries:
        _require_exact_keys(
            entry,
            {"key", "path", "sha256", "bytes"},
            "execution source entry",
        )
        key = _text(entry.get("key"), "execution source key")
        path = _resolve_relative(
            root,
            _text(entry.get("path"), "execution source path"),
        )
        expected_digest = pins.get(
            key,
            _digest(entry.get("sha256"), "test execution source sha256"),
        )
        if (
            entry.get("sha256") != expected_digest
            or _sha256_file(path) != expected_digest
            or entry.get("bytes") != path.stat().st_size
        ):
            raise ValueError(f"execution source bundle file {key} drifted")
    return bundle


def _validate_active_protocol_resource(
    *,
    root: pathlib.Path,
    protocol: RelationshipProductHorizonProtocol,
    payload: Mapping[str, object],
) -> None:
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "repository_path",
            "path",
            "protocol_id",
            "raw_sha256",
            "raw_bytes",
        },
        "active protocol resource",
    )
    expected_path = (
        pathlib.PurePosixPath("inputs")
        / "execution_sources"
        / "tree"
        / pathlib.PurePosixPath(_LOCAL_EXECUTION_ACTIVE_PROTOCOL_PATH)
    ).as_posix()
    if (
        payload.get("schema_version") != _ACTIVE_PROTOCOL_RESOURCE_SCHEMA_VERSION
        or payload.get("repository_path") != _LOCAL_EXECUTION_ACTIVE_PROTOCOL_PATH
        or payload.get("path") != expected_path
        or payload.get("protocol_id") != protocol.protocol_id
        or payload.get("raw_sha256") != protocol.raw_sha256
    ):
        raise ValueError("active protocol resource lineage drifted")
    active_path = _resolve_relative(root, expected_path)
    active_bytes = active_path.read_bytes()
    if (
        _sha256_bytes(active_bytes) != protocol.raw_sha256
        or payload.get("raw_bytes") != len(active_bytes)
        or active_bytes != (root / "protocol.json").read_bytes()
    ):
        raise ValueError("active protocol resource bytes drifted")


def _validate_local_execution_source_tree_bundle(
    *,
    root: pathlib.Path,
    protocol: RelationshipProductHorizonProtocol,
    payload: Mapping[str, object],
) -> None:
    _validate_content_addressed(payload, "local Python source tree")
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "selector",
            "entrypoints",
            "resource_paths",
            "active_protocol_path",
            "canonicalization",
            "tree_sha256",
            "file_count",
            "canonical_bytes",
            "files",
            "artifact_id",
        },
        "local execution source tree",
    )
    expected_summary = {
        "schema_version": _LOCAL_EXECUTION_SOURCE_TREE_SCHEMA_VERSION,
        "selector": _LOCAL_EXECUTION_SOURCE_TREE_SELECTOR,
        "entrypoints": list(_LOCAL_EXECUTION_SOURCE_ENTRYPOINTS),
        "resource_paths": list(_LOCAL_EXECUTION_RESOURCE_PATHS),
        "active_protocol_path": _LOCAL_EXECUTION_ACTIVE_PROTOCOL_PATH,
        "canonicalization": _LOCAL_EXECUTION_SOURCE_CANONICALIZATION,
        "tree_sha256": protocol.local_execution_source_tree_sha256,
        "file_count": protocol.local_execution_source_file_count,
        "canonical_bytes": protocol.local_execution_source_canonical_bytes,
    }
    if {key: payload.get(key) for key in expected_summary} != expected_summary:
        raise ValueError("local Python source-tree summary differs from protocol")

    entries = tuple(
        _mapping(item, "local Python source-tree entry")
        for item in _list(payload.get("files"), "local Python source-tree files")
    )
    if len(entries) != protocol.local_execution_source_file_count:
        raise ValueError("local execution source-tree file count drifted")
    identity_files: list[Mapping[str, object]] = []
    observed_paths: list[str] = []
    observed_casefolds: set[str] = set()
    for entry in entries:
        _require_exact_keys(
            entry,
            {
                "repository_path",
                "path",
                "canonicalization_kind",
                "canonical_sha256",
                "canonical_bytes",
                "raw_sha256",
                "raw_bytes",
            },
            "local execution source-tree entry",
        )
        repository_path = _text(
            entry.get("repository_path"),
            "local Python source repository_path",
        )
        pure_repository_path = pathlib.PurePosixPath(repository_path)
        is_python_source = (
            len(pure_repository_path.parts) >= 4
            and pure_repository_path.parts[0] == "packages"
            and pure_repository_path.parts[2] == "src"
            and pure_repository_path.suffix.lower() == ".py"
        )
        is_entrypoint = repository_path in _LOCAL_EXECUTION_SOURCE_ENTRYPOINTS
        is_resource = repository_path in _LOCAL_EXECUTION_RESOURCE_PATHS
        if (
            pure_repository_path.is_absolute()
            or ".." in pure_repository_path.parts
            or not (is_python_source or is_entrypoint or is_resource)
            or pure_repository_path.as_posix() != repository_path
        ):
            raise ValueError("local execution source repository path is outside selector")
        folded = repository_path.casefold()
        if folded in observed_casefolds:
            raise ValueError("local Python source-tree path casefold collision")
        observed_casefolds.add(folded)
        observed_paths.append(repository_path)
        expected_artifact_path = (
            pathlib.PurePosixPath("inputs")
            / "execution_sources"
            / "tree"
            / pure_repository_path
        ).as_posix()
        if entry.get("path") != expected_artifact_path:
            raise ValueError("local Python source-tree artifact path drifted")
        artifact_path = _resolve_relative(root, expected_artifact_path)
        raw_bytes = artifact_path.read_bytes()
        canonical_bytes, canonicalization_kind = (
            _canonical_local_execution_source_bytes(
                repository_path=repository_path,
                raw_bytes=raw_bytes,
            )
        )
        canonical_sha256 = _digest(
            entry.get("canonical_sha256"),
            "local Python source canonical_sha256",
        )
        raw_sha256 = _digest(
            entry.get("raw_sha256"),
            "local Python source raw_sha256",
        )
        if (
            entry.get("canonicalization_kind") != canonicalization_kind
            or _sha256_bytes(canonical_bytes) != canonical_sha256
            or entry.get("canonical_bytes") != len(canonical_bytes)
            or _sha256_bytes(raw_bytes) != raw_sha256
            or entry.get("raw_bytes") != len(raw_bytes)
        ):
            raise ValueError(f"local Python source-tree file drifted: {repository_path}")
        identity_files.append(
            {
                "repository_path": repository_path,
                "canonicalization_kind": canonicalization_kind,
                "canonical_sha256": canonical_sha256,
                "canonical_bytes": len(canonical_bytes),
            }
        )
    expected_order = sorted(observed_paths, key=lambda value: value.encode("utf-8"))
    if observed_paths != expected_order:
        raise ValueError("local Python source-tree path ordering drifted")
    identity = {
        "schema_version": _LOCAL_EXECUTION_SOURCE_TREE_SCHEMA_VERSION,
        "selector": _LOCAL_EXECUTION_SOURCE_TREE_SELECTOR,
        "entrypoints": list(_LOCAL_EXECUTION_SOURCE_ENTRYPOINTS),
        "resource_paths": list(_LOCAL_EXECUTION_RESOURCE_PATHS),
        "active_protocol_path": _LOCAL_EXECUTION_ACTIVE_PROTOCOL_PATH,
        "canonicalization": _LOCAL_EXECUTION_SOURCE_CANONICALIZATION,
        "files": identity_files,
    }
    if (
        sha256_json(identity) != protocol.local_execution_source_tree_sha256
        or sum(
            _integer(item["canonical_bytes"], "canonical source bytes")
            for item in identity_files
        )
        != protocol.local_execution_source_canonical_bytes
    ):
        raise ValueError("local Python source-tree root digest drifted")
    expected_tree_paths = sorted(
        [
            _text(item.get("path"), "local execution source-tree path")
            for item in entries
        ]
        + [
            (
                pathlib.PurePosixPath("inputs")
                / "execution_sources"
                / "tree"
                / pathlib.PurePosixPath(_LOCAL_EXECUTION_ACTIVE_PROTOCOL_PATH)
            ).as_posix()
        ],
        key=lambda value: value.encode("utf-8"),
    )
    tree_root = root / "inputs" / "execution_sources" / "tree"
    observed_tree_paths = sorted(
        _relative_posix(root, path)
        for path in tree_root.rglob("*")
        if path.is_file()
    )
    if observed_tree_paths != expected_tree_paths:
        raise ValueError("local execution source tree contains missing or extra files")
    observed_file_ids: set[tuple[int, int]] = set()
    for relative_path in observed_tree_paths:
        artifact_path = root / pathlib.PurePosixPath(relative_path)
        _assert_non_reparse_source_path(
            artifact_path,
            repository_root=root,
        )
        metadata = artifact_path.stat()
        if metadata.st_nlink != 1:
            raise ValueError("local execution source tree rejects hard-linked files")
        file_id = (metadata.st_dev, metadata.st_ino)
        if file_id in observed_file_ids:
            raise ValueError("local execution source tree rejects hard-linked files")
        observed_file_ids.add(file_id)


def _source_tree_entry_by_repository_path(
    source_tree: Mapping[str, object],
) -> Mapping[str, Mapping[str, object]]:
    entries = tuple(
        _mapping(item, "local execution source-tree entry")
        for item in _list(source_tree.get("files"), "local execution source-tree files")
    )
    indexed = {
        _text(item.get("repository_path"), "source-tree repository_path"): item
        for item in entries
    }
    if len(indexed) != len(entries):
        raise ValueError("local execution source tree has duplicate repository paths")
    return indexed


def _expected_module_repository_path(
    *,
    module_name: str,
    source_entries: Mapping[str, Mapping[str, object]],
) -> str:
    suffix = f"/{module_name.replace('.', '/')}.py"
    matches = tuple(
        path
        for path in source_entries
        if f"/{path}".endswith(suffix)
    )
    if len(matches) != 1:
        raise ValueError(
            f"critical local module {module_name} is not unique in the source tree"
        )
    return matches[0]


def _local_source_top_level_module_names(
    source_entries: Mapping[str, Mapping[str, object]],
) -> tuple[str, ...]:
    names: set[str] = set()
    for repository_path in source_entries:
        parts = pathlib.PurePosixPath(repository_path).parts
        if (
            len(parts) >= 4
            and parts[0] == "packages"
            and parts[2] == "src"
            and repository_path.endswith(".py")
        ):
            names.add(pathlib.PurePosixPath(parts[3]).stem)
    return tuple(sorted(names, key=lambda value: value.encode("utf-8")))


def _module_name_for_repository_path(repository_path: str) -> str:
    parts = pathlib.PurePosixPath(repository_path).parts
    if (
        len(parts) < 4
        or parts[0] != "packages"
        or parts[2] != "src"
        or not repository_path.endswith(".py")
    ):
        raise ValueError(
            f"local module repository path is not importable Python: {repository_path}"
        )
    module_parts = list(parts[3:])
    if module_parts[-1] == "__init__.py":
        module_parts.pop()
    else:
        module_parts[-1] = pathlib.PurePosixPath(module_parts[-1]).stem
    if not module_parts:
        raise ValueError(
            f"local module repository path has no module name: {repository_path}"
        )
    return ".".join(module_parts)


def _local_module_repository_paths_by_name(
    source_entries: Mapping[str, Mapping[str, object]],
) -> Mapping[str, str]:
    indexed: dict[str, str] = {}
    for repository_path in source_entries:
        parts = pathlib.PurePosixPath(repository_path).parts
        if not (
            len(parts) >= 4
            and parts[0] == "packages"
            and parts[2] == "src"
            and repository_path.endswith(".py")
        ):
            continue
        module_name = _module_name_for_repository_path(repository_path)
        if module_name in indexed:
            raise ValueError(
                f"loaded local module name/path mapping is not unique: {module_name}"
            )
        indexed[module_name] = repository_path
    return indexed


def _expected_volvence_zero_namespace_paths(
    source_entries: Mapping[str, Mapping[str, object]],
) -> tuple[str, ...]:
    paths = {
        pathlib.PurePosixPath(*pathlib.PurePosixPath(repository_path).parts[:4]).as_posix()
        for repository_path in source_entries
        if (
            len(pathlib.PurePosixPath(repository_path).parts) >= 5
            and pathlib.PurePosixPath(repository_path).parts[0] == "packages"
            and pathlib.PurePosixPath(repository_path).parts[2] == "src"
            and pathlib.PurePosixPath(repository_path).parts[3]
            == "volvence_zero"
            and repository_path.endswith(".py")
        )
    }
    if not paths:
        raise ValueError("local source tree has no volvence_zero namespace roots")
    return tuple(sorted(paths, key=lambda value: value.encode("utf-8")))


def _loaded_local_module_origins(
    *,
    tree_root: pathlib.Path,
    source_entries: Mapping[str, Mapping[str, object]],
) -> list[Mapping[str, object]]:
    top_level_names = _local_source_top_level_module_names(source_entries)
    expected_module_paths = _local_module_repository_paths_by_name(source_entries)
    origins: list[Mapping[str, object]] = []
    resolved_tree_root = tree_root.resolve()
    for module_name, module in sorted(
        sys.modules.items(),
        key=lambda item: item[0].encode("utf-8"),
    ):
        if module is None or not any(
            module_name == name or module_name.startswith(f"{name}.")
            for name in top_level_names
        ):
            continue
        module_values = vars(module)
        module_file = module_values.get("__file__")
        if module_file is None and module_name == "volvence_zero":
            continue
        if not isinstance(module_file, str) or not module_file:
            raise RuntimeError(
                f"loaded local module has no auditable source file: {module_name}"
            )
        module_path = pathlib.Path(module_file).resolve()
        try:
            repository_path = module_path.relative_to(
                resolved_tree_root
            ).as_posix()
        except ValueError as exc:
            raise RuntimeError(
                f"loaded local module escaped the mirrored tree: {module_name}"
            ) from exc
        entry = source_entries.get(repository_path)
        if entry is None:
            raise RuntimeError(
                f"loaded local module is absent from the source tree: {module_name}"
            )
        raw_sha256 = _digest(
            entry.get("raw_sha256"),
            f"loaded local module {module_name} raw sha256",
        )
        if _sha256_file(module_path) != raw_sha256:
            raise RuntimeError(
                f"loaded local module differs from the source tree: {module_name}"
            )
        expected_repository_path = expected_module_paths.get(module_name)
        if repository_path != expected_repository_path:
            raise RuntimeError(
                f"loaded local module name/path mapping drifted: {module_name}"
            )
        origins.append(
            {
                "module_name": module_name,
                "repository_path": repository_path,
                "raw_sha256": raw_sha256,
            }
        )
    return origins


def _v2_worker_source_lineage(
    *,
    root: pathlib.Path,
    protocol: RelationshipProductHorizonProtocol,
    request: Mapping[str, object],
) -> Mapping[str, object]:
    if not protocol.is_v2:
        raise ValueError("worker source lineage is a v2-only contract")
    bundle = _load_json(root / "inputs" / "execution_sources" / "bundle.json")
    _validate_content_addressed(bundle, "worker execution source bundle")
    bundle_id = _digest(
        request.get("execution_source_bundle_artifact_id"),
        "worker execution source bundle artifact_id",
    )
    if bundle.get("artifact_id") != bundle_id:
        raise ValueError("worker execution source bundle lineage drifted")
    source_tree = _mapping(
        bundle.get("local_execution_source_tree"),
        "worker local execution source tree",
    )
    _validate_content_addressed(source_tree, "worker local execution source tree")
    tree_sha256 = _digest(
        request.get("local_execution_source_tree_sha256"),
        "worker local execution source tree sha256",
    )
    if (
        source_tree.get("tree_sha256") != tree_sha256
        or tree_sha256 != protocol.local_execution_source_tree_sha256
    ):
        raise ValueError("worker local execution source tree root drifted")
    source_entries = _source_tree_entry_by_repository_path(source_tree)
    tree_root = root / "inputs" / "execution_sources" / "tree"
    namespace_module = sys.modules.get("volvence_zero")
    namespace_values = vars(namespace_module) if namespace_module is not None else {}
    namespace_path = namespace_values.get("__path__")
    if namespace_path is None:
        raise RuntimeError("volvence_zero namespace is not loaded")
    resolved_tree_root = tree_root.resolve()
    namespace_search_locations: list[str] = []
    for value in namespace_path:
        try:
            repository_path = pathlib.Path(value).resolve().relative_to(
                resolved_tree_root
            ).as_posix()
        except ValueError as exc:
            raise RuntimeError(
                "volvence_zero namespace escaped the mirrored source tree"
            ) from exc
        namespace_search_locations.append(repository_path)
    expected_namespace_locations = list(
        _expected_volvence_zero_namespace_paths(source_entries)
    )
    if namespace_search_locations != expected_namespace_locations:
        raise RuntimeError("volvence_zero namespace search locations drifted")
    worker_repository_path = "scripts/run_relationship_lab_product_horizon.py"
    worker_entry = source_entries.get(worker_repository_path)
    if worker_entry is None:
        raise ValueError("worker CLI is absent from the local execution source tree")
    worker_path = pathlib.Path(sys.argv[0]).resolve()
    expected_worker_path = (
        tree_root / pathlib.PurePosixPath(worker_repository_path)
    ).resolve()
    worker_raw_sha256 = _digest(
        worker_entry.get("raw_sha256"),
        "worker CLI raw sha256",
    )
    if (
        worker_path != expected_worker_path
        or _sha256_file(worker_path) != worker_raw_sha256
    ):
        raise RuntimeError("worker process did not boot from the mirrored CLI")
    module_origins: list[Mapping[str, object]] = []
    for module_name in _CRITICAL_LOCAL_MODULE_NAMES:
        module = sys.modules.get(module_name)
        module_file = module.__file__ if module is not None else None
        if not isinstance(module_file, str) or not module_file:
            raise RuntimeError(f"critical local module is not loaded: {module_name}")
        module_path = pathlib.Path(module_file).resolve()
        try:
            repository_path = module_path.relative_to(tree_root.resolve()).as_posix()
        except ValueError as exc:
            raise RuntimeError(
                f"critical local module escaped the mirrored tree: {module_name}"
            ) from exc
        expected_repository_path = _expected_module_repository_path(
            module_name=module_name,
            source_entries=source_entries,
        )
        entry = source_entries[expected_repository_path]
        raw_sha256 = _digest(
            entry.get("raw_sha256"),
            f"critical module {module_name} raw sha256",
        )
        if (
            repository_path != expected_repository_path
            or _sha256_file(module_path) != raw_sha256
        ):
            raise RuntimeError(
                f"critical local module differs from the mirrored tree: {module_name}"
            )
        module_origins.append(
            {
                "module_name": module_name,
                "repository_path": repository_path,
                "raw_sha256": raw_sha256,
            }
        )
    return _with_artifact_id(
        {
            "schema_version": RELATIONSHIP_PRODUCT_WORKER_SOURCE_LINEAGE_SCHEMA_VERSION,
            "execution_source_bundle_artifact_id": bundle_id,
            "local_execution_source_tree_sha256": tree_sha256,
            "worker_script_repository_path": worker_repository_path,
            "worker_script_raw_sha256": worker_raw_sha256,
            "volvence_zero_namespace_search_locations": (
                namespace_search_locations
            ),
            "critical_module_origins": module_origins,
            "loaded_local_module_origins": _loaded_local_module_origins(
                tree_root=tree_root,
                source_entries=source_entries,
            ),
        }
    )


def _validate_v2_worker_source_lineage(
    payload: Mapping[str, object],
    *,
    root: pathlib.Path,
    protocol: RelationshipProductHorizonProtocol,
    execution_source_bundle_artifact_id: str,
) -> None:
    _validate_content_addressed(payload, "worker source lineage")
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "execution_source_bundle_artifact_id",
            "local_execution_source_tree_sha256",
            "worker_script_repository_path",
            "worker_script_raw_sha256",
            "volvence_zero_namespace_search_locations",
            "critical_module_origins",
            "loaded_local_module_origins",
            "artifact_id",
        },
        "worker source lineage",
    )
    if (
        payload.get("schema_version")
        != RELATIONSHIP_PRODUCT_WORKER_SOURCE_LINEAGE_SCHEMA_VERSION
        or payload.get("execution_source_bundle_artifact_id")
        != execution_source_bundle_artifact_id
        or payload.get("local_execution_source_tree_sha256")
        != protocol.local_execution_source_tree_sha256
        or payload.get("worker_script_repository_path")
        != "scripts/run_relationship_lab_product_horizon.py"
    ):
        raise ValueError("worker source lineage projection drifted")
    bundle = _load_json(root / "inputs" / "execution_sources" / "bundle.json")
    if bundle.get("artifact_id") != execution_source_bundle_artifact_id:
        raise ValueError("worker source lineage bundle is not the validated bundle")
    source_tree = _mapping(
        bundle.get("local_execution_source_tree"),
        "worker source lineage tree",
    )
    source_entries = _source_tree_entry_by_repository_path(source_tree)
    expected_module_paths = _local_module_repository_paths_by_name(source_entries)
    expected_namespace_locations = list(
        _expected_volvence_zero_namespace_paths(source_entries)
    )
    if (
        payload.get("volvence_zero_namespace_search_locations")
        != expected_namespace_locations
    ):
        raise ValueError("worker volvence_zero namespace lineage drifted")
    worker_entry = source_entries["scripts/run_relationship_lab_product_horizon.py"]
    if payload.get("worker_script_raw_sha256") != worker_entry.get("raw_sha256"):
        raise ValueError("worker source lineage CLI digest drifted")
    origins = tuple(
        _mapping(item, "critical module origin")
        for item in _list(
            payload.get("critical_module_origins"),
            "critical module origins",
        )
    )
    if tuple(item.get("module_name") for item in origins) != _CRITICAL_LOCAL_MODULE_NAMES:
        raise ValueError("worker source lineage critical module set/order drifted")
    tree_root = root / "inputs" / "execution_sources" / "tree"
    for origin in origins:
        _require_exact_keys(
            origin,
            {"module_name", "repository_path", "raw_sha256"},
            "critical module origin",
        )
        module_name = _text(origin.get("module_name"), "critical module name")
        repository_path = _expected_module_repository_path(
            module_name=module_name,
            source_entries=source_entries,
        )
        entry = source_entries[repository_path]
        artifact_path = tree_root / pathlib.PurePosixPath(repository_path)
        if (
            origin.get("repository_path") != repository_path
            or origin.get("raw_sha256") != entry.get("raw_sha256")
            or _sha256_file(artifact_path)
            != _digest(entry.get("raw_sha256"), "critical module raw sha256")
        ):
            raise ValueError(f"critical module source lineage drifted: {module_name}")
    loaded_origins = tuple(
        _mapping(item, "loaded local module origin")
        for item in _list(
            payload.get("loaded_local_module_origins"),
            "loaded local module origins",
        )
    )
    loaded_names = tuple(
        _text(item.get("module_name"), "loaded local module name")
        for item in loaded_origins
    )
    if (
        loaded_names
        != tuple(sorted(loaded_names, key=lambda value: value.encode("utf-8")))
        or len(set(loaded_names)) != len(loaded_names)
        or not set(_CRITICAL_LOCAL_MODULE_NAMES).issubset(loaded_names)
    ):
        raise ValueError("worker loaded local module lineage set/order drifted")
    for origin in loaded_origins:
        _require_exact_keys(
            origin,
            {"module_name", "repository_path", "raw_sha256"},
            "loaded local module origin",
        )
        repository_path = _text(
            origin.get("repository_path"),
            "loaded local module repository_path",
        )
        module_name = _text(
            origin.get("module_name"),
            "loaded local module name",
        )
        expected_repository_path = expected_module_paths.get(module_name)
        if repository_path != expected_repository_path:
            raise ValueError("loaded local module name/path mapping drifted")
        entry = source_entries.get(repository_path)
        if entry is None:
            raise ValueError("loaded local module is absent from source tree")
        artifact_path = tree_root / pathlib.PurePosixPath(repository_path)
        if (
            origin.get("raw_sha256") != entry.get("raw_sha256")
            or _sha256_file(artifact_path)
            != _digest(entry.get("raw_sha256"), "loaded local module raw sha256")
        ):
            raise ValueError("loaded local module source lineage drifted")


def _subprocess_environment_contract(
    protocol: RelationshipProductHorizonProtocol,
) -> Mapping[str, str | None]:
    return (
        _SUBPROCESS_ENVIRONMENT_CONTRACT_V2
        if protocol.is_v2
        else _SUBPROCESS_ENVIRONMENT_CONTRACT
    )


def _child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment.update(_SUBPROCESS_ENVIRONMENT_CONTRACT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONSAFEPATH"] = "1"
    return environment


def _resolve_relative(root: pathlib.Path, value: str) -> pathlib.Path:
    candidate = (root / pathlib.PurePosixPath(value)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"worker path escapes run root: {value}") from exc
    return candidate


def _relative_posix(root: pathlib.Path, path: pathlib.Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _write_json_create_only(path: pathlib.Path, payload: Mapping[str, object]) -> None:
    _write_text_create_only(path, canonical_json(payload) + "\n")


def _write_text_create_only(path: pathlib.Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _write_bytes_create_only(path: pathlib.Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _load_json(path: pathlib.Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    return _mapping(payload, str(path))


def _with_artifact_id(core: Mapping[str, object]) -> Mapping[str, object]:
    if "artifact_id" in core:
        raise ValueError("content-addressed core cannot already carry artifact_id")
    return {**core, "artifact_id": sha256_json(core)}


def _validate_content_addressed(payload: Mapping[str, object], source: str) -> None:
    artifact_id = payload.get("artifact_id")
    _require_sha256(artifact_id, f"{source}.artifact_id")
    core = {key: value for key, value in payload.items() if key != "artifact_id"}
    if artifact_id != sha256_json(core):
        raise ValueError(f"{source} artifact_id mismatch")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_file(path: pathlib.Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_u64(*parts: str) -> int:
    if not parts or any(not isinstance(part, str) or not part for part in parts):
        raise ValueError("sha256_u64 parts must be non-empty strings")
    digest = hashlib.sha256(canonical_json(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _list(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return value


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{field} must be finite numeric")
    return float(value)


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def _digest(value: object, field: str) -> str:
    _require_sha256(value, field)
    assert isinstance(value, str)
    return value


def _require_sha256(value: object, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be lowercase sha256")


def _require_exact_keys(value: object, expected: set[str], source: str) -> None:
    mapping = _mapping(value, source)
    if set(mapping) != expected:
        missing = sorted(expected - set(mapping))
        extra = sorted(set(mapping) - expected)
        raise ValueError(f"{source} fields drifted; missing={missing}, extra={extra}")


__all__ = [
    "RELATIONSHIP_PRODUCT_HORIZON_SCHEMA_VERSION",
    "RelationshipProductArm",
    "RelationshipProductCampaignSelection",
    "RelationshipProductHorizonProtocol",
    "load_relationship_product_horizon_protocol",
    "relationship_product_horizon_protocol_path",
    "relationship_product_required_semantic_texts",
    "run_relationship_product_decision_worker",
    "run_relationship_product_horizon_campaign",
    "run_relationship_product_onboarding_worker",
    "validate_relationship_product_horizon_campaign",
]
