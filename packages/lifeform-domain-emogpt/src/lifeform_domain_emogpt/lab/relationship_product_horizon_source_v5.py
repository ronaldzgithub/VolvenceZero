"""Identity-disjoint Product Horizon source-v5 rendering over the frozen v4 causal engine.

The v4 source has already been consumed by adaptive development.  This owner keeps its
typed schedule and reactive-environment mathematics as a byte-pinned dependency, while
publishing a new protocol identity, new seeds, and a separately frozen public language
catalog.  No model, reader, current-decision outcome settlement, PE, credit, or gate
update is run here; the four public onboarding outcomes per root remain historical input.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import pathlib
from typing import Any, Mapping

from lifeform_domain_emogpt.lab.contracts import canonical_json
from lifeform_domain_emogpt.lab.environment import ReactiveRelationshipEnvironment
from lifeform_domain_emogpt.lab.relationship_product_horizon_source_v4 import (
    RELATIONSHIP_PRODUCT_HORIZON_EVALUATOR_SCHEMA_VERSION,
    RELATIONSHIP_PRODUCT_HORIZON_PUBLIC_VIEW_SCHEMA_VERSION,
    RELATIONSHIP_PRODUCT_HORIZON_SOURCE_SCHEMA_VERSION,
    HorizonPublicDecisionSession,
    HorizonPublicOnboardingSession,
    HorizonPublicRoot,
    RelationshipProductHorizonEvaluatorBundle,
    RelationshipProductHorizonPublicView,
    RelationshipProductHorizonSourceProtocol,
    build_relationship_product_horizon_environment,
    build_relationship_product_horizon_evaluator_bundle,
    build_relationship_product_horizon_public_view,
    load_relationship_product_horizon_source_protocol,
    relationship_product_horizon_source_protocol_path,
)


RELATIONSHIP_PRODUCT_HORIZON_SOURCE_V5_SCHEMA_VERSION = (
    "relationship-product-horizon-source.v5"
)
RELATIONSHIP_PRODUCT_HORIZON_PUBLIC_RENDERING_V5 = (
    "relationship-product-horizon-public-renderer.v5"
)
_PROTOCOL_FILENAME = "relationship_product_horizon_source_v5.json"
_OWNER_MODULE = "lifeform_domain_emogpt.lab.relationship_product_horizon_source_v5"
_PUBLIC_META_LITERALS = (
    "模型",
    "评估器",
    "隐藏状态",
    "系统",
    "跨会话",
    "长上下文",
    "评测",
    "测试",
    "提示词",
    "记忆是否",
    "恢复和读取",
    "检验",
    "与关系判断无关",
    "只用于说明注意力背景",
    "隐藏的语义标签",
)
_ACTION_SURFACE = (
    ("stay_present_without_probe", "留在这里听，不追问、不代办，也不替对方下结论"),
    ("respect_space_with_return_option", "尊重暂停，同时明确保留以后回来继续交流的入口"),
    ("neutral_noop", "不实施关系干预，仅维持当前公开状态"),
)


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_int(value: object, field_name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field_name} must be an integer")
    return value


def _require_bool(value: object, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _require_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], *, source: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValueError(f"{source} fields drifted; missing={missing}, extra={extra}")


def _require_text_tuple(value: object, field_name: str, *, count: int) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"{field_name} must contain exactly {count} strings")
    result = tuple(_require_text(item, f"{field_name}[{index}]") for index, item in enumerate(value))
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must contain unique strings")
    return result


def _parse_unique_json(raw_bytes: bytes, source: pathlib.Path) -> dict[str, object]:
    if b"\r" in raw_bytes or not raw_bytes.endswith(b"\n") or raw_bytes.endswith(b"\n\n"):
        raise ValueError(f"{source} must be LF-only UTF-8 ending in one LF")
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{source} must be UTF-8") from exc

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{source} contains duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"{source} contains non-finite JSON number: {value}")
            ),
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{source} must contain a JSON object")
    return payload


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _reader_inventory_sha256(public: RelationshipProductHorizonPublicView) -> str:
    inventory = sorted((_sha256_text(text), text) for text in _reader_texts(public))
    return hashlib.sha256(canonical_json(inventory).encode("utf-8")).hexdigest()


def _stable_index(protocol_id: str, purpose: str, row_id: str, count: int) -> int:
    digest = hashlib.sha256(
        canonical_json(
            {"protocol_id": protocol_id, "purpose": purpose, "row_id": row_id}
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % count


@dataclass(frozen=True)
class RelationshipProductHorizonSourceV5Catalog:
    domain_contexts: tuple[tuple[str, str], ...]
    role_contexts: tuple[tuple[str, str], ...]
    condition_surfaces: tuple[tuple[str, tuple[str, ...]], ...]
    segment_contexts: tuple[tuple[str, tuple[str, ...]], ...]
    reflections: tuple[str, ...]
    neutral_contexts: tuple[str, ...]
    historical_reactions: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for field_name, value in self.__dict__.items():
            if type(value) is not tuple:
                raise ValueError(f"catalog.{field_name} must be immutable")

    def domain(self, domain_id: str) -> str:
        return dict(self.domain_contexts)[domain_id]

    def role(self, role_id: str) -> str:
        return dict(self.role_contexts)[role_id]

    def condition(self, condition_id: str) -> tuple[str, ...]:
        return dict(self.condition_surfaces)[condition_id]

    def segment(self, segment_id: str) -> tuple[str, ...]:
        return dict(self.segment_contexts)[segment_id]

    def reaction(self, outcome_id: str) -> str:
        return dict(self.historical_reactions)[outcome_id]


@dataclass(frozen=True)
class RelationshipProductHorizonSourceV5Protocol:
    protocol_id: str
    cohort_id: str
    rendering_version: str
    engine_protocol: RelationshipProductHorizonSourceProtocol
    catalog: RelationshipProductHorizonSourceV5Catalog
    claim_boundary: str
    schema_version: str = RELATIONSHIP_PRODUCT_HORIZON_SOURCE_V5_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_PRODUCT_HORIZON_SOURCE_V5_SCHEMA_VERSION:
            raise ValueError("source-v5 schema version mismatch")
        if len(self.protocol_id) != 64 or any(
            character not in "0123456789abcdef" for character in self.protocol_id
        ):
            raise ValueError("source-v5 protocol identity must be lowercase SHA-256")
        if self.engine_protocol.protocol_id != self.protocol_id:
            raise ValueError("source-v5 engine projection protocol identity drifted")
        if self.engine_protocol.cohort_id != self.cohort_id:
            raise ValueError("source-v5 engine projection cohort drifted")
        if self.rendering_version != RELATIONSHIP_PRODUCT_HORIZON_PUBLIC_RENDERING_V5:
            raise ValueError("source-v5 rendering version drifted")
        _require_text(self.claim_boundary, "claim_boundary")


def relationship_product_horizon_source_v5_protocol_path() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1] / "lab_protocols" / _PROTOCOL_FILENAME


def _catalog_from_payload(
    raw: Mapping[str, Any],
    *,
    base: RelationshipProductHorizonSourceProtocol,
) -> RelationshipProductHorizonSourceV5Catalog:
    _require_exact_keys(
        raw,
        {
            "domain_contexts",
            "role_contexts",
            "condition_surfaces",
            "segment_contexts",
            "reflections",
            "neutral_contexts",
            "historical_reactions",
        },
        source="rendering.catalog",
    )

    def text_map(name: str, expected_keys: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
        value = _require_mapping(raw[name], f"rendering.catalog.{name}")
        _require_exact_keys(value, set(expected_keys), source=f"rendering.catalog.{name}")
        return tuple(
            (key, _require_text(value[key], f"rendering.catalog.{name}.{key}"))
            for key in expected_keys
        )

    def text_list_map(
        name: str,
        expected_keys: tuple[str, ...],
        *,
        count: int,
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        value = _require_mapping(raw[name], f"rendering.catalog.{name}")
        _require_exact_keys(value, set(expected_keys), source=f"rendering.catalog.{name}")
        return tuple(
            (
                key,
                _require_text_tuple(
                    value[key],
                    f"rendering.catalog.{name}.{key}",
                    count=count,
                ),
            )
            for key in expected_keys
        )

    catalog = RelationshipProductHorizonSourceV5Catalog(
        domain_contexts=text_map("domain_contexts", base.domain_ids),
        role_contexts=text_map("role_contexts", base.role_ids),
        condition_surfaces=text_list_map(
            "condition_surfaces", base.condition_ids, count=12
        ),
        segment_contexts=text_list_map(
            "segment_contexts",
            tuple(item.segment_id for item in base.segment_specs),
            count=4,
        ),
        reflections=_require_text_tuple(raw["reflections"], "rendering.catalog.reflections", count=12),
        neutral_contexts=_require_text_tuple(
            raw["neutral_contexts"], "rendering.catalog.neutral_contexts", count=12
        ),
        historical_reactions=text_map(
            "historical_reactions", ("helped", "felt_heard", "missed", "over_directive")
        ),
    )
    all_public_text = tuple(dict(catalog.domain_contexts).values()) + tuple(
        dict(catalog.role_contexts).values()
    ) + tuple(
        text for _key, values in catalog.condition_surfaces for text in values
    ) + tuple(
        text for _key, values in catalog.segment_contexts for text in values
    ) + catalog.reflections + catalog.neutral_contexts + tuple(
        dict(catalog.historical_reactions).values()
    )
    if len(set(all_public_text)) != len(all_public_text):
        raise ValueError("source-v5 public catalog entries must be globally unique")
    sealed_literals = (
        *base.condition_ids,
        *(item.segment_id for item in base.segment_specs),
        *(item.policy_id for item in base.policy_profiles),
    )
    if any(literal in text for literal in sealed_literals for text in all_public_text):
        raise ValueError("source-v5 public catalog leaked a sealed literal")
    if any(literal in text for literal in _PUBLIC_META_LITERALS for text in all_public_text):
        raise ValueError("source-v5 public catalog leaked a meta-evaluation literal")
    return catalog


def load_relationship_product_horizon_source_v5_protocol(
    protocol_path: pathlib.Path | None = None,
) -> RelationshipProductHorizonSourceV5Protocol:
    """Load the independent source-v5 protocol without model or outcome execution."""

    path = pathlib.Path(protocol_path or relationship_product_horizon_source_v5_protocol_path())
    raw_bytes = path.read_bytes()
    raw = _parse_unique_json(raw_bytes, path)
    _require_exact_keys(
        raw,
        {
            "schema_version",
            "owner",
            "base_source",
            "cohort",
            "rendering",
            "firewall",
            "claim_boundary",
        },
        source="source-v5 protocol",
    )
    if raw["schema_version"] != RELATIONSHIP_PRODUCT_HORIZON_SOURCE_V5_SCHEMA_VERSION:
        raise ValueError("source-v5 loader refuses another schema version")

    owner = _require_mapping(raw["owner"], "owner")
    _require_exact_keys(
        owner,
        {
            "module",
            "source_role",
            "causal_engine_dependency",
            "settlement_owner",
            "runtime_owner_added",
            "runtime_slot_added",
            "difficulty_tuned_from_prior_outcome",
        },
        source="owner",
    )
    if owner != {
        "module": _OWNER_MODULE,
        "source_role": "identity_disjoint_unconsumed_synthetic_source_candidate",
        "causal_engine_dependency": (
            "lifeform_domain_emogpt.lab.relationship_product_horizon_source_v4"
        ),
        "settlement_owner": (
            "lifeform_domain_emogpt.lab.environment.ReactiveRelationshipEnvironment"
        ),
        "runtime_owner_added": False,
        "runtime_slot_added": False,
        "difficulty_tuned_from_prior_outcome": False,
    }:
        raise ValueError("source-v5 owner boundary drifted")

    base_path = relationship_product_horizon_source_protocol_path()
    base_owner_path = pathlib.Path(__file__).with_name(
        "relationship_product_horizon_source_v4.py"
    )
    base_bytes = base_path.read_bytes()
    base_owner_bytes = base_owner_path.read_bytes()
    base = load_relationship_product_horizon_source_protocol(base_path)
    spent_public = build_relationship_product_horizon_public_view(base)
    spent_evaluator = build_relationship_product_horizon_evaluator_bundle(base)
    base_source = _require_mapping(raw["base_source"], "base_source")
    _require_exact_keys(
        base_source,
        {
            "schema_version",
            "current_engine_owner_raw_sha256",
            "current_engine_owner_raw_bytes",
            "protocol_raw_sha256",
            "protocol_raw_bytes",
            "protocol_id",
            "public_view_schema_version",
            "evaluator_schema_version",
            "spent_public_plan_sha256",
            "spent_sealed_bundle_sha256",
            "spent_reader_text_unique_count",
            "spent_reader_text_inventory_sha256",
        },
        source="base_source",
    )
    expected_base = {
        "schema_version": RELATIONSHIP_PRODUCT_HORIZON_SOURCE_SCHEMA_VERSION,
        "current_engine_owner_raw_sha256": _sha256_bytes(base_owner_bytes),
        "current_engine_owner_raw_bytes": len(base_owner_bytes),
        "protocol_raw_sha256": _sha256_bytes(base_bytes),
        "protocol_raw_bytes": len(base_bytes),
        "protocol_id": base.protocol_id,
        "public_view_schema_version": RELATIONSHIP_PRODUCT_HORIZON_PUBLIC_VIEW_SCHEMA_VERSION,
        "evaluator_schema_version": RELATIONSHIP_PRODUCT_HORIZON_EVALUATOR_SCHEMA_VERSION,
        "spent_public_plan_sha256": spent_public.public_plan_sha256,
        "spent_sealed_bundle_sha256": spent_evaluator.sealed_bundle_sha256,
        "spent_reader_text_unique_count": len(_reader_texts(spent_public)),
        "spent_reader_text_inventory_sha256": _reader_inventory_sha256(spent_public),
    }
    if base_source != expected_base:
        raise ValueError("source-v5 base causal engine pin drifted")

    cohort = _require_mapping(raw["cohort"], "cohort")
    _require_exact_keys(
        cohort,
        {
            "cohort_id",
            "evidence_role",
            "root_count",
            "onboarding_sessions_per_root",
            "collection_decisions_per_root",
            "evaluation_decisions_per_root",
            "master_seed_namespace",
            "environment_seed_namespace",
            "per_arm_exogenous_root_clone",
            "arm_identity_affects_source_or_environment_seed",
        },
        source="cohort",
    )
    if cohort["evidence_role"] != "identity_disjoint_unconsumed_synthetic_source_candidate":
        raise ValueError("source-v5 evidence role drifted")
    counts = (
        _require_int(cohort["root_count"], "cohort.root_count"),
        _require_int(
            cohort["onboarding_sessions_per_root"],
            "cohort.onboarding_sessions_per_root",
        ),
        _require_int(
            cohort["collection_decisions_per_root"],
            "cohort.collection_decisions_per_root",
        ),
        _require_int(
            cohort["evaluation_decisions_per_root"],
            "cohort.evaluation_decisions_per_root",
        ),
    )
    expected_counts = (
        base.root_count,
        base.onboarding_sessions_per_root,
        base.collection_decisions_per_root,
        base.evaluation_decisions_per_root,
    )
    if counts != expected_counts:
        raise ValueError("source-v5 causal horizon inventory drifted")
    if not _require_bool(
        cohort["per_arm_exogenous_root_clone"], "cohort.per_arm_exogenous_root_clone"
    ):
        raise ValueError("source-v5 requires per-arm exogenous root clones")
    if _require_bool(
        cohort["arm_identity_affects_source_or_environment_seed"],
        "cohort.arm_identity_affects_source_or_environment_seed",
    ):
        raise ValueError("source-v5 arm identity cannot enter source or environment seeds")
    master_namespace = _require_text(
        cohort["master_seed_namespace"], "cohort.master_seed_namespace"
    )
    environment_namespace = _require_text(
        cohort["environment_seed_namespace"], "cohort.environment_seed_namespace"
    )
    if master_namespace == base.master_seed_namespace or environment_namespace == base.environment_seed_namespace:
        raise ValueError("source-v5 seed namespaces must not reuse source-v4")

    rendering = _require_mapping(raw["rendering"], "rendering")
    _require_exact_keys(
        rendering,
        {
            "version",
            "minimum_public_source_characters_per_root",
            "minimum_public_source_utf8_bytes_per_root",
            "reader_text_exact_disjoint_from_spent_source_v4_required",
            "catalog",
        },
        source="rendering",
    )
    if rendering["version"] != RELATIONSHIP_PRODUCT_HORIZON_PUBLIC_RENDERING_V5:
        raise ValueError("source-v5 rendering version drifted")
    if not _require_bool(
        rendering["reader_text_exact_disjoint_from_spent_source_v4_required"],
        "rendering.reader_text_exact_disjoint_from_spent_source_v4_required",
    ):
        raise ValueError("source-v5 must require exact reader-text disjointness")
    minimum_characters = _require_int(
        rendering["minimum_public_source_characters_per_root"],
        "rendering.minimum_public_source_characters_per_root",
    )
    minimum_bytes = _require_int(
        rendering["minimum_public_source_utf8_bytes_per_root"],
        "rendering.minimum_public_source_utf8_bytes_per_root",
    )
    if minimum_characters < base.minimum_public_source_characters_per_root:
        raise ValueError("source-v5 public character pressure cannot be weaker than source-v4")
    if minimum_bytes < base.minimum_public_source_utf8_bytes_per_root:
        raise ValueError("source-v5 public byte pressure cannot be weaker than source-v4")
    catalog = _catalog_from_payload(
        _require_mapping(rendering["catalog"], "rendering.catalog"), base=base
    )

    firewall = _require_mapping(raw["firewall"], "firewall")
    expected_firewall = {
        "public_view_contains_sealed_condition": False,
        "public_view_contains_policy_or_preferred_action": False,
        "public_view_contains_current_decision_sealed_environment_truth": False,
        "collection_forced_action_owned_by_source": False,
        "evaluation_or_judge_feedback_to_learning": False,
        "source_v4_outcome_or_credit_used_to_generate_or_select_v5": False,
        "spent_v4_sealed_bundle_identity_rebuild_for_exclusion_only": True,
        "spent_v4_credit_record_read_count": 0,
        "public_historical_onboarding_outcome_record_count": 448,
        "sealed_evaluator_onboarding_truth_row_count": 448,
        "sealed_evaluator_decision_row_count": 5_376,
        "reactive_environment_construction_count": 0,
        "decision_action_outcome_settlement_count": 0,
        "prediction_error_count": 0,
        "credit_count": 0,
        "gate_update_count": 0,
        "reader_fit_count": 0,
        "embedding_inference_count": 0,
        "model_output_count": 0,
        "cuda_invocation_count": 0,
        "rehearsal_execution_count": 0,
        "exact_disjoint_from_spent_source_v4": True,
        "exact_disjoint_from_source_v3_and_all_adaptive_inputs": False,
        "semantic_novelty_established": False,
        "dgp_independence_established": False,
        "formal_evidence_authorized": False,
        "unseen_evidence_authorized": False,
        "integrated_execution_authorized": False,
        "human_sample_claimed": False,
    }
    _require_exact_keys(firewall, set(expected_firewall), source="firewall")
    if firewall != expected_firewall:
        raise ValueError("source-v5 firewall drifted")

    protocol_id = hashlib.sha256(canonical_json(raw).encode("utf-8")).hexdigest()
    cohort_id = _require_text(cohort["cohort_id"], "cohort.cohort_id")
    claim_boundary = _require_text(raw["claim_boundary"], "claim_boundary")
    engine_protocol = replace(
        base,
        protocol_id=protocol_id,
        cohort_id=cohort_id,
        master_seed_namespace=master_namespace,
        environment_seed_namespace=environment_namespace,
        minimum_public_source_characters_per_root=minimum_characters,
        minimum_public_source_utf8_bytes_per_root=minimum_bytes,
        claim_boundary=claim_boundary,
    )
    return RelationshipProductHorizonSourceV5Protocol(
        protocol_id=protocol_id,
        cohort_id=cohort_id,
        rendering_version=_require_text(rendering["version"], "rendering.version"),
        engine_protocol=engine_protocol,
        catalog=catalog,
        claim_boundary=claim_boundary,
    )


def _reader_texts(public: RelationshipProductHorizonPublicView) -> frozenset[str]:
    return frozenset(
        [
            session.user_utterance
            for root in public.roots
            for session in root.onboarding_sessions
        ]
        + [
            session.current_input
            for root in public.roots
            for session in root.decision_sessions
        ]
    )


def relationship_product_horizon_source_v5_reader_text_inventory(
    public: RelationshipProductHorizonPublicView,
    *,
    protocol: RelationshipProductHorizonSourceV5Protocol,
) -> tuple[tuple[str, str], ...]:
    """Publish the exact unique public reader texts as immutable digest/text pairs."""

    if public.protocol_id != protocol.protocol_id or public.cohort_id != protocol.cohort_id:
        raise ValueError("source-v5 reader inventory public lineage drifted")
    by_digest: dict[str, str] = {}
    for value in _reader_texts(public):
        digest = _sha256_text(value)
        existing = by_digest.setdefault(digest, value)
        if existing != value:
            raise RuntimeError("SHA-256 collision in source-v5 reader text inventory")
    return tuple(sorted(by_digest.items()))


def _build_public_root(
    *,
    protocol: RelationshipProductHorizonSourceV5Protocol,
    root_index: int,
    evaluator: RelationshipProductHorizonEvaluatorBundle,
) -> HorizonPublicRoot:
    manifest = evaluator.root_manifests[root_index]
    subject_id = manifest.subject_id
    onboarding_rows = tuple(
        item for item in evaluator.onboarding_sessions if item.subject_id == subject_id
    )
    decision_rows = tuple(
        item for item in evaluator.decision_sessions if item.subject_id == subject_id
    )
    catalog = protocol.catalog

    onboarding: list[HorizonPublicOnboardingSession] = []
    for row in onboarding_rows:
        surface_index = _stable_index(
            protocol.protocol_id, "onboarding-surface", row.session_id, 12
        )
        reflection_index = _stable_index(
            protocol.protocol_id, "onboarding-reflection", row.session_id, 12
        )
        domain_index = _stable_index(
            protocol.protocol_id, "onboarding-domain", row.session_id, len(protocol.engine_protocol.domain_ids)
        )
        role_index = _stable_index(
            protocol.protocol_id, "onboarding-role", row.session_id, len(protocol.engine_protocol.role_ids)
        )
        neutral_index = _stable_index(
            protocol.protocol_id, "onboarding-neutral", row.session_id, 12
        )
        onboarding.append(
            HorizonPublicOnboardingSession(
                session_id=row.session_id,
                session_index=row.session_index,
                virtual_day=row.virtual_day,
                public_context_chunk="".join(
                    (
                        catalog.domain(protocol.engine_protocol.domain_ids[domain_index]),
                        catalog.role(protocol.engine_protocol.role_ids[role_index]),
                        catalog.neutral_contexts[neutral_index],
                    )
                ),
                user_utterance="".join(
                    (
                        "这条较早记录保留的是当时的原话。",
                        catalog.condition(row.condition_id)[surface_index],
                        catalog.reflections[reflection_index],
                    )
                ),
                exposed_action_id=row.exposed_action_id,
                observed_outcome_id=row.observed_outcome_id,
                rendered_user_reaction=catalog.reaction(row.observed_outcome_id),
            )
        )

    decisions: list[HorizonPublicDecisionSession] = []
    by_index = {item.decision_index: item for item in decision_rows}
    for row in decision_rows:
        surface_index = int(row.surface_recipe_id.rsplit("-", 1)[1])
        segment_index = _stable_index(
            protocol.protocol_id, "segment-context", row.session_id, 4
        )
        reflection_index = _stable_index(
            protocol.protocol_id, "decision-reflection", row.session_id, 12
        )
        neutral_index = _stable_index(
            protocol.protocol_id, "decision-neutral", row.session_id, 12
        )
        correction_target = (
            by_index[row.correction_target_index].session_id
            if row.correction_target_index is not None
            else None
        )
        decisions.append(
            HorizonPublicDecisionSession(
                session_id=row.session_id,
                decision_id=row.decision_id,
                decision_index=row.decision_index,
                virtual_day=row.virtual_day,
                public_context_chunk="".join(
                    (
                        catalog.domain(row.domain_id),
                        catalog.role(row.role_id),
                        f"这是这段连续记录里的第{row.decision_index + 1}次当下更新；这次我没有把之前的经过全部重述。",
                        catalog.neutral_contexts[neutral_index],
                    )
                ),
                current_input="".join(
                    (
                        catalog.condition(row.condition_id)[surface_index],
                        catalog.segment(row.segment_id)[segment_index],
                        catalog.reflections[reflection_index],
                    )
                ),
                public_correction_target_session_id=correction_target,
                action_surface=_ACTION_SURFACE,
            )
        )
    root = HorizonPublicRoot(
        subject_id=subject_id,
        onboarding_sessions=tuple(onboarding),
        decision_sessions=tuple(decisions),
    )
    if root.public_source_characters < protocol.engine_protocol.minimum_public_source_characters_per_root:
        raise ValueError("source-v5 public character pressure fell below the frozen minimum")
    if root.public_source_utf8_bytes < protocol.engine_protocol.minimum_public_source_utf8_bytes_per_root:
        raise ValueError("source-v5 public UTF-8 pressure fell below the frozen minimum")
    return root


def _assert_disjoint_from_spent_source_v4(
    *,
    public: RelationshipProductHorizonPublicView,
    evaluator: RelationshipProductHorizonEvaluatorBundle,
) -> None:
    spent_public = build_relationship_product_horizon_public_view()
    spent_evaluator = build_relationship_product_horizon_evaluator_bundle()
    if _reader_texts(public) & _reader_texts(spent_public):
        raise ValueError("source-v5 reader texts overlap spent source-v4")

    identity_sets = (
        ({item.subject_id for item in public.roots}, {item.subject_id for item in spent_public.roots}),
        (
            {item.public_trajectory_sha256 for item in public.roots},
            {item.public_trajectory_sha256 for item in spent_public.roots},
        ),
        (
            {
                item.session_id
                for root in public.roots
                for item in (*root.onboarding_sessions, *root.decision_sessions)
            },
            {
                item.session_id
                for root in spent_public.roots
                for item in (*root.onboarding_sessions, *root.decision_sessions)
            },
        ),
        (
            {item.decision_id for root in public.roots for item in root.decision_sessions},
            {item.decision_id for root in spent_public.roots for item in root.decision_sessions},
        ),
        (
            {item.root_seed for item in evaluator.root_manifests},
            {item.root_seed for item in spent_evaluator.root_manifests},
        ),
        (
            {item.tape_seed for item in evaluator.root_manifests},
            {item.tape_seed for item in spent_evaluator.root_manifests},
        ),
        (
            {item.world_clone_id for item in evaluator.root_manifests},
            {item.world_clone_id for item in spent_evaluator.root_manifests},
        ),
        (
            {item.causal_tape_signature for item in evaluator.root_manifests},
            {item.causal_tape_signature for item in spent_evaluator.root_manifests},
        ),
        (
            {item.scene_id for item in evaluator.decision_sessions},
            {item.scene_id for item in spent_evaluator.decision_sessions},
        ),
        (
            {item.environment_seed for item in evaluator.decision_sessions},
            {item.environment_seed for item in spent_evaluator.decision_sessions},
        ),
    )
    if any(left & right for left, right in identity_sets):
        raise ValueError("source-v5 causal identity overlaps spent source-v4")


def _assert_no_meta_evaluation_literals(value: object) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _assert_no_meta_evaluation_literals(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_meta_evaluation_literals(child)
    elif isinstance(value, str):
        leaked = tuple(literal for literal in _PUBLIC_META_LITERALS if literal in value)
        if leaked:
            raise ValueError(f"source-v5 public payload leaked meta-evaluation literals: {leaked}")


def build_relationship_product_horizon_source_v5_projections(
    protocol: RelationshipProductHorizonSourceV5Protocol | None = None,
) -> tuple[RelationshipProductHorizonPublicView, RelationshipProductHorizonEvaluatorBundle]:
    """Build exact public and sealed projections without settling any outcome."""

    source = protocol or load_relationship_product_horizon_source_v5_protocol()
    base_evaluator = build_relationship_product_horizon_evaluator_bundle(source.engine_protocol)
    roots = tuple(
        _build_public_root(protocol=source, root_index=index, evaluator=base_evaluator)
        for index in range(source.engine_protocol.root_count)
    )
    public = RelationshipProductHorizonPublicView(
        protocol_id=source.protocol_id,
        cohort_id=source.cohort_id,
        roots=roots,
    )
    _assert_no_meta_evaluation_literals(public.to_sut_payload())
    root_manifests = tuple(
        replace(manifest, public_trajectory_sha256=root.public_trajectory_sha256)
        for manifest, root in zip(base_evaluator.root_manifests, roots, strict=True)
    )
    evaluator = replace(base_evaluator, root_manifests=root_manifests)
    if tuple(item.subject_id for item in public.roots) != tuple(
        item.subject_id for item in evaluator.root_manifests
    ):
        raise ValueError("source-v5 public/evaluator subject join drifted")
    if tuple(item.public_trajectory_sha256 for item in evaluator.root_manifests) != tuple(
        root.public_trajectory_sha256 for root in public.roots
    ):
        raise ValueError("source-v5 public/evaluator trajectory join drifted")
    _assert_disjoint_from_spent_source_v4(public=public, evaluator=evaluator)
    return public, evaluator


def build_relationship_product_horizon_source_v5_public_view(
    protocol: RelationshipProductHorizonSourceV5Protocol | None = None,
) -> RelationshipProductHorizonPublicView:
    return build_relationship_product_horizon_source_v5_projections(protocol)[0]


def build_relationship_product_horizon_source_v5_evaluator_bundle(
    protocol: RelationshipProductHorizonSourceV5Protocol | None = None,
) -> RelationshipProductHorizonEvaluatorBundle:
    return build_relationship_product_horizon_source_v5_projections(protocol)[1]


def build_relationship_product_horizon_source_v5_environment(
    evaluator_bundle: RelationshipProductHorizonEvaluatorBundle,
    *,
    subject_id: str,
) -> ReactiveRelationshipEnvironment:
    return build_relationship_product_horizon_environment(
        evaluator_bundle, subject_id=subject_id
    )


__all__ = [
    "RELATIONSHIP_PRODUCT_HORIZON_PUBLIC_RENDERING_V5",
    "RELATIONSHIP_PRODUCT_HORIZON_SOURCE_V5_SCHEMA_VERSION",
    "RelationshipProductHorizonSourceV5Catalog",
    "RelationshipProductHorizonSourceV5Protocol",
    "build_relationship_product_horizon_source_v5_environment",
    "build_relationship_product_horizon_source_v5_evaluator_bundle",
    "build_relationship_product_horizon_source_v5_projections",
    "build_relationship_product_horizon_source_v5_public_view",
    "load_relationship_product_horizon_source_v5_protocol",
    "relationship_product_horizon_source_v5_protocol_path",
    "relationship_product_horizon_source_v5_reader_text_inventory",
]
