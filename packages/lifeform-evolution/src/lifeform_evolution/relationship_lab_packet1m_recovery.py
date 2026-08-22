"""Auditable renderer-transport recovery for P1m generation.

The recovery is authorized only because generation protocol v1 failed before
one accepted scenario rendering and before any qualification answer.  It binds
the same semantic recipe and pair plan, requires a non-scenario real-model
preflight, and durably records every raw field before JSON composition.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import pathlib
import tempfile

from lifeform_domain_emogpt.lab import (
    RELATIONSHIP_P1M_FIELD_COUNT,
    RELATIONSHIP_P1M_PREFLIGHT_FIELD_COUNT,
    RELATIONSHIP_TRANSFER_P1M_V1_PACKAGE_NAME,
    RelationshipP1mFieldOutput,
    RelationshipP1mFieldPlan,
    RelationshipP1mGenerationRecipe,
    RelationshipP1mPairPlan,
    RelationshipP1mRendererTransport,
    canonical_json,
    sha256_json,
    validate_relationship_p1m_field_output,
)
from lifeform_evolution.relationship_lab_packet1m import (
    RELATIONSHIP_P1M_SOURCE_VERDICT,
    relationship_p1m_pair_plan_sha256,
)


RELATIONSHIP_P1M_PREFLIGHT_SCHEMA_VERSION = "relationship-p1m-renderer-preflight.v5"
RELATIONSHIP_P1M_RECOVERY_PROTOCOL_SCHEMA_VERSION = (
    "relationship-p1m-generation-recovery-protocol.v5"
)
RELATIONSHIP_P1M_FIELD_BATCH_SCHEMA_VERSION = "relationship-p1m-field-batch.v5"
RELATIONSHIP_P1M_RAW_FIELD_ATTEMPT_SCHEMA_VERSION = (
    "relationship-p1m-raw-field-attempt.v5"
)
RELATIONSHIP_P1M_RECOVERY_NEXT_ACTION = (
    "materialize_same_fsm_package_then_freeze_first_qualification_consumer"
)

_HEX_DIGITS = frozenset("0123456789abcdef")
_RECOVERY_CLAIM_BOUNDARY = (
    "P1m final generation recovery is limited to renderer transport after four "
    "preserved attempts ended before any accepted scenario or consumer output. "
    "It binds the same semantic recipe, pair plan, truth, pair count, statistical "
    "gates, and stop rule; a deterministic typed surface realizer now replaces "
    "stochastic paraphrasing and every raw field must match its expected hash. "
    "Generation success remains instrument preparation, not evidence of any "
    "four-able axis."
)


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_sha256(value: object, field_name: str) -> str:
    text = _require_text(value, field_name)
    if len(text) != 64 or any(char not in _HEX_DIGITS for char in text):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return text


def _require_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be boolean")
    return value


def _require_timestamp(value: object, field_name: str) -> str:
    text = _require_text(value, field_name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return text


def _require_exact_keys(
    value: object,
    expected: set[str],
    *,
    field_name: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValueError(
            f"{field_name} fields do not match; missing={missing}, extra={extra}"
        )
    return value


def _atomic_write_text(path: pathlib.Path, content: str) -> None:
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(content)
        handle.flush()
        temporary = pathlib.Path(handle.name)
    temporary.replace(target)


def relationship_p1m_field_plan_sha256(
    plans: tuple[RelationshipP1mFieldPlan, ...],
) -> str:
    return sha256_json(
        [
            {
                "field_key": item.field_key,
                "field_kind": item.field_kind,
                "renderer_input_sha256": item.renderer_input_sha256,
                "expected_output_sha256": item.expected_output_sha256,
                "seed": item.seed,
                "minimum_length": item.minimum_length,
                "maximum_length": item.maximum_length,
                "forbidden_tokens": list(item.forbidden_tokens),
            }
            for item in plans
        ]
    )


@dataclass(frozen=True)
class RelationshipP1mRendererPreflightReport:
    created_at_iso: str
    transport_id: str
    model_id: str
    weights_sha256: str
    generation_config_sha256: str
    surface_seed_inventory_sha256: str
    runtime_device: str
    torch_dtype: str
    field_plan_sha256: str
    outputs: tuple[RelationshipP1mFieldOutput, ...]
    required_valid_rate: float
    valid_rate: float
    passed: bool
    evidence_role: str
    scenario_outputs: int
    consumer_outputs: int
    schema_version: str = RELATIONSHIP_P1M_PREFLIGHT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1M_PREFLIGHT_SCHEMA_VERSION:
            raise ValueError("P1m renderer preflight schema mismatch")
        _require_timestamp(self.created_at_iso, "P1m preflight created_at")
        for field_name, value in (
            ("transport_id", self.transport_id),
            ("weights_sha256", self.weights_sha256),
            ("generation_config_sha256", self.generation_config_sha256),
            (
                "surface_seed_inventory_sha256",
                self.surface_seed_inventory_sha256,
            ),
            ("field_plan_sha256", self.field_plan_sha256),
        ):
            _require_sha256(value, field_name)
        for value in (self.model_id, self.runtime_device, self.torch_dtype):
            _require_text(value, "P1m preflight runtime field")
        if len(self.outputs) != RELATIONSHIP_P1M_PREFLIGHT_FIELD_COUNT:
            raise ValueError("P1m preflight output count mismatch")
        if self.required_valid_rate != 1.0 or self.valid_rate != 1.0 or not self.passed:
            raise ValueError("P1m renderer preflight must pass 4/4")
        if self.evidence_role != "non_scenario_transport_only":
            raise ValueError("P1m preflight evidence role drift")
        if self.scenario_outputs != 0 or self.consumer_outputs != 0:
            raise ValueError("P1m preflight cannot contain scenario/consumer output")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "created_at_iso": self.created_at_iso,
            "transport_id": self.transport_id,
            "model_id": self.model_id,
            "weights_sha256": self.weights_sha256,
            "generation_config_sha256": self.generation_config_sha256,
            "surface_seed_inventory_sha256": (
                self.surface_seed_inventory_sha256
            ),
            "runtime_device": self.runtime_device,
            "torch_dtype": self.torch_dtype,
            "field_plan_sha256": self.field_plan_sha256,
            "outputs": [item.to_payload() for item in self.outputs],
            "required_valid_rate": self.required_valid_rate,
            "valid_rate": self.valid_rate,
            "passed": self.passed,
            "evidence_role": self.evidence_role,
            "scenario_outputs": self.scenario_outputs,
            "consumer_outputs": self.consumer_outputs,
        }

    @property
    def artifact_id(self) -> str:
        return sha256_json(self.to_payload())


def build_relationship_p1m_renderer_preflight_report(
    *,
    transport: RelationshipP1mRendererTransport,
    field_plans: tuple[RelationshipP1mFieldPlan, ...],
    outputs: tuple[RelationshipP1mFieldOutput, ...],
    model_id: str,
    weights_sha256: str,
    generation_config_sha256: str,
    runtime_device: str,
    torch_dtype: str,
    created_at_iso: str,
) -> RelationshipP1mRendererPreflightReport:
    if len(field_plans) != len(outputs):
        raise ValueError("P1m preflight plans/outputs mismatch")
    for plan, output in zip(field_plans, outputs, strict=True):
        if (
            plan.field_key != output.field_key
            or plan.renderer_input_sha256 != output.renderer_input_sha256
            or plan.seed != output.seed
        ):
            raise ValueError("P1m preflight output lineage mismatch")
    return RelationshipP1mRendererPreflightReport(
        created_at_iso=created_at_iso,
        transport_id=transport.transport_id,
        model_id=model_id,
        weights_sha256=weights_sha256,
        generation_config_sha256=generation_config_sha256,
        surface_seed_inventory_sha256=(
            transport.surface_seed_inventory_sha256
        ),
        runtime_device=runtime_device,
        torch_dtype=torch_dtype,
        field_plan_sha256=relationship_p1m_field_plan_sha256(field_plans),
        outputs=outputs,
        required_valid_rate=transport.preflight_required_valid_rate,
        valid_rate=1.0,
        passed=True,
        evidence_role=transport.preflight_evidence_role,
        scenario_outputs=0,
        consumer_outputs=0,
    )


def write_relationship_p1m_renderer_preflight_report(
    report: RelationshipP1mRendererPreflightReport,
    *,
    output_dir: pathlib.Path,
) -> pathlib.Path:
    path = pathlib.Path(output_dir) / "renderer_preflight.json"
    _atomic_write_text(
        path,
        json.dumps(
            {**report.to_payload(), "artifact_id": report.artifact_id},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return path


def load_relationship_p1m_renderer_preflight_report(
    path: pathlib.Path,
    *,
    field_plans: tuple[RelationshipP1mFieldPlan, ...],
) -> RelationshipP1mRendererPreflightReport:
    raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    root = _require_exact_keys(
        raw,
        {
            "schema_version",
            "created_at_iso",
            "transport_id",
            "model_id",
            "weights_sha256",
            "generation_config_sha256",
            "surface_seed_inventory_sha256",
            "runtime_device",
            "torch_dtype",
            "field_plan_sha256",
            "outputs",
            "required_valid_rate",
            "valid_rate",
            "passed",
            "evidence_role",
            "scenario_outputs",
            "consumer_outputs",
            "artifact_id",
        },
        field_name="P1m renderer preflight",
    )
    raw_outputs = root["outputs"]
    if not isinstance(raw_outputs, list) or len(raw_outputs) != len(field_plans):
        raise ValueError("P1m preflight durable outputs mismatch")
    outputs: list[RelationshipP1mFieldOutput] = []
    for raw_output, plan in zip(raw_outputs, field_plans, strict=True):
        item = _require_exact_keys(
            raw_output,
            {
                "field_key",
                "renderer_input_sha256",
                "seed",
                "raw_output",
                "normalized_text",
            },
            field_name="P1m preflight output",
        )
        output = validate_relationship_p1m_field_output(
            _require_text(item["raw_output"], "P1m preflight raw output"),
            plan=plan,
        )
        if output.to_payload() != item:
            raise ValueError("P1m preflight output payload drift")
        outputs.append(output)
    report = RelationshipP1mRendererPreflightReport(
        created_at_iso=_require_timestamp(root["created_at_iso"], "created_at"),
        transport_id=_require_sha256(root["transport_id"], "transport_id"),
        model_id=_require_text(root["model_id"], "model_id"),
        weights_sha256=_require_sha256(root["weights_sha256"], "weights"),
        generation_config_sha256=_require_sha256(
            root["generation_config_sha256"], "generation config"
        ),
        surface_seed_inventory_sha256=_require_sha256(
            root["surface_seed_inventory_sha256"], "surface seed inventory"
        ),
        runtime_device=_require_text(root["runtime_device"], "runtime device"),
        torch_dtype=_require_text(root["torch_dtype"], "torch dtype"),
        field_plan_sha256=_require_sha256(
            root["field_plan_sha256"], "field plan"
        ),
        outputs=tuple(outputs),
        required_valid_rate=float(root["required_valid_rate"]),
        valid_rate=float(root["valid_rate"]),
        passed=_require_bool(root["passed"], "passed"),
        evidence_role=_require_text(root["evidence_role"], "evidence role"),
        scenario_outputs=_require_int(root["scenario_outputs"], "scenario outputs"),
        consumer_outputs=_require_int(root["consumer_outputs"], "consumer outputs"),
        schema_version=_require_text(root["schema_version"], "schema version"),
    )
    if report.artifact_id != _require_sha256(root["artifact_id"], "artifact id"):
        raise ValueError("P1m preflight artifact id mismatch")
    if report.field_plan_sha256 != relationship_p1m_field_plan_sha256(field_plans):
        raise ValueError("P1m preflight field plan drift")
    return report


@dataclass(frozen=True)
class RelationshipP1mGenerationRecoveryProtocol:
    frozen_at_iso: str
    source_p1k_report_artifact_id: str
    source_p1k_verdict: str
    source_failed_protocol_id: str
    source_incident_sha256: str
    recipe_id: str
    package_name: str
    pair_count: int
    pair_plan_sha256: str
    transport_id: str
    preflight_artifact_id: str
    model_source: str
    model_id: str
    weights_sha256: str
    prompt_sha256: str
    output_schema_sha256: str
    surface_seed_inventory_sha256: str
    generation_config_sha256: str
    runtime_device: str
    torch_dtype: str
    accepted_scenario_renderings_before_freeze: int
    consumer_outputs_before_freeze: int
    semantic_recipe_changed: bool
    qualification_gate_changed: bool
    evaluation_feedback_allowed: bool
    claim_boundary: str
    next_action: str = RELATIONSHIP_P1M_RECOVERY_NEXT_ACTION
    schema_version: str = RELATIONSHIP_P1M_RECOVERY_PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1M_RECOVERY_PROTOCOL_SCHEMA_VERSION:
            raise ValueError("P1m recovery protocol schema mismatch")
        _require_timestamp(self.frozen_at_iso, "P1m recovery frozen_at")
        for field_name, value in (
            ("source_p1k_report_artifact_id", self.source_p1k_report_artifact_id),
            ("source_failed_protocol_id", self.source_failed_protocol_id),
            ("source_incident_sha256", self.source_incident_sha256),
            ("recipe_id", self.recipe_id),
            ("pair_plan_sha256", self.pair_plan_sha256),
            ("transport_id", self.transport_id),
            ("preflight_artifact_id", self.preflight_artifact_id),
            ("weights_sha256", self.weights_sha256),
            ("prompt_sha256", self.prompt_sha256),
            ("output_schema_sha256", self.output_schema_sha256),
            (
                "surface_seed_inventory_sha256",
                self.surface_seed_inventory_sha256,
            ),
            ("generation_config_sha256", self.generation_config_sha256),
        ):
            _require_sha256(value, field_name)
        if self.source_p1k_verdict != RELATIONSHIP_P1M_SOURCE_VERDICT:
            raise ValueError("P1m recovery source P1k verdict mismatch")
        if self.package_name != RELATIONSHIP_TRANSFER_P1M_V1_PACKAGE_NAME:
            raise ValueError("P1m recovery package mismatch")
        if self.pair_count != 24:
            raise ValueError("P1m recovery pair count changed")
        for value in (
            self.model_source,
            self.model_id,
            self.runtime_device,
            self.torch_dtype,
        ):
            _require_text(value, "P1m recovery renderer field")
        if self.accepted_scenario_renderings_before_freeze != 0:
            raise ValueError("P1m recovery must freeze before accepted scenario output")
        if self.consumer_outputs_before_freeze != 0:
            raise ValueError("P1m recovery must freeze before qualification output")
        if (
            self.semantic_recipe_changed
            or self.qualification_gate_changed
            or self.evaluation_feedback_allowed
        ):
            raise ValueError("P1m recovery exceeded renderer-only scope")
        if self.claim_boundary != _RECOVERY_CLAIM_BOUNDARY:
            raise ValueError("P1m recovery claim boundary drift")
        if self.next_action != RELATIONSHIP_P1M_RECOVERY_NEXT_ACTION:
            raise ValueError("P1m recovery next action drift")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "frozen_at_iso": self.frozen_at_iso,
            "source": {
                "p1k_report_artifact_id": self.source_p1k_report_artifact_id,
                "p1k_verdict": self.source_p1k_verdict,
                "failed_generation_protocol_id": self.source_failed_protocol_id,
                "generation_incident_sha256": self.source_incident_sha256,
            },
            "semantic_freeze": {
                "recipe_id": self.recipe_id,
                "package_name": self.package_name,
                "pair_count": self.pair_count,
                "pair_plan_sha256": self.pair_plan_sha256,
                "semantic_recipe_changed": self.semantic_recipe_changed,
                "qualification_gate_changed": self.qualification_gate_changed,
            },
            "transport": {
                "transport_id": self.transport_id,
                "preflight_artifact_id": self.preflight_artifact_id,
                "model_source": self.model_source,
                "model_id": self.model_id,
                "weights_sha256": self.weights_sha256,
                "prompt_sha256": self.prompt_sha256,
                "output_schema_sha256": self.output_schema_sha256,
                "surface_seed_inventory_sha256": (
                    self.surface_seed_inventory_sha256
                ),
                "generation_config_sha256": self.generation_config_sha256,
                "runtime_device": self.runtime_device,
                "torch_dtype": self.torch_dtype,
            },
            "freeze_guards": {
                "accepted_scenario_renderings_before_freeze": (
                    self.accepted_scenario_renderings_before_freeze
                ),
                "consumer_outputs_before_freeze": self.consumer_outputs_before_freeze,
                "evaluation_feedback_allowed": self.evaluation_feedback_allowed,
            },
            "claim_boundary": self.claim_boundary,
            "next_action": self.next_action,
        }

    @property
    def protocol_id(self) -> str:
        return sha256_json(self.to_payload())


def freeze_relationship_p1m_generation_recovery_protocol(
    *,
    recipe: RelationshipP1mGenerationRecipe,
    pair_plans: tuple[RelationshipP1mPairPlan, ...],
    transport: RelationshipP1mRendererTransport,
    preflight: RelationshipP1mRendererPreflightReport,
    source_p1k_report_artifact_id: str,
    source_p1k_verdict: str,
    source_incident_sha256: str,
    weights_sha256: str,
    runtime_device: str,
    torch_dtype: str,
    frozen_at_iso: str,
) -> RelationshipP1mGenerationRecoveryProtocol:
    if transport.source_recipe_id != recipe.recipe_id:
        raise ValueError("P1m recovery transport/recipe mismatch")
    if preflight.transport_id != transport.transport_id or not preflight.passed:
        raise ValueError("P1m recovery requires passing bound transport preflight")
    if (
        preflight.weights_sha256 != weights_sha256
        or preflight.runtime_device != runtime_device
        or preflight.torch_dtype != torch_dtype
        or preflight.model_id != transport.model_id
        or preflight.surface_seed_inventory_sha256
        != transport.surface_seed_inventory_sha256
        or preflight.generation_config_sha256
        != transport.generation_config_sha256
    ):
        raise ValueError("P1m recovery renderer differs from preflight")
    return RelationshipP1mGenerationRecoveryProtocol(
        frozen_at_iso=frozen_at_iso,
        source_p1k_report_artifact_id=source_p1k_report_artifact_id,
        source_p1k_verdict=source_p1k_verdict,
        source_failed_protocol_id=transport.source_failed_generation_protocol_id,
        source_incident_sha256=source_incident_sha256,
        recipe_id=recipe.recipe_id,
        package_name=recipe.package_name,
        pair_count=recipe.pair_count,
        pair_plan_sha256=relationship_p1m_pair_plan_sha256(pair_plans),
        transport_id=transport.transport_id,
        preflight_artifact_id=preflight.artifact_id,
        model_source=transport.model_source,
        model_id=transport.model_id,
        weights_sha256=weights_sha256,
        prompt_sha256=transport.prompt_sha256,
        output_schema_sha256=transport.output_schema_sha256,
        surface_seed_inventory_sha256=(
            transport.surface_seed_inventory_sha256
        ),
        generation_config_sha256=transport.generation_config_sha256,
        runtime_device=runtime_device,
        torch_dtype=torch_dtype,
        accepted_scenario_renderings_before_freeze=0,
        consumer_outputs_before_freeze=0,
        semantic_recipe_changed=False,
        qualification_gate_changed=False,
        evaluation_feedback_allowed=False,
        claim_boundary=_RECOVERY_CLAIM_BOUNDARY,
    )


def write_relationship_p1m_generation_recovery_protocol(
    protocol: RelationshipP1mGenerationRecoveryProtocol,
    *,
    output_dir: pathlib.Path,
) -> pathlib.Path:
    path = pathlib.Path(output_dir) / "generation_protocol.json"
    _atomic_write_text(
        path,
        json.dumps(
            {**protocol.to_payload(), "protocol_id": protocol.protocol_id},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return path


def load_relationship_p1m_generation_recovery_protocol(
    path: pathlib.Path,
) -> RelationshipP1mGenerationRecoveryProtocol:
    raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    root = _require_exact_keys(
        raw,
        {
            "schema_version",
            "frozen_at_iso",
            "source",
            "semantic_freeze",
            "transport",
            "freeze_guards",
            "claim_boundary",
            "next_action",
            "protocol_id",
        },
        field_name="P1m recovery protocol",
    )
    source = _require_exact_keys(
        root["source"],
        {
            "p1k_report_artifact_id",
            "p1k_verdict",
            "failed_generation_protocol_id",
            "generation_incident_sha256",
        },
        field_name="P1m recovery source",
    )
    semantic = _require_exact_keys(
        root["semantic_freeze"],
        {
            "recipe_id",
            "package_name",
            "pair_count",
            "pair_plan_sha256",
            "semantic_recipe_changed",
            "qualification_gate_changed",
        },
        field_name="P1m recovery semantic freeze",
    )
    transport = _require_exact_keys(
        root["transport"],
        {
            "transport_id",
            "preflight_artifact_id",
            "model_source",
            "model_id",
            "weights_sha256",
            "prompt_sha256",
            "output_schema_sha256",
            "surface_seed_inventory_sha256",
            "generation_config_sha256",
            "runtime_device",
            "torch_dtype",
        },
        field_name="P1m recovery transport",
    )
    guards = _require_exact_keys(
        root["freeze_guards"],
        {
            "accepted_scenario_renderings_before_freeze",
            "consumer_outputs_before_freeze",
            "evaluation_feedback_allowed",
        },
        field_name="P1m recovery guards",
    )
    protocol = RelationshipP1mGenerationRecoveryProtocol(
        frozen_at_iso=_require_timestamp(root["frozen_at_iso"], "frozen_at"),
        source_p1k_report_artifact_id=_require_sha256(
            source["p1k_report_artifact_id"], "P1m source report"
        ),
        source_p1k_verdict=_require_text(source["p1k_verdict"], "P1m verdict"),
        source_failed_protocol_id=_require_sha256(
            source["failed_generation_protocol_id"], "P1m failed protocol"
        ),
        source_incident_sha256=_require_sha256(
            source["generation_incident_sha256"], "P1m incident"
        ),
        recipe_id=_require_sha256(semantic["recipe_id"], "P1m recipe"),
        package_name=_require_text(semantic["package_name"], "P1m package"),
        pair_count=_require_int(semantic["pair_count"], "P1m pair count"),
        pair_plan_sha256=_require_sha256(
            semantic["pair_plan_sha256"], "P1m pair plan"
        ),
        transport_id=_require_sha256(transport["transport_id"], "P1m transport"),
        preflight_artifact_id=_require_sha256(
            transport["preflight_artifact_id"], "P1m preflight"
        ),
        model_source=_require_text(transport["model_source"], "P1m model source"),
        model_id=_require_text(transport["model_id"], "P1m model id"),
        weights_sha256=_require_sha256(transport["weights_sha256"], "P1m weights"),
        prompt_sha256=_require_sha256(transport["prompt_sha256"], "P1m prompt"),
        output_schema_sha256=_require_sha256(
            transport["output_schema_sha256"], "P1m output schema"
        ),
        surface_seed_inventory_sha256=_require_sha256(
            transport["surface_seed_inventory_sha256"],
            "P1m surface seed inventory",
        ),
        generation_config_sha256=_require_sha256(
            transport["generation_config_sha256"], "P1m generation config"
        ),
        runtime_device=_require_text(transport["runtime_device"], "P1m device"),
        torch_dtype=_require_text(transport["torch_dtype"], "P1m dtype"),
        accepted_scenario_renderings_before_freeze=_require_int(
            guards["accepted_scenario_renderings_before_freeze"],
            "P1m prior scenario renderings",
        ),
        consumer_outputs_before_freeze=_require_int(
            guards["consumer_outputs_before_freeze"], "P1m prior consumer outputs"
        ),
        semantic_recipe_changed=_require_bool(
            semantic["semantic_recipe_changed"], "P1m semantic changed"
        ),
        qualification_gate_changed=_require_bool(
            semantic["qualification_gate_changed"], "P1m gate changed"
        ),
        evaluation_feedback_allowed=_require_bool(
            guards["evaluation_feedback_allowed"], "P1m feedback"
        ),
        claim_boundary=_require_text(root["claim_boundary"], "P1m claim boundary"),
        next_action=_require_text(root["next_action"], "P1m next action"),
        schema_version=_require_text(root["schema_version"], "P1m schema version"),
    )
    if protocol.protocol_id != _require_sha256(root["protocol_id"], "protocol id"):
        raise ValueError("P1m recovery protocol id mismatch")
    return protocol


@dataclass(frozen=True)
class RelationshipP1mFieldBatchRecord:
    protocol_id: str
    record_index: int
    pair_id: str
    field_plan_sha256: str
    outputs: tuple[RelationshipP1mFieldOutput, ...]
    schema_version: str = RELATIONSHIP_P1M_FIELD_BATCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1M_FIELD_BATCH_SCHEMA_VERSION:
            raise ValueError("P1m field batch schema mismatch")
        _require_sha256(self.protocol_id, "P1m field batch protocol")
        _require_sha256(self.field_plan_sha256, "P1m field batch plan")
        if not 0 <= self.record_index < 24:
            raise ValueError("P1m field batch index out of range")
        _require_text(self.pair_id, "P1m field batch pair")
        if len(self.outputs) != RELATIONSHIP_P1M_FIELD_COUNT:
            raise ValueError("P1m field batch requires 13 outputs")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "record_index": self.record_index,
            "pair_id": self.pair_id,
            "field_plan_sha256": self.field_plan_sha256,
            "outputs": [item.to_payload() for item in self.outputs],
        }

    @property
    def artifact_id(self) -> str:
        return sha256_json(self.to_payload())


@dataclass(frozen=True)
class RelationshipP1mRawFieldAttempt:
    protocol_id: str
    record_index: int
    pair_id: str
    field_plan_sha256: str
    raw_outputs: tuple[str, ...]
    schema_version: str = RELATIONSHIP_P1M_RAW_FIELD_ATTEMPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1M_RAW_FIELD_ATTEMPT_SCHEMA_VERSION:
            raise ValueError("P1m raw field attempt schema mismatch")
        _require_sha256(self.protocol_id, "P1m raw attempt protocol")
        _require_sha256(self.field_plan_sha256, "P1m raw attempt field plan")
        if not 0 <= self.record_index < 24:
            raise ValueError("P1m raw attempt index out of range")
        _require_text(self.pair_id, "P1m raw attempt pair")
        if len(self.raw_outputs) != RELATIONSHIP_P1M_FIELD_COUNT:
            raise ValueError("P1m raw attempt requires 13 outputs")
        if any(not isinstance(item, str) for item in self.raw_outputs):
            raise ValueError("P1m raw attempt outputs must be strings")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "record_index": self.record_index,
            "pair_id": self.pair_id,
            "field_plan_sha256": self.field_plan_sha256,
            "raw_outputs": list(self.raw_outputs),
        }

    @property
    def artifact_id(self) -> str:
        return sha256_json(self.to_payload())


def load_relationship_p1m_raw_field_attempts(
    *,
    output_dir: pathlib.Path,
    protocol: RelationshipP1mGenerationRecoveryProtocol,
    pair_plans: tuple[RelationshipP1mPairPlan, ...],
    field_plans_by_pair: tuple[tuple[RelationshipP1mFieldPlan, ...], ...],
) -> tuple[RelationshipP1mRawFieldAttempt, ...]:
    path = pathlib.Path(output_dir) / "raw_field_attempts.jsonl"
    if not path.is_file():
        return ()
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) > len(pair_plans):
        raise ValueError("P1m raw attempt ledger exceeds pair plan")
    records: list[RelationshipP1mRawFieldAttempt] = []
    for index, line in enumerate(lines):
        root = _require_exact_keys(
            json.loads(line),
            {
                "schema_version",
                "protocol_id",
                "record_index",
                "pair_id",
                "field_plan_sha256",
                "raw_outputs",
                "artifact_id",
            },
            field_name="P1m raw field attempt",
        )
        raw_outputs = root["raw_outputs"]
        if not isinstance(raw_outputs, list):
            raise ValueError("P1m raw field outputs must be an array")
        if any(not isinstance(item, str) for item in raw_outputs):
            raise ValueError("P1m raw field output must be text")
        record = RelationshipP1mRawFieldAttempt(
            protocol_id=_require_sha256(root["protocol_id"], "protocol id"),
            record_index=_require_int(root["record_index"], "record index"),
            pair_id=_require_text(root["pair_id"], "pair id"),
            field_plan_sha256=_require_sha256(
                root["field_plan_sha256"], "field plan"
            ),
            raw_outputs=tuple(raw_outputs),
            schema_version=_require_text(root["schema_version"], "schema version"),
        )
        plans = field_plans_by_pair[index]
        if (
            record.protocol_id != protocol.protocol_id
            or record.record_index != index
            or record.pair_id != pair_plans[index].pair_id
            or record.field_plan_sha256 != relationship_p1m_field_plan_sha256(plans)
        ):
            raise ValueError("P1m raw field attempt lineage drift")
        if record.artifact_id != _require_sha256(root["artifact_id"], "artifact id"):
            raise ValueError("P1m raw field attempt artifact id mismatch")
        records.append(record)
    return tuple(records)


def persist_relationship_p1m_raw_field_attempt(
    *,
    output_dir: pathlib.Path,
    protocol: RelationshipP1mGenerationRecoveryProtocol,
    pair_plans: tuple[RelationshipP1mPairPlan, ...],
    field_plans_by_pair: tuple[tuple[RelationshipP1mFieldPlan, ...], ...],
    raw_outputs: tuple[str, ...],
) -> RelationshipP1mRawFieldAttempt:
    records = load_relationship_p1m_raw_field_attempts(
        output_dir=output_dir,
        protocol=protocol,
        pair_plans=pair_plans,
        field_plans_by_pair=field_plans_by_pair,
    )
    index = len(records)
    if index >= len(pair_plans):
        raise ValueError("P1m raw field attempts are complete")
    record = RelationshipP1mRawFieldAttempt(
        protocol_id=protocol.protocol_id,
        record_index=index,
        pair_id=pair_plans[index].pair_id,
        field_plan_sha256=relationship_p1m_field_plan_sha256(
            field_plans_by_pair[index]
        ),
        raw_outputs=raw_outputs,
    )
    payloads = [
        {**item.to_payload(), "artifact_id": item.artifact_id}
        for item in (*records, record)
    ]
    _atomic_write_text(
        pathlib.Path(output_dir) / "raw_field_attempts.jsonl",
        "".join(canonical_json(item) + "\n" for item in payloads),
    )
    return record


def load_relationship_p1m_field_batches(
    *,
    output_dir: pathlib.Path,
    protocol: RelationshipP1mGenerationRecoveryProtocol,
    field_plans_by_pair: tuple[tuple[RelationshipP1mFieldPlan, ...], ...],
) -> tuple[RelationshipP1mFieldBatchRecord, ...]:
    path = pathlib.Path(output_dir) / "field_outputs.jsonl"
    if not path.is_file():
        return ()
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) > len(field_plans_by_pair):
        raise ValueError("P1m field ledger exceeds pair plan")
    records: list[RelationshipP1mFieldBatchRecord] = []
    for index, line in enumerate(lines):
        root = _require_exact_keys(
            json.loads(line),
            {
                "schema_version",
                "protocol_id",
                "record_index",
                "pair_id",
                "field_plan_sha256",
                "outputs",
                "artifact_id",
            },
            field_name="P1m field batch",
        )
        plans = field_plans_by_pair[index]
        raw_outputs = root["outputs"]
        if not isinstance(raw_outputs, list) or len(raw_outputs) != len(plans):
            raise ValueError("P1m durable field output count mismatch")
        outputs: list[RelationshipP1mFieldOutput] = []
        for raw_output, plan in zip(raw_outputs, plans, strict=True):
            item = _require_exact_keys(
                raw_output,
                {
                    "field_key",
                    "renderer_input_sha256",
                    "seed",
                    "raw_output",
                    "normalized_text",
                },
                field_name="P1m durable field output",
            )
            output = validate_relationship_p1m_field_output(
                _require_text(item["raw_output"], "P1m raw field output"),
                plan=plan,
            )
            if output.to_payload() != item:
                raise ValueError("P1m durable field output drift")
            outputs.append(output)
        record = RelationshipP1mFieldBatchRecord(
            protocol_id=_require_sha256(root["protocol_id"], "protocol id"),
            record_index=_require_int(root["record_index"], "record index"),
            pair_id=_require_text(root["pair_id"], "pair id"),
            field_plan_sha256=_require_sha256(
                root["field_plan_sha256"], "field plan"
            ),
            outputs=tuple(outputs),
            schema_version=_require_text(root["schema_version"], "schema version"),
        )
        if record.protocol_id != protocol.protocol_id or record.record_index != index:
            raise ValueError("P1m field ledger order/protocol drift")
        expected_pair_id = plans[0].field_key.split(":", 1)[0]
        if record.pair_id != expected_pair_id:
            raise ValueError("P1m field ledger pair drift")
        if record.field_plan_sha256 != relationship_p1m_field_plan_sha256(plans):
            raise ValueError("P1m field ledger plan drift")
        if record.artifact_id != _require_sha256(root["artifact_id"], "artifact id"):
            raise ValueError("P1m field batch artifact id mismatch")
        records.append(record)
    return tuple(records)


def persist_relationship_p1m_field_batch(
    *,
    output_dir: pathlib.Path,
    protocol: RelationshipP1mGenerationRecoveryProtocol,
    pair_plans: tuple[RelationshipP1mPairPlan, ...],
    field_plans_by_pair: tuple[tuple[RelationshipP1mFieldPlan, ...], ...],
    outputs: tuple[RelationshipP1mFieldOutput, ...],
) -> RelationshipP1mFieldBatchRecord:
    records = load_relationship_p1m_field_batches(
        output_dir=output_dir,
        protocol=protocol,
        field_plans_by_pair=field_plans_by_pair,
    )
    index = len(records)
    if index >= len(pair_plans):
        raise ValueError("P1m field generation is complete")
    plans = field_plans_by_pair[index]
    if tuple(item.field_key for item in outputs) != tuple(
        item.field_key for item in plans
    ):
        raise ValueError("P1m field batch output order mismatch")
    record = RelationshipP1mFieldBatchRecord(
        protocol_id=protocol.protocol_id,
        record_index=index,
        pair_id=pair_plans[index].pair_id,
        field_plan_sha256=relationship_p1m_field_plan_sha256(plans),
        outputs=outputs,
    )
    payloads = [
        {**item.to_payload(), "artifact_id": item.artifact_id}
        for item in (*records, record)
    ]
    _atomic_write_text(
        pathlib.Path(output_dir) / "field_outputs.jsonl",
        "".join(canonical_json(item) + "\n" for item in payloads),
    )
    return record


def relationship_p1m_field_ledger_sha256(
    records: tuple[RelationshipP1mFieldBatchRecord, ...],
) -> str:
    return hashlib.sha256(
        "".join(
            canonical_json({**item.to_payload(), "artifact_id": item.artifact_id})
            + "\n"
            for item in records
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "RELATIONSHIP_P1M_FIELD_BATCH_SCHEMA_VERSION",
    "RELATIONSHIP_P1M_PREFLIGHT_SCHEMA_VERSION",
    "RELATIONSHIP_P1M_RECOVERY_NEXT_ACTION",
    "RELATIONSHIP_P1M_RECOVERY_PROTOCOL_SCHEMA_VERSION",
    "RELATIONSHIP_P1M_RAW_FIELD_ATTEMPT_SCHEMA_VERSION",
    "RelationshipP1mFieldBatchRecord",
    "RelationshipP1mGenerationRecoveryProtocol",
    "RelationshipP1mRendererPreflightReport",
    "RelationshipP1mRawFieldAttempt",
    "build_relationship_p1m_renderer_preflight_report",
    "freeze_relationship_p1m_generation_recovery_protocol",
    "load_relationship_p1m_field_batches",
    "load_relationship_p1m_generation_recovery_protocol",
    "load_relationship_p1m_raw_field_attempts",
    "load_relationship_p1m_renderer_preflight_report",
    "persist_relationship_p1m_field_batch",
    "persist_relationship_p1m_raw_field_attempt",
    "relationship_p1m_field_ledger_sha256",
    "relationship_p1m_field_plan_sha256",
    "write_relationship_p1m_generation_recovery_protocol",
    "write_relationship_p1m_renderer_preflight_report",
]
