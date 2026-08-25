"""P1m generated-instrument freeze and first-attempt evidence contracts.

This module owns only content-addressed offline evidence.  The relationship
vertical owns the FSM and rendered dataset; P1m freezes renderer lineage before
the first surface output and later freezes the consumer before the first answer.
No renderer or evaluator output may enter Volvence memory, PE, credit, reward,
controller state, or steering.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import pathlib
import tempfile

from lifeform_domain_emogpt.lab import (
    RELATIONSHIP_P1M_REQUIRED_PAIR_COUNT,
    RELATIONSHIP_TRANSFER_P1M_V1_PACKAGE_NAME,
    RelationshipP1mGenerationRecipe,
    RelationshipP1mPairPlan,
    RelationshipP1mSurfaceRendering,
    canonical_json,
    parse_relationship_p1m_surface_rendering,
    sha256_json,
)


RELATIONSHIP_P1M_GENERATION_PROTOCOL_SCHEMA_VERSION = (
    "relationship-p1m-generation-protocol.v1"
)
RELATIONSHIP_P1M_GENERATION_RECORD_SCHEMA_VERSION = (
    "relationship-p1m-generation-record.v1"
)
RELATIONSHIP_P1M_GENERATION_ATTESTATION_SCHEMA_VERSION = (
    "relationship-p1m-generation-attestation.v1"
)
RELATIONSHIP_P1M_SOURCE_VERDICT = "substrate_cannot_apply_disclosed_policy"
RELATIONSHIP_P1M_GENERATION_NEXT_ACTION = (
    "freeze_forced_choice_and_named_reader_before_first_qualification_answer"
)

_HEX_DIGITS = frozenset("0123456789abcdef")
_GENERATION_CLAIM_BOUNDARY = (
    "P1m generation freezes the deterministic FSM recipe, local surface-only "
    "renderer lineage, 24 mirrored pair inputs, retry seeds, and zero consumer "
    "outputs before any rendering. The renderer cannot choose an action or "
    "change truth. A sealed generated package is only an instrument input; it "
    "does not qualify a consumer or prove Volvence advantage or any of the four "
    "capability axes."
)
_ATTESTATION_CLAIM_BOUNDARY = (
    "This attestation proves that exactly 24 frozen pair plans were rendered by "
    "the bound local model and materialized into a validated 48-scene mirrored "
    "development package without consumer answers. It says nothing about answer "
    "quality, Volvence advantage, Readable/Learnable/Steerable capability, "
    "production ACTIVE, or product value."
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


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def relationship_p1m_pair_plan_sha256(
    plans: tuple[RelationshipP1mPairPlan, ...],
) -> str:
    return sha256_json(
        [
            {
                "pair_index": item.pair_index,
                "pair_id": item.pair_id,
                "split": item.split.value,
                "renderer_input_sha256": item.renderer_input_sha256,
                "attempt_seeds": list(item.attempt_seeds),
            }
            for item in plans
        ]
    )


@dataclass(frozen=True)
class RelationshipP1mGenerationProtocol:
    frozen_at_iso: str
    source_p1k_report_artifact_id: str
    source_p1k_verdict: str
    recipe_id: str
    package_name: str
    pair_count: int
    pair_plan_sha256: str
    pair_input_sha256: tuple[str, ...]
    renderer_model_source: str
    renderer_model_id: str
    renderer_weights_sha256: str
    renderer_prompt_sha256: str
    renderer_output_schema_sha256: str
    renderer_generation_config_sha256: str
    runtime_device: str
    torch_dtype: str
    renderer_outputs_before_freeze: int
    consumer_outputs_before_freeze: int
    first_attempt_only: bool
    evaluation_feedback_allowed: bool
    claim_boundary: str
    next_action: str = RELATIONSHIP_P1M_GENERATION_NEXT_ACTION
    schema_version: str = RELATIONSHIP_P1M_GENERATION_PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1M_GENERATION_PROTOCOL_SCHEMA_VERSION:
            raise ValueError("P1m generation protocol schema mismatch")
        _require_timestamp(self.frozen_at_iso, "P1m generation frozen_at_iso")
        for field_name, value in (
            ("source_p1k_report_artifact_id", self.source_p1k_report_artifact_id),
            ("recipe_id", self.recipe_id),
            ("pair_plan_sha256", self.pair_plan_sha256),
            ("renderer_weights_sha256", self.renderer_weights_sha256),
            ("renderer_prompt_sha256", self.renderer_prompt_sha256),
            ("renderer_output_schema_sha256", self.renderer_output_schema_sha256),
            (
                "renderer_generation_config_sha256",
                self.renderer_generation_config_sha256,
            ),
        ):
            _require_sha256(value, field_name)
        if self.source_p1k_verdict != RELATIONSHIP_P1M_SOURCE_VERDICT:
            raise ValueError("P1m generation requires the terminal P1k floor verdict")
        if self.package_name != RELATIONSHIP_TRANSFER_P1M_V1_PACKAGE_NAME:
            raise ValueError("P1m generation package mismatch")
        if self.pair_count != RELATIONSHIP_P1M_REQUIRED_PAIR_COUNT:
            raise ValueError("P1m generation requires exactly 24 pairs")
        if len(self.pair_input_sha256) != self.pair_count:
            raise ValueError("P1m generation pair input hashes are incomplete")
        if len(set(self.pair_input_sha256)) != self.pair_count:
            raise ValueError("P1m generation pair input hashes must be unique")
        for value in self.pair_input_sha256:
            _require_sha256(value, "P1m pair input hash")
        for field_name, value in (
            ("renderer_model_source", self.renderer_model_source),
            ("renderer_model_id", self.renderer_model_id),
            ("runtime_device", self.runtime_device),
            ("torch_dtype", self.torch_dtype),
        ):
            _require_text(value, field_name)
        if self.renderer_outputs_before_freeze != 0:
            raise ValueError("P1m generation must freeze before renderer output")
        if self.consumer_outputs_before_freeze != 0:
            raise ValueError("P1m generation must freeze before consumer output")
        if not self.first_attempt_only:
            raise ValueError("P1m generation must be first-attempt only")
        if self.evaluation_feedback_allowed:
            raise ValueError("P1m generation evaluation feedback must be forbidden")
        if self.claim_boundary != _GENERATION_CLAIM_BOUNDARY:
            raise ValueError("P1m generation claim boundary drift")
        if self.next_action != RELATIONSHIP_P1M_GENERATION_NEXT_ACTION:
            raise ValueError("P1m generation next action drift")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "frozen_at_iso": self.frozen_at_iso,
            "source": {
                "p1k_report_artifact_id": self.source_p1k_report_artifact_id,
                "p1k_verdict": self.source_p1k_verdict,
            },
            "recipe": {
                "recipe_id": self.recipe_id,
                "package_name": self.package_name,
                "pair_count": self.pair_count,
                "pair_plan_sha256": self.pair_plan_sha256,
                "pair_input_sha256": list(self.pair_input_sha256),
            },
            "renderer": {
                "model_source": self.renderer_model_source,
                "model_id": self.renderer_model_id,
                "weights_sha256": self.renderer_weights_sha256,
                "prompt_sha256": self.renderer_prompt_sha256,
                "output_schema_sha256": self.renderer_output_schema_sha256,
                "generation_config_sha256": (
                    self.renderer_generation_config_sha256
                ),
                "runtime_device": self.runtime_device,
                "torch_dtype": self.torch_dtype,
            },
            "freeze_guards": {
                "renderer_outputs_before_freeze": self.renderer_outputs_before_freeze,
                "consumer_outputs_before_freeze": self.consumer_outputs_before_freeze,
                "first_attempt_only": self.first_attempt_only,
                "evaluation_feedback_allowed": self.evaluation_feedback_allowed,
            },
            "claim_boundary": self.claim_boundary,
            "next_action": self.next_action,
        }

    @property
    def protocol_id(self) -> str:
        return sha256_json(self.to_payload())


def freeze_relationship_p1m_generation_protocol(
    *,
    recipe: RelationshipP1mGenerationRecipe,
    plans: tuple[RelationshipP1mPairPlan, ...],
    source_p1k_report_artifact_id: str,
    source_p1k_verdict: str,
    renderer_weights_sha256: str,
    runtime_device: str,
    torch_dtype: str,
    frozen_at_iso: str,
) -> RelationshipP1mGenerationProtocol:
    if len(plans) != recipe.pair_count:
        raise ValueError("P1m generation protocol requires the full pair plan")
    return RelationshipP1mGenerationProtocol(
        frozen_at_iso=frozen_at_iso,
        source_p1k_report_artifact_id=source_p1k_report_artifact_id,
        source_p1k_verdict=source_p1k_verdict,
        recipe_id=recipe.recipe_id,
        package_name=recipe.package_name,
        pair_count=recipe.pair_count,
        pair_plan_sha256=relationship_p1m_pair_plan_sha256(plans),
        pair_input_sha256=tuple(item.renderer_input_sha256 for item in plans),
        renderer_model_source=recipe.renderer.model_source,
        renderer_model_id=recipe.renderer.model_id,
        renderer_weights_sha256=renderer_weights_sha256,
        renderer_prompt_sha256=recipe.renderer.prompt_sha256,
        renderer_output_schema_sha256=recipe.renderer.output_schema_sha256,
        renderer_generation_config_sha256=(
            recipe.renderer.generation_config_sha256
        ),
        runtime_device=runtime_device,
        torch_dtype=torch_dtype,
        renderer_outputs_before_freeze=0,
        consumer_outputs_before_freeze=0,
        first_attempt_only=True,
        evaluation_feedback_allowed=False,
        claim_boundary=_GENERATION_CLAIM_BOUNDARY,
    )


def write_relationship_p1m_generation_protocol(
    protocol: RelationshipP1mGenerationProtocol,
    *,
    output_dir: pathlib.Path,
) -> pathlib.Path:
    path = pathlib.Path(output_dir) / "generation_protocol.json"
    payload = {**protocol.to_payload(), "protocol_id": protocol.protocol_id}
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return path


def load_relationship_p1m_generation_protocol(
    path: pathlib.Path,
) -> RelationshipP1mGenerationProtocol:
    raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    root = _require_exact_keys(
        raw,
        {
            "schema_version",
            "frozen_at_iso",
            "source",
            "recipe",
            "renderer",
            "freeze_guards",
            "claim_boundary",
            "next_action",
            "protocol_id",
        },
        field_name="P1m generation protocol",
    )
    source = _require_exact_keys(
        root["source"],
        {"p1k_report_artifact_id", "p1k_verdict"},
        field_name="P1m generation source",
    )
    recipe = _require_exact_keys(
        root["recipe"],
        {
            "recipe_id",
            "package_name",
            "pair_count",
            "pair_plan_sha256",
            "pair_input_sha256",
        },
        field_name="P1m generation recipe",
    )
    renderer = _require_exact_keys(
        root["renderer"],
        {
            "model_source",
            "model_id",
            "weights_sha256",
            "prompt_sha256",
            "output_schema_sha256",
            "generation_config_sha256",
            "runtime_device",
            "torch_dtype",
        },
        field_name="P1m generation renderer",
    )
    guards = _require_exact_keys(
        root["freeze_guards"],
        {
            "renderer_outputs_before_freeze",
            "consumer_outputs_before_freeze",
            "first_attempt_only",
            "evaluation_feedback_allowed",
        },
        field_name="P1m generation guards",
    )
    raw_inputs = recipe["pair_input_sha256"]
    if not isinstance(raw_inputs, list):
        raise ValueError("P1m pair input hashes must be an array")
    protocol = RelationshipP1mGenerationProtocol(
        frozen_at_iso=_require_timestamp(root["frozen_at_iso"], "frozen_at_iso"),
        source_p1k_report_artifact_id=_require_sha256(
            source["p1k_report_artifact_id"], "P1m source report"
        ),
        source_p1k_verdict=_require_text(source["p1k_verdict"], "P1m verdict"),
        recipe_id=_require_sha256(recipe["recipe_id"], "P1m recipe id"),
        package_name=_require_text(recipe["package_name"], "P1m package name"),
        pair_count=_require_int(recipe["pair_count"], "P1m pair count"),
        pair_plan_sha256=_require_sha256(
            recipe["pair_plan_sha256"], "P1m pair plan hash"
        ),
        pair_input_sha256=tuple(
            _require_sha256(item, "P1m pair input hash") for item in raw_inputs
        ),
        renderer_model_source=_require_text(
            renderer["model_source"], "P1m renderer model source"
        ),
        renderer_model_id=_require_text(
            renderer["model_id"], "P1m renderer model id"
        ),
        renderer_weights_sha256=_require_sha256(
            renderer["weights_sha256"], "P1m renderer weights"
        ),
        renderer_prompt_sha256=_require_sha256(
            renderer["prompt_sha256"], "P1m renderer prompt"
        ),
        renderer_output_schema_sha256=_require_sha256(
            renderer["output_schema_sha256"], "P1m renderer schema"
        ),
        renderer_generation_config_sha256=_require_sha256(
            renderer["generation_config_sha256"], "P1m renderer config"
        ),
        runtime_device=_require_text(
            renderer["runtime_device"], "P1m renderer device"
        ),
        torch_dtype=_require_text(renderer["torch_dtype"], "P1m renderer dtype"),
        renderer_outputs_before_freeze=_require_int(
            guards["renderer_outputs_before_freeze"], "P1m prior renderer outputs"
        ),
        consumer_outputs_before_freeze=_require_int(
            guards["consumer_outputs_before_freeze"], "P1m prior consumer outputs"
        ),
        first_attempt_only=_require_bool(
            guards["first_attempt_only"], "P1m first attempt"
        ),
        evaluation_feedback_allowed=_require_bool(
            guards["evaluation_feedback_allowed"], "P1m feedback"
        ),
        claim_boundary=_require_text(root["claim_boundary"], "P1m claim boundary"),
        next_action=_require_text(root["next_action"], "P1m next action"),
        schema_version=_require_text(root["schema_version"], "P1m schema version"),
    )
    if protocol.protocol_id != _require_sha256(root["protocol_id"], "P1m protocol id"):
        raise ValueError("P1m generation protocol id mismatch")
    return protocol


def validate_relationship_p1m_generation_protocol(
    protocol: RelationshipP1mGenerationProtocol,
    *,
    recipe: RelationshipP1mGenerationRecipe,
    plans: tuple[RelationshipP1mPairPlan, ...],
    renderer_weights_sha256: str,
) -> None:
    if protocol.recipe_id != recipe.recipe_id:
        raise ValueError("P1m generation recipe lineage mismatch")
    if protocol.pair_plan_sha256 != relationship_p1m_pair_plan_sha256(plans):
        raise ValueError("P1m generation pair plan lineage mismatch")
    if protocol.pair_input_sha256 != tuple(
        item.renderer_input_sha256 for item in plans
    ):
        raise ValueError("P1m generation input lineage mismatch")
    if protocol.renderer_model_source != recipe.renderer.model_source:
        raise ValueError("P1m renderer source drift")
    if protocol.renderer_model_id != recipe.renderer.model_id:
        raise ValueError("P1m renderer id drift")
    if protocol.renderer_weights_sha256 != renderer_weights_sha256:
        raise ValueError("P1m renderer weights drift")
    if (
        protocol.renderer_prompt_sha256 != recipe.renderer.prompt_sha256
        or protocol.renderer_output_schema_sha256
        != recipe.renderer.output_schema_sha256
        or protocol.renderer_generation_config_sha256
        != recipe.renderer.generation_config_sha256
    ):
        raise ValueError("P1m renderer asset/config lineage drift")


@dataclass(frozen=True)
class RelationshipP1mGenerationRecord:
    protocol_id: str
    record_index: int
    pair_id: str
    renderer_input_sha256: str
    rendering: RelationshipP1mSurfaceRendering
    schema_version: str = RELATIONSHIP_P1M_GENERATION_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1M_GENERATION_RECORD_SCHEMA_VERSION:
            raise ValueError("P1m generation record schema mismatch")
        _require_sha256(self.protocol_id, "P1m record protocol id")
        if not 0 <= self.record_index < RELATIONSHIP_P1M_REQUIRED_PAIR_COUNT:
            raise ValueError("P1m generation record index out of range")
        if self.pair_id != self.rendering.pair_id:
            raise ValueError("P1m generation record pair mismatch")
        if self.renderer_input_sha256 != self.rendering.renderer_input_sha256:
            raise ValueError("P1m generation record input hash mismatch")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "record_index": self.record_index,
            "pair_id": self.pair_id,
            "renderer_input_sha256": self.renderer_input_sha256,
            "rendering": self.rendering.to_payload(include_raw=True),
        }

    @property
    def artifact_id(self) -> str:
        return sha256_json(self.to_payload())


def _generation_record_from_payload(
    raw: object,
    *,
    plan: RelationshipP1mPairPlan,
) -> RelationshipP1mGenerationRecord:
    root = _require_exact_keys(
        raw,
        {
            "schema_version",
            "protocol_id",
            "record_index",
            "pair_id",
            "renderer_input_sha256",
            "rendering",
            "artifact_id",
        },
        field_name="P1m generation record",
    )
    rendering_raw = _require_exact_keys(
        root["rendering"],
        {
            "schema_version",
            "pair_id",
            "renderer_input_sha256",
            "seed",
            "attempt_index",
            "raw_output",
            "history_utterances",
            "current_input",
            "reactions_a",
            "reactions_b",
        },
        field_name="P1m generation rendering",
    )
    rendering = parse_relationship_p1m_surface_rendering(
        _require_text(rendering_raw["raw_output"], "P1m raw renderer output"),
        plan=plan,
        seed=_require_int(rendering_raw["seed"], "P1m rendering seed"),
        attempt_index=_require_int(
            rendering_raw["attempt_index"], "P1m rendering attempt"
        ),
    )
    if rendering.to_payload(include_raw=True) != rendering_raw:
        raise ValueError("P1m parsed rendering diverges from durable payload")
    record = RelationshipP1mGenerationRecord(
        protocol_id=_require_sha256(root["protocol_id"], "P1m record protocol"),
        record_index=_require_int(root["record_index"], "P1m record index"),
        pair_id=_require_text(root["pair_id"], "P1m record pair"),
        renderer_input_sha256=_require_sha256(
            root["renderer_input_sha256"], "P1m record input hash"
        ),
        rendering=rendering,
        schema_version=_require_text(root["schema_version"], "P1m record schema"),
    )
    if record.artifact_id != _require_sha256(root["artifact_id"], "P1m record id"):
        raise ValueError("P1m generation record artifact id mismatch")
    return record


def load_relationship_p1m_generation_records(
    *,
    output_dir: pathlib.Path,
    protocol: RelationshipP1mGenerationProtocol,
    plans: tuple[RelationshipP1mPairPlan, ...],
) -> tuple[RelationshipP1mGenerationRecord, ...]:
    path = pathlib.Path(output_dir) / "renderings.jsonl"
    if not path.is_file():
        return ()
    lines = path.read_text(encoding="utf-8").splitlines()
    records: list[RelationshipP1mGenerationRecord] = []
    if len(lines) > len(plans):
        raise ValueError("P1m generation ledger exceeds frozen plan")
    for index, line in enumerate(lines):
        if not line.strip():
            raise ValueError("P1m generation ledger contains an empty line")
        record = _generation_record_from_payload(json.loads(line), plan=plans[index])
        if record.protocol_id != protocol.protocol_id:
            raise ValueError("P1m generation record protocol drift")
        if record.record_index != index:
            raise ValueError("P1m generation record indices are not contiguous")
        if (
            record.pair_id != plans[index].pair_id
            or record.renderer_input_sha256 != plans[index].renderer_input_sha256
        ):
            raise ValueError("P1m generation record does not match frozen order")
        if record.rendering.seed not in plans[index].attempt_seeds:
            raise ValueError("P1m generation record used an unregistered seed")
        records.append(record)
    return tuple(records)


def persist_relationship_p1m_generation_record(
    *,
    output_dir: pathlib.Path,
    protocol: RelationshipP1mGenerationProtocol,
    plans: tuple[RelationshipP1mPairPlan, ...],
    rendering: RelationshipP1mSurfaceRendering,
) -> RelationshipP1mGenerationRecord:
    records = load_relationship_p1m_generation_records(
        output_dir=output_dir,
        protocol=protocol,
        plans=plans,
    )
    index = len(records)
    if index >= len(plans):
        raise ValueError("P1m generation is already complete")
    plan = plans[index]
    record = RelationshipP1mGenerationRecord(
        protocol_id=protocol.protocol_id,
        record_index=index,
        pair_id=plan.pair_id,
        renderer_input_sha256=plan.renderer_input_sha256,
        rendering=rendering,
    )
    payloads = [
        {**item.to_payload(), "artifact_id": item.artifact_id}
        for item in (*records, record)
    ]
    _atomic_write_text(
        pathlib.Path(output_dir) / "renderings.jsonl",
        "".join(canonical_json(item) + "\n" for item in payloads),
    )
    return record


def relationship_p1m_rendering_ledger_sha256(
    records: tuple[RelationshipP1mGenerationRecord, ...],
) -> str:
    ledger = "".join(
        canonical_json({**item.to_payload(), "artifact_id": item.artifact_id})
        + "\n"
        for item in records
    )
    return hashlib.sha256(ledger.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RelationshipP1mGenerationAttestation:
    created_at_iso: str
    protocol_id: str
    recipe_id: str
    package_name: str
    pair_count: int
    scene_count: int
    rendering_ledger_sha256: str
    dataset_fingerprint: str
    package_file_sha256: tuple[tuple[str, str], ...]
    renderer_outputs: int
    consumer_outputs: int
    first_attempt_sealed: bool
    strict_loader_passed: bool
    claim_boundary: str
    next_action: str = RELATIONSHIP_P1M_GENERATION_NEXT_ACTION
    schema_version: str = RELATIONSHIP_P1M_GENERATION_ATTESTATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1M_GENERATION_ATTESTATION_SCHEMA_VERSION:
            raise ValueError("P1m generation attestation schema mismatch")
        _require_timestamp(self.created_at_iso, "P1m attestation created_at")
        for field_name, value in (
            ("protocol_id", self.protocol_id),
            ("recipe_id", self.recipe_id),
            ("rendering_ledger_sha256", self.rendering_ledger_sha256),
            ("dataset_fingerprint", self.dataset_fingerprint),
        ):
            _require_sha256(value, field_name)
        if self.package_name != RELATIONSHIP_TRANSFER_P1M_V1_PACKAGE_NAME:
            raise ValueError("P1m attestation package mismatch")
        if self.pair_count != 24 or self.scene_count != 48:
            raise ValueError("P1m attestation package size mismatch")
        expected_names = (
            "generation_recipe.json",
            "generator_truth.json",
            "manifest.yaml",
            "rendered_observations.json",
            "scenes.yaml",
            "ssot_fragment.json",
            "test_suite.yaml",
        )
        if tuple(name for name, _ in self.package_file_sha256) != expected_names:
            raise ValueError("P1m attestation package file set/order mismatch")
        for _name, digest in self.package_file_sha256:
            _require_sha256(digest, "P1m package file hash")
        if self.renderer_outputs != 24 or self.consumer_outputs != 0:
            raise ValueError("P1m attestation output counts are invalid")
        if not self.first_attempt_sealed or not self.strict_loader_passed:
            raise ValueError("P1m generated package must be sealed and validated")
        if self.claim_boundary != _ATTESTATION_CLAIM_BOUNDARY:
            raise ValueError("P1m attestation claim boundary drift")
        if self.next_action != RELATIONSHIP_P1M_GENERATION_NEXT_ACTION:
            raise ValueError("P1m attestation next action drift")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "created_at_iso": self.created_at_iso,
            "protocol_id": self.protocol_id,
            "recipe_id": self.recipe_id,
            "package_name": self.package_name,
            "pair_count": self.pair_count,
            "scene_count": self.scene_count,
            "rendering_ledger_sha256": self.rendering_ledger_sha256,
            "dataset_fingerprint": self.dataset_fingerprint,
            "package_file_sha256": [
                {"path": name, "sha256": digest}
                for name, digest in self.package_file_sha256
            ],
            "renderer_outputs": self.renderer_outputs,
            "consumer_outputs": self.consumer_outputs,
            "first_attempt_sealed": self.first_attempt_sealed,
            "strict_loader_passed": self.strict_loader_passed,
            "claim_boundary": self.claim_boundary,
            "next_action": self.next_action,
        }

    @property
    def artifact_id(self) -> str:
        return sha256_json(self.to_payload())


def build_relationship_p1m_generation_attestation(
    *,
    protocol: RelationshipP1mGenerationProtocol,
    records: tuple[RelationshipP1mGenerationRecord, ...],
    dataset_fingerprint: str,
    package_dir: pathlib.Path,
    created_at_iso: str,
) -> RelationshipP1mGenerationAttestation:
    if len(records) != protocol.pair_count:
        raise ValueError("P1m attestation requires a complete rendering ledger")
    names = (
        "generation_recipe.json",
        "generator_truth.json",
        "manifest.yaml",
        "rendered_observations.json",
        "scenes.yaml",
        "ssot_fragment.json",
        "test_suite.yaml",
    )
    file_hashes = tuple(
        (name, _sha256_file(pathlib.Path(package_dir) / name)) for name in names
    )
    return RelationshipP1mGenerationAttestation(
        created_at_iso=created_at_iso,
        protocol_id=protocol.protocol_id,
        recipe_id=protocol.recipe_id,
        package_name=protocol.package_name,
        pair_count=protocol.pair_count,
        scene_count=protocol.pair_count * 2,
        rendering_ledger_sha256=relationship_p1m_rendering_ledger_sha256(records),
        dataset_fingerprint=dataset_fingerprint,
        package_file_sha256=file_hashes,
        renderer_outputs=len(records),
        consumer_outputs=0,
        first_attempt_sealed=True,
        strict_loader_passed=True,
        claim_boundary=_ATTESTATION_CLAIM_BOUNDARY,
    )


def write_relationship_p1m_generation_attestation(
    attestation: RelationshipP1mGenerationAttestation,
    *,
    output_dir: pathlib.Path,
) -> pathlib.Path:
    path = pathlib.Path(output_dir) / "generation_attestation.json"
    payload = {**attestation.to_payload(), "artifact_id": attestation.artifact_id}
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return path


def load_relationship_p1m_generation_attestation(
    path: pathlib.Path,
) -> RelationshipP1mGenerationAttestation:
    raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    root = _require_exact_keys(
        raw,
        {
            "schema_version",
            "created_at_iso",
            "protocol_id",
            "recipe_id",
            "package_name",
            "pair_count",
            "scene_count",
            "rendering_ledger_sha256",
            "dataset_fingerprint",
            "package_file_sha256",
            "renderer_outputs",
            "consumer_outputs",
            "first_attempt_sealed",
            "strict_loader_passed",
            "claim_boundary",
            "next_action",
            "artifact_id",
        },
        field_name="P1m generation attestation",
    )
    raw_files = root["package_file_sha256"]
    if not isinstance(raw_files, list):
        raise ValueError("P1m attestation package files must be an array")
    files: list[tuple[str, str]] = []
    for item in raw_files:
        parsed = _require_exact_keys(
            item,
            {"path", "sha256"},
            field_name="P1m attestation package file",
        )
        files.append(
            (
                _require_text(parsed["path"], "P1m package path"),
                _require_sha256(parsed["sha256"], "P1m package sha256"),
            )
        )
    attestation = RelationshipP1mGenerationAttestation(
        created_at_iso=_require_timestamp(root["created_at_iso"], "created_at"),
        protocol_id=_require_sha256(root["protocol_id"], "P1m protocol id"),
        recipe_id=_require_sha256(root["recipe_id"], "P1m recipe id"),
        package_name=_require_text(root["package_name"], "P1m package name"),
        pair_count=_require_int(root["pair_count"], "P1m pair count"),
        scene_count=_require_int(root["scene_count"], "P1m scene count"),
        rendering_ledger_sha256=_require_sha256(
            root["rendering_ledger_sha256"], "P1m rendering ledger"
        ),
        dataset_fingerprint=_require_sha256(
            root["dataset_fingerprint"], "P1m dataset fingerprint"
        ),
        package_file_sha256=tuple(files),
        renderer_outputs=_require_int(
            root["renderer_outputs"], "P1m renderer outputs"
        ),
        consumer_outputs=_require_int(
            root["consumer_outputs"], "P1m consumer outputs"
        ),
        first_attempt_sealed=_require_bool(
            root["first_attempt_sealed"], "P1m first attempt sealed"
        ),
        strict_loader_passed=_require_bool(
            root["strict_loader_passed"], "P1m strict loader"
        ),
        claim_boundary=_require_text(root["claim_boundary"], "P1m claim boundary"),
        next_action=_require_text(root["next_action"], "P1m next action"),
        schema_version=_require_text(root["schema_version"], "P1m schema version"),
    )
    if attestation.artifact_id != _require_sha256(
        root["artifact_id"], "P1m attestation id"
    ):
        raise ValueError("P1m generation attestation id mismatch")
    return attestation


def validate_relationship_p1m_generation_attestation_files(
    attestation: RelationshipP1mGenerationAttestation,
    *,
    output_dir: pathlib.Path,
    protocol: RelationshipP1mGenerationProtocol,
    records: tuple[RelationshipP1mGenerationRecord, ...],
) -> None:
    if attestation.protocol_id != protocol.protocol_id:
        raise ValueError("P1m attestation protocol lineage mismatch")
    if attestation.recipe_id != protocol.recipe_id:
        raise ValueError("P1m attestation recipe lineage mismatch")
    if attestation.rendering_ledger_sha256 != (
        relationship_p1m_rendering_ledger_sha256(records)
    ):
        raise ValueError("P1m attestation rendering ledger drift")
    for name, expected in attestation.package_file_sha256:
        if _sha256_file(pathlib.Path(output_dir) / name) != expected:
            raise ValueError(f"P1m package file drift: {name}")


__all__ = [
    "RELATIONSHIP_P1M_GENERATION_ATTESTATION_SCHEMA_VERSION",
    "RELATIONSHIP_P1M_GENERATION_NEXT_ACTION",
    "RELATIONSHIP_P1M_GENERATION_PROTOCOL_SCHEMA_VERSION",
    "RELATIONSHIP_P1M_GENERATION_RECORD_SCHEMA_VERSION",
    "RELATIONSHIP_P1M_SOURCE_VERDICT",
    "RelationshipP1mGenerationAttestation",
    "RelationshipP1mGenerationProtocol",
    "RelationshipP1mGenerationRecord",
    "build_relationship_p1m_generation_attestation",
    "freeze_relationship_p1m_generation_protocol",
    "load_relationship_p1m_generation_attestation",
    "load_relationship_p1m_generation_protocol",
    "load_relationship_p1m_generation_records",
    "persist_relationship_p1m_generation_record",
    "relationship_p1m_pair_plan_sha256",
    "relationship_p1m_rendering_ledger_sha256",
    "validate_relationship_p1m_generation_attestation_files",
    "validate_relationship_p1m_generation_protocol",
    "write_relationship_p1m_generation_attestation",
    "write_relationship_p1m_generation_protocol",
]
