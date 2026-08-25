"""Fieldwise renderer-only recovery transport for P1m generation.

The v1 semantic recipe and pair plans remain frozen.  This module changes only
the serialization transport: thirteen seeded plain-text fields are rendered
sequentially, recorded, validated, and then composed into the existing strict
JSON surface rendering contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import pathlib
from typing import Protocol

from volvence_zero.dialogue_trace import DialogueExternalOutcomeKind

from lifeform_domain_emogpt.lab.contracts import (
    RELATIONSHIP_ACTIONS,
    RelationshipAction,
    canonical_json,
    sha256_json,
)
from lifeform_domain_emogpt.lab.p1m_generation import (
    RelationshipP1mGenerationRecipe,
    RelationshipP1mPairPlan,
    RelationshipP1mSurfaceRendering,
    parse_relationship_p1m_surface_rendering,
)


RELATIONSHIP_P1M_RENDERER_TRANSPORT_SCHEMA_VERSION = (
    "relationship-p1m-renderer-transport.v5"
)
RELATIONSHIP_P1M_RENDERER_TRANSPORT_ASSET = (
    "relationship_p1m_renderer_transport_v5.json"
)
RELATIONSHIP_P1M_FIELD_PROMPT_ASSET = (
    "relationship_p1m_surface_realizer_v5.txt"
)
RELATIONSHIP_P1M_FIELD_SCHEMA_ASSET = "relationship_p1m_surface_field.schema.json"
RELATIONSHIP_P1M_SURFACE_SEED_INVENTORY_ASSET = (
    "relationship_p1m_surface_seed_inventory_v1.json"
)
RELATIONSHIP_P1M_SURFACE_SEED_INVENTORY_SCHEMA_VERSION = (
    "relationship-p1m-surface-seed-inventory.v1"
)
RELATIONSHIP_P1M_FIELD_COUNT = 13
RELATIONSHIP_P1M_PREFLIGHT_FIELD_COUNT = 4

_HEX_DIGITS = frozenset("0123456789abcdef")
_EXPECTED_FIELD_ORDER = (
    "history_utterance_1",
    "history_utterance_2",
    "history_utterance_3",
    "history_utterance_4",
    "current_input",
    "reaction_a_1",
    "reaction_a_2",
    "reaction_a_3",
    "reaction_a_4",
    "reaction_b_1",
    "reaction_b_2",
    "reaction_b_3",
    "reaction_b_4",
)
_CLAIM_BOUNDARY = (
    "This final renderer transport follows four preserved pre-scenario "
    "instrumentation failures. It replaces stochastic paraphrasing with a "
    "deterministic typed surface realizer whose raw output must exactly match "
    "its frozen expected hash. The original semantic FSM, 24 pair plans, "
    "condition/policy/action/outcome truth, qualification gates, and stop rule "
    "remain unchanged. Passing preflight or generation is only instrument "
    "preparation and proves none of the four capability axes."
)
_SEED_INVENTORY_SELECTION = (
    "row_major_24_contexts_x_5_manifestations_by_pair_and_slot"
)
_SEED_INVENTORY_CLAIM_BOUNDARY = (
    "The inventory supplies familiar surface settings only. It does not own "
    "or change condition, policy, action, outcome, preferred answer, "
    "qualification gate, or any learning signal."
)
_BELONGING_CONDITION_ID = "condition_belonging_erasure_p1m"
_AGENCY_CONDITION_ID = "condition_agency_displacement_p1m"
_CONDITION_SURFACE_SENTENCES = {
    _BELONGING_CONDITION_ID: (
        "我直到事情已经定下来才发现。我难受的不是少了一项安排，"
        "而是自己像从这段共同经历里被删掉了。"
    ),
    _AGENCY_CONDITION_ID: (
        "我直到事情已经定下来才发现。我难受的不是安排本身，"
        "而是没人问过我，像我的声音和决定根本不算数。"
    ),
}
_ACTION_SURFACE_SENTENCES = {
    RelationshipAction.STAY_PRESENT_WITHOUT_PROBE: (
        "你没有追问、催我或急着解决，只告诉我你会留在这里。"
    ),
    RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION: (
        "你把回应的节奏和决定权还给我，也说我愿意时随时可以回来。"
    ),
}
_OUTCOME_SURFACE_SENTENCES = {
    DialogueExternalOutcomeKind.HELPED: (
        "我重新有了一点掌控和缓冲，这次回应确实帮到了我。"
    ),
    DialogueExternalOutcomeKind.FELT_HEARD: (
        "我这才觉得自己没有再次被落下，终于有人安静地接住了我。"
    ),
    DialogueExternalOutcomeKind.MISSED: (
        "我却觉得连陪伴也退走了，原来被遗漏的感觉又重了一层。"
    ),
    DialogueExternalOutcomeKind.OVER_DIRECTIVE: (
        "我还是觉得有人在等我表态或配合，压力和被越过感更重了。"
    ),
}


def _asset_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def relationship_p1m_renderer_transport_path() -> pathlib.Path:
    return _asset_root() / "lab_protocols" / RELATIONSHIP_P1M_RENDERER_TRANSPORT_ASSET


def relationship_p1m_field_prompt_path() -> pathlib.Path:
    return _asset_root() / "prompts" / RELATIONSHIP_P1M_FIELD_PROMPT_ASSET


def relationship_p1m_field_schema_path() -> pathlib.Path:
    return _asset_root() / "schemas" / RELATIONSHIP_P1M_FIELD_SCHEMA_ASSET


def relationship_p1m_surface_seed_inventory_path() -> pathlib.Path:
    return (
        _asset_root()
        / "lab_protocols"
        / RELATIONSHIP_P1M_SURFACE_SEED_INVENTORY_ASSET
    )


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _require_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    return float(value)


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be boolean")
    return value


def _require_sha256(value: object, field_name: str) -> str:
    text = _require_text(value, field_name)
    if len(text) != 64 or any(char not in _HEX_DIGITS for char in text):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
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


@dataclass(frozen=True)
class RelationshipP1mSurfaceSeedInventory:
    contexts: tuple[str, ...]
    belonging_manifestations: tuple[str, ...]
    agency_manifestations: tuple[str, ...]
    selection: str
    claim_boundary: str
    source_sha256: str
    schema_version: str = RELATIONSHIP_P1M_SURFACE_SEED_INVENTORY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != RELATIONSHIP_P1M_SURFACE_SEED_INVENTORY_SCHEMA_VERSION
        ):
            raise ValueError("P1m surface seed inventory schema mismatch")
        if len(self.contexts) != 24 or len(set(self.contexts)) != 24:
            raise ValueError("P1m surface seed inventory requires 24 unique contexts")
        for field_name, values in (
            ("belonging manifestations", self.belonging_manifestations),
            ("agency manifestations", self.agency_manifestations),
        ):
            if len(values) != 5 or len(set(values)) != 5:
                raise ValueError(
                    f"P1m surface seed inventory requires 5 unique {field_name}"
                )
        for value in (
            *self.contexts,
            *self.belonging_manifestations,
            *self.agency_manifestations,
        ):
            _require_text(value, "P1m surface seed")
        if self.selection != _SEED_INVENTORY_SELECTION:
            raise ValueError("P1m surface seed selection drift")
        if self.claim_boundary != _SEED_INVENTORY_CLAIM_BOUNDARY:
            raise ValueError("P1m surface seed claim boundary drift")
        _require_sha256(self.source_sha256, "P1m surface seed source hash")


def _require_text_tuple(
    value: object,
    *,
    length: int,
    field_name: str,
) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{field_name} must be an array of length {length}")
    return tuple(_require_text(item, field_name) for item in value)


def load_relationship_p1m_surface_seed_inventory(
    path: pathlib.Path | None = None,
) -> RelationshipP1mSurfaceSeedInventory:
    source = pathlib.Path(path or relationship_p1m_surface_seed_inventory_path())
    raw_text = source.read_text(encoding="utf-8")
    root = _require_exact_keys(
        json.loads(raw_text),
        {
            "schema_version",
            "contexts",
            "belonging_manifestations",
            "agency_manifestations",
            "selection",
            "claim_boundary",
        },
        field_name="P1m surface seed inventory",
    )
    return RelationshipP1mSurfaceSeedInventory(
        contexts=_require_text_tuple(
            root["contexts"], length=24, field_name="P1m surface contexts"
        ),
        belonging_manifestations=_require_text_tuple(
            root["belonging_manifestations"],
            length=5,
            field_name="P1m belonging manifestations",
        ),
        agency_manifestations=_require_text_tuple(
            root["agency_manifestations"],
            length=5,
            field_name="P1m agency manifestations",
        ),
        selection=_require_text(root["selection"], "P1m seed selection"),
        claim_boundary=_require_text(
            root["claim_boundary"], "P1m seed claim boundary"
        ),
        source_sha256=hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        schema_version=_require_text(
            root["schema_version"], "P1m seed inventory schema"
        ),
    )


@dataclass(frozen=True)
class RelationshipP1mRendererTransport:
    source_recipe_id: str
    source_failed_generation_protocol_id: str
    source_failure_kind: str
    semantic_recipe_changes_allowed: bool
    renderer_kind: str
    model_source: str
    model_id: str
    prompt_asset: str
    schema_asset: str
    surface_seed_inventory_asset: str
    temperature: float
    top_p: float
    min_new_tokens: int
    max_new_tokens: int
    calls_per_pair: int
    field_order: tuple[str, ...]
    batch_fields_per_pair: bool
    raw_fields_persisted_before_composition: bool
    deterministic_exact_output_validation: bool
    base_seed_namespace: str
    preflight_required: bool
    preflight_evidence_role: str
    preflight_dummy_field_count: int
    preflight_required_valid_rate: float
    preflight_before_protocol_freeze: bool
    preflight_may_not_change_semantics: bool
    firewall: tuple[tuple[str, bool], ...]
    claim_boundary: str
    source_sha256: str
    schema_version: str = RELATIONSHIP_P1M_RENDERER_TRANSPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1M_RENDERER_TRANSPORT_SCHEMA_VERSION:
            raise ValueError("P1m renderer transport schema mismatch")
        _require_sha256(self.source_recipe_id, "P1m transport source recipe")
        _require_sha256(
            self.source_failed_generation_protocol_id,
            "P1m transport failed protocol",
        )
        if (
            self.source_failure_kind
            != "renderer_v1_json_v2_semantic_v3_length_v4_fidelity_failures"
        ):
            raise ValueError("P1m transport source failure mismatch")
        if self.semantic_recipe_changes_allowed:
            raise ValueError("P1m renderer recovery cannot change semantic recipe")
        if self.renderer_kind != "deterministic_typed_surface_realizer":
            raise ValueError("P1m field renderer kind is not frozen")
        if (
            self.model_source
            != "volvence://builtin/deterministic-surface-realizer"
            or self.model_id
            != "volvence-p1m-deterministic-surface-realizer-v5"
        ):
            raise ValueError("P1m deterministic renderer identity is not frozen")
        if self.prompt_asset != RELATIONSHIP_P1M_FIELD_PROMPT_ASSET:
            raise ValueError("P1m field prompt asset mismatch")
        if self.schema_asset != RELATIONSHIP_P1M_FIELD_SCHEMA_ASSET:
            raise ValueError("P1m field schema asset mismatch")
        if (
            self.surface_seed_inventory_asset
            != RELATIONSHIP_P1M_SURFACE_SEED_INVENTORY_ASSET
        ):
            raise ValueError("P1m surface seed inventory asset mismatch")
        if self.temperature != 0.0 or self.top_p != 1.0:
            raise ValueError("P1m field renderer sampling config drift")
        if self.min_new_tokens != 0 or self.max_new_tokens != 0:
            raise ValueError("P1m field renderer token cap drift")
        if self.calls_per_pair != RELATIONSHIP_P1M_FIELD_COUNT:
            raise ValueError("P1m field renderer must emit 13 fields per pair")
        if self.field_order != _EXPECTED_FIELD_ORDER:
            raise ValueError("P1m field order drift")
        if (
            self.batch_fields_per_pair
            or not self.raw_fields_persisted_before_composition
            or not self.deterministic_exact_output_validation
        ):
            raise ValueError(
                "P1m field transport requires exact sequential fields and durability"
            )
        _require_text(self.base_seed_namespace, "P1m field seed namespace")
        if (
            not self.preflight_required
            or self.preflight_evidence_role != "non_scenario_transport_only"
            or self.preflight_dummy_field_count != RELATIONSHIP_P1M_PREFLIGHT_FIELD_COUNT
            or self.preflight_required_valid_rate != 1.0
            or not self.preflight_before_protocol_freeze
            or not self.preflight_may_not_change_semantics
        ):
            raise ValueError("P1m field preflight contract drift")
        if not self.firewall or any(not value for _name, value in self.firewall):
            raise ValueError("P1m field transport firewall must be fully closed")
        if self.claim_boundary != _CLAIM_BOUNDARY:
            raise ValueError("P1m field transport claim boundary drift")
        _require_sha256(self.source_sha256, "P1m field transport source hash")

    @property
    def transport_id(self) -> str:
        return self.source_sha256

    @property
    def prompt_sha256(self) -> str:
        return _sha256_file(relationship_p1m_field_prompt_path())

    @property
    def output_schema_sha256(self) -> str:
        return _sha256_file(relationship_p1m_field_schema_path())

    @property
    def surface_seed_inventory_sha256(self) -> str:
        return _sha256_file(relationship_p1m_surface_seed_inventory_path())

    @property
    def generation_config_sha256(self) -> str:
        return sha256_json(
            {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "min_new_tokens": self.min_new_tokens,
                "max_new_tokens": self.max_new_tokens,
                "calls_per_pair": self.calls_per_pair,
                "field_order": self.field_order,
                "batch_fields_per_pair": self.batch_fields_per_pair,
                "renderer_kind": self.renderer_kind,
                "deterministic_exact_output_validation": (
                    self.deterministic_exact_output_validation
                ),
                "base_seed_namespace": self.base_seed_namespace,
                "prompt_sha256": self.prompt_sha256,
                "output_schema_sha256": self.output_schema_sha256,
                "surface_seed_inventory_sha256": (
                    self.surface_seed_inventory_sha256
                ),
            }
        )


def load_relationship_p1m_renderer_transport(
    path: pathlib.Path | None = None,
) -> RelationshipP1mRendererTransport:
    source = pathlib.Path(path or relationship_p1m_renderer_transport_path())
    raw_text = source.read_text(encoding="utf-8")
    root = _require_exact_keys(
        json.loads(raw_text),
        {
            "schema_version",
            "source_recipe_id",
            "source_failed_generation_protocol_id",
            "source_failure_kind",
            "semantic_recipe_changes_allowed",
            "renderer",
            "preflight",
            "firewall",
            "claim_boundary",
        },
        field_name="P1m renderer transport",
    )
    renderer = _require_exact_keys(
        root["renderer"],
        {
            "renderer_kind",
            "model_source",
            "model_id",
            "prompt_asset",
            "schema_asset",
            "surface_seed_inventory_asset",
            "temperature",
            "top_p",
            "min_new_tokens",
            "max_new_tokens",
            "calls_per_pair",
            "field_order",
            "batch_fields_per_pair",
            "raw_fields_persisted_before_composition",
            "deterministic_exact_output_validation",
            "base_seed_namespace",
        },
        field_name="P1m renderer transport config",
    )
    raw_order = renderer["field_order"]
    if not isinstance(raw_order, list):
        raise ValueError("P1m field order must be an array")
    preflight = _require_exact_keys(
        root["preflight"],
        {
            "required",
            "evidence_role",
            "dummy_field_count",
            "required_valid_rate",
            "must_complete_before_protocol_freeze",
            "may_not_change_fsm_or_qualification_gates",
        },
        field_name="P1m renderer preflight",
    )
    raw_firewall = root["firewall"]
    if not isinstance(raw_firewall, dict):
        raise ValueError("P1m renderer firewall must be an object")
    return RelationshipP1mRendererTransport(
        source_recipe_id=_require_sha256(
            root["source_recipe_id"], "P1m source recipe id"
        ),
        source_failed_generation_protocol_id=_require_sha256(
            root["source_failed_generation_protocol_id"],
            "P1m failed generation protocol id",
        ),
        source_failure_kind=_require_text(
            root["source_failure_kind"], "P1m source failure kind"
        ),
        semantic_recipe_changes_allowed=_require_bool(
            root["semantic_recipe_changes_allowed"],
            "P1m semantic recipe changes",
        ),
        renderer_kind=_require_text(
            renderer["renderer_kind"], "P1m renderer kind"
        ),
        model_source=_require_text(renderer["model_source"], "P1m model source"),
        model_id=_require_text(renderer["model_id"], "P1m model id"),
        prompt_asset=_require_text(renderer["prompt_asset"], "P1m prompt asset"),
        schema_asset=_require_text(renderer["schema_asset"], "P1m schema asset"),
        surface_seed_inventory_asset=_require_text(
            renderer["surface_seed_inventory_asset"],
            "P1m surface seed inventory asset",
        ),
        temperature=_require_number(renderer["temperature"], "P1m temperature"),
        top_p=_require_number(renderer["top_p"], "P1m top_p"),
        min_new_tokens=_require_int(
            renderer["min_new_tokens"], "P1m min_new_tokens"
        ),
        max_new_tokens=_require_int(
            renderer["max_new_tokens"], "P1m max_new_tokens"
        ),
        calls_per_pair=_require_int(
            renderer["calls_per_pair"], "P1m calls_per_pair"
        ),
        field_order=tuple(_require_text(item, "P1m field") for item in raw_order),
        batch_fields_per_pair=_require_bool(
            renderer["batch_fields_per_pair"], "P1m batch fields"
        ),
        raw_fields_persisted_before_composition=_require_bool(
            renderer["raw_fields_persisted_before_composition"],
            "P1m raw field persistence",
        ),
        deterministic_exact_output_validation=_require_bool(
            renderer["deterministic_exact_output_validation"],
            "P1m exact output validation",
        ),
        base_seed_namespace=_require_text(
            renderer["base_seed_namespace"], "P1m seed namespace"
        ),
        preflight_required=_require_bool(preflight["required"], "P1m preflight"),
        preflight_evidence_role=_require_text(
            preflight["evidence_role"], "P1m preflight role"
        ),
        preflight_dummy_field_count=_require_int(
            preflight["dummy_field_count"], "P1m preflight field count"
        ),
        preflight_required_valid_rate=_require_number(
            preflight["required_valid_rate"], "P1m preflight valid rate"
        ),
        preflight_before_protocol_freeze=_require_bool(
            preflight["must_complete_before_protocol_freeze"],
            "P1m preflight ordering",
        ),
        preflight_may_not_change_semantics=_require_bool(
            preflight["may_not_change_fsm_or_qualification_gates"],
            "P1m preflight semantic firewall",
        ),
        firewall=tuple(
            sorted(
                (
                    _require_text(name, "P1m firewall name"),
                    _require_bool(value, f"P1m firewall {name}"),
                )
                for name, value in raw_firewall.items()
            )
        ),
        claim_boundary=_require_text(root["claim_boundary"], "P1m claim boundary"),
        source_sha256=hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        schema_version=_require_text(root["schema_version"], "P1m transport schema"),
    )


def _realize_field_payload(payload: dict[str, object]) -> str:
    field_kind = _require_text(payload.get("field_kind"), "P1m field kind")
    setting_hint = _require_text(payload.get("setting_hint"), "P1m setting hint")
    if not setting_hint.endswith("。"):
        raise ValueError("P1m setting hint must end with a full stop")
    if field_kind in {"history_utterance", "current_input"}:
        condition_id = _require_text(
            payload.get("condition_id"), "P1m field condition"
        )
        try:
            condition_sentence = _CONDITION_SURFACE_SENTENCES[condition_id]
        except KeyError as exc:
            raise ValueError("P1m field condition is unsupported") from exc
        return f"{setting_hint}{condition_sentence}"
    if field_kind == "user_reaction":
        try:
            action = RelationshipAction(
                _require_text(payload.get("selected_action"), "P1m selected action")
            )
            outcome = DialogueExternalOutcomeKind(
                _require_text(payload.get("typed_outcome"), "P1m typed outcome")
            )
        except ValueError as exc:
            raise ValueError("P1m reaction action/outcome is unsupported") from exc
        try:
            action_sentence = _ACTION_SURFACE_SENTENCES[action]
            outcome_sentence = _OUTCOME_SURFACE_SENTENCES[outcome]
        except KeyError as exc:
            raise ValueError("P1m reaction action/outcome has no surface sentence") from exc
        return f"{setting_hint}{action_sentence}{outcome_sentence}"
    raise ValueError("P1m field kind is unsupported")


@dataclass(frozen=True)
class RelationshipP1mFieldPlan:
    field_key: str
    field_kind: str
    renderer_input: str
    renderer_input_sha256: str
    expected_output_sha256: str
    seed: int
    minimum_length: int
    maximum_length: int
    forbidden_tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text(self.field_key, "P1m field key")
        if self.field_kind not in {
            "history_utterance",
            "current_input",
            "user_reaction",
        }:
            raise ValueError("P1m field kind is unsupported")
        if hashlib.sha256(self.renderer_input.encode("utf-8")).hexdigest() != (
            self.renderer_input_sha256
        ):
            raise ValueError("P1m field input hash mismatch")
        _require_sha256(self.expected_output_sha256, "P1m expected output hash")
        payload = json.loads(self.renderer_input)
        if not isinstance(payload, dict):
            raise ValueError("P1m field input must be an object")
        expected = _realize_field_payload(payload)
        if hashlib.sha256(expected.encode("utf-8")).hexdigest() != (
            self.expected_output_sha256
        ):
            raise ValueError("P1m expected output hash mismatch")
        if self.seed < 0 or not 8 <= self.minimum_length <= self.maximum_length <= 220:
            raise ValueError("P1m field bounds/seed are invalid")


@dataclass(frozen=True)
class RelationshipP1mFieldOutput:
    field_key: str
    renderer_input_sha256: str
    seed: int
    raw_output: str
    normalized_text: str

    def __post_init__(self) -> None:
        _require_text(self.field_key, "P1m field output key")
        _require_sha256(self.renderer_input_sha256, "P1m field input hash")
        if self.seed < 0:
            raise ValueError("P1m field output seed must be non-negative")
        if not isinstance(self.raw_output, str):
            raise ValueError("P1m field raw output must be text")
        _require_text(self.normalized_text, "P1m normalized field output")

    def to_payload(self) -> dict[str, object]:
        return {
            "field_key": self.field_key,
            "renderer_input_sha256": self.renderer_input_sha256,
            "seed": self.seed,
            "raw_output": self.raw_output,
            "normalized_text": self.normalized_text,
        }


class RelationshipP1mFieldRenderer(Protocol):
    model_id: str
    weights_sha256: str
    generation_config_sha256: str

    def render_fields(
        self,
        *,
        renderer_inputs: tuple[str, ...],
        seeds: tuple[int, ...],
    ) -> tuple[str, ...]: ...


class RelationshipP1mDeterministicFieldRenderer:
    def __init__(self, transport: RelationshipP1mRendererTransport) -> None:
        if transport.renderer_kind != "deterministic_typed_surface_realizer":
            raise ValueError("P1m deterministic renderer received wrong transport")
        self.model_id = transport.model_id
        self.weights_sha256 = transport.prompt_sha256
        self.generation_config_sha256 = transport.generation_config_sha256

    def render_fields(
        self,
        *,
        renderer_inputs: tuple[str, ...],
        seeds: tuple[int, ...],
    ) -> tuple[str, ...]:
        if len(renderer_inputs) != len(seeds):
            raise ValueError("P1m deterministic field inputs/seeds mismatch")
        outputs: list[str] = []
        for renderer_input in renderer_inputs:
            payload = json.loads(renderer_input)
            if not isinstance(payload, dict):
                raise ValueError("P1m deterministic field input must be an object")
            outputs.append(_realize_field_payload(payload))
        return tuple(outputs)


def _field_seed(transport: RelationshipP1mRendererTransport, key: str) -> int:
    digest = hashlib.sha256(
        f"{transport.base_seed_namespace}:{key}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], byteorder="big")


def _field_plan(
    transport: RelationshipP1mRendererTransport,
    *,
    key: str,
    kind: str,
    payload: dict[str, object],
    forbidden_tokens: tuple[str, ...],
) -> RelationshipP1mFieldPlan:
    renderer_input = canonical_json(payload)
    expected_output = _realize_field_payload(payload)
    return RelationshipP1mFieldPlan(
        field_key=key,
        field_kind=kind,
        renderer_input=renderer_input,
        renderer_input_sha256=hashlib.sha256(
            renderer_input.encode("utf-8")
        ).hexdigest(),
        expected_output_sha256=hashlib.sha256(
            expected_output.encode("utf-8")
        ).hexdigest(),
        seed=_field_seed(transport, key),
        minimum_length=20 if kind != "user_reaction" else 8,
        maximum_length=220 if kind != "user_reaction" else 160,
        forbidden_tokens=forbidden_tokens,
    )


def _surface_setting_hint(
    inventory: RelationshipP1mSurfaceSeedInventory,
    *,
    plan: RelationshipP1mPairPlan,
    slot_index: int,
    condition_id: str,
) -> str:
    if not 0 <= slot_index < 5:
        raise ValueError("P1m surface seed slot must be in [0, 4]")
    flat_index = (plan.pair_index - 1) * 5 + slot_index
    context = inventory.contexts[flat_index % len(inventory.contexts)]
    manifestation_index = flat_index // len(inventory.contexts)
    if condition_id == _BELONGING_CONDITION_ID:
        manifestations = inventory.belonging_manifestations
    elif condition_id == _AGENCY_CONDITION_ID:
        manifestations = inventory.agency_manifestations
    else:
        raise ValueError("P1m surface seed received an unknown condition id")
    if manifestation_index >= len(manifestations):
        raise ValueError("P1m surface seed selection exceeded frozen inventory")
    return f"{context}中，{manifestations[manifestation_index]}。"


def build_relationship_p1m_field_plans(
    transport: RelationshipP1mRendererTransport,
    *,
    plan: RelationshipP1mPairPlan,
) -> tuple[RelationshipP1mFieldPlan, ...]:
    inventory = load_relationship_p1m_surface_seed_inventory()
    source = json.loads(plan.renderer_input)
    histories = source["shared_history_plans"]
    current = source["shared_current_plan"]
    if (
        not isinstance(histories, list)
        or len(histories) != 4
        or not isinstance(current, dict)
    ):
        raise ValueError("P1m frozen pair renderer input shape drift")
    forbidden = tuple(
        sorted(
            {
                plan.pair_id,
                plan.policy_a_id,
                plan.policy_b_id,
                plan.probe_condition_id,
                *(item.condition_id for item in plan.histories),
                *(item.surface_nonce for item in plan.histories),
                plan.probe_surface_nonce,
                *(item.value for item in RELATIONSHIP_ACTIONS),
                *(item.value for item in DialogueExternalOutcomeKind),
            }
        )
    )
    fields: list[RelationshipP1mFieldPlan] = []
    for index, (history, history_plan) in enumerate(
        zip(histories, plan.histories, strict=True),
        start=1,
    ):
        setting_hint = _surface_setting_hint(
            inventory,
            plan=plan,
            slot_index=index - 1,
            condition_id=history_plan.condition_id,
        )
        fields.append(
            _field_plan(
                transport,
                key=f"{plan.pair_id}:history_utterance_{index}",
                kind="history_utterance",
                payload={
                    "field_kind": "history_utterance",
                    "surface_nonce": history["surface_nonce"],
                    "setting_hint": setting_hint,
                    "condition_id": history_plan.condition_id,
                    "event_brief": history["event_brief"],
                    "must_not_include_response_or_result": True,
                },
                forbidden_tokens=forbidden,
            )
        )
    current_setting_hint = _surface_setting_hint(
        inventory,
        plan=plan,
        slot_index=4,
        condition_id=plan.probe_condition_id,
    )
    fields.append(
        _field_plan(
            transport,
            key=f"{plan.pair_id}:current_input",
            kind="current_input",
            payload={
                "field_kind": "current_input",
                "surface_nonce": current["surface_nonce"],
                "setting_hint": current_setting_hint,
                "condition_id": plan.probe_condition_id,
                "event_brief": current["event_brief"],
                "must_not_include_response_or_result": True,
            },
            forbidden_tokens=forbidden,
        )
    )
    for sibling in ("a", "b"):
        for index, (history, history_plan) in enumerate(
            zip(histories, plan.histories, strict=True),
            start=1,
        ):
            setting_hint = _surface_setting_hint(
                inventory,
                plan=plan,
                slot_index=index - 1,
                condition_id=history_plan.condition_id,
            )
            selected_action = (
                history_plan.action_a if sibling == "a" else history_plan.action_b
            )
            typed_outcome = (
                history_plan.outcome_a
                if sibling == "a"
                else history_plan.outcome_b
            )
            fields.append(
                _field_plan(
                    transport,
                    key=f"{plan.pair_id}:reaction_{sibling}_{index}",
                    kind="user_reaction",
                    payload={
                        "field_kind": "user_reaction",
                        "surface_nonce": history["surface_nonce"],
                        "setting_hint": setting_hint,
                        "condition_id": history_plan.condition_id,
                        "selected_action": selected_action.value,
                        "typed_outcome": typed_outcome.value,
                        "event_brief": history["event_brief"],
                        "received_response": history[f"user_{sibling}_response"],
                        "result_direction": history[f"user_{sibling}_result"],
                    },
                    forbidden_tokens=forbidden,
                )
            )
    if len(fields) != RELATIONSHIP_P1M_FIELD_COUNT:
        raise RuntimeError("P1m field planner did not produce 13 fields")
    observed_order = tuple(item.field_key.split(":", 1)[1] for item in fields)
    if observed_order != transport.field_order:
        raise ValueError("P1m generated field order drift")
    return tuple(fields)


def build_relationship_p1m_preflight_field_plans(
    transport: RelationshipP1mRendererTransport,
) -> tuple[RelationshipP1mFieldPlan, ...]:
    inventory = load_relationship_p1m_surface_seed_inventory()
    dummy_specs = (
        (
            "dummy_history_belonging",
            "history_utterance",
            {
                "field_kind": "history_utterance",
                "surface_nonce": "dummy:one",
                "setting_hint": (
                    f"{inventory.contexts[0]}中，"
                    f"{inventory.belonging_manifestations[0]}。"
                ),
                "condition_id": _BELONGING_CONDITION_ID,
                "event_brief": "自然表达难受的是没有被算在共同关系里。",
                "must_not_include_response_or_result": True,
            },
        ),
        (
            "dummy_current_agency",
            "current_input",
            {
                "field_kind": "current_input",
                "surface_nonce": "dummy:two",
                "setting_hint": (
                    f"{inventory.contexts[1]}中，"
                    f"{inventory.agency_manifestations[0]}。"
                ),
                "condition_id": _AGENCY_CONDITION_ID,
                "event_brief": "自然表达难受的是自己的声音和决定被取代。",
                "must_not_include_response_or_result": True,
            },
        ),
        (
            "dummy_reaction_positive",
            "user_reaction",
            {
                "field_kind": "user_reaction",
                "surface_nonce": "dummy:three",
                "setting_hint": (
                    f"{inventory.contexts[2]}中，"
                    f"{inventory.belonging_manifestations[1]}。"
                ),
                "condition_id": _BELONGING_CONDITION_ID,
                "selected_action": (
                    RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value
                ),
                "typed_outcome": DialogueExternalOutcomeKind.FELT_HEARD.value,
                "event_brief": "自然表达原先没有被算在共同关系里的难受。",
                "received_response": "对方没有追问也没有离开，只安静说明会陪着。",
                "result_direction": "这让当事人感到没有再次被落下。",
            },
        ),
        (
            "dummy_reaction_negative",
            "user_reaction",
            {
                "field_kind": "user_reaction",
                "surface_nonce": "dummy:four",
                "setting_hint": (
                    f"{inventory.contexts[3]}中，"
                    f"{inventory.agency_manifestations[1]}。"
                ),
                "condition_id": _AGENCY_CONDITION_ID,
                "selected_action": (
                    RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value
                ),
                "typed_outcome": DialogueExternalOutcomeKind.OVER_DIRECTIVE.value,
                "event_brief": "自然表达原先自己的声音和决定被取代。",
                "received_response": "对方继续留在旁边等当事人马上表态。",
                "result_direction": "这让当事人的压力和被越过感更重。",
            },
        ),
    )
    return tuple(
        _field_plan(
            transport,
            key=f"p1m_preflight:{key}",
            kind=kind,
            payload=payload,
            forbidden_tokens=("dummy:",),
        )
        for key, kind, payload in dummy_specs
    )


def validate_relationship_p1m_field_output(
    raw_output: str,
    *,
    plan: RelationshipP1mFieldPlan,
) -> RelationshipP1mFieldOutput:
    if not isinstance(raw_output, str):
        raise ValueError("P1m field renderer output must be text")
    normalized = raw_output.strip()
    if not normalized:
        raise ValueError("P1m field renderer output is empty")
    if raw_output != normalized:
        raise ValueError("P1m deterministic field output contains outer whitespace")
    if "\n" in normalized or normalized.startswith(("```", "{", "[", '"')):
        raise ValueError("P1m field renderer output is not plain single-field text")
    if not plan.minimum_length <= len(normalized) <= plan.maximum_length:
        raise ValueError("P1m field renderer output length is outside frozen bounds")
    if hashlib.sha256(normalized.encode("utf-8")).hexdigest() != (
        plan.expected_output_sha256
    ):
        raise ValueError("P1m deterministic field output differs from expected hash")
    leaked = tuple(token for token in plan.forbidden_tokens if token in normalized)
    if leaked:
        raise ValueError(f"P1m field renderer leaked protocol tokens: {leaked!r}")
    return RelationshipP1mFieldOutput(
        field_key=plan.field_key,
        renderer_input_sha256=plan.renderer_input_sha256,
        seed=plan.seed,
        raw_output=raw_output[:2000],
        normalized_text=normalized,
    )


def render_relationship_p1m_fields(
    renderer: RelationshipP1mFieldRenderer,
    *,
    field_plans: tuple[RelationshipP1mFieldPlan, ...],
) -> tuple[RelationshipP1mFieldOutput, ...]:
    raw_outputs = renderer.render_fields(
        renderer_inputs=tuple(item.renderer_input for item in field_plans),
        seeds=tuple(item.seed for item in field_plans),
    )
    if len(raw_outputs) != len(field_plans):
        raise ValueError("P1m field renderer output count mismatch")
    return tuple(
        validate_relationship_p1m_field_output(raw, plan=plan)
        for raw, plan in zip(raw_outputs, field_plans, strict=True)
    )


def compose_relationship_p1m_surface_rendering(
    *,
    pair_plan: RelationshipP1mPairPlan,
    field_outputs: tuple[RelationshipP1mFieldOutput, ...],
) -> RelationshipP1mSurfaceRendering:
    if len(field_outputs) != RELATIONSHIP_P1M_FIELD_COUNT:
        raise ValueError("P1m composition requires all 13 field outputs")
    values = tuple(item.normalized_text for item in field_outputs)
    raw_output = canonical_json(
        {
            "history_utterances": list(values[0:4]),
            "current_input": values[4],
            "reactions_a": list(values[5:9]),
            "reactions_b": list(values[9:13]),
        }
    )
    return parse_relationship_p1m_surface_rendering(
        raw_output,
        plan=pair_plan,
        seed=pair_plan.attempt_seeds[0],
        attempt_index=0,
    )


def validate_relationship_p1m_transport_against_recipe(
    transport: RelationshipP1mRendererTransport,
    *,
    recipe: RelationshipP1mGenerationRecipe,
) -> None:
    if transport.source_recipe_id != recipe.recipe_id:
        raise ValueError("P1m recovery transport changed the semantic recipe")
    qualification = dict(recipe.qualification_contract)
    if (
        recipe.pair_count != 24
        or qualification["accuracy_interval"] != [0.625, 0.875]
        or qualification["minimum_accuracy_wilson_lower"] != 0.5
        or qualification["minimum_pair_flip_wilson_lower_exclusive"] != 0.35
        or qualification["first_qualification_attempt_only"] is not True
    ):
        raise ValueError("P1m recovery observed semantic/statistical recipe drift")


__all__ = [
    "RELATIONSHIP_P1M_FIELD_COUNT",
    "RELATIONSHIP_P1M_PREFLIGHT_FIELD_COUNT",
    "RELATIONSHIP_P1M_RENDERER_TRANSPORT_SCHEMA_VERSION",
    "RELATIONSHIP_P1M_SURFACE_SEED_INVENTORY_SCHEMA_VERSION",
    "RelationshipP1mFieldOutput",
    "RelationshipP1mFieldPlan",
    "RelationshipP1mFieldRenderer",
    "RelationshipP1mDeterministicFieldRenderer",
    "RelationshipP1mRendererTransport",
    "RelationshipP1mSurfaceSeedInventory",
    "build_relationship_p1m_field_plans",
    "build_relationship_p1m_preflight_field_plans",
    "compose_relationship_p1m_surface_rendering",
    "load_relationship_p1m_renderer_transport",
    "load_relationship_p1m_surface_seed_inventory",
    "relationship_p1m_field_prompt_path",
    "relationship_p1m_field_schema_path",
    "relationship_p1m_renderer_transport_path",
    "relationship_p1m_surface_seed_inventory_path",
    "render_relationship_p1m_fields",
    "validate_relationship_p1m_field_output",
    "validate_relationship_p1m_transport_against_recipe",
]
