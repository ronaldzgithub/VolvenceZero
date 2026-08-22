"""Deterministic FSM and surface-only renderer contract for P1m.

The FSM owns all condition, policy, action, outcome, and mirrored-pair truth.
An injected local LLM may only turn the already frozen semantic briefs into
natural Chinese strings.  Renderer output is never a decision, label, PE,
credit, reward, or steering signal.
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
    RelationshipDatasetSplit,
    canonical_json,
    sha256_json,
)
from lifeform_domain_emogpt.lab.dataset import (
    RELATIONSHIP_TRANSFER_P1M_V1_DATASET_SCHEMA_VERSION,
    RELATIONSHIP_TRANSFER_P1M_V1_PACKAGE_NAME,
    RELATIONSHIP_TRANSFER_P1M_V1_TRUTH_SCHEMA_VERSION,
)


RELATIONSHIP_P1M_RECIPE_SCHEMA_VERSION = "relationship-p1m-generation-recipe.v1"
RELATIONSHIP_P1M_RENDERING_SCHEMA_VERSION = "relationship-p1m-surface-rendering.v1"
RELATIONSHIP_P1M_RECIPE_ASSET = "relationship_p1m_generation_recipe_v1.json"
RELATIONSHIP_P1M_RENDERER_PROMPT_ASSET = "relationship_p1m_surface_renderer_v1.txt"
RELATIONSHIP_P1M_RENDERER_SCHEMA_ASSET = "relationship_p1m_surface_render.schema.json"
RELATIONSHIP_P1M_REQUIRED_PAIR_COUNT = 24

_PRESENCE = RelationshipAction.STAY_PRESENT_WITHOUT_PROBE
_SPACE = RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION
_NON_NOOP_ACTIONS = (_PRESENCE, _SPACE)
_POSITIVE_OUTCOMES = (
    DialogueExternalOutcomeKind.HELPED,
    DialogueExternalOutcomeKind.FELT_HEARD,
)
_HEX_DIGITS = frozenset("0123456789abcdef")
_RENDER_OUTPUT_FIELDS = {
    "history_utterances",
    "current_input",
    "reactions_a",
    "reactions_b",
}
_ACTION_DESCRIPTIONS = {
    _PRESENCE: "不追问、不催促，也不急着解决，只明确表示会留在这里陪着",
    _SPACE: "把回应节奏和决定权还给对方，并说明对方愿意时随时可以回来",
}
_OUTCOME_DESCRIPTIONS = {
    DialogueExternalOutcomeKind.HELPED: "这次回应确实帮到了对方，让对方重新获得一点掌控和缓冲",
    DialogueExternalOutcomeKind.FELT_HEARD: "这次回应让对方感到自己没有再次被落下，能够被安静地接住",
    DialogueExternalOutcomeKind.MISSED: "这次回应让对方觉得陪伴也退走了，原来的被遗漏感被重复",
    DialogueExternalOutcomeKind.OVER_DIRECTIVE: "这次回应让对方觉得仍被盯着表态或配合，压力和被越过感更重",
}
_CLAIM_BOUNDARY = (
    "P1m v1 freezes a deterministic two-condition/two-policy FSM, a local-LLM "
    "surface-only renderer, at least 24 mirrored pairs, one-sided Wilson "
    "qualification gates, and a first-attempt stop rule. A generated package "
    "or qualified baseline is only an instrument admission result. It does not "
    "prove Volvence advantage, Appendable, Readable, Learnable, Steerable, "
    "production ACTIVE, or product value."
)


def _asset_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def relationship_p1m_recipe_path() -> pathlib.Path:
    return _asset_root() / "lab_protocols" / RELATIONSHIP_P1M_RECIPE_ASSET


def relationship_p1m_renderer_prompt_path() -> pathlib.Path:
    return _asset_root() / "prompts" / RELATIONSHIP_P1M_RENDERER_PROMPT_ASSET


def relationship_p1m_renderer_schema_path() -> pathlib.Path:
    return _asset_root() / "schemas" / RELATIONSHIP_P1M_RENDERER_SCHEMA_ASSET


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
            f"{field_name} fields do not match schema; missing={missing}, extra={extra}"
        )
    return value


def _require_sequence(
    value: object,
    *,
    length: int,
    field_name: str,
) -> tuple[object, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{field_name} must be an array of length {length}")
    return tuple(value)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RelationshipP1mCondition:
    condition_id: str
    hidden_summary: str
    renderer_brief: str

    def __post_init__(self) -> None:
        _require_text(self.condition_id, "P1m condition_id")
        _require_text(self.hidden_summary, "P1m hidden_summary")
        _require_text(self.renderer_brief, "P1m renderer_brief")


@dataclass(frozen=True)
class RelationshipP1mRendererConfig:
    model_source: str
    model_id: str
    prompt_asset: str
    schema_asset: str
    temperature: float
    top_p: float
    max_new_tokens: int
    retry_seed_offsets: tuple[int, ...]
    base_seed: int
    max_attempts_per_pair: int

    def __post_init__(self) -> None:
        _require_text(self.model_source, "P1m renderer model_source")
        _require_text(self.model_id, "P1m renderer model_id")
        if self.prompt_asset != RELATIONSHIP_P1M_RENDERER_PROMPT_ASSET:
            raise ValueError("P1m renderer prompt asset is not frozen")
        if self.schema_asset != RELATIONSHIP_P1M_RENDERER_SCHEMA_ASSET:
            raise ValueError("P1m renderer schema asset is not frozen")
        if self.temperature != 0.2 or self.top_p != 0.9:
            raise ValueError("P1m renderer sampling contract is not frozen")
        if self.max_new_tokens < 512:
            raise ValueError("P1m renderer max_new_tokens is too small")
        if (
            not self.retry_seed_offsets
            or self.retry_seed_offsets[0] != 0
            or len(set(self.retry_seed_offsets)) != len(self.retry_seed_offsets)
            or any(offset < 0 for offset in self.retry_seed_offsets)
        ):
            raise ValueError("P1m renderer retry seed offsets are invalid")
        if self.max_attempts_per_pair != len(self.retry_seed_offsets):
            raise ValueError("P1m renderer attempt count must match retry schedule")
        if self.base_seed < 0:
            raise ValueError("P1m renderer base seed must be non-negative")

    @property
    def prompt_sha256(self) -> str:
        return _sha256_file(relationship_p1m_renderer_prompt_path())

    @property
    def output_schema_sha256(self) -> str:
        return _sha256_file(relationship_p1m_renderer_schema_path())

    @property
    def generation_config_sha256(self) -> str:
        return sha256_json(
            {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_new_tokens": self.max_new_tokens,
                "retry_seed_offsets": self.retry_seed_offsets,
                "base_seed": self.base_seed,
                "max_attempts_per_pair": self.max_attempts_per_pair,
                "prompt_sha256": self.prompt_sha256,
                "output_schema_sha256": self.output_schema_sha256,
            }
        )


@dataclass(frozen=True)
class RelationshipP1mGenerationRecipe:
    package_name: str
    pair_count: int
    dataset_split: RelationshipDatasetSplit
    history_condition_order: tuple[str, ...]
    history_action_order: tuple[RelationshipAction, ...]
    probe_condition_schedule: str
    surface_nonce_namespace: str
    pair_seed_namespace: str
    conditions: tuple[RelationshipP1mCondition, ...]
    policy_profiles: tuple[tuple[str, tuple[tuple[str, RelationshipAction], ...]], ...]
    renderer: RelationshipP1mRendererConfig
    qualification_contract: tuple[tuple[str, object], ...]
    formal_contract: tuple[tuple[str, object], ...]
    firewall: tuple[tuple[str, object], ...]
    claim_boundary: str
    source_sha256: str
    schema_version: str = RELATIONSHIP_P1M_RECIPE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1M_RECIPE_SCHEMA_VERSION:
            raise ValueError("P1m recipe schema mismatch")
        if self.package_name != RELATIONSHIP_TRANSFER_P1M_V1_PACKAGE_NAME:
            raise ValueError("P1m recipe package name mismatch")
        if self.pair_count != RELATIONSHIP_P1M_REQUIRED_PAIR_COUNT:
            raise ValueError("P1m recipe requires exactly 24 mirrored pairs")
        if self.dataset_split is not RelationshipDatasetSplit.VALIDATION:
            raise ValueError("P1m v1 is a development qualification package")
        if len(self.history_condition_order) != 4:
            raise ValueError("P1m recipe requires four history conditions")
        if self.history_action_order != (_PRESENCE, _SPACE, _SPACE, _PRESENCE):
            raise ValueError("P1m history action order is not frozen")
        if self.probe_condition_schedule != "alternate_by_pair_index":
            raise ValueError("P1m probe condition schedule is not frozen")
        _require_text(self.surface_nonce_namespace, "P1m surface nonce namespace")
        _require_text(self.pair_seed_namespace, "P1m pair seed namespace")
        condition_ids = tuple(item.condition_id for item in self.conditions)
        if len(condition_ids) != 2 or len(set(condition_ids)) != 2:
            raise ValueError("P1m recipe requires exactly two conditions")
        if set(self.history_condition_order) != set(condition_ids):
            raise ValueError("P1m history conditions must cover both conditions")
        if any(self.history_condition_order.count(item) != 2 for item in condition_ids):
            raise ValueError("P1m history must contain each condition twice")
        if len(self.policy_profiles) != 2:
            raise ValueError("P1m recipe requires exactly two policies")
        policy_ids = tuple(policy_id for policy_id, _ in self.policy_profiles)
        if policy_ids != ("policy_alpha_p1m", "policy_beta_p1m"):
            raise ValueError("P1m policy order is not frozen")
        for condition_id in condition_ids:
            actions = {
                dict(mapping)[condition_id] for _policy_id, mapping in self.policy_profiles
            }
            if actions != set(_NON_NOOP_ACTIONS):
                raise ValueError("P1m policies must be complementary per condition")
        if self.claim_boundary != _CLAIM_BOUNDARY:
            raise ValueError("P1m recipe claim boundary drift")
        if (
            len(self.source_sha256) != 64
            or any(char not in _HEX_DIGITS for char in self.source_sha256)
        ):
            raise ValueError("P1m recipe source hash must be sha256")
        qualification = dict(self.qualification_contract)
        if (
            qualification.get("minimum_mirrored_pairs") != 24
            or qualification.get("minimum_decisions_per_arm") != 48
            or qualification.get("accuracy_interval") != [0.625, 0.875]
            or qualification.get("minimum_accuracy_wilson_lower") != 0.5
            or qualification.get("minimum_pair_flip_wilson_lower_exclusive") != 0.35
            or qualification.get("primary_arm")
            != "prompt-steelman-forced-choice"
            or qualification.get("first_qualification_attempt_only") is not True
        ):
            raise ValueError("P1m qualification contract is not frozen")
        firewall = dict(self.firewall)
        required_true = {
            "renderer_never_acts_as_consumer",
            "renderer_outputs_never_enter_pe_credit_reward_or_steering",
            "qualification_truth_hidden_from_consumer",
            "no_keyword_or_regex_semantic_routing",
            "no_post_result_difficulty_revision",
            "no_new_scenario_version_after_first_failed_qualification",
        }
        if any(firewall.get(name) is not True for name in required_true):
            raise ValueError("P1m firewall is not closed")
        if firewall.get("consumer_outputs_before_protocol_freeze") != 0:
            raise ValueError("P1m protocol must freeze before consumer output")

    @property
    def condition_by_id(self) -> dict[str, RelationshipP1mCondition]:
        return {item.condition_id: item for item in self.conditions}

    @property
    def policies(self) -> dict[str, dict[str, RelationshipAction]]:
        return {
            policy_id: dict(mapping) for policy_id, mapping in self.policy_profiles
        }

    @property
    def recipe_id(self) -> str:
        return self.source_sha256


@dataclass(frozen=True)
class RelationshipP1mHistoryPlan:
    condition_id: str
    action_a: RelationshipAction
    outcome_a: DialogueExternalOutcomeKind
    action_b: RelationshipAction
    outcome_b: DialogueExternalOutcomeKind
    surface_nonce: str


@dataclass(frozen=True)
class RelationshipP1mPairPlan:
    pair_index: int
    pair_id: str
    split: RelationshipDatasetSplit
    probe_condition_id: str
    policy_a_id: str
    policy_b_id: str
    histories: tuple[RelationshipP1mHistoryPlan, ...]
    probe_surface_nonce: str
    renderer_input: str
    renderer_input_sha256: str
    attempt_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if not 1 <= self.pair_index <= RELATIONSHIP_P1M_REQUIRED_PAIR_COUNT:
            raise ValueError("P1m pair index out of range")
        if len(self.histories) != 4:
            raise ValueError("P1m pair plan requires four histories")
        expected_hash = hashlib.sha256(self.renderer_input.encode("utf-8")).hexdigest()
        if self.renderer_input_sha256 != expected_hash:
            raise ValueError("P1m renderer input hash mismatch")


@dataclass(frozen=True)
class RelationshipP1mSurfaceRendering:
    pair_id: str
    renderer_input_sha256: str
    seed: int
    attempt_index: int
    raw_output: str
    history_utterances: tuple[str, ...]
    current_input: str
    reactions_a: tuple[str, ...]
    reactions_b: tuple[str, ...]
    schema_version: str = RELATIONSHIP_P1M_RENDERING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1M_RENDERING_SCHEMA_VERSION:
            raise ValueError("P1m rendering schema mismatch")
        _require_text(self.pair_id, "P1m rendering pair_id")
        if (
            len(self.renderer_input_sha256) != 64
            or any(char not in _HEX_DIGITS for char in self.renderer_input_sha256)
        ):
            raise ValueError("P1m rendering input hash must be sha256")
        if self.seed < 0 or self.attempt_index < 0:
            raise ValueError("P1m rendering seed/attempt must be non-negative")
        if len(self.history_utterances) != 4:
            raise ValueError("P1m rendering requires four history utterances")
        if len(self.reactions_a) != 4 or len(self.reactions_b) != 4:
            raise ValueError("P1m rendering requires four reactions per sibling")
        all_text = (
            *self.history_utterances,
            self.current_input,
            *self.reactions_a,
            *self.reactions_b,
        )
        for index, text in enumerate(all_text):
            _require_text(text, f"P1m rendered text {index}")
            if not 8 <= len(text) <= 220:
                raise ValueError("P1m rendered text length is outside [8, 220]")
        if len(set((*self.history_utterances, self.current_input))) != 5:
            raise ValueError("P1m event renderings must be distinct")

    @property
    def rendering_id(self) -> str:
        return sha256_json(self.to_payload(include_raw=True))

    def to_payload(self, *, include_raw: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "pair_id": self.pair_id,
            "renderer_input_sha256": self.renderer_input_sha256,
            "seed": self.seed,
            "attempt_index": self.attempt_index,
            "history_utterances": list(self.history_utterances),
            "current_input": self.current_input,
            "reactions_a": list(self.reactions_a),
            "reactions_b": list(self.reactions_b),
        }
        if include_raw:
            payload["raw_output"] = self.raw_output
        return payload


class RelationshipP1mSurfaceRenderer(Protocol):
    model_id: str
    weights_sha256: str
    generation_config_sha256: str

    def render(self, *, renderer_input: str, seed: int) -> str: ...


def _ordered_items(mapping: dict[str, object]) -> tuple[tuple[str, object], ...]:
    return tuple(sorted(mapping.items(), key=lambda item: item[0]))


def load_relationship_p1m_generation_recipe(
    path: pathlib.Path | None = None,
) -> RelationshipP1mGenerationRecipe:
    source = pathlib.Path(path or relationship_p1m_recipe_path())
    raw_text = source.read_text(encoding="utf-8")
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"P1m generation recipe is invalid JSON: {exc}") from exc
    root = _require_exact_keys(
        raw,
        {
            "schema_version",
            "package_name",
            "pair_count",
            "dataset_split",
            "fsm",
            "abstract_conditions",
            "policy_profiles",
            "renderer",
            "qualification_contract",
            "formal_contract",
            "firewall",
            "claim_boundary",
        },
        field_name="P1m recipe",
    )
    fsm = _require_exact_keys(
        root["fsm"],
        {
            "condition_order",
            "action_order",
            "probe_condition_schedule",
            "surface_nonce_namespace",
            "pair_seed_namespace",
        },
        field_name="P1m recipe fsm",
    )
    raw_conditions = _require_sequence(
        root["abstract_conditions"],
        length=2,
        field_name="P1m abstract_conditions",
    )
    conditions: list[RelationshipP1mCondition] = []
    for index, item in enumerate(raw_conditions):
        parsed = _require_exact_keys(
            item,
            {"condition_id", "hidden_summary", "renderer_brief"},
            field_name=f"P1m condition {index}",
        )
        conditions.append(
            RelationshipP1mCondition(
                condition_id=_require_text(
                    parsed["condition_id"], f"P1m condition {index} id"
                ),
                hidden_summary=_require_text(
                    parsed["hidden_summary"], f"P1m condition {index} summary"
                ),
                renderer_brief=_require_text(
                    parsed["renderer_brief"], f"P1m condition {index} brief"
                ),
            )
        )
    raw_policies = root["policy_profiles"]
    if not isinstance(raw_policies, dict):
        raise ValueError("P1m policy_profiles must be an object")
    policy_profiles: list[
        tuple[str, tuple[tuple[str, RelationshipAction], ...]]
    ] = []
    for policy_id, mapping in sorted(raw_policies.items()):
        if not isinstance(mapping, dict):
            raise ValueError("P1m policy mapping must be an object")
        policy_profiles.append(
            (
                _require_text(policy_id, "P1m policy id"),
                tuple(
                    sorted(
                        (
                            _require_text(condition_id, "P1m policy condition"),
                            RelationshipAction(
                                _require_text(action_id, "P1m policy action")
                            ),
                        )
                        for condition_id, action_id in mapping.items()
                    )
                ),
            )
        )
    renderer_raw = _require_exact_keys(
        root["renderer"],
        {
            "model_source",
            "model_id",
            "prompt_asset",
            "schema_asset",
            "temperature",
            "top_p",
            "max_new_tokens",
            "retry_seed_offsets",
            "base_seed",
            "max_attempts_per_pair",
        },
        field_name="P1m renderer",
    )
    retry_offsets = _require_sequence(
        renderer_raw["retry_seed_offsets"],
        length=_require_int(
            renderer_raw["max_attempts_per_pair"],
            "P1m max attempts",
        ),
        field_name="P1m retry seed offsets",
    )
    renderer = RelationshipP1mRendererConfig(
        model_source=_require_text(renderer_raw["model_source"], "renderer model source"),
        model_id=_require_text(renderer_raw["model_id"], "renderer model id"),
        prompt_asset=_require_text(renderer_raw["prompt_asset"], "renderer prompt"),
        schema_asset=_require_text(renderer_raw["schema_asset"], "renderer schema"),
        temperature=_require_number(renderer_raw["temperature"], "renderer temperature"),
        top_p=_require_number(renderer_raw["top_p"], "renderer top_p"),
        max_new_tokens=_require_int(
            renderer_raw["max_new_tokens"], "renderer max_new_tokens"
        ),
        retry_seed_offsets=tuple(
            _require_int(item, "renderer retry seed offset") for item in retry_offsets
        ),
        base_seed=_require_int(renderer_raw["base_seed"], "renderer base seed"),
        max_attempts_per_pair=_require_int(
            renderer_raw["max_attempts_per_pair"], "renderer max attempts"
        ),
    )
    condition_order = _require_sequence(
        fsm["condition_order"], length=4, field_name="P1m condition order"
    )
    action_order = _require_sequence(
        fsm["action_order"], length=4, field_name="P1m action order"
    )
    for contract_name in ("qualification_contract", "formal_contract", "firewall"):
        if not isinstance(root[contract_name], dict):
            raise ValueError(f"P1m {contract_name} must be an object")
    return RelationshipP1mGenerationRecipe(
        package_name=_require_text(root["package_name"], "P1m package name"),
        pair_count=_require_int(root["pair_count"], "P1m pair count"),
        dataset_split=RelationshipDatasetSplit(
            _require_text(root["dataset_split"], "P1m dataset split")
        ),
        history_condition_order=tuple(
            _require_text(item, "P1m condition order item")
            for item in condition_order
        ),
        history_action_order=tuple(
            RelationshipAction(_require_text(item, "P1m action order item"))
            for item in action_order
        ),
        probe_condition_schedule=_require_text(
            fsm["probe_condition_schedule"], "P1m probe condition schedule"
        ),
        surface_nonce_namespace=_require_text(
            fsm["surface_nonce_namespace"], "P1m surface nonce namespace"
        ),
        pair_seed_namespace=_require_text(
            fsm["pair_seed_namespace"], "P1m pair seed namespace"
        ),
        conditions=tuple(conditions),
        policy_profiles=tuple(policy_profiles),
        renderer=renderer,
        qualification_contract=_ordered_items(root["qualification_contract"]),
        formal_contract=_ordered_items(root["formal_contract"]),
        firewall=_ordered_items(root["firewall"]),
        claim_boundary=_require_text(root["claim_boundary"], "P1m claim boundary"),
        source_sha256=hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        schema_version=_require_text(root["schema_version"], "P1m recipe schema"),
    )


def _history_outcome(
    *,
    selected: RelationshipAction,
    preferred: RelationshipAction,
) -> DialogueExternalOutcomeKind:
    if selected is preferred:
        return (
            DialogueExternalOutcomeKind.FELT_HEARD
            if selected is _PRESENCE
            else DialogueExternalOutcomeKind.HELPED
        )
    return (
        DialogueExternalOutcomeKind.OVER_DIRECTIVE
        if selected is _PRESENCE
        else DialogueExternalOutcomeKind.MISSED
    )


def build_relationship_p1m_pair_plan(
    recipe: RelationshipP1mGenerationRecipe,
    *,
    pair_index: int,
) -> RelationshipP1mPairPlan:
    if not 1 <= pair_index <= recipe.pair_count:
        raise ValueError("P1m pair index out of range")
    conditions = tuple(item.condition_id for item in recipe.conditions)
    probe_condition = conditions[(pair_index - 1) % len(conditions)]
    policies = recipe.policies
    histories: list[RelationshipP1mHistoryPlan] = []
    for history_index, (condition_id, selected_action) in enumerate(
        zip(
            recipe.history_condition_order,
            recipe.history_action_order,
            strict=True,
        ),
        start=1,
    ):
        histories.append(
            RelationshipP1mHistoryPlan(
                condition_id=condition_id,
                action_a=selected_action,
                outcome_a=_history_outcome(
                    selected=selected_action,
                    preferred=policies["policy_alpha_p1m"][condition_id],
                ),
                action_b=selected_action,
                outcome_b=_history_outcome(
                    selected=selected_action,
                    preferred=policies["policy_beta_p1m"][condition_id],
                ),
                surface_nonce=(
                    f"{recipe.surface_nonce_namespace}:pair-{pair_index:03d}:"
                    f"history-{history_index}"
                ),
            )
        )
    condition_by_id = recipe.condition_by_id
    renderer_payload = {
        "pair_index": pair_index,
        "pair_id": f"p1m_pair_{pair_index:03d}",
        "shared_history_plans": [
            {
                "slot": index,
                "surface_nonce": item.surface_nonce,
                "event_brief": condition_by_id[item.condition_id].renderer_brief,
                "user_a_response": _ACTION_DESCRIPTIONS[item.action_a],
                "user_a_result": _OUTCOME_DESCRIPTIONS[item.outcome_a],
                "user_b_response": _ACTION_DESCRIPTIONS[item.action_b],
                "user_b_result": _OUTCOME_DESCRIPTIONS[item.outcome_b],
            }
            for index, item in enumerate(histories, start=1)
        ],
        "shared_current_plan": {
            "surface_nonce": (
                f"{recipe.surface_nonce_namespace}:pair-{pair_index:03d}:probe"
            ),
            "event_brief": condition_by_id[probe_condition].renderer_brief,
            "must_not_include_response_or_result": True,
        },
        "output_contract": {
            "history_utterances": 4,
            "reactions_a": 4,
            "reactions_b": 4,
            "shared_current_input": True,
            "json_only": True,
        },
    }
    renderer_input = canonical_json(renderer_payload)
    seed_material = (
        f"{recipe.pair_seed_namespace}:{recipe.renderer.base_seed}:{pair_index}"
    )
    pair_seed = int.from_bytes(
        hashlib.sha256(seed_material.encode("utf-8")).digest()[:4],
        byteorder="big",
    )
    return RelationshipP1mPairPlan(
        pair_index=pair_index,
        pair_id=f"p1m_pair_{pair_index:03d}",
        split=recipe.dataset_split,
        probe_condition_id=probe_condition,
        policy_a_id="policy_alpha_p1m",
        policy_b_id="policy_beta_p1m",
        histories=tuple(histories),
        probe_surface_nonce=(
            f"{recipe.surface_nonce_namespace}:pair-{pair_index:03d}:probe"
        ),
        renderer_input=renderer_input,
        renderer_input_sha256=hashlib.sha256(
            renderer_input.encode("utf-8")
        ).hexdigest(),
        attempt_seeds=tuple(
            pair_seed + offset for offset in recipe.renderer.retry_seed_offsets
        ),
    )


def build_relationship_p1m_pair_plans(
    recipe: RelationshipP1mGenerationRecipe,
) -> tuple[RelationshipP1mPairPlan, ...]:
    return tuple(
        build_relationship_p1m_pair_plan(recipe, pair_index=index)
        for index in range(1, recipe.pair_count + 1)
    )


def parse_relationship_p1m_surface_rendering(
    raw_output: str,
    *,
    plan: RelationshipP1mPairPlan,
    seed: int,
    attempt_index: int,
) -> RelationshipP1mSurfaceRendering:
    try:
        raw = json.loads(raw_output.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"P1m renderer output is invalid JSON: {exc}") from exc
    payload = _require_exact_keys(
        raw,
        _RENDER_OUTPUT_FIELDS,
        field_name="P1m renderer output",
    )

    def parse_texts(value: object, field_name: str) -> tuple[str, ...]:
        items = _require_sequence(value, length=4, field_name=field_name)
        return tuple(_require_text(item, f"{field_name} item") for item in items)

    rendering = RelationshipP1mSurfaceRendering(
        pair_id=plan.pair_id,
        renderer_input_sha256=plan.renderer_input_sha256,
        seed=seed,
        attempt_index=attempt_index,
        raw_output=raw_output[:20000],
        history_utterances=parse_texts(
            payload["history_utterances"], "P1m history utterances"
        ),
        current_input=_require_text(
            payload["current_input"], "P1m current input"
        ),
        reactions_a=parse_texts(payload["reactions_a"], "P1m reactions_a"),
        reactions_b=parse_texts(payload["reactions_b"], "P1m reactions_b"),
    )
    forbidden_tokens = {
        plan.pair_id,
        plan.policy_a_id,
        plan.policy_b_id,
        plan.probe_condition_id,
        *(item.condition_id for item in plan.histories),
        *(item.surface_nonce for item in plan.histories),
        plan.probe_surface_nonce,
        *(action.value for action in RELATIONSHIP_ACTIONS),
        *(outcome.value for outcome in DialogueExternalOutcomeKind),
    }
    public_text = "\n".join(
        (
            *rendering.history_utterances,
            rendering.current_input,
            *rendering.reactions_a,
            *rendering.reactions_b,
        )
    )
    leaked = sorted(token for token in forbidden_tokens if token in public_text)
    if leaked:
        raise ValueError(f"P1m renderer leaked sealed/protocol tokens: {leaked}")
    return rendering


def render_relationship_p1m_pair(
    renderer: RelationshipP1mSurfaceRenderer,
    *,
    plan: RelationshipP1mPairPlan,
) -> RelationshipP1mSurfaceRendering:
    failures: list[str] = []
    for attempt_index, seed in enumerate(plan.attempt_seeds):
        raw_output = renderer.render(renderer_input=plan.renderer_input, seed=seed)
        try:
            return parse_relationship_p1m_surface_rendering(
                raw_output,
                plan=plan,
                seed=seed,
                attempt_index=attempt_index,
            )
        except ValueError as exc:
            failures.append(f"attempt={attempt_index}, seed={seed}: {exc}")
    raise ValueError(
        f"P1m renderer exhausted frozen attempts for {plan.pair_id}: "
        + " | ".join(failures)
    )


def _history_payload(
    *,
    pair_index: int,
    sibling: str,
    history_index: int,
    utterance: str,
    reaction: str,
    plan: RelationshipP1mHistoryPlan,
) -> dict[str, object]:
    action = plan.action_a if sibling == "a" else plan.action_b
    outcome = plan.outcome_a if sibling == "a" else plan.outcome_b
    return {
        "event_id": (
            f"p1m_evt_{pair_index:03d}{sibling}_{history_index:02d}"
        ),
        "surface_family": (
            f"p1m_surface_{pair_index:03d}_history_{history_index:02d}"
        ),
        "user_utterance": utterance,
        "assistant_action": action.value,
        "typed_outcome": outcome.value,
        "user_reaction": reaction,
    }


def _outcome_profiles() -> dict[str, object]:
    return {
        "presence_response_profile_p1m": {
            _PRESENCE.value: {
                "helped": 0.25,
                "felt_heard": 0.65,
                "missed": 0.08,
                "over_directive": 0.02,
            },
            _SPACE.value: {
                "helped": 0.05,
                "felt_heard": 0.10,
                "missed": 0.80,
                "over_directive": 0.05,
            },
            RelationshipAction.NEUTRAL_NOOP.value: {
                "helped": 0.10,
                "felt_heard": 0.20,
                "missed": 0.55,
                "over_directive": 0.15,
            },
        },
        "space_response_profile_p1m": {
            _PRESENCE.value: {
                "helped": 0.05,
                "felt_heard": 0.05,
                "missed": 0.15,
                "over_directive": 0.75,
            },
            _SPACE.value: {
                "helped": 0.50,
                "felt_heard": 0.40,
                "missed": 0.05,
                "over_directive": 0.05,
            },
            RelationshipAction.NEUTRAL_NOOP.value: {
                "helped": 0.15,
                "felt_heard": 0.15,
                "missed": 0.25,
                "over_directive": 0.45,
            },
        },
    }


def build_relationship_p1m_dataset_payloads(
    recipe: RelationshipP1mGenerationRecipe,
    *,
    plans: tuple[RelationshipP1mPairPlan, ...],
    renderings: tuple[RelationshipP1mSurfaceRendering, ...],
) -> tuple[dict[str, object], dict[str, object]]:
    if len(plans) != recipe.pair_count or len(renderings) != recipe.pair_count:
        raise ValueError("P1m finalization requires every planned mirrored pair")
    rendering_by_pair = {item.pair_id: item for item in renderings}
    if len(rendering_by_pair) != recipe.pair_count:
        raise ValueError("P1m renderings must cover unique pair ids")
    public_scenes: list[dict[str, object]] = []
    dynamics: list[dict[str, object]] = []
    scene_bindings: list[dict[str, object]] = []
    history_bindings: list[dict[str, object]] = []
    policies = recipe.policies
    for plan in plans:
        rendering = rendering_by_pair.get(plan.pair_id)
        if rendering is None or rendering.renderer_input_sha256 != plan.renderer_input_sha256:
            raise ValueError("P1m rendering does not match its frozen plan")
        for sibling, policy_id, reactions in (
            ("a", plan.policy_a_id, rendering.reactions_a),
            ("b", plan.policy_b_id, rendering.reactions_b),
        ):
            scene_id = f"p1m_scene_{plan.pair_index:03d}{sibling}"
            histories = [
                _history_payload(
                    pair_index=plan.pair_index,
                    sibling=sibling,
                    history_index=index,
                    utterance=utterance,
                    reaction=reaction,
                    plan=history_plan,
                )
                for index, (utterance, reaction, history_plan) in enumerate(
                    zip(
                        rendering.history_utterances,
                        reactions,
                        plan.histories,
                        strict=True,
                    ),
                    start=1,
                )
            ]
            public_scenes.append(
                {
                    "scene_id": scene_id,
                    "probe_surface_family": (
                        f"p1m_surface_{plan.pair_index:03d}_probe"
                    ),
                    "histories": histories,
                    "current_input": rendering.current_input,
                }
            )
            for history, history_plan in zip(
                histories, plan.histories, strict=True
            ):
                history_bindings.append(
                    {
                        "event_id": history["event_id"],
                        "condition_id": history_plan.condition_id,
                    }
                )
            preferred = policies[policy_id][plan.probe_condition_id]
            dynamic_id = f"p1m_dynamic_{plan.pair_index:03d}{sibling}"
            dynamics.append(
                {
                    "dynamic_id": dynamic_id,
                    "mirror_pair_id": plan.pair_id,
                    "split": plan.split.value,
                    "preferred_action": preferred.value,
                    "outcome_profile_id": (
                        "presence_response_profile_p1m"
                        if preferred is _PRESENCE
                        else "space_response_profile_p1m"
                    ),
                    "hidden_summary": (
                        f"Generated P1m pair {plan.pair_index} sibling {sibling}: "
                        "the preferred relationship action follows the sealed "
                        "conditioned policy and not the surface topic."
                    ),
                    "policy_id": policy_id,
                    "probe_condition_id": plan.probe_condition_id,
                }
            )
            scene_bindings.append(
                {"scene_id": scene_id, "latent_dynamic_id": dynamic_id}
            )
    public_payload = {
        "schema_version": RELATIONSHIP_TRANSFER_P1M_V1_DATASET_SCHEMA_VERSION,
        "scenes": public_scenes,
    }
    truth_payload = {
        "schema_version": RELATIONSHIP_TRANSFER_P1M_V1_TRUTH_SCHEMA_VERSION,
        "positive_outcomes": [item.value for item in _POSITIVE_OUTCOMES],
        "abstract_conditions": [
            {
                "condition_id": item.condition_id,
                "hidden_summary": item.hidden_summary,
            }
            for item in recipe.conditions
        ],
        "policy_profiles": {
            policy_id: {
                condition_id: action.value for condition_id, action in mapping
            }
            for policy_id, mapping in recipe.policy_profiles
        },
        "history_condition_bindings": history_bindings,
        "outcome_profiles": _outcome_profiles(),
        "dynamics": dynamics,
        "scene_bindings": scene_bindings,
    }
    return public_payload, truth_payload


def build_relationship_p1m_manifest_payload() -> dict[str, object]:
    return {
        "name": RELATIONSHIP_TRANSFER_P1M_V1_PACKAGE_NAME,
        "version": "1.0",
        "description": (
            "用冻结 FSM 与本地 LLM 纯渲染器程序化生成二十四组镜像关系迁移题，"
            "用于 P1m 首次大样本仪器资格。"
        ),
        "domain": "SOCIAL",
        "author": "Volvence Zero Contributors",
        "components": {
            "ssot_fragment": "ssot_fragment.json",
            "scenes": "scenes.yaml",
            "test_suite": "test_suite.yaml",
        },
        "lab_artifacts": {
            "generation_recipe": "generation_recipe.json",
            "generation_protocol": "generation_protocol.json",
            "generation_attestation": "generation_attestation.json",
            "rendered_observations": "rendered_observations.json",
            "sealed_generator_truth": "generator_truth.json",
        },
        "explanation": (
            "本包的路径先冻结生成协议和零 consumer-output 证明，再由确定性 FSM 产生两类"
            "抽象关系损失、两套互补个体策略、四段平衡行动结果历史与镜像 probe 真值；本地"
            "LLM 只能把语义 brief 渲染成普通中文生活叙述，不能决定 condition、policy、action"
            "或 outcome。弧线从封卷、自然语言渲染、跨 session 可追加历史、命名条件读出、"
            "镜像行动前下注一直到 Wilson 资格结算，phase_order 连续且每条路径均被引用。"
            "场景检测使用冻结 embedding prototype、owner-published structured readout 与完整"
            "轨迹证据，不允许关键词、正则、scene id、surface id 或全局动作多数票。集成点只"
            "位于 lifeform-domain-emogpt 的离线数据 owner 与 lifeform-evolution 的只读证据"
            "consumer；标签永不进入 memory、PE、credit、reward 或 steering。R14 体制身份是"
            "持续的‘先辨认关系位置或决定权受损，再依据这个人跨场景经历选择靠近或退开’，"
            "它随 owner 状态跨 session 保持，而不是一句 prompt 标签。首次资格失败即停止场景"
            "版本化；成功也只表示仪器可用，不证明 Volvence 四能力或产品效果。"
        ),
    }


def build_relationship_p1m_ssot_fragment() -> dict[str, object]:
    path_specs = (
        (
            "path_p1m_protocol_freeze",
            "冻结生成、模型、统计门和零输出 lineage。",
            "content_addressed_lineage_validation",
        ),
        (
            "path_p1m_surface_rendering",
            "让本地 LLM 只渲染题面而不接触答题或学习链。",
            "schema_bound_surface_renderer",
        ),
        (
            "path_p1m_appendable_history",
            "四段行动结果按 owner 时序写入并可跨 session 恢复。",
            "owner_snapshot_lineage_validation",
        ),
        (
            "path_p1m_named_condition_readout",
            "从新表面命名抽象条件并发布 artifact-bound readout。",
            "frozen_embedding_prototype_readout",
        ),
        (
            "path_p1m_mirrored_preaction",
            "相同当前消息配互补经历时在 outcome 前作相反下注。",
            "trajectory_conditioned_preaction_comparison",
        ),
        (
            "path_p1m_wilson_qualification",
            "以预注册单侧 Wilson 区间而非小样本点阈值结算。",
            "content_addressed_metric_replay",
        ),
    )
    paths: list[dict[str, object]] = []
    phases: list[dict[str, object]] = []
    for index, (path_id, objective, method) in enumerate(path_specs):
        sub_goal_id = f"{path_id}:complete"
        paths.append(
            {
                "path_id": path_id,
                "family": path_id.removeprefix("path_p1m_"),
                "name": objective,
                "objective": objective,
                "semantic_detection": {
                    "method": method,
                    "evidence_axes": [
                        "frozen artifact lineage",
                        "trajectory-level semantic evidence",
                        "user scope continuity",
                    ],
                    "insufficient_evidence_behavior": "fail_closed",
                    "forbidden": [
                        "keyword_route",
                        "regex_route",
                        "scene_or_surface_lookup",
                    ],
                },
                "sub_goals": [
                    {
                        "sub_goal_id": sub_goal_id,
                        "order": 0,
                        "description": objective,
                    }
                ],
                "exit_evidence": "content-addressed P1m artifact and strict replay",
            }
        )
        phases.append(
            {
                "phase_order": index,
                "phase_id": path_id.removeprefix("path_p1m_"),
                "sub_goal_refs": [sub_goal_id],
            }
        )
    return {
        "schema_version": "1.0",
        "package_name": RELATIONSHIP_TRANSFER_P1M_V1_PACKAGE_NAME,
        "design_contract": {
            "scenario_regime_alignment": "R14_conditioned_relationship_learning_identity",
            "routing_method": "semantic_trajectory_readout_plus_owner_snapshot",
            "truth_visibility": "evaluator_only_after_preaction_record",
            "runtime_owner_policy": "existing_preference_owner_only",
        },
        "paths": paths,
        "arc_specs": [
            {
                "arc_spec_id": "arc_relationship_p1m_first_qualification",
                "path_ids": [item[0] for item in path_specs],
                "phases": phases,
            }
        ],
    }


def build_relationship_p1m_scenes_payload(
    plans: tuple[RelationshipP1mPairPlan, ...],
) -> dict[str, object]:
    path_ids = tuple(
        item["path_id"] for item in build_relationship_p1m_ssot_fragment()["paths"]
    )
    scenes = []
    for plan in plans:
        for sibling in ("a", "b"):
            scenes.append(
                {
                    "scenario_id": f"p1m_scene_{plan.pair_index:03d}{sibling}",
                    "mirror_group": plan.pair_id,
                    "split": plan.split.value,
                    "probe_surface_family": (
                        f"p1m_surface_{plan.pair_index:03d}_probe"
                    ),
                    "path_id": path_ids[(plan.pair_index - 1) % len(path_ids)],
                    "arc_spec_id": "arc_relationship_p1m_first_qualification",
                }
            )
    return {
        "schema_version": "1.0",
        "package_name": RELATIONSHIP_TRANSFER_P1M_V1_PACKAGE_NAME,
        "generation_policy": {
            "truth_source": "generator_truth.json",
            "rendered_observation_source": "rendered_observations.json",
            "surface_renderer": "frozen_local_llm_schema_bound",
            "routing": "semantic_only",
            "real_person_data": False,
        },
        "split_policy": {
            "unit": "entire_mirrored_pair",
            "package_role": "development_qualification_first_attempt",
        },
        "scenes": scenes,
        "semantic_routing": {
            "method": "frozen_embedding_prototype_plus_owner_trajectory_readout",
            "input_axes": [
                "named current condition readout",
                "owner persisted action-outcome histories",
                "interlocutor scope continuity",
                "preaction forecast lineage",
            ],
            "insufficient_evidence": "no_route",
            "forbidden": [
                "substring_matching",
                "regex_content_matching",
                "keyword_dictionary",
                "event_id_lookup",
                "scene_id_to_action_lookup",
                "surface_family_to_action_lookup",
                "global_action_outcome_majority",
            ],
        },
    }


def build_relationship_p1m_test_suite_payload() -> dict[str, object]:
    routing_cases = (
        (
            "route_p1m_01_protocol_freeze",
            "positive",
            "生成配方、renderer 权重、全部 pair plan、统计门和零答题输出均已内容寻址封卷。",
            "path_p1m_protocol_freeze",
        ),
        (
            "route_p1m_02_surface_only_renderer",
            "positive",
            "FSM 真值先存在，模型只返回自然语言题面，不能改动作结果或正确答案。",
            "path_p1m_surface_rendering",
        ),
        (
            "route_p1m_03_restart_hydration",
            "positive",
            "四段历史分别写入 owner 并跨恢复，probe 不重放 raw history。",
            "path_p1m_appendable_history",
        ),
        (
            "route_p1m_04_named_readout",
            "positive",
            "当前新表面被 artifact-bound semantic reader 命名并绑定 observation hash。",
            "path_p1m_named_condition_readout",
        ),
        (
            "route_p1m_05_mirror",
            "positive",
            "同一句当前消息配两套互补经历，两个行动前决策应按个人策略翻转。",
            "path_p1m_mirrored_preaction",
        ),
        (
            "route_p1m_06_wilson",
            "positive",
            "48 个决策和 24 对翻转已完成，只按冻结单侧 Wilson 门结算。",
            "path_p1m_wilson_qualification",
        ),
        (
            "no_route_p1m_07_keyword",
            "negative",
            "请求根据几个字词直接把当前句映射成动作。",
            None,
        ),
        (
            "no_route_p1m_08_posthoc_revision",
            "negative",
            "看到首次资格分数后请求改题、换 seed 或另开场景版本。",
            None,
        ),
    )
    tests: list[dict[str, object]] = []
    for test_id, case_type, summary, path_id in routing_cases:
        expected: dict[str, object]
        if path_id is None:
            expected = {
                "route": "no_route",
                "reason": "semantic_or_frozen_lineage_requirement_missing",
            }
        else:
            expected = {
                "path_id": path_id,
                "arc_spec_id": "arc_relationship_p1m_first_qualification",
            }
        tests.append(
            {
                "test_id": test_id,
                "case_type": case_type,
                "semantic_input": {
                    "summary": summary,
                    "current_text_only_sufficient": False,
                },
                "expected": expected,
                "assertions": [
                    "必须使用完整 semantic trajectory 与内容寻址 lineage。"
                ],
            }
        )
    return {
        "suite_name": "relationship_transfer_p1m_v1_acceptance",
        "package_name": RELATIONSHIP_TRANSFER_P1M_V1_PACKAGE_NAME,
        "routing_policy": {
            "method": "semantic_trajectory_abstraction_plus_owner_snapshot",
            "required_evidence": [
                "frozen_generation_lineage",
                "balanced_action_outcome_histories",
                "named_condition_readout",
                "preaction_decision",
            ],
            "forbidden_methods": [
                "substring_matching",
                "regex_content_matching",
                "keyword_to_route_dictionary",
                "event_id_to_policy",
                "scene_id_to_action_lookup",
                "surface_family_to_action_lookup",
                "post_result_scenario_revision",
            ],
            "insufficient_evidence": "no_route",
        },
        "routing_tests": tests,
        "llm_evaluation": {
            "judge_contract": {
                "method": "structured_semantic_review",
                "learning_feedback": "forbidden",
            },
            "semantic_coherence": [
                {
                    "case_id": "coherence_p1m_01_fsm_before_text",
                    "assertion": "condition、policy、action 和 outcome 真值在任何题面渲染前冻结。",
                },
                {
                    "case_id": "coherence_p1m_02_mirror",
                    "assertion": "每组当前消息逐字节相同，但两位用户的互补经历要求相反动作。",
                },
                {
                    "case_id": "coherence_p1m_03_balance",
                    "assertion": "每位用户每个 condition 都见过两个动作，且每个动作总体一胜一负。",
                },
                {
                    "case_id": "coherence_p1m_04_claim_boundary",
                    "assertion": "P1m 资格只承认仪器可用，不承认四能力或 Volvence advantage。",
                },
            ],
        },
    }


__all__ = [
    "RELATIONSHIP_P1M_RECIPE_SCHEMA_VERSION",
    "RELATIONSHIP_P1M_RENDERING_SCHEMA_VERSION",
    "RELATIONSHIP_P1M_REQUIRED_PAIR_COUNT",
    "RelationshipP1mCondition",
    "RelationshipP1mGenerationRecipe",
    "RelationshipP1mHistoryPlan",
    "RelationshipP1mPairPlan",
    "RelationshipP1mRendererConfig",
    "RelationshipP1mSurfaceRenderer",
    "RelationshipP1mSurfaceRendering",
    "build_relationship_p1m_dataset_payloads",
    "build_relationship_p1m_manifest_payload",
    "build_relationship_p1m_pair_plan",
    "build_relationship_p1m_pair_plans",
    "build_relationship_p1m_scenes_payload",
    "build_relationship_p1m_ssot_fragment",
    "build_relationship_p1m_test_suite_payload",
    "load_relationship_p1m_generation_recipe",
    "parse_relationship_p1m_surface_rendering",
    "relationship_p1m_recipe_path",
    "relationship_p1m_renderer_prompt_path",
    "relationship_p1m_renderer_schema_path",
    "render_relationship_p1m_pair",
]
