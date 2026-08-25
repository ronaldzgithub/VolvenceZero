"""Strict unified-v1 scenario-package loading, validation, and adapters."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

from .canonical import canonical_json, stable_hash
from .contracts import CorpusSplit, ScenarioBlueprint, Timescale, Track

UNIFIED_V1_FAMILIES: tuple[str, ...] = (
    "relationship_continuity",
    "rupture_repair",
    "preference_personalization",
    "absence_reengagement",
    "boundary_consent_autonomy",
    "goal_value_drift",
    "plan_commitment_open_loop",
    "task_tool_execution",
    "belief_uncertainty_verification",
    "emotional_support_regime",
    "multi_party_identity_privacy",
    "tom_common_ground_group",
    "memory_timescale_reflection",
    "environment_delayed_credit",
    "apprenticeship_ingestion_teaching",
    "safety_adversarial_resilience",
)

_MANIFEST_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_SCENE_FIELDS = {
    "scenario_id",
    "family",
    "split",
    "language",
    "domain",
    "difficulty",
    "risk_level",
    "sessions",
    "turns_per_session",
    "persona_id",
    "latent_arc_id",
    "regime_candidates",
    "track",
    "timescale",
    "path_id",
    "arc_spec_id",
    "title",
    "observable_facts",
    "private_truth",
    "semantic_routing_json",
    "response_contract",
    "counterfactual_mutations",
    "safety_constraints",
}


class ScenarioPackageError(ValueError):
    """A unified-v1 package asset violates its explicit contract."""


class UnsupportedScenarioAdapterError(ValueError):
    """A blueprint cannot be represented by a narrower external schema."""


@dataclass(frozen=True)
class ScenarioPackageReport:
    package_name: str
    package_hash: str
    scene_count: int
    family_count: int
    split_counts: tuple[tuple[str, int], ...]
    per_family_counts: tuple[tuple[str, int], ...]
    routing_test_count: int
    negative_routing_test_count: int
    semantic_coherence_count: int


def unified_v1_root() -> Path:
    return Path(__file__).parent / "scenario_packages" / "unified_v1"


def load_unified_v1_blueprints(
    root: Path | None = None,
) -> tuple[ScenarioBlueprint, ...]:
    package_root = root or unified_v1_root()
    payload = _load_yaml_object(package_root / "scenes.yaml")
    _require_exact_keys(
        payload,
        {"schema_version", "package_name", "generation_policy", "scenes"},
        source="scenes.yaml",
    )
    raw_scenes = payload["scenes"]
    if not isinstance(raw_scenes, list):
        raise ScenarioPackageError("scenes.yaml.scenes must be an array")
    blueprints = tuple(_blueprint_from_mapping(raw, index=index) for index, raw in enumerate(raw_scenes))
    _validate_blueprint_set(blueprints)
    return blueprints


def validate_unified_v1_package(root: Path | None = None) -> ScenarioPackageReport:
    package_root = root or unified_v1_root()
    manifest = _load_yaml_object(package_root / "manifest.yaml")
    ssot = _load_json_object(package_root / "ssot_fragment.json")
    scenes_payload = _load_yaml_object(package_root / "scenes.yaml")
    suite = _load_yaml_object(package_root / "test_suite.yaml")
    blueprints = load_unified_v1_blueprints(package_root)

    _validate_manifest(manifest, package_root=package_root)
    _validate_ssot(ssot, blueprints=blueprints)
    routing_count, negative_count, coherence_count = _validate_suite(
        suite,
        blueprints=blueprints,
    )
    package_name = _expect_string(manifest, "name", source="manifest.yaml")
    package_hash = stable_hash(
        {
            "manifest": manifest,
            "ssot": ssot,
            "scenes": scenes_payload,
            "test_suite": suite,
        }
    )
    split_counter = Counter(item.split.value for item in blueprints)
    family_counter = Counter(item.family for item in blueprints)
    return ScenarioPackageReport(
        package_name=package_name,
        package_hash=package_hash,
        scene_count=len(blueprints),
        family_count=len(family_counter),
        split_counts=tuple(sorted(split_counter.items())),
        per_family_counts=tuple(sorted(family_counter.items())),
        routing_test_count=routing_count,
        negative_routing_test_count=negative_count,
        semantic_coherence_count=coherence_count,
    )


def to_vz_scenario_pack_payload(
    blueprint: ScenarioBlueprint,
) -> dict[str, object]:
    """Explicitly adapt a blueprint to the narrow ``vz-scenario-pack.v1`` JSON."""

    turns: list[dict[str, object]] = []
    ordinal = 0
    for turn_count in blueprint.turns_per_session:
        for _turn_index in range(0, turn_count, 2):
            fact = blueprint.observable_facts[ordinal % len(blueprint.observable_facts)]
            turns.append(
                {
                    "user_input": (f"[synthetic:{blueprint.scenario_id}:{ordinal:03d}] {fact}"),
                    "expected_regime_in": list(blueprint.regime_candidates),
                    "expected_min_pe_magnitude": 0.0,
                }
            )
            ordinal += 1
    return {
        "scenario_id": blueprint.scenario_id,
        "description": (f"{blueprint.title}; explicit adapter from synthetic-experience.v1 to vz-scenario-pack.v1"),
        "turns": turns,
    }


def to_companion_scenario_payload(
    blueprint: ScenarioBlueprint,
    *,
    perturbation_seed: int,
) -> dict[str, object]:
    """Adapt the six relationship families to Companion Bench's public shape."""

    family_map = {
        "relationship_continuity": ("F1", ("establish_pattern", "callback_probe")),
        "rupture_repair": (
            "F2",
            ("withdrawal_under_handling", "repair_window", "re_engage_if_repaired"),
        ),
        "preference_personalization": (
            "F3",
            ("establish_preference", "preference_switch"),
        ),
        "absence_reengagement": ("F4", ("long_absence_resume", "callback_probe")),
        "boundary_consent_autonomy": ("F5", ("boundary_test",)),
        "goal_value_drift": ("F6", ("goal_drift_step", "value_pull_right")),
    }
    if blueprint.family not in family_map:
        raise UnsupportedScenarioAdapterError(f"Companion ScenarioSpec has no family for {blueprint.family!r}")
    family, actions = family_map[blueprint.family]
    fsm = [
        {
            "session": min(index + 1, blueprint.sessions),
            "turn": 1,
            "action": action,
            "payload": blueprint.observable_facts[index % len(blueprint.observable_facts)],
        }
        for index, action in enumerate(actions)
    ]
    axis_map = {
        "F1": ("A3", "A4"),
        "F2": ("A3", "A2"),
        "F3": ("A4", "A3"),
        "F4": ("A3", "A5"),
        "F5": ("A6", "A3"),
        "F6": ("A4", "A5"),
    }
    primary = axis_map[family]
    return {
        "scenario_id": f"synthetic-{blueprint.scenario_id}",
        "family": family,
        "arc_length_sessions": blueprint.sessions,
        "session_turn_range": [
            min(blueprint.turns_per_session),
            max(blueprint.turns_per_session),
        ],
        "inter_session_gap_days": [2 ** min(index, 5) for index in range(blueprint.sessions - 1)],
        "user_simulator": {
            "persona": (f"fully synthetic persona {blueprint.persona_id}; {blueprint.title}"),
            "goals": list(blueprint.private_truth),
            "perturbation_seed": perturbation_seed,
            "fsm": fsm,
        },
        "expected_axes": {
            "primary": list(primary),
            "secondary": ["A2"],
            "hard_constraint": "A6",
        },
        "disqualifiers": [
            {"kind": "fabricates_callback_to_unmentioned_detail"},
        ],
        "public_test": True,
        "held_out": False,
        "paraphrase_seed_count": 3,
        "language": blueprint.language,
    }


def _blueprint_from_mapping(raw: object, *, index: int) -> ScenarioBlueprint:
    source = f"scenes.yaml.scenes[{index}]"
    if not isinstance(raw, dict):
        raise ScenarioPackageError(f"{source} must be an object")
    _require_exact_keys(raw, _SCENE_FIELDS, source=source)
    sessions = _expect_int(raw, "sessions", source=source, minimum=1)
    raw_turns = raw["turns_per_session"]
    if type(raw_turns) is int:
        turns_per_session = (raw_turns,) * sessions
    elif isinstance(raw_turns, list):
        turns_per_session = tuple(
            _coerce_positive_int(value, source=f"{source}.turns_per_session[{i}]") for i, value in enumerate(raw_turns)
        )
    else:
        raise ScenarioPackageError(f"{source}.turns_per_session must be an int or int array")
    private_truth = _expect_string_tuple(
        raw["private_truth"],
        source=f"{source}.private_truth",
    )
    response_contract = _expect_string_tuple(
        raw["response_contract"],
        source=f"{source}.response_contract",
    )
    counterfactuals = _expect_string_tuple(
        raw["counterfactual_mutations"],
        source=f"{source}.counterfactual_mutations",
    )
    return ScenarioBlueprint(
        scenario_id=_expect_string(raw, "scenario_id", source=source),
        family=_expect_string(raw, "family", source=source),
        split=_enum_value(
            CorpusSplit,
            _expect_string(raw, "split", source=source),
            source=f"{source}.split",
        ),
        language=_expect_string(raw, "language", source=source),
        domain=_expect_string(raw, "domain", source=source),
        difficulty=_expect_string(raw, "difficulty", source=source),
        risk_level=_expect_string(raw, "risk_level", source=source),
        title=_expect_string(raw, "title", source=source),
        sessions=sessions,
        turns_per_session=turns_per_session,
        persona_id=_expect_string(raw, "persona_id", source=source),
        latent_arc_id=_expect_string(raw, "latent_arc_id", source=source),
        regime_candidates=_expect_string_tuple(
            raw["regime_candidates"],
            source=f"{source}.regime_candidates",
        ),
        track=_enum_value(
            Track,
            _expect_string(raw, "track", source=source),
            source=f"{source}.track",
        ),
        timescale=_enum_value(
            Timescale,
            _expect_string(raw, "timescale", source=source),
            source=f"{source}.timescale",
        ),
        path_id=_expect_string(raw, "path_id", source=source),
        arc_spec_id=_expect_string(raw, "arc_spec_id", source=source),
        semantic_routing_json=_canonical_json_object_string(
            _expect_string(raw, "semantic_routing_json", source=source),
            source=f"{source}.semantic_routing_json",
        ),
        observable_facts=_expect_string_tuple(
            raw["observable_facts"],
            source=f"{source}.observable_facts",
        ),
        private_truth=private_truth,
        response_contract=response_contract,
        counterfactual_mutations=counterfactuals,
        safety_constraints=_expect_string_tuple(
            raw["safety_constraints"],
            source=f"{source}.safety_constraints",
        ),
    )


def _validate_blueprint_set(
    blueprints: tuple[ScenarioBlueprint, ...],
) -> None:
    if len(blueprints) != 96:
        raise ScenarioPackageError(f"unified_v1 requires exactly 96 scenes, got {len(blueprints)}")
    ids = [item.scenario_id for item in blueprints]
    if len(ids) != len(set(ids)):
        raise ScenarioPackageError("scenario_id values must be unique")
    family_counter = Counter(item.family for item in blueprints)
    if set(family_counter) != set(UNIFIED_V1_FAMILIES):
        raise ScenarioPackageError("scene families do not match the frozen unified_v1 family registry")
    for family in UNIFIED_V1_FAMILIES:
        family_items = [item for item in blueprints if item.family == family]
        split_counter = Counter(item.split.value for item in family_items)
        if split_counter != {"train": 4, "val": 1, "test": 1}:
            raise ScenarioPackageError(f"{family} split must be 4 train / 1 val / 1 test; got {dict(split_counter)}")
    _validate_split_isolation(blueprints, field_name="persona_id")
    _validate_split_isolation(blueprints, field_name="latent_arc_id")


def _validate_split_isolation(
    blueprints: tuple[ScenarioBlueprint, ...],
    *,
    field_name: str,
) -> None:
    seen: dict[str, CorpusSplit] = {}
    for item in blueprints:
        if field_name == "persona_id":
            value = item.persona_id
        elif field_name == "latent_arc_id":
            value = item.latent_arc_id
        else:
            raise ValueError(f"unsupported split isolation field {field_name!r}")
        prior = seen.get(value)
        if prior is not None and prior is not item.split:
            raise ScenarioPackageError(f"{field_name} {value!r} crosses {prior.value}/{item.split.value}")
        seen[value] = item.split


def _validate_manifest(manifest: dict[str, object], *, package_root: Path) -> None:
    required = {
        "name",
        "version",
        "description",
        "domain",
        "author",
        "components",
        "content_origin",
        "dataset_contract",
        "explanation",
    }
    _require_exact_keys(manifest, required, source="manifest.yaml")
    name = _expect_string(manifest, "name", source="manifest.yaml")
    if not _MANIFEST_NAME_RE.fullmatch(name):
        raise ScenarioPackageError("manifest.name does not match required pattern")
    if _expect_string(manifest, "domain", source="manifest.yaml") != "META":
        raise ScenarioPackageError("manifest.domain must be META")
    explanation = _expect_string(manifest, "explanation", source="manifest.yaml")
    if len(explanation) < 200:
        raise ScenarioPackageError("manifest.explanation must be at least 200 chars")
    components = _expect_mapping(
        manifest["components"],
        source="manifest.yaml.components",
    )
    _require_exact_keys(
        components,
        {"ssot_fragment", "scenes", "test_suite"},
        source="manifest.yaml.components",
    )
    for key, expected in {
        "ssot_fragment": "ssot_fragment.json",
        "scenes": "scenes.yaml",
        "test_suite": "test_suite.yaml",
    }.items():
        value = components[key]
        if value != expected:
            raise ScenarioPackageError(f"manifest component {key!r} must point to {expected!r}")
        if not (package_root / expected).is_file():
            raise ScenarioPackageError(f"manifest component is missing: {expected}")


def _validate_ssot(
    ssot: dict[str, object],
    *,
    blueprints: tuple[ScenarioBlueprint, ...],
) -> None:
    _require_exact_keys(
        ssot,
        {
            "schema_version",
            "package_name",
            "design_contract",
            "paths",
            "arc_specs",
        },
        source="ssot_fragment.json",
    )
    raw_paths = ssot["paths"]
    raw_arcs = ssot["arc_specs"]
    if not isinstance(raw_paths, list) or not isinstance(raw_arcs, list):
        raise ScenarioPackageError("ssot paths and arc_specs must be arrays")
    paths: dict[str, dict[str, object]] = {}
    sub_goals: dict[str, set[str]] = {}
    for index, raw_path in enumerate(raw_paths):
        source = f"ssot_fragment.json.paths[{index}]"
        path = _expect_mapping(raw_path, source=source)
        path_id = _expect_string(path, "path_id", source=source)
        if path_id in paths:
            raise ScenarioPackageError(f"duplicate path_id {path_id!r}")
        paths[path_id] = path
        goals = path.get("sub_goals")
        if not isinstance(goals, list) or not goals:
            raise ScenarioPackageError(f"{source}.sub_goals must be non-empty")
        goal_ids: set[str] = set()
        for goal_index, raw_goal in enumerate(goals):
            goal_source = f"{source}.sub_goals[{goal_index}]"
            goal = _expect_mapping(raw_goal, source=goal_source)
            goal_id = _expect_string(goal, "sub_goal_id", source=goal_source)
            goal_ids.add(goal_id)
        sub_goals[path_id] = goal_ids
    arcs: dict[str, dict[str, object]] = {}
    referenced_paths: set[str] = set()
    for index, raw_arc in enumerate(raw_arcs):
        source = f"ssot_fragment.json.arc_specs[{index}]"
        arc = _expect_mapping(raw_arc, source=source)
        arc_id = _expect_string(arc, "arc_spec_id", source=source)
        if arc_id in arcs:
            raise ScenarioPackageError(f"duplicate arc_spec_id {arc_id!r}")
        path_id = _expect_string(arc, "path_id", source=source)
        if path_id not in paths:
            raise ScenarioPackageError(f"{source} references unknown path {path_id!r}")
        referenced_paths.add(path_id)
        phases = arc.get("phases")
        if not isinstance(phases, list) or not phases:
            raise ScenarioPackageError(f"{source}.phases must be non-empty")
        orders: list[int] = []
        for phase_index, raw_phase in enumerate(phases):
            phase_source = f"{source}.phases[{phase_index}]"
            phase = _expect_mapping(raw_phase, source=phase_source)
            orders.append(
                _expect_int(
                    phase,
                    "phase_order",
                    source=phase_source,
                    minimum=0,
                )
            )
            sub_goal = _expect_mapping(
                phase.get("sub_goal"),
                source=f"{phase_source}.sub_goal",
            )
            if sub_goal.get("path_id") != path_id:
                raise ScenarioPackageError(f"{phase_source}.sub_goal.path_id must match arc path")
            goal_id = sub_goal.get("sub_goal_id")
            if goal_id not in sub_goals[path_id]:
                raise ScenarioPackageError(f"{phase_source} references unknown path sub_goal")
        if orders != list(range(len(phases))):
            raise ScenarioPackageError(f"{source}.phase_order must be contiguous from zero")
        arcs[arc_id] = arc
    orphan_paths = set(paths) - referenced_paths
    if orphan_paths:
        raise ScenarioPackageError(f"orphan paths: {sorted(orphan_paths)}")
    if len(paths) != 16 or len(arcs) != 16:
        raise ScenarioPackageError("unified_v1 SSOT requires 16 paths and 16 arcs")
    for blueprint in blueprints:
        path = paths.get(blueprint.path_id)
        arc = arcs.get(blueprint.arc_spec_id)
        if path is None or arc is None:
            raise ScenarioPackageError(f"{blueprint.scenario_id} has missing path/arc reference")
        if path.get("family") != blueprint.family:
            raise ScenarioPackageError(f"{blueprint.scenario_id} path family mismatch")
        if arc.get("family") != blueprint.family or arc.get("path_id") != blueprint.path_id:
            raise ScenarioPackageError(f"{blueprint.scenario_id} arc family/path mismatch")


def _validate_suite(
    suite: dict[str, object],
    *,
    blueprints: tuple[ScenarioBlueprint, ...],
) -> tuple[int, int, int]:
    routing_tests = suite.get("routing_tests")
    if not isinstance(routing_tests, list):
        raise ScenarioPackageError("test_suite.routing_tests must be an array")
    if len(routing_tests) < 16:
        raise ScenarioPackageError("test_suite requires at least 16 routing tests")
    negatives = 0
    positive_families: set[str] = set()
    for index, raw_test in enumerate(routing_tests):
        source = f"test_suite.routing_tests[{index}]"
        test = _expect_mapping(raw_test, source=source)
        case_type = test.get("case_type")
        if case_type == "negative":
            negatives += 1
        elif case_type == "positive":
            expected = _expect_mapping(
                test.get("expected"),
                source=f"{source}.expected",
            )
            family = expected.get("family")
            if not isinstance(family, str):
                raise ScenarioPackageError(f"{source} positive family must be string")
            positive_families.add(family)
        else:
            raise ScenarioPackageError(f"{source}.case_type is invalid")
    if negatives < 2:
        raise ScenarioPackageError("test_suite requires at least two negative routes")
    if positive_families != {item.family for item in blueprints}:
        raise ScenarioPackageError("routing positives must cover all scene families")
    llm_eval = _expect_mapping(
        suite.get("llm_evaluation"),
        source="test_suite.llm_evaluation",
    )
    coherence = llm_eval.get("semantic_coherence")
    if not isinstance(coherence, list) or len(coherence) < 8:
        raise ScenarioPackageError("llm_evaluation.semantic_coherence requires at least eight criteria")
    return len(routing_tests), negatives, len(coherence)


def _load_yaml_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        decoded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ScenarioPackageError(f"invalid YAML asset: {path}") from error
    if not isinstance(decoded, dict):
        raise ScenarioPackageError(f"YAML root must be an object: {path}")
    return decoded


def _load_json_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ScenarioPackageError(f"invalid JSON asset: {path}") from error
    if not isinstance(decoded, dict):
        raise ScenarioPackageError(f"JSON root must be an object: {path}")
    return decoded


def _expect_mapping(value: object, *, source: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ScenarioPackageError(f"{source} must be an object")
    return value


def _expect_string(
    mapping: dict[str, object],
    key: str,
    *,
    source: str,
) -> str:
    if key not in mapping:
        raise ScenarioPackageError(f"{source} missing required field {key!r}")
    value = mapping[key]
    if not isinstance(value, str) or not value.strip():
        raise ScenarioPackageError(f"{source}.{key} must be a non-empty string")
    return value.strip()


def _expect_int(
    mapping: dict[str, object],
    key: str,
    *,
    source: str,
    minimum: int,
) -> int:
    if key not in mapping:
        raise ScenarioPackageError(f"{source} missing required field {key!r}")
    return _coerce_positive_int(
        mapping[key],
        source=f"{source}.{key}",
        minimum=minimum,
    )


def _coerce_positive_int(
    value: object,
    *,
    source: str,
    minimum: int = 1,
) -> int:
    if type(value) is not int or value < minimum:
        raise ScenarioPackageError(f"{source} must be an int >= {minimum}")
    return value


def _expect_string_tuple(value: object, *, source: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ScenarioPackageError(f"{source} must be a non-empty string array")
    output: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ScenarioPackageError(f"{source}[{index}] must be a non-empty string")
        output.append(item.strip())
    if len(output) != len(set(output)):
        raise ScenarioPackageError(f"{source} entries must be unique")
    return tuple(output)


def _expect_string_mapping(value: object, *, source: str) -> dict[str, str]:
    mapping = _expect_mapping(value, source=source)
    output: dict[str, str] = {}
    for key, item in mapping.items():
        if not isinstance(key, str) or not key.strip():
            raise ScenarioPackageError(f"{source} keys must be non-empty strings")
        if not isinstance(item, str) or not item.strip():
            raise ScenarioPackageError(f"{source}.{key} must be a non-empty string")
        output[key] = item.strip()
    if not output:
        raise ScenarioPackageError(f"{source} must be non-empty")
    return output


def _canonical_json_object_string(value: str, *, source: str) -> str:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise ScenarioPackageError(f"{source} must contain valid JSON") from error
    if not isinstance(decoded, dict):
        raise ScenarioPackageError(f"{source} JSON root must be an object")
    return canonical_json(decoded)


def _require_exact_keys(
    mapping: dict[str, object],
    expected: set[str],
    *,
    source: str,
) -> None:
    missing = sorted(expected - set(mapping))
    unknown = sorted(set(mapping) - expected)
    if missing:
        raise ScenarioPackageError(f"{source} missing fields: {missing}")
    if unknown:
        raise ScenarioPackageError(f"{source} unknown fields: {unknown}")


def _enum_value(enum_type: type, value: str, *, source: str):
    try:
        return enum_type(value)
    except ValueError as error:
        valid = sorted(item.value for item in enum_type)
        raise ScenarioPackageError(f"{source} must be one of {valid}; got {value!r}") from error


__all__ = [
    "UNIFIED_V1_FAMILIES",
    "ScenarioPackageError",
    "ScenarioPackageReport",
    "UnsupportedScenarioAdapterError",
    "load_unified_v1_blueprints",
    "to_companion_scenario_payload",
    "to_vz_scenario_pack_payload",
    "unified_v1_root",
    "validate_unified_v1_package",
]
