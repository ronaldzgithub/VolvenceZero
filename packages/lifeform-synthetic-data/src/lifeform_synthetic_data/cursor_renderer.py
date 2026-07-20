"""Cursor-authored text assets and deterministic zero-API rendering."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from .canonical import canonical_json, stable_hash
from .contracts import (
    ExperienceTrajectory,
    GenerationTier,
    KeyValue,
    QualityRecord,
    QualitySeverity,
    ScenarioBlueprint,
    TurnRole,
)
from .llm import JsonCompletion, LLMResponseError, TokenUsage
from .prompt_manager import parse_render_request

ASSET_SCHEMA_VERSION = "cursor-authored-render.v1"
MODEL_FAMILY = "cursor-agent/gpt-5.6-sol-authored-v1"
DEFAULT_ASSET_BUNDLE = "unified_v1"
_ASSET_BUNDLE_RE = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class AuthoredTurn:
    turn_index: int
    role: str
    variants: tuple[str, ...]


@dataclass(frozen=True)
class AuthoredSession:
    session_index: int
    turns: tuple[AuthoredTurn, ...]


@dataclass(frozen=True)
class AuthoredScenario:
    scenario_id: str
    family: str
    language: str
    sessions: tuple[AuthoredSession, ...]


@dataclass(frozen=True)
class CursorAssetReport:
    asset_hash: str
    family_count: int
    scenario_count: int
    turn_count: int
    variant_count: int


@dataclass(frozen=True)
class CursorAssetNoveltyReport:
    candidate_asset_hash: str
    reference_asset_hash: str
    candidate_variant_count: int
    reference_variant_count: int
    normalized_overlap_count: int


class CursorAuthoredJsonClient:
    """Select from checked-in Cursor-authored variants without an API call."""

    def __init__(
        self,
        *,
        asset_root: Path | None = None,
        asset_bundle: str | None = None,
    ) -> None:
        resolved_bundle = asset_bundle or (
            DEFAULT_ASSET_BUNDLE if asset_root is None else "custom"
        )
        _validate_asset_bundle_id(resolved_bundle)
        root = asset_root or bundled_asset_root(resolved_bundle)
        scenarios, report = load_cursor_assets(root)
        self._scenarios = {item.scenario_id: item for item in scenarios}
        self._report = report
        self._asset_bundle = resolved_bundle
        model_family = (
            MODEL_FAMILY
            if resolved_bundle == DEFAULT_ASSET_BUNDLE
            else f"{MODEL_FAMILY}/{resolved_bundle}"
        )
        self._model_id = f"{model_family}@{report.asset_hash[:16]}"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def report(self) -> CursorAssetReport:
        return self._report

    def enrich_truth(
        self,
        trajectory: ExperienceTrajectory,
    ) -> ExperienceTrajectory:
        """Compile authored user events into truth before selecting any text."""

        if trajectory.generation_tier is not GenerationTier.STRUCTURAL:
            raise ValueError("Cursor truth enrichment requires a structural trajectory")
        scenario = self._scenarios.get(trajectory.scenario_ref)
        if scenario is None:
            raise ValueError(f"no Cursor-authored truth asset for {trajectory.scenario_ref!r}")
        authored_sessions = scenario.sessions
        if len(trajectory.sessions) != len(authored_sessions):
            raise ValueError("Cursor-authored session count mismatch")

        canonical_by_frame: dict[str, str] = {}
        for session, authored_session in zip(
            trajectory.sessions,
            authored_sessions,
            strict=True,
        ):
            if len(session.turns) != len(authored_session.turns):
                raise ValueError("Cursor-authored turn count mismatch")
            for turn, authored_turn in zip(
                session.turns,
                authored_session.turns,
                strict=True,
            ):
                if turn.role.value != authored_turn.role:
                    raise ValueError(f"Cursor-authored role mismatch at {turn.turn_id!r}")
                if turn.role is TurnRole.USER:
                    if turn.latent_frame_ref is None:
                        raise ValueError(f"Cursor-authored user turn lacks truth frame: {turn.turn_id}")
                    canonical_by_frame[turn.latent_frame_ref] = authored_turn.variants[0]

        enriched_frames = []
        for frame in trajectory.truth_frames:
            canonical_fact = canonical_by_frame.get(frame.frame_id)
            if canonical_fact is None:
                raise ValueError(f"Cursor asset has no canonical fact for {frame.frame_id!r}")
            existing = {item.key: item.value for item in frame.observable_facts}
            scenario_anchor = existing.pop("fact")
            enriched_frames.append(
                replace(
                    frame,
                    observable_facts=(
                        KeyValue(key="fact", value=canonical_fact),
                        KeyValue(key="scenario_anchor", value=scenario_anchor),
                        *tuple(KeyValue(key=key, value=value) for key, value in existing.items()),
                    ),
                )
            )

        asset_hash = self._report.asset_hash
        enriched_scenario_hash = stable_hash(
            {
                "base_scenario_hash": trajectory.scenario_hash,
                "cursor_asset_hash": asset_hash,
                "scenario_ref": trajectory.scenario_ref,
            }
        )
        bundle_metadata = (
            ()
            if self._asset_bundle == DEFAULT_ASSET_BUNDLE
            else (
                KeyValue(
                    key="cursor_render_asset_bundle",
                    value=self._asset_bundle,
                ),
            )
        )
        return replace(
            trajectory,
            scenario_hash=enriched_scenario_hash,
            truth_frames=tuple(enriched_frames),
            quality=trajectory.quality
            + (
                QualityRecord(
                    quality_id=(f"{trajectory.trajectory_id}:quality:cursor-authored-truth"),
                    check_kind="cursor_authored_truth_before_render",
                    passed=True,
                    severity=QualitySeverity.INFO,
                    score=1.0,
                    evidence_refs=(asset_hash,),
                    description=(
                        "Canonical user events were compiled from checked-in "
                        "Cursor assets before expression variants were selected."
                    ),
                ),
            ),
            provenance=replace(
                trajectory.provenance,
                scenario_hash=enriched_scenario_hash,
                generator_version=(f"{trajectory.provenance.generator_version}+cursor-assets-{asset_hash[:16]}"),
            ),
            metadata=trajectory.metadata
            + (
                KeyValue(
                    key="cursor_render_asset_hash",
                    value=asset_hash,
                ),
            )
            + bundle_metadata,
        )

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> JsonCompletion:
        if not system_prompt.strip():
            raise LLMResponseError("centralized renderer system prompt is empty")
        request = parse_render_request(user_prompt)
        trajectory_id = _required_string(request, "trajectory_id")
        scenario_ref = _required_string(request, "scenario_ref")
        variation_seed = request.get("variation_seed")
        if type(variation_seed) is not int or variation_seed < 0:
            raise LLMResponseError("variation_seed must be a non-negative integer")
        raw_slots = request.get("slots")
        if not isinstance(raw_slots, list):
            raise LLMResponseError("render request slots must be an array")
        scenario = self._scenarios.get(scenario_ref)
        if scenario is None:
            raise LLMResponseError(f"no Cursor-authored render asset for scenario {scenario_ref!r}")
        authored_turns = tuple(turn for session in scenario.sessions for turn in session.turns)
        if len(raw_slots) != len(authored_turns):
            raise LLMResponseError(
                f"scenario {scenario_ref!r} expects {len(authored_turns)} slots, received {len(raw_slots)}"
            )

        rendered_slots: list[dict[str, str]] = []
        for turn_ordinal, (raw_slot, authored) in enumerate(zip(raw_slots, authored_turns, strict=True)):
            if not isinstance(raw_slot, dict):
                raise LLMResponseError("render request slot must be an object")
            turn_id = _required_string(raw_slot, "turn_id")
            role = _required_string(raw_slot, "role")
            if role != authored.role:
                raise LLMResponseError(f"authored role mismatch for {turn_id!r}: {authored.role!r} != {role!r}")
            variant_index = _variant_index(
                variation_seed=variation_seed,
                turn_ordinal=turn_ordinal,
                turn_id=turn_id,
                asset_hash=self._report.asset_hash,
                variant_count=len(authored.variants),
            )
            rendered_slots.append(
                {
                    "turn_id": turn_id,
                    "text": authored.variants[variant_index],
                }
            )

        payload = {
            "trajectory_id": trajectory_id,
            "slots": rendered_slots,
        }
        request_id = stable_hash(
            {
                "asset_hash": self._report.asset_hash,
                "trajectory_id": trajectory_id,
                "variation_seed": variation_seed,
            }
        )
        return JsonCompletion(
            model_id=self.model_id,
            request_id=f"cursor-authored:{request_id}",
            payload_json=canonical_json(payload),
            usage=TokenUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            ),
            cost_usd=0.0,
        )


def bundled_asset_root(
    asset_bundle: str = DEFAULT_ASSET_BUNDLE,
) -> Path:
    _validate_asset_bundle_id(asset_bundle)
    package_root = Path(__file__).parent
    if asset_bundle == DEFAULT_ASSET_BUNDLE:
        return package_root / "render_assets"
    return package_root / "render_asset_bundles" / asset_bundle


def load_cursor_assets(
    root: Path | None = None,
) -> tuple[tuple[AuthoredScenario, ...], CursorAssetReport]:
    asset_root = root or bundled_asset_root()
    paths = tuple(sorted(asset_root.glob("*.json")))
    if not paths:
        raise FileNotFoundError(f"no Cursor render assets under {asset_root}")

    scenarios: list[AuthoredScenario] = []
    families: set[str] = set()
    payload_hashes: list[str] = []
    for path in paths:
        payload = _load_json_object(path)
        _require_exact_fields(
            payload,
            {"schema_version", "family", "scenarios"},
            source=path.name,
        )
        if payload["schema_version"] != ASSET_SCHEMA_VERSION:
            raise ValueError(f"{path.name} has unsupported asset schema")
        family = _required_string(payload, "family")
        if path.stem != family:
            raise ValueError(f"Cursor asset filename {path.name!r} must match family {family!r}")
        if family in families:
            raise ValueError(f"duplicate Cursor asset family {family!r}")
        families.add(family)
        raw_scenarios = payload["scenarios"]
        if not isinstance(raw_scenarios, list) or not raw_scenarios:
            raise ValueError(f"{path.name}.scenarios must be a non-empty array")
        scenarios.extend(
            _parse_scenario(
                raw,
                family=family,
                source=f"{path.name}.scenarios[{index}]",
            )
            for index, raw in enumerate(raw_scenarios)
        )
        payload_hashes.append(stable_hash(payload))

    scenario_ids = [item.scenario_id for item in scenarios]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("Cursor render assets repeat a scenario_id")
    turn_count = sum(len(session.turns) for scenario in scenarios for session in scenario.sessions)
    variant_count = sum(
        len(turn.variants) for scenario in scenarios for session in scenario.sessions for turn in session.turns
    )
    report = CursorAssetReport(
        asset_hash=stable_hash(
            {
                "schema_version": ASSET_SCHEMA_VERSION,
                "payload_hashes": payload_hashes,
            }
        ),
        family_count=len(families),
        scenario_count=len(scenarios),
        turn_count=turn_count,
        variant_count=variant_count,
    )
    return tuple(sorted(scenarios, key=lambda item: item.scenario_id)), report


def validate_cursor_assets(
    blueprints: tuple[ScenarioBlueprint, ...],
    *,
    root: Path | None = None,
) -> CursorAssetReport:
    scenarios, report = load_cursor_assets(root)
    expected = {item.scenario_id: item for item in blueprints}
    observed = {item.scenario_id: item for item in scenarios}
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        unknown = sorted(set(observed) - set(expected))
        raise ValueError(f"Cursor render scenario coverage mismatch; missing={missing}, unknown={unknown}")
    expected_families = {item.family for item in blueprints}
    if report.family_count != len(expected_families):
        raise ValueError(f"Cursor render family coverage is {report.family_count}, expected {len(expected_families)}")

    for scenario_id, scenario in observed.items():
        blueprint = expected[scenario_id]
        if scenario.family != blueprint.family:
            raise ValueError(f"{scenario_id} family does not match blueprint")
        if scenario.language != blueprint.language:
            raise ValueError(f"{scenario_id} language does not match blueprint")
        if len(scenario.sessions) != blueprint.sessions:
            raise ValueError(f"{scenario_id} session count does not match blueprint")
        for session, expected_turn_count in zip(
            scenario.sessions,
            blueprint.turns_per_session,
            strict=True,
        ):
            if len(session.turns) != expected_turn_count:
                raise ValueError(f"{scenario_id} session {session.session_index} turn count does not match blueprint")
        private_truth = tuple(text.strip() for text in blueprint.private_truth)
        leaked = [
            f"{scenario_id}:{session.session_index}:{turn.turn_index}"
            for session in scenario.sessions
            for turn in session.turns
            if any(truth and truth in variant for truth in private_truth for variant in turn.variants)
        ]
        if leaked:
            raise ValueError(f"{scenario_id} variants copy private truth at {leaked[:5]}")
    return report


def validate_cursor_asset_novelty(
    *,
    candidate_root: Path,
    reference_root: Path | None = None,
) -> CursorAssetNoveltyReport:
    candidate, candidate_report = load_cursor_assets(candidate_root)
    reference, reference_report = load_cursor_assets(
        reference_root or bundled_asset_root(),
    )
    candidate_entries = [
        (
            _normalized_variant(variant),
            (
                f"{scenario.family}:{scenario.scenario_id}:"
                f"s{session.session_index}:t{turn.turn_index}:v{variant_index}"
            ),
        )
        for scenario in candidate
        for session in scenario.sessions
        for turn in session.turns
        for variant_index, variant in enumerate(turn.variants)
    ]
    candidate_variant_list = [value for value, _ in candidate_entries]
    candidate_variants = set(candidate_variant_list)
    if len(candidate_variants) != len(candidate_variant_list):
        duplicate_values = {
            value
            for value, count in Counter(candidate_variant_list).items()
            if count > 1
        }
        duplicate_refs = sorted(
            ref
            for value, ref in candidate_entries
            if value in duplicate_values
        )
        raise ValueError(
            "Cursor asset bundle repeats normalized variants internally; "
            f"count={len(candidate_variant_list) - len(candidate_variants)}, "
            f"sample_refs={duplicate_refs[:10]}"
        )
    reference_variants = {
        _normalized_variant(variant)
        for scenario in reference
        for session in scenario.sessions
        for turn in session.turns
        for variant in turn.variants
    }
    overlaps = candidate_variants & reference_variants
    if overlaps:
        overlap_hashes = sorted(
            hashlib.sha256(value.encode("utf-8")).hexdigest()
            for value in overlaps
        )
        overlap_refs = sorted(
            ref
            for value, ref in candidate_entries
            if value in overlaps
        )
        raise ValueError(
            "Cursor asset bundle reuses normalized variants from the "
            f"reference bundle; count={len(overlaps)}, "
            f"sample_refs={overlap_refs[:10]}, "
            f"sample_hashes={overlap_hashes[:5]}"
        )
    return CursorAssetNoveltyReport(
        candidate_asset_hash=candidate_report.asset_hash,
        reference_asset_hash=reference_report.asset_hash,
        candidate_variant_count=candidate_report.variant_count,
        reference_variant_count=reference_report.variant_count,
        normalized_overlap_count=0,
    )


def _parse_scenario(
    raw: object,
    *,
    family: str,
    source: str,
) -> AuthoredScenario:
    if not isinstance(raw, dict):
        raise TypeError(f"{source} must be an object")
    _require_exact_fields(
        raw,
        {"scenario_id", "language", "sessions"},
        source=source,
    )
    scenario_id = _required_string(raw, "scenario_id")
    language = _required_string(raw, "language")
    if language not in {"zh", "en", "bilingual"}:
        raise ValueError(f"{source}.language is invalid")
    raw_sessions = raw["sessions"]
    if not isinstance(raw_sessions, list) or not raw_sessions:
        raise ValueError(f"{source}.sessions must be a non-empty array")
    sessions = tuple(
        _parse_session(item, source=f"{source}.sessions[{index}]") for index, item in enumerate(raw_sessions)
    )
    for index, session in enumerate(sessions):
        if session.session_index != index:
            raise ValueError(f"{source} session_index must equal position")
    return AuthoredScenario(
        scenario_id=scenario_id,
        family=family,
        language=language,
        sessions=sessions,
    )


def _parse_session(raw: object, *, source: str) -> AuthoredSession:
    if not isinstance(raw, dict):
        raise TypeError(f"{source} must be an object")
    _require_exact_fields(raw, {"session_index", "turns"}, source=source)
    session_index = raw["session_index"]
    if type(session_index) is not int or session_index < 0:
        raise ValueError(f"{source}.session_index must be non-negative integer")
    raw_turns = raw["turns"]
    if not isinstance(raw_turns, list) or not raw_turns:
        raise ValueError(f"{source}.turns must be a non-empty array")
    turns = tuple(_parse_turn(item, source=f"{source}.turns[{index}]") for index, item in enumerate(raw_turns))
    for index, turn in enumerate(turns):
        expected_role = "user" if index % 2 == 0 else "assistant"
        if turn.turn_index != index:
            raise ValueError(f"{source} turn_index must equal position")
        if turn.role != expected_role:
            raise ValueError(f"{source} roles must alternate from user")
    return AuthoredSession(session_index=session_index, turns=turns)


def _parse_turn(raw: object, *, source: str) -> AuthoredTurn:
    if not isinstance(raw, dict):
        raise TypeError(f"{source} must be an object")
    _require_exact_fields(raw, {"turn_index", "role", "variants"}, source=source)
    turn_index = raw["turn_index"]
    if type(turn_index) is not int or turn_index < 0:
        raise ValueError(f"{source}.turn_index must be non-negative integer")
    role = _required_string(raw, "role")
    if role not in {"user", "assistant"}:
        raise ValueError(f"{source}.role is invalid")
    raw_variants = raw["variants"]
    if not isinstance(raw_variants, list) or len(raw_variants) != 4:
        raise ValueError(f"{source}.variants must contain exactly four strings")
    variants = tuple(raw_variants)
    if any(not isinstance(item, str) or not item.strip() for item in variants):
        raise ValueError(f"{source}.variants must contain non-empty strings")
    if len(set(variants)) != len(variants):
        raise ValueError(f"{source}.variants must be unique")
    return AuthoredTurn(
        turn_index=turn_index,
        role=role,
        variants=variants,
    )


def _variant_index(
    *,
    variation_seed: int,
    turn_ordinal: int,
    turn_id: str,
    asset_hash: str,
    variant_count: int,
) -> int:
    if turn_ordinal < 4:
        return (variation_seed // (variant_count**turn_ordinal)) % variant_count
    digest = hashlib.sha256(f"{variation_seed}:{turn_id}:{asset_hash}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % variant_count


def _normalized_variant(value: str) -> str:
    return " ".join(value.casefold().split())


def _validate_asset_bundle_id(value: str) -> None:
    if _ASSET_BUNDLE_RE.fullmatch(value) is None:
        raise ValueError(
            "Cursor asset bundle must match ^[a-z][a-z0-9_]*$"
        )


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid Cursor render asset: {path}") from error
    if not isinstance(decoded, dict):
        raise TypeError(f"Cursor render asset root must be object: {path}")
    return decoded


def _required_string(mapping: dict[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _require_exact_fields(
    mapping: dict[str, object],
    expected: set[str],
    *,
    source: str,
) -> None:
    if set(mapping) != expected:
        raise ValueError(
            f"{source} fields mismatch; "
            f"missing={sorted(expected - set(mapping))}, "
            f"unknown={sorted(set(mapping) - expected)}"
        )


__all__ = [
    "ASSET_SCHEMA_VERSION",
    "DEFAULT_ASSET_BUNDLE",
    "MODEL_FAMILY",
    "AuthoredScenario",
    "AuthoredSession",
    "AuthoredTurn",
    "CursorAssetNoveltyReport",
    "CursorAssetReport",
    "CursorAuthoredJsonClient",
    "bundled_asset_root",
    "load_cursor_assets",
    "validate_cursor_asset_novelty",
    "validate_cursor_assets",
]
