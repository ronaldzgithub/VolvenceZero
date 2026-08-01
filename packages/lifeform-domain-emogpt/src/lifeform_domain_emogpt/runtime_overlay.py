"""Reviewed companion playbook overlay with explicit runtime wiring.

The overlay is a declarative, additive input to the existing
``strategy_playbook`` owner.  It never owns runtime state and it cannot change
its own wiring level: DISABLED/SHADOW/ACTIVE is supplied by trusted code outside
the editable asset.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from volvence_zero.application import PlaybookRule
from volvence_zero.runtime import WiringLevel


COMPANION_PLAYBOOK_OVERLAY_SCHEMA_VERSION = "companion-playbook-overlay.v1"
COMPANION_PLAYBOOK_OVERLAY_OWNER = "lifeform-domain-emogpt"
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "overlay_id",
        "version",
        "owner",
        "playbook_rules",
        "description",
    }
)
_RULE_FIELDS = frozenset(
    {
        "rule_id",
        "problem_pattern",
        "recommended_regime",
        "recommended_ordering",
        "recommended_pacing",
        "avoid_patterns",
        "knowledge_weight_hint",
        "experience_weight_hint",
        "applicability_scope",
        "confidence",
        "description",
    }
)


@dataclass(frozen=True)
class CompanionPlaybookOverlay:
    schema_version: str
    overlay_id: str
    version: str
    owner: str
    playbook_rules: tuple[PlaybookRule, ...]
    content_sha256: str
    description: str


@dataclass(frozen=True)
class CompanionPlaybookOverlayResolution:
    wiring_level: WiringLevel
    overlay: CompanionPlaybookOverlay | None
    baseline_rules: tuple[PlaybookRule, ...]
    candidate_rules: tuple[PlaybookRule, ...]
    live_rules: tuple[PlaybookRule, ...]
    applied: bool
    description: str


def companion_playbook_overlay_path() -> Path:
    return Path(__file__).resolve().parent / "runtime_assets" / "companion_playbook_overlay.json"


def load_companion_playbook_overlay(
    path: Path | None = None,
) -> CompanionPlaybookOverlay:
    source = (path or companion_playbook_overlay_path()).expanduser().resolve()
    try:
        payload_bytes = source.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read companion playbook overlay {source}: {exc}") from exc
    try:
        raw = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid companion playbook overlay JSON {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("companion playbook overlay must be a JSON object")
    _require_exact_fields(raw, _TOP_LEVEL_FIELDS, "companion playbook overlay")
    schema_version = _required_string(raw, "schema_version", "overlay")
    if schema_version != COMPANION_PLAYBOOK_OVERLAY_SCHEMA_VERSION:
        raise ValueError(
            "companion playbook overlay schema_version must be "
            f"{COMPANION_PLAYBOOK_OVERLAY_SCHEMA_VERSION!r}"
        )
    owner = _required_string(raw, "owner", "overlay")
    if owner != COMPANION_PLAYBOOK_OVERLAY_OWNER:
        raise ValueError(
            "companion playbook overlay owner must remain "
            f"{COMPANION_PLAYBOOK_OVERLAY_OWNER!r}"
        )
    rules_raw = raw["playbook_rules"]
    if not isinstance(rules_raw, list):
        raise ValueError("companion playbook overlay playbook_rules must be a list")
    rules = tuple(_parse_rule(item, index=index) for index, item in enumerate(rules_raw))
    identifiers = tuple(rule.rule_id for rule in rules)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("companion playbook overlay rule_id values must be unique")
    patterns = tuple(rule.problem_pattern for rule in rules)
    if len(patterns) != len(set(patterns)):
        raise ValueError("companion playbook overlay problem_pattern values must be unique")
    return CompanionPlaybookOverlay(
        schema_version=schema_version,
        overlay_id=_required_string(raw, "overlay_id", "overlay"),
        version=_required_string(raw, "version", "overlay"),
        owner=owner,
        playbook_rules=rules,
        content_sha256=hashlib.sha256(payload_bytes).hexdigest(),
        description=_required_string(raw, "description", "overlay"),
    )


def resolve_companion_playbook_overlay(
    *,
    baseline_rules: tuple[PlaybookRule, ...],
    wiring_level: WiringLevel = WiringLevel.DISABLED,
    overlay_path: Path | None = None,
) -> CompanionPlaybookOverlayResolution:
    if wiring_level is WiringLevel.DISABLED:
        return CompanionPlaybookOverlayResolution(
            wiring_level=wiring_level,
            overlay=None,
            baseline_rules=baseline_rules,
            candidate_rules=baseline_rules,
            live_rules=baseline_rules,
            applied=False,
            description="Companion playbook overlay is DISABLED; the asset was not read.",
        )
    overlay = load_companion_playbook_overlay(overlay_path)
    baseline_ids = {rule.rule_id for rule in baseline_rules}
    overlay_ids = {rule.rule_id for rule in overlay.playbook_rules}
    duplicate_ids = sorted(baseline_ids & overlay_ids)
    if duplicate_ids:
        raise ValueError(
            "companion playbook overlay may not replace baseline rule_id values: "
            f"{duplicate_ids}"
        )
    baseline_patterns = {rule.problem_pattern for rule in baseline_rules}
    overlay_patterns = {rule.problem_pattern for rule in overlay.playbook_rules}
    duplicate_patterns = sorted(baseline_patterns & overlay_patterns)
    if duplicate_patterns:
        raise ValueError(
            "companion playbook overlay is additive and may not replace baseline "
            f"problem_pattern values: {duplicate_patterns}"
        )
    candidate = baseline_rules + overlay.playbook_rules
    applied = wiring_level is WiringLevel.ACTIVE
    return CompanionPlaybookOverlayResolution(
        wiring_level=wiring_level,
        overlay=overlay,
        baseline_rules=baseline_rules,
        candidate_rules=candidate,
        live_rules=candidate if applied else baseline_rules,
        applied=applied,
        description=(
            f"Companion playbook overlay {overlay.overlay_id} loaded as "
            f"{wiring_level.value}; candidate_rules={len(candidate)}, applied={applied}."
        ),
    )


def _parse_rule(value: Any, *, index: int) -> PlaybookRule:
    context = f"playbook_rules[{index}]"
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    _require_exact_fields(value, _RULE_FIELDS, context)
    rule_id = _required_string(value, "rule_id", context)
    if not rule_id.startswith("rid-companion:forge:"):
        raise ValueError(f"{context}.rule_id must start with 'rid-companion:forge:'")
    recommended_regime = value["recommended_regime"]
    if recommended_regime is not None and (
        not isinstance(recommended_regime, str) or not recommended_regime.strip()
    ):
        raise ValueError(f"{context}.recommended_regime must be null or a non-empty string")
    return PlaybookRule(
        rule_id=rule_id,
        problem_pattern=_required_string(value, "problem_pattern", context),
        recommended_regime=recommended_regime.strip() if recommended_regime else None,
        recommended_ordering=_string_tuple(value, "recommended_ordering", context, allow_empty=False),
        recommended_pacing=_required_string(value, "recommended_pacing", context),
        avoid_patterns=_string_tuple(value, "avoid_patterns", context, allow_empty=True),
        knowledge_weight_hint=_bounded_number(value, "knowledge_weight_hint", context),
        experience_weight_hint=_bounded_number(value, "experience_weight_hint", context),
        applicability_scope=_string_tuple(value, "applicability_scope", context, allow_empty=False),
        confidence=_bounded_number(value, "confidence", context),
        description=_required_string(value, "description", context),
    )


def _require_exact_fields(value: dict[str, Any], expected: frozenset[str], context: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValueError(f"{context} fields do not match schema; missing={missing}, extra={extra}")


def _required_string(value: dict[str, Any], key: str, context: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return item.strip()


def _string_tuple(
    value: dict[str, Any],
    key: str,
    context: str,
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    item = value.get(key)
    if not isinstance(item, list) or (not allow_empty and not item):
        raise ValueError(f"{context}.{key} must be a {'possibly empty' if allow_empty else 'non-empty'} list")
    if not all(isinstance(part, str) and part.strip() for part in item):
        raise ValueError(f"{context}.{key} must contain only non-empty strings")
    normalized = tuple(part.strip() for part in item)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{context}.{key} values must be unique")
    return normalized


def _bounded_number(value: dict[str, Any], key: str, context: str) -> float:
    item = value.get(key)
    if not isinstance(item, (int, float)) or isinstance(item, bool):
        raise ValueError(f"{context}.{key} must be numeric")
    numeric = float(item)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{context}.{key} must be finite and within [0, 1]")
    return numeric


__all__ = [
    "COMPANION_PLAYBOOK_OVERLAY_OWNER",
    "COMPANION_PLAYBOOK_OVERLAY_SCHEMA_VERSION",
    "CompanionPlaybookOverlay",
    "CompanionPlaybookOverlayResolution",
    "companion_playbook_overlay_path",
    "load_companion_playbook_overlay",
    "resolve_companion_playbook_overlay",
]
