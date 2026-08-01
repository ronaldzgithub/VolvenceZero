from __future__ import annotations

import json
from pathlib import Path

import pytest

from lifeform_domain_emogpt import (
    build_companion_package,
    load_companion_playbook_overlay,
    resolve_companion_package_overlay,
)
from volvence_zero.application import compile_domain_experience_package
from volvence_zero.runtime import WiringLevel


def _overlay_payload() -> dict[str, object]:
    return {
        "schema_version": "companion-playbook-overlay.v1",
        "overlay_id": "companion-forge-reviewed-v1",
        "version": "1.0.0",
        "owner": "lifeform-domain-emogpt",
        "playbook_rules": [
            {
                "rule_id": "rid-companion:forge:repair-after-memory-gap",
                "problem_pattern": "relationship-memory-gap-repair",
                "recommended_regime": "repair_and_deescalation",
                "recommended_ordering": [
                    "acknowledge_memory_gap",
                    "avoid_invented_callback",
                    "invite_user_restoration",
                ],
                "recommended_pacing": "repair-first",
                "avoid_patterns": ["invented-callback", "confident-fabrication"],
                "knowledge_weight_hint": 0.3,
                "experience_weight_hint": 0.8,
                "applicability_scope": ["risk-medium", "relationship-continuity"],
                "confidence": 0.82,
                "description": "Repair a continuity gap without inventing shared history.",
            }
        ],
        "description": "Reviewed test overlay.",
    }


def _write_overlay(path: Path, payload: dict[str, object] | None = None) -> Path:
    path.write_text(
        json.dumps(payload or _overlay_payload(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def test_disabled_is_historical_package_and_does_not_read_asset(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"

    package = build_companion_package(
        playbook_overlay_wiring=WiringLevel.DISABLED,
        playbook_overlay_path=missing,
    )

    assert len(package.playbook_rules) == 4
    assert all(not rule.rule_id.startswith("rid-companion:forge:") for rule in package.playbook_rules)


def test_shadow_exposes_candidate_without_changing_live_package(tmp_path: Path) -> None:
    overlay_path = _write_overlay(tmp_path / "overlay.json")

    resolution = resolve_companion_package_overlay(
        wiring_level=WiringLevel.SHADOW,
        overlay_path=overlay_path,
    )
    shadow = build_companion_package(
        playbook_overlay_wiring=WiringLevel.SHADOW,
        playbook_overlay_path=overlay_path,
    )

    assert resolution.overlay is not None
    assert len(resolution.baseline_rules) == 4
    assert len(resolution.candidate_rules) == 5
    assert resolution.live_rules == resolution.baseline_rules
    assert resolution.applied is False
    assert shadow.playbook_rules == resolution.baseline_rules


def test_active_compiles_overlay_into_existing_application_owner(tmp_path: Path) -> None:
    overlay_path = _write_overlay(tmp_path / "overlay.json")

    active = build_companion_package(
        playbook_overlay_wiring=WiringLevel.ACTIVE,
        playbook_overlay_path=overlay_path,
    )
    compiled = compile_domain_experience_package(active)

    assert len(active.playbook_rules) == 5
    assert active.playbook_rules[-1].rule_id == "rid-companion:forge:repair-after-memory-gap"
    assert compiled.validation_report.valid
    assert compiled.rare_heavy_checkpoint.distilled_playbook_rules == active.playbook_rules


def test_asset_cannot_self_authorize_wiring(tmp_path: Path) -> None:
    payload = _overlay_payload()
    payload["wiring_level"] = "active"
    path = _write_overlay(tmp_path / "overlay.json", payload)

    with pytest.raises(ValueError, match="fields do not match schema"):
        load_companion_playbook_overlay(path)


def test_overlay_cannot_replace_baseline_problem_pattern(tmp_path: Path) -> None:
    payload = _overlay_payload()
    rules = payload["playbook_rules"]
    assert isinstance(rules, list)
    rules[0]["problem_pattern"] = "response-misread-as-dismissive"
    path = _write_overlay(tmp_path / "overlay.json", payload)

    with pytest.raises(ValueError, match="may not replace baseline problem_pattern"):
        resolve_companion_package_overlay(
            wiring_level=WiringLevel.SHADOW,
            overlay_path=path,
        )
