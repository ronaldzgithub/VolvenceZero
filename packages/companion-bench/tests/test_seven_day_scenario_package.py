from __future__ import annotations

import json
from pathlib import Path
import re

import yaml

from companion_bench.spec import load_scenario_yaml


ROOT = Path(__file__).resolve().parents[1] / "src/companion_bench"
PACKAGE = ROOT / "scenario_packages/seven_day_companion_v1"
SCENARIOS = ROOT / "scenarios/seven_day"


def test_package_manifest_and_required_components() -> None:
    manifest = yaml.safe_load((PACKAGE / "manifest.yaml").read_text())
    assert re.fullmatch(r"[a-z][a-z0-9_]*", manifest["name"])
    assert len(manifest["explanation"]) >= 200
    assert manifest["components"] == {
        "ssot_fragment": "ssot_fragment.json",
        "scenes": "scenes.yaml",
        "test_suite": "test_suite.yaml",
    }
    for relative in manifest["components"].values():
        assert (PACKAGE / relative).is_file()


def test_every_path_is_referenced_and_phase_order_is_contiguous() -> None:
    ssot = json.loads((PACKAGE / "ssot_fragment.json").read_bytes())
    path_ids = {path["path_id"] for path in ssot["paths"]}
    referenced = {arc["path_id"] for arc in ssot["arc_specs"]}
    assert path_ids == referenced
    for arc in ssot["arc_specs"]:
        assert [phase["phase_order"] for phase in arc["phases"]] == list(
            range(7)
        )


def test_six_scenarios_cover_three_personas_and_two_arc_types() -> None:
    scenes = yaml.safe_load((PACKAGE / "scenes.yaml").read_text())
    assert len(scenes["scenes"]) == 6
    assert {scene["persona_id"] for scene in scenes["scenes"]} == {
        "researcher",
        "nurse",
        "designer",
    }
    assert {scene["arc_type"] for scene in scenes["scenes"]} == {
        "progressive_warmth",
        "rupture_repair",
    }
    for scene in scenes["scenes"]:
        spec = load_scenario_yaml(SCENARIOS / f"{scene['scenario_id']}.yaml")
        assert spec.arc_length_sessions == 7
        assert spec.session_turn_range == (5, 5)
        assert spec.inter_session_gap_days == (1, 1, 1, 1, 1, 1)


def test_l4_event_coverage_and_semantic_routing_contract() -> None:
    scenes = yaml.safe_load((PACKAGE / "scenes.yaml").read_text())
    for schedule in scenes["daily_event_schedule"].values():
        tags = {
            tag for day in schedule for tag in day["required_event_tags"]
        }
        assert tags == {"callback", "emotion", "boundary"}
        assert [day["day_index"] for day in schedule] == list(range(1, 8))
    routing = scenes["semantic_routing"]
    assert "embedding" in routing["method"]
    assert "structured_output" in routing["method"]
    assert set(routing["forbidden"]) == {
        "substring_matching",
        "regex_content_matching",
        "keyword_dictionary",
    }


def test_test_suite_has_required_positive_negative_and_coherence_cases() -> None:
    suite = yaml.safe_load((PACKAGE / "test_suite.yaml").read_text())
    assert len(suite["routing_tests"]) >= 6
    assert any(
        test["case_type"] == "negative" for test in suite["routing_tests"]
    )
    assert len(suite["llm_evaluation"]["semantic_coherence"]) >= 3
