from __future__ import annotations

import sys
from pathlib import Path

import yaml

from praxist.plugins.workflow_stages.research_loop.backend.frontier import (
    _matches_lane_filters,
)

TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT))


def _task() -> dict[str, object]:
    return yaml.safe_load((TASK_ROOT / "task.yaml").read_text(encoding="utf-8"))


def _summary(*, passed: bool, random_control: bool = False) -> dict[str, object]:
    exit_classification = "PASS" if passed else "DOMAIN_LOCAL"
    metrics = {
        "qualification_margin": 0.05 if passed else -0.25,
        "same_view_balanced_accuracy": 0.875,
        "cross_view_balanced_accuracy": 0.8 if passed else 0.65,
        "worst_view_balanced_accuracy": 0.625 if passed else 0.375,
        "causal_target_margin_effect": 0.08,
        "random_control_separation": 0.03,
        "scored_complete": True,
        "protocol_integrity_passed": True,
        "qualification_passed": passed,
        "promotion_eligible": passed and not random_control,
        "protocol_integrity_failed": False,
        "random_control": random_control,
        "domain_local": exit_classification == "DOMAIN_LOCAL",
        "instrument_invalid": False,
        "is_smoke_eval": False,
        "partial": False,
        "scout_only": False,
        "validation_only": False,
        "validation_only_result": False,
        "late_after_generation_boundary": False,
        "suspect_protocol": False,
        "suspect_leakage": False,
    }
    return {
        "tier": "complete",
        "frontier_lane": "negative_control" if random_control else "performance",
        "promotion_lane": "negative_control" if random_control else "performance",
        "metrics": metrics,
    }


def test_frontier_lanes_keep_pass_incubator_and_control_distinct() -> None:
    lanes = {lane["name"]: lane for lane in _task()["evaluation"]["frontier_lanes"]}

    assert _matches_lane_filters(_summary(passed=True), lanes["confirmed"])
    assert not _matches_lane_filters(_summary(passed=False), lanes["confirmed"])
    assert _matches_lane_filters(_summary(passed=False), lanes["incubator"])
    assert _matches_lane_filters(
        _summary(passed=False, random_control=True), lanes["diagnostic"]
    )
    assert not _matches_lane_filters(
        _summary(passed=False, random_control=True), lanes["incubator"]
    )


def test_task_declares_resume_safe_runtime_and_exact_role_assets() -> None:
    task = _task()
    assert task["runtime_environment"]["python"] == "../../../.venv/bin/python"
    assert task["compute_budget"]["resource_scheduler"]["max_concurrent_experiments"] == 1
    assert task["dig_lite"]["generation_scope"] == "initial_only"
    assert task["quality_diversity"]["later_generations_enabled"] is True
    assert task["gems"]["enabled"] is False

    role_refs = task["praxist_plugins"]["panel"]["roles"]
    role_ids = {ref.partition(":")[2] for ref in role_refs}
    assert role_ids == {
        "peer_generalist",
        "starter",
        "solver",
        "analyst",
        "builder_pi",
        "skeptic_pi",
        "portfolio_pi",
        "external_validity_pi",
        "chair",
    }
    for role_id in role_ids:
        assert (TASK_ROOT / "roles" / role_id / "role.yaml").is_file()
        assert (TASK_ROOT / "roles" / role_id / "skill.md").is_file()
