from __future__ import annotations

import copy
import json
import math
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from praxist.plugins.workflow_stages.research_loop.backend import frontier
from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger


TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT))

from evaluations.context_replay import run as evaluator  # noqa: E402


TASK_CONFIG = yaml.safe_load((TASK_ROOT / "task.yaml").read_text(encoding="utf-8"))
DATASET_MANIFEST = {
    "visibility": "public-development-replay",
    "source_base_revision": "test-revision",
    "corpus": {"brain_trajectory_tree_sha256": "a" * 64},
}


def _measured_metrics(
    *,
    complete: bool,
    scaling_margin: float = 0.04,
    selection_gate_passed: bool = True,
    render_retention_gate_passed: bool = True,
    protocol_integrity_passed: bool = True,
    strict_budget_passed: bool = True,
    suspect_protocol: bool = False,
    suspect_leakage: bool = False,
    late_after_generation_boundary: bool = False,
) -> dict[str, object]:
    retention_gate_passed = selection_gate_passed and render_retention_gate_passed
    scaling_gate_passed = scaling_margin >= 0.0
    promotion_eligible = (
        complete
        and retention_gate_passed
        and scaling_gate_passed
        and strict_budget_passed
        and protocol_integrity_passed
        and not suspect_protocol
        and not suspect_leakage
        and not late_after_generation_boundary
    )
    return {
        "scaling_margin": scaling_margin,
        "context_token_ratio": evaluator.MAX_TOKEN_RATIO - scaling_margin,
        "worst_chain_scaling_margin": scaling_margin - 0.01,
        "recalled_entry_selection_coverage": 0.90,
        "failed_entry_selection_coverage": 1.0,
        "recalled_entry_retention": 1.0,
        "failed_entry_retention": 1.0,
        "strict_budget_pass_rate": 1.0 if strict_budget_passed else 0.0,
        "mean_context_tokens": 500.0 - scaling_margin,
        "candidate_context_tokens": 36000,
        "steelman_context_tokens": 500000,
        "recalled_entries_available": 500,
        "recalled_entries_selected": 450,
        "recalled_entries_retained": 450,
        "failed_entries_available": 100,
        "failed_entries_selected": 100,
        "failed_entries_retained": 100,
        "evaluation_units": 8 if complete else 1,
        "evaluated_contexts": 72 if complete else 9,
        "evaluator_wall_seconds": 1.0,
        "protocol_integrity_passed": protocol_integrity_passed,
        "protocol_integrity_failed": not protocol_integrity_passed,
        "selection_gate_passed": selection_gate_passed,
        "render_retention_gate_passed": render_retention_gate_passed,
        "retention_gate_passed": retention_gate_passed,
        "scaling_gate_passed": scaling_gate_passed,
        "strict_budget_passed": strict_budget_passed,
        "scored_complete": complete,
        "promotion_eligible": promotion_eligible,
        "is_smoke_eval": not complete,
        "partial": not complete,
        "scout_only": not complete,
        "validation_only": not complete,
        "validation_only_result": not complete,
        "suspect_protocol": suspect_protocol,
        "suspect_leakage": suspect_leakage,
        "late_after_generation_boundary": late_after_generation_boundary,
    }


def _summary(
    variant_id: str,
    *,
    mode: str = "complete",
    scaling_margin: float = 0.04,
    **metric_overrides: object,
) -> dict[str, object]:
    policy = {
        **evaluator.DEFAULT_POLICY,
        "schema_version": evaluator.POLICY_VERSION,
        "variant_id": variant_id,
    }
    effective_policy = {key: policy[key] for key in evaluator.DEFAULT_POLICY}
    metrics = _measured_metrics(
        complete=mode == "complete",
        scaling_margin=scaling_margin,
        **metric_overrides,
    )
    return evaluator._build_summary(
        mode=mode,
        policy=policy,
        effective_policy=effective_policy,
        manifest=DATASET_MANIFEST,
        metrics=metrics,
        producer={"generation_id": 0, "peer_id": "contract-peer"},
    )


def _write_policy(path: Path, payload: dict[str, object]) -> Path:
    path.mkdir()
    (path / "policy.json").write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def test_effective_config_defaults_and_replication_are_exact(tmp_path: Path) -> None:
    minimal = {
        "schema_version": evaluator.POLICY_VERSION,
        "variant_id": "config_contract",
    }
    explicit = {**minimal, **evaluator.DEFAULT_POLICY}
    changed = {**explicit, "max_context_chars": 3000}

    minimal_policy, minimal_effective = evaluator._load_policy(
        _write_policy(tmp_path / "minimal", minimal)
    )
    explicit_policy, explicit_effective = evaluator._load_policy(
        _write_policy(tmp_path / "explicit", explicit)
    )
    changed_policy, changed_effective = evaluator._load_policy(
        _write_policy(tmp_path / "changed", changed)
    )
    metrics = _measured_metrics(complete=True)

    minimal_summary = evaluator._build_summary(
        mode="complete",
        policy=minimal_policy,
        effective_policy=minimal_effective,
        manifest=DATASET_MANIFEST,
        metrics=copy.deepcopy(metrics),
    )
    explicit_summary = evaluator._build_summary(
        mode="complete",
        policy=explicit_policy,
        effective_policy=explicit_effective,
        manifest=DATASET_MANIFEST,
        metrics=copy.deepcopy(metrics),
    )
    changed_summary = evaluator._build_summary(
        mode="complete",
        policy=changed_policy,
        effective_policy=changed_effective,
        manifest=DATASET_MANIFEST,
        metrics=copy.deepcopy(metrics),
    )

    digest = str(minimal_summary["effective_config_digest"])
    assert digest == explicit_summary["effective_config_digest"]
    assert digest != changed_summary["effective_config_digest"]

    exact_replication = evaluator._build_summary(
        mode="complete",
        policy=explicit_policy,
        effective_policy=explicit_effective,
        manifest=DATASET_MANIFEST,
        metrics=copy.deepcopy(metrics),
        replication_of_effective_config_sha256=digest,
    )
    mismatched_replication = evaluator._build_summary(
        mode="complete",
        policy=changed_policy,
        effective_policy=changed_effective,
        manifest=DATASET_MANIFEST,
        metrics=copy.deepcopy(metrics),
        replication_of_effective_config_sha256=digest,
    )
    assert exact_replication["replication_effective_config_status"] == "matched"
    assert mismatched_replication["replication_effective_config_status"] == "mismatched"

    with pytest.raises(evaluator.EvaluationError, match="violates policy.schema.json"):
        evaluator._load_policy(
            _write_policy(tmp_path / "invalid", {**minimal, "unknown_surface": True})
        )


def test_lane_routing_uses_evaluator_summaries_and_keeps_incubator_reachable(
    tmp_path: Path,
) -> None:
    lanes = TASK_CONFIG["evaluation"]["frontier_lanes"]
    parent_lanes = [lane for lane in lanes if lane["parent_eligible"]]
    clean = _summary("clean_contract")

    assert {lane["name"] for lane in parent_lanes} == {"confirmed", "incubator"}
    assert all(frontier._matches_lane_filters(clean, lane) for lane in parent_lanes)

    preliminary = _summary("preliminary_contract", mode="preliminary")
    protocol_failed = _summary(
        "protocol_failed_contract",
        protocol_integrity_passed=False,
    )
    suspect = _summary("suspect_contract", suspect_protocol=True)
    late = _summary("late_contract", late_after_generation_boundary=True)
    for non_parent in (preliminary, protocol_failed, suspect, late):
        assert not any(
            frontier._matches_lane_filters(non_parent, lane) for lane in parent_lanes
        )

    summaries = [
        _summary("primary_best", scaling_margin=0.08),
        _summary("primary_second", scaling_margin=0.07),
        _summary("primary_third", scaling_margin=0.06),
        _summary("incubator_after_top_k", scaling_margin=0.05),
    ]
    findings = [
        {**summary, "id": str(summary["variant_id"]), "finding_type": "result"}
        for summary in summaries
    ]
    store = frontier.FrontierStore(
        tmp_path / "frontier",
        promote_top_k=TASK_CONFIG["generation_policy"]["promote_top_k"],
        primary_metric=TASK_CONFIG["evaluation"]["primary_metric"],
        metric_direction=TASK_CONFIG["evaluation"]["direction"],
        frontier_lanes=lanes,
        maturity_policy=TASK_CONFIG["evaluation"]["maturity_policy"],
    )
    store.promote(0, findings)
    lane_frontiers = store.get_manifest()["lane_frontiers"]

    assert [entry["variant_name"] for entry in lane_frontiers["confirmed"]] == [
        "primary_best",
        "primary_second",
        "primary_third",
    ]
    assert [entry["variant_name"] for entry in lane_frontiers["incubator"]] == [
        "incubator_after_top_k"
    ]


def _trigger(
    root: Path,
    *,
    started_minutes_ago: float,
    active_work: dict[str, int],
) -> synthesis_trigger.SynthesisTrigger:
    trigger_config = TASK_CONFIG["synthesis_trigger"]
    generation = TASK_CONFIG["generation_policy"]
    gen_dir = root / "gen_0"
    gen_dir.mkdir(parents=True)
    return synthesis_trigger.SynthesisTrigger(
        run_dir=root,
        gen_dir=gen_dir,
        gen_id=0,
        gen_start_time=time.time() - started_minutes_ago * 60.0,
        min_findings=trigger_config["min_findings"],
        min_interval_minutes=trigger_config["min_interval_minutes"],
        max_interval_minutes=trigger_config["max_interval_minutes"],
        min_contributing_peers=trigger_config["min_contributing_peers"],
        adaptive_policy=trigger_config["adaptive"],
        maturity_policy=TASK_CONFIG["evaluation"]["maturity_policy"],
        mature_quorum_fraction=trigger_config["mature_quorum_fraction"],
        cohort_size=generation["cohort_size"],
        cohort_active_peers_callback=lambda: active_work["count"],
    )


def _patch_trigger_state(
    trigger: synthesis_trigger.SynthesisTrigger,
    *,
    mature: dict[str, int],
    protected: dict[str, int],
    findings: int,
    peers: int,
):
    return (
        patch.object(trigger, "_query_gen_state", return_value=(findings, peers)),
        patch.object(trigger, "_query_adaptive_state", return_value=(0.0, 0)),
        patch.object(trigger, "_query_mature_state", side_effect=lambda: mature["count"]),
        patch.object(
            trigger,
            "mature_result_count",
            side_effect=lambda synchronize=False: mature["count"],
        ),
        patch.object(
            trigger,
            "_active_protected_pid_count",
            side_effect=lambda: protected["count"],
        ),
    )


def test_closing_policy_requires_maturity_and_preserves_liveness(tmp_path: Path) -> None:
    config = TASK_CONFIG["synthesis_trigger"]
    cohort = TASK_CONFIG["generation_policy"]["cohort_size"]
    expected_quorum = math.ceil(cohort * config["mature_quorum_fraction"])
    assert expected_quorum == 1

    active = {"count": 1}
    mature = {"count": 0}
    protected = {"count": 0}
    trigger = _trigger(
        tmp_path / "normal",
        started_minutes_ago=config["min_interval_minutes"] + 1,
        active_work=active,
    )
    state_patches = _patch_trigger_state(
        trigger,
        mature=mature,
        protected=protected,
        findings=config["min_findings"],
        peers=config["min_contributing_peers"],
    )
    with ExitStack() as stack:
        for state_patch in state_patches:
            stack.enter_context(state_patch)
        stack.enter_context(patch.object(trigger, "begin_assessment", return_value=True))
        stack.enter_context(
            patch.object(
                trigger,
                "begin_closing",
                side_effect=lambda _snapshot: setattr(trigger, "_closing", True),
            )
        )
        below_quorum = trigger.evaluate()
        assert not below_quorum.fired
        assert below_quorum.reason == "assessment_mature_topup"
        assert below_quorum.required_mature_result_peers == expected_quorum

        mature["count"] = expected_quorum
        active["count"] = 0
        protected["count"] = 1
        draining = trigger.evaluate()
        assert not draining.fired
        assert draining.reason == "draining_active_evals"
        assert draining.active_protected_pids == 1

        protected["count"] = 0
        closed = trigger.evaluate()
        assert closed.fired
        assert closed.reason == "mature_quorum"

    safety_active = {"count": 1}
    safety_trigger = _trigger(
        tmp_path / "safety",
        started_minutes_ago=config["max_interval_minutes"] + 1,
        active_work=safety_active,
    )
    safety_patches = _patch_trigger_state(
        safety_trigger,
        mature={"count": 0},
        protected={"count": 0},
        findings=0,
        peers=0,
    )
    with safety_patches[0], safety_patches[1], safety_patches[2], safety_patches[3], safety_patches[4]:
        safety = safety_trigger.evaluate()
    assert safety.fired
    assert safety.reason == "safety_cap"
    assert safety.mature_result_peers == 0
    assert safety.required_mature_result_peers == expected_quorum

    drained_active = {"count": 0}
    drained_trigger = _trigger(
        tmp_path / "drained",
        started_minutes_ago=2,
        active_work=drained_active,
    )
    drained_patches = _patch_trigger_state(
        drained_trigger,
        mature={"count": 0},
        protected={"count": 0},
        findings=0,
        peers=0,
    )
    with drained_patches[0], drained_patches[1], drained_patches[2], drained_patches[3], drained_patches[4]:
        drained = drained_trigger.evaluate()
    assert drained.fired
    assert drained.reason == "cohort_drained_insufficient_mature"
