from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from praxist.plugins.workflow_stages.research_loop.backend.findings_collection import (
    _materialize_result_artifacts,
)
from praxist.plugins.workflow_stages.research_loop.backend.frontier import (
    FrontierStore,
    _matches_lane_filters,
)

TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT))

from evaluations.context_replay import run  # noqa: E402


def _rows(
    *,
    candidate_tokens: int,
    selected_entries: int,
    failed_selection_coverage: float = 1.0,
    chain_count: int = 8,
) -> tuple[run.ReplayRow, ...]:
    failed_available = (3, 3, 3, 3, 2, 2, 2, 2)
    if failed_selection_coverage == 0.95:
        failed_selected = (2, 3, 3, 3, 2, 2, 2, 2)
    else:
        failed_selected = failed_available
    return tuple(
        run.ReplayRow(
            chain_index=chain_index,
            episode_index=1,
            context_chars=candidate_tokens * 4,
            context_tokens=candidate_tokens,
            steelman_context_tokens=1000,
            available_entries=8,
            selected_entries=selected_entries,
            retained_entries=selected_entries,
            available_failed_entries=failed_available[chain_index],
            selected_failed_entries=failed_selected[chain_index],
            retained_failed_entries=failed_selected[chain_index],
            strict_budget_passed=True,
        )
        for chain_index in range(chain_count)
    )


def _write_summary(
    *,
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    variant_id: str,
    rows: tuple[run.ReplayRow, ...],
    max_recalled_entries: int,
    mode: str = "complete",
    replication_digest: str = "",
) -> tuple[Path, dict[str, Any]]:
    variant_dir = root / "variants" / variant_id
    variant_dir.mkdir(parents=True, exist_ok=True)
    (variant_dir / "policy.json").write_text(
        json.dumps(
            {
                "schema_version": run.POLICY_VERSION,
                "variant_id": variant_id,
                "max_recalled_entries": max_recalled_entries,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest = {
        "visibility": "public-development-replay",
        "source_base_revision": "fixture-revision",
        "corpus": {"brain_trajectory_tree_sha256": "a" * 64},
    }

    async def fake_replay(**_kwargs: Any) -> tuple[run.ReplayRow, ...]:
        return rows

    monkeypatch.setattr(run, "_verify_corpus", lambda: (manifest, root, {}))
    monkeypatch.setattr(run, "_replay", fake_replay)
    monkeypatch.setenv("PRAXIST_GENERATION_ID", "0")
    monkeypatch.setenv("PRAXIST_PEER_ID", f"peer-{variant_id}")
    output_dir = root / "results" / variant_id / mode
    summary_path = run.evaluate(
        argparse.Namespace(
            variant_dir=variant_dir,
            output_dir=output_dir,
            mode=mode,
            replication_of_effective_config_sha256=replication_digest,
        )
    )
    return summary_path, json.loads(summary_path.read_text(encoding="utf-8"))


def _task() -> dict[str, Any]:
    return yaml.safe_load((TASK_ROOT / "task.yaml").read_text(encoding="utf-8"))


def test_metrics_require_selection_floor_and_full_selected_lines() -> None:
    metrics = run._metrics(
        _rows(candidate_tokens=80, selected_entries=6),
        elapsed=1.0,
        complete=True,
    )
    assert metrics["recalled_entry_selection_coverage"] == pytest.approx(0.75)
    assert metrics["failed_entry_selection_coverage"] == pytest.approx(1.0)
    assert metrics["recalled_entry_retention"] == pytest.approx(1.0)
    assert metrics["failed_entry_retention"] == pytest.approx(1.0)
    assert metrics["retention_gate_passed"] is True
    assert metrics["scaling_gate_passed"] is True
    assert metrics["promotion_eligible"] is True

    truncated = list(_rows(candidate_tokens=80, selected_entries=6))
    truncated[0] = run.ReplayRow(
        **{
            **truncated[0].__dict__,
            "retained_entries": truncated[0].selected_entries - 1,
        }
    )
    truncated_metrics = run._metrics(tuple(truncated), elapsed=1.0, complete=True)
    assert truncated_metrics["render_retention_gate_passed"] is False
    assert truncated_metrics["promotion_eligible"] is False


def test_resolved_defaults_and_replication_digest_are_stable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, first = _write_summary(
        monkeypatch=monkeypatch,
        root=tmp_path / "first",
        variant_id="default_implicit",
        rows=_rows(candidate_tokens=80, selected_entries=6),
        max_recalled_entries=6,
    )
    digest = first["effective_config_digest"]
    _, replicated = _write_summary(
        monkeypatch=monkeypatch,
        root=tmp_path / "replicated",
        variant_id="default_explicit",
        rows=_rows(candidate_tokens=80, selected_entries=6),
        max_recalled_entries=6,
        replication_digest=digest,
    )
    assert replicated["effective_config_digest"] == digest
    assert replicated["replication_effective_config_status"] == "matched"

    _, changed = _write_summary(
        monkeypatch=monkeypatch,
        root=tmp_path / "changed",
        variant_id="changed_policy",
        rows=_rows(candidate_tokens=80, selected_entries=7),
        max_recalled_entries=7,
        replication_digest=digest,
    )
    assert changed["effective_config_digest"] != digest
    assert changed["replication_effective_config_status"] == "mismatched"


def test_actual_summary_materialization_preserves_attribution_and_ratios(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    summary_path, summary = _write_summary(
        monkeypatch=monkeypatch,
        root=tmp_path,
        variant_id="materialized_candidate",
        rows=_rows(candidate_tokens=80, selected_entries=6),
        max_recalled_entries=6,
        mode="preliminary",
    )
    findings = _materialize_result_artifacts(
        run_dir=tmp_path,
        gen_id=0,
        default_lane="task_candidate",
        default_family="coding_memory_inheritance_policy",
        scoring_metric_keys=["scaling_margin"],
        result_maturity_policy=_task()["evaluation"]["maturity_policy"],
    )
    assert summary_path.name == "evaluation_summary.json"
    assert len(findings) == 1
    finding = findings[0]
    assert finding["generation_id"] == 0
    assert finding["peer_id"] == "peer-materialized_candidate"
    assert finding["metrics"]["effort_ratio"] == summary["effort_ratio"] == 0.125
    assert finding["metrics"]["coverage_ratio"] == summary["coverage_ratio"] == 0.125
    assert finding["metrics"]["excluded_from_durable_frontier"] is True
    assert finding["metrics"]["exclusion_reason"] == "preliminary_or_incomplete_evidence"


def test_parent_lanes_are_reachable_and_non_parent_signals_are_excluded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = _task()
    complete_cases = (
        ("parent_a", 50, 6, 1.0),
        ("parent_b", 60, 7, 0.95),
        ("parent_c", 70, 7, 1.0),
        ("parent_d", 80, 8, 1.0),
    )
    for variant_id, tokens, selected, failed_coverage in complete_cases:
        _write_summary(
            monkeypatch=monkeypatch,
            root=tmp_path,
            variant_id=variant_id,
            rows=_rows(
                candidate_tokens=tokens,
                selected_entries=selected,
                failed_selection_coverage=failed_coverage,
            ),
            max_recalled_entries=selected,
        )
    _, preliminary = _write_summary(
        monkeypatch=monkeypatch,
        root=tmp_path,
        variant_id="preliminary_signal",
        rows=_rows(candidate_tokens=80, selected_entries=6, chain_count=1),
        max_recalled_entries=6,
        mode="preliminary",
    )

    findings = _materialize_result_artifacts(
        run_dir=tmp_path,
        gen_id=0,
        default_lane="task_candidate",
        default_family="coding_memory_inheritance_policy",
        scoring_metric_keys=["scaling_margin"],
        result_maturity_policy=task["evaluation"]["maturity_policy"],
    )
    store = FrontierStore(
        tmp_path / "frontier-regression",
        promote_top_k=task["generation_policy"]["promote_top_k"],
        primary_metric=task["evaluation"]["primary_metric"],
        metric_direction=task["evaluation"]["direction"],
        frontier_lanes=task["evaluation"]["frontier_lanes"],
        maturity_policy=task["evaluation"]["maturity_policy"],
    )
    picks = store._select_lane_frontier(findings)
    picked = {(item["variant_name"], item["_promoted_for_lane"]) for item in picks}
    assert {("parent_a", "confirmed"), ("parent_b", "confirmed"), ("parent_c", "confirmed")} <= picked
    assert ("parent_d", "incubator") in picked

    parent_lanes = [
        lane for lane in task["evaluation"]["frontier_lanes"] if lane["parent_eligible"]
    ]
    preliminary_finding = next(
        finding for finding in findings if finding["variant_name"] == "preliminary_signal"
    )
    assert preliminary["parent_authorized"] is False
    assert not any(_matches_lane_filters(preliminary_finding, lane) for lane in parent_lanes)

    confirmed_finding = next(
        finding for finding in findings if finding["variant_name"] == "parent_a"
    )
    protocol_failed = copy.deepcopy(confirmed_finding)
    protocol_failed["metrics"]["protocol_integrity_passed"] = False
    protocol_failed["metrics"]["protocol_integrity_failed"] = True
    assert not any(_matches_lane_filters(protocol_failed, lane) for lane in parent_lanes)

    suspect = copy.deepcopy(confirmed_finding)
    suspect["metrics"]["suspect_protocol"] = True
    assert not any(_matches_lane_filters(suspect, lane) for lane in parent_lanes)


def test_same_result_artifact_alias_consumes_one_lane_slot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = _task()
    _write_summary(
        monkeypatch=monkeypatch,
        root=tmp_path,
        variant_id="canonical_parent",
        rows=_rows(candidate_tokens=80, selected_entries=6),
        max_recalled_entries=6,
    )
    findings = _materialize_result_artifacts(
        run_dir=tmp_path,
        gen_id=0,
        default_lane="task_candidate",
        default_family="coding_memory_inheritance_policy",
        scoring_metric_keys=["scaling_margin"],
        result_maturity_policy=task["evaluation"]["maturity_policy"],
    )
    alias = copy.deepcopy(findings[0])
    alias["id"] = "alias-id"
    alias["variant_name"] = "alias_name"
    store = FrontierStore(
        tmp_path / "frontier-alias-regression",
        primary_metric=task["evaluation"]["primary_metric"],
        metric_direction=task["evaluation"]["direction"],
        frontier_lanes=task["evaluation"]["frontier_lanes"],
        maturity_policy=task["evaluation"]["maturity_policy"],
    )
    picks = store._select_lane_frontier([findings[0], alias])
    assert len(picks) == 1
