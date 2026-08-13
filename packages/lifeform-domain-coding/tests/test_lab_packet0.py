"""Packet 0 tests: generator determinism, oracle teeth, episode plumbing.

These are the machinery guarantees the Packet 0 verdict relies on:

* same spec => identical tree, different seed => different tree;
* every task category's reference solution passes the oracle
  (solvability — templates cannot rot silently);
* invariant sabotage passes acceptance but fails regression (the
  golden failure mode memory is supposed to prevent);
* acceptance sabotage fails acceptance;
* test tampering in the hand's workspace cannot influence the oracle;
* the scripted episode loop runs end-to-end, merges on pass, cleans
  worktrees, and hashes trajectories;
* held-out sealing reproduces hashes and leaves no tree behind.
"""

from __future__ import annotations

import pathlib

import pytest

from lifeform_domain_coding.lab.calibration import (
    CalibrationConfig,
    check_environment_determinism,
    run_calibration,
)
from lifeform_domain_coding.lab.generation import (
    EnvSpec,
    generate_environment,
)
from lifeform_domain_coding.lab.heldout import seal_heldout_variants, verify_sealed_variant
from lifeform_domain_coding.lab.oracle import evaluate_episode
from lifeform_domain_coding.lab.tasks import (
    CATEGORY_ADD_HELPER,
    CATEGORY_CONFIG_FEATURE,
    CATEGORY_EXTEND_REPORT,
    CATEGORY_FIX_BUG,
    CATEGORY_REFACTOR_ALIAS,
    generate_task_chain,
    representative_tasks,
)
from lifeform_domain_coding.lab.workspace import ChainWorkspace, apply_edit

_SPEC = EnvSpec(env_seed=424242)


def _materialize(tmp_path: pathlib.Path, name: str = "env") -> pathlib.Path:
    root = tmp_path / name
    generate_environment(_SPEC, root)
    return root


def test_generation_is_deterministic(tmp_path: pathlib.Path) -> None:
    first = generate_environment(_SPEC, tmp_path / "a")
    second = generate_environment(_SPEC, tmp_path / "b")
    other = generate_environment(EnvSpec(env_seed=424243), tmp_path / "c")
    assert first.tree_hash == second.tree_hash
    assert first.tree_hash != other.tree_hash
    chain_a = generate_task_chain(_SPEC, chain_seed=0, length=6)
    chain_b = generate_task_chain(_SPEC, chain_seed=0, length=6)
    assert [task.task_id for task in chain_a] == [task.task_id for task in chain_b]


def test_pristine_environment_passes_full_suite(tmp_path: pathlib.Path) -> None:
    root = _materialize(tmp_path)
    task = representative_tasks(_SPEC)[0]
    for edit in task.reference_edits:
        apply_edit(root, edit)
    outcome = evaluate_episode(spec=_SPEC, task=task, workspace_root=root)
    assert outcome.passed, (outcome.failed_test_ids, outcome.error_test_ids)


@pytest.mark.parametrize("task_index", range(5))
def test_reference_solutions_pass(tmp_path: pathlib.Path, task_index: int) -> None:
    task = representative_tasks(_SPEC)[task_index]
    root = _materialize(tmp_path)
    for edit in task.prestate_edits:
        apply_edit(root, edit)
    for edit in task.reference_edits:
        apply_edit(root, edit)
    outcome = evaluate_episode(spec=_SPEC, task=task, workspace_root=root)
    assert outcome.passed, (
        task.task_id,
        outcome.failed_test_ids,
        outcome.error_test_ids,
    )


@pytest.mark.parametrize(
    "task_index, expected_category",
    [(1, CATEGORY_EXTEND_REPORT), (2, CATEGORY_CONFIG_FEATURE), (4, CATEGORY_REFACTOR_ALIAS)],
)
def test_invariant_sabotage_is_the_golden_failure_mode(
    tmp_path: pathlib.Path, task_index: int, expected_category: str
) -> None:
    """Invariant sabotage passes acceptance but fails the full suite."""

    task = representative_tasks(_SPEC)[task_index]
    assert task.category == expected_category
    assert task.invariant_sabotage_edits
    root = _materialize(tmp_path)
    for edit in task.invariant_sabotage_edits:
        apply_edit(root, edit)
    outcome = evaluate_episode(spec=_SPEC, task=task, workspace_root=root)
    assert not outcome.passed
    assert not outcome.regression_passed
    assert outcome.invariant_violations, outcome.failed_test_ids
    assert set(outcome.invariant_violations) & set(task.invariant_risk) or task.category == (
        CATEGORY_REFACTOR_ALIAS
    )
    if task.category in (CATEGORY_EXTEND_REPORT, CATEGORY_CONFIG_FEATURE):
        assert outcome.acceptance_passed, outcome.failed_test_ids


@pytest.mark.parametrize("task_index", [0, 3])
def test_acceptance_sabotage_fails_acceptance(tmp_path: pathlib.Path, task_index: int) -> None:
    task = representative_tasks(_SPEC)[task_index]
    assert task.category in (CATEGORY_ADD_HELPER, CATEGORY_FIX_BUG)
    root = _materialize(tmp_path)
    for edit in task.prestate_edits:
        apply_edit(root, edit)
    for edit in task.acceptance_sabotage_edits:
        apply_edit(root, edit)
    outcome = evaluate_episode(spec=_SPEC, task=task, workspace_root=root)
    assert not outcome.acceptance_passed
    assert not outcome.passed


def test_unfixed_bug_fails_regression(tmp_path: pathlib.Path) -> None:
    task = representative_tasks(_SPEC)[3]
    assert task.category == CATEGORY_FIX_BUG
    root = _materialize(tmp_path)
    for edit in task.prestate_edits:
        apply_edit(root, edit)
    outcome = evaluate_episode(spec=_SPEC, task=task, workspace_root=root)
    assert not outcome.passed
    assert not outcome.acceptance_passed


def test_oracle_ignores_tampered_tests(tmp_path: pathlib.Path) -> None:
    """Rewriting the workspace test tree cannot influence the verdict."""

    task = representative_tasks(_SPEC)[2]
    root = _materialize(tmp_path)
    for edit in task.invariant_sabotage_edits:
        apply_edit(root, edit)
    invariants_path = root / "tests" / "full" / "test_invariants.py"
    invariants_path.write_text("def test_everything_is_fine():\n    assert True\n", encoding="utf-8")
    outcome = evaluate_episode(spec=_SPEC, task=task, workspace_root=root)
    assert not outcome.regression_passed
    assert outcome.invariant_violations


async def test_scripted_calibration_end_to_end(tmp_path: pathlib.Path) -> None:
    config = CalibrationConfig(
        run_id="test-run",
        output_root=tmp_path / "artifacts",
        chains=1,
        episodes_per_chain=3,
        scripted_invariant_sabotage_rate=0.0,
        scripted_acceptance_sabotage_rate=0.0,
        heldout_variants=1,
        min_free_disk_bytes=1,
    )
    report = await run_calibration(config)
    episodes = report["episodes"]
    assert len(episodes) == 3
    assert all(row["passed"] for row in episodes), episodes
    assert all(len(row["trajectory_sha256"]) == 64 for row in episodes)
    assert not any(row["tests_tampered"] for row in episodes)
    run_dir = tmp_path / "artifacts" / "test-run"
    assert (run_dir / "report.json").is_file()
    assert (run_dir / "report.md").is_file()
    worktrees = run_dir / "chains" / "chain-00" / "worktrees"
    assert not any(worktrees.iterdir()), "episode worktrees must be cleaned up"
    assert (run_dir / "heldout" / "sealed_variants.json").is_file()
    heldout_trees = list(run_dir.rglob("hv0_app"))
    assert not heldout_trees, "held-out trees must never land in the run dir"


async def test_scripted_sabotage_modes_produce_failures(tmp_path: pathlib.Path) -> None:
    config = CalibrationConfig(
        run_id="test-run-sabotage",
        output_root=tmp_path / "artifacts",
        chains=1,
        episodes_per_chain=6,
        scripted_invariant_sabotage_rate=0.5,
        scripted_acceptance_sabotage_rate=0.5,
        heldout_variants=1,
        min_free_disk_bytes=1,
    )
    report = await run_calibration(config)
    episodes = report["episodes"]
    assert len(episodes) == 6
    assert not any(row["passed"] for row in episodes), "all-sabotage hand must fail every episode"
    assert report["oracle_band"]["pass_rate"] == 0.0


def test_environment_determinism_check() -> None:
    config = CalibrationConfig(
        run_id="unused",
        output_root=pathlib.Path("/tmp/unused"),
        chains=1,
        episodes_per_chain=3,
    )
    result = check_environment_determinism(config)
    assert result["environment_deterministic"] is True


def test_heldout_sealing_round_trip(tmp_path: pathlib.Path) -> None:
    manifest_path = tmp_path / "sealed.json"
    sealed = seal_heldout_variants(base_spec=_SPEC, count=2, manifest_path=manifest_path)
    assert manifest_path.is_file()
    assert len(sealed) == 2
    assert sealed[0].tree_sha256 != sealed[1].tree_sha256
    assert verify_sealed_variant(sealed[0])
    generated_trees = list(tmp_path.rglob("hv0_app"))
    assert not generated_trees


def test_chain_workspace_merge_and_reset(tmp_path: pathlib.Path) -> None:
    workspace = ChainWorkspace(spec=_SPEC, chain_root=tmp_path / "chain")
    environment = workspace.initialize()
    task = representative_tasks(_SPEC)[0]
    handle = workspace.begin_episode(0, task)
    for edit in task.reference_edits:
        apply_edit(handle.worktree, edit)
    _bytes, merged_hash = workspace.finalize_episode(handle, passed=True, task=task)
    assert merged_hash != environment.tree_hash, "merge must change the canonical tree"
    handle_two = workspace.begin_episode(1, task)
    util_text = (handle_two.worktree / _SPEC.package_name / "util.py").read_text(encoding="utf-8")
    assert "def clamp(" in util_text, "episode 1 must start from the merged state"
    _bytes2, hash_after_fail = workspace.finalize_episode(handle_two, passed=False, task=task)
    assert hash_after_fail == merged_hash, "failed episodes must not change the canonical tree"
