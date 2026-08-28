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

import hashlib
import pathlib

import pytest

from lifeform_domain_coding.lab.calibration import (
    CalibrationConfig,
    check_environment_determinism,
    run_calibration,
)
from lifeform_domain_coding.lab.generation import (
    ALL_CONVENTION_IDS,
    CONVENTION_ANNOTATED_SIGNATURE,
    CONVENTION_DOCSTRING_CONTRACT,
    CONVENTION_EXPORT_ALL,
    CONVENTION_SYMBOL_OWNER,
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


_CONVENTION_SPEC = EnvSpec(env_seed=424242, convention_ids=(CONVENTION_EXPORT_ALL,))


@pytest.mark.parametrize("task_index", range(5))
def test_convention_reference_solutions_pass(tmp_path: pathlib.Path, task_index: int) -> None:
    """With the difficulty knob on, references stay solvable (compliance edits)."""

    task = representative_tasks(_CONVENTION_SPEC)[task_index]
    root = tmp_path / "env"
    generate_environment(_CONVENTION_SPEC, root)
    for edit in task.prestate_edits:
        apply_edit(root, edit)
    for edit in task.reference_edits:
        apply_edit(root, edit)
    outcome = evaluate_episode(spec=_CONVENTION_SPEC, task=task, workspace_root=root)
    assert outcome.passed, (
        task.task_id,
        outcome.failed_test_ids,
        outcome.error_test_ids,
    )
    assert outcome.invariant_violations == ()


def test_convention_violation_fails_and_is_attributed(tmp_path: pathlib.Path) -> None:
    """A behaviourally-correct solution that skips ``__all__`` fails the
    hidden house test, and the violation is attributed by id — the
    memory-realisable failure mode the difficulty knob exists for."""

    task = representative_tasks(_CONVENTION_SPEC)[0]  # add_helper
    root = tmp_path / "env"
    generate_environment(_CONVENTION_SPEC, root)
    non_compliant = [
        edit
        for edit in task.reference_edits
        if "__all__" not in edit.new
    ]
    assert len(non_compliant) == len(task.reference_edits) - 1
    for edit in non_compliant:
        apply_edit(root, edit)
    outcome = evaluate_episode(spec=_CONVENTION_SPEC, task=task, workspace_root=root)
    assert not outcome.passed
    assert not outcome.acceptance_passed
    assert outcome.regression_passed
    assert CONVENTION_EXPORT_ALL in outcome.invariant_violations
    # Post-submit CI evidence must be actionable: the assertion head
    # names __all__ so a remembering hand can fix the next episode.
    convention_details = [
        detail for detail in outcome.failure_details if "test_house_" in detail
    ]
    assert convention_details, outcome.failure_details
    assert any("__all__" in detail for detail in convention_details)


def test_convention_invariant_sabotage_still_passes_acceptance(tmp_path: pathlib.Path) -> None:
    """Invariant sabotage must keep its defining property (passes
    acceptance, fails regression) with the difficulty knob on."""

    task = representative_tasks(_CONVENTION_SPEC)[1]  # extend_report
    root = tmp_path / "env"
    generate_environment(_CONVENTION_SPEC, root)
    for edit in task.invariant_sabotage_edits:
        apply_edit(root, edit)
    outcome = evaluate_episode(spec=_CONVENTION_SPEC, task=task, workspace_root=root)
    assert outcome.acceptance_passed, outcome.failed_test_ids
    assert not outcome.regression_passed
    assert CONVENTION_EXPORT_ALL not in outcome.invariant_violations


def test_convention_off_by_default_keeps_legacy_tasks(tmp_path: pathlib.Path) -> None:
    task_default = representative_tasks(_SPEC)[0]
    assert "test_house_" not in task_default.acceptance_test_source
    assert all("__all__" not in edit.new for edit in task_default.reference_edits)


# --- Additional house conventions (2026-08-27) ------------------------------

#: Representative-task indices whose category introduces a new public symbol,
#: i.e. the only tasks any house convention can be enforced on.
_SYMBOL_TASK_INDICES = (0, 1, 2)  # add_helper, extend_report, config_feature
_SYMBOL_FREE_TASK_INDICES = (3, 4)  # fix_bug, refactor_alias

_NEW_CONVENTIONS = (
    CONVENTION_ANNOTATED_SIGNATURE,
    CONVENTION_DOCSTRING_CONTRACT,
    CONVENTION_SYMBOL_OWNER,
)


def _convention_payload(convention_ids: tuple[str, ...]) -> str:
    """Canonical text of everything a spec's conventions inject into tasks."""

    spec = EnvSpec(env_seed=424242, convention_ids=convention_ids)
    parts: list[str] = []
    for task in representative_tasks(spec):
        parts.extend((f"### REP {task.task_id} description", task.description))
        parts.extend((f"### REP {task.task_id} acceptance", task.acceptance_test_source))
        for label, group in (
            ("ref", task.reference_edits),
            ("accsab", task.acceptance_sabotage_edits),
            ("invsab", task.invariant_sabotage_edits),
        ):
            for index, edit in enumerate(group):
                parts.append(f"### REP {task.task_id} {label}[{index}] {edit.path}")
                parts.extend((f"OLD>{edit.old}", f"NEW>{edit.new}"))
    for task in generate_task_chain(spec, chain_seed=0, length=10):
        parts.extend((f"### CHAIN {task.task_id} acceptance", task.acceptance_test_source))
        for index, edit in enumerate(task.reference_edits):
            parts.append(f"### CHAIN {task.task_id} ref[{index}] {edit.path}")
            parts.extend((f"OLD>{edit.old}", f"NEW>{edit.new}"))
    return "\u0000".join(parts)


def test_export_all_configuration_is_byte_frozen() -> None:
    """Adding conventions must not perturb ``convention_export_all``.

    The 2026-08-13 Packet 2 formal chains were generated with this single
    convention active and are replayed from it, so any drift here silently
    invalidates frozen evidence. Verified byte-identical against the pre-change
    tree when the three 2026-08-27 conventions were introduced.
    """

    digest = hashlib.sha256(
        _convention_payload((CONVENTION_EXPORT_ALL,)).encode("utf-8")
    ).hexdigest()
    assert digest == "60dec70c8b484fb32140f444ddd33e880949f25f21c5a7e709bce8a62a878b15"


@pytest.mark.parametrize("convention_id", _NEW_CONVENTIONS)
@pytest.mark.parametrize("task_index", _SYMBOL_TASK_INDICES)
def test_new_convention_reference_solutions_pass(
    tmp_path: pathlib.Path, convention_id: str, task_index: int
) -> None:
    """Each new convention keeps its tasks solvable via the compliance edit."""

    spec = EnvSpec(env_seed=424242, convention_ids=(convention_id,))
    task = representative_tasks(spec)[task_index]
    assert f"test_house_{convention_id}" in task.acceptance_test_source
    root = tmp_path / "env"
    generate_environment(spec, root)
    for edit in task.prestate_edits:
        apply_edit(root, edit)
    for edit in task.reference_edits:
        apply_edit(root, edit)
    outcome = evaluate_episode(spec=spec, task=task, workspace_root=root)
    assert outcome.passed, (task.task_id, outcome.failed_test_ids, outcome.failure_details)
    assert outcome.invariant_violations == ()


@pytest.mark.parametrize("convention_id", _NEW_CONVENTIONS)
def test_new_convention_violation_is_attributed(
    tmp_path: pathlib.Path, convention_id: str
) -> None:
    """A behaviourally-correct but non-compliant solution fails, by id.

    The non-compliant solution is exactly the same task's reference with all
    conventions switched off, so the only thing under test is the convention.
    """

    spec_on = EnvSpec(env_seed=424242, convention_ids=(convention_id,))
    spec_off = EnvSpec(env_seed=424242)
    task_on = representative_tasks(spec_on)[0]  # add_helper
    task_off = representative_tasks(spec_off)[0]
    assert task_on.task_id == task_off.task_id
    root = tmp_path / "env"
    generate_environment(spec_on, root)
    for edit in task_off.reference_edits:
        apply_edit(root, edit)
    outcome = evaluate_episode(spec=spec_on, task=task_on, workspace_root=root)
    assert not outcome.passed
    assert not outcome.acceptance_passed
    assert outcome.regression_passed, outcome.failed_test_ids
    assert convention_id in outcome.invariant_violations
    # The assertion head must name the rule, so a remembering hand can act on it.
    house_details = [detail for detail in outcome.failure_details if "test_house_" in detail]
    assert house_details, outcome.failure_details


def test_all_conventions_compose(tmp_path: pathlib.Path) -> None:
    """All four conventions active at once stay independently satisfiable."""

    spec = EnvSpec(env_seed=424242, convention_ids=ALL_CONVENTION_IDS)
    task = representative_tasks(spec)[0]
    for convention_id in ALL_CONVENTION_IDS:
        assert f"test_house_{convention_id}" in task.acceptance_test_source
    root = tmp_path / "env"
    generate_environment(spec, root)
    for edit in task.reference_edits:
        apply_edit(root, edit)
    outcome = evaluate_episode(spec=spec, task=task, workspace_root=root)
    assert outcome.passed, (outcome.failed_test_ids, outcome.failure_details)
    assert outcome.invariant_violations == ()


@pytest.mark.parametrize("task_index", _SYMBOL_FREE_TASK_INDICES)
def test_conventions_skip_tasks_without_a_new_public_symbol(task_index: int) -> None:
    """fix_bug / refactor_alias add no public symbol, so no rule is injected."""

    spec = EnvSpec(env_seed=424242, convention_ids=ALL_CONVENTION_IDS)
    task = representative_tasks(spec)[task_index]
    assert "test_house_" not in task.acceptance_test_source


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
