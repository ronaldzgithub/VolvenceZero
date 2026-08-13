"""Episode outcome oracle (coding-lab Packet 0).

The oracle settles an episode from a *fresh evaluation workspace*:

* the agent's package tree is copied in;
* the pristine test tree is REGENERATED from the environment spec, so
  any test edits the hand made in its workspace are discarded
  (test-tampering cannot influence the verdict);
* the task's hidden acceptance tests are injected;
* one pytest run produces machine-readable results (junitxml).

The outcome is a frozen dataclass; episode success requires both the
acceptance tests and the full regression suite to pass.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass

from lifeform_domain_coding.lab.generation import EnvSpec, latent_invariants, write_pristine_tests
from lifeform_domain_coding.lab.tasks import ChainTask

_ACCEPTANCE_TEST_RELPATH = "tests/acceptance/test_task_acceptance.py"
_ORACLE_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True)
class OracleOutcome:
    """Settled outcome of one episode."""

    task_id: str
    passed: bool
    acceptance_passed: bool
    regression_passed: bool
    failed_test_ids: tuple[str, ...]
    error_test_ids: tuple[str, ...]
    invariant_violations: tuple[str, ...]
    tests_collected: int
    duration_seconds: float
    pytest_exit_code: int


def _junit_failed_ids(junit_path: pathlib.Path) -> tuple[tuple[str, ...], tuple[str, ...], int]:
    """Parse junitxml into (failed ids, errored ids, collected count).

    Ids are normalised to ``relative/path.py::test_name`` so they can be
    matched against the invariant registry.
    """

    tree = ElementTree.parse(junit_path)
    failed: list[str] = []
    errored: list[str] = []
    collected = 0
    for testcase in tree.iter("testcase"):
        collected += 1
        classname = testcase.attrib.get("classname", "")
        name = testcase.attrib.get("name", "")
        node_id = f"{classname.replace('.', '/')}.py::{name}"
        if testcase.find("failure") is not None:
            failed.append(node_id)
        elif testcase.find("error") is not None:
            errored.append(node_id)
    return tuple(failed), tuple(errored), collected


def _map_invariant_violations(
    spec: EnvSpec, failed_or_errored: tuple[str, ...]
) -> tuple[str, ...]:
    violations: list[str] = []
    for invariant in latent_invariants(spec):
        expected_ids = {
            f"{relpath.removesuffix('.py')}.py::{test_name}".replace("\\", "/")
            for relpath, test_name in invariant.regression_tests
        }
        if expected_ids & set(failed_or_errored):
            violations.append(invariant.invariant_id)
    return tuple(violations)


def evaluate_episode(
    *,
    spec: EnvSpec,
    task: ChainTask,
    workspace_root: pathlib.Path,
    python_executable: str | None = None,
) -> OracleOutcome:
    """Settle ``task`` against the agent's ``workspace_root``.

    Only ``<package_name>/`` is taken from the workspace; tests are
    regenerated pristine and the acceptance file is injected fresh.
    """

    python_bin = python_executable or sys.executable
    package_dir = workspace_root / spec.package_name
    if not package_dir.is_dir():
        raise FileNotFoundError(
            f"workspace {workspace_root!s} does not contain package dir {spec.package_name!r}"
        )
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="coding-lab-oracle-") as eval_root_str:
        eval_root = pathlib.Path(eval_root_str)
        shutil.copytree(
            package_dir,
            eval_root / spec.package_name,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        write_pristine_tests(spec, eval_root)
        acceptance_path = eval_root / _ACCEPTANCE_TEST_RELPATH
        acceptance_path.parent.mkdir(parents=True, exist_ok=True)
        acceptance_path.write_text(task.acceptance_test_source, encoding="utf-8")
        junit_path = eval_root / "oracle-junit.xml"
        completed = subprocess.run(
            [
                python_bin,
                "-m",
                "pytest",
                "-q",
                "--tb=no",
                "-p",
                "no:cacheprovider",
                f"--junitxml={junit_path}",
                "tests",
            ],
            cwd=eval_root,
            capture_output=True,
            text=True,
            timeout=_ORACLE_TIMEOUT_SECONDS,
        )
        if not junit_path.is_file():
            raise RuntimeError(
                "oracle pytest produced no junitxml "
                f"(exit={completed.returncode}); stderr tail: {completed.stderr[-800:]!r}"
            )
        failed, errored, collected = _junit_failed_ids(junit_path)
    duration = time.monotonic() - started
    failed_or_errored = tuple(failed) + tuple(errored)
    acceptance_prefix = _ACCEPTANCE_TEST_RELPATH.removesuffix(".py")
    acceptance_failures = tuple(
        node_id for node_id in failed_or_errored if node_id.startswith(acceptance_prefix)
    )
    regression_failures = tuple(
        node_id for node_id in failed_or_errored if not node_id.startswith(acceptance_prefix)
    )
    if collected == 0:
        # A collection catastrophe (syntax error in the package) surfaces
        # as zero collected tests; that is an episode failure, not an
        # infrastructure error.
        return OracleOutcome(
            task_id=task.task_id,
            passed=False,
            acceptance_passed=False,
            regression_passed=False,
            failed_test_ids=(),
            error_test_ids=("<collection-failed>",),
            invariant_violations=(),
            tests_collected=0,
            duration_seconds=duration,
            pytest_exit_code=completed.returncode,
        )
    acceptance_passed = not acceptance_failures
    regression_passed = not regression_failures
    return OracleOutcome(
        task_id=task.task_id,
        passed=acceptance_passed and regression_passed,
        acceptance_passed=acceptance_passed,
        regression_passed=regression_passed,
        failed_test_ids=failed,
        error_test_ids=errored,
        invariant_violations=_map_invariant_violations(spec, regression_failures),
        tests_collected=collected,
        duration_seconds=duration,
        pytest_exit_code=completed.returncode,
    )


__all__ = ["OracleOutcome", "evaluate_episode"]
