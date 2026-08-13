"""Chain workspace management (coding-lab Packet 0).

One :class:`ChainWorkspace` owns an inner git repository holding the
canonical evolving state of a task chain. Each episode runs in a fresh
``git worktree`` of the current HEAD:

* pre-state edits (bug injection) are applied to the worktree only;
* the hand works inside the worktree through the sandboxed affordance
  backends;
* on a PASS verdict, the package tree (and only the package tree — the
  hand's test edits never merge) is copied back and committed, becoming
  the next episode's base;
* the worktree is removed afterwards (disk discipline; bytes are
  accounted before removal).

All git identity/date variables are pinned so a chain replay produces
identical trees. Tree equality is checked with our own file hashing,
never git object ids.
"""

from __future__ import annotations

import ast
import pathlib
import shutil
import subprocess
from dataclasses import dataclass

from lifeform_domain_coding.lab.generation import (
    EnvSpec,
    GeneratedEnvironment,
    compute_tree_hash,
    generate_environment,
)
from lifeform_domain_coding.lab.tasks import (
    ChainTask,
    FunctionReplace,
    PrestateEdit,
)

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "coding-lab",
    "GIT_AUTHOR_EMAIL": "coding-lab@volvence.local",
    "GIT_COMMITTER_NAME": "coding-lab",
    "GIT_COMMITTER_EMAIL": "coding-lab@volvence.local",
    "GIT_AUTHOR_DATE": "2026-01-01T00:00:00 +0000",
    "GIT_COMMITTER_DATE": "2026-01-01T00:00:00 +0000",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "HOME": "/tmp",
}


def directory_bytes(root: pathlib.Path) -> int:
    """Total size of regular files under ``root`` (bytes accounting)."""

    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def apply_edit(workspace: pathlib.Path, edit: PrestateEdit) -> None:
    """Apply one prestate edit inside ``workspace`` (fail loudly).

    ``FileEdit`` is an exact-string anchor; ``FunctionReplace`` is an
    AST-anchored whole-function reset that survives hand-authored drift
    in the evolving chain repository (see its docstring in ``tasks``).
    """

    if isinstance(edit, FunctionReplace):
        _apply_function_replace(workspace, edit)
        return
    target = workspace / edit.path
    if not edit.old:
        if not target.parent.is_dir():
            raise FileNotFoundError(f"edit target parent missing: {target.parent!s}")
        existing = target.read_text(encoding="utf-8") if target.is_file() else ""
        target.write_text(existing + edit.new, encoding="utf-8")
        return
    content = target.read_text(encoding="utf-8")
    occurrences = content.count(edit.old)
    if occurrences != 1:
        raise ValueError(
            f"edit anchor must occur exactly once in {edit.path!r}, found {occurrences}: {edit.old[:80]!r}"
        )
    target.write_text(content.replace(edit.old, edit.new, 1), encoding="utf-8")


def _apply_function_replace(workspace: pathlib.Path, edit: FunctionReplace) -> None:
    target = workspace / edit.path
    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source)
    candidates = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == edit.function_name
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"function {edit.function_name!r} must occur exactly once at top "
            f"level of {edit.path!r}, found {len(candidates)}"
        )
    node = candidates[0]
    start = (node.decorator_list[0].lineno if node.decorator_list else node.lineno) - 1
    end = node.end_lineno
    assert end is not None  # ast guarantees end_lineno on parsed sources
    lines = source.splitlines(keepends=True)
    replacement = edit.replacement_source
    if not replacement.endswith("\n"):
        replacement += "\n"
    target.write_text(
        "".join(lines[:start]) + replacement + "".join(lines[end:]),
        encoding="utf-8",
    )


@dataclass(frozen=True)
class EpisodeWorkspaceHandle:
    """Live worktree for one episode."""

    episode_index: int
    worktree: pathlib.Path


class ChainWorkspace:
    """Owns the canonical evolving repo state for one task chain."""

    def __init__(self, *, spec: EnvSpec, chain_root: pathlib.Path) -> None:
        self._spec = spec
        self._chain_root = pathlib.Path(chain_root)
        self._repo_dir = self._chain_root / "repo"
        self._worktrees_dir = self._chain_root / "worktrees"
        self._environment: GeneratedEnvironment | None = None

    @property
    def spec(self) -> EnvSpec:
        return self._spec

    @property
    def repo_dir(self) -> pathlib.Path:
        return self._repo_dir

    @property
    def environment(self) -> GeneratedEnvironment:
        if self._environment is None:
            raise RuntimeError("ChainWorkspace.initialize() has not been called")
        return self._environment

    def _git(self, *args: str, cwd: pathlib.Path | None = None) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd if cwd is not None else self._repo_dir),
            env=_GIT_ENV,
            capture_output=True,
            text=True,
            timeout=60.0,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(args)} failed (exit={completed.returncode}): {completed.stderr.strip()!r}"
            )
        return completed.stdout

    def initialize(self) -> GeneratedEnvironment:
        """Generate the environment and commit it as the chain base."""

        self._repo_dir.mkdir(parents=True, exist_ok=True)
        self._worktrees_dir.mkdir(parents=True, exist_ok=True)
        environment = generate_environment(self._spec, self._repo_dir)
        self._git("init", "--quiet", "--initial-branch=main")
        self._git("add", "-A")
        self._git("commit", "--quiet", "-m", "coding-lab: environment base")
        self._environment = environment
        return environment

    def begin_episode(self, episode_index: int, task: ChainTask) -> EpisodeWorkspaceHandle:
        """Create a fresh worktree at HEAD and apply pre-state edits."""

        worktree = self._worktrees_dir / f"episode-{episode_index:03d}"
        if worktree.exists():
            raise FileExistsError(f"worktree already exists: {worktree!s}")
        self._git("worktree", "add", "--detach", "--quiet", str(worktree), "HEAD")
        for edit in task.prestate_edits:
            apply_edit(worktree, edit)
        return EpisodeWorkspaceHandle(episode_index=episode_index, worktree=worktree)

    def tests_tampered(self, handle: EpisodeWorkspaceHandle) -> bool:
        """True when the hand modified the test tree in its worktree."""

        status = self._git("status", "--porcelain", "--", "tests", "conftest.py", cwd=handle.worktree)
        return bool(status.strip())

    def finalize_episode(
        self,
        handle: EpisodeWorkspaceHandle,
        *,
        passed: bool,
        task: ChainTask,
    ) -> tuple[int, str]:
        """Account bytes, merge the package tree on PASS, drop the worktree.

        Returns ``(workspace_bytes, post_merge_tree_hash)``.
        """

        workspace_bytes = directory_bytes(handle.worktree)
        if passed:
            source_pkg = handle.worktree / self._spec.package_name
            target_pkg = self._repo_dir / self._spec.package_name
            if not source_pkg.is_dir():
                raise FileNotFoundError(f"episode worktree lost its package dir: {source_pkg!s}")
            shutil.rmtree(target_pkg)
            shutil.copytree(source_pkg, target_pkg, ignore=shutil.ignore_patterns("__pycache__"))
            self._git("add", "-A", "--", self._spec.package_name)
            self._git(
                "commit",
                "--quiet",
                "--allow-empty",
                "-m",
                f"coding-lab: merge episode {handle.episode_index:03d} ({task.task_id})",
            )
        self._git("worktree", "remove", "--force", str(handle.worktree))
        tree_hash, _ = compute_tree_hash(self._repo_dir)
        return workspace_bytes, tree_hash


__all__ = [
    "ChainWorkspace",
    "EpisodeWorkspaceHandle",
    "apply_edit",
    "directory_bytes",
]
