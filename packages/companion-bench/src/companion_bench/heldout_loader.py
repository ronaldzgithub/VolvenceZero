# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Held-out scenario loader.

The held-out set lives outside this wheel — in a private git submodule
mounted at ``external/companion-bench-heldout/`` (preferred) or the
legacy alias ``external/lscb-heldout/`` (kept for git-history continuity).
See RFC §3 P3, §8.6. This module knows how to find that submodule,
refuse to load if it is missing in environments that require it, and
surface a clean message otherwise.

Held-out semantics:

* Held-out scenarios MUST have ``held_out: true`` and ``public_test: false``
  in their YAML; the spec parser already enforces mutual exclusion.
* The leaderboard cites held-out scenarios only by hash — never by
  body — so the harness emits hashes alongside scores.
* When the submodule is absent (open-source clone, public CI on a PR),
  the harness logs a single line and proceeds with public-only
  evaluation. ``require_heldout=True`` callers (release tier) will
  raise ``HeldOutMissingError`` instead.
"""

from __future__ import annotations

import pathlib
import warnings

from companion_bench.spec import ScenarioSpec, load_scenarios_dir


_DEFAULT_HELDOUT_DIR_ENV = "COMPANION_BENCH_HELDOUT_DIR"

# Preferred path first; legacy alias second. The loader auto-resolves to
# whichever exists at runtime so existing checkouts keep working while new
# clones use the rebranded name.
_DEFAULT_HELDOUT_PATHS: tuple[tuple[str, ...], ...] = (
    ("external", "companion-bench-heldout", "scenarios"),
    ("external", "lscb-heldout", "scenarios"),
)


class HeldOutMissingError(RuntimeError):
    """Raised when a release-tier caller cannot locate the submodule."""


def default_heldout_dir(repo_root: pathlib.Path) -> pathlib.Path:
    """Path the harness expects the submodule to live at.

    Returns the preferred ``companion-bench-heldout`` path if it exists,
    otherwise falls back to the legacy ``lscb-heldout`` path. If neither
    exists on disk the preferred path is returned (callers that ``require``
    presence will then raise a clear error).
    """
    for parts in _DEFAULT_HELDOUT_PATHS:
        candidate = repo_root.joinpath(*parts)
        if candidate.exists():
            return candidate
    return repo_root.joinpath(*_DEFAULT_HELDOUT_PATHS[0])


def is_heldout_present(heldout_dir: pathlib.Path) -> bool:
    """Return True when the submodule directory contains at least one YAML."""
    if not heldout_dir.exists():
        return False
    for _ in heldout_dir.rglob("*.yaml"):
        return True
    return False


def load_heldout_scenarios(
    *,
    heldout_dir: pathlib.Path,
    require: bool = False,
) -> tuple[ScenarioSpec, ...]:
    """Load held-out scenarios from ``heldout_dir``.

    ``require=False`` (default): if the directory is missing or empty
    we return an empty tuple and emit a single warning. This keeps
    public CI / smoke tests from failing simply because the private
    submodule is not checked out.

    ``require=True`` (release tier): missing or empty heldout pool
    raises :class:`HeldOutMissingError`. Wired up in CI behind a tier
    flag so a PR cannot accidentally trigger a release run.
    """
    if not is_heldout_present(heldout_dir):
        if require:
            raise HeldOutMissingError(
                f"held-out submodule not found at {heldout_dir}. "
                f"Initialise it with `git submodule update --init "
                f"external/companion-bench-heldout` (or the legacy "
                f"`external/lscb-heldout` alias; requires deploy key)."
            )
        warnings.warn(
            f"held-out submodule not present at {heldout_dir}; "
            f"running against public scenarios only. This is expected "
            f"on public clones; release-tier CI must enable submodules.",
            stacklevel=2,
        )
        return ()
    specs = load_scenarios_dir(heldout_dir, include_held_out=True)
    # Defensive: every scenario in this dir must be flagged held_out.
    bad = [s.scenario_id for s in specs if not s.held_out]
    if bad:
        raise ValueError(
            f"scenarios under {heldout_dir} not marked held_out=true: {bad}; "
            f"refusing to load to prevent public-set contamination"
        )
    return specs
