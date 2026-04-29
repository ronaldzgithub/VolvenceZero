"""Rot-guard test for ``examples/coding_end_to_end_demo.py``.

The demo is an application-layer example, deliberately placed
OUTSIDE ``packages/``. This test imports it via a file-path
``spec_from_file_location`` handshake so we don't need to add the
``examples/`` directory to sys.path (or turn it into a package).

Assertions:

* The demo runs to completion without raising.
* ``audit.turns_run`` matches the scripted turn count.
* At least one affordance invocation SUCCEEDED and one was
  DENIED_BY_BOUNDARY (the no-confirmation write_file path
  proves the safety gate is live).
* At least one mid-reflection thinking artifact was collected
  (proves the thinking adapter wiring reached the session
  lifecycle hooks).

Plus a discipline guard: the demo file itself does NOT import
from ``volvence_zero.*`` \u2014 the application/kernel boundary is
checked by grep so future edits can't silently leak kernel
internals into the example.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest


# ---------------------------------------------------------------------------
# Path to the demo file
# ---------------------------------------------------------------------------


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DEMO_PATH = _REPO_ROOT / "examples" / "coding_end_to_end_demo.py"


def _load_demo_module():
    """Load the demo script as a module without mutating sys.path."""
    spec = importlib.util.spec_from_file_location(
        "coding_end_to_end_demo", _DEMO_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec so the module's own
    # relative-ish references (e.g. dataclass metaclass registration)
    # land consistently.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Integration test: demo runs end-to-end
# ---------------------------------------------------------------------------


async def test_coding_end_to_end_demo_runs_cleanly() -> None:
    demo = _load_demo_module()
    audit = await demo.main()
    # Scripted turn count.
    assert audit.turns_run == 5, (
        f"expected 5 turns scripted in the demo, got {audit.turns_run}"
    )
    # Multiple invocations, including the demo's deliberate
    # no-confirmation DENIED path.
    assert audit.affordance_invocations >= 4
    assert audit.succeeded_invocations >= 2, (
        f"expected >= 2 SUCCEEDED invocations (read_file + run_test + "
        f"write_file with confirmation); got {audit.succeeded_invocations}"
    )
    assert audit.denied_invocations >= 1, (
        "expected >= 1 DENIED invocation (the deliberately-unconfirmed "
        "write_file); safety gate regression if zero"
    )
    # Thinking adapter must have published at least one world-lane or
    # self-lane mid-reflection artifact.
    assert audit.thinking_artifacts >= 1, (
        "expected >= 1 thinking artifact collected; thinking adapter "
        "wiring is broken if zero"
    )


# ---------------------------------------------------------------------------
# Discipline guard: application/kernel boundary
# ---------------------------------------------------------------------------


def test_coding_demo_does_not_import_kernel_directly() -> None:
    """The demo is the canonical ``lifeform-*``-only example.

    A regression here would mean someone reached into
    ``volvence_zero.*`` (kernel) from application code, which
    violates the wheel-boundary rule from SPLIT.md. We grep the
    raw file contents so the check survives whatever import-time
    tricks a future edit might introduce.
    """
    text = _DEMO_PATH.read_text(encoding="utf-8")
    # Allow the string ``volvence_zero`` to appear in comments /
    # docstrings (it shows up as contextual references in prose).
    # The hard rule is that no line is a real ``import`` or
    # ``from ... import ...`` statement touching the kernel.
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "volvence_zero" in stripped and (
            stripped.startswith("import ") or stripped.startswith("from ")
        ):
            pytest.fail(
                f"examples/coding_end_to_end_demo.py must not import "
                f"from volvence_zero (kernel). Offending line: {line!r}"
            )


def test_coding_demo_imports_only_lifeform_wheels() -> None:
    """Positive companion to the negative guard above: list the
    top-level imports and assert each is in the allowed lifeform-
    wheel set (plus stdlib).
    """
    text = _DEMO_PATH.read_text(encoding="utf-8")
    allowed_third_party_prefixes = (
        "lifeform_core",
        "lifeform_domain_coding",
        "lifeform_affordance",
        "lifeform_thinking",
        "lifeform_ingestion",
        "lifeform_expression",
    )
    # Stdlib modules we allow; extend if the demo genuinely needs more.
    allowed_stdlib_prefixes = (
        "asyncio",
        "pathlib",
        "sys",
        "tempfile",
        "dataclasses",
        "typing",
        "__future__",
    )
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("from ") and " import " in stripped:
            module = stripped.split(" ", 2)[1]
        elif stripped.startswith("import "):
            module = stripped.split(" ", 2)[1].split(",")[0].strip()
        else:
            continue
        top = module.split(".")[0]
        if top in allowed_stdlib_prefixes:
            continue
        if any(module.startswith(p) for p in allowed_third_party_prefixes):
            continue
        pytest.fail(
            f"examples/coding_end_to_end_demo.py imports disallowed "
            f"module {module!r}. Allowed lifeform wheels: "
            f"{allowed_third_party_prefixes!r}; allowed stdlib: "
            f"{allowed_stdlib_prefixes!r}"
        )
