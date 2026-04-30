"""Rot-guard test for ``examples/companionship_end_to_end_demo.py``.

Mirrors the discipline of ``test_coding_end_to_end_demo.py``:

* The demo runs to completion without raising.
* Audit counters match the scripted shape (5 natural turns +
  some ingestion turns).
* Discipline guards: no ``volvence_zero.*`` (kernel) imports;
  only ``lifeform-*`` wheels + a small stdlib whitelist.

This test deliberately does NOT assert specific kernel behaviour
(regime transitions, trust direction, rapport changes). The demo
is a *diagnostic instrument* \u2014 its value comes from the printed
trajectory exposing what the kernel actually does, including
weaknesses. Asserting the desired emotional-arc behaviour here
would lock in the current (acknowledged-imperfect) classifier
output and turn the rot-guard into a regression-blocker for
future companion-vertical work. We only check that the run
mechanics work and the import discipline holds.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DEMO_PATH = _REPO_ROOT / "examples" / "companionship_end_to_end_demo.py"


def _load_demo_module():
    spec = importlib.util.spec_from_file_location(
        "companionship_end_to_end_demo", _DEMO_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


async def test_companionship_end_to_end_demo_runs_cleanly() -> None:
    demo = _load_demo_module()
    audit = await demo.main()
    # 5 natural turns scripted (first contact / disclosure / guided /
    # rupture / repair).
    assert audit.natural_turns == 5
    # Background memo splits into >= 1 ingestion chunk + turn.
    assert audit.ingestion_turns >= 1
    # Trajectory snapshots: after-ingest, 5 turns, after-close = 7
    # rows minimum.
    assert len(audit.trajectory) >= 7
    # Every trajectory row has a non-empty regime tag (so the demo
    # is actually reading session.latest_active_snapshots).
    for record in audit.trajectory:
        assert record.regime, (
            f"trajectory row {record.label!r} has empty regime tag; "
            f"snapshot reading broken"
        )


# ---------------------------------------------------------------------------
# Discipline: kernel-import boundary
# ---------------------------------------------------------------------------


def test_companionship_demo_does_not_import_kernel_directly() -> None:
    text = _DEMO_PATH.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "volvence_zero" in stripped and (
            stripped.startswith("import ") or stripped.startswith("from ")
        ):
            pytest.fail(
                f"examples/companionship_end_to_end_demo.py must not import "
                f"from volvence_zero (kernel). Offending line: {line!r}"
            )


def test_companionship_demo_imports_only_lifeform_wheels() -> None:
    text = _DEMO_PATH.read_text(encoding="utf-8")
    allowed_third_party_prefixes = (
        "lifeform_core",
        "lifeform_domain_emogpt",
        "lifeform_affordance",
        "lifeform_thinking",
        "lifeform_ingestion",
        "lifeform_expression",
    )
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
            f"examples/companionship_end_to_end_demo.py imports disallowed "
            f"module {module!r}. Allowed lifeform wheels: "
            f"{allowed_third_party_prefixes!r}; allowed stdlib: "
            f"{allowed_stdlib_prefixes!r}"
        )
