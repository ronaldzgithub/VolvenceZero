"""Rot-guard for ``examples/companionship_real_substrate_demo.py``.

Two layers of coverage:

1. **Always-on (CI-safe)**: synthetic side + import discipline.
   The synthetic side runs without any model download and exercises
   the full pipeline, so a regression in the wiring shows up here
   even on a network-isolated CI runner.

2. **Opt-in (real substrate)**: gated by
   ``VZ_RUN_REAL_SUBSTRATE_DEMO=1``. When set, runs the FULL demo
   end-to-end (downloads Qwen 0.5B if not cached) and asserts the
   substrate switch produced ANY measurable effect on the regime
   classifier OR the 12-axis InterlocutorState. We don't assert
   specific axis values \u2014 the calibrator is acknowledged-imperfect
   and we don't want this test to lock in current numbers.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys

import pytest


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DEMO_PATH = _REPO_ROOT / "examples" / "companionship_real_substrate_demo.py"


def _load_demo_module():
    spec = importlib.util.spec_from_file_location(
        "companionship_real_substrate_demo", _DEMO_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Layer 1: synthetic-only (always run)
# ---------------------------------------------------------------------------


async def test_synthetic_side_runs_cleanly() -> None:
    """The synthetic half of the demo runs without raising and
    produces a trajectory of expected shape (1 ingest + 5 turns
    + 1 close = 7 rows).
    """
    demo = _load_demo_module()
    result = await demo._run_synthetic()
    assert result.is_real is False
    assert result.name == "synthetic"
    assert len(result.trajectory) == 7
    for record in result.trajectory:
        assert record.regime, (
            f"trajectory row {record.label!r} has empty regime tag; "
            f"snapshot reading broken"
        )


# ---------------------------------------------------------------------------
# Layer 2: real substrate (opt-in via env var)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("VZ_RUN_REAL_SUBSTRATE_DEMO") != "1",
    reason=(
        "Real-substrate demo requires Qwen download (~500 MB). "
        "Set VZ_RUN_REAL_SUBSTRATE_DEMO=1 to run."
    ),
)
async def test_real_substrate_produces_measurable_substrate_effect() -> None:
    """Run the demo end-to-end and assert the substrate switch
    produced SOME observable difference vs synthetic.

    We do NOT assert direction (rapport up / resistance down) or
    specific regime transitions because the calibrator is
    acknowledged-imperfect; locking in current numbers would
    block future calibrator work. The check is "the substrate
    layer matters at all" \u2014 a regression where real and
    synthetic produce identical trajectories means the substrate
    pipeline broke.
    """
    demo = _load_demo_module()
    result = await demo.main()
    assert result["real"].is_real is True, (
        f"real substrate did not load; got status="
        f"{result['real'].status_label!r}"
    )
    summary = result["summary"]
    # Either at least one regime disagreement, or at least one
    # 12-axis dimension moved by >= 0.05. Either is sufficient
    # evidence that the substrate switch reached downstream layers.
    regime_disagreement = summary["regime_phase_disagreement_count"] > 0
    axis_movement = max(
        summary["engagement"],
        summary["focus"],
        summary["rapport"],
        summary["resistance"],
        summary["openness"],
        summary["trust"],
    ) >= 0.05
    assert regime_disagreement or axis_movement, (
        f"real substrate produced an effectively-identical trajectory to "
        f"synthetic; substrate pipeline likely broke. summary={summary!r}"
    )


# ---------------------------------------------------------------------------
# Discipline guards (always run, mirror the other example tests)
# ---------------------------------------------------------------------------


def test_real_substrate_demo_does_not_import_kernel_directly() -> None:
    text = _DEMO_PATH.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "volvence_zero" in stripped and (
            stripped.startswith("import ") or stripped.startswith("from ")
        ):
            pytest.fail(
                f"examples/companionship_real_substrate_demo.py must not "
                f"import from volvence_zero (kernel). Offending line: {line!r}"
            )


def test_real_substrate_demo_imports_only_lifeform_wheels() -> None:
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
        "os",
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
            f"examples/companionship_real_substrate_demo.py imports disallowed "
            f"module {module!r}. Allowed: lifeform-* + stdlib whitelist."
        )
