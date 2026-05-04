"""v0 gate test for the Real Open Dialogue Learning Loop (M6)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lifeform_evolution.open_dialogue_v0_gate import run_v0_gate


def test_v0_gate_passes_on_full_loop(tmp_path: Path) -> None:
    out_dir = tmp_path / "artifacts" / "open_dialogue"
    scope_root = tmp_path / "memory_scope_root"
    report = run_v0_gate(out_dir=out_dir, scope_root_dir=scope_root)

    # The gate must pass as a whole.
    assert report.passed, (
        "v0 gate failed: "
        + "; ".join(
            f"{name}={'ok' if ok else 'FAIL'} ({detail})"
            for name, ok, detail in report.gate_items
        )
    )

    # Artifacts are on disk.
    summary_path = out_dir / "v0_gate_report.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["passed"] is True

    # Treatment produced at least one durable rupture_repair entry.
    assert report.treatment.durable_rupture_repair_count >= 1
    # Matched control (no shared memory) produced none.
    assert report.matched_control.durable_rupture_repair_count == 0
    # No cross-user leakage.
    assert report.leakage_alice_viewed_from_bob == 0
    assert report.leakage_bob_viewed_from_alice == 0
    # Negative invariant: no external signal => no durable entry.
    assert report.negative_invariant_rupture_repair_count == 0


def test_v0_gate_fails_loudly_if_treatment_has_no_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulate a broken pipeline by monkeypatching the gate harness to
    skip the MISSED submit; the report must then report failure on the
    treatment gate item.
    """

    from lifeform_evolution import open_dialogue_v0_gate

    async def _broken_treatment(runner) -> None:  # type: ignore[no-untyped-def]
        # Run the same turns but NEVER submit the external outcomes.
        await runner.run_turn(
            "I have been thinking about whether to leave my job; I feel stuck."
        )
        await runner.run_turn("That felt cold.")
        runner.begin_new_context(reason="scene-end-session-a-broken")
        await runner.drain_session_post_slow_loop()

    monkeypatch.setattr(
        open_dialogue_v0_gate,
        "_run_treatment_session_a",
        _broken_treatment,
    )
    out_dir = tmp_path / "broken_artifacts"
    scope_root = tmp_path / "broken_scope"
    report = run_v0_gate(out_dir=out_dir, scope_root_dir=scope_root)
    assert report.passed is False
    treatment_item = next(
        (item for item in report.gate_items if item[0] == "treatment_produces_durable_rupture_repair"),
        None,
    )
    assert treatment_item is not None
    assert treatment_item[1] is False
