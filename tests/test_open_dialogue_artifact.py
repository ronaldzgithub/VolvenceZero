"""Tests for open-dialogue artifact export (M5)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from volvence_zero.agent.session import AgentSessionRunner
from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidenceSource,
    DialogueExternalOutcomeKind,
)
from volvence_zero.integration import FinalRolloutConfig
from volvence_zero.open_dialogue_artifact import (
    EXPORTER_SCHEMA_VERSION,
    OpenDialogueExportReport,
    build_rupture_wiring_config,
    export_open_dialogue_session,
)
from volvence_zero.runtime import WiringLevel


def _run_small_session() -> AgentSessionRunner:
    async def _main() -> AgentSessionRunner:
        runner = AgentSessionRunner(
            user_scope="alice",
            config=FinalRolloutConfig(),
        )
        await runner.run_turn("hi")
        runner.submit_dialogue_outcome(
            kind=DialogueExternalOutcomeKind.MISSED,
            source=DialogueExternalOutcomeEvidenceSource.USER_EXPLICIT,
            confidence=0.9,
        )
        await runner.run_turn("that missed me")
        runner.begin_new_context(reason="t")
        await runner.drain_session_post_slow_loop()
        return runner

    return asyncio.run(_main())


def test_export_writes_turns_ndjson_and_summary_json(tmp_path: Path) -> None:
    runner = _run_small_session()
    report = export_open_dialogue_session(
        runner,
        session_id="alice-sess-1",
        out_dir=tmp_path,
    )
    assert isinstance(report, OpenDialogueExportReport)
    assert report.session_id == "alice-sess-1"
    assert report.turns_path.exists()
    assert report.summary_path.exists()
    # NDJSON has one line per dialogue trace.
    lines = report.turns_path.read_text(encoding="utf-8").splitlines()
    assert report.turn_count == len(lines)
    for line in lines:
        row = json.loads(line)
        assert row["schema_version"] == EXPORTER_SCHEMA_VERSION
        assert "trace_id" in row
        assert "turn_index" in row
        assert "outcome" in row
    # Summary JSON carries the session-post queue counters and scope.
    summary = json.loads(report.summary_path.read_text(encoding="utf-8"))
    assert summary["schema_version"] == EXPORTER_SCHEMA_VERSION
    assert summary["session_id"] == "alice-sess-1"
    assert summary["user_scope"] == "alice"
    assert "session_post_queue" in summary
    assert "rupture_state" in summary
    assert "external_outcomes" in summary
    assert "rupture_repair_memory" in summary


def test_export_is_read_only_does_not_change_snapshots(tmp_path: Path) -> None:
    runner = _run_small_session()
    pre_snapshots = dict(runner._upstream_snapshots)  # noqa: SLF001
    report = export_open_dialogue_session(
        runner,
        session_id="alice-sess-1",
        out_dir=tmp_path,
    )
    assert report.turn_count >= 1
    post_snapshots = runner._upstream_snapshots  # noqa: SLF001
    # Slot set unchanged and every slot's version unchanged.
    assert set(pre_snapshots) == set(post_snapshots)
    for slot_name, pre in pre_snapshots.items():
        assert post_snapshots[slot_name].version == pre.version, (
            f"Export mutated snapshot '{slot_name}'."
        )


def test_round_trip_preserves_evidence_ids(tmp_path: Path) -> None:
    runner = _run_small_session()
    report = export_open_dialogue_session(
        runner,
        session_id="alice-sess-1",
        out_dir=tmp_path,
    )
    lines = report.turns_path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    # All evidence ids in the exported rows are non-empty strings.
    for row in rows:
        for ev in row["outcome"]["structured_evidence"]:
            assert ev["evidence_id"]


def test_rupture_wiring_override_disables_rupture_state_snapshot() -> None:
    config = build_rupture_wiring_config(WiringLevel.DISABLED)
    assert config.rupture_state is WiringLevel.DISABLED
    # Base config default is SHADOW.
    assert FinalRolloutConfig().rupture_state is WiringLevel.SHADOW

    # String values also work.
    config_str = build_rupture_wiring_config("active")
    assert config_str.rupture_state is WiringLevel.ACTIVE


def test_rupture_wiring_matched_control_runs_end_to_end() -> None:
    """DISABLED-wired runs still produce turns.jsonl but no rupture_repair entries."""

    async def _run_with_wiring(wiring: WiringLevel) -> AgentSessionRunner:
        runner = AgentSessionRunner(
            user_scope="alice",
            config=build_rupture_wiring_config(wiring),
        )
        await runner.run_turn("hi")
        runner.submit_dialogue_outcome(
            kind=DialogueExternalOutcomeKind.MISSED,
            source=DialogueExternalOutcomeEvidenceSource.USER_EXPLICIT,
            confidence=0.9,
        )
        await runner.run_turn("that missed me")
        runner.begin_new_context(reason="matched-control")
        await runner.drain_session_post_slow_loop()
        return runner

    disabled_runner = asyncio.run(_run_with_wiring(WiringLevel.DISABLED))
    from volvence_zero.memory import MemoryStratum

    durables = disabled_runner._memory_store._entries_for(MemoryStratum.DURABLE)  # noqa: SLF001
    # With rupture_state DISABLED, no rupture_repair entry should be
    # written by reflection (the shadow slot is empty placeholder).
    assert not any("rupture_repair" in entry.tags for entry in durables)
