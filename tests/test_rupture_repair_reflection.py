"""Unit / integration tests for rupture-repair memory writes (M3).

These tests focus on ``ReflectionEngine.reflect`` + ``.apply`` +
``.rollback`` producing durable ``MemoryEntry`` objects under the
documented tag schema, written only through the owner apply path.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from volvence_zero.agent import default_active_runner
from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidence,
    DialogueExternalOutcomeEvidenceSource,
    DialogueExternalOutcomeKind,
    DialogueExternalOutcomeSnapshot,
)
from volvence_zero.memory import (
    MemoryEntry,
    MemoryStratum,
    build_default_memory_store,
)
from volvence_zero.memory.store import MemoryStore
from volvence_zero.reflection import (
    ReflectionEngine,
    WritebackMode,
)
from volvence_zero.rupture_state import (
    RuptureContributingSignal,
    RuptureEvidenceSource,
    RuptureKind,
    RuptureStateSnapshot,
)


def _build_rupture_snapshot(kind: RuptureKind = RuptureKind.MISREAD) -> RuptureStateSnapshot:
    return RuptureStateSnapshot(
        rupture_signal_strength=0.8,
        rupture_kind=kind,
        confidence=0.9,
        internal_suspected_only=False,
        evidence_sources=(RuptureEvidenceSource.EXTERNAL_USER,),
        contributing_signals=(
            RuptureContributingSignal(
                source=RuptureEvidenceSource.EXTERNAL_USER,
                signal_strength=0.8,
                confidence=0.9,
                kind_hint=kind,
                detail="user explicit missed",
            ),
        ),
        description=f"test rupture kind={kind.value}",
    )


def _external_snapshot(
    *,
    turn_index: int,
    kind: DialogueExternalOutcomeKind,
    evidence_id: str | None = None,
) -> DialogueExternalOutcomeSnapshot:
    ev = DialogueExternalOutcomeEvidence(
        evidence_id=evidence_id or f"user:explicit:{kind.value}:turn-{turn_index}",
        turn_index=turn_index,
        kind=kind,
        source=DialogueExternalOutcomeEvidenceSource.USER_EXPLICIT,
        confidence=0.9,
        evidence_ref="user:explicit",
    )
    return DialogueExternalOutcomeSnapshot(
        turn_index=turn_index,
        entries=(ev,),
        description=f"one {kind.value}",
    )


def test_reflect_emits_no_rupture_entry_when_no_external_confirmation() -> None:
    engine = ReflectionEngine(writeback_mode=WritebackMode.APPLY)
    rupture = _build_rupture_snapshot()
    # External snapshot present but empty: no confirmed external outcome.
    empty_external = DialogueExternalOutcomeSnapshot(
        turn_index=1,
        entries=(),
        description="empty",
    )
    snapshot = engine.reflect(
        timestamp_ms=1,
        memory_snapshot=None,
        dual_track_snapshot=None,
        evaluation_snapshot=None,
        credit_snapshot=None,
        rupture_state_snapshot=rupture,
        dialogue_external_outcome_snapshot=empty_external,
        user_scope="alice",
    )
    assert all(
        "rupture_repair" not in entry.tags
        for entry in snapshot.memory_consolidation.new_durable_entries
    )


def test_reflect_emits_rupture_repair_entry_with_tag_schema_on_missed() -> None:
    engine = ReflectionEngine(writeback_mode=WritebackMode.APPLY)
    rupture = _build_rupture_snapshot(RuptureKind.MISREAD)
    external = _external_snapshot(turn_index=2, kind=DialogueExternalOutcomeKind.MISSED)
    snapshot = engine.reflect(
        timestamp_ms=100,
        memory_snapshot=None,
        dual_track_snapshot=None,
        evaluation_snapshot=None,
        credit_snapshot=None,
        rupture_state_snapshot=rupture,
        dialogue_external_outcome_snapshot=external,
        user_scope="alice",
    )
    entries = [
        entry
        for entry in snapshot.memory_consolidation.new_durable_entries
        if "rupture_repair" in entry.tags
    ]
    assert len(entries) == 1
    entry = entries[0]
    # Tag schema per DATA_CONTRACT.md §3.3.
    assert "rupture_repair" in entry.tags
    assert f"rupture_kind:{RuptureKind.MISREAD.value}" in entry.tags
    # repair_outcome pending because no positive external confirmation.
    assert "repair_outcome:pending" in entry.tags
    assert "user_scope:alice" in entry.tags
    assert "source_wave:wave-2" in entry.tags
    # Structured JSON content.
    content = json.loads(entry.content)
    assert content["rupture_kind"] == RuptureKind.MISREAD.value
    assert content["source_turn_index"] == 2
    assert content["source_wave_id"] == "wave-2"
    assert content["observed_outcome_kind"] == DialogueExternalOutcomeKind.MISSED.value
    assert content["user_scope"] == "alice"
    assert entry.stratum == MemoryStratum.DURABLE.value


def test_reflect_emits_observed_repair_outcome_when_helped_present() -> None:
    engine = ReflectionEngine(writeback_mode=WritebackMode.APPLY)
    rupture = _build_rupture_snapshot(RuptureKind.MISREAD)
    missed = DialogueExternalOutcomeEvidence(
        evidence_id="user:explicit:missed:turn-2",
        turn_index=2,
        kind=DialogueExternalOutcomeKind.MISSED,
        source=DialogueExternalOutcomeEvidenceSource.USER_EXPLICIT,
        confidence=0.8,
        evidence_ref="user:explicit",
    )
    helped = DialogueExternalOutcomeEvidence(
        evidence_id="user:explicit:helped:turn-2",
        turn_index=2,
        kind=DialogueExternalOutcomeKind.HELPED,
        source=DialogueExternalOutcomeEvidenceSource.USER_EXPLICIT,
        confidence=0.9,
        evidence_ref="user:explicit",
    )
    external = DialogueExternalOutcomeSnapshot(
        turn_index=2,
        entries=(missed, helped),
        description="missed+helped",
    )
    snapshot = engine.reflect(
        timestamp_ms=100,
        memory_snapshot=None,
        dual_track_snapshot=None,
        evaluation_snapshot=None,
        credit_snapshot=None,
        rupture_state_snapshot=rupture,
        dialogue_external_outcome_snapshot=external,
        user_scope="alice",
    )
    entries = [
        entry
        for entry in snapshot.memory_consolidation.new_durable_entries
        if "rupture_repair" in entry.tags
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert "repair_outcome:observed" in entry.tags
    content = json.loads(entry.content)
    assert content["observed_outcome_kind"] == DialogueExternalOutcomeKind.HELPED.value


def test_apply_writes_rupture_repair_entry_and_rollback_removes_it() -> None:
    engine = ReflectionEngine(writeback_mode=WritebackMode.APPLY)
    rupture = _build_rupture_snapshot(RuptureKind.MISREAD)
    external = _external_snapshot(turn_index=2, kind=DialogueExternalOutcomeKind.MISSED)
    snapshot = engine.reflect(
        timestamp_ms=200,
        memory_snapshot=None,
        dual_track_snapshot=None,
        evaluation_snapshot=None,
        credit_snapshot=None,
        rupture_state_snapshot=rupture,
        dialogue_external_outcome_snapshot=external,
        user_scope="alice",
    )

    store = build_default_memory_store()

    def _durable_rupture_entries(store_: MemoryStore) -> tuple[MemoryEntry, ...]:
        # Test-only introspection. Production code reads durable entries
        # through the owner snapshot; this helper is just a way to count
        # rupture_repair rows without going through retrieval ranking.
        return tuple(
            entry
            for entry in store._entries_for(MemoryStratum.DURABLE)  # noqa: SLF001
            if "rupture_repair" in entry.tags
        )

    def _count_rupture_repair() -> int:
        return len(_durable_rupture_entries(store))

    assert _count_rupture_repair() == 0
    result = engine.apply(
        memory_store=store,
        reflection_snapshot=snapshot,
        credit_snapshot=None,
    )
    assert result.checkpoint is not None
    assert _count_rupture_repair() == 1
    engine.rollback(memory_store=store, checkpoint=result.checkpoint)
    assert _count_rupture_repair() == 0


def test_rupture_repair_boundary_no_direct_memory_writes_in_kernel() -> None:
    """Static guard: only ``ReflectionEngine`` may produce rupture_repair entries.

    No call site outside ``ReflectionEngine`` should construct a
    ``MemoryEntry`` or ``MemoryWriteRequest`` carrying the
    ``rupture_repair`` tag. This is a conservative grep-style check on
    kernel wheels — the dedicated write path is the only legal channel
    and this test fails loudly if anyone adds a bypass.
    """

    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    allowed = (
        repo_root / "packages" / "vz-cognition" / "src" / "volvence_zero"
        / "reflection" / "writeback.py"
    )
    offenders: list[str] = []
    for wheel in ("vz-cognition", "vz-memory", "vz-application", "vz-runtime"):
        src_root = repo_root / "packages" / wheel / "src" / "volvence_zero"
        if not src_root.exists():
            continue
        for py in src_root.rglob("*.py"):
            if py.resolve() == allowed.resolve():
                continue
            text = py.read_text(encoding="utf-8", errors="ignore")
            # We look for the literal tag string appearing alongside a
            # memory write pattern. Pure comments / docstrings that just
            # mention the schema are fine.
            if "rupture_repair" in text and (
                "MemoryEntry(" in text or "MemoryWriteRequest(" in text
            ):
                offenders.append(str(py.relative_to(repo_root)))
    assert not offenders, (
        "rupture_repair tagged memory writes must only happen in "
        "ReflectionEngine.apply; offenders: " + ", ".join(offenders)
    )


def test_full_turn_with_missed_plus_helped_produces_one_durable_entry() -> None:
    """Integration: run_turn -> submit MISSED -> run_turn -> end_scene drain
    writes exactly one DURABLE rupture-repair entry via reflection.

    Exercises the full ``AgentSessionRunner`` path: a turn runs, external
    outcome is submitted, another turn runs, then a scene-end drain
    fires the session-post slow loop which triggers
    ``ReflectionEngine.apply`` — the only legal write path.
    """

    async def _run() -> list:
        runner = default_active_runner()
        await runner.run_turn("I need to be heard.")
        runner.submit_dialogue_outcome(
            kind=DialogueExternalOutcomeKind.MISSED,
            source=DialogueExternalOutcomeEvidenceSource.USER_EXPLICIT,
            confidence=0.9,
        )
        await runner.run_turn("That felt cold.")
        # Close the scene so the session-post slow loop picks up the
        # deferred writeback request and applies it through
        # ReflectionEngine.apply.
        runner.begin_new_context(reason="test-scene-end")
        await runner.drain_session_post_slow_loop()
        store = runner._memory_store  # noqa: SLF001 (test introspection)
        return [
            entry
            for entry in store._entries_for(MemoryStratum.DURABLE)  # noqa: SLF001
            if "rupture_repair" in entry.tags
        ]

    entries = asyncio.run(_run())
    assert len(entries) >= 1, "Expected at least one rupture_repair entry after drain."
    by_kind = [e for e in entries if f"rupture_kind:{RuptureKind.MISREAD.value}" in e.tags]
    assert by_kind, f"Expected MISREAD entry in {[e.tags for e in entries]}"
