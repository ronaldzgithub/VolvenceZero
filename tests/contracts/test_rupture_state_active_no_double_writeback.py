"""Phase 1 W1.B contract test: rupture_state ACTIVE produces exactly
one rupture-repair durable entry per (user_scope, source_wave_id,
rupture_kind, repair_outcome) tuple.

Two paths can produce rupture-repair memory entries:

1. ``ReflectionModule.process`` -> ``ReflectionEngine.reflect`` ->
   :func:`rupture_repair_memory_entries`. This path runs during
   propagate when ``rupture_state`` is ACTIVE.
2. ``run_final_wiring_turn`` post-propagate ->
   :func:`enrich_reflection_snapshot_with_rupture_repair`. This path
   exists for back-compat when ``rupture_state`` is explicitly SHADOW.

Promoting ``rupture_state`` to ACTIVE without an idempotency guard
would cause both paths to fire and append the same entry twice. The
helper therefore deduplicates by ``MemoryEntry.entry_id``, which is
deterministic over the (scope, wave, kind, outcome) tuple. This test
pins both that exact one entry is emitted under either wiring level
and that the published entry_id schema is stable.

Test design: feed a typed external MISSED outcome into the runtime
turn, run with ``rupture_state`` ACTIVE and SHADOW, and inspect the
rupture-repair durable entries that the reflection snapshot carries
into the session-post writeback request. Both paths must publish the
same set of entry_ids and the count of entries with prefix
``rupture_repair:`` must be exactly one in both runs.
"""

from __future__ import annotations

import asyncio
from typing import Any

from volvence_zero.dialogue_external_outcome import DialogueExternalOutcomeModule
from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidence,
    DialogueExternalOutcomeEvidenceSource,
    DialogueExternalOutcomeKind,
)
from volvence_zero.integration import FinalRolloutConfig, run_final_wiring_turn
from volvence_zero.runtime import WiringLevel
from volvence_zero.rupture_state import (
    RuptureKind,
    RuptureStateModule,
    RuptureStateSnapshot,
)
from volvence_zero.substrate import FeatureSignal, FeatureSurfaceSubstrateAdapter


def _substrate() -> FeatureSurfaceSubstrateAdapter:
    return FeatureSurfaceSubstrateAdapter(
        model_id="rupture-no-double-writeback-model",
        feature_surface=(
            FeatureSignal(
                name="rupture_double_writeback_context",
                values=(0.5,),
                source="adapter",
            ),
        ),
    )


def _outcome_module_with_missed(turn_index: int) -> DialogueExternalOutcomeModule:
    module = DialogueExternalOutcomeModule(wiring_level=WiringLevel.ACTIVE)
    module.set_turn_index(turn_index)
    module.append_evidence(
        DialogueExternalOutcomeEvidence(
            evidence_id=f"user:explicit:missed:turn-{turn_index}",
            turn_index=turn_index,
            kind=DialogueExternalOutcomeKind.MISSED,
            source=DialogueExternalOutcomeEvidenceSource.USER_EXPLICIT,
            confidence=0.9,
            evidence_ref="user:explicit:missed",
        )
    )
    return module


def _run_turn_with_rupture_wiring(level: WiringLevel) -> dict[str, Any]:
    config = FinalRolloutConfig(rupture_state=level)
    rupture_module = RuptureStateModule(wiring_level=level)
    outcome_module = _outcome_module_with_missed(turn_index=1)
    result = asyncio.run(
        run_final_wiring_turn(
            config=config,
            substrate_adapter=_substrate(),
            user_input="that felt cold",
            session_id="rupture-no-double-writeback",
            wave_id="wave-1",
            turn_index=1,
            dialogue_external_outcome_module=outcome_module,
            rupture_state_module=rupture_module,
            apply_slow_writeback=False,
        )
    )
    return {
        "active": result.active_snapshots,
        "shadow": result.shadow_snapshots,
        "writeback_request": result.session_post_writeback_request,
    }


def _rupture_repair_entry_ids(writeback_request: Any) -> tuple[str, ...]:
    """Return the entry_ids of all rupture-repair durable entries
    queued for the session-post slow loop.
    """
    if writeback_request is None:
        return ()
    durable = writeback_request.reflection_snapshot.memory_consolidation.new_durable_entries
    return tuple(
        entry.entry_id
        for entry in durable
        if entry.entry_id.startswith("rupture_repair:")
    )


def test_rupture_state_active_publishes_to_active_snapshots() -> None:
    run = _run_turn_with_rupture_wiring(WiringLevel.ACTIVE)
    assert "rupture_state" in run["active"], (
        "rupture_state ACTIVE should publish to active_snapshots"
    )
    assert "rupture_state" not in run["shadow"], (
        "rupture_state ACTIVE must NOT also publish to shadow_snapshots"
    )
    snapshot = run["active"]["rupture_state"].value
    assert isinstance(snapshot, RuptureStateSnapshot)
    assert snapshot.rupture_kind is RuptureKind.MISREAD


def test_rupture_state_active_emits_exactly_one_durable_entry() -> None:
    run = _run_turn_with_rupture_wiring(WiringLevel.ACTIVE)
    entry_ids = _rupture_repair_entry_ids(run["writeback_request"])
    assert len(entry_ids) == 1, (
        f"Expected exactly one rupture-repair durable entry under ACTIVE; "
        f"got {len(entry_ids)}: {entry_ids!r}. A count > 1 means the "
        "propagate-internal path and the post-propagate enrichment "
        "path both wrote the same entry without entry_id dedup."
    )
    expected_prefix = "rupture_repair:anonymous:wave-1:misread:"
    assert entry_ids[0].startswith(expected_prefix), (
        f"entry_id schema drifted: expected prefix "
        f"{expected_prefix!r}, got {entry_ids[0]!r}"
    )


def test_rupture_state_shadow_back_compat_still_emits_one_entry() -> None:
    """Explicitly SHADOW wiring must keep emitting exactly one durable
    entry through the post-propagate enrichment path. This guards the
    back-compat surface that experimental / test setups rely on.
    """
    run = _run_turn_with_rupture_wiring(WiringLevel.SHADOW)
    entry_ids = _rupture_repair_entry_ids(run["writeback_request"])
    assert len(entry_ids) == 1, (
        f"SHADOW back-compat: expected exactly one rupture-repair "
        f"durable entry; got {len(entry_ids)}: {entry_ids!r}"
    )


def test_active_and_shadow_paths_publish_identical_entry_ids() -> None:
    """Both wiring levels must produce the same entry_id for the same
    typed input. This is the core matched-control invariant: promotion
    is a wiring/visibility change, not a behavior change for the
    durable rupture-repair memory.
    """
    active_run = _run_turn_with_rupture_wiring(WiringLevel.ACTIVE)
    shadow_run = _run_turn_with_rupture_wiring(WiringLevel.SHADOW)
    active_ids = _rupture_repair_entry_ids(active_run["writeback_request"])
    shadow_ids = _rupture_repair_entry_ids(shadow_run["writeback_request"])
    assert active_ids == shadow_ids, (
        f"ACTIVE and SHADOW paths produced different rupture-repair "
        f"entry_id sets: active={active_ids!r}, shadow={shadow_ids!r}"
    )
