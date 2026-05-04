"""Integration smoke tests for rupture_state + dialogue_external_outcome.

These tests run the full ``run_final_wiring_turn`` path to prove that:

* ``rupture_state`` is published as a SHADOW snapshot every turn;
* the SHADOW snapshot does not leak into ``active_snapshots`` (R8 / M1);
* ``dialogue_external_outcome`` is published as an ACTIVE snapshot;
* ``rupture_state`` sees non-empty evidence when an external MISSED
  entry is appended via ``DialogueExternalOutcomeModule.append_evidence``.
"""

from __future__ import annotations

import asyncio

from volvence_zero.dialogue_external_outcome import DialogueExternalOutcomeModule
from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidence,
    DialogueExternalOutcomeEvidenceSource,
    DialogueExternalOutcomeKind,
    DialogueExternalOutcomeSnapshot,
)
from volvence_zero.integration import FinalRolloutConfig
from volvence_zero.integration.final_wiring import run_final_wiring_turn
from volvence_zero.rupture_state import (
    RuptureEvidenceSource,
    RuptureKind,
    RuptureStateModule,
    RuptureStateSnapshot,
)
from volvence_zero.runtime import WiringLevel
from volvence_zero.substrate import FeatureSurfaceSubstrateAdapter, FeatureSignal


def _substrate_adapter() -> FeatureSurfaceSubstrateAdapter:
    return FeatureSurfaceSubstrateAdapter(
        model_id="rupture-state-test",
        feature_surface=(
            FeatureSignal(name="test_context", values=(0.5,), source="adapter"),
        ),
    )


def test_rupture_state_publishes_shadow_each_turn() -> None:
    config = FinalRolloutConfig()
    adapter = _substrate_adapter()
    result = asyncio.run(
        run_final_wiring_turn(
            config=config,
            substrate_adapter=adapter,
            user_input="hi there",
            session_id="sess-rupture-shadow",
            wave_id="wave-0",
            turn_index=0,
            apply_slow_writeback=False,
        )
    )

    shadow = result.shadow_snapshots
    active = result.active_snapshots
    assert "rupture_state" in shadow, (
        "rupture_state must publish a SHADOW snapshot every turn"
    )
    assert "rupture_state" not in active, (
        "rupture_state is SHADOW-only in v0; it must NOT leak into active_snapshots"
    )
    snapshot = shadow["rupture_state"].value
    assert isinstance(snapshot, RuptureStateSnapshot)
    # Initial turn: no external signal and bootstrap PE; no rupture.
    assert snapshot.rupture_kind is None
    assert snapshot.internal_suspected_only in (True, False)
    # dialogue_external_outcome is ACTIVE — the single legal channel.
    assert "dialogue_external_outcome" in active


def test_external_missed_flows_into_shadow_rupture_state() -> None:
    config = FinalRolloutConfig()
    adapter = _substrate_adapter()

    # Build the external-outcome module explicitly so we can pre-load
    # one typed MISSED evidence; the owner publishes it on its next
    # turn, and rupture_state reads it through snapshot.
    outcome_module = DialogueExternalOutcomeModule(
        wiring_level=WiringLevel.ACTIVE,
    )
    outcome_module.set_turn_index(1)
    outcome_module.append_evidence(
        DialogueExternalOutcomeEvidence(
            evidence_id="user:explicit:missed:turn-1",
            turn_index=1,
            kind=DialogueExternalOutcomeKind.MISSED,
            source=DialogueExternalOutcomeEvidenceSource.USER_EXPLICIT,
            confidence=0.9,
            evidence_ref="user:explicit:missed",
        )
    )

    rupture_module = RuptureStateModule(wiring_level=WiringLevel.SHADOW)

    result = asyncio.run(
        run_final_wiring_turn(
            config=config,
            substrate_adapter=adapter,
            user_input="that felt cold",
            session_id="sess-rupture-missed",
            wave_id="wave-1",
            turn_index=1,
            dialogue_external_outcome_module=outcome_module,
            rupture_state_module=rupture_module,
            apply_slow_writeback=False,
        )
    )

    rupture_snapshot = result.shadow_snapshots["rupture_state"].value
    assert isinstance(rupture_snapshot, RuptureStateSnapshot)
    assert rupture_snapshot.rupture_kind is RuptureKind.MISREAD
    assert rupture_snapshot.internal_suspected_only is False
    assert RuptureEvidenceSource.EXTERNAL_USER in rupture_snapshot.evidence_sources

    # Crucially: the external-outcome snapshot must carry the evidence.
    outcome_snapshot = result.active_snapshots["dialogue_external_outcome"].value
    assert isinstance(outcome_snapshot, DialogueExternalOutcomeSnapshot)
    assert len(outcome_snapshot.entries) == 1
    assert outcome_snapshot.entries[0].kind is DialogueExternalOutcomeKind.MISSED
