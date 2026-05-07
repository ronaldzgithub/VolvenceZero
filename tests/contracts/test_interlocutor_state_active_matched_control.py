"""Matched-control E2E gate for interlocutor_state SHADOW -> ACTIVE.

The W2 SSOT cleanup landed ``InterlocutorStateModule`` as the single
producer of the 12-axis readout but kept the wiring level at SHADOW
because the on-record promotion gate required matched-control
evidence: planner / synthesizer / lifeform consume the same readout
when the slot is in ``active_snapshots`` vs ``shadow_snapshots``.

This contract test runs the full kernel turn twice with otherwise
identical inputs and asserts:

* SHADOW path: the snapshot lives in ``shadow_snapshots``;
  ``active_snapshots`` does not contain ``interlocutor_state``.
* ACTIVE path: the snapshot lives in ``active_snapshots``;
  ``shadow_snapshots`` does not contain ``interlocutor_state``.
* The published ``InterlocutorState`` value is byte-for-byte
  identical across the two runs (axes, zone bools, confidence,
  rationale, owner-authored description). Promotion changes only
  visibility, not the readout itself.
* No other active slot's value digest is disturbed by the wiring
  flip. This guards against accidentally adding ``interlocutor_state``
  as a kernel-side dependency without a contract review.

The test is parameterless on purpose: the wiring flip is the change
under test, so we hard-code both ``WiringLevel.SHADOW`` and
``WiringLevel.ACTIVE`` rather than rely on the current default.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any

from volvence_zero.integration import FinalRolloutConfig, run_final_wiring_turn
from volvence_zero.interlocutor import (
    InterlocutorState,
    InterlocutorStateSnapshot,
)
from volvence_zero.runtime import Snapshot, WiringLevel
from volvence_zero.substrate import FeatureSignal, FeatureSurfaceSubstrateAdapter


_MATCHED_CONTROL_MODEL_ID = "interlocutor-matched-control-model"
_MATCHED_CONTROL_FEATURE = (
    FeatureSignal(
        name="interlocutor_matched_control_context",
        values=(0.5,),
        source="adapter",
    ),
)


def _run_turn_with_wiring(level: WiringLevel) -> dict[str, Any]:
    """Run one ``run_final_wiring_turn`` with ``interlocutor_state`` at level."""

    config = FinalRolloutConfig(interlocutor_state=level)
    result = asyncio.run(
        run_final_wiring_turn(
            config=config,
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id=_MATCHED_CONTROL_MODEL_ID,
                feature_surface=_MATCHED_CONTROL_FEATURE,
            ),
            session_id="interlocutor-matched-control-session",
            wave_id="interlocutor-matched-control-wave",
        )
    )
    return {
        "active": result.active_snapshots,
        "shadow": result.shadow_snapshots,
    }


def _interlocutor_snapshot(snapshots: dict[str, Snapshot[Any]]) -> InterlocutorStateSnapshot:
    snapshot = snapshots["interlocutor_state"]
    assert isinstance(snapshot.value, InterlocutorStateSnapshot)
    return snapshot.value


def test_shadow_publishes_to_shadow_only_active_publishes_to_active_only() -> None:
    shadow_run = _run_turn_with_wiring(WiringLevel.SHADOW)
    active_run = _run_turn_with_wiring(WiringLevel.ACTIVE)

    assert "interlocutor_state" in shadow_run["shadow"]
    assert "interlocutor_state" not in shadow_run["active"]

    assert "interlocutor_state" in active_run["active"]
    assert "interlocutor_state" not in active_run["shadow"]


def test_active_promotion_does_not_change_published_state() -> None:
    """Same upstream + same readout -> same 12-axis state.

    Promotion is a pure visibility change. If this assertion ever
    fires, something is mutating the readout based on the wiring
    level itself, which would be a contract violation.
    """

    shadow_payload = _interlocutor_snapshot(
        _run_turn_with_wiring(WiringLevel.SHADOW)["shadow"]
    )
    active_payload = _interlocutor_snapshot(
        _run_turn_with_wiring(WiringLevel.ACTIVE)["active"]
    )

    shadow_state: InterlocutorState = shadow_payload.state
    active_state: InterlocutorState = active_payload.state

    assert dataclasses.asdict(shadow_state) == dataclasses.asdict(active_state)
    assert shadow_payload.description == active_payload.description


def _drifted_slot_names(
    a: dict[str, Snapshot[Any]], b: dict[str, Snapshot[Any]]
) -> set[str]:
    """Return active slot names whose value differs between two runs."""
    shared = set(a) & set(b)
    return {name for name in shared if a[name].value != b[name].value}


def test_active_promotion_does_not_widen_active_slot_set() -> None:
    """Promoting ``interlocutor_state`` MUST shift exactly one slot
    from ``shadow_snapshots`` to ``active_snapshots`` and nothing
    else. No kernel-side module currently declares
    ``interlocutor_state`` as a dependency; this test pins that
    invariant so adding such a dependency requires an explicit
    contract review.
    """

    shadow_run = _run_turn_with_wiring(WiringLevel.SHADOW)
    active_run = _run_turn_with_wiring(WiringLevel.ACTIVE)

    only_in_active_run = set(active_run["active"]) - set(shadow_run["active"])
    only_in_shadow_run = set(shadow_run["active"]) - set(active_run["active"])
    assert only_in_active_run == {"interlocutor_state"}, only_in_active_run
    assert only_in_shadow_run == set(), only_in_shadow_run


def test_active_promotion_does_not_introduce_new_value_drift() -> None:
    """Active-to-active drift caused by the wiring flip must not
    exceed the baseline drift caused by run-to-run non-determinism.

    Several owner snapshots (``credit`` / ``case_memory`` /
    ``dual_track`` / ``memory`` / ``reflection``) carry incremental
    state (counters / timestamps / running stats) that yields a
    legitimate baseline drift between any two independent
    ``run_final_wiring_turn`` calls regardless of wiring. The
    matched-control invariant is "promoting interlocutor_state does
    not introduce ADDITIONAL drift on top of that baseline".

    If a kernel module silently takes ``interlocutor_state`` as a
    dependency, its slot will appear in the SHADOW-vs-ACTIVE drift
    set but not in the SHADOW-vs-SHADOW baseline set, and this
    test will fire.
    """

    baseline_a = _run_turn_with_wiring(WiringLevel.SHADOW)
    baseline_b = _run_turn_with_wiring(WiringLevel.SHADOW)
    baseline_drift = _drifted_slot_names(
        baseline_a["active"], baseline_b["active"]
    )

    shadow_run = _run_turn_with_wiring(WiringLevel.SHADOW)
    active_run = _run_turn_with_wiring(WiringLevel.ACTIVE)
    promotion_drift = _drifted_slot_names(
        shadow_run["active"], active_run["active"]
    )

    new_drift = promotion_drift - baseline_drift
    assert not new_drift, (
        "Promoting interlocutor_state SHADOW->ACTIVE introduced new "
        f"value drift on slots that are stable across baseline runs: "
        f"{sorted(new_drift)}. Either a kernel module silently took "
        "interlocutor_state as a dependency, or the propagation "
        "order changed. Both require a contract review."
    )
