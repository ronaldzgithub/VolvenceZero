"""Phase 2 W2.B matched-control gate for session_post_slow_loop.

The module's ``default_wiring_level`` was already ACTIVE; the
``FinalRolloutConfig`` override was the only obstacle to publishing
the deferred-job queue snapshot into ``active_snapshots``. This
contract test pins the W2.B flip:

* SHADOW config: snapshot lives in ``shadow_snapshots`` only;
* ACTIVE config: snapshot lives in ``active_snapshots`` only and the
  payload is byte-for-byte identical to the SHADOW publication when
  the upstream input is identical;
* Promotion does not introduce new value drift on unrelated slots
  beyond the run-to-run baseline. ``session_post_slow_loop`` declares
  no kernel dependencies, so we expect zero new drift.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any

from volvence_zero.agent.session import AgentSessionRunner
from volvence_zero.agent.session_post_slow_loop import (
    SessionPostSlowLoopSnapshot,
)
from volvence_zero.integration import FinalRolloutConfig
from volvence_zero.runtime import Snapshot, WiringLevel


def _run_turn_via_runner(level: WiringLevel) -> dict[str, dict[str, Snapshot[Any]]]:
    """``session_post_slow_loop`` is published by the
    ``AgentSessionRunner`` after each turn (it is not part of the
    ``run_final_wiring_turn`` ``modules`` list). Use the runner so the
    snapshot lands in active / shadow according to its wiring.
    """
    runner = AgentSessionRunner(
        session_id="session-post-mc",
        config=FinalRolloutConfig(session_post_slow_loop=level),
    )
    result = asyncio.run(runner.run_turn("hi there"))
    return {
        "active": result.active_snapshots,
        "shadow": result.shadow_snapshots,
    }


def _drifted_slot_names(
    a: dict[str, Snapshot[Any]], b: dict[str, Snapshot[Any]]
) -> set[str]:
    shared = set(a) & set(b)
    return {name for name in shared if a[name].value != b[name].value}


def test_shadow_publishes_to_shadow_only_active_publishes_to_active_only() -> None:
    shadow_run = _run_turn_via_runner(WiringLevel.SHADOW)
    active_run = _run_turn_via_runner(WiringLevel.ACTIVE)
    assert "session_post_slow_loop" in shadow_run["shadow"]
    assert "session_post_slow_loop" not in shadow_run["active"]
    assert "session_post_slow_loop" in active_run["active"]
    assert "session_post_slow_loop" not in active_run["shadow"]


def test_active_promotion_does_not_change_published_state() -> None:
    shadow_value = _run_turn_via_runner(WiringLevel.SHADOW)["shadow"][
        "session_post_slow_loop"
    ].value
    active_value = _run_turn_via_runner(WiringLevel.ACTIVE)["active"][
        "session_post_slow_loop"
    ].value
    assert isinstance(shadow_value, SessionPostSlowLoopSnapshot)
    assert isinstance(active_value, SessionPostSlowLoopSnapshot)
    assert dataclasses.asdict(shadow_value) == dataclasses.asdict(active_value)


def test_active_promotion_does_not_introduce_new_value_drift() -> None:
    """``session_post_slow_loop`` has no kernel dependencies so its
    promotion must not cause any other slot to drift beyond the
    baseline run-to-run noise. If a kernel module silently takes
    ``session_post_slow_loop`` as a dependency this test will fire.
    """
    baseline_a = _run_turn_via_runner(WiringLevel.SHADOW)
    baseline_b = _run_turn_via_runner(WiringLevel.SHADOW)
    baseline_drift = _drifted_slot_names(baseline_a["active"], baseline_b["active"])

    shadow_run = _run_turn_via_runner(WiringLevel.SHADOW)
    active_run = _run_turn_via_runner(WiringLevel.ACTIVE)
    promotion_drift = _drifted_slot_names(shadow_run["active"], active_run["active"])

    new_drift = promotion_drift - baseline_drift
    assert not new_drift, (
        "Promoting session_post_slow_loop SHADOW->ACTIVE introduced "
        f"new value drift on slots that are stable across baseline runs: "
        f"{sorted(new_drift)}."
    )
