"""CP-20 (GAP-07): service-level followup execution loop.

Covers the gate chain (consent / cooldown / budget / tenant / readonly),
the canonical ``FOLLOWUP_DUE`` turn execution with ``followup:<id>``
provenance, and a simulated 24h idle arc proving proactive behaviour does
not flood and never crosses a consent boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from lifeform_core import TurnTriggerKind
from lifeform_service.session_manager import (
    FollowupExecutionReport,
    SessionManager,
)
from volvence_zero.semantic_state import BoundaryConsentSnapshot


def _consent_snapshot(*, blocked: bool) -> BoundaryConsentSnapshot:
    return BoundaryConsentSnapshot(
        granted_consents=(),
        missing_consents=(),
        denied_boundaries=(),
        memory_consent="granted",
        external_action_consent="granted" if not blocked else "denied",
        compliance_score=1.0,
        control_signal=0.0,
        description="synthetic consent snapshot for followup loop tests",
        external_action_blocked=blocked,
    )


@dataclass
class _SnapshotEnvelope:
    value: object


@dataclass
class _ProactiveTurn:
    text: str
    trigger_kind: TurnTriggerKind
    provenance: str | None


@dataclass
class _Followup:
    followup_id: str
    description: str = "check in on the open thread"


class _FakeSession:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.tick_count = 0
        self.pending: list[_Followup] = []
        self.turns: list[_ProactiveTurn] = []
        self.consent_value: object | None = _consent_snapshot(blocked=False)
        self._followup_seq = 0

    async def advance_tick(self, system_ticks: int, *, reason: str = "") -> tuple[object, ...]:
        self.tick_count += system_ticks
        # One new due followup per tick batch — a worst-case chatty vitals
        # owner, to stress the anti-flood gates.
        self._followup_seq += 1
        self.pending.append(_Followup(f"{self.session_id}-due-{self._followup_seq}"))
        return tuple(object() for _ in range(system_ticks))

    def due_followups(self) -> tuple[_Followup, ...]:
        return tuple(self.pending)

    def acknowledge_followup(self, followup_id: str) -> bool:
        before = len(self.pending)
        self.pending = [f for f in self.pending if f.followup_id != followup_id]
        return len(self.pending) < before

    @property
    def latest_active_snapshots(self) -> dict[str, object]:
        if self.consent_value is None:
            return {}
        return {"boundary_consent": _SnapshotEnvelope(self.consent_value)}

    async def run_turn(
        self,
        user_input: str,
        *,
        trigger_kind: TurnTriggerKind = TurnTriggerKind.USER_INPUT,
        environment_provenance: str | None = None,
        environment_consent_context: tuple[str, ...] = (),
    ) -> object:
        self.turns.append(
            _ProactiveTurn(
                text=user_input,
                trigger_kind=trigger_kind,
                provenance=environment_provenance,
            )
        )
        return object()


class _FakeLifeform:
    def __init__(self) -> None:
        self.sessions: dict[str, _FakeSession] = {}

    async def start(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    def create_session(self, *, session_id: str) -> _FakeSession:
        session = _FakeSession(session_id)
        self.sessions[session_id] = session
        return session


@dataclass
class _FakeClock:
    now: float = 0.0
    calls: int = field(default=0)

    def __call__(self) -> float:
        self.calls += 1
        return self.now


def _manager(clock: _FakeClock) -> SessionManager:
    return SessionManager(
        lifeform_factory=lambda _runtime: _FakeLifeform(),
        vertical_name="fake",
        idle_eviction_seconds=None,
        clock=clock,
    )


async def _session(manager: SessionManager, session_id: str) -> _FakeSession:
    await manager.create_session(session_id=session_id)
    entry_session = manager._sessions[session_id].session  # test-only reach-in
    assert isinstance(entry_session, _FakeSession)
    return entry_session


async def test_executed_followup_runs_canonical_turn_with_lineage() -> None:
    clock = _FakeClock()
    manager = _manager(clock)
    session = await _session(manager, "s1")
    await manager.advance_autonomous_ticks(system_ticks=1)

    reports = await manager.execute_due_followups()

    executed = [r for r in reports if r.executed]
    assert len(executed) == 1
    assert isinstance(executed[0], FollowupExecutionReport)
    assert executed[0].reason == "executed"
    assert len(session.turns) == 1
    turn = session.turns[0]
    assert turn.trigger_kind is TurnTriggerKind.FOLLOWUP_DUE
    assert turn.provenance == f"followup:{executed[0].followup_id}"
    # Executed followup was acknowledged (retired from the due list).
    assert session.due_followups() == ()


async def test_consent_gate_blocks_and_missing_snapshot_is_conservative() -> None:
    clock = _FakeClock()
    manager = _manager(clock)
    blocked = await _session(manager, "blocked")
    unknown = await _session(manager, "unknown")
    blocked.consent_value = _consent_snapshot(blocked=True)
    unknown.consent_value = None
    await manager.advance_autonomous_ticks(system_ticks=1)

    reports = await manager.execute_due_followups()

    by_session: dict[str, list[FollowupExecutionReport]] = {}
    for report in reports:
        by_session.setdefault(report.session_id, []).append(report)
    assert all(r.reason == "consent-blocked" for r in by_session["blocked"])
    assert all(r.reason == "consent-unknown" for r in by_session["unknown"])
    assert blocked.turns == []
    assert unknown.turns == []


async def test_cooldown_and_budget_gates_are_typed() -> None:
    clock = _FakeClock()
    manager = _manager(clock)
    session = await _session(manager, "s1")
    await manager.advance_autonomous_ticks(system_ticks=1)
    await manager.advance_autonomous_ticks(system_ticks=1)
    assert len(session.due_followups()) == 2

    # cooldown=0 isolates the budget gate: per_session_max=1 lets exactly
    # one execute; the second is budget-exhausted.
    first = await manager.execute_due_followups(
        per_session_max=1, cooldown_seconds=0.0
    )
    reasons = sorted(r.reason for r in first)
    assert reasons == ["budget-exhausted", "executed"]

    # With a cooldown window, the remaining one is cooldown-blocked.
    second = await manager.execute_due_followups(cooldown_seconds=600.0)
    assert [r.reason for r in second] == ["cooldown-active"]
    # After the cooldown window passes, it executes.
    clock.now += 601.0
    third = await manager.execute_due_followups(cooldown_seconds=600.0)
    assert [r.reason for r in third] == ["executed"]


async def test_tenant_allowlist_gate() -> None:
    clock = _FakeClock()
    manager = _manager(clock)
    session = await _session(manager, "s1")
    await manager.advance_autonomous_ticks(system_ticks=1)

    denied = await manager.execute_due_followups(
        tenant_allowlist=frozenset({"some-other-user"})
    )
    assert [r.reason for r in denied] == ["tenant-denied"]
    assert session.turns == []


async def test_invalid_parameters_fail_loudly() -> None:
    manager = _manager(_FakeClock())
    with pytest.raises(ValueError, match="max_turns"):
        await manager.execute_due_followups(max_turns=0)
    with pytest.raises(ValueError, match="per_session_max"):
        await manager.execute_due_followups(per_session_max=0)
    with pytest.raises(ValueError, match="cooldown_seconds"):
        await manager.execute_due_followups(cooldown_seconds=-1.0)


async def test_24h_idle_arc_no_flooding_and_no_consent_violation() -> None:
    """Simulated 24h idle arc (fake clock, 5-minute service cadence).

    Two chatty sessions (one consented, one consent-blocked) accumulate a
    due followup every cadence step. Exit conditions from the plan:
    proactive behaviour must not flood (bounded by the 1h cooldown, not by
    the 288 opportunities) and must never cross the consent boundary.
    """

    clock = _FakeClock()
    manager = _manager(clock)
    allowed = await _session(manager, "allowed")
    blocked = await _session(manager, "blocked")
    blocked.consent_value = _consent_snapshot(blocked=True)

    step_seconds = 300.0  # 5-minute cadence
    steps = int(24 * 3600 / step_seconds)  # 288 steps = 24h
    for _ in range(steps):
        clock.now += step_seconds
        await manager.advance_autonomous_ticks(system_ticks=1)
        await manager.execute_due_followups(
            max_turns=2,
            per_session_max=1,
            cooldown_seconds=3600.0,  # at most one proactive turn per hour
        )

    # Anti-flood: 288 opportunities, but the cooldown caps engagement at
    # ~one per hour (24) — allow +1 for boundary alignment.
    assert 1 <= len(allowed.turns) <= 25, (
        f"proactive turns flooded: {len(allowed.turns)} in 24h"
    )
    assert all(
        turn.trigger_kind is TurnTriggerKind.FOLLOWUP_DUE for turn in allowed.turns
    )
    # Consent boundary: zero proactive turns for the blocked session.
    assert blocked.turns == []
