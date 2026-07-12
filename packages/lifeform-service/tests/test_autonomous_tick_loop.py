from __future__ import annotations

from dataclasses import dataclass

from lifeform_service.session_manager import AutonomousTickReport, SessionManager


@dataclass
class _Followup:
    followup_id: str


class _FakeSession:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.tick_count = 0
        self._due = 0

    async def advance_tick(self, system_ticks: int, *, reason: str = "") -> tuple[object, ...]:
        self.tick_count += system_ticks
        if self.tick_count >= 2:
            self._due = 1
        return tuple(object() for _ in range(system_ticks))

    def due_followups(self) -> tuple[_Followup, ...]:
        return tuple(_Followup(f"due-{index}") for index in range(self._due))


class _FakeLifeform:
    def __init__(self) -> None:
        self.started = False
        self.shutdown_called = False
        self.sessions: dict[str, _FakeSession] = {}

    async def start(self) -> None:
        self.started = True

    async def shutdown(self) -> None:
        self.shutdown_called = True

    def create_session(self, *, session_id: str) -> _FakeSession:
        session = _FakeSession(session_id)
        self.sessions[session_id] = session
        return session


def _factory(_runtime) -> _FakeLifeform:
    return _FakeLifeform()


async def test_autonomous_tick_loop_advances_with_budget_and_surfaces_due_followups() -> None:
    manager = SessionManager(
        lifeform_factory=_factory,
        vertical_name="fake",
        idle_eviction_seconds=None,
    )
    await manager.create_session(session_id="s1")
    await manager.create_session(session_id="s2")

    first = await manager.advance_autonomous_ticks(
        system_ticks=2,
        max_sessions=1,
        reason="test-autonomous-loop",
    )

    assert isinstance(first[0], AutonomousTickReport)
    by_id = {report.session_id: report for report in first}
    assert by_id["s1"].ticks_advanced == 2
    assert by_id["s1"].due_followup_count == 1
    assert by_id["s1"].paused is False
    assert by_id["s2"].ticks_advanced == 0
    assert by_id["s2"].paused is True
    assert by_id["s2"].reason == "budget-exhausted"

    second = await manager.advance_autonomous_ticks(system_ticks=1)
    by_id = {report.session_id: report for report in second}
    assert by_id["s1"].ticks_advanced == 1
    assert by_id["s2"].ticks_advanced == 1


async def test_autonomous_tick_loop_rejects_invalid_budget() -> None:
    manager = SessionManager(
        lifeform_factory=_factory,
        vertical_name="fake",
        idle_eviction_seconds=None,
    )
    try:
        await manager.advance_autonomous_ticks(system_ticks=0)
    except ValueError as exc:
        assert "system_ticks" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")

