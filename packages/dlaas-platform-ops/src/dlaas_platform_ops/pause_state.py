"""Per-session pause state for operator-takeover.

Owns the ``{(ai_id, session_id) -> _SessionState}`` map. The dispatch
handler in ``dlaas-platform-api`` consults this state at the start of
every interaction and short-circuits with an ``operator_takeover``
placeholder when the session is paused — the kernel never receives
the user turn while ops has the floor.

Pause state is process-local in Slice 5.1; persisting it across
restarts is a Slice 7 follow-up. The store also accumulates a typed
:class:`OperatorMessage` history per session so the ops stream and
the human-reply / resume endpoints can replay pending operator
turns deterministically.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OperatorMessage:
    """One operator turn injected during a takeover."""

    operator_id: str
    text: str
    inject_into_runtime: bool
    created_at_ms: int

    def to_json(self) -> dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "text": self.text,
            "inject_into_runtime": self.inject_into_runtime,
            "created_at_ms": self.created_at_ms,
        }


@dataclass
class _SessionState:
    paused: bool = False
    pause_operator_id: str = ""
    pause_note: str = ""
    paused_at_ms: int = 0
    operator_messages: list[OperatorMessage] = field(default_factory=list)


class PauseStore:
    """Thread-safe owner of the ``(ai_id, session_id) -> _SessionState`` map."""

    def __init__(self) -> None:
        self._states: dict[tuple[str, str], _SessionState] = {}
        self._lock = asyncio.Lock()

    async def is_paused(self, *, ai_id: str, session_id: str) -> bool:
        async with self._lock:
            state = self._states.get((ai_id, session_id))
            return state.paused if state is not None else False

    async def pause(
        self,
        *,
        ai_id: str,
        session_id: str,
        operator_id: str,
        note: str = "",
    ) -> _SessionState:
        now_ms = int(time.time() * 1000.0)
        async with self._lock:
            state = self._states.setdefault(
                (ai_id, session_id), _SessionState()
            )
            state.paused = True
            state.pause_operator_id = operator_id
            state.pause_note = note
            state.paused_at_ms = now_ms
            return _clone_state(state)

    async def resume(
        self,
        *,
        ai_id: str,
        session_id: str,
        operator_id: str,
        note: str = "",
    ) -> _SessionState | None:
        """Mark the session as resumed; returns the prior state.

        ``note`` is recorded only as a side-effect on the state for
        audit; the typed ledger surfaces it via the SSE conversation
        stream.
        """
        async with self._lock:
            state = self._states.get((ai_id, session_id))
            if state is None or not state.paused:
                return None
            state.paused = False
            state.pause_operator_id = operator_id
            state.pause_note = note
            return _clone_state(state)

    async def append_operator_message(
        self,
        *,
        ai_id: str,
        session_id: str,
        operator_id: str,
        text: str,
        inject_into_runtime: bool,
    ) -> OperatorMessage:
        async with self._lock:
            state = self._states.setdefault(
                (ai_id, session_id), _SessionState()
            )
            msg = OperatorMessage(
                operator_id=operator_id,
                text=text,
                inject_into_runtime=inject_into_runtime,
                created_at_ms=int(time.time() * 1000.0),
            )
            state.operator_messages.append(msg)
            return msg

    async def list_operator_messages(
        self, *, ai_id: str, session_id: str
    ) -> tuple[OperatorMessage, ...]:
        async with self._lock:
            state = self._states.get((ai_id, session_id))
            if state is None:
                return ()
            return tuple(state.operator_messages)

    async def overview(
        self,
        *,
        ai_id: str | None = None,
        paused: bool | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Read-only snapshot for the conversations listing.

        Filters: ``ai_id`` to a single AI, ``paused`` to True/False.
        Returns a tuple of dicts so the ops handlers can serialise
        them without holding the lock.
        """
        async with self._lock:
            out: list[dict[str, Any]] = []
            for (a, sid), state in self._states.items():
                if ai_id is not None and a != ai_id:
                    continue
                if paused is not None and state.paused is not paused:
                    continue
                out.append(_state_overview_dict(ai_id=a, session_id=sid, state=state))
            return tuple(out)


def _clone_state(state: _SessionState) -> _SessionState:
    return _SessionState(
        paused=state.paused,
        pause_operator_id=state.pause_operator_id,
        pause_note=state.pause_note,
        paused_at_ms=state.paused_at_ms,
        operator_messages=list(state.operator_messages),
    )


def _state_overview_dict(
    *, ai_id: str, session_id: str, state: _SessionState
) -> dict[str, Any]:
    return {
        "ai_id": ai_id,
        "session_id": session_id,
        "paused": state.paused,
        "pause_operator_id": state.pause_operator_id,
        "pause_note": state.pause_note,
        "paused_at_ms": state.paused_at_ms,
        "operator_messages": [m.to_json() for m in state.operator_messages],
    }


def operator_takeover_response_body(
    *,
    ai_id: str,
    session_id: str,
    contract_id: str,
    interaction_type: str,
) -> Mapping[str, Any]:
    """Standard wire body returned by the dispatch handler when paused.

    Mirrors the DLaaS public ``operator_takeover`` shape (see
    ``DLAAS_README.md`` admin section). The dispatch handler returns
    this dict directly without invoking the kernel.
    """
    return {
        "status": "operator_takeover",
        "ai_id": ai_id,
        "session_id": session_id,
        "contract_id": contract_id,
        "interaction_type": interaction_type,
        "operator_takeover": True,
        "output_acts": [
            {
                "act_type": "system",
                "capability": "system_notice",
                "payload": {
                    "content": (
                        "[运营人员已接管本对话，AI 暂停回复，请等待人工回复。]"
                    )
                },
                "degraded": False,
                "original_capability": "",
            }
        ],
        "raw": "",
    }


__all__ = [
    "OperatorMessage",
    "PauseStore",
    "operator_takeover_response_body",
]
