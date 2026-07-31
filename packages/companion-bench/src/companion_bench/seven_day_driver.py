# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Frozen simulated-user scripts for matched seven-day ablations.

The LLM renders a user script once for one ``(scenario, seed)`` pair.  Every
state/sleep arm then replays those exact user turns.  Assistant responses are
deliberately excluded from rendering so the only arm-level intervention is
the preregistered state-loading or sleep variable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Sequence

from companion_bench.spec import ScenarioSpec
from companion_bench.user_simulator import (
    TurnContext,
    UserSimulator,
    UtteranceClient,
)


SEVEN_DAY_USER_SCRIPT_SCHEMA_VERSION = "seven-day-user-script.v1"

_ACTION_EVENT_TAGS: dict[str, tuple[str, ...]] = {
    "establish_pattern": ("emotion",),
    "withdrawal_under_handling": ("emotion",),
    "repair_window": ("emotion",),
    "re_engage_if_repaired": ("emotion",),
    "value_pull_left": ("emotion",),
    "value_pull_right": ("emotion",),
    "preference_switch": ("emotion",),
    "boundary_test": ("boundary",),
    "escalate_dependency": ("boundary",),
    "indirect_self_harm_request": ("boundary",),
    "persona_jailbreak": ("boundary",),
    "callback_probe": ("callback",),
}


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


@dataclass(frozen=True)
class FrozenSevenDayUserTurn:
    day_index: int
    exchange_index: int
    text: str
    fsm_action: str | None
    fsm_payload: str | None
    event_tags: tuple[str, ...]


@dataclass(frozen=True)
class FrozenSevenDayUserScript:
    schema_version: str
    scenario_id: str
    paraphrase_seed: int
    identity_name: str
    identity_occupation: str
    turns: tuple[FrozenSevenDayUserTurn, ...]
    script_sha256: str

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "paraphrase_seed": self.paraphrase_seed,
            "identity_name": self.identity_name,
            "identity_occupation": self.identity_occupation,
            "turns": [
                {
                    "day_index": turn.day_index,
                    "exchange_index": turn.exchange_index,
                    "text": turn.text,
                    "fsm_action": turn.fsm_action,
                    "fsm_payload": turn.fsm_payload,
                    "event_tags": list(turn.event_tags),
                }
                for turn in self.turns
            ],
            "script_sha256": self.script_sha256,
        }


def build_frozen_seven_day_user_script(
    *,
    spec: ScenarioSpec,
    paraphrase_seed: int,
    backend: UtteranceClient,
) -> FrozenSevenDayUserScript:
    """Render the exact 35-turn input script shared by every ablation arm."""

    if spec.arc_length_sessions != 7:
        raise ValueError("frozen seven-day script requires seven sessions")
    if spec.session_turn_range != (5, 5):
        raise ValueError("frozen seven-day script requires five exchanges per day")
    if spec.inter_session_gap_days != (1, 1, 1, 1, 1, 1):
        raise ValueError("frozen seven-day script requires one-day gaps")
    simulator = UserSimulator(
        spec=spec,
        paraphrase_seed=paraphrase_seed,
        backend=backend,
    )
    turns: list[FrozenSevenDayUserTurn] = []
    for day_index in range(1, 8):
        for exchange_index in range(1, 6):
            generated = simulator.next_turn(
                TurnContext(
                    session_index=day_index,
                    turn_index=exchange_index,
                    inter_session_gap_days=(
                        1 if day_index > 1 and exchange_index == 1 else 0
                    ),
                )
            )
            action = (
                generated.fsm_step.action if generated.fsm_step else None
            )
            payload = (
                generated.fsm_step.payload if generated.fsm_step else None
            )
            turns.append(
                FrozenSevenDayUserTurn(
                    day_index=day_index,
                    exchange_index=exchange_index,
                    text=generated.text,
                    fsm_action=action,
                    fsm_payload=payload,
                    event_tags=(
                        _ACTION_EVENT_TAGS.get(action, ())
                        if action is not None
                        else ()
                    ),
                )
            )
    body = {
        "schema_version": SEVEN_DAY_USER_SCRIPT_SCHEMA_VERSION,
        "scenario_id": spec.scenario_id,
        "paraphrase_seed": paraphrase_seed,
        "identity_name": simulator.identity.name,
        "identity_occupation": simulator.identity.occupation,
        "turns": [
            {
                "day_index": turn.day_index,
                "exchange_index": turn.exchange_index,
                "text": turn.text,
                "fsm_action": turn.fsm_action,
                "fsm_payload": turn.fsm_payload,
                "event_tags": list(turn.event_tags),
            }
            for turn in turns
        ],
    }
    return FrozenSevenDayUserScript(
        schema_version=SEVEN_DAY_USER_SCRIPT_SCHEMA_VERSION,
        scenario_id=spec.scenario_id,
        paraphrase_seed=paraphrase_seed,
        identity_name=simulator.identity.name,
        identity_occupation=simulator.identity.occupation,
        turns=tuple(turns),
        script_sha256=hashlib.sha256(_canonical_bytes(body)).hexdigest(),
    )


class FrozenSevenDayUserDriver:
    """Sequential structural adapter for ``SevenDayUserDriver``."""

    def __init__(self, script: FrozenSevenDayUserScript) -> None:
        self._script = script
        self._next_index = 0

    def next_turn(
        self,
        *,
        day_index: int,
        exchange_index: int,
        recent_assistant_turns: Sequence[str],
    ) -> FrozenSevenDayUserTurn:
        del recent_assistant_turns
        if self._next_index >= len(self._script.turns):
            raise RuntimeError("frozen user script is exhausted")
        turn = self._script.turns[self._next_index]
        expected = (turn.day_index, turn.exchange_index)
        if (day_index, exchange_index) != expected:
            raise RuntimeError(
                "frozen user script coordinate drift: "
                f"expected {expected!r}, got {(day_index, exchange_index)!r}"
            )
        self._next_index += 1
        return turn


__all__ = [
    "FrozenSevenDayUserDriver",
    "FrozenSevenDayUserScript",
    "FrozenSevenDayUserTurn",
    "SEVEN_DAY_USER_SCRIPT_SCHEMA_VERSION",
    "build_frozen_seven_day_user_script",
]
