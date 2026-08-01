from __future__ import annotations

import json
from pathlib import Path

import pytest

from companion_bench.seven_day_driver import (
    FrozenSevenDayUserDriver,
    build_frozen_seven_day_user_script,
    load_frozen_seven_day_user_script,
)
from companion_bench.spec import load_scenario_yaml
SCENARIO = (
    Path(__file__).resolve().parents[1]
    / "src/companion_bench/scenarios/seven_day/"
    / "F1-seven-day-warmth-researcher.yaml"
)


class _StyleFake:
    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        seed: int,
    ) -> str:
        del system_prompt, user_prompt, temperature
        return ("Honestly,", "Right now,")[seed % 2]


def _script():
    return build_frozen_seven_day_user_script(
        spec=load_scenario_yaml(SCENARIO),
        paraphrase_seed=1401,
        backend=_StyleFake(),
    )


def test_script_is_exact_and_byte_deterministic() -> None:
    first = _script()
    second = _script()
    assert len(first.turns) == 35
    assert first.script_sha256 == second.script_sha256
    assert first.turns == second.turns
    assert {
        tag for turn in first.turns for tag in turn.event_tags
    } == {"callback", "emotion", "boundary"}


def test_driver_ignores_arm_specific_assistant_history() -> None:
    first = FrozenSevenDayUserDriver(_script())
    second = FrozenSevenDayUserDriver(_script())
    for day_index in range(1, 8):
        for exchange_index in range(1, 6):
            left = first.next_turn(
                day_index=day_index,
                exchange_index=exchange_index,
                recent_assistant_turns=("left arm",),
            )
            right = second.next_turn(
                day_index=day_index,
                exchange_index=exchange_index,
                recent_assistant_turns=("right arm",),
            )
            assert left == right


def test_driver_fails_on_coordinate_drift() -> None:
    driver = FrozenSevenDayUserDriver(_script())
    with pytest.raises(RuntimeError, match="coordinate drift"):
        driver.next_turn(
            day_index=2,
            exchange_index=1,
            recent_assistant_turns=(),
        )


def test_frozen_script_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    script = _script()
    path = tmp_path / "script.json"
    path.write_text(
        json.dumps(script.to_json(), ensure_ascii=False),
        encoding="utf-8",
    )
    assert load_frozen_seven_day_user_script(path) == script

    payload = script.to_json()
    payload["turns"][0]["text"] = "I tampered with this."
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="digest drift"):
        load_frozen_seven_day_user_script(path)
