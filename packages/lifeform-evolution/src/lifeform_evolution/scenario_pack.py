"""Scenario-pack loader: load scripted scenarios from JSON files.

A *scenario pack* is one JSON file describing one ``ScriptedScenario``. A
*scenario directory* is a folder containing many such files. This module
lets verticals (lifeform-domain-* packages, product teams, evaluation
researchers) define their own benchmarks **as data** and feed them into
``lifeform-bench`` / ``-trace`` / ``-loop`` / ``-multi-loop`` without
modifying ``lifeform-evolution``.

The JSON schema is intentionally tiny:

```json
{
  "scenario_id": "low-mood-disclosure",
  "description": "Free-form English description...",
  "turns": [
    {
      "user_input": "I have been feeling stuck lately.",
      "expected_regime_in": ["emotional_support", "acquaintance_building"],
      "expected_min_pe_magnitude": 0.0
    }
  ]
}
```

Validation keeps the loader honest: missing ``scenario_id``, missing
``user_input``, malformed turn types, etc. all raise
``ScenarioPackError`` with a precise path so the offending file/turn is
discoverable in CI logs.

The loader is **read-only**. It does not import ``lifeform-domain-*`` or
peek at any kernel state \u2014 it just turns JSON bytes into ``ScriptedScenario``
instances. That keeps the module decoupled from any specific vertical.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Iterable
from dataclasses import asdict
from typing import Any

from lifeform_evolution.benchmark import ScriptedScenario, ScriptedTurn


SCHEMA_VERSION = "vz-scenario-pack.v1"


class ScenarioPackError(ValueError):
    """Raised when a scenario JSON file fails validation."""


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_scenario_pack(path: str | pathlib.Path) -> ScriptedScenario:
    """Load a single ``ScriptedScenario`` from a JSON file.

    Raises ``FileNotFoundError`` if the path doesn't exist, and
    ``ScenarioPackError`` for any structural / type problem.
    """
    file_path = pathlib.Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Scenario pack not found: {file_path}")
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ScenarioPackError(
            f"Scenario pack at {file_path} is not valid JSON: {exc}"
        ) from exc
    return _scenario_from_payload(payload, source=str(file_path))


def load_scenario_pack_dir(
    path: str | pathlib.Path,
    *,
    glob: str = "*.json",
    sort: bool = True,
) -> tuple[ScriptedScenario, ...]:
    """Load every JSON scenario in a directory.

    Args:
        path: directory containing one ``.json`` file per scenario.
        glob: glob pattern relative to ``path`` (default matches ``*.json``).
            Subdirectories are not traversed by default; use a recursive glob
            (``**/*.json``) if you want that behaviour.
        sort: if True (default), scenarios are returned in lexicographic
            order of their file names. This makes runs deterministic across
            machines for the same on-disk pack.

    Returns an empty tuple if the directory exists but contains no matches.
    Raises ``FileNotFoundError`` if the directory itself does not exist, and
    ``ScenarioPackError`` if any file fails validation.
    """
    dir_path = pathlib.Path(path)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Scenario directory not found: {dir_path}")
    files = list(dir_path.glob(glob))
    if sort:
        files.sort()
    scenarios = tuple(load_scenario_pack(f) for f in files if f.is_file())
    return scenarios


def load_scenarios(path: str | pathlib.Path) -> tuple[ScriptedScenario, ...]:
    """Convenience: load a single file OR every JSON file in a directory.

    Use this when a CLI flag accepts either form. The discrimination is by
    ``Path.is_file()`` / ``is_dir()``; symbolic links and special files are
    treated as files.
    """
    p = pathlib.Path(path)
    if p.is_file():
        return (load_scenario_pack(p),)
    if p.is_dir():
        return load_scenario_pack_dir(p)
    raise FileNotFoundError(
        f"Scenarios path is neither a file nor a directory: {p}"
    )


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def dump_scenario_pack(
    scenario: ScriptedScenario,
    path: str | pathlib.Path,
    *,
    indent: int | None = 2,
) -> pathlib.Path:
    """Write a ``ScriptedScenario`` to disk in the v1 JSON shape.

    The on-disk format omits keys whose values are empty tuples / None to
    keep the file readable. ``load_scenario_pack`` round-trips this form.
    """
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_payload_from_scenario(scenario), encoding="utf-8")
    return out.resolve()


def dump_scenario_packs(
    scenarios: Iterable[ScriptedScenario],
    out_dir: str | pathlib.Path,
    *,
    indent: int | None = 2,
) -> tuple[pathlib.Path, ...]:
    """Write each ``ScriptedScenario`` to a separate file in ``out_dir``.

    Filenames are derived from ``scenario_id`` (sanitised \u2014 forward slashes
    and other non-portable characters become hyphens).
    """
    target = pathlib.Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    written: list[pathlib.Path] = []
    for scenario in scenarios:
        name = _sanitise_filename(scenario.scenario_id)
        written.append(dump_scenario_pack(scenario, target / f"{name}.json", indent=indent))
    return tuple(written)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _scenario_from_payload(
    payload: object,
    *,
    source: str,
) -> ScriptedScenario:
    if not isinstance(payload, dict):
        raise ScenarioPackError(
            f"{source}: top-level value must be an object, got {type(payload).__name__}."
        )
    scenario_id = payload.get("scenario_id")
    if not isinstance(scenario_id, str) or not scenario_id.strip():
        raise ScenarioPackError(
            f"{source}: 'scenario_id' must be a non-empty string."
        )
    description = payload.get("description", "")
    if description is None:
        description = ""
    if not isinstance(description, str):
        raise ScenarioPackError(
            f"{source}: 'description' must be a string."
        )
    raw_turns = payload.get("turns")
    if not isinstance(raw_turns, list) or not raw_turns:
        raise ScenarioPackError(
            f"{source}: 'turns' must be a non-empty list."
        )

    turns: list[ScriptedTurn] = []
    for turn_index, raw_turn in enumerate(raw_turns):
        turns.append(
            _turn_from_payload(
                raw_turn,
                source=f"{source}:turns[{turn_index}]",
            )
        )
    return ScriptedScenario(
        scenario_id=scenario_id,
        description=description,
        turns=tuple(turns),
    )


def _turn_from_payload(payload: object, *, source: str) -> ScriptedTurn:
    if not isinstance(payload, dict):
        raise ScenarioPackError(
            f"{source}: turn must be an object, got {type(payload).__name__}."
        )
    user_input = payload.get("user_input")
    if not isinstance(user_input, str) or not user_input.strip():
        raise ScenarioPackError(
            f"{source}: 'user_input' must be a non-empty string."
        )
    raw_expected = payload.get("expected_regime_in", [])
    if raw_expected is None:
        raw_expected = []
    if not isinstance(raw_expected, list) or not all(isinstance(x, str) for x in raw_expected):
        raise ScenarioPackError(
            f"{source}: 'expected_regime_in' must be a list of strings."
        )
    raw_pe = payload.get("expected_min_pe_magnitude")
    if raw_pe is not None and not isinstance(raw_pe, (int, float)):
        raise ScenarioPackError(
            f"{source}: 'expected_min_pe_magnitude' must be a number or null."
        )
    return ScriptedTurn(
        user_input=user_input,
        expected_regime_in=tuple(raw_expected),
        expected_min_pe_magnitude=float(raw_pe) if raw_pe is not None else None,
    )


def _payload_from_scenario(scenario: ScriptedScenario, *, indent: int | None = 2) -> str:
    payload: dict[str, Any] = {
        "scenario_id": scenario.scenario_id,
        "description": scenario.description,
        "turns": [
            _payload_from_turn(turn) for turn in scenario.turns
        ],
    }
    return json.dumps(payload, indent=indent, ensure_ascii=False) + "\n"


def _payload_from_turn(turn: ScriptedTurn) -> dict[str, Any]:
    payload: dict[str, Any] = {"user_input": turn.user_input}
    if turn.expected_regime_in:
        payload["expected_regime_in"] = list(turn.expected_regime_in)
    if turn.expected_min_pe_magnitude is not None:
        payload["expected_min_pe_magnitude"] = turn.expected_min_pe_magnitude
    return payload


_FILENAME_REPLACE = {ord(c): "-" for c in r"/\\:*?<>|\""}


def _sanitise_filename(name: str) -> str:
    cleaned = name.translate(_FILENAME_REPLACE).strip(" .-")
    return cleaned or "scenario"
