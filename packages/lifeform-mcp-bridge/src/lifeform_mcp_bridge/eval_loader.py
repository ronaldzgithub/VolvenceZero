"""Filesystem loader for ``eval-scenarios/*.json`` from external bundles.

Per ``docs/specs/mcp-bridge.md`` § "Eval scenarios", evaluation
scenarios are NOT MCP RPC — they are static JSON artifacts shipped
in the external repo. This loader walks ``<repo_root>/eval-scenarios``
(by default), parses each ``*.json`` file, validates the minimal
shape, and returns a tuple of ``MCPEvalScenario`` records.

The loader does NOT couple to ``lifeform-evolution``'s benchmark
runner directly — that wheel can decide how to consume these
records (e.g. wrap them in its own ``BenchmarkScenario`` shape).
This keeps the bridge wheel free of a dependency on the bench
infrastructure.

Validation rules (minimal so external repos can iterate fast):

* File must be valid UTF-8 JSON
* Top-level must be an object
* ``name`` must be a non-empty string
* ``turns`` must be a non-empty list of strings (one per simulated
  user turn) OR a list of objects with at least ``input`` field
* All other fields pass through verbatim
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_LOG = logging.getLogger("lifeform_mcp_bridge.eval_loader")


class EvalScenarioLoadError(Exception):
    """Raised when an eval scenario JSON file is structurally invalid.

    Distinct from ``MCPBridgeError`` because eval loading does not
    cross the MCP wire — it is local filesystem reading. Catching
    this separately lets a caller skip a bad scenario without
    aborting the bundle bringup.
    """


@dataclass(frozen=True)
class MCPEvalScenario:
    """One loaded scenario, validated.

    ``raw`` carries every field the JSON file declared so consumers
    that understand richer schemas (lifeform-evolution benchmarks,
    companion-bench harness) can read what they need without going
    back to disk.
    """

    name: str
    source_path: str
    description: str
    turns: tuple[Any, ...]
    raw: Mapping[str, Any]


def load_scenarios(
    *,
    repo_root: str | Path,
    subdirectory: str = "eval-scenarios",
) -> tuple[MCPEvalScenario, ...]:
    """Walk ``<repo_root>/<subdirectory>/*.json`` and return scenarios.

    Missing directory returns an empty tuple (the external repo may
    not ship eval scenarios). Per-file load errors raise
    ``EvalScenarioLoadError`` with the offending path; the caller
    decides whether to abort.
    """
    root = Path(repo_root) / subdirectory
    if not root.is_dir():
        return ()
    scenarios: list[MCPEvalScenario] = []
    for path in sorted(root.rglob("*.json")):
        scenarios.append(_load_one(path))
    return tuple(scenarios)


def _load_one(path: Path) -> MCPEvalScenario:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvalScenarioLoadError(
            f"could not read eval scenario {str(path)!r}: {exc}"
        ) from exc
    try:
        document = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise EvalScenarioLoadError(
            f"eval scenario {str(path)!r} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(document, Mapping):
        raise EvalScenarioLoadError(
            f"eval scenario {str(path)!r}: top-level JSON must be an "
            f"object; got {type(document).__name__}."
        )
    name = document.get("name")
    if not isinstance(name, str) or not name.strip():
        raise EvalScenarioLoadError(
            f"eval scenario {str(path)!r}: 'name' must be a non-empty "
            f"string."
        )
    description = str(document.get("description", ""))
    raw_turns = document.get("turns", [])
    if not isinstance(raw_turns, list) or not raw_turns:
        raise EvalScenarioLoadError(
            f"eval scenario {str(path)!r}: 'turns' must be a "
            f"non-empty list."
        )
    for index, turn in enumerate(raw_turns):
        if isinstance(turn, str):
            if not turn.strip():
                raise EvalScenarioLoadError(
                    f"eval scenario {str(path)!r}: turns[{index}] is "
                    f"an empty string."
                )
        elif isinstance(turn, Mapping):
            if not isinstance(turn.get("input"), str) or not turn.get("input").strip():
                raise EvalScenarioLoadError(
                    f"eval scenario {str(path)!r}: turns[{index}] is "
                    f"a mapping missing a non-empty 'input' field."
                )
        else:
            raise EvalScenarioLoadError(
                f"eval scenario {str(path)!r}: turns[{index}] must be "
                f"a string or mapping; got {type(turn).__name__}."
            )
    return MCPEvalScenario(
        name=name,
        source_path=str(path),
        description=description,
        turns=tuple(raw_turns),
        raw=dict(document),
    )


__all__ = [
    "EvalScenarioLoadError",
    "MCPEvalScenario",
    "load_scenarios",
]
