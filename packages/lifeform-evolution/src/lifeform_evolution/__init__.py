"""Lifeform evolution layer — scripted benchmarks + evidence dashboards.

Public API:

* ``ScriptedScenario``, ``ScriptedTurn``, ``BenchmarkReport``, ``TurnReport`` —
  benchmark types.
* ``low_mood_disclosure_scenario`` — built-in scenario for sanity testing.
* ``run_benchmark`` / ``run_benchmark_async`` — runner entry points.
* ``format_report`` — pretty-print a report for CLI output.

The CLI entry point is ``lifeform-bench`` (see ``lifeform_evolution.cli``).
"""

from __future__ import annotations

from lifeform_evolution.benchmark import (
    BenchmarkReport,
    ScriptedScenario,
    ScriptedTurn,
    TurnReport,
    all_built_in_scenarios,
    casual_social_checkin_scenario,
    format_report,
    low_mood_disclosure_scenario,
    run_benchmark,
    run_benchmark_async,
    trust_rupture_repair_scenario,
)
from lifeform_evolution.trace_collector import (
    TraceCollector,
    TraceScenarioReport,
    TraceTurnRecord,
)

__all__ = (
    "BenchmarkReport",
    "ScriptedScenario",
    "ScriptedTurn",
    "TraceCollector",
    "TraceScenarioReport",
    "TraceTurnRecord",
    "TurnReport",
    "all_built_in_scenarios",
    "casual_social_checkin_scenario",
    "format_report",
    "low_mood_disclosure_scenario",
    "run_benchmark",
    "run_benchmark_async",
    "trust_rupture_repair_scenario",
)
