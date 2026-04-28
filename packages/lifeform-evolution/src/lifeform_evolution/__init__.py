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
from lifeform_evolution.dataset_adapter import (
    trace_record_to_training_trace,
    trace_records_from_ndjson,
    trace_records_to_training_dataset,
)
from lifeform_evolution.learning_loop import (
    DistributionSnapshot,
    LearningLoopReport,
    format_learning_loop_report,
    run_learning_loop,
    run_learning_loop_async,
)
from lifeform_evolution.multi_round_loop import (
    MultiRoundLearningLoopReport,
    RoundReport,
    format_multi_round_report,
    run_multi_round_loop,
    run_multi_round_loop_async,
)
from lifeform_evolution.ssl_demo import (
    SSLDemoReport,
    format_ssl_demo_report,
    run_ssl_demo,
    run_ssl_demo_from_ndjson,
)
from lifeform_evolution.trace_collector import (
    TraceCollector,
    TraceScenarioReport,
    TraceTurnRecord,
)

__all__ = (
    "BenchmarkReport",
    "DistributionSnapshot",
    "LearningLoopReport",
    "MultiRoundLearningLoopReport",
    "RoundReport",
    "ScriptedScenario",
    "ScriptedTurn",
    "SSLDemoReport",
    "TraceCollector",
    "TraceScenarioReport",
    "TraceTurnRecord",
    "TurnReport",
    "all_built_in_scenarios",
    "casual_social_checkin_scenario",
    "format_learning_loop_report",
    "format_multi_round_report",
    "format_report",
    "format_ssl_demo_report",
    "low_mood_disclosure_scenario",
    "run_benchmark",
    "run_benchmark_async",
    "run_learning_loop",
    "run_learning_loop_async",
    "run_multi_round_loop",
    "run_multi_round_loop_async",
    "run_ssl_demo",
    "run_ssl_demo_from_ndjson",
    "trace_record_to_training_trace",
    "trace_records_from_ndjson",
    "trace_records_to_training_dataset",
    "trust_rupture_repair_scenario",
)
