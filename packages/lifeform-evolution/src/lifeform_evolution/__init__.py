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
from lifeform_evolution.companion_evidence import (
    CompanionEvidenceGate,
    CompanionEvidenceReport,
    companion_evidence_report_to_dict,
    format_companion_evidence_report,
    run_companion_evidence,
    run_companion_evidence_async,
)
from lifeform_evolution.family_report import (
    FamilyEvaluation,
    FamilyId,
    FamilyMetric,
    FamilyReport,
    compute_family_report,
    family_report_to_dict,
    format_family_report,
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
    RoundDeltaVsBaseline,
    RoundQualityMetrics,
    RoundReport,
    format_multi_round_report,
    run_multi_round_loop,
    run_multi_round_loop_async,
)
from lifeform_evolution.regime_calibrator import (
    RegimeCalibrationReport,
    RegimeCalibrationRoundReport,
    format_regime_calibration_report,
    run_regime_calibrator,
    run_regime_calibrator_async,
)
from lifeform_evolution.regime_io import (
    RegimeBootstrapArtifact,
    load_regime_bootstrap,
    load_regime_bootstrap_only,
    save_regime_bootstrap,
)
from lifeform_evolution.social_cognition_evidence import (
    SocialCognitionEvidenceGate,
    SocialCognitionEvidenceReport,
    format_social_cognition_evidence_report,
    run_social_cognition_evidence,
    run_social_cognition_evidence_async,
    social_cognition_evidence_report_to_dict,
)
from lifeform_evolution.super_loop import (
    SuperLoopReport,
    SuperLoopRoundReport,
    format_super_loop_report,
    run_super_loop,
    run_super_loop_async,
)
from lifeform_evolution.scenario_pack import (
    ScenarioPackError,
    dump_scenario_pack,
    dump_scenario_packs,
    load_scenario_pack,
    load_scenario_pack_dir,
    load_scenarios,
)
from lifeform_evolution.snapshot_io import (
    SnapshotArtifact,
    load_snapshot,
    load_snapshot_only,
    save_snapshot,
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
    "CompanionEvidenceGate",
    "CompanionEvidenceReport",
    "DistributionSnapshot",
    "FamilyEvaluation",
    "FamilyId",
    "FamilyMetric",
    "FamilyReport",
    "LearningLoopReport",
    "MultiRoundLearningLoopReport",
    "RegimeBootstrapArtifact",
    "RegimeCalibrationReport",
    "RegimeCalibrationRoundReport",
    "RoundDeltaVsBaseline",
    "RoundQualityMetrics",
    "RoundReport",
    "ScenarioPackError",
    "ScriptedScenario",
    "ScriptedTurn",
    "SnapshotArtifact",
    "SSLDemoReport",
    "SocialCognitionEvidenceGate",
    "SocialCognitionEvidenceReport",
    "SuperLoopReport",
    "SuperLoopRoundReport",
    "TraceCollector",
    "TraceScenarioReport",
    "TraceTurnRecord",
    "TurnReport",
    "all_built_in_scenarios",
    "casual_social_checkin_scenario",
    "compute_family_report",
    "companion_evidence_report_to_dict",
    "dump_scenario_pack",
    "dump_scenario_packs",
    "family_report_to_dict",
    "format_family_report",
    "format_companion_evidence_report",
    "format_learning_loop_report",
    "format_multi_round_report",
    "format_regime_calibration_report",
    "format_report",
    "format_social_cognition_evidence_report",
    "format_ssl_demo_report",
    "format_super_loop_report",
    "load_regime_bootstrap",
    "load_regime_bootstrap_only",
    "load_scenario_pack",
    "load_scenario_pack_dir",
    "load_scenarios",
    "load_snapshot",
    "load_snapshot_only",
    "low_mood_disclosure_scenario",
    "run_benchmark",
    "run_benchmark_async",
    "run_companion_evidence",
    "run_companion_evidence_async",
    "run_learning_loop",
    "run_learning_loop_async",
    "run_multi_round_loop",
    "run_multi_round_loop_async",
    "run_regime_calibrator",
    "run_regime_calibrator_async",
    "run_social_cognition_evidence",
    "run_social_cognition_evidence_async",
    "run_ssl_demo",
    "run_ssl_demo_from_ndjson",
    "run_super_loop",
    "run_super_loop_async",
    "save_regime_bootstrap",
    "save_snapshot",
    "social_cognition_evidence_report_to_dict",
    "trace_record_to_training_trace",
    "trace_records_from_ndjson",
    "trace_records_to_training_dataset",
    "trust_rupture_repair_scenario",
)
