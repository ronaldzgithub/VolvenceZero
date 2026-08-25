"""Lifeform evolution public API with side-effect-free lazy exports.

Importing :mod:`lifeform_evolution` must not import model-capable runtime
modules. Public attributes keep their historical locations, but each owner
module is imported only when that attribute is requested.
"""

from __future__ import annotations

from importlib import import_module


_LAZY_EXPORT_GROUPS = (
    (
        "lifeform_evolution.benchmark",
        (
            "BenchmarkReport",
            "ScriptedScenario",
            "ScriptedTurn",
            "TurnReport",
            "all_built_in_scenarios",
            "casual_social_checkin_scenario",
            "emotional_decision_support_scenario",
            "format_report",
            "low_mood_disclosure_scenario",
            "run_benchmark",
            "run_benchmark_async",
            "trust_rupture_repair_scenario",
        ),
    ),
    (
        "lifeform_evolution.companion_evidence",
        (
            "CompanionEvidenceGate",
            "CompanionEvidenceReport",
            "companion_evidence_report_to_dict",
            "format_companion_evidence_report",
            "run_companion_evidence",
            "run_companion_evidence_async",
        ),
    ),
    (
        "lifeform_evolution.closed_alpha_preflight",
        (
            "ClosedAlphaPreflightReport",
            "format_closed_alpha_preflight_report",
            "run_closed_alpha_preflight",
        ),
    ),
    (
        "lifeform_evolution.family_report",
        (
            "FamilyEvaluation",
            "FamilyId",
            "FamilyMetric",
            "FamilyReport",
            "compute_family_report",
            "family_report_to_dict",
            "format_family_report",
        ),
    ),
    (
        "lifeform_evolution.dataset_adapter",
        (
            "trace_record_to_training_trace",
            "trace_records_from_ndjson",
            "trace_records_to_training_dataset",
        ),
    ),
    (
        "lifeform_evolution.learning_loop",
        (
            "DistributionSnapshot",
            "LearningLoopReport",
            "format_learning_loop_report",
            "run_learning_loop",
            "run_learning_loop_async",
        ),
    ),
    (
        "lifeform_evolution.multi_round_loop",
        (
            "MultiRoundLearningLoopReport",
            "RoundDeltaVsBaseline",
            "RoundQualityMetrics",
            "RoundReport",
            "format_multi_round_report",
            "run_multi_round_loop",
            "run_multi_round_loop_async",
        ),
    ),
    (
        "lifeform_evolution.regime_calibrator",
        (
            "RegimeCalibrationReport",
            "RegimeCalibrationRoundReport",
            "format_regime_calibration_report",
            "run_regime_calibrator",
            "run_regime_calibrator_async",
        ),
    ),
    (
        "lifeform_evolution.regime_io",
        (
            "RegimeBootstrapArtifact",
            "load_regime_bootstrap",
            "load_regime_bootstrap_only",
            "save_regime_bootstrap",
        ),
    ),
    (
        "lifeform_evolution.relationship_repair_alpha_gate",
        (
            "RepairAlphaArmReport",
            "RepairAlphaGateReport",
            "format_relationship_repair_alpha_report",
            "run_relationship_repair_alpha_gate",
            "run_relationship_repair_alpha_gate_async",
        ),
    ),
    (
        "lifeform_evolution.relationship_assistant_pilot",
        (
            "PilotDayEvidence",
            "PilotTranscriptTurn",
            "RelationshipAssistantPilotHarness",
        ),
    ),
    (
        "lifeform_evolution.seven_day_companion",
        (
            "HTTPSevenDayCompanionService",
            "ProcessRestartEvidence",
            "SevenDayCompanionOrchestrator",
            "SevenDayCompanionRun",
            "SevenDayDayEvidence",
            "SevenDayScenarioSchedule",
            "SevenDayScheduleDay",
            "SevenDayTurnEvidence",
            "SimulatedSourceAttestation",
            "SimulatedUserTurn",
            "StateInterventionEvidence",
        ),
    ),
    (
        "lifeform_evolution.seven_day_state_control",
        (
            "SEVEN_DAY_SHUFFLED_SOURCE_DAYS",
            "SevenDayFilesystemStateController",
        ),
    ),
    (
        "lifeform_evolution.seven_day_process_host",
        (
            "ServiceProcessStart",
            "ServiceProcessStop",
            "StateControlledSubprocessLifecycle",
            "SubprocessSevenDayServiceHost",
        ),
    ),
    (
        "lifeform_evolution.social_cognition_evidence",
        (
            "SocialCognitionEvidenceGate",
            "SocialCognitionEvidenceReport",
            "format_social_cognition_evidence_report",
            "run_social_cognition_evidence",
            "run_social_cognition_evidence_async",
            "social_cognition_evidence_report_to_dict",
        ),
    ),
    (
        "lifeform_evolution.super_loop",
        (
            "SuperLoopReport",
            "SuperLoopRoundReport",
            "format_super_loop_report",
            "run_super_loop",
            "run_super_loop_async",
        ),
    ),
    (
        "lifeform_evolution.scenario_pack",
        (
            "ScenarioPackError",
            "dump_scenario_pack",
            "dump_scenario_packs",
            "load_scenario_pack",
            "load_scenario_pack_dir",
            "load_scenarios",
        ),
    ),
    (
        "lifeform_evolution.semantic_proposal_ablation",
        (
            "DEFAULT_ABLATION_PROBE_CASES",
            "AblationArmResult",
            "AblationProbeCase",
            "AblationProbeTurn",
            "ScriptedProposalSpec",
            "ScriptedSemanticProposalRuntime",
            "SemanticProposalAblationError",
            "SemanticProposalAblationReport",
            "SemanticProposalAblationRunResult",
            "SlotExpectation",
            "run_semantic_proposal_ablation_async",
        ),
    ),
    (
        "lifeform_evolution.snapshot_io",
        (
            "SnapshotArtifact",
            "load_snapshot",
            "load_snapshot_only",
            "save_snapshot",
        ),
    ),
    (
        "lifeform_evolution.ssl_demo",
        (
            "SSLDemoReport",
            "format_ssl_demo_report",
            "run_ssl_demo",
            "run_ssl_demo_from_ndjson",
        ),
    ),
    (
        "lifeform_evolution.trace_collector",
        (
            "TraceCollector",
            "TraceScenarioReport",
            "TraceTurnRecord",
        ),
    ),
)

_LAZY_EXPORTS = {
    export_name: module_name for module_name, export_names in _LAZY_EXPORT_GROUPS for export_name in export_names
}
__all__ = tuple(_LAZY_EXPORTS)


def __getattr__(name: str) -> object:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
