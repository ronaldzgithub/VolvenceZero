from __future__ import annotations

import asyncio
import json
from dataclasses import replace

from volvence_zero.agent import (
    AgentSessionRunner,
    DEFAULT_DIALOGUE_CASE_VARIANTS,
    DEFAULT_DIALOGUE_PARAPHRASE_FAMILIES,
    DEFAULT_DIALOGUE_REPLAY_SEEDS,
    DEFAULT_DIALOGUE_PROOF_CASES,
    DEFAULT_OPEN_DIALOGUE_SCENARIOS,
    DEFAULT_RARE_HEAVY_CANDIDATE_CONFIGS,
    OpenDialogueEpisodeState,
    OpenDialogueScenario,
    DialogueArtifactAcceptanceGateConfig,
    DialogueComprehensiveStage,
    DialogueNLEssenceAcceptanceConfig,
    DialogueNLEssenceAcceptanceDecision,
    DialogueNLEssenceAssessmentReport,
    DialogueBenchmarkReport,
    DialogueRealComprehensiveBenchmarkConfig,
    DialogueSharedRunnerFactories,
    evaluate_dialogue_nl_essence_acceptance,
    build_dialogue_nl_essence_assessment,
    build_dialogue_emergence_dashboard,
    build_dialogue_emergence_dashboard_payload,
    build_pe_dominance_case_diagnosis_report,
    build_pe_dominance_comparison_report,
    build_replay_selection_training_traces,
    build_dialogue_replay_selection_artifact,
    build_real_dialogue_comprehensive_runner_factories,
    build_standard_dialogue_runner,
    default_rare_heavy_candidate_configs,
    default_dialogue_ablation_profiles,
    default_dialogue_strong_proof_profiles,
    default_dialogue_comprehensive_profiles,
    default_open_dialogue_ablation_profiles,
    default_dialogue_real_proof_config,
    dialogue_case_variants,
    dialogue_paraphrase_families,
    open_dialogue_scenarios,
    DialogueBenchmarkTurn,
    ScriptedDialogueCase,
    build_deterministic_user_simulator,
    build_dialogue_case_report,
    build_open_dialogue_case_report,
    build_dialogue_replay_ranking_report,
    get_open_dialogue_scenario,
    generate_stochastic_dialogue_case_variants,
    run_replay_selection_artifact_acceptance_benchmark,
    run_dialogue_pe_eta_ablation_benchmark,
    run_dialogue_pe_eta_benchmark,
    run_dialogue_pe_eta_case,
    run_dialogue_pe_eta_comprehensive_benchmark,
    run_dialogue_pe_eta_longitudinal_benchmark,
    run_dialogue_pe_eta_perturbation_benchmark,
    run_dialogue_pe_eta_systematic_replay_benchmark,
    run_open_dialogue_benchmark,
    run_open_dialogue_ablation_benchmark,
    run_open_dialogue_case,
    run_real_dialogue_pe_eta_comprehensive_benchmark,
    run_real_dialogue_pe_eta_comprehensive_benchmark_staged,
    run_multi_artifact_acceptance_benchmark,
    train_rare_heavy_artifact_from_replay_selection,
    export_dialogue_emergence_dashboard_artifact,
)
from volvence_zero.joint_loop import JointLoopSchedule
from volvence_zero.substrate import LocalSubstrateRuntimeMode, SyntheticOpenWeightResidualRuntime
from volvence_zero.agent.dialogue_benchmark import evaluate_dialogue_artifact_acceptance


def _synthetic_runner(case: ScriptedDialogueCase) -> AgentSessionRunner:
    runtime = SyntheticOpenWeightResidualRuntime(model_id=f"dialogue-proof:{case.case_id}")
    runtime.runtime_origin = "synthetic-proof"
    return AgentSessionRunner(
        session_id=f"dialogue-proof:{case.case_id}",
        default_residual_runtime=runtime,
        joint_schedule=JointLoopSchedule(ssl_interval=1, rl_interval=2),
    )


def _synthetic_ablation_runner(profile_label: str, case: ScriptedDialogueCase) -> AgentSessionRunner:
    runtime = SyntheticOpenWeightResidualRuntime(model_id=f"dialogue-ablation:{profile_label}:{case.case_id}")
    runtime.runtime_origin = f"synthetic-{profile_label}"
    return build_standard_dialogue_runner(
        profile_label=profile_label,
        case=case,
        residual_runtime=runtime,
    )


def _synthetic_perturbation_runner(profile_label: str, variant) -> AgentSessionRunner:
    runtime = SyntheticOpenWeightResidualRuntime(
        model_id=f"dialogue-perturb:{profile_label}:{variant.case.case_id}"
    )
    runtime.runtime_origin = f"synthetic-perturb-{profile_label}"
    return build_standard_dialogue_runner(
        profile_label=profile_label,
        case=variant.case,
        residual_runtime=runtime,
    )


def _synthetic_systematic_runner(profile_label: str, variant) -> AgentSessionRunner:
    runtime = SyntheticOpenWeightResidualRuntime(
        model_id=f"dialogue-systematic:{profile_label}:{variant.case.case_id}"
    )
    runtime.runtime_origin = f"synthetic-systematic-{profile_label}"
    return build_standard_dialogue_runner(
        profile_label=profile_label,
        case=variant.case,
        residual_runtime=runtime,
    )


def _synthetic_acceptance_runner(variant) -> AgentSessionRunner:
    runtime = SyntheticOpenWeightResidualRuntime(model_id="dialogue-acceptance:shared")
    runtime.runtime_origin = "synthetic-acceptance"
    return build_standard_dialogue_runner(
        profile_label="pe-eta",
        case=variant.case,
        residual_runtime=runtime,
    )


def _synthetic_open_ablation_runner(profile_label: str, scenario: OpenDialogueScenario) -> AgentSessionRunner:
    runtime = SyntheticOpenWeightResidualRuntime(model_id=f"open-dialogue:{profile_label}:{scenario.scenario_id}")
    runtime.runtime_origin = f"synthetic-open-{profile_label}"
    return build_standard_dialogue_runner(
        profile_label=profile_label,
        case=ScriptedDialogueCase(
            case_id=f"open-dialogue:{scenario.scenario_id}",
            description=scenario.description,
            user_inputs=scenario.opening_turns,
        ),
        residual_runtime=runtime,
    )


def _synthetic_shared_factories() -> DialogueSharedRunnerFactories:
    runtime = SyntheticOpenWeightResidualRuntime(model_id="dialogue-staged-shared")
    runtime.runtime_origin = "synthetic-staged-shared"
    return DialogueSharedRunnerFactories(
        residual_runtime=runtime,
        canonical_runner_factory=lambda profile_label, case: build_standard_dialogue_runner(
            profile_label=profile_label,
            case=case,
            residual_runtime=runtime,
        ),
        perturbation_runner_factory=lambda profile_label, variant: build_standard_dialogue_runner(
            profile_label=profile_label,
            case=variant.case,
            residual_runtime=runtime,
        ),
        open_runner_factory=lambda profile_label, scenario: build_standard_dialogue_runner(
            profile_label=profile_label,
            case=ScriptedDialogueCase(
                case_id=f"open-dialogue:{scenario.scenario_id}",
                description=scenario.description,
                user_inputs=scenario.opening_turns,
            ),
            residual_runtime=runtime,
        ),
        systematic_runner_factory=lambda profile_label, variant: build_standard_dialogue_runner(
            profile_label=profile_label,
            case=variant.case,
            residual_runtime=runtime,
        ),
        acceptance_runner_factory=lambda variant: build_standard_dialogue_runner(
            profile_label="pe-eta",
            case=variant.case,
            residual_runtime=runtime,
        ),
        description="synthetic shared benchmark factories",
    )


def _benchmark_turn(
    *,
    turn_index: int,
    pe: float,
    reward: float,
    action: str,
    regime: str,
    abstract_action: str,
    switch_gate: float,
    delayed_metric: float,
    pe_triggered: bool = False,
    action_family_version: int = 0,
    bounded_writeback_applied: bool = False,
    reflection_promotion_eligible: bool = False,
    session_post_completed_job_count: int = 0,
    evolution_decision: str | None = None,
    evolution_category: str | None = None,
    cross_session_verdict: str = "",
    nested_profile_active: bool = True,
    nested_context_reset_applied: bool = False,
    nested_context_reset_total_count: int = 0,
    slow_to_fast_init_benefit: float = 0.0,
    slow_to_fast_target_distance_before: float = 0.0,
    slow_to_fast_target_distance_after: float = 0.0,
    slow_to_fast_target_alignment_gain: float = 0.0,
) -> DialogueBenchmarkTurn:
    return DialogueBenchmarkTurn(
        turn_index=turn_index,
        wave_id=f"wave-{turn_index}",
        user_input=f"turn-{turn_index}",
        acceptance_passed=True,
        active_regime=regime,
        active_abstract_action=abstract_action,
        joint_schedule_action=action,
        switch_gate=switch_gate,
        action_family_version=action_family_version,
        prediction_error_magnitude=pe,
        prediction_error_reward=reward,
        task_error=pe * 0.5,
        relationship_error=pe * 0.3,
        regime_error=pe * 0.2,
        action_error=pe * 0.4,
        has_prediction_chain=True,
        bounded_writeback_applied=bounded_writeback_applied,
        reflection_promotion_eligible=reflection_promotion_eligible,
        session_post_completed_job_count=session_post_completed_job_count,
        rare_heavy_recommended=pe_triggered,
        rare_heavy_applied=False,
        evolution_decision=evolution_decision,
        evolution_category=evolution_category,
        cross_session_verdict=cross_session_verdict,
        nested_profile_active=nested_profile_active,
        nested_context_reset_applied=nested_context_reset_applied,
        nested_context_reset_total_count=nested_context_reset_total_count,
        slow_to_fast_init_benefit=slow_to_fast_init_benefit,
        outcome_metrics=(
            ("learning:joint_learning_progress", delayed_metric),
            ("relationship:cross_track_stability", delayed_metric),
        ),
        description="synthetic benchmark turn",
        slow_to_fast_target_distance_before=slow_to_fast_target_distance_before,
        slow_to_fast_target_distance_after=slow_to_fast_target_distance_after,
        slow_to_fast_target_alignment_gain=slow_to_fast_target_alignment_gain,
    )


def test_dialogue_benchmark_exposes_default_scripted_cases():
    case_ids = tuple(case.case_id for case in DEFAULT_DIALOGUE_PROOF_CASES)

    assert case_ids == ("repair", "task_clarification", "repeated_failure", "goal_drift")
    assert all(len(case.user_inputs) >= 6 for case in DEFAULT_DIALOGUE_PROOF_CASES)


def test_dialogue_benchmark_exposes_default_ablation_profiles():
    assert default_dialogue_ablation_profiles() == ("pe-eta", "pe-drive-off", "eta-off", "timescale-off")


def test_dialogue_benchmark_exposes_default_strong_proof_profiles():
    assert default_dialogue_strong_proof_profiles() == (
        "pe-eta",
        "pe-eta-no-semantic-label",
        "pe-eta-no-reflection-cache",
        "pe-eta-pe-readout-only",
        "pe-drive-off",
        "eta-off",
        "timescale-off",
    )


def test_dialogue_benchmark_exposes_default_comprehensive_profiles():
    assert default_dialogue_comprehensive_profiles() == (
        "pe-eta",
        "pe-eta-online-only",
        "pe-eta-no-writeback",
        "pe-eta-no-rare-heavy",
        "pe-drive-off",
        "eta-off",
        "timescale-off",
    )


def test_dialogue_benchmark_exposes_default_case_variants():
    variants = dialogue_case_variants()

    assert len(variants) == len(DEFAULT_DIALOGUE_CASE_VARIANTS)
    labels = {(variant.base_case_id, variant.variant_label) for variant in variants}
    assert ("repair", "wording_shift") in labels
    assert ("goal_drift", "pressure_shift_late") in labels
    assert all(variant.case.case_id.startswith(f"{variant.base_case_id}__") for variant in variants)


def test_dialogue_benchmark_exposes_default_paraphrase_families():
    families = dialogue_paraphrase_families()

    assert len(families) == len(DEFAULT_DIALOGUE_PARAPHRASE_FAMILIES)
    family_ids = {(family.base_case_id, family.family_label) for family in families}
    assert ("repair", "repair_family") in family_ids
    assert ("goal_drift", "goal_drift_family") in family_ids


def test_dialogue_benchmark_exposes_default_open_dialogue_scenarios():
    scenarios = open_dialogue_scenarios()

    assert scenarios == DEFAULT_OPEN_DIALOGUE_SCENARIOS
    assert tuple(scenario.scenario_id for scenario in scenarios) == (
        "open_repair",
        "open_clarification",
        "open_failure_loop",
        "open_goal_shift",
    )


def test_dialogue_benchmark_exposes_default_open_ablation_profiles():
    assert default_open_dialogue_ablation_profiles() == ("pe-eta", "pe-drive-off", "eta-off")


def test_build_deterministic_user_simulator_is_reproducible_for_same_seed():
    scenario = get_open_dialogue_scenario("open_repair")
    simulator_a = build_deterministic_user_simulator(scenario_id=scenario.scenario_id, seed=7, max_turns=4)
    simulator_b = build_deterministic_user_simulator(scenario_id=scenario.scenario_id, seed=7, max_turns=4)
    opening_a = simulator_a.next_turn()
    opening_b = simulator_b.next_turn()
    assert opening_a == opening_b

    escalation_turn = _benchmark_turn(
        turn_index=1,
        pe=0.45,
        reward=-0.12,
        action="evidence-only",
        regime="support",
        abstract_action="stay-cold",
        switch_gate=0.05,
        delayed_metric=0.1,
    )
    followup_a = simulator_a.next_turn(last_turn=escalation_turn)
    followup_b = simulator_b.next_turn(last_turn=escalation_turn)

    assert followup_a == followup_b
    assert simulator_a.episode_state == simulator_b.episode_state


def test_build_open_dialogue_case_report_uses_open_acceptance_surface():
    scenario = OpenDialogueScenario(
        scenario_id="open-proof",
        description="Synthetic open scenario for report coverage.",
        opening_turns=("start",),
        escalation_turns=("push",),
        stabilization_turns=("stabilize",),
        consolidation_turns=("compress",),
        max_turns=4,
    )
    turns = (
        _benchmark_turn(
            turn_index=1,
            pe=0.32,
            reward=-0.08,
            action="evidence-only",
            regime="repair",
            abstract_action="probe",
            switch_gate=0.1,
            delayed_metric=0.1,
        ),
        _benchmark_turn(
            turn_index=2,
            pe=0.28,
            reward=-0.07,
            action="ssl-only-pe",
            regime="repair",
            abstract_action="repair",
            switch_gate=0.4,
            delayed_metric=0.2,
            bounded_writeback_applied=True,
            reflection_promotion_eligible=True,
            session_post_completed_job_count=1,
            pe_triggered=True,
        ),
        _benchmark_turn(
            turn_index=3,
            pe=0.06,
            reward=-0.01,
            action="evidence-only",
            regime="support",
            abstract_action="stabilize",
            switch_gate=0.3,
            delayed_metric=0.5,
        ),
        _benchmark_turn(
            turn_index=4,
            pe=0.03,
            reward=0.01,
            action="evidence-only",
            regime="support",
            abstract_action="consolidate",
            switch_gate=0.2,
            delayed_metric=0.7,
        ),
    )

    report = build_open_dialogue_case_report(
        scenario=scenario,
        final_episode_state=OpenDialogueEpisodeState(
            scenario_id=scenario.scenario_id,
            turn_index=4,
            pressure_level=0,
            adaptive_response_count=1,
            calm_turn_count=2,
            last_stage="consolidation",
            completed=True,
            stop_reason="stable-consolidation",
        ),
        turns=turns,
    )

    assert report.final_episode_state.completed is True
    assert report.pe_triggered_turn_count > 0
    assert report.online_learning_turn_count > 0
    assert report.late_episode_stability_score >= 0.5
    check_map = dict(report.acceptance_checks)
    assert check_map["episode-runs-to-completion"] is True
    assert check_map["multi-timescale-evidence-observed"] is True
    assert report.passed is True


def test_run_open_dialogue_case_reuses_runner_turn_path():
    scenario = get_open_dialogue_scenario("open_repair")
    report = asyncio.run(
        run_open_dialogue_case(
            scenario=scenario,
            runner=_synthetic_runner(
                ScriptedDialogueCase(
                    case_id="open-repair-adapter",
                    description="Synthetic runner shell for open dialogue.",
                    user_inputs=("seed",),
                )
            ),
            seed=3,
            max_turns=3,
        )
    )

    assert len(report.turns) == 3
    assert report.final_episode_state.completed is True
    assert report.final_episode_state.stop_reason == "max-turns"
    assert all(turn.wave_id.startswith("wave-") for turn in report.turns)


def test_run_open_dialogue_benchmark_aggregates_open_reports():
    benchmark = asyncio.run(
        run_open_dialogue_benchmark(
            scenarios=(get_open_dialogue_scenario("open_repair"),),
            runner_factory=lambda scenario: _synthetic_runner(
                ScriptedDialogueCase(
                    case_id=f"open-benchmark:{scenario.scenario_id}",
                    description="Synthetic open benchmark runner shell.",
                    user_inputs=("seed",),
                )
            ),
            seed=2,
        )
    )

    assert benchmark.total_case_count == 1
    assert len(benchmark.case_reports) == 1
    assert "episode_runs_to_completion" in dict(benchmark.metric_means)


def test_run_open_dialogue_ablation_benchmark_compares_small_profile_matrix():
    scenario = get_open_dialogue_scenario("open_repair")
    comparison = asyncio.run(
        run_open_dialogue_ablation_benchmark(
            scenarios=(scenario,),
            profile_labels=default_open_dialogue_ablation_profiles(),
            runner_factory=_synthetic_open_ablation_runner,
            seed=5,
        )
    )

    assert comparison.baseline_label == "pe-eta"
    assert tuple(path.path_label for path in comparison.path_reports) == (
        "pe-eta",
        "pe-drive-off",
        "eta-off",
    )
    assert len(comparison.case_deltas_from_baseline) == 1
    scenario_id, per_path = comparison.case_deltas_from_baseline[0]
    assert scenario_id == scenario.scenario_id
    assert {label for label, _ in per_path} == {"pe-eta", "pe-drive-off", "eta-off"}
    metric_deltas = dict(comparison.metric_deltas_from_baseline)
    assert "pe-drive-off" in metric_deltas
    assert "eta-off" in metric_deltas
    assert "pe_triggered_turn_count" in dict(metric_deltas["pe-drive-off"])


def test_open_ablation_runner_respects_profile_runtime_semantics():
    scenario = get_open_dialogue_scenario("open_repair")
    pe_drive_runner = _synthetic_open_ablation_runner("pe-drive-off", scenario)
    eta_off_runner = _synthetic_open_ablation_runner("eta-off", scenario)

    assert pe_drive_runner._external_prediction_error_drive is False
    assert eta_off_runner._external_prediction_error_drive is False
    assert eta_off_runner._joint_schedule.ssl_interval == 0
    assert eta_off_runner._joint_schedule.rl_interval == 0


def test_run_dialogue_pe_eta_case_collects_pe_and_eta_trajectories():
    case = ScriptedDialogueCase(
        case_id="mini-proof",
        description="Short scripted case for integration coverage.",
        user_inputs=(
            "I need help but the first answer was not enough.",
            "Try again and adjust to the feedback.",
            "Now tell me what changed in your frame.",
        ),
    )

    report = asyncio.run(
        run_dialogue_pe_eta_case(
            case=case,
            runner=_synthetic_runner(case),
        )
    )

    assert report.case.case_id == "mini-proof"
    assert len(report.turns) == 3
    assert report.prediction_chain_turn_count >= 1
    assert isinstance(report.turns[0].outcome_metrics, tuple)
    assert report.turns[0].joint_schedule_action
    assert report.turns[0].nested_profile_active is True


def test_run_dialogue_pe_eta_benchmark_runs_complete_scripted_suite():
    report = asyncio.run(
        run_dialogue_pe_eta_benchmark(
            runner_factory=_synthetic_runner,
        )
    )

    assert report.total_case_count == len(DEFAULT_DIALOGUE_PROOF_CASES)
    assert len(report.case_reports) == len(DEFAULT_DIALOGUE_PROOF_CASES)
    assert report.description
    metric_means = dict(report.metric_means)
    assert "bounded_writeback_turn_count" in metric_means
    assert "rare_heavy_recommended_count" in metric_means
    for case_report in report.case_reports:
        assert len(case_report.turns) == len(case_report.case.user_inputs)
        assert isinstance(case_report.acceptance_checks, tuple)


def test_run_dialogue_pe_eta_ablation_benchmark_collects_path_deltas():
    report = asyncio.run(
        run_dialogue_pe_eta_ablation_benchmark(
            cases=DEFAULT_DIALOGUE_PROOF_CASES[:2],
            runner_factory=_synthetic_ablation_runner,
        )
    )

    assert report.baseline_label == "pe-eta"
    assert len(report.path_reports) == 4
    assert len(report.case_deltas_from_baseline) == 2
    case_id, path_deltas = report.case_deltas_from_baseline[0]
    assert case_id in {"repair", "task_clarification"}
    delta_map = dict(path_deltas)
    assert "pe-eta" in delta_map
    assert "pe-drive-off" in delta_map
    assert "eta-off" in delta_map
    assert "timescale-off" in delta_map
    assert dict(delta_map["pe-eta"])["passed"] == 0.0
    assert dict(delta_map["pe-drive-off"])["pe_triggered_turn_count"] <= 0.0
    assert "recovery_lag_turns" in dict(delta_map["pe-eta"])
    assert "pressure_localization_score" in dict(delta_map["pe-eta"])
    assert "over_response_cost" in dict(delta_map["pe-eta"])
    assert "pressure_response_precision" in dict(delta_map["pe-eta"])
    assert "pressure_response_recall" in dict(delta_map["pe-eta"])
    assert "stability_after_recovery_score" in dict(delta_map["pe-eta"])
    profile_means = dict(report.metric_deltas_from_baseline)
    assert "pe-eta" in profile_means
    assert "pe-drive-off" in profile_means
    assert "bounded_writeback_turn_count" in dict(profile_means["pe-drive-off"])


def test_run_dialogue_pe_eta_perturbation_benchmark_collects_variant_reports():
    report = asyncio.run(
        run_dialogue_pe_eta_perturbation_benchmark(
            variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
            runner_factory=_synthetic_perturbation_runner,
        )
    )

    assert len(report.variant_cases) == 2
    assert report.ablation_report.baseline_label == "pe-eta"
    assert len(report.ablation_report.path_reports) == 4
    variant_ids = {variant.case.case_id for variant in report.variant_cases}
    case_ids = {
        case_id
        for case_id, _ in report.ablation_report.case_deltas_from_baseline
    }
    assert case_ids == variant_ids


def test_generate_stochastic_dialogue_case_variants_is_deterministic():
    first = generate_stochastic_dialogue_case_variants(seeds=(0, 1))
    second = generate_stochastic_dialogue_case_variants(seeds=(0, 1))

    assert tuple(variant.case.case_id for variant in first) == tuple(variant.case.case_id for variant in second)
    assert all("__seed_" in variant.case.case_id for variant in first)
    assert len(first) == len(DEFAULT_DIALOGUE_PARAPHRASE_FAMILIES) * 2


def test_build_dialogue_replay_ranking_report_sorts_by_diagnostic_score():
    perturbation_report = asyncio.run(
        run_dialogue_pe_eta_perturbation_benchmark(
            variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
            runner_factory=_synthetic_perturbation_runner,
        )
    )

    ranking = build_dialogue_replay_ranking_report(
        variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
        ablation_report=perturbation_report.ablation_report,
    )

    assert len(ranking.entries) == 2
    assert ranking.entries[0].diagnostic_score >= ranking.entries[1].diagnostic_score
    assert ranking.entries[0].gap_vs_eta_no_pe >= 0.0


def test_build_dialogue_replay_selection_artifact_keeps_top_k_entries():
    perturbation_report = asyncio.run(
        run_dialogue_pe_eta_perturbation_benchmark(
            variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:3],
            runner_factory=_synthetic_perturbation_runner,
        )
    )
    ranking = build_dialogue_replay_ranking_report(
        variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:3],
        ablation_report=perturbation_report.ablation_report,
    )
    artifact = build_dialogue_replay_selection_artifact(
        variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:3],
        replay_ranking_report=ranking,
        top_k=2,
    )

    assert artifact.artifact_id == "dialogue-replay-selection"
    assert len(artifact.selected_variants) == 2
    assert len(artifact.ranking_entries) == 2


def test_build_replay_selection_training_traces_matches_selected_variants():
    perturbation_report = asyncio.run(
        run_dialogue_pe_eta_perturbation_benchmark(
            variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
            runner_factory=_synthetic_perturbation_runner,
        )
    )
    ranking = build_dialogue_replay_ranking_report(
        variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
        ablation_report=perturbation_report.ablation_report,
    )
    artifact = build_dialogue_replay_selection_artifact(
        variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
        replay_ranking_report=ranking,
        top_k=2,
    )
    traces = build_replay_selection_training_traces(artifact)

    assert len(traces) == 2
    assert traces[0].trace_id.startswith("replay-selection:")


def test_train_rare_heavy_artifact_from_replay_selection_exports_artifact():
    perturbation_report = asyncio.run(
        run_dialogue_pe_eta_perturbation_benchmark(
            variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
            runner_factory=_synthetic_perturbation_runner,
        )
    )
    ranking = build_dialogue_replay_ranking_report(
        variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
        ablation_report=perturbation_report.ablation_report,
    )
    artifact = build_dialogue_replay_selection_artifact(
        variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
        replay_ranking_report=ranking,
        top_k=2,
    )
    rare_heavy_artifact = train_rare_heavy_artifact_from_replay_selection(artifact)

    assert rare_heavy_artifact.artifact_id.endswith(":rare-heavy")


def test_run_replay_selection_artifact_acceptance_benchmark_returns_case_reports():
    perturbation_report = asyncio.run(
        run_dialogue_pe_eta_perturbation_benchmark(
            variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
            runner_factory=_synthetic_perturbation_runner,
        )
    )
    ranking = build_dialogue_replay_ranking_report(
        variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
        ablation_report=perturbation_report.ablation_report,
    )
    selection_artifact = build_dialogue_replay_selection_artifact(
        variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
        replay_ranking_report=ranking,
        top_k=2,
    )
    acceptance_report = asyncio.run(
        run_replay_selection_artifact_acceptance_benchmark(
            selection_artifact,
            runner_factory=_synthetic_acceptance_runner,
        )
    )

    assert len(acceptance_report.case_reports) == 2
    assert acceptance_report.artifact.artifact_id.endswith(":rare-heavy")
    assert acceptance_report.decision.accepted is False
    assert acceptance_report.decision.rollback_applied is True
    assert all(report.rollback_operations for report in acceptance_report.case_reports)


def test_replay_selection_artifact_acceptance_can_be_forced_accept():
    perturbation_report = asyncio.run(
        run_dialogue_pe_eta_perturbation_benchmark(
            variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
            runner_factory=_synthetic_perturbation_runner,
        )
    )
    ranking = build_dialogue_replay_ranking_report(
        variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
        ablation_report=perturbation_report.ablation_report,
    )
    selection_artifact = build_dialogue_replay_selection_artifact(
        variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
        replay_ranking_report=ranking,
        top_k=2,
    )
    acceptance_report = asyncio.run(
        run_replay_selection_artifact_acceptance_benchmark(
            selection_artifact,
            runner_factory=_synthetic_acceptance_runner,
            gate_config=DialogueArtifactAcceptanceGateConfig(
                min_mean_score_delta=-1.0,
                min_passed_case_delta=-10,
                min_positive_case_fraction=0.0,
                min_worst_case_delta=-1.0,
            ),
        )
    )

    assert acceptance_report.decision.accepted is True
    assert acceptance_report.decision.rollback_applied is False
    assert all(not report.rollback_operations for report in acceptance_report.case_reports)


def test_run_dialogue_pe_eta_systematic_replay_benchmark_collects_generated_variants():
    report = asyncio.run(
        run_dialogue_pe_eta_systematic_replay_benchmark(
            seeds=(0,),
            include_fixed_variants=False,
            runner_factory=_synthetic_systematic_runner,
        )
    )

    assert len(report.variant_cases) == len(DEFAULT_DIALOGUE_PARAPHRASE_FAMILIES)
    assert len(report.replay_ranking_report.entries) == len(report.variant_cases)
    assert report.perturbation_report.ablation_report.baseline_label == "pe-eta"


def test_run_dialogue_pe_eta_comprehensive_benchmark_runs_end_to_end():
    report = asyncio.run(
        run_dialogue_pe_eta_comprehensive_benchmark(
            canonical_cases=DEFAULT_DIALOGUE_PROOF_CASES[:1],
            open_scenarios=DEFAULT_OPEN_DIALOGUE_SCENARIOS[:1],
            variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:1],
            seeds=(0,),
            families=DEFAULT_DIALOGUE_PARAPHRASE_FAMILIES[:1],
            selection_top_k=1,
            candidate_configs=DEFAULT_RARE_HEAVY_CANDIDATE_CONFIGS[:1],
            canonical_runner_factory=_synthetic_ablation_runner,
            open_runner_factory=_synthetic_open_ablation_runner,
            perturbation_runner_factory=_synthetic_perturbation_runner,
            systematic_runner_factory=_synthetic_systematic_runner,
            acceptance_runner_factory=_synthetic_acceptance_runner,
        )
    )

    assert report.profile_labels == default_dialogue_comprehensive_profiles()
    assert report.canonical_ablation_report.baseline_label == "pe-eta"
    assert report.longitudinal_report.cross_session_report.verdict in {"growing", "stable", "regressing", "insufficient-data"}
    assert isinstance(report.essence_report, DialogueNLEssenceAssessmentReport)
    assert isinstance(report.essence_acceptance, DialogueNLEssenceAcceptanceDecision)
    assert report.open_ablation_report is not None
    assert report.open_ablation_report.baseline_label == "pe-eta"
    assert report.emergence_dashboard.baseline_label == "pe-eta"
    assert report.emergence_dashboard.strong_proof_panels
    assert report.emergence_dashboard.open_environment_panels
    assert len(report.selection_artifact.selected_variants) == 1
    assert len(report.artifact_comparison_report.candidate_reports) == 1
    assert report.description


def test_build_dialogue_emergence_dashboard_compresses_strong_proof_and_open_env_evidence():
    report = asyncio.run(
        run_dialogue_pe_eta_comprehensive_benchmark(
            canonical_cases=DEFAULT_DIALOGUE_PROOF_CASES[:1],
            open_scenarios=DEFAULT_OPEN_DIALOGUE_SCENARIOS[:1],
            variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:1],
            seeds=(0,),
            families=DEFAULT_DIALOGUE_PARAPHRASE_FAMILIES[:1],
            selection_top_k=1,
            candidate_configs=DEFAULT_RARE_HEAVY_CANDIDATE_CONFIGS[:1],
            canonical_runner_factory=_synthetic_ablation_runner,
            open_runner_factory=_synthetic_open_ablation_runner,
            perturbation_runner_factory=_synthetic_perturbation_runner,
            systematic_runner_factory=_synthetic_systematic_runner,
            acceptance_runner_factory=_synthetic_acceptance_runner,
        )
    )

    dashboard = build_dialogue_emergence_dashboard(report)

    assert dashboard.baseline_label == "pe-eta"
    assert dashboard.canonical_case_count == 1
    assert dashboard.open_scenario_count == 1
    assert dashboard.strong_proof_panels
    assert dashboard.open_environment_panels
    assert dashboard.strongest_scaffold_path_label is not None
    assert dashboard.strongest_open_path_label is not None
    assert isinstance(dashboard.interpretation, str)
    assert "canonical_pass_rate=" in dashboard.description


def test_build_dialogue_emergence_dashboard_payload_exposes_summary_keys():
    report = asyncio.run(
        run_dialogue_pe_eta_comprehensive_benchmark(
            canonical_cases=DEFAULT_DIALOGUE_PROOF_CASES[:1],
            open_scenarios=DEFAULT_OPEN_DIALOGUE_SCENARIOS[:1],
            variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:1],
            seeds=(0,),
            families=DEFAULT_DIALOGUE_PARAPHRASE_FAMILIES[:1],
            selection_top_k=1,
            candidate_configs=DEFAULT_RARE_HEAVY_CANDIDATE_CONFIGS[:1],
            canonical_runner_factory=_synthetic_ablation_runner,
            open_runner_factory=_synthetic_open_ablation_runner,
            perturbation_runner_factory=_synthetic_perturbation_runner,
            systematic_runner_factory=_synthetic_systematic_runner,
            acceptance_runner_factory=_synthetic_acceptance_runner,
        )
    )

    payload = build_dialogue_emergence_dashboard_payload(report)

    assert payload["baseline_label"] == "pe-eta"
    assert payload["canonical"]["case_count"] == 1
    assert payload["open_environment"]["scenario_count"] == 1
    assert payload["strong_proof_panels"]
    assert payload["open_environment_panels"]
    assert "rare_heavy_gate" in payload
    assert "essence" in payload


def test_export_dialogue_emergence_dashboard_artifact_writes_json(tmp_path):
    report = asyncio.run(
        run_dialogue_pe_eta_comprehensive_benchmark(
            canonical_cases=DEFAULT_DIALOGUE_PROOF_CASES[:1],
            open_scenarios=DEFAULT_OPEN_DIALOGUE_SCENARIOS[:1],
            variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:1],
            seeds=(0,),
            families=DEFAULT_DIALOGUE_PARAPHRASE_FAMILIES[:1],
            selection_top_k=1,
            candidate_configs=DEFAULT_RARE_HEAVY_CANDIDATE_CONFIGS[:1],
            canonical_runner_factory=_synthetic_ablation_runner,
            open_runner_factory=_synthetic_open_ablation_runner,
            perturbation_runner_factory=_synthetic_perturbation_runner,
            systematic_runner_factory=_synthetic_systematic_runner,
            acceptance_runner_factory=_synthetic_acceptance_runner,
        )
    )

    output_path = tmp_path / "emergence_dashboard.json"
    written_path = export_dialogue_emergence_dashboard_artifact(
        report,
        output_path=output_path,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert written_path == output_path
    assert payload["baseline_label"] == "pe-eta"
    assert payload["canonical"]["case_count"] == 1
    assert payload["open_environment"]["scenario_count"] == 1
    assert payload["strong_proof_panels"]
    assert payload["description"] == report.emergence_dashboard.description


def test_run_dialogue_pe_eta_longitudinal_benchmark_emits_cross_session_report():
    report = asyncio.run(
        run_dialogue_pe_eta_longitudinal_benchmark(
            cases=DEFAULT_DIALOGUE_PROOF_CASES[:2],
            runner_factory=lambda: _synthetic_runner(DEFAULT_DIALOGUE_PROOF_CASES[0]),
        )
    )

    assert len(report.case_reports) == 2
    assert len(report.session_reports) == 2
    assert report.cross_session_report.verdict in {"growing", "stable", "regressing"}
    assert report.description


def test_build_dialogue_nl_essence_assessment_uses_nested_and_cross_session_evidence():
    ablation_report = asyncio.run(
        run_dialogue_pe_eta_ablation_benchmark(
            cases=DEFAULT_DIALOGUE_PROOF_CASES[:1],
            profile_labels=("pe-eta", "pe-eta-no-rare-heavy", "pe-drive-off", "eta-off", "timescale-off"),
            runner_factory=_synthetic_ablation_runner,
        )
    )
    benchmark_report = next(
        path.benchmark_report
        for path in ablation_report.path_reports
        if path.path_label == "pe-eta"
    )
    longitudinal_report = asyncio.run(
        run_dialogue_pe_eta_longitudinal_benchmark(
            cases=DEFAULT_DIALOGUE_PROOF_CASES[:2],
            runner_factory=lambda: _synthetic_runner(DEFAULT_DIALOGUE_PROOF_CASES[0]),
        )
    )

    assessment = build_dialogue_nl_essence_assessment(
        path_label="pe-eta",
        benchmark_report=benchmark_report,
        comparison_report=ablation_report,
        cross_session_report=longitudinal_report.cross_session_report,
    )

    assert assessment.path_label == "pe-eta"
    assert assessment.total_gate_count == 7
    gate_map = {gate.gate_id: gate for gate in assessment.gates}
    assert "multi-timescale-default" in gate_map
    assert "online-fast-pe-coupling" in gate_map
    assert "cross-session-growth" in gate_map
    assert "rare-heavy-net-benefit" in gate_map


def test_dialogue_nl_essence_acceptance_fails_closed_when_required_gate_is_missing():
    benchmark_report = asyncio.run(
        run_dialogue_pe_eta_benchmark(
            cases=DEFAULT_DIALOGUE_PROOF_CASES[:1],
            runner_factory=_synthetic_runner,
        )
    )

    assessment = build_dialogue_nl_essence_assessment(
        path_label="pe-eta",
        benchmark_report=benchmark_report,
    )
    decision = evaluate_dialogue_nl_essence_acceptance(assessment)

    assert isinstance(decision, DialogueNLEssenceAcceptanceDecision)
    assert decision.accepted is False
    assert "failed-gate:rare-heavy-net-benefit" in decision.reasons


def test_slow_shapes_fast_gate_reports_config_artifact_when_case_count_is_below_proof_minimum():
    ablation_report = asyncio.run(
        run_dialogue_pe_eta_ablation_benchmark(
            cases=DEFAULT_DIALOGUE_PROOF_CASES[:1],
            profile_labels=("pe-eta", "pe-eta-no-rare-heavy", "pe-drive-off", "eta-off", "timescale-off"),
            runner_factory=_synthetic_ablation_runner,
        )
    )
    benchmark_report = next(
        path.benchmark_report
        for path in ablation_report.path_reports
        if path.path_label == "pe-eta"
    )
    longitudinal_report = asyncio.run(
        run_dialogue_pe_eta_longitudinal_benchmark(
            cases=DEFAULT_DIALOGUE_PROOF_CASES[:1],
            runner_factory=lambda: _synthetic_runner(DEFAULT_DIALOGUE_PROOF_CASES[0]),
        )
    )

    assessment = build_dialogue_nl_essence_assessment(
        path_label="pe-eta",
        benchmark_report=benchmark_report,
        comparison_report=ablation_report,
        cross_session_report=longitudinal_report.cross_session_report,
        longitudinal_report=longitudinal_report,
        proof_min_canonical_cases=2,
    )

    gate_map = {gate.gate_id: gate for gate in assessment.gates}
    slow_shapes_fast = gate_map["slow-shapes-fast"]

    assert slow_shapes_fast.passed is False
    evidence = dict(slow_shapes_fast.evidence)
    assert evidence["failure_mode"] == "config-artifact"
    assert evidence["proof_min_case_count_satisfied"] == 0.0


def test_dialogue_nl_essence_acceptance_passes_with_longitudinal_nested_evidence():
    ablation_report = asyncio.run(
        run_dialogue_pe_eta_ablation_benchmark(
            cases=DEFAULT_DIALOGUE_PROOF_CASES[:2],
            profile_labels=("pe-eta", "pe-eta-no-rare-heavy", "pe-drive-off", "eta-off", "timescale-off"),
            runner_factory=_synthetic_ablation_runner,
        )
    )
    benchmark_report = next(
        path.benchmark_report
        for path in ablation_report.path_reports
        if path.path_label == "pe-eta"
    )
    longitudinal_report = asyncio.run(
        run_dialogue_pe_eta_longitudinal_benchmark(
            cases=DEFAULT_DIALOGUE_PROOF_CASES[:2],
            runner_factory=lambda: _synthetic_runner(DEFAULT_DIALOGUE_PROOF_CASES[0]),
        )
    )
    assessment = build_dialogue_nl_essence_assessment(
        path_label="pe-eta",
        benchmark_report=benchmark_report,
        comparison_report=ablation_report,
        cross_session_report=longitudinal_report.cross_session_report,
        longitudinal_report=longitudinal_report,
    )

    decision = evaluate_dialogue_nl_essence_acceptance(
        assessment,
        config=DialogueNLEssenceAcceptanceConfig(
            required_gate_ids=(
                "pe-first",
                "multi-timescale-default",
                "slow-shapes-fast",
                "judge-gated-evolution",
                "cross-session-growth",
            ),
            min_passed_gate_count=5,
        ),
    )

    assert decision.accepted is True
    assert not decision.blocked_gate_ids


def test_rare_heavy_net_benefit_gate_fails_closed_without_no_rare_heavy_comparison():
    benchmark_report = asyncio.run(
        run_dialogue_pe_eta_benchmark(
            cases=DEFAULT_DIALOGUE_PROOF_CASES[:2],
            runner_factory=_synthetic_runner,
        )
    )
    longitudinal_report = asyncio.run(
        run_dialogue_pe_eta_longitudinal_benchmark(
            cases=DEFAULT_DIALOGUE_PROOF_CASES[:2],
            runner_factory=lambda: _synthetic_runner(DEFAULT_DIALOGUE_PROOF_CASES[0]),
        )
    )

    assessment = build_dialogue_nl_essence_assessment(
        path_label="pe-eta",
        benchmark_report=benchmark_report,
        cross_session_report=longitudinal_report.cross_session_report,
        longitudinal_report=longitudinal_report,
    )

    gate_map = {gate.gate_id: gate for gate in assessment.gates}
    rare_heavy_gate = gate_map["rare-heavy-net-benefit"]
    assert rare_heavy_gate.passed is False
    assert dict(rare_heavy_gate.evidence)["failure_mode"] == "missing-no-rare-heavy-comparison"


def test_explicit_pe_schedule_labels_do_not_receive_credit_below_runtime_reward_threshold():
    case = ScriptedDialogueCase(
        case_id="pe-threshold-alignment",
        description="Explicit -pe labels should still satisfy runtime PE schedule semantics.",
        user_inputs=("turn-1", "turn-2"),
        expected_pressure_turns=(1,),
    )
    turns = (
        _benchmark_turn(
            turn_index=1,
            pe=0.05,
            reward=0.10,
            action="evidence-only",
            regime="repair",
            abstract_action="observe",
            switch_gate=0.1,
            delayed_metric=0.2,
        ),
        _benchmark_turn(
            turn_index=2,
            pe=0.04,
            reward=0.01,
            action="ssl-only-pe",
            regime="repair",
            abstract_action="adapt",
            switch_gate=0.7,
            delayed_metric=0.3,
        ),
    )

    report = build_dialogue_case_report(
        case=case,
        turns=turns,
        allow_interval_carryover_credit=False,
    )

    assert report.high_pe_turn_count == 1
    assert report.pe_triggered_turn_count == 0


def test_case_report_tracks_store_nested_reset_deltas_without_turn_level_overcount():
    case = ScriptedDialogueCase(
        case_id="nested-reset-diagnostics",
        description="Case report should distinguish turn-level reset flags from owner-level reset count.",
        user_inputs=("turn-1", "turn-2", "turn-3"),
        expected_pressure_turns=(1,),
    )
    turns = (
        _benchmark_turn(
            turn_index=1,
            pe=0.25,
            reward=0.2,
            action="ssl-only-pe",
            regime="repair",
            abstract_action="adapt",
            switch_gate=0.8,
            delayed_metric=0.2,
            pe_triggered=True,
            nested_context_reset_applied=True,
            nested_context_reset_total_count=1,
            slow_to_fast_init_benefit=0.02,
            slow_to_fast_target_distance_before=0.02,
            slow_to_fast_target_distance_after=0.0,
            slow_to_fast_target_alignment_gain=0.02,
        ),
        _benchmark_turn(
            turn_index=2,
            pe=0.1,
            reward=0.02,
            action="evidence-only",
            regime="repair",
            abstract_action="stabilize",
            switch_gate=0.2,
            delayed_metric=0.4,
            nested_context_reset_total_count=1,
            slow_to_fast_init_benefit=0.02,
            slow_to_fast_target_distance_before=0.0,
            slow_to_fast_target_distance_after=0.0,
            slow_to_fast_target_alignment_gain=0.0,
        ),
        _benchmark_turn(
            turn_index=3,
            pe=0.08,
            reward=0.01,
            action="evidence-only",
            regime="repair",
            abstract_action="stabilize",
            switch_gate=0.1,
            delayed_metric=0.6,
            nested_context_reset_total_count=1,
            slow_to_fast_init_benefit=0.02,
            slow_to_fast_target_distance_before=0.0,
            slow_to_fast_target_distance_after=0.0,
            slow_to_fast_target_alignment_gain=0.0,
        ),
    )

    report = build_dialogue_case_report(case=case, turns=turns)

    assert report.nested_context_reset_count == 1
    assert report.store_nested_context_reset_count == 1
    assert report.boundary_reset_observed_on_first_turn is True
    assert report.first_turn_slow_to_fast_init_benefit == 0.02
    assert report.mean_reset_turn_slow_to_fast_init_benefit == 0.02
    assert report.first_turn_slow_to_fast_target_distance_before == 0.02
    assert report.first_turn_slow_to_fast_target_distance_after == 0.0
    assert report.first_turn_slow_to_fast_target_alignment_gain == 0.02
    assert report.mean_reset_turn_slow_to_fast_target_distance_before == 0.02
    assert report.mean_reset_turn_slow_to_fast_target_distance_after == 0.0
    assert report.mean_reset_turn_slow_to_fast_target_alignment_gain == 0.02


def test_case_report_surfaces_rare_heavy_preimport_selection_telemetry():
    case = ScriptedDialogueCase(
        case_id="rare-heavy-preimport-telemetry",
        description="Case report should expose pre-import rare-heavy selection and rejection evidence.",
        user_inputs=("turn-1", "turn-2"),
        expected_pressure_turns=(1,),
    )
    report = build_dialogue_case_report(
        case=case,
        turns=(
            _benchmark_turn(
                turn_index=1,
                pe=0.28,
                reward=0.18,
                action="ssl-only-pe",
                regime="repair",
                abstract_action="adapt",
                switch_gate=0.75,
                delayed_metric=0.2,
                pe_triggered=True,
            ),
            _benchmark_turn(
                turn_index=2,
                pe=0.10,
                reward=0.01,
                action="full-cycle-pe",
                regime="repair",
                abstract_action="stabilize",
                switch_gate=0.25,
                delayed_metric=0.6,
            ),
        ),
    )

    updated_turns = (
        replace(
            report.turns[0],
            rare_heavy_recommended=True,
            rare_heavy_import_decision="rejected-pre-import",
            rare_heavy_reject_reason="pre-import-positive-fraction-too-low",
            rare_heavy_pre_import_passed=False,
            rare_heavy_pre_import_mean_score_delta=-0.08,
            rare_heavy_candidate_alignment=1.0,
            rare_heavy_candidate_adapter_parameter_count=12,
        ),
        replace(
            report.turns[1],
            rare_heavy_recommended=True,
            rare_heavy_applied=True,
            rare_heavy_import_decision="imported",
            rare_heavy_pre_import_passed=True,
            rare_heavy_pre_import_mean_score_delta=0.12,
            rare_heavy_candidate_alignment=1.0,
            rare_heavy_candidate_adapter_parameter_count=18,
        ),
    )
    updated_report = build_dialogue_case_report(case=case, turns=updated_turns)
    assert updated_report.rare_heavy_pre_import_pass_count == 1
    assert updated_report.rare_heavy_pre_import_reject_count == 1
    assert updated_report.mean_rare_heavy_pre_import_score_delta > 0.0
    assert updated_report.mean_rare_heavy_candidate_alignment == 1.0
    assert updated_report.max_rare_heavy_candidate_adapter_parameter_count == 18


def test_slow_shapes_fast_gate_identifies_already_near_target_when_benefit_is_small():
    case = ScriptedDialogueCase(
        case_id="nested-already-near-target",
        description="Small displacement can still mean reset is healthy if the state was already near its target.",
        user_inputs=("turn-1", "turn-2"),
        expected_pressure_turns=(1,),
    )
    case_report = build_dialogue_case_report(
        case=case,
        turns=(
            _benchmark_turn(
                turn_index=1,
                pe=0.22,
                reward=0.12,
                action="ssl-only-pe",
                regime="repair",
                abstract_action="adapt",
                switch_gate=0.8,
                delayed_metric=0.2,
                pe_triggered=True,
                nested_context_reset_applied=True,
                nested_context_reset_total_count=1,
                slow_to_fast_init_benefit=0.002,
                slow_to_fast_target_distance_before=0.002,
                slow_to_fast_target_distance_after=0.001,
                slow_to_fast_target_alignment_gain=0.001,
            ),
            _benchmark_turn(
                turn_index=2,
                pe=0.09,
                reward=0.01,
                action="evidence-only",
                regime="repair",
                abstract_action="stabilize",
                switch_gate=0.2,
                delayed_metric=0.5,
                nested_context_reset_total_count=1,
                slow_to_fast_target_distance_before=0.0,
                slow_to_fast_target_distance_after=0.0,
                slow_to_fast_target_alignment_gain=0.0,
            ),
        ),
    )
    benchmark_report = DialogueBenchmarkReport(
        case_reports=(case_report,),
        passed_case_count=int(case_report.passed),
        total_case_count=1,
        metric_means=tuple(
            (key, float(value))
            for key, value in (
                ("prediction_chain_turn_count", float(case_report.prediction_chain_turn_count)),
                ("pe_triggered_turn_count", float(case_report.pe_triggered_turn_count)),
                ("delayed_improvement_observed", float(case_report.delayed_improvement_observed)),
                ("online_learning_turn_count", float(case_report.online_learning_turn_count)),
                ("bounded_writeback_turn_count", float(case_report.bounded_writeback_turn_count)),
                ("nested_profile_active_turn_count", float(case_report.nested_profile_active_turn_count)),
                ("learned_memory_primary_turn_count", float(case_report.learned_memory_primary_turn_count)),
                ("store_nested_context_reset_count", float(case_report.store_nested_context_reset_count)),
                ("mean_reset_turn_slow_to_fast_init_benefit", case_report.mean_reset_turn_slow_to_fast_init_benefit),
                (
                    "mean_reset_turn_slow_to_fast_target_distance_before",
                    case_report.mean_reset_turn_slow_to_fast_target_distance_before,
                ),
                (
                    "mean_reset_turn_slow_to_fast_target_distance_after",
                    case_report.mean_reset_turn_slow_to_fast_target_distance_after,
                ),
                (
                    "mean_reset_turn_slow_to_fast_target_alignment_gain",
                    case_report.mean_reset_turn_slow_to_fast_target_alignment_gain,
                ),
                ("boundary_reset_observed_on_first_turn", float(case_report.boundary_reset_observed_on_first_turn)),
                ("evolution_judge_turn_count", float(case_report.evolution_judge_turn_count)),
                ("evolution_judge_structural_allow_count", float(case_report.evolution_judge_structural_allow_count)),
                ("core_guided_recall_turn_count", float(case_report.core_guided_recall_turn_count)),
                ("mean_learned_recall_confidence", case_report.mean_learned_recall_confidence),
                ("max_artifact_consolidation_count", float(case_report.max_artifact_consolidation_count)),
            )
        ),
        description="single-case benchmark report",
    )
    assessment = build_dialogue_nl_essence_assessment(
        path_label="pe-eta",
        benchmark_report=benchmark_report,
        proof_min_canonical_cases=1,
    )
    slow_shapes_fast_gate = {gate.gate_id: gate for gate in assessment.gates}["slow-shapes-fast"]

    assert slow_shapes_fast_gate.passed is False
    evidence = dict(slow_shapes_fast_gate.evidence)
    assert evidence["failure_mode"] == "already-near-target"
    assert evidence["alignment_interpretation"] == "already-near-target"
    assert evidence["weak_benefit_explained_by_target_proximity"] == 1.0


def test_build_real_dialogue_comprehensive_runner_factories_share_runtime():
    factories = build_real_dialogue_comprehensive_runner_factories(
        runtime_mode=LocalSubstrateRuntimeMode.BUILTIN_ONLY,
    )
    case = DEFAULT_DIALOGUE_PROOF_CASES[0]
    variant = DEFAULT_DIALOGUE_CASE_VARIANTS[0]
    scenario = DEFAULT_OPEN_DIALOGUE_SCENARIOS[0]
    runner = factories.canonical_runner_factory("pe-eta", case)
    acceptance_runner = factories.acceptance_runner_factory(variant)
    open_runner = factories.open_runner_factory("pe-eta", scenario)

    assert runner._default_residual_runtime is factories.residual_runtime
    assert acceptance_runner._default_residual_runtime is factories.residual_runtime
    assert open_runner._default_residual_runtime is factories.residual_runtime
    assert factories.description


def test_run_real_dialogue_pe_eta_comprehensive_benchmark_completes_with_builtin_runtime():
    progress_messages: list[str] = []
    report = asyncio.run(
        run_real_dialogue_pe_eta_comprehensive_benchmark(
            config=default_dialogue_real_proof_config(),
            progress_callback=progress_messages.append,
        )
    )

    assert report.canonical_ablation_report.baseline_label == "pe-eta"
    assert len(report.longitudinal_report.case_reports) == 2
    assert report.longitudinal_report.cross_session_report.verdict in {"growing", "stable", "regressing", "insufficient-data"}
    assert report.essence_report.total_gate_count == 7
    assert isinstance(report.essence_acceptance, DialogueNLEssenceAcceptanceDecision)
    assert report.open_ablation_report is not None
    assert report.open_ablation_report.baseline_label == "pe-eta"
    assert report.emergence_dashboard.strong_proof_panels
    assert report.emergence_dashboard.open_environment_panels
    assert len(report.selection_artifact.selected_variants) == 1
    assert len(report.artifact_comparison_report.candidate_reports) == 1
    assert "rare_heavy_gate_passed=" in report.description
    assert "rare_heavy_gate_failure_mode=" in report.description
    assert "emergence_interpretation=" in report.description
    assert "shared runtime" in report.description
    assert "open_scenarios=" in report.description
    assert progress_messages[0].startswith("Building shared real residual runtime")
    assert progress_messages[-1] == "Real comprehensive benchmark finished."


def test_run_real_dialogue_pe_eta_comprehensive_benchmark_staged_persists_and_resumes(tmp_path):
    config = default_dialogue_real_proof_config(runtime_mode=LocalSubstrateRuntimeMode.BUILTIN_ONLY)
    first_progress: list[str] = []
    first_report = asyncio.run(
        run_real_dialogue_pe_eta_comprehensive_benchmark_staged(
            output_dir=tmp_path,
            config=config,
            shared_factories=_synthetic_shared_factories(),
            longitudinal_runner_factory=lambda: _synthetic_runner(DEFAULT_DIALOGUE_PROOF_CASES[0]),
            progress_callback=first_progress.append,
        )
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert DialogueComprehensiveStage.FINAL_REPORT.value in manifest["completed_stages"]
    assert manifest["summary"]["stage"] == DialogueComprehensiveStage.FINAL_REPORT.value
    assert manifest["summary"]["rare_heavy_gate"]["present"] is True
    assert "failure_mode" in manifest["summary"]["rare_heavy_gate"]
    assert "essence_accepted" in manifest["summary"]
    assert manifest["summary"]["artifact_acceptance"]["candidate_label"] is not None
    assert "override_mode" in manifest["summary"]["artifact_acceptance"]
    assert "reasons" in manifest["summary"]["artifact_acceptance"]
    assert "pre_import_evidence" in manifest["summary"]["artifact_acceptance"]
    assert "pre_import_pass_fraction" in manifest["summary"]["artifact_acceptance"]["pre_import_evidence"]
    assert "mean_pre_import_score_delta" in manifest["summary"]["artifact_acceptance"]["pre_import_evidence"]
    assert manifest["summary"]["emergence_dashboard"]["baseline_label"] == "pe-eta"
    assert "interpretation" in manifest["summary"]["emergence_dashboard"]
    assert manifest["summary"]["open_profile_count"] >= 1
    assert manifest["summary"]["open_scenario_count"] >= 1
    assert first_report.selection_artifact.selected_variants
    assert "rare_heavy_gate_passed=" in first_report.description
    assert "emergence_interpretation=" in first_report.description
    assert "open_scenarios=" in first_report.description
    assert any("Running canonical ablation" in message for message in first_progress)
    assert any("Running open-environment ablation" in message for message in first_progress)

    second_progress: list[str] = []
    resumed_report = asyncio.run(
        run_real_dialogue_pe_eta_comprehensive_benchmark_staged(
            output_dir=tmp_path,
            config=config,
            resume=True,
            progress_callback=second_progress.append,
        )
    )

    assert resumed_report.description == first_report.description
    assert second_progress == ["Resuming final comprehensive report from checkpoint."]


def test_default_dialogue_replay_seeds_are_exposed():
    assert DEFAULT_DIALOGUE_REPLAY_SEEDS == (0, 1, 2)


def test_default_dialogue_real_proof_config_requires_multiple_canonical_cases():
    config = default_dialogue_real_proof_config()

    assert isinstance(config, DialogueRealComprehensiveBenchmarkConfig)
    assert config.profile_labels == default_dialogue_strong_proof_profiles()
    assert config.open_profile_labels == default_open_dialogue_ablation_profiles()
    assert config.canonical_case_limit == 2
    assert config.open_scenario_limit == 1
    assert config.proof_min_canonical_cases == 2
    assert config.runtime_mode == LocalSubstrateRuntimeMode.BUILTIN_ONLY


def test_pe_drive_off_runner_disables_external_prediction_error_drive():
    case = DEFAULT_DIALOGUE_PROOF_CASES[0]
    runner = build_standard_dialogue_runner(profile_label="pe-drive-off", case=case)

    assert runner._external_prediction_error_drive is False
    assert runner._joint_schedule.pe_ssl_threshold < 900.0


def test_timescale_off_runner_uses_non_nested_memory_profile():
    case = DEFAULT_DIALOGUE_PROOF_CASES[0]
    runner = build_standard_dialogue_runner(profile_label="timescale-off", case=case)
    memory_snapshot = runner._memory_store.snapshot(retrieved_entries=())
    lifecycle_metrics = dict(memory_snapshot.lifecycle_metrics)

    assert lifecycle_metrics["nested_profile_active"] == 0.0


def test_eta_off_runner_uses_learned_lite_temporal_policy_and_no_joint_learning():
    case = DEFAULT_DIALOGUE_PROOF_CASES[0]
    runner = build_standard_dialogue_runner(profile_label="eta-off", case=case)

    assert runner._temporal_policy.mode.value == "learned-lite"
    assert runner._joint_schedule.ssl_interval == 0
    assert runner._joint_schedule.rl_interval == 0
    assert runner._external_prediction_error_drive is False
    assert runner._external_prediction_error_drive is False


def test_scaffold_ablation_runners_expose_expected_temporal_controls():
    case = DEFAULT_DIALOGUE_PROOF_CASES[0]
    no_label_runner = build_standard_dialogue_runner(
        profile_label="pe-eta-no-semantic-label",
        case=case,
    )
    no_reflection_cache_runner = build_standard_dialogue_runner(
        profile_label="pe-eta-no-reflection-cache",
        case=case,
    )

    assert no_label_runner._world_temporal_policy.mode.value == "full-learned"
    assert no_reflection_cache_runner._world_temporal_policy.mode.value == "full-learned"
    assert no_label_runner._external_prediction_error_drive is True
    assert no_reflection_cache_runner._external_prediction_error_drive is True


def test_pe_readout_only_runner_keeps_pe_visible_without_primary_dominance():
    case = DEFAULT_DIALOGUE_PROOF_CASES[0]
    runner = build_standard_dialogue_runner(
        profile_label="pe-eta-pe-readout-only",
        case=case,
    )

    assert runner._external_prediction_error_drive is False
    assert runner._prediction_error_readout_only is True
    assert runner._primary_prediction_error_dominance_enabled is False
    assert runner._joint_loop.primary_prediction_error_dominance_enabled is False
    assert runner._joint_loop._world_sandbox._env.primary_prediction_error_enabled is False


def test_scaffold_ablation_profiles_preserve_latent_family_and_pe_schedule():
    report = asyncio.run(
        run_dialogue_pe_eta_ablation_benchmark(
            cases=DEFAULT_DIALOGUE_PROOF_CASES[:1],
            profile_labels=("pe-eta", "pe-eta-no-semantic-label", "pe-eta-no-reflection-cache"),
            runner_factory=_synthetic_ablation_runner,
        )
    )

    path_reports = {
        path.path_label: path.benchmark_report.case_reports[0]
        for path in report.path_reports
    }
    no_label = path_reports["pe-eta-no-semantic-label"]
    no_reflection_cache = path_reports["pe-eta-no-reflection-cache"]

    assert max(turn.action_family_version for turn in no_label.turns) > 0
    assert no_label.pe_triggered_turn_count > 0
    assert all(turn.active_abstract_action.startswith("latent-family-v") for turn in no_label.turns)

    assert max(turn.action_family_version for turn in no_reflection_cache.turns) > 0
    assert no_reflection_cache.pe_triggered_turn_count > 0
    assert no_reflection_cache.pressure_response_precision >= 0.0
    assert no_reflection_cache.stability_after_recovery_score >= 0.0

    strong_proof_metric_deltas = dict(report.strong_proof_metric_deltas)
    assert "pe-eta-no-semantic-label" in strong_proof_metric_deltas
    assert "pe-eta-no-reflection-cache" in strong_proof_metric_deltas


def test_strong_proof_deltas_surface_scaffold_and_pe_readout_profiles_explicitly():
    report = asyncio.run(
        run_dialogue_pe_eta_ablation_benchmark(
            cases=DEFAULT_DIALOGUE_PROOF_CASES[:1],
            profile_labels=(
                "pe-eta",
                "pe-eta-no-semantic-label",
                "pe-eta-no-reflection-cache",
                "pe-eta-pe-readout-only",
            ),
            runner_factory=_synthetic_ablation_runner,
        )
    )

    strong_case_map = {
        case_id: dict(path_deltas)
        for case_id, path_deltas in report.strong_proof_case_deltas
    }
    assert "repair" in strong_case_map
    assert {
        "pe-eta-no-semantic-label",
        "pe-eta-no-reflection-cache",
        "pe-eta-pe-readout-only",
    } <= set(strong_case_map["repair"])
    strong_metric_map = dict(report.strong_proof_metric_deltas)
    assert "pe-eta-no-semantic-label" in strong_metric_map
    assert "pe-eta-no-reflection-cache" in strong_metric_map
    assert "pe-eta-pe-readout-only" in strong_metric_map


def test_pe_readout_only_profile_preserves_prediction_chain_but_reduces_pe_trigger_dominance():
    report = asyncio.run(
        run_dialogue_pe_eta_ablation_benchmark(
            cases=DEFAULT_DIALOGUE_PROOF_CASES[:1],
            profile_labels=("pe-eta", "pe-eta-pe-readout-only"),
            runner_factory=_synthetic_ablation_runner,
        )
    )

    path_reports = {
        path.path_label: path.benchmark_report.case_reports[0]
        for path in report.path_reports
    }
    pe_eta = path_reports["pe-eta"]
    pe_readout_only = path_reports["pe-eta-pe-readout-only"]
    delta_map = dict(report.case_deltas_from_baseline[0][1])

    assert pe_readout_only.prediction_chain_turn_count > 0
    assert max(turn.action_family_version for turn in pe_readout_only.turns) > 0
    assert pe_readout_only.pe_triggered_turn_count <= pe_eta.pe_triggered_turn_count
    assert dict(delta_map["pe-eta-pe-readout-only"])["pe_triggered_turn_count"] <= 0.0


def test_pe_dominance_comparison_report_quantifies_mechanism_retention():
    comparison = asyncio.run(
        run_dialogue_pe_eta_ablation_benchmark(
            cases=DEFAULT_DIALOGUE_PROOF_CASES[:1],
            profile_labels=("pe-eta", "pe-drive-off", "pe-eta-pe-readout-only"),
            runner_factory=_synthetic_ablation_runner,
        )
    )

    report = build_pe_dominance_comparison_report(comparison)

    assert report.baseline_label == "pe-eta"
    assert report.pe_drive_off_label == "pe-drive-off"
    assert report.pe_readout_only_label == "pe-eta-pe-readout-only"
    assert report.mechanism_retention_ratio >= 0.0
    assert report.pe_visibility_retention_ratio > 0.0
    assert isinstance(report.interpretation, str)
    assert "mechanism_retention=" in report.description


def test_pe_dominance_case_diagnosis_report_identifies_worst_case_and_failure_mode():
    comparison = asyncio.run(
        run_dialogue_pe_eta_ablation_benchmark(
            cases=DEFAULT_DIALOGUE_PROOF_CASES[:2],
            profile_labels=("pe-eta", "pe-drive-off", "pe-eta-pe-readout-only"),
            runner_factory=_synthetic_ablation_runner,
        )
    )

    diagnosis = build_pe_dominance_case_diagnosis_report(comparison)

    assert diagnosis.baseline_label == "pe-eta"
    assert diagnosis.pe_drive_off_label == "pe-drive-off"
    assert diagnosis.pe_readout_only_label == "pe-eta-pe-readout-only"
    assert diagnosis.worst_case_id is not None
    assert diagnosis.dominant_failure_mode in {
        "schedule-driven",
        "reward-driven",
        "latent-fragility-driven",
    }
    assert len(diagnosis.case_diagnoses) == 2
    assert diagnosis.case_diagnoses[0].degradation_severity >= diagnosis.case_diagnoses[-1].degradation_severity
    assert "dominant_failure_mode=" in diagnosis.description


def test_default_rare_heavy_candidate_configs_are_exposed():
    configs = default_rare_heavy_candidate_configs()

    assert configs == DEFAULT_RARE_HEAVY_CANDIDATE_CONFIGS
    assert len(configs) >= 3


def test_eta_no_pe_baseline_does_not_receive_interval_carryover_credit():
    report = asyncio.run(
        run_dialogue_pe_eta_ablation_benchmark(
            cases=DEFAULT_DIALOGUE_PROOF_CASES[:1],
            runner_factory=_synthetic_ablation_runner,
        )
    )

    case_id, path_deltas = report.case_deltas_from_baseline[0]
    assert case_id == "repair"
    delta_map = {label: dict(deltas) for label, deltas in path_deltas}

    assert delta_map["pe-eta"]["pe_triggered_turn_count"] == 0.0
    assert delta_map["pe-drive-off"]["pe_triggered_turn_count"] < 0.0


def test_run_multi_artifact_acceptance_benchmark_orders_candidates():
    perturbation_report = asyncio.run(
        run_dialogue_pe_eta_perturbation_benchmark(
            variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
            runner_factory=_synthetic_perturbation_runner,
        )
    )
    ranking = build_dialogue_replay_ranking_report(
        variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
        ablation_report=perturbation_report.ablation_report,
    )
    selection_artifact = build_dialogue_replay_selection_artifact(
        variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
        replay_ranking_report=ranking,
        top_k=2,
    )
    comparison = asyncio.run(
        run_multi_artifact_acceptance_benchmark(
            selection_artifact,
            runner_factory=_synthetic_acceptance_runner,
        )
    )

    assert len(comparison.candidate_reports) == len(DEFAULT_RARE_HEAVY_CANDIDATE_CONFIGS)
    assert comparison.chosen_candidate_label is not None
    assert comparison.candidate_reports[0].candidate_score >= comparison.candidate_reports[-1].candidate_score


def test_run_multi_artifact_acceptance_benchmark_can_choose_accepted_candidate():
    perturbation_report = asyncio.run(
        run_dialogue_pe_eta_perturbation_benchmark(
            variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
            runner_factory=_synthetic_perturbation_runner,
        )
    )
    ranking = build_dialogue_replay_ranking_report(
        variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
        ablation_report=perturbation_report.ablation_report,
    )
    selection_artifact = build_dialogue_replay_selection_artifact(
        variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
        replay_ranking_report=ranking,
        top_k=2,
    )
    comparison = asyncio.run(
        run_multi_artifact_acceptance_benchmark(
            selection_artifact,
            runner_factory=_synthetic_acceptance_runner,
            gate_config=DialogueArtifactAcceptanceGateConfig(
                min_mean_score_delta=-1.0,
                min_passed_case_delta=-10,
                min_positive_case_fraction=0.0,
                min_worst_case_delta=-1.0,
            ),
        )
    )

    assert comparison.chosen_candidate_label is not None
    assert comparison.chosen_accepted is True


def test_artifact_acceptance_can_use_minimal_graded_gain_override_when_only_passed_case_delta_is_missing():
    decision = evaluate_dialogue_artifact_acceptance(
        mean_score_delta=0.42,
        passed_case_delta=0,
        positive_case_fraction=0.8,
        worst_case_delta=0.0,
        substrate_checkpoint_present=True,
        substrate_update_count=2,
        substrate_source_batch_count=2,
        substrate_mean_sequence_length=1.2,
        substrate_mean_residual_magnitude=0.05,
        substrate_import_success_fraction=1.0,
        gate_config=DialogueArtifactAcceptanceGateConfig(),
    )

    assert decision.accepted is True
    assert decision.override_mode == "graded-gain"
    assert "passed-case-delta-below-threshold" not in decision.reasons


def test_artifact_acceptance_rejects_when_graded_gain_still_has_negative_tail():
    decision = evaluate_dialogue_artifact_acceptance(
        mean_score_delta=0.42,
        passed_case_delta=0,
        positive_case_fraction=0.8,
        worst_case_delta=-0.01,
        substrate_checkpoint_present=True,
        substrate_update_count=2,
        substrate_source_batch_count=2,
        substrate_mean_sequence_length=1.2,
        substrate_mean_residual_magnitude=0.05,
        substrate_import_success_fraction=1.0,
        gate_config=DialogueArtifactAcceptanceGateConfig(),
    )

    assert decision.accepted is False
    assert decision.override_mode == "none"
    assert "passed-case-delta-below-threshold" in decision.reasons


def test_replay_selection_acceptance_reports_substrate_evidence():
    perturbation_report = asyncio.run(
        run_dialogue_pe_eta_perturbation_benchmark(
            variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
            profile_labels=("pe-eta", "pe-eta-no-rare-heavy", "pe-drive-off", "eta-off", "timescale-off"),
            runner_factory=_synthetic_perturbation_runner,
        )
    )
    ranking = build_dialogue_replay_ranking_report(
        variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
        ablation_report=perturbation_report.ablation_report,
    )
    selection_artifact = build_dialogue_replay_selection_artifact(
        variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
        replay_ranking_report=ranking,
        top_k=2,
    )

    acceptance_report = asyncio.run(
        run_replay_selection_artifact_acceptance_benchmark(
            selection_artifact,
            runner_factory=_synthetic_acceptance_runner,
        )
    )

    evidence = dict(acceptance_report.substrate_evidence)
    pre_import_evidence = dict(acceptance_report.pre_import_evidence)
    assert acceptance_report.artifact.substrate_checkpoint is not None
    assert evidence["substrate_checkpoint_present"] == 1.0
    assert evidence["substrate_update_count"] >= 1.0
    assert evidence["substrate_import_success_fraction"] == 1.0
    assert "pre_import_pass_fraction" in pre_import_evidence
    assert "mean_pre_import_score_delta" in pre_import_evidence


def test_ablation_report_quantifies_rare_heavy_gap_against_no_rare_heavy():
    report = asyncio.run(
        run_dialogue_pe_eta_ablation_benchmark(
            cases=DEFAULT_DIALOGUE_PROOF_CASES[:1],
            profile_labels=("pe-eta", "pe-eta-no-rare-heavy", "pe-drive-off", "eta-off", "timescale-off"),
            runner_factory=_synthetic_ablation_runner,
        )
    )

    assert report.rare_heavy_metric_deltas
    metric_delta_map = dict(report.rare_heavy_metric_deltas)
    assert metric_delta_map["rare_heavy_applied_count"] >= 0.0
    assert len(report.rare_heavy_case_deltas) == 1
    case_id, case_deltas = report.rare_heavy_case_deltas[0]
    assert case_id == "repair"
    assert "rare_heavy_applied_count" in dict(case_deltas)


def test_replay_ranking_quantifies_gap_vs_no_rare_heavy_when_profile_is_present():
    perturbation_report = asyncio.run(
        run_dialogue_pe_eta_perturbation_benchmark(
            variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
            profile_labels=("pe-eta", "pe-eta-no-rare-heavy", "pe-drive-off", "eta-off", "timescale-off"),
            runner_factory=_synthetic_perturbation_runner,
        )
    )

    ranking = build_dialogue_replay_ranking_report(
        variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
        ablation_report=perturbation_report.ablation_report,
    )

    assert ranking.entries
    assert ranking.mean_gap_vs_no_rare_heavy >= 0.0
    assert all(isinstance(entry.no_rare_heavy_score, float) for entry in ranking.entries)


def test_dialogue_case_report_flags_missing_pe_and_temporal_change():
    case = ScriptedDialogueCase(
        case_id="no-proof",
        description="A degenerate case with no PE or temporal variation.",
        user_inputs=("a", "b", "c"),
    )
    turns = (
        _benchmark_turn(
            turn_index=1,
            pe=0.0,
            reward=0.0,
            action="evidence-only",
            regime="problem_solving",
            abstract_action="steady",
            switch_gate=0.10,
            delayed_metric=0.20,
        ),
        _benchmark_turn(
            turn_index=2,
            pe=0.0,
            reward=0.0,
            action="evidence-only",
            regime="problem_solving",
            abstract_action="steady",
            switch_gate=0.10,
            delayed_metric=0.20,
        ),
        _benchmark_turn(
            turn_index=3,
            pe=0.0,
            reward=0.0,
            action="evidence-only",
            regime="problem_solving",
            abstract_action="steady",
            switch_gate=0.10,
            delayed_metric=0.20,
        ),
    )

    report = build_dialogue_case_report(case=case, turns=turns)

    assert report.passed is False
    assert "high_pe_detected" in report.reasons
    assert "pe_triggered_temporal_response" in report.reasons
    assert "temporal_trajectory_nonconstant" in report.reasons
    assert "delayed_improvement_observed" in report.reasons


def test_dialogue_case_report_passes_when_pe_drives_temporal_and_delayed_change():
    case = ScriptedDialogueCase(
        case_id="proof-positive",
        description="Synthetic positive proof case.",
        user_inputs=("a", "b", "c", "d"),
    )
    turns = (
        _benchmark_turn(
            turn_index=1,
            pe=0.85,
            reward=-0.50,
            action="full-cycle-pe",
            regime="repair_and_deescalation",
            abstract_action="repair",
            switch_gate=0.72,
            delayed_metric=0.20,
            pe_triggered=True,
        ),
        _benchmark_turn(
            turn_index=2,
            pe=0.62,
            reward=-0.30,
            action="ssl-only-pe",
            regime="repair_and_deescalation",
            abstract_action="stabilize",
            switch_gate=0.61,
            delayed_metric=0.36,
            bounded_writeback_applied=True,
            reflection_promotion_eligible=True,
        ),
        _benchmark_turn(
            turn_index=3,
            pe=0.35,
            reward=-0.10,
            action="full-cycle",
            regime="problem_solving",
            abstract_action="plan",
            switch_gate=0.42,
            delayed_metric=0.55,
        ),
        _benchmark_turn(
            turn_index=4,
            pe=0.18,
            reward=0.12,
            action="evidence-only",
            regime="problem_solving",
            abstract_action="plan",
            switch_gate=0.28,
            delayed_metric=0.70,
        ),
    )

    report = build_dialogue_case_report(case=case, turns=turns)

    assert report.passed is True
    checks = dict(report.acceptance_checks)
    assert checks["prediction_chain_present"] is True
    assert checks["pe_triggered_temporal_response"] is True
    assert checks["temporal_trajectory_nonconstant"] is True
    assert checks["delayed_improvement_observed"] is True
    assert report.recovery_lag_turns == 0
    assert report.pressure_localization_score == 1.0
    assert report.over_response_cost == 0.0
    assert report.pressure_response_precision == 1.0
    assert report.pressure_response_recall == 1.0
    assert report.stability_after_recovery_score == 0.0
    assert report.bounded_writeback_turn_count == 1
    assert report.reflection_promotion_eligible_turn_count == 1


def test_dialogue_case_report_counts_cross_turn_pe_trigger():
    case = ScriptedDialogueCase(
        case_id="cross-turn-trigger",
        description="High PE on one turn should justify a PE schedule on the next turn.",
        user_inputs=("a", "b", "c", "d"),
    )
    turns = (
        _benchmark_turn(
            turn_index=1,
            pe=0.24,
            reward=-0.06,
            action="ssl-only",
            regime="problem_solving",
            abstract_action="diagnose",
            switch_gate=0.58,
            delayed_metric=0.25,
        ),
        _benchmark_turn(
            turn_index=2,
            pe=0.12,
            reward=-0.02,
            action="ssl-only-pe",
            regime="guided_exploration",
            abstract_action="reframe",
            switch_gate=0.81,
            delayed_metric=0.38,
        ),
        _benchmark_turn(
            turn_index=3,
            pe=0.10,
            reward=-0.01,
            action="full-cycle",
            regime="problem_solving",
            abstract_action="plan",
            switch_gate=0.43,
            delayed_metric=0.52,
        ),
        _benchmark_turn(
            turn_index=4,
            pe=0.08,
            reward=0.01,
            action="evidence-only",
            regime="problem_solving",
            abstract_action="plan",
            switch_gate=0.29,
            delayed_metric=0.63,
        ),
    )

    report = build_dialogue_case_report(case=case, turns=turns)

    assert report.passed is True
    assert report.pe_triggered_turn_count >= 1
    checks = dict(report.acceptance_checks)
    assert checks["pe_triggered_temporal_response"] is True
    assert report.recovery_lag_turns == 1
    assert report.pressure_localization_score == 1.0
    assert report.over_response_cost == 0.0
    assert report.pressure_response_precision == 1.0
    assert report.pressure_response_recall == 1.0


def test_dialogue_case_report_counts_cross_turn_full_cycle_after_high_pe():
    case = ScriptedDialogueCase(
        case_id="cross-turn-full-cycle",
        description="High PE on one turn should count when the next turn runs a non-evidence cycle.",
        user_inputs=("a", "b", "c"),
    )
    turns = (
        _benchmark_turn(
            turn_index=1,
            pe=0.31,
            reward=-0.08,
            action="ssl-only",
            regime="problem_solving",
            abstract_action="diagnose",
            switch_gate=0.66,
            delayed_metric=0.28,
        ),
        _benchmark_turn(
            turn_index=2,
            pe=0.11,
            reward=-0.02,
            action="full-cycle",
            regime="problem_solving",
            abstract_action="reframe",
            switch_gate=0.74,
            delayed_metric=0.44,
        ),
        _benchmark_turn(
            turn_index=3,
            pe=0.07,
            reward=0.01,
            action="evidence-only",
            regime="problem_solving",
            abstract_action="plan",
            switch_gate=0.32,
            delayed_metric=0.61,
        ),
    )

    report = build_dialogue_case_report(case=case, turns=turns)

    assert report.pe_triggered_turn_count >= 1
    assert dict(report.acceptance_checks)["pe_triggered_temporal_response"] is True
    assert report.recovery_lag_turns == 1
    assert report.pressure_localization_score == 1.0
    assert report.over_response_cost == 0.0
    assert report.pressure_response_precision == 1.0
    assert report.pressure_response_recall == 1.0


def test_dialogue_case_report_quantifies_recovery_lag_and_pressure_localization():
    case = ScriptedDialogueCase(
        case_id="quantitative-proof",
        description="Synthetic case for quantitative response metrics.",
        user_inputs=("a", "b", "c", "d", "e"),
        expected_pressure_turns=(2, 3),
    )
    turns = (
        _benchmark_turn(
            turn_index=1,
            pe=0.02,
            reward=0.0,
            action="evidence-only",
            regime="problem_solving",
            abstract_action="steady",
            switch_gate=0.10,
            delayed_metric=0.20,
        ),
        _benchmark_turn(
            turn_index=2,
            pe=0.26,
            reward=-0.06,
            action="ssl-only",
            regime="problem_solving",
            abstract_action="diagnose",
            switch_gate=0.52,
            delayed_metric=0.24,
        ),
        _benchmark_turn(
            turn_index=3,
            pe=0.22,
            reward=-0.05,
            action="full-cycle",
            regime="guided_exploration",
            abstract_action="reframe",
            switch_gate=0.67,
            delayed_metric=0.40,
        ),
        _benchmark_turn(
            turn_index=4,
            pe=0.21,
            reward=-0.02,
            action="ssl-only",
            regime="problem_solving",
            abstract_action="plan",
            switch_gate=0.33,
            delayed_metric=0.57,
        ),
        _benchmark_turn(
            turn_index=5,
            pe=0.09,
            reward=0.02,
            action="ssl-only-pe",
            regime="problem_solving",
            abstract_action="plan",
            switch_gate=0.78,
            delayed_metric=0.64,
        ),
    )

    report = build_dialogue_case_report(case=case, turns=turns)

    assert report.recovery_lag_turns == 1
    assert report.pressure_localization_score == 2 / 3
    assert report.over_response_cost == 0.2
    assert report.pressure_response_precision == 2 / 3
    assert report.pressure_response_recall == 1.0
    assert report.stability_after_recovery_score == 0.0


def test_goal_drift_case_no_longer_fails_pe_trigger_check_with_synthetic_runner():
    goal_drift_case = next(case for case in DEFAULT_DIALOGUE_PROOF_CASES if case.case_id == "goal_drift")

    report = asyncio.run(
        run_dialogue_pe_eta_case(
            case=goal_drift_case,
            runner=_synthetic_runner(goal_drift_case),
        )
    )

    checks = dict(report.acceptance_checks)
    assert checks["high_pe_detected"] is True
    assert checks["pe_triggered_temporal_response"] is True
