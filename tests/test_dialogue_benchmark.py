from __future__ import annotations

import asyncio

from volvence_zero.agent import (
    AgentSessionRunner,
    DEFAULT_DIALOGUE_CASE_VARIANTS,
    DEFAULT_DIALOGUE_PARAPHRASE_FAMILIES,
    DEFAULT_DIALOGUE_REPLAY_SEEDS,
    DEFAULT_DIALOGUE_PROOF_CASES,
    DEFAULT_RARE_HEAVY_CANDIDATE_CONFIGS,
    DialogueArtifactAcceptanceGateConfig,
    build_replay_selection_training_traces,
    build_dialogue_replay_selection_artifact,
    build_standard_dialogue_runner,
    default_rare_heavy_candidate_configs,
    default_dialogue_ablation_profiles,
    dialogue_case_variants,
    dialogue_paraphrase_families,
    DialogueBenchmarkTurn,
    ScriptedDialogueCase,
    build_dialogue_case_report,
    build_dialogue_replay_ranking_report,
    generate_stochastic_dialogue_case_variants,
    run_replay_selection_artifact_acceptance_benchmark,
    run_dialogue_pe_eta_ablation_benchmark,
    run_dialogue_pe_eta_benchmark,
    run_dialogue_pe_eta_case,
    run_dialogue_pe_eta_perturbation_benchmark,
    run_dialogue_pe_eta_systematic_replay_benchmark,
    run_multi_artifact_acceptance_benchmark,
    train_rare_heavy_artifact_from_replay_selection,
)
from volvence_zero.joint_loop import JointLoopSchedule
from volvence_zero.substrate import SyntheticOpenWeightResidualRuntime


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
    runtime = SyntheticOpenWeightResidualRuntime(
        model_id=f"dialogue-acceptance:{variant.case.case_id}"
    )
    runtime.runtime_origin = "synthetic-acceptance"
    return build_standard_dialogue_runner(
        profile_label="pe-eta",
        case=variant.case,
        residual_runtime=runtime,
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
        rare_heavy_recommended=pe_triggered,
        rare_heavy_applied=False,
        outcome_metrics=(
            ("learning:joint_learning_progress", delayed_metric),
            ("relationship:cross_track_stability", delayed_metric),
        ),
        description="synthetic benchmark turn",
    )


def test_dialogue_benchmark_exposes_default_scripted_cases():
    case_ids = tuple(case.case_id for case in DEFAULT_DIALOGUE_PROOF_CASES)

    assert case_ids == ("repair", "task_clarification", "repeated_failure", "goal_drift")
    assert all(len(case.user_inputs) >= 6 for case in DEFAULT_DIALOGUE_PROOF_CASES)


def test_dialogue_benchmark_exposes_default_ablation_profiles():
    assert default_dialogue_ablation_profiles() == ("pe-eta", "eta-no-pe", "heuristic-baseline")


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


def test_run_dialogue_pe_eta_benchmark_runs_complete_scripted_suite():
    report = asyncio.run(
        run_dialogue_pe_eta_benchmark(
            runner_factory=_synthetic_runner,
        )
    )

    assert report.total_case_count == len(DEFAULT_DIALOGUE_PROOF_CASES)
    assert len(report.case_reports) == len(DEFAULT_DIALOGUE_PROOF_CASES)
    assert report.description
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
    assert len(report.path_reports) == 3
    assert len(report.case_deltas_from_baseline) == 2
    case_id, path_deltas = report.case_deltas_from_baseline[0]
    assert case_id in {"repair", "task_clarification"}
    delta_map = dict(path_deltas)
    assert "pe-eta" in delta_map
    assert "eta-no-pe" in delta_map
    assert "heuristic-baseline" in delta_map
    assert dict(delta_map["pe-eta"])["passed"] == 0.0
    assert dict(delta_map["eta-no-pe"])["pe_triggered_turn_count"] <= 0.0
    assert "recovery_lag_turns" in dict(delta_map["pe-eta"])
    assert "pressure_localization_score" in dict(delta_map["pe-eta"])
    assert "over_response_cost" in dict(delta_map["pe-eta"])
    assert "pressure_response_precision" in dict(delta_map["pe-eta"])
    assert "pressure_response_recall" in dict(delta_map["pe-eta"])
    assert "stability_after_recovery_score" in dict(delta_map["pe-eta"])


def test_run_dialogue_pe_eta_perturbation_benchmark_collects_variant_reports():
    report = asyncio.run(
        run_dialogue_pe_eta_perturbation_benchmark(
            variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
            runner_factory=_synthetic_perturbation_runner,
        )
    )

    assert len(report.variant_cases) == 2
    assert report.ablation_report.baseline_label == "pe-eta"
    assert len(report.ablation_report.path_reports) == 3
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


def test_default_dialogue_replay_seeds_are_exposed():
    assert DEFAULT_DIALOGUE_REPLAY_SEEDS == (0, 1, 2)


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
    assert delta_map["eta-no-pe"]["pe_triggered_turn_count"] < 0.0


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
