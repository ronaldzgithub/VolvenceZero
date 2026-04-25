from __future__ import annotations

from volvence_zero.agent.eta_proof_benchmark import (
    ETAInternalRLAcceptanceConfig,
    ETAOpenWeightRuntimeConfig,
    ETA_REAL_RESIDUAL_MIN_HOOK_COVERAGE,
    build_eta_open_weight_paper_suite_manifest,
    build_eta_proof_paper_suite_manifest,
    build_default_eta_proof_environment,
    build_eta_internal_rl_assessment,
    _calibrate_case_for_real_snapshots,
    default_eta_proof_cases,
    default_eta_proof_profiles,
    default_eta_proof_routes,
    evaluate_eta_internal_rl_acceptance,
    export_eta_internal_rl_paper_suite_artifact_bundle,
    run_eta_open_weight_residual_benchmark,
    run_eta_internal_rl_backend_robustness_benchmark,
    run_eta_internal_rl_paper_suite,
    run_eta_internal_rl_proof_benchmark,
)
from volvence_zero.integration.final_wiring import FinalRolloutConfig
from volvence_zero.runtime import WiringLevel
from volvence_zero.internal_rl import InternalRLProofEpisode, InternalRLProofSubgoal, InternalRLSandbox
from volvence_zero.memory import Track
from volvence_zero.substrate import (
    ResidualSequenceStep,
    SubstrateSnapshot,
    SurfaceKind,
    SyntheticOpenWeightResidualRuntime,
    build_training_trace,
)
from volvence_zero.temporal.metacontroller_components import summarize_residual_activations


def _snapshot_from_step(trace_id: str, step: object) -> SubstrateSnapshot:
    return SubstrateSnapshot(
        model_id=trace_id,
        is_frozen=True,
        surface_kind=SurfaceKind.RESIDUAL_STREAM,
        token_logits=(0.1, 0.2),
        feature_surface=step.feature_surface,
        residual_activations=step.residual_activations,
        residual_sequence=(
            ResidualSequenceStep(
                step=step.step,
                token=step.token,
                feature_surface=step.feature_surface,
                residual_activations=step.residual_activations,
                description=f"proof token {step.token}",
            ),
        ),
        unavailable_fields=(),
        description=f"proof trace step {step.step}",
    )


class _RecordingOpenWeightRuntime(SyntheticOpenWeightResidualRuntime):
    def __init__(self) -> None:
        super().__init__(model_id="recording-open-weight-runtime")
        self.applied_source_texts: list[str] = []

    def apply_control(self, *, source_text, substrate_snapshot, applied_control, track_scale=(1.0, 1.0, 1.0)):
        self.applied_source_texts.append(source_text)
        return super().apply_control(
            source_text=source_text,
            substrate_snapshot=substrate_snapshot,
            applied_control=applied_control,
            track_scale=track_scale,
        )


def test_internal_rl_proof_rollout_emits_delayed_credit_assignments():
    trace = build_training_trace(trace_id="proof-rollout", source_text="steady guidance alignment planning support")
    snapshots = tuple(_snapshot_from_step(trace.trace_id, step) for step in trace.steps)
    first_signature = summarize_residual_activations(trace.steps[0].residual_activations, trace.steps[0].feature_surface)
    last_signature = summarize_residual_activations(trace.steps[-1].residual_activations, trace.steps[-1].feature_surface)
    proof_episode = InternalRLProofEpisode(
        episode_id="proof-rollout",
        subgoals=(
            InternalRLProofSubgoal(
                subgoal_id="alpha",
                target_signature=first_signature,
                completion_threshold=0.45,
                credit_horizon=2,
            ),
            InternalRLProofSubgoal(
                subgoal_id="beta",
                target_signature=last_signature,
                completion_threshold=0.45,
                credit_horizon=2,
            ),
        ),
    )
    sandbox = InternalRLSandbox()

    rollout = sandbox.rollout(
        rollout_id="proof-rollout",
        substrate_steps=snapshots,
        track=Track.SHARED,
        replacement_mode="causal-binary",
        proof_episode=proof_episode,
    )

    assert rollout.reward_mode == "proof-delayed"
    assert rollout.proof_episode_id == "proof-rollout"
    assert rollout.delayed_credit_assignments
    assert rollout.transitions[0].reward_mode == "proof-sparse"
    assert rollout.transitions[0].raw_reward != rollout.transitions[0].reward or rollout.total_reward != 0.0
    assert any(transition.reward != transition.raw_reward for transition in rollout.transitions)
    assert any(transition.reward > 0.0 for transition in rollout.transitions[:-1])


def test_eta_proof_benchmark_exposes_default_cases_and_profiles():
    cases = default_eta_proof_cases()
    routes = default_eta_proof_routes()

    assert len(cases) == 7
    assert len(routes) == len(cases)
    assert sum(1 for case in cases if case.split == "train") == 3
    assert sum(1 for case in cases if case.split == "heldout") == 2
    assert any(case.branch_depth >= 4 for case in cases if case.split == "heldout")
    assert {case.split_detail for case in cases if case.split == "heldout"} == {
        "heldout-composition",
        "heldout-long-loop",
    }
    assert all(case.reward_taxonomy for case in cases)
    assert all(case.route_length == len(case.route_signature) for case in cases)
    assert all(case.distractor_count == len(case.proof_episode.distractor_signatures) for case in cases)
    assert default_eta_proof_profiles() == (
        "full-internal-rl",
        "full-bootstrap-init",
        "full-no-fast-prior",
        "full-no-optimize",
        "full-no-replacement",
        "learned-lite-causal",
        "noop-backend",
    )


def test_eta_real_residual_calibration_rewrites_subgoal_signatures_owner_side():
    case = default_eta_proof_cases()[0]
    trace = build_training_trace(trace_id="calibration-trace", source_text=case.source_text)
    snapshots = tuple(_snapshot_from_step(trace.trace_id, step) for step in trace.steps)

    calibrated = _calibrate_case_for_real_snapshots(case, snapshots=snapshots, enabled=True)

    assert calibrated.case_id == case.case_id
    assert tuple(subgoal.subgoal_id for subgoal in calibrated.proof_episode.subgoals) == tuple(
        subgoal.subgoal_id for subgoal in case.proof_episode.subgoals
    )
    assert calibrated.proof_episode.subgoals[0].target_signature != case.proof_episode.subgoals[0].target_signature
    assert calibrated.proof_episode.subgoals[0].observation_weight >= case.proof_episode.subgoals[0].observation_weight
    assert "Calibrated against frozen real residual prefixes" in calibrated.proof_episode.description


def test_default_eta_proof_environment_builds_cases_from_routes():
    environment = build_default_eta_proof_environment()
    route = next(route for route in default_eta_proof_routes() if route.case_id == "heldout-epsilon-beta-alpha-delta")

    case = environment.build_case(route)

    assert case.environment_id == "eta-mini-hierarchy"
    assert case.route_signature[0] == "entry"
    assert case.branch_depth >= 4
    assert len(case.proof_episode.subgoals) == 4
    assert case.proof_episode.distractor_signatures
    assert case.split_detail == "heldout-long-loop"
    reward_kinds = {source.component_name: source.kind for source in case.proof_episode.reward_taxonomy}
    assert reward_kinds["proof_terminal_success"] == "terminal"
    assert reward_kinds["proof_subgoal_complete"] == "delayed"
    assert reward_kinds["proof_subgoal_progress"] == "diagnostic"
    assert reward_kinds["proof_observation_alignment"] == "diagnostic"
    assert reward_kinds["proof_intervention_effect"] == "diagnostic"
    assert case.proof_episode.reward_profile == "proof-sparse-terminal-delayed"
    assert case.proof_episode.reward_optimizer_visible("proof_subgoal_progress") is False
    assert case.proof_episode.reward_optimizer_visible("proof_observation_alignment") is False
    assert case.proof_episode.reward_optimizer_visible("proof_intervention_effect") is False


def test_default_eta_proof_environment_supports_reset_and_step():
    environment = build_default_eta_proof_environment()
    route = next(route for route in default_eta_proof_routes() if route.case_id == "train-corridor-alpha-beta-delta")

    state = environment.reset(route)
    observation = environment.observe(state)
    assert state.current_location_id == "entry"
    assert observation.available_targets

    first_step = environment.step(state, target_id="alpha")
    second_step = environment.step(first_step.next_state, target_id="beta")
    final_step = environment.step(second_step.next_state, target_id="delta")

    assert first_step.feedback.route_advanced is True
    assert first_step.feedback.objective_completed is True
    assert second_step.feedback.structural_role in {"corridor", "branch", "return", "loop"}
    assert final_step.next_state.done is True
    assert final_step.next_state.success is True
    assert final_step.next_state.completed_objective_ids == ("alpha", "beta", "delta")


def test_run_eta_internal_rl_proof_benchmark_emits_profile_reports():
    report = run_eta_internal_rl_proof_benchmark(
        profile_labels=("full-internal-rl", "full-bootstrap-init", "full-no-fast-prior"),
        train_epochs=1,
    )

    assert report.baseline_label == "full-internal-rl"
    assert report.backend_label == "trace"
    assert report.rollout_batch_count >= 1
    assert len(report.profile_reports) == 3
    metric_names = {name for name, _ in report.profile_reports[0].metric_means}
    assert "heldout_strong_success_rate" in metric_names
    assert "heldout_strong_success_std" in metric_names
    assert "heldout_family_reuse_rate" in metric_names
    assert "mean_reward_sparsity" in metric_names
    assert "reward_shaping_leakage" in metric_names
    assert "mean_route_length" in metric_names
    assert "mean_distractor_count" in metric_names
    assert "mean_steps_per_abstract_action" in metric_names
    assert "persistence_window_success_rate" in metric_names
    assert "premature_switch_rate" in metric_names
    assert "always_switch_rate" in metric_names
    assert "never_switch_rate" in metric_names
    assert "intervention_application_count" in metric_names
    assert "episode_replacement_effect_delta" in metric_names
    assert "residual_signal_quality" in metric_names
    assert "heldout_family_miss_rate" in metric_names
    assert "heldout_credit_window_miss_rate" in metric_names
    assert "heldout_terminal_credit_coverage" in metric_names
    assert "temporal_fast_prior_strength" in metric_names
    assert "temporal_fast_prior_switch_delta" in metric_names
    assert "credit_to_family_write_count" in metric_names
    assert "long_horizon_payoff_coverage" in metric_names
    assert "family_competition_mean" in metric_names
    assert "bootstrap_init_used" in metric_names
    assert report.profile_reports[0].training_update_count > 0
    assert report.profile_reports[0].rollout_batch_count > 0
    assert report.profile_reports[0].mean_rollouts_per_update >= 1.0
    assert report.profile_reports[0].training_parameter_change_rate > 0.0
    assert report.profile_reports[0].mean_parameter_change_norm >= 0.0
    assert report.profile_reports[0].episode_reports
    first_episode = report.profile_reports[0].episode_reports[0]
    assert first_episode.reward_profile == "proof-sparse-terminal-delayed"
    assert first_episode.reward_source_mix
    reward_components = {component_name for component_name, _, _ in first_episode.reward_source_mix}
    assert "proof_observation_alignment" in reward_components
    assert "proof_intervention_effect" in reward_components
    assert first_episode.reward_shaping_leakage == 0.0
    assert first_episode.reward_sparsity == 1.0
    assert first_episode.mean_steps_per_abstract_action >= 1.0
    assert first_episode.median_steps_per_abstract_action >= 1.0
    assert 0.0 <= first_episode.persistence_window_success_rate <= 1.0
    assert 0.0 <= first_episode.premature_switch_rate <= 1.0
    assert 0.0 <= first_episode.always_switch_rate <= 1.0
    assert 0.0 <= first_episode.never_switch_rate <= 1.0
    assert first_episode.intervention_application_count >= 0
    assert first_episode.mean_replacement_effect_delta >= -1.0
    assert 0.0 <= first_episode.residual_signal_quality <= 1.0
    assert first_episode.first_missed_subgoal
    assert 0.0 <= first_episode.family_miss_rate <= 1.0
    assert 0.0 <= first_episode.credit_window_miss_rate <= 1.0
    assert 0.0 <= first_episode.terminal_credit_coverage <= 1.0


def test_eta_proof_matched_controls_share_case_reward_specs():
    report = run_eta_internal_rl_proof_benchmark(
        profile_labels=("full-internal-rl", "full-no-optimize", "learned-lite-causal"),
        train_epochs=1,
    )

    specs_by_case: dict[str, set[tuple[str, str, int, int]]] = {}
    for profile_report in report.profile_reports:
        for episode_report in profile_report.episode_reports:
            specs_by_case.setdefault(episode_report.case_id, set()).add(
                (
                    episode_report.split_detail,
                    episode_report.reward_profile,
                    episode_report.route_length,
                    episode_report.distractor_count,
                )
            )

    assert specs_by_case
    assert all(len(specs) == 1 for specs in specs_by_case.values())


def test_eta_proof_benchmark_exposes_bootstrap_init_profile() -> None:
    report = run_eta_internal_rl_proof_benchmark(
        profile_labels=("full-internal-rl", "full-bootstrap-init"),
        train_epochs=1,
    )

    metric_map = {
        profile.profile_label: dict(profile.metric_means)
        for profile in report.profile_reports
    }

    assert metric_map["full-internal-rl"]["bootstrap_init_used"] == 0.0
    assert metric_map["full-bootstrap-init"]["bootstrap_init_used"] == 1.0


def test_eta_proof_benchmark_exposes_no_fast_prior_ablation_gaps():
    report = run_eta_internal_rl_proof_benchmark(
        profile_labels=("full-internal-rl", "full-no-fast-prior"),
        train_epochs=1,
    )

    baseline_metrics = dict(next(item.metric_means for item in report.profile_reports if item.profile_label == "full-internal-rl"))
    assert "heldout_family_reuse_gap_vs_no_fast_prior" in baseline_metrics
    assert "heldout_credit_alignment_gap_vs_no_fast_prior" in baseline_metrics
    assert "heldout_strong_success_gap_vs_no_fast_prior" in baseline_metrics
    assert "slow_to_fast_init_benefit" in baseline_metrics
    assert "family_reuse_after_reset" in baseline_metrics
    assert "heldout_gain_after_consolidation" in baseline_metrics
    ablation_deltas = dict(next(values for label, values in report.metric_deltas_from_baseline if label == "full-no-fast-prior"))
    assert "heldout_family_reuse_rate" in ablation_deltas
    assert "heldout_credit_alignment" in ablation_deltas
    assert "heldout_strong_success_rate" in ablation_deltas


def test_eta_proof_benchmark_accumulates_temporal_fast_prior_metrics_after_training():
    report = run_eta_internal_rl_proof_benchmark(
        profile_labels=("full-internal-rl",),
        train_epochs=1,
    )

    metric_map = dict(report.profile_reports[0].metric_means)
    assert metric_map["temporal_fast_prior_strength"] > 0.0
    assert metric_map["temporal_fast_prior_switch_delta"] != 0.0
    assert metric_map["credit_to_family_write_count"] > 0.0
    assert metric_map["long_horizon_payoff_coverage"] > 0.0
    assert metric_map["family_competition_mean"] >= 0.0


def test_eta_backend_robustness_benchmark_compares_trace_and_synthetic():
    report = run_eta_internal_rl_backend_robustness_benchmark()

    assert report.profile_label == "full-internal-rl"
    assert len(report.profile_reports) == 2
    backend_labels = {profile_report.backend_label for profile_report in report.profile_reports}
    assert backend_labels == {"trace", "synthetic-open-weight"}
    delta_names = {name for name, _ in report.metric_deltas}
    assert "heldout_terminal_success_rate" in delta_names


def test_eta_open_weight_residual_benchmark_uses_real_runtime_snapshots():
    config = ETAOpenWeightRuntimeConfig(
        max_prefix_steps=2,
        runtime_mode="builtin-only",
        fallback_mode="allow-builtin",
        local_files_only=False,
        require_real_backend=False,
    )

    report = run_eta_open_weight_residual_benchmark(
        cases=default_eta_proof_cases()[:2],
        profile_labels=("full-internal-rl", "noop-backend"),
        runtime_config=config,
    )

    assert report.backend_label == "transformers-open-weight"
    full_metrics = dict(
        next(profile.metric_means for profile in report.profile_reports if profile.profile_label == "full-internal-rl")
    )
    assert full_metrics["real_open_weight_step_count"] > 0.0
    assert full_metrics["real_open_weight_capture_rate"] == 1.0
    assert full_metrics["real_open_weight_hook_coverage"] >= ETA_REAL_RESIDUAL_MIN_HOOK_COVERAGE
    assert full_metrics["real_open_weight_fallback_rate"] == 1.0
    assert full_metrics["intervention_application_count"] > 0.0
    assert full_metrics["episode_replacement_effect_delta"] >= -1.0
    assert full_metrics["residual_signal_quality"] > 0.0
    assert full_metrics["mean_steps_per_abstract_action"] >= 1.0
    assert 0.0 <= full_metrics["persistence_window_success_rate"] <= 1.0


def test_eta_open_weight_intervention_uses_prefix_aligned_source_texts():
    case = next(case for case in default_eta_proof_cases() if len(case.source_text.split()) >= 5)
    runtime = _RecordingOpenWeightRuntime()

    report = run_eta_open_weight_residual_benchmark(
        cases=(case,),
        profile_labels=("full-internal-rl",),
        runtime=runtime,
        runtime_config=ETAOpenWeightRuntimeConfig(
            max_prefix_steps=2,
            require_real_backend=False,
            calibrate_proof_signatures=False,
        ),
        train_epochs=1,
    )

    metrics = dict(report.profile_reports[0].metric_means)
    assert runtime.applied_source_texts
    assert any(source_text != case.source_text for source_text in runtime.applied_source_texts)
    assert metrics["real_open_weight_intervention_protocol_valid"] == 1.0


def test_eta_open_weight_paper_suite_manifest_and_backend_report_include_real_backend():
    manifest = build_eta_open_weight_paper_suite_manifest(suite_tier="ci-smoke")
    primary_metric_names = {metric.metric_name for metric in manifest.primary_metrics}
    assert "real_residual_policy_gap_vs_control" in primary_metric_names

    report = run_eta_internal_rl_paper_suite(
        manifest=manifest,
        open_weight_config=ETAOpenWeightRuntimeConfig(
            max_prefix_steps=2,
            runtime_mode="builtin-only",
            fallback_mode="allow-builtin",
            local_files_only=False,
            require_real_backend=False,
        ),
    )

    assert report.manifest.suite_kind == "eta-open-weight-residual-proof"
    assert report.reference_benchmark_report is not None
    assert report.reference_benchmark_report.backend_label == "transformers-open-weight"
    assert report.reference_backend_report is not None
    backend_labels = {profile_report.backend_label for profile_report in report.reference_backend_report.profile_reports}
    assert "transformers-open-weight" in backend_labels
    assert "trace" in backend_labels
    assert report.pairwise_effects
    assert report.claim_verdicts
    claim_map = {claim.claim_id: claim for claim in report.claim_verdicts}
    real_claim_evidence = dict(claim_map["claim_eta_real_open_weight_residual_control"].evidence)
    assert "residual_signal_quality" in real_claim_evidence
    assert "episode_replacement_effect_delta" in real_claim_evidence
    assert "real_residual_policy_gap_vs_control" in real_claim_evidence
    assert real_claim_evidence["required_min_hook_coverage"] == ETA_REAL_RESIDUAL_MIN_HOOK_COVERAGE
    assert real_claim_evidence["real_open_weight_hook_coverage"] >= ETA_REAL_RESIDUAL_MIN_HOOK_COVERAGE
    assert real_claim_evidence["real_open_weight_fallback_rate"] == 1.0
    assert claim_map["claim_eta_real_open_weight_residual_control"].status == "fail"


def test_eta_open_weight_real_backend_defaults_fail_closed_without_local_model():
    try:
        run_eta_open_weight_residual_benchmark(
            cases=default_eta_proof_cases()[:1],
            profile_labels=("full-internal-rl",),
            runtime_config=ETAOpenWeightRuntimeConfig(model_id="missing-local-model", max_prefix_steps=1),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        assert "fallback" in str(exc).lower() or type(exc).__name__ in {"OSError", "ValueError"}
    else:
        raise AssertionError("Expected strict real residual config to fail closed without a local transformers model.")


def test_eta_open_weight_real_snapshots_cover_proof_subgoal_budget():
    config = ETAOpenWeightRuntimeConfig(
        max_prefix_steps=1,
        runtime_mode="builtin-only",
        fallback_mode="allow-builtin",
        local_files_only=False,
        require_real_backend=False,
    )

    report = run_eta_open_weight_residual_benchmark(
        cases=default_eta_proof_cases()[-1:],
        profile_labels=("full-internal-rl",),
        runtime_config=config,
    )

    min_steps = sum(max(subgoal.min_persistence, 1) for subgoal in default_eta_proof_cases()[-1].proof_episode.subgoals)
    metrics = dict(report.profile_reports[0].metric_means)
    assert metrics["intervention_application_count"] >= min_steps


def test_final_rollout_config_exposes_eta_open_weight_runtime_gate():
    config = FinalRolloutConfig()

    assert config.level_for("eta_open_weight_runtime", WiringLevel.DISABLED) is WiringLevel.SHADOW
    assert config.is_active("eta_open_weight_runtime") is False

    promoted = FinalRolloutConfig(eta_open_weight_runtime=WiringLevel.ACTIVE)
    assert promoted.is_active("eta_open_weight_runtime") is True


def test_eta_internal_rl_acceptance_fails_closed_without_backend_report():
    benchmark_report = run_eta_internal_rl_proof_benchmark(
        profile_labels=("full-internal-rl", "full-no-optimize"),
        train_epochs=1,
    )
    assessment = build_eta_internal_rl_assessment(benchmark_report=benchmark_report)
    decision = evaluate_eta_internal_rl_acceptance(assessment)

    assert decision.accepted is False
    assert "missing-backend-report" in decision.reasons


def test_eta_internal_rl_assessment_exposes_policy_update_gate():
    benchmark_report = run_eta_internal_rl_proof_benchmark()
    backend_report = run_eta_internal_rl_backend_robustness_benchmark()
    assessment = build_eta_internal_rl_assessment(
        benchmark_report=benchmark_report,
        backend_report=backend_report,
    )

    gate_map = {gate.gate_id: gate for gate in assessment.gates}
    assert "policy-update-evidence" in gate_map
    assert gate_map["policy-update-evidence"].passed is True
    assert "statistical-batch-evidence" in gate_map
    statistical_evidence = dict(gate_map["statistical-batch-evidence"].evidence)
    assert statistical_evidence["mean_rollouts_per_update"] >= 0.0


def test_eta_internal_rl_assessment_exposes_gate_specific_best_control_evidence():
    benchmark_report = run_eta_internal_rl_proof_benchmark()
    backend_report = run_eta_internal_rl_backend_robustness_benchmark()
    assessment = build_eta_internal_rl_assessment(
        benchmark_report=benchmark_report,
        backend_report=backend_report,
    )

    gate_map = {gate.gate_id: gate for gate in assessment.gates}
    success_evidence = dict(gate_map["sparse-reward-success"].evidence)
    reuse_evidence = dict(gate_map["abstract-action-reuse"].evidence)
    composition_evidence = dict(gate_map["heldout-composition"].evidence)

    assert success_evidence["best_control_terminal_success_label"] in {
        "full-no-fast-prior",
        "full-no-optimize",
        "full-no-replacement",
        "learned-lite-causal",
        "noop-backend",
    }
    assert success_evidence["best_control_strong_success_label"] in {
        "full-no-fast-prior",
        "full-no-optimize",
        "full-no-replacement",
        "learned-lite-causal",
        "noop-backend",
    }
    assert reuse_evidence["best_control_reuse_label"] in {
        "full-no-fast-prior",
        "full-no-optimize",
        "full-no-replacement",
        "learned-lite-causal",
        "noop-backend",
    }
    assert reuse_evidence["best_control_credit_alignment_label"] in {
        "full-no-fast-prior",
        "full-no-optimize",
        "full-no-replacement",
        "learned-lite-causal",
        "noop-backend",
    }
    assert composition_evidence["best_control_subgoal_completion_label"] in {
        "full-no-fast-prior",
        "full-no-optimize",
        "full-no-replacement",
        "learned-lite-causal",
        "noop-backend",
    }


def test_eta_proof_benchmark_strong_success_tracks_task_core_more_than_switch_shape():
    benchmark_report = run_eta_internal_rl_proof_benchmark()
    report_map = {
        profile.profile_label: dict(profile.metric_means)
        for profile in benchmark_report.profile_reports
    }

    full_metrics = report_map["full-internal-rl"]
    no_replacement_metrics = report_map["full-no-replacement"]
    noop_metrics = report_map["noop-backend"]

    assert full_metrics["heldout_task_success_core"] == 1.0
    assert no_replacement_metrics["heldout_task_success_core"] == 1.0
    assert abs(
        full_metrics["heldout_strong_success_rate"] - no_replacement_metrics["heldout_strong_success_rate"]
    ) <= 0.02
    assert full_metrics["heldout_mechanism_evidence_score"] > noop_metrics["heldout_mechanism_evidence_score"]
    assert full_metrics["heldout_strong_success_rate"] > noop_metrics["heldout_strong_success_rate"]


def test_eta_internal_rl_default_acceptance_now_passes_with_mechanism_tie_break():
    benchmark_report = run_eta_internal_rl_proof_benchmark()
    backend_report = run_eta_internal_rl_backend_robustness_benchmark()
    assessment = build_eta_internal_rl_assessment(
        benchmark_report=benchmark_report,
        backend_report=backend_report,
    )
    decision = evaluate_eta_internal_rl_acceptance(assessment)
    gate_map = {gate.gate_id: gate for gate in assessment.gates}
    sparse_reward_evidence = dict(gate_map["sparse-reward-success"].evidence)

    assert decision.accepted is True
    assert not decision.reasons
    assert gate_map["sparse-reward-success"].passed is True
    assert sparse_reward_evidence["mechanism_tie_break_margin"] >= 0.02
    assert sparse_reward_evidence["effective_success_margin"] >= 0.02


def test_eta_internal_rl_acceptance_passes_with_relaxed_thresholds():
    benchmark_report = run_eta_internal_rl_proof_benchmark(train_epochs=1)
    backend_report = run_eta_internal_rl_backend_robustness_benchmark()
    assessment = build_eta_internal_rl_assessment(
        benchmark_report=benchmark_report,
        backend_report=backend_report,
    )
    assert assessment.passed_gate_count >= 1

    decision = evaluate_eta_internal_rl_acceptance(
        assessment,
        config=ETAInternalRLAcceptanceConfig(
            required_gate_ids=tuple(gate.gate_id for gate in assessment.gates if gate.passed),
            min_success_delta=-1.0,
            min_terminal_success_rate=0.0,
            min_reuse_rate=0.0,
            min_credit_alignment=0.0,
            min_strong_success_rate=0.0,
            max_backend_success_gap=1.0,
            min_policy_update_rate=0.0,
            min_rollouts_per_update=0.0,
            min_parameter_change_norm=0.0,
            min_replacement_effect_delta=0.0,
            max_heldout_success_std=1.0,
            min_passed_gate_count=max(assessment.passed_gate_count, 1),
        ),
    )

    assert decision.accepted is True


def test_eta_proof_paper_suite_manifest_freezes_primary_metrics():
    manifest = build_eta_proof_paper_suite_manifest(suite_tier="paper-suite-small")

    assert manifest.suite_kind == "eta-internal-rl-proof"
    assert manifest.baseline_label == "full-internal-rl"
    assert manifest.repeat_count == 5
    assert manifest.seed_schedule == (0, 1, 2, 3, 4)
    assert any(metric.metric_name == "heldout_strong_success_rate" for metric in manifest.primary_metrics)
    assert any(metric.metric_name == "mean_rollouts_per_update" for metric in manifest.secondary_metrics)


def test_run_eta_internal_rl_paper_suite_emits_interval_summaries(tmp_path):
    manifest = build_eta_proof_paper_suite_manifest(suite_tier="ci-smoke")

    report = run_eta_internal_rl_paper_suite(
        manifest=manifest,
        output_dir=tmp_path,
    )

    assert report.manifest.suite_id == "eta-proof-ci-smoke"
    assert report.run_summaries
    assert report.primary_metric_summaries
    assert any(summary.metric_name == "mean_rollouts_per_update" for summary in report.secondary_metric_summaries)
    assert report.pairwise_effects
    assert report.claim_verdicts
    claim_ids = {claim.claim_id for claim in report.claim_verdicts}
    assert "claim_eta_internal_rl_sparse_reward_advantage" in claim_ids
    assert "claim_eta_scaffold_free_temporal_abstraction" in claim_ids
    assert "claim_nl_slow_loop_improves_eta_fast_path" in claim_ids
    nl_claim = next(claim for claim in report.claim_verdicts if claim.claim_id == "claim_nl_slow_loop_improves_eta_fast_path")
    nl_evidence = dict(nl_claim.evidence)
    assert "credit_to_family_write_count" in nl_evidence
    assert "long_horizon_payoff_coverage" in nl_evidence
    assert report.interpretation_summary is not None
    assert report.interpretation_summary.interpretation
    assert report.interpretation_summary.review_summary
    assert report.interpretation_summary.dominant_failure_mode
    assert report.interpretation_summary.strongest_competing_control
    assert report.interpretation_summary.strongest_competing_control in {
        "full-no-fast-prior",
        "full-no-optimize",
        "full-no-replacement",
        "learned-lite-causal",
        "noop-backend",
    }
    assert report.provenance.manifest_hash
    assert report.reference_assessment is not None


def test_eta_paper_suite_artifact_bundle_exports_reference_reports(tmp_path):
    manifest = build_eta_proof_paper_suite_manifest(suite_tier="ci-smoke")
    report = run_eta_internal_rl_paper_suite(manifest=manifest)

    written_paths = export_eta_internal_rl_paper_suite_artifact_bundle(
        report,
        output_dir=tmp_path,
    )

    assert written_paths
    assert (tmp_path / "paper_suite_manifest.json").exists()
    assert (tmp_path / "reference_benchmark_report.json").exists()
    assert (tmp_path / "reference_assessment.json").exists()
    assert (tmp_path / "paper_suite_interpretation_summary.json").exists()
    assert (tmp_path / "evidence_bundle.json").exists()
