from __future__ import annotations

from volvence_zero.agent.eta_proof_benchmark import (
    ETAInternalRLAcceptanceConfig,
    build_eta_proof_paper_suite_manifest,
    build_default_eta_proof_environment,
    build_eta_internal_rl_assessment,
    default_eta_proof_cases,
    default_eta_proof_profiles,
    default_eta_proof_routes,
    evaluate_eta_internal_rl_acceptance,
    export_eta_internal_rl_paper_suite_artifact_bundle,
    run_eta_internal_rl_backend_robustness_benchmark,
    run_eta_internal_rl_paper_suite,
    run_eta_internal_rl_proof_benchmark,
)
from volvence_zero.internal_rl import InternalRLProofEpisode, InternalRLProofSubgoal, InternalRLSandbox
from volvence_zero.memory import Track
from volvence_zero.substrate import ResidualSequenceStep, SubstrateSnapshot, SurfaceKind, build_training_trace
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


def test_eta_proof_benchmark_exposes_default_cases_and_profiles():
    cases = default_eta_proof_cases()
    routes = default_eta_proof_routes()

    assert len(cases) == 7
    assert len(routes) == len(cases)
    assert sum(1 for case in cases if case.split == "train") == 3
    assert sum(1 for case in cases if case.split == "heldout") == 2
    assert any(case.branch_depth >= 4 for case in cases if case.split == "heldout")
    assert default_eta_proof_profiles() == (
        "full-internal-rl",
        "full-bootstrap-init",
        "full-no-fast-prior",
        "full-no-optimize",
        "full-no-replacement",
        "learned-lite-causal",
        "noop-backend",
    )


def test_default_eta_proof_environment_builds_cases_from_routes():
    environment = build_default_eta_proof_environment()
    route = next(route for route in default_eta_proof_routes() if route.case_id == "heldout-epsilon-beta-alpha-delta")

    case = environment.build_case(route)

    assert case.environment_id == "eta-mini-hierarchy"
    assert case.route_signature[0] == "entry"
    assert case.branch_depth >= 4
    assert len(case.proof_episode.subgoals) == 4
    assert case.proof_episode.distractor_signatures


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
    assert "temporal_fast_prior_strength" in metric_names
    assert "temporal_fast_prior_switch_delta" in metric_names
    assert "bootstrap_init_used" in metric_names
    assert report.profile_reports[0].training_update_count > 0
    assert report.profile_reports[0].rollout_batch_count > 0
    assert report.profile_reports[0].mean_rollouts_per_update >= 1.0
    assert report.profile_reports[0].training_parameter_change_rate > 0.0
    assert report.profile_reports[0].mean_parameter_change_norm >= 0.0
    assert report.profile_reports[0].episode_reports


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


def test_eta_backend_robustness_benchmark_compares_trace_and_synthetic():
    report = run_eta_internal_rl_backend_robustness_benchmark()

    assert report.profile_label == "full-internal-rl"
    assert len(report.profile_reports) == 2
    backend_labels = {profile_report.backend_label for profile_report in report.profile_reports}
    assert backend_labels == {"trace", "synthetic-open-weight"}
    delta_names = {name for name, _ in report.metric_deltas}
    assert "heldout_terminal_success_rate" in delta_names


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
