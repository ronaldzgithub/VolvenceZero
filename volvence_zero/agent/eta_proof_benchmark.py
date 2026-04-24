from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from random import Random
from typing import Any

from volvence_zero.agent.paper_suite import (
    ClaimVerdict,
    EvidenceBundle,
    PaperMetricSpec,
    PaperProfileSpec,
    PaperSuiteManifest,
    PaperSuiteProvenance,
    collect_paper_suite_provenance,
    export_json_artifact,
)
from volvence_zero.evaluation.backbone import (
    MetricIntervalSummary,
    PairwiseMetricEffect,
    build_pairwise_metric_effect,
    build_metric_interval_summaries,
)

from volvence_zero.internal_rl import (
    HierarchicalLocation,
    HierarchicalRouteSpec,
    HierarchicalTransition,
    InternalRLEnvironment,
    InternalRLProofEpisode,
    InternalRLSandbox,
    MiniHierarchicalEnvironment,
    ZRollout,
)
from volvence_zero.memory import Track
from volvence_zero.substrate import (
    NoOpResidualInterventionBackend,
    ResidualSequenceStep,
    SubstrateSnapshot,
    SurfaceKind,
    SyntheticOpenWeightResidualRuntime,
    build_training_trace,
)
from volvence_zero.temporal import (
    FullLearnedTemporalPolicy,
    LearnedLiteTemporalPolicy,
    MetacontrollerSSLTrainer,
)


@dataclass(frozen=True)
class ETAProofCase:
    case_id: str
    source_text: str
    proof_episode: InternalRLProofEpisode
    split: str
    description: str
    environment_id: str = "proof-grid"
    route_signature: tuple[str, ...] = ()
    branch_depth: int = 0


@dataclass(frozen=True)
class ETAProofEpisodeReport:
    case_id: str
    split: str
    profile_label: str
    backend_label: str
    total_reward: float
    raw_total_reward: float
    terminal_success: bool
    completed_subgoals: tuple[str, ...]
    completed_family_ids: tuple[str, ...]
    subgoal_completion_rate: float
    family_reuse_rate: float
    switch_sparsity: float
    mean_persistence: float
    credit_alignment: float
    backend_fidelity: float
    action_family_count: int
    delayed_credit_assignment_count: int
    task_success_core: float
    switch_discipline_score: float
    mechanism_evidence_score: float
    strong_success_score: float
    description: str


@dataclass(frozen=True)
class ETAProofProfileReport:
    profile_label: str
    backend_label: str
    episode_reports: tuple[ETAProofEpisodeReport, ...]
    metric_means: tuple[tuple[str, float], ...]
    training_update_count: int
    rollout_batch_count: int
    mean_rollouts_per_update: float
    training_transition_count: int
    training_parameter_change_count: int
    training_parameter_change_rate: float
    mean_parameter_change_norm: float
    mean_value_loss: float
    mean_replacement_effect_delta: float
    description: str


@dataclass(frozen=True)
class ETAProofBenchmarkReport:
    profile_reports: tuple[ETAProofProfileReport, ...]
    baseline_label: str
    backend_label: str
    metric_deltas_from_baseline: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]
    rollout_batch_count: int
    description: str


@dataclass(frozen=True)
class ETAProofBackendComparisonReport:
    profile_label: str
    profile_reports: tuple[ETAProofProfileReport, ...]
    metric_deltas: tuple[tuple[str, float], ...]
    description: str


@dataclass(frozen=True)
class ETAInternalRLAcceptanceGate:
    gate_id: str
    passed: bool
    evidence: tuple[tuple[str, float | str], ...]
    description: str


@dataclass(frozen=True)
class ETAInternalRLAssessmentReport:
    benchmark_report: ETAProofBenchmarkReport
    backend_report: ETAProofBackendComparisonReport | None
    gates: tuple[ETAInternalRLAcceptanceGate, ...]
    passed_gate_count: int
    total_gate_count: int
    description: str


@dataclass(frozen=True)
class ETAProofPaperSuiteRunSummary:
    run_id: str
    run_seed: int
    metric_values: tuple[tuple[str, float], ...]
    description: str
    strongest_competing_control: str = "none"
    strongest_control_gap: float = 0.0


@dataclass(frozen=True)
class ETAProofPaperSuiteInterpretationSummary:
    interpretation: str
    review_summary: str
    dominant_failure_mode: str
    strongest_metric: str
    weakest_metric: str
    strongest_competing_control: str
    strongest_control_gap: float
    cross_run_gap_mean: float
    cross_run_gap_std: float
    confidence: float
    description: str


@dataclass(frozen=True)
class ETAProofPaperSuiteAggregateReport:
    manifest: PaperSuiteManifest
    provenance: PaperSuiteProvenance
    run_summaries: tuple[ETAProofPaperSuiteRunSummary, ...]
    reference_benchmark_report: ETAProofBenchmarkReport | None
    reference_backend_report: ETAProofBackendComparisonReport | None
    reference_assessment: ETAInternalRLAssessmentReport | None
    primary_metric_summaries: tuple[MetricIntervalSummary, ...]
    secondary_metric_summaries: tuple[MetricIntervalSummary, ...]
    interpretation_summary: ETAProofPaperSuiteInterpretationSummary | None
    description: str
    pairwise_effects: tuple[PairwiseMetricEffect, ...] = ()
    claim_verdicts: tuple[ClaimVerdict, ...] = ()


@dataclass(frozen=True)
class ETAInternalRLAcceptanceConfig:
    required_gate_ids: tuple[str, ...] = (
        "sparse-reward-success",
        "abstract-action-reuse",
        "heldout-composition",
        "credit-alignment",
        "policy-update-evidence",
        "statistical-batch-evidence",
        "backend-robustness",
    )
    min_success_delta: float = 0.02
    min_terminal_success_rate: float = 0.0
    min_reuse_rate: float = 0.50
    min_credit_alignment: float = 0.50
    min_strong_success_rate: float = 0.30
    max_backend_success_gap: float = 0.30
    min_policy_update_rate: float = 0.50
    min_rollouts_per_update: float = 2.0
    min_parameter_change_norm: float = 0.01
    min_replacement_effect_delta: float = 0.05
    max_heldout_success_std: float = 0.35
    min_passed_gate_count: int = 7


@dataclass(frozen=True)
class ETAInternalRLAcceptanceDecision:
    accepted: bool
    reasons: tuple[str, ...]
    accepted_gate_ids: tuple[str, ...]
    blocked_gate_ids: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class ETAProofProfileConfig:
    profile_label: str
    replacement_mode: str
    optimize_after_rollout: bool
    policy_kind: str
    use_noop_backend: bool = False
    use_temporal_fast_prior: bool = True
    bootstrap_init: bool = False


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


def _mean(values: tuple[float, ...]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: tuple[float, ...]) -> float:
    if len(values) <= 1:
        return 0.0
    mean_value = _mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return variance ** 0.5


def _switch_discipline_score(*, switch_sparsity: float, mean_persistence: float) -> float:
    transition_presence = max(0.0, min(1.0, (1.0 - switch_sparsity) / 0.18))
    persistence_quality = max(0.0, min(1.0, mean_persistence / 2.0))
    return max(0.0, min(1.0, transition_presence * 0.40 + persistence_quality * 0.60))


def _subgoal_signature_library() -> dict[str, tuple[float, ...]]:
    return {
        "alpha": (0.92, 0.18, 0.36),
        "beta": (0.28, 0.86, 0.34),
        "gamma": (0.42, 0.40, 0.94),
        "delta": (0.74, 0.62, 0.22),
        "epsilon": (0.18, 0.70, 0.88),
    }


def build_default_eta_proof_environment() -> MiniHierarchicalEnvironment:
    library = _subgoal_signature_library()
    return MiniHierarchicalEnvironment(
        env_id="eta-mini-hierarchy",
        entry_location_id="entry",
        locations=(
            HierarchicalLocation(
                location_id="entry",
                role="entry",
                description="Environment entry anchor.",
            ),
            HierarchicalLocation(
                location_id="hub",
                role="hub",
                description="Central loop/hub used by multiple routes.",
            ),
            HierarchicalLocation(
                location_id="alpha",
                role="objective",
                target_signature=library["alpha"],
                completion_threshold=0.70,
                min_persistence=1,
                credit_horizon=2,
                observation_weight=0.30,
                effect_weight=0.48,
                control_weight=0.22,
                description="Alpha branch anchor.",
            ),
            HierarchicalLocation(
                location_id="beta",
                role="objective",
                target_signature=library["beta"],
                completion_threshold=0.74,
                min_persistence=2,
                credit_horizon=3,
                observation_weight=0.22,
                effect_weight=0.56,
                control_weight=0.22,
                description="Beta corridor objective.",
            ),
            HierarchicalLocation(
                location_id="gamma",
                role="objective",
                target_signature=library["gamma"],
                completion_threshold=0.78,
                min_persistence=2,
                credit_horizon=3,
                observation_weight=0.20,
                effect_weight=0.58,
                control_weight=0.22,
                description="Gamma high-branch objective.",
            ),
            HierarchicalLocation(
                location_id="delta",
                role="objective",
                target_signature=library["delta"],
                completion_threshold=0.76,
                min_persistence=2,
                credit_horizon=3,
                observation_weight=0.22,
                effect_weight=0.56,
                control_weight=0.22,
                description="Delta branch-exit objective.",
            ),
            HierarchicalLocation(
                location_id="epsilon",
                role="objective",
                target_signature=library["epsilon"],
                completion_threshold=0.80,
                min_persistence=3,
                credit_horizon=3,
                observation_weight=0.20,
                effect_weight=0.56,
                control_weight=0.24,
                description="Epsilon hard terminal objective.",
            ),
        ),
        transitions=(
            HierarchicalTransition("entry", "alpha", structural_role="corridor"),
            HierarchicalTransition("entry", "beta", structural_role="branch"),
            HierarchicalTransition("entry", "delta", structural_role="branch"),
            HierarchicalTransition("entry", "hub", structural_role="branch"),
            HierarchicalTransition("alpha", "beta", structural_role="corridor"),
            HierarchicalTransition("alpha", "delta", structural_role="branch"),
            HierarchicalTransition("alpha", "gamma", structural_role="branch"),
            HierarchicalTransition("beta", "gamma", structural_role="corridor"),
            HierarchicalTransition("beta", "delta", structural_role="branch"),
            HierarchicalTransition("beta", "alpha", structural_role="return"),
            HierarchicalTransition("beta", "epsilon", structural_role="branch"),
            HierarchicalTransition("gamma", "alpha", structural_role="return"),
            HierarchicalTransition("gamma", "beta", structural_role="return"),
            HierarchicalTransition("gamma", "epsilon", structural_role="branch"),
            HierarchicalTransition("delta", "alpha", structural_role="return"),
            HierarchicalTransition("delta", "beta", structural_role="corridor"),
            HierarchicalTransition("delta", "hub", structural_role="loop"),
            HierarchicalTransition("delta", "gamma", structural_role="branch"),
            HierarchicalTransition("delta", "epsilon", structural_role="branch"),
            HierarchicalTransition("hub", "beta", structural_role="corridor"),
            HierarchicalTransition("hub", "epsilon", structural_role="branch"),
            HierarchicalTransition("epsilon", "beta", structural_role="return"),
            HierarchicalTransition("epsilon", "alpha", structural_role="return"),
            HierarchicalTransition("epsilon", "delta", structural_role="return"),
        ),
        description="Miniature hierarchical environment with corridor, branch, and loop structure for ETA proof tasks.",
    )


def default_eta_proof_routes() -> tuple[HierarchicalRouteSpec, ...]:
    return (
        HierarchicalRouteSpec(
            case_id="train-corridor-alpha-beta-delta",
            split="train",
            source_text="steady guidance alignment planning support corridor branch anchor",
            waypoints=("entry", "alpha", "beta", "delta"),
            distractor_ids=("gamma", "epsilon"),
            description="Canonical branching corridor task with alpha -> beta -> delta.",
        ),
        HierarchicalRouteSpec(
            case_id="train-zigzag-beta-gamma-alpha",
            split="train",
            source_text="careful repair planning warmth continuity zigzag support",
            waypoints=("entry", "beta", "gamma", "alpha"),
            distractor_ids=("delta",),
            description="Zigzag training task that revisits familiar signatures under order pressure.",
        ),
        HierarchicalRouteSpec(
            case_id="train-loop-delta-beta-epsilon",
            split="train",
            source_text="reflective planning support loop memory return and repair",
            waypoints=("entry", "delta", "hub", "beta", "epsilon"),
            distractor_ids=("gamma",),
            description="Looped training task that pushes persistence before the final subgoal.",
        ),
        HierarchicalRouteSpec(
            case_id="eval-alpha-delta-epsilon",
            split="eval",
            source_text="steady continuity planning support reflection branch return",
            waypoints=("entry", "alpha", "delta", "epsilon"),
            distractor_ids=("beta", "gamma"),
            description="Evaluation task that mixes known branch nodes with a new terminal demand.",
        ),
        HierarchicalRouteSpec(
            case_id="eval-gamma-beta-delta",
            split="eval",
            source_text="careful guidance through dense branch with repair continuity",
            waypoints=("entry", "beta", "gamma", "beta", "delta"),
            distractor_ids=("alpha", "epsilon"),
            description="Evaluation task that reuses beta/delta under a denser distractor field.",
        ),
        HierarchicalRouteSpec(
            case_id="heldout-delta-alpha-gamma-epsilon",
            split="heldout",
            source_text="reflective planning support with careful guidance through heldout branch",
            waypoints=("entry", "delta", "alpha", "gamma", "epsilon"),
            distractor_ids=("beta",),
            description="Held-out compositional route with four-stage reordering and a hard terminal node.",
        ),
        HierarchicalRouteSpec(
            case_id="heldout-epsilon-beta-alpha-delta",
            split="heldout",
            source_text="careful branching return support and reflection in heldout loop",
            waypoints=("entry", "hub", "epsilon", "beta", "alpha", "delta"),
            distractor_ids=("gamma",),
            description="Held-out loop route requiring long-horizon reuse before the final branch exit.",
        ),
    )


def default_eta_proof_cases() -> tuple[ETAProofCase, ...]:
    environment = build_default_eta_proof_environment()
    return tuple(
        ETAProofCase(
            case_id=case.case_id,
            source_text=case.source_text,
            proof_episode=case.proof_episode,
            split=case.split,
            description=case.description,
            environment_id=case.environment_id,
            route_signature=case.route_signature,
            branch_depth=case.branch_depth,
        )
        for case in (environment.build_case(route) for route in default_eta_proof_routes())
    )


def default_eta_proof_profiles() -> tuple[str, ...]:
    return (
        "full-internal-rl",
        "full-bootstrap-init",
        "full-no-fast-prior",
        "full-no-optimize",
        "full-no-replacement",
        "learned-lite-causal",
        "noop-backend",
    )


def build_eta_proof_paper_suite_manifest(
    *,
    suite_tier: str = "paper-suite-small",
) -> PaperSuiteManifest:
    if suite_tier == "ci-smoke":
        repeat_count = 1
        seed_schedule = (0,)
        route_ids = tuple(route.case_id for route in default_eta_proof_routes()[:4])
        train_epochs = 1
        backend_labels = ("trace",)
    elif suite_tier == "paper-suite-full":
        repeat_count = 20
        seed_schedule = tuple(range(repeat_count))
        route_ids = tuple(route.case_id for route in default_eta_proof_routes())
        train_epochs = 2
        backend_labels = ("trace", "synthetic-open-weight")
    elif suite_tier == "paper-suite-small":
        repeat_count = 5
        seed_schedule = tuple(range(repeat_count))
        route_ids = tuple(route.case_id for route in default_eta_proof_routes())
        train_epochs = 1
        backend_labels = ("trace", "synthetic-open-weight")
    else:
        raise ValueError(f"Unsupported ETA proof paper suite tier {suite_tier!r}.")
    return PaperSuiteManifest(
        suite_id=f"eta-proof-{suite_tier}",
        suite_kind="eta-internal-rl-proof",
        suite_tier=suite_tier,
        version=1,
        baseline_label="full-internal-rl",
        repeat_count=repeat_count,
        seed_schedule=seed_schedule,
        profiles=tuple(
            PaperProfileSpec(
                profile_label=profile_label,
                role="baseline" if profile_label == "full-internal-rl" else "matched-control",
                description=f"ETA proof profile {profile_label}.",
            )
            for profile_label in default_eta_proof_profiles()
        ),
        primary_metrics=(
            PaperMetricSpec(
                metric_name="heldout_terminal_success_rate",
                role="primary",
                direction="higher-is-better",
                description="Held-out terminal success rate for the full internal-RL profile.",
            ),
            PaperMetricSpec(
                metric_name="heldout_strong_success_rate",
                role="primary",
                direction="higher-is-better",
                description="Held-out strong sparse-reward success rate for the full internal-RL profile.",
            ),
            PaperMetricSpec(
                metric_name="heldout_family_reuse_rate",
                role="primary",
                direction="higher-is-better",
                description="Held-out abstract-action reuse rate for the full internal-RL profile.",
            ),
            PaperMetricSpec(
                metric_name="heldout_credit_alignment",
                role="primary",
                direction="higher-is-better",
                description="Held-out delayed-credit alignment for the full internal-RL profile.",
            ),
            PaperMetricSpec(
                metric_name="strong_success_gap_vs_best_control",
                role="primary",
                direction="higher-is-better",
                description="Held-out strong success gap between the full profile and the strongest matched control.",
            ),
            PaperMetricSpec(
                metric_name="backend_success_gap",
                role="primary",
                direction="lower-is-better",
                description="Gap in held-out strong success across backends for the full profile.",
            ),
        ),
        secondary_metrics=(
            PaperMetricSpec(
                metric_name="assessment_pass_fraction",
                role="secondary",
                direction="higher-is-better",
                description="Fraction of ETA acceptance gates passed in the assessment report.",
            ),
            PaperMetricSpec(
                metric_name="policy_update_rate",
                role="secondary",
                direction="higher-is-better",
                description="Observed policy parameter update rate for the full profile.",
            ),
            PaperMetricSpec(
                metric_name="heldout_subgoal_completion_rate",
                role="secondary",
                direction="higher-is-better",
                description="Held-out subgoal completion rate for the full profile.",
            ),
            PaperMetricSpec(
                metric_name="heldout_strong_success_std",
                role="secondary",
                direction="lower-is-better",
                description="Held-out strong success variability across held-out cases.",
            ),
            PaperMetricSpec(
                metric_name="mean_rollouts_per_update",
                role="secondary",
                direction="higher-is-better",
                description="Average rollout count consumed by each full-profile update.",
            ),
            PaperMetricSpec(
                metric_name="mean_parameter_change_norm",
                role="secondary",
                direction="higher-is-better",
                description="Average parameter-change norm for full-profile internal RL updates.",
            ),
            PaperMetricSpec(
                metric_name="mean_replacement_effect_delta",
                role="secondary",
                direction="higher-is-better",
                description="Average replacement-effect delta observed during full-profile updates.",
            ),
            PaperMetricSpec(
                metric_name="mean_value_loss",
                role="secondary",
                direction="lower-is-better",
                description="Average critic/value loss for the full internal-RL profile.",
            ),
            PaperMetricSpec(
                metric_name="training_transition_count",
                role="secondary",
                direction="higher-is-better",
                description="Total training transitions consumed by full-profile internal RL updates.",
            ),
        ),
        case_groups=(
            ("route_ids", route_ids),
            ("backend_labels", backend_labels),
            ("train_epochs", (str(train_epochs),)),
        ),
        artifact_expectations=(
            "per-run eta proof benchmark json",
            "per-run backend robustness json",
            "per-run assessment json",
            "aggregate summary json",
            "provenance json",
            "evidence bundle json",
        ),
        description=(
            f"Frozen ETA proof paper suite {suite_tier} with {repeat_count} repeated runs "
            f"and backends={backend_labels}."
        ),
    )


def _profile_config(profile_label: str) -> ETAProofProfileConfig:
    if profile_label == "full-internal-rl":
        return ETAProofProfileConfig(
            profile_label=profile_label,
            replacement_mode="causal-binary",
            optimize_after_rollout=True,
            policy_kind="full",
        )
    if profile_label == "full-bootstrap-init":
        return ETAProofProfileConfig(
            profile_label=profile_label,
            replacement_mode="causal-binary",
            optimize_after_rollout=True,
            policy_kind="full",
            bootstrap_init=True,
        )
    if profile_label == "full-no-fast-prior":
        return ETAProofProfileConfig(
            profile_label=profile_label,
            replacement_mode="causal-binary",
            optimize_after_rollout=True,
            policy_kind="full",
            use_temporal_fast_prior=False,
        )
    if profile_label == "full-no-optimize":
        return ETAProofProfileConfig(
            profile_label=profile_label,
            replacement_mode="causal-binary",
            optimize_after_rollout=False,
            policy_kind="full",
        )
    if profile_label == "full-no-replacement":
        return ETAProofProfileConfig(
            profile_label=profile_label,
            replacement_mode="baseline",
            optimize_after_rollout=True,
            policy_kind="full",
        )
    if profile_label == "noop-backend":
        return ETAProofProfileConfig(
            profile_label=profile_label,
            replacement_mode="causal-binary",
            optimize_after_rollout=True,
            policy_kind="full",
            use_noop_backend=True,
        )
    if profile_label == "learned-lite-causal":
        return ETAProofProfileConfig(
            profile_label=profile_label,
            replacement_mode="causal-binary",
            optimize_after_rollout=True,
            policy_kind="learned-lite",
        )
    if profile_label == "metacontroller-no-rl":
        return ETAProofProfileConfig(
            profile_label=profile_label,
            replacement_mode="baseline",
            optimize_after_rollout=False,
            policy_kind="full",
        )
    if profile_label == "learned-lite-baseline":
        return ETAProofProfileConfig(
            profile_label=profile_label,
            replacement_mode="baseline",
            optimize_after_rollout=False,
            policy_kind="learned-lite",
        )
    raise ValueError(f"Unsupported ETA proof profile {profile_label!r}")


def _bootstrap_snapshot_for_cases(
    cases: tuple[ETAProofCase, ...],
) -> object | None:
    if not cases:
        return None
    policy = FullLearnedTemporalPolicy()
    trainer = MetacontrollerSSLTrainer(n_z=policy.parameter_store.n_z)
    policy.parameter_store.set_learning_phase("ssl", structure_frozen=False)
    for case in cases:
        trainer.optimize(
            policy=policy,
            trace=build_training_trace(
                trace_id=f"{case.case_id}:bootstrap",
                source_text=case.source_text,
            ),
        )
    policy.parameter_store.set_learning_phase("runtime", structure_frozen=True)
    return policy.export_rare_heavy_snapshot()


def _build_sandbox(
    *,
    profile: ETAProofProfileConfig,
    backend_label: str,
    bootstrap_snapshot: object | None = None,
) -> InternalRLSandbox:
    if profile.policy_kind == "full":
        if bootstrap_snapshot is not None:
            policy = FullLearnedTemporalPolicy.from_bootstrap_snapshot(bootstrap_snapshot)
        else:
            policy = FullLearnedTemporalPolicy()
    elif profile.policy_kind == "learned-lite":
        policy = LearnedLiteTemporalPolicy()
    else:
        raise ValueError(f"Unsupported policy kind {profile.policy_kind!r}")
    env = (
        InternalRLEnvironment(control_backend=NoOpResidualInterventionBackend())
        if profile.use_noop_backend
        else InternalRLEnvironment()
    )
    if backend_label == "synthetic-open-weight" and not profile.use_noop_backend:
        runtime = SyntheticOpenWeightResidualRuntime(model_id=f"eta-proof:{profile.profile_label}")
    elif backend_label == "trace":
        runtime = None
    else:
        raise ValueError(f"Unsupported backend label {backend_label!r}")
    return InternalRLSandbox(policy=policy, env=env, residual_runtime=runtime)


def _build_case_snapshots(case: ETAProofCase) -> tuple[SubstrateSnapshot, ...]:
    trace = build_training_trace(trace_id=case.case_id, source_text=case.source_text)
    return tuple(_snapshot_from_step(trace.trace_id, step) for step in trace.steps)


def _credit_alignment_score(rollout: ZRollout) -> float:
    sandbox = InternalRLSandbox()
    return sandbox._delayed_credit_alignment((rollout,))


def _episode_report(
    *,
    case: ETAProofCase,
    rollout: ZRollout,
    profile_label: str,
    backend_label: str,
    family_registry: dict[str, set[str]],
) -> ETAProofEpisodeReport:
    completed_pairs = tuple(zip(rollout.completed_subgoals, rollout.completed_family_ids, strict=True))
    reusable_pairs = tuple(
        (subgoal_id, family_id)
        for subgoal_id, family_id in completed_pairs
        if family_id != "unassigned" and family_registry.get(subgoal_id)
    )
    reuse_hits = sum(
        1
        for subgoal_id, family_id in reusable_pairs
        if family_id in family_registry[subgoal_id]
    )
    subgoal_completion_rate = len(rollout.completed_subgoals) / max(len(case.proof_episode.subgoals), 1)
    family_reuse_rate = reuse_hits / max(len(reusable_pairs), 1) if reusable_pairs else 0.0
    switch_sparsity = _mean(tuple(1.0 - transition.controller_state.switch_gate for transition in rollout.transitions))
    mean_persistence = _mean(tuple(float(transition.controller_state.steps_since_switch) for transition in rollout.transitions))
    raw_total_reward = sum(transition.raw_reward for transition in rollout.transitions)
    credit_alignment = _credit_alignment_score(rollout)
    task_success_core = (
        float(rollout.terminal_success) * 0.40
        + subgoal_completion_rate * 0.25
        + family_reuse_rate * 0.20
        + credit_alignment * 0.15
    )
    switch_discipline_score = _switch_discipline_score(
        switch_sparsity=switch_sparsity,
        mean_persistence=mean_persistence,
    )
    backend_quality = min(1.0, _mean(tuple(transition.backend_fidelity for transition in rollout.transitions)) / 0.6)
    mechanism_evidence_score = backend_quality * 0.70 + switch_discipline_score * 0.30
    strong_success_score = (
        task_success_core * 0.90
        + mechanism_evidence_score * 0.10
    )
    if not rollout.terminal_success:
        strong_success_score *= 0.65
    return ETAProofEpisodeReport(
        case_id=case.case_id,
        split=case.split,
        profile_label=profile_label,
        backend_label=backend_label,
        total_reward=rollout.total_reward,
        raw_total_reward=raw_total_reward,
        terminal_success=rollout.terminal_success,
        completed_subgoals=rollout.completed_subgoals,
        completed_family_ids=rollout.completed_family_ids,
        subgoal_completion_rate=subgoal_completion_rate,
        family_reuse_rate=family_reuse_rate,
        switch_sparsity=switch_sparsity,
        mean_persistence=mean_persistence,
        credit_alignment=credit_alignment,
        backend_fidelity=_mean(tuple(transition.backend_fidelity for transition in rollout.transitions)),
        action_family_count=len({family_id for family_id in rollout.completed_family_ids if family_id != "unassigned"}),
        delayed_credit_assignment_count=len(rollout.delayed_credit_assignments),
        task_success_core=task_success_core,
        switch_discipline_score=switch_discipline_score,
        mechanism_evidence_score=mechanism_evidence_score,
        strong_success_score=strong_success_score,
        description=(
            f"{profile_label} on {case.case_id} ({backend_label}) "
            f"success={rollout.terminal_success} completed={len(rollout.completed_subgoals)}/{len(case.proof_episode.subgoals)}."
        ),
    )


def _update_family_registry(registry: dict[str, set[str]], rollout: ZRollout) -> None:
    for subgoal_id, family_id in zip(rollout.completed_subgoals, rollout.completed_family_ids, strict=True):
        if family_id == "unassigned":
            continue
        registry.setdefault(subgoal_id, set()).add(family_id)


def _profile_metric_means(episode_reports: tuple[ETAProofEpisodeReport, ...]) -> tuple[tuple[str, float], ...]:
    eval_reports = tuple(report for report in episode_reports if report.split == "eval")
    heldout_reports = tuple(report for report in episode_reports if report.split == "heldout")
    return (
        ("mean_total_reward", _mean(tuple(report.total_reward for report in episode_reports))),
        ("mean_raw_total_reward", _mean(tuple(report.raw_total_reward for report in episode_reports))),
        ("terminal_success_rate", _mean(tuple(float(report.terminal_success) for report in episode_reports))),
        ("eval_terminal_success_rate", _mean(tuple(float(report.terminal_success) for report in eval_reports))),
        ("heldout_terminal_success_rate", _mean(tuple(float(report.terminal_success) for report in heldout_reports))),
        ("mean_subgoal_completion_rate", _mean(tuple(report.subgoal_completion_rate for report in episode_reports))),
        ("eval_subgoal_completion_rate", _mean(tuple(report.subgoal_completion_rate for report in eval_reports))),
        ("heldout_subgoal_completion_rate", _mean(tuple(report.subgoal_completion_rate for report in heldout_reports))),
        ("mean_family_reuse_rate", _mean(tuple(report.family_reuse_rate for report in episode_reports))),
        ("eval_family_reuse_rate", _mean(tuple(report.family_reuse_rate for report in eval_reports))),
        ("heldout_family_reuse_rate", _mean(tuple(report.family_reuse_rate for report in heldout_reports))),
        ("mean_switch_sparsity", _mean(tuple(report.switch_sparsity for report in episode_reports))),
        ("mean_credit_alignment", _mean(tuple(report.credit_alignment for report in episode_reports))),
        ("eval_credit_alignment", _mean(tuple(report.credit_alignment for report in eval_reports))),
        ("heldout_credit_alignment", _mean(tuple(report.credit_alignment for report in heldout_reports))),
        ("mean_task_success_core", _mean(tuple(report.task_success_core for report in episode_reports))),
        ("eval_task_success_core", _mean(tuple(report.task_success_core for report in eval_reports))),
        ("heldout_task_success_core", _mean(tuple(report.task_success_core for report in heldout_reports))),
        ("mean_switch_discipline_score", _mean(tuple(report.switch_discipline_score for report in episode_reports))),
        ("heldout_switch_discipline_score", _mean(tuple(report.switch_discipline_score for report in heldout_reports))),
        ("mean_mechanism_evidence_score", _mean(tuple(report.mechanism_evidence_score for report in episode_reports))),
        ("heldout_mechanism_evidence_score", _mean(tuple(report.mechanism_evidence_score for report in heldout_reports))),
        ("mean_strong_success_rate", _mean(tuple(report.strong_success_score for report in episode_reports))),
        ("eval_strong_success_rate", _mean(tuple(report.strong_success_score for report in eval_reports))),
        ("heldout_strong_success_rate", _mean(tuple(report.strong_success_score for report in heldout_reports))),
        ("heldout_strong_success_std", _std(tuple(report.strong_success_score for report in heldout_reports))),
        ("mean_backend_fidelity", _mean(tuple(report.backend_fidelity for report in episode_reports))),
    )


def _control_reports_for_gate(
    *,
    report_map: dict[str, ETAProofProfileReport],
    exclude_labels: tuple[str, ...] = ("full-internal-rl", "full-bootstrap-init"),
) -> tuple[ETAProofProfileReport, ...]:
    return tuple(
        report
        for label, report in report_map.items()
        if label not in exclude_labels
    )


def _best_control_metric(
    *,
    control_reports: tuple[ETAProofProfileReport, ...],
    metric_name: str,
) -> tuple[str, float]:
    best_label = "none"
    best_value = 0.0
    for report in control_reports:
        candidate_value = dict(report.metric_means).get(metric_name, 0.0)
        if candidate_value > best_value or best_label == "none":
            best_label = report.profile_label
            best_value = candidate_value
    return best_label, best_value


def _eta_mechanism_strength(report: ETAProofProfileReport) -> float:
    metrics = dict(report.metric_means)
    return round(
        max(0.0, metrics.get("temporal_fast_prior_strength", 0.0)) * 0.025
        + max(0.0, report.mean_replacement_effect_delta) * 1.2,
        4,
    )


def _best_eta_sparse_reward_control(
    *,
    control_reports: tuple[ETAProofProfileReport, ...],
) -> ETAProofProfileReport | None:
    best_report: ETAProofProfileReport | None = None
    best_success = float("-inf")
    best_mechanism_strength = float("-inf")
    for report in control_reports:
        candidate_success = dict(report.metric_means).get("heldout_strong_success_rate", 0.0)
        candidate_mechanism_strength = _eta_mechanism_strength(report)
        if candidate_success > best_success + 1e-9:
            best_report = report
            best_success = candidate_success
            best_mechanism_strength = candidate_mechanism_strength
            continue
        if abs(candidate_success - best_success) <= 1e-9 and candidate_mechanism_strength > best_mechanism_strength:
            best_report = report
            best_success = candidate_success
            best_mechanism_strength = candidate_mechanism_strength
    return best_report


def run_eta_internal_rl_proof_benchmark(
    *,
    cases: tuple[ETAProofCase, ...] = default_eta_proof_cases(),
    profile_labels: tuple[str, ...] = default_eta_proof_profiles(),
    baseline_label: str = "full-internal-rl",
    backend_label: str = "trace",
    train_epochs: int = 2,
) -> ETAProofBenchmarkReport:
    profile_reports: list[ETAProofProfileReport] = []
    benchmark_rollout_batch_count = 0
    for profile_label in profile_labels:
        profile = _profile_config(profile_label)
        train_cases = tuple(case for case in cases if case.split == "train")
        eval_cases = tuple(case for case in cases if case.split != "train")
        bootstrap_snapshot = (
            _bootstrap_snapshot_for_cases(train_cases or eval_cases)
            if profile.bootstrap_init and profile.policy_kind == "full"
            else None
        )
        sandbox = _build_sandbox(
            profile=profile,
            backend_label=backend_label,
            bootstrap_snapshot=bootstrap_snapshot,
        )
        family_registry: dict[str, set[str]] = {}
        training_update_count = 0
        rollout_batch_count = 0
        training_transition_count = 0
        training_parameter_change_count = 0
        parameter_change_norms: list[float] = []
        value_losses: list[float] = []
        replacement_effect_deltas: list[float] = []
        for epoch in range(train_epochs):
            train_rollouts: list[ZRollout] = []
            for case in train_cases:
                snapshots = _build_case_snapshots(case)
                if backend_label == "synthetic-open-weight" and not profile.use_noop_backend:
                    sandbox.configure_runtime_backend(source_text=case.source_text)
                if profile.replacement_mode in {"causal", "causal-binary"} or profile.optimize_after_rollout:
                    sandbox.policy.parameter_store.set_learning_phase("rl", structure_frozen=True)
                else:
                    sandbox.policy.parameter_store.set_learning_phase("runtime")
                train_rollout = sandbox.rollout(
                    rollout_id=f"{profile_label}:{case.case_id}:train:{epoch}",
                    substrate_steps=snapshots,
                    track=Track.SHARED,
                    replacement_mode=profile.replacement_mode,
                    proof_episode=case.proof_episode,
                )
                _update_family_registry(family_registry, train_rollout)
                train_rollouts.append(train_rollout)
            if profile.optimize_after_rollout and train_rollouts:
                optimize_report = sandbox.optimize(tuple(train_rollouts))
                training_update_count += 1
                rollout_batch_count += 1
                training_transition_count += sum(
                    len(train_rollout.transitions) for train_rollout in train_rollouts
                )
                if optimize_report.parameters_changed:
                    training_parameter_change_count += 1
                parameter_change_norms.append(optimize_report.parameter_change_norm)
                value_losses.append(optimize_report.value_loss)
                replacement_effect_deltas.append(optimize_report.replacement_effect_delta)
            sandbox.ingest_temporal_fast_prior(
                tuple(train_rollouts),
                enabled=profile.use_temporal_fast_prior,
            )
        episode_reports: list[ETAProofEpisodeReport] = []
        for case in train_cases + eval_cases:
            snapshots = _build_case_snapshots(case)
            if backend_label == "synthetic-open-weight" and not profile.use_noop_backend:
                sandbox.configure_runtime_backend(source_text=case.source_text)
            if profile.replacement_mode in {"causal", "causal-binary"}:
                sandbox.policy.parameter_store.set_learning_phase("rl", structure_frozen=True)
            else:
                sandbox.policy.parameter_store.set_learning_phase("runtime")
            rollout = sandbox.rollout(
                rollout_id=f"{profile_label}:{case.case_id}:eval",
                substrate_steps=snapshots,
                track=Track.SHARED,
                replacement_mode=profile.replacement_mode,
                proof_episode=case.proof_episode,
            )
            episode_reports.append(
                _episode_report(
                    case=case,
                    rollout=rollout,
                    profile_label=profile_label,
                    backend_label=backend_label,
                    family_registry=family_registry,
                )
            )
            _update_family_registry(family_registry, rollout)
        metric_means = _profile_metric_means(tuple(episode_reports))
        metric_means = metric_means + (
            ("temporal_fast_prior_strength", sandbox.policy.parameter_store.latest_fast_prior_strength),
            ("temporal_fast_prior_switch_delta", sandbox.policy.parameter_store.latest_fast_prior_switch_pressure_delta),
            ("temporal_fast_prior_action_bias", sandbox.policy.parameter_store.latest_fast_prior_action_bias),
            ("temporal_fast_prior_family_bias", sandbox.policy.parameter_store.latest_fast_prior_family_bias),
            ("bootstrap_init_used", 1.0 if bootstrap_snapshot is not None else 0.0),
        )
        benchmark_rollout_batch_count += rollout_batch_count
        profile_reports.append(
            ETAProofProfileReport(
                profile_label=profile_label,
                backend_label=backend_label,
                episode_reports=tuple(episode_reports),
                metric_means=metric_means,
                training_update_count=training_update_count,
                rollout_batch_count=rollout_batch_count,
                mean_rollouts_per_update=(
                    len(train_cases)
                    if training_update_count > 0
                    else 0.0
                ),
                training_transition_count=training_transition_count,
                training_parameter_change_count=training_parameter_change_count,
                training_parameter_change_rate=(
                    training_parameter_change_count / training_update_count
                    if training_update_count > 0
                    else 0.0
                ),
                mean_parameter_change_norm=_mean(tuple(parameter_change_norms)),
                mean_value_loss=_mean(tuple(value_losses)),
                mean_replacement_effect_delta=_mean(tuple(replacement_effect_deltas)),
                description=(
                    f"{profile_label} produced {len(episode_reports)} ETA proof episode reports "
                    f"on backend={backend_label} with batch_updates={rollout_batch_count}."
                ),
            )
        )
    baseline_report = next(report for report in profile_reports if report.profile_label == baseline_label)
    baseline_metrics = dict(baseline_report.metric_means)
    no_fast_prior_report = next(
        (report for report in profile_reports if report.profile_label == "full-no-fast-prior"),
        None,
    )
    deltas: list[tuple[str, tuple[tuple[str, float], ...]]] = []
    for report in profile_reports:
        if report.profile_label == baseline_label:
            continue
        deltas.append(
            (
                report.profile_label,
                tuple(
                    (metric_name, value - baseline_metrics.get(metric_name, 0.0))
                    for metric_name, value in report.metric_means
                ),
            )
        )
    if no_fast_prior_report is not None:
        no_fast_prior_metrics = dict(no_fast_prior_report.metric_means)
        baseline_metric_means = baseline_report.metric_means + (
            (
                "heldout_family_reuse_gap_vs_no_fast_prior",
                baseline_metrics.get("heldout_family_reuse_rate", 0.0)
                - no_fast_prior_metrics.get("heldout_family_reuse_rate", 0.0),
            ),
            (
                "heldout_credit_alignment_gap_vs_no_fast_prior",
                baseline_metrics.get("heldout_credit_alignment", 0.0)
                - no_fast_prior_metrics.get("heldout_credit_alignment", 0.0),
            ),
            (
                "heldout_strong_success_gap_vs_no_fast_prior",
                baseline_metrics.get("heldout_strong_success_rate", 0.0)
                - no_fast_prior_metrics.get("heldout_strong_success_rate", 0.0),
            ),
        )
        profile_reports = [
            ETAProofProfileReport(
                profile_label=report.profile_label,
                backend_label=report.backend_label,
                episode_reports=report.episode_reports,
                metric_means=baseline_metric_means if report.profile_label == baseline_label else report.metric_means,
                training_update_count=report.training_update_count,
                rollout_batch_count=report.rollout_batch_count,
                mean_rollouts_per_update=report.mean_rollouts_per_update,
                training_transition_count=report.training_transition_count,
                training_parameter_change_count=report.training_parameter_change_count,
                training_parameter_change_rate=report.training_parameter_change_rate,
                mean_parameter_change_norm=report.mean_parameter_change_norm,
                mean_value_loss=report.mean_value_loss,
                mean_replacement_effect_delta=report.mean_replacement_effect_delta,
                description=report.description,
            )
            for report in profile_reports
        ]
    return ETAProofBenchmarkReport(
        profile_reports=tuple(profile_reports),
        baseline_label=baseline_label,
        backend_label=backend_label,
        metric_deltas_from_baseline=tuple(deltas),
        rollout_batch_count=benchmark_rollout_batch_count,
        description=(
            f"ETA internal-RL proof benchmark ran {len(profile_reports)} profiles on backend={backend_label} "
            f"across {len(cases)} cases."
        ),
    )


def run_eta_internal_rl_backend_robustness_benchmark(
    *,
    cases: tuple[ETAProofCase, ...] = default_eta_proof_cases(),
    profile_label: str = "full-internal-rl",
) -> ETAProofBackendComparisonReport:
    trace_report = run_eta_internal_rl_proof_benchmark(
        cases=cases,
        profile_labels=(profile_label,),
        baseline_label=profile_label,
        backend_label="trace",
    ).profile_reports[0]
    synthetic_report = run_eta_internal_rl_proof_benchmark(
        cases=cases,
        profile_labels=(profile_label,),
        baseline_label=profile_label,
        backend_label="synthetic-open-weight",
    ).profile_reports[0]
    trace_metrics = dict(trace_report.metric_means)
    synthetic_metrics = dict(synthetic_report.metric_means)
    return ETAProofBackendComparisonReport(
        profile_label=profile_label,
        profile_reports=(trace_report, synthetic_report),
        metric_deltas=tuple(
            (metric_name, synthetic_metrics.get(metric_name, 0.0) - trace_metrics.get(metric_name, 0.0))
            for metric_name, _ in trace_report.metric_means
        ),
        description=(
            f"Backend robustness comparison for {profile_label} across trace and synthetic-open-weight backends."
        ),
    )


def build_eta_internal_rl_assessment(
    *,
    benchmark_report: ETAProofBenchmarkReport,
    backend_report: ETAProofBackendComparisonReport | None = None,
) -> ETAInternalRLAssessmentReport:
    report_map = {report.profile_label: report for report in benchmark_report.profile_reports}
    full_report = report_map["full-internal-rl"]
    gate_control_reports = _control_reports_for_gate(report_map=report_map)
    full_metrics = dict(full_report.metric_means)
    best_sparse_reward_control = _best_eta_sparse_reward_control(control_reports=gate_control_reports)
    best_terminal_success_label, best_control_terminal_success = _best_control_metric(
        control_reports=gate_control_reports,
        metric_name="heldout_terminal_success_rate",
    )
    best_strong_success_label = best_sparse_reward_control.profile_label if best_sparse_reward_control is not None else "none"
    best_control_strong_success = (
        dict(best_sparse_reward_control.metric_means).get("heldout_strong_success_rate", 0.0)
        if best_sparse_reward_control is not None
        else 0.0
    )
    best_control_mechanism_strength = (
        _eta_mechanism_strength(best_sparse_reward_control)
        if best_sparse_reward_control is not None
        else 0.0
    )
    full_mechanism_strength = _eta_mechanism_strength(full_report)
    raw_success_delta = full_metrics.get("heldout_strong_success_rate", 0.0) - best_control_strong_success
    mechanism_tie_break_margin = (
        max(0.0, full_mechanism_strength - best_control_mechanism_strength)
        if abs(raw_success_delta) <= 1e-9
        else 0.0
    )
    effective_success_margin = max(raw_success_delta, mechanism_tie_break_margin)
    best_subgoal_completion_label, best_control_subgoal_completion = _best_control_metric(
        control_reports=gate_control_reports,
        metric_name="heldout_subgoal_completion_rate",
    )
    best_reuse_label, best_control_reuse = _best_control_metric(
        control_reports=gate_control_reports,
        metric_name="heldout_family_reuse_rate",
    )
    best_credit_alignment_label, best_control_credit_alignment = _best_control_metric(
        control_reports=gate_control_reports,
        metric_name="heldout_credit_alignment",
    )
    backend_success_gap = 1.0
    backend_min_success = 0.0
    if backend_report is not None:
        backend_success = tuple(
            dict(report.metric_means).get("heldout_strong_success_rate", 0.0)
            for report in backend_report.profile_reports
        )
        backend_success_gap = max(backend_success) - min(backend_success) if backend_success else 1.0
        backend_min_success = min(backend_success) if backend_success else 0.0
    gates = (
        ETAInternalRLAcceptanceGate(
            gate_id="sparse-reward-success",
            passed=(
                full_metrics.get("heldout_terminal_success_rate", 0.0) >= best_control_terminal_success
                and full_metrics.get("heldout_strong_success_rate", 0.0) >= 0.30
                and effective_success_margin >= ETAInternalRLAcceptanceConfig.min_success_delta
            ),
            evidence=(
                ("heldout_terminal_success_rate", full_metrics.get("heldout_terminal_success_rate", 0.0)),
                ("best_control_terminal_success_label", best_terminal_success_label),
                ("best_control_terminal_success", best_control_terminal_success),
                ("heldout_strong_success_rate", full_metrics.get("heldout_strong_success_rate", 0.0)),
                ("best_control_strong_success_label", best_strong_success_label),
                ("best_control_strong_success", best_control_strong_success),
                ("raw_success_delta", raw_success_delta),
                ("full_mechanism_strength", full_mechanism_strength),
                ("best_control_mechanism_strength", best_control_mechanism_strength),
                ("mechanism_tie_break_margin", mechanism_tie_break_margin),
                ("effective_success_margin", effective_success_margin),
            ),
            description=(
                "Full internal RL should beat matched controls on held-out terminal success and strong sparse-reward success, "
                "or win tied sparse-reward outcomes with stronger mechanism evidence."
            ),
        ),
        ETAInternalRLAcceptanceGate(
            gate_id="abstract-action-reuse",
            passed=(
                full_metrics.get("heldout_family_reuse_rate", 0.0) >= 0.50
                and full_metrics.get("heldout_credit_alignment", 0.0) >= 0.50
                and full_metrics.get("heldout_family_reuse_rate", 0.0) >= best_control_reuse
                and full_metrics.get("heldout_credit_alignment", 0.0) >= best_control_credit_alignment
            ),
            evidence=(
                ("heldout_family_reuse_rate", full_metrics.get("heldout_family_reuse_rate", 0.0)),
                ("best_control_reuse_label", best_reuse_label),
                ("best_control_reuse", best_control_reuse),
                ("heldout_credit_alignment", full_metrics.get("heldout_credit_alignment", 0.0)),
                ("best_control_credit_alignment_label", best_credit_alignment_label),
                ("best_control_credit_alignment", best_control_credit_alignment),
            ),
            description="Discovered abstract-action families should be reused across cases, not recreated every time.",
        ),
        ETAInternalRLAcceptanceGate(
            gate_id="heldout-composition",
            passed=(
                full_metrics.get("heldout_strong_success_rate", 0.0) >= 0.30
                and full_metrics.get("heldout_subgoal_completion_rate", 0.0) >= 0.25
                and full_metrics.get("heldout_strong_success_rate", 0.0) >= best_control_strong_success
                and full_metrics.get("heldout_subgoal_completion_rate", 0.0) >= best_control_subgoal_completion
            ),
            evidence=(
                ("heldout_strong_success_rate", full_metrics.get("heldout_strong_success_rate", 0.0)),
                ("best_control_strong_success_label", best_strong_success_label),
                ("best_control_strong_success", best_control_strong_success),
                ("heldout_subgoal_completion_rate", full_metrics.get("heldout_subgoal_completion_rate", 0.0)),
                ("best_control_subgoal_completion_label", best_subgoal_completion_label),
                ("best_control_subgoal_completion", best_control_subgoal_completion),
                ("mean_switch_sparsity", full_metrics.get("mean_switch_sparsity", 0.0)),
            ),
            description="Held-out subgoal recombinations should remain solvable with non-trivial completion.",
        ),
        ETAInternalRLAcceptanceGate(
            gate_id="credit-alignment",
            passed=(
                full_metrics.get("heldout_credit_alignment", 0.0) >= 0.50
                and full_metrics.get("heldout_credit_alignment", 0.0) >= best_control_credit_alignment
            ),
            evidence=(
                ("heldout_credit_alignment", full_metrics.get("heldout_credit_alignment", 0.0)),
                ("best_control_credit_alignment_label", best_credit_alignment_label),
                ("best_control_credit_alignment", best_control_credit_alignment),
                ("heldout_strong_success_rate", full_metrics.get("heldout_strong_success_rate", 0.0)),
                ("mean_total_reward", full_metrics.get("mean_total_reward", 0.0)),
                ("mean_raw_total_reward", full_metrics.get("mean_raw_total_reward", 0.0)),
            ),
            description="Delayed sparse reward should align with the abstract-action windows that preceded success.",
        ),
        ETAInternalRLAcceptanceGate(
            gate_id="policy-update-evidence",
            passed=(
                full_report.training_update_count > 0
                and full_report.training_parameter_change_rate >= 0.50
                and full_report.mean_parameter_change_norm > 0.01
            ),
            evidence=(
                ("training_update_count", float(full_report.training_update_count)),
                ("rollout_batch_count", float(full_report.rollout_batch_count)),
                ("mean_rollouts_per_update", full_report.mean_rollouts_per_update),
                ("training_transition_count", float(full_report.training_transition_count)),
                ("training_parameter_change_count", float(full_report.training_parameter_change_count)),
                ("training_parameter_change_rate", full_report.training_parameter_change_rate),
                ("mean_parameter_change_norm", full_report.mean_parameter_change_norm),
                ("mean_value_loss", full_report.mean_value_loss),
            ),
            description="Internal RL should show concrete parameter-update evidence during training, not only better outcomes.",
        ),
        ETAInternalRLAcceptanceGate(
            gate_id="statistical-batch-evidence",
            passed=(
                full_report.mean_rollouts_per_update >= 2.0
                and full_report.mean_replacement_effect_delta >= 0.05
                and full_metrics.get("heldout_strong_success_std", 0.0) <= 0.35
            ),
            evidence=(
                ("mean_rollouts_per_update", full_report.mean_rollouts_per_update),
                ("mean_replacement_effect_delta", full_report.mean_replacement_effect_delta),
                ("heldout_strong_success_std", full_metrics.get("heldout_strong_success_std", 0.0)),
                (
                    "heldout_control_gap",
                    full_metrics.get("heldout_strong_success_rate", 0.0) - best_control_strong_success,
                ),
            ),
            description=(
                "Stronger ETA claims require batch-level rollout evidence, non-trivial replacement effect, "
                "and bounded held-out variance."
            ),
        ),
        ETAInternalRLAcceptanceGate(
            gate_id="backend-robustness",
            passed=backend_report is not None and backend_min_success >= 0.30,
            evidence=(
                ("backend_min_success", backend_min_success),
                ("backend_success_gap", backend_success_gap),
                ("backend_report", "present" if backend_report is not None else "missing"),
            ),
            description="The proof should persist across trace and synthetic-open-weight backends.",
        ),
    )
    passed_gate_count = sum(1 for gate in gates if gate.passed)
    return ETAInternalRLAssessmentReport(
        benchmark_report=benchmark_report,
        backend_report=backend_report,
        gates=gates,
        passed_gate_count=passed_gate_count,
        total_gate_count=len(gates),
        description=f"ETA internal-RL assessment passed {passed_gate_count}/{len(gates)} gates.",
    )


def evaluate_eta_internal_rl_acceptance(
    assessment: ETAInternalRLAssessmentReport,
    *,
    config: ETAInternalRLAcceptanceConfig | None = None,
) -> ETAInternalRLAcceptanceDecision:
    active_config = config or ETAInternalRLAcceptanceConfig()
    gate_map = {gate.gate_id: gate for gate in assessment.gates}
    blocked_gate_ids: list[str] = []
    reasons: list[str] = []
    for gate_id in active_config.required_gate_ids:
        gate = gate_map.get(gate_id)
        if gate is None:
            blocked_gate_ids.append(gate_id)
            reasons.append(f"missing-gate:{gate_id}")
            continue
        if not gate.passed:
            blocked_gate_ids.append(gate_id)
            reasons.append(f"failed-gate:{gate_id}")
    full_metrics = dict(
        next(
            report.metric_means
            for report in assessment.benchmark_report.profile_reports
            if report.profile_label == "full-internal-rl"
        )
    )
    full_profile_report = next(
        report
        for report in assessment.benchmark_report.profile_reports
        if report.profile_label == "full-internal-rl"
    )
    control_reports = _control_reports_for_gate(
        report_map={
            report.profile_label: report
            for report in assessment.benchmark_report.profile_reports
        }
    )
    _, best_control_success = _best_control_metric(
        control_reports=control_reports,
        metric_name="heldout_strong_success_rate",
    )
    sparse_reward_evidence = dict(gate_map["sparse-reward-success"].evidence) if "sparse-reward-success" in gate_map else {}
    effective_success_margin = sparse_reward_evidence.get(
        "effective_success_margin",
        full_metrics.get("heldout_strong_success_rate", 0.0) - best_control_success,
    )
    if full_metrics.get("heldout_terminal_success_rate", 0.0) < active_config.min_terminal_success_rate:
        reasons.append("heldout-success-below-threshold")
    if effective_success_margin < active_config.min_success_delta:
        reasons.append("success-delta-below-threshold")
    if full_metrics.get("heldout_strong_success_rate", 0.0) < active_config.min_strong_success_rate:
        reasons.append("heldout-strong-success-below-threshold")
    if full_metrics.get("heldout_family_reuse_rate", 0.0) < active_config.min_reuse_rate:
        reasons.append("family-reuse-below-threshold")
    if full_metrics.get("heldout_credit_alignment", 0.0) < active_config.min_credit_alignment:
        reasons.append("credit-alignment-below-threshold")
    if full_profile_report.training_parameter_change_rate < active_config.min_policy_update_rate:
        reasons.append("policy-update-rate-below-threshold")
    if full_profile_report.mean_rollouts_per_update < active_config.min_rollouts_per_update:
        reasons.append("rollouts-per-update-below-threshold")
    if full_profile_report.mean_parameter_change_norm < active_config.min_parameter_change_norm:
        reasons.append("parameter-change-norm-below-threshold")
    if full_profile_report.mean_replacement_effect_delta < active_config.min_replacement_effect_delta:
        reasons.append("replacement-effect-delta-below-threshold")
    if full_metrics.get("heldout_strong_success_std", 0.0) > active_config.max_heldout_success_std:
        reasons.append("heldout-success-std-above-threshold")
    if assessment.backend_report is None:
        reasons.append("missing-backend-report")
    else:
        backend_metrics = dict(assessment.backend_report.metric_deltas)
        success_gap = abs(backend_metrics.get("heldout_strong_success_rate", 0.0))
        if success_gap > active_config.max_backend_success_gap:
            reasons.append("backend-gap-above-threshold")
    if assessment.passed_gate_count < active_config.min_passed_gate_count:
        reasons.append("passed-gate-count-below-threshold")
    accepted_gate_ids = tuple(gate.gate_id for gate in assessment.gates if gate.passed)
    return ETAInternalRLAcceptanceDecision(
        accepted=not reasons,
        reasons=tuple(reasons),
        accepted_gate_ids=accepted_gate_ids,
        blocked_gate_ids=tuple(sorted(set(blocked_gate_ids))),
        description=(
            f"ETA internal-RL acceptance accepted={not reasons} "
            f"passed={assessment.passed_gate_count}/{assessment.total_gate_count}."
        ),
    )


def _eta_metric_value_map(report: ETAProofProfileReport) -> dict[str, float]:
    return dict(report.metric_means)


def _eta_paper_suite_metric_values(
    *,
    benchmark_report: ETAProofBenchmarkReport,
    backend_report: ETAProofBackendComparisonReport,
    assessment: ETAInternalRLAssessmentReport,
) -> tuple[tuple[str, float], ...]:
    report_map = {
        profile_report.profile_label: profile_report
        for profile_report in benchmark_report.profile_reports
    }
    full_report = report_map["full-internal-rl"]
    full_metrics = _eta_metric_value_map(full_report)
    control_reports = _control_reports_for_gate(report_map=report_map)
    best_sparse_reward_control = _best_eta_sparse_reward_control(control_reports=control_reports)
    best_control_strong_success = (
        _eta_metric_value_map(best_sparse_reward_control).get("heldout_strong_success_rate", 0.0)
        if best_sparse_reward_control is not None
        else 0.0
    )
    best_control_mechanism_strength = (
        _eta_mechanism_strength(best_sparse_reward_control)
        if best_sparse_reward_control is not None
        else 0.0
    )
    raw_strong_success_gap = full_metrics.get("heldout_strong_success_rate", 0.0) - best_control_strong_success
    mechanism_tie_break_gap = (
        max(0.0, _eta_mechanism_strength(full_report) - best_control_mechanism_strength)
        if abs(raw_strong_success_gap) <= 1e-9
        else 0.0
    )
    effective_strong_success_gap = max(raw_strong_success_gap, mechanism_tie_break_gap)
    backend_strong_success_values = tuple(
        dict(profile_report.metric_means).get("heldout_strong_success_rate", 0.0)
        for profile_report in backend_report.profile_reports
    )
    return (
        ("heldout_terminal_success_rate", full_metrics.get("heldout_terminal_success_rate", 0.0)),
        ("heldout_strong_success_rate", full_metrics.get("heldout_strong_success_rate", 0.0)),
        ("heldout_family_reuse_rate", full_metrics.get("heldout_family_reuse_rate", 0.0)),
        ("heldout_credit_alignment", full_metrics.get("heldout_credit_alignment", 0.0)),
        ("heldout_strong_success_std", full_metrics.get("heldout_strong_success_std", 0.0)),
        (
            "strong_success_gap_vs_best_control",
            effective_strong_success_gap,
        ),
        (
            "backend_success_gap",
            max(backend_strong_success_values) - min(backend_strong_success_values)
            if backend_strong_success_values
            else 0.0,
        ),
        (
            "assessment_pass_fraction",
            assessment.passed_gate_count / assessment.total_gate_count
            if assessment.total_gate_count > 0
            else 0.0,
        ),
        ("policy_update_rate", full_report.training_parameter_change_rate),
        ("mean_rollouts_per_update", full_report.mean_rollouts_per_update),
        ("mean_parameter_change_norm", full_report.mean_parameter_change_norm),
        ("mean_replacement_effect_delta", full_report.mean_replacement_effect_delta),
        ("mean_value_loss", full_report.mean_value_loss),
        ("training_transition_count", float(full_report.training_transition_count)),
        ("heldout_subgoal_completion_rate", full_metrics.get("heldout_subgoal_completion_rate", 0.0)),
    )


def _eta_metric_samples(
    *,
    run_summaries: tuple[ETAProofPaperSuiteRunSummary, ...],
    metric_names: tuple[str, ...],
) -> dict[str, tuple[float, ...]]:
    summary_maps = [dict(summary.metric_values) for summary in run_summaries]
    return {
        metric_name: tuple(summary_map.get(metric_name, 0.0) for summary_map in summary_maps)
        for metric_name in metric_names
    }


def _eta_failure_mode_from_assessment(assessment: ETAInternalRLAssessmentReport | None) -> str:
    if assessment is None:
        return "no-assessment"
    blocked_gate_ids = tuple(
        gate.gate_id
        for gate in assessment.gates
        if not gate.passed
    )
    if "statistical-batch-evidence" in blocked_gate_ids:
        return "weak-statistical-evidence"
    if "policy-update-evidence" in blocked_gate_ids:
        return "weak-policy-update"
    if "credit-alignment" in blocked_gate_ids:
        return "credit-misalignment"
    if "heldout-composition" in blocked_gate_ids:
        return "heldout-composition-fragile"
    if "abstract-action-reuse" in blocked_gate_ids:
        return "family-reuse-fragile"
    if "backend-robustness" in blocked_gate_ids:
        return "backend-fragility"
    if "sparse-reward-success" in blocked_gate_ids:
        return "sparse-reward-weakness"
    if not blocked_gate_ids:
        return "no-dominant-failure-mode"
    return blocked_gate_ids[0]


def _build_eta_paper_suite_interpretation_summary(
    *,
    reference_assessment: ETAInternalRLAssessmentReport | None,
    reference_benchmark_report: ETAProofBenchmarkReport | None,
    run_summaries: tuple[ETAProofPaperSuiteRunSummary, ...],
    primary_metric_summaries: tuple[MetricIntervalSummary, ...],
    secondary_metric_summaries: tuple[MetricIntervalSummary, ...],
) -> ETAProofPaperSuiteInterpretationSummary | None:
    if reference_assessment is None:
        return None
    all_summaries = primary_metric_summaries + secondary_metric_summaries
    if not all_summaries:
        return ETAProofPaperSuiteInterpretationSummary(
            interpretation="eta-proof-summary-unavailable",
            review_summary="ETA proof paper suite did not produce summary metrics.",
            dominant_failure_mode=_eta_failure_mode_from_assessment(reference_assessment),
            strongest_metric="none",
            weakest_metric="none",
            strongest_competing_control="none",
            strongest_control_gap=0.0,
            cross_run_gap_mean=0.0,
            cross_run_gap_std=0.0,
            confidence=0.0,
            description="ETA proof paper suite did not produce any metric summaries.",
        )
    strengths = {summary.metric_name: summary.mean for summary in all_summaries}
    strongest_metric = max(strengths.items(), key=lambda item: item[1])[0]
    weakest_metric = min(strengths.items(), key=lambda item: item[1])[0]
    dominant_failure_mode = _eta_failure_mode_from_assessment(reference_assessment)
    strongest_competing_control = "none"
    strongest_control_gap = 0.0
    cross_run_gap_mean = 0.0
    cross_run_gap_std = 0.0
    if reference_benchmark_report is not None:
        report_map = {
            profile_report.profile_label: profile_report
            for profile_report in reference_benchmark_report.profile_reports
        }
        control_reports = _control_reports_for_gate(report_map=report_map)
        strongest_control_report = _best_eta_sparse_reward_control(control_reports=control_reports)
        strongest_competing_control = strongest_control_report.profile_label if strongest_control_report is not None else "none"
        best_control_strong_success = (
            dict(strongest_control_report.metric_means).get("heldout_strong_success_rate", 0.0)
            if strongest_control_report is not None
            else 0.0
        )
        full_report = report_map.get("full-internal-rl")
        full_metrics = dict(full_report.metric_means) if full_report is not None else {}
        raw_gap = full_metrics.get("heldout_strong_success_rate", 0.0) - best_control_strong_success
        mechanism_gap = (
            max(0.0, _eta_mechanism_strength(full_report) - _eta_mechanism_strength(strongest_control_report))
            if full_report is not None and strongest_control_report is not None and abs(raw_gap) <= 1e-9
            else 0.0
        )
        strongest_control_gap = round(
            max(raw_gap, mechanism_gap),
            4,
        )
    if run_summaries:
        control_counts: dict[str, int] = {}
        control_gaps: list[float] = []
        for summary in run_summaries:
            control_counts[summary.strongest_competing_control] = (
                control_counts.get(summary.strongest_competing_control, 0) + 1
            )
            control_gaps.append(summary.strongest_control_gap)
        strongest_competing_control = max(
            control_counts.items(),
            key=lambda item: item[1],
        )[0]
        cross_run_gap_mean = round(sum(control_gaps) / len(control_gaps), 4)
        cross_run_gap_std = round(_std(tuple(control_gaps)), 4)
    pass_fraction = (
        reference_assessment.passed_gate_count / reference_assessment.total_gate_count
        if reference_assessment.total_gate_count > 0
        else 0.0
    )
    confidence = round(max(0.0, min(1.0, pass_fraction * 0.75 + 0.25)), 4)
    if dominant_failure_mode == "no-dominant-failure-mode":
        interpretation = (
            f"eta-proof-strongest={strongest_metric}; no dominant failure mode; "
            f"weakest_metric={weakest_metric}"
        )
    else:
        interpretation = (
            f"eta-proof-strongest={strongest_metric}; failure_mode={dominant_failure_mode}; "
            f"weakest_metric={weakest_metric}"
        )
    if dominant_failure_mode == "no-dominant-failure-mode":
        review_summary = (
            f"Full internal RL remains ahead of {strongest_competing_control} by "
            f"{cross_run_gap_mean:.3f} +/- {cross_run_gap_std:.3f} on held-out strong success across runs; "
            f"no dominant failure mode surfaced."
        )
    else:
        review_summary = (
            f"Primary weakness is {dominant_failure_mode}; strongest competing control is "
            f"{strongest_competing_control}; cross-run held-out strong-success gap is "
            f"{cross_run_gap_mean:.3f} +/- {cross_run_gap_std:.3f}."
        )
    return ETAProofPaperSuiteInterpretationSummary(
        interpretation=interpretation,
        review_summary=review_summary,
        dominant_failure_mode=dominant_failure_mode,
        strongest_metric=strongest_metric,
        weakest_metric=weakest_metric,
        strongest_competing_control=strongest_competing_control,
        strongest_control_gap=strongest_control_gap,
        cross_run_gap_mean=cross_run_gap_mean,
        cross_run_gap_std=cross_run_gap_std,
        confidence=confidence,
        description=(
            f"ETA paper-suite interpretation strongest={strongest_metric} weakest={weakest_metric} "
            f"failure_mode={dominant_failure_mode} competitor={strongest_competing_control} "
            f"gap={strongest_control_gap:.3f} cross_run_gap={cross_run_gap_mean:.3f}+/-{cross_run_gap_std:.3f} "
            f"confidence={confidence:.2f}."
        ),
    )


def _repo_root_from_eta_module() -> Path:
    return Path(__file__).resolve().parents[2]


def run_eta_internal_rl_paper_suite(
    *,
    manifest: PaperSuiteManifest | None = None,
    output_dir: str | Path | None = None,
) -> ETAProofPaperSuiteAggregateReport:
    active_manifest = manifest or build_eta_proof_paper_suite_manifest()
    route_ids = {
        route_id
        for name, values in active_manifest.case_groups
        if name == "route_ids"
        for route_id in values
    }
    train_epochs = int(
        next(values[0] for name, values in active_manifest.case_groups if name == "train_epochs")
    )
    cases = tuple(case for case in default_eta_proof_cases() if case.case_id in route_ids)
    run_summaries: list[ETAProofPaperSuiteRunSummary] = []
    reference_benchmark_report: ETAProofBenchmarkReport | None = None
    reference_backend_report: ETAProofBackendComparisonReport | None = None
    reference_assessment: ETAInternalRLAssessmentReport | None = None
    for run_index, run_seed in enumerate(active_manifest.seed_schedule[: active_manifest.repeat_count], start=1):
        run_random = Random(run_seed)
        ordered_cases = tuple(
            sorted(
                cases,
                key=lambda case: (run_random.random(), case.case_id),
            )
        )
        benchmark_report = run_eta_internal_rl_proof_benchmark(
            cases=ordered_cases,
            profile_labels=tuple(profile.profile_label for profile in active_manifest.profiles),
            baseline_label=active_manifest.baseline_label,
            backend_label="trace",
            train_epochs=train_epochs,
        )
        backend_report = run_eta_internal_rl_backend_robustness_benchmark(
            cases=ordered_cases,
            profile_label=active_manifest.baseline_label,
        )
        assessment = build_eta_internal_rl_assessment(
            benchmark_report=benchmark_report,
            backend_report=backend_report,
        )
        reference_benchmark_report = reference_benchmark_report or benchmark_report
        reference_backend_report = reference_backend_report or backend_report
        reference_assessment = reference_assessment or assessment
        report_map = {
            profile_report.profile_label: profile_report
            for profile_report in benchmark_report.profile_reports
        }
        control_reports = _control_reports_for_gate(report_map=report_map)
        strongest_control_report = _best_eta_sparse_reward_control(control_reports=control_reports)
        strongest_competing_control = strongest_control_report.profile_label if strongest_control_report is not None else "none"
        full_report = report_map.get("full-internal-rl")
        full_metrics = dict(full_report.metric_means) if full_report is not None else {}
        best_control_strong_success = (
            dict(strongest_control_report.metric_means).get("heldout_strong_success_rate", 0.0)
            if strongest_control_report is not None
            else 0.0
        )
        raw_gap = full_metrics.get("heldout_strong_success_rate", 0.0) - best_control_strong_success
        mechanism_gap = (
            max(0.0, _eta_mechanism_strength(full_report) - _eta_mechanism_strength(strongest_control_report))
            if full_report is not None and strongest_control_report is not None and abs(raw_gap) <= 1e-9
            else 0.0
        )
        run_summary = ETAProofPaperSuiteRunSummary(
            run_id=f"{active_manifest.suite_id}:run-{run_index:02d}",
            run_seed=run_seed,
            metric_values=_eta_paper_suite_metric_values(
                benchmark_report=benchmark_report,
                backend_report=backend_report,
                assessment=assessment,
            ),
            strongest_competing_control=strongest_competing_control,
            strongest_control_gap=round(
                max(raw_gap, mechanism_gap),
                4,
            ),
            description=(
                f"ETA proof paper suite run {run_index} summarized "
                f"{len(benchmark_report.profile_reports)} profiles for seed={run_seed}."
            ),
        )
        run_summaries.append(run_summary)
        if output_dir is not None:
            target_dir = Path(output_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            export_json_artifact(
                payload=benchmark_report,
                output_path=target_dir / f"eta_run_{run_index:02d}_benchmark.json",
            )
            export_json_artifact(
                payload=backend_report,
                output_path=target_dir / f"eta_run_{run_index:02d}_backend.json",
            )
            export_json_artifact(
                payload=assessment,
                output_path=target_dir / f"eta_run_{run_index:02d}_assessment.json",
            )
            export_json_artifact(
                payload=run_summary,
                output_path=target_dir / f"eta_run_{run_index:02d}_summary.json",
            )
    primary_metric_summaries = build_metric_interval_summaries(
        metric_samples=_eta_metric_samples(
            run_summaries=tuple(run_summaries),
            metric_names=tuple(metric.metric_name for metric in active_manifest.primary_metrics),
        )
    )
    secondary_metric_summaries = build_metric_interval_summaries(
        metric_samples=_eta_metric_samples(
            run_summaries=tuple(run_summaries),
            metric_names=tuple(metric.metric_name for metric in active_manifest.secondary_metrics),
        )
    )
    provenance = collect_paper_suite_provenance(
        manifest=active_manifest,
        repo_root=_repo_root_from_eta_module(),
        runtime_descriptor={
            "suite_kind": active_manifest.suite_kind,
            "train_epochs": str(train_epochs),
            "backend_mode": "trace+synthetic-open-weight",
        },
    )
    interpretation_summary = _build_eta_paper_suite_interpretation_summary(
        reference_assessment=reference_assessment,
        reference_benchmark_report=reference_benchmark_report,
        run_summaries=tuple(run_summaries),
        primary_metric_summaries=primary_metric_summaries,
        secondary_metric_summaries=secondary_metric_summaries,
    )
    pairwise_effects = _build_eta_paper_suite_pairwise_effects(tuple(run_summaries))
    aggregate_report = ETAProofPaperSuiteAggregateReport(
        manifest=active_manifest,
        provenance=provenance,
        run_summaries=tuple(run_summaries),
        reference_benchmark_report=reference_benchmark_report,
        reference_backend_report=reference_backend_report,
        reference_assessment=reference_assessment,
        primary_metric_summaries=primary_metric_summaries,
        secondary_metric_summaries=secondary_metric_summaries,
        interpretation_summary=interpretation_summary,
        description=(
            f"ETA proof paper suite {active_manifest.suite_id} aggregated "
            f"{len(run_summaries)} repeated runs."
        ),
        pairwise_effects=pairwise_effects,
        claim_verdicts=(),
    )
    aggregate_report = replace(
        aggregate_report,
        claim_verdicts=_build_eta_claim_verdicts(aggregate_report),
    )
    if output_dir is not None:
        export_eta_internal_rl_paper_suite_artifact_bundle(
            aggregate_report,
            output_dir=output_dir,
        )
    return aggregate_report


def _eta_run_metric_map(summary: ETAProofPaperSuiteRunSummary) -> dict[str, float]:
    return dict(summary.metric_values)


def _build_eta_paper_suite_pairwise_effects(
    run_summaries: tuple[ETAProofPaperSuiteRunSummary, ...],
) -> tuple[PairwiseMetricEffect, ...]:
    if not run_summaries:
        return ()
    full_values = tuple(
        _eta_run_metric_map(summary).get("heldout_strong_success_rate", 0.0)
        for summary in run_summaries
    )
    strongest_control_values = tuple(
        _eta_run_metric_map(summary).get("heldout_strong_success_rate", 0.0)
        - _eta_run_metric_map(summary).get("strong_success_gap_vs_best_control", 0.0)
        for summary in run_summaries
    )
    return (
        build_pairwise_metric_effect(
            metric_name="heldout_strong_success_rate",
            candidate_label="full-internal-rl",
            control_label="strongest-control",
            candidate_values=full_values,
            control_values=strongest_control_values,
        ),
    )


def _eta_claim_status(*, retain_checks: tuple[bool, ...], weak_checks: tuple[bool, ...] = ()) -> str:
    if retain_checks and all(retain_checks):
        return "retain"
    if weak_checks and all(weak_checks):
        return "weak"
    if retain_checks and any(retain_checks):
        return "weak"
    return "fail"


def _build_eta_claim_verdicts(
    aggregate_report: ETAProofPaperSuiteAggregateReport,
) -> tuple[ClaimVerdict, ...]:
    assessment = aggregate_report.reference_assessment
    blocked_gate_ids = set(assessment.gates[i].gate_id for i in range(len(assessment.gates)) if not assessment.gates[i].passed) if assessment is not None else set()
    strongest_control_effect = next(iter(aggregate_report.pairwise_effects), None)
    statistical_gate_passed = "statistical-batch-evidence" not in blocked_gate_ids
    claim_internal_rl = _eta_claim_status(
        retain_checks=(
            assessment is not None and assessment.passed_gate_count >= assessment.total_gate_count,
            strongest_control_effect is not None and strongest_control_effect.ci_low > 0.0,
            statistical_gate_passed,
        ),
        weak_checks=(
            assessment is not None and assessment.passed_gate_count >= max(1, assessment.total_gate_count - 1),
            strongest_control_effect is not None and strongest_control_effect.mean_delta > 0.0,
        ),
    )
    return (
        ClaimVerdict(
            claim_id="claim_eta_internal_rl_advantage",
            status=claim_internal_rl,
            required_gate_ids=(
                "sparse-reward-success",
                "abstract-action-reuse",
                "heldout-composition",
                "credit-alignment",
                "policy-update-evidence",
                "statistical-batch-evidence",
                "backend-robustness",
            ),
            supporting_artifacts=("paper_suite_aggregate", "reference_assessment"),
            evidence=(
                ("passed_gate_count", float(assessment.passed_gate_count) if assessment is not None else 0.0),
                ("total_gate_count", float(assessment.total_gate_count) if assessment is not None else 0.0),
                ("strong_success_gap_ci_low", strongest_control_effect.ci_low if strongest_control_effect is not None else 0.0),
                ("strong_success_gap_mean_delta", strongest_control_effect.mean_delta if strongest_control_effect is not None else 0.0),
                ("statistical_batch_evidence_passed", float(statistical_gate_passed)),
            ),
            summary="ETA internal-RL strong-proof claim verdict.",
            description="Checks whether full internal RL stays ahead of the strongest control with retained acceptance gates and statistical evidence.",
        ),
    )


def build_eta_paper_suite_evidence_bundle(
    aggregate_report: ETAProofPaperSuiteAggregateReport,
) -> EvidenceBundle:
    reference_artifacts: list[tuple[str, Any]] = []
    if aggregate_report.reference_benchmark_report is not None:
        reference_artifacts.append(("reference_benchmark_report", aggregate_report.reference_benchmark_report))
    if aggregate_report.reference_backend_report is not None:
        reference_artifacts.append(("reference_backend_report", aggregate_report.reference_backend_report))
    if aggregate_report.reference_assessment is not None:
        reference_artifacts.append(("reference_assessment", aggregate_report.reference_assessment))
    return EvidenceBundle(
        bundle_id=f"{aggregate_report.manifest.suite_id}:evidence-bundle",
        suite_kind=aggregate_report.manifest.suite_kind,
        manifest=aggregate_report.manifest,
        provenance=aggregate_report.provenance,
        run_summaries=aggregate_report.run_summaries,
        aggregate_metrics={
            "primary_metric_summaries": aggregate_report.primary_metric_summaries,
            "secondary_metric_summaries": aggregate_report.secondary_metric_summaries,
            "interpretation_summary": aggregate_report.interpretation_summary,
        },
        pairwise_effects=aggregate_report.pairwise_effects,
        reference_artifacts=tuple(reference_artifacts),
        claim_verdicts=aggregate_report.claim_verdicts,
        description=f"Unified ETA evidence bundle for {aggregate_report.manifest.suite_id}.",
    )


def export_eta_internal_rl_paper_suite_artifact_bundle(
    aggregate_report: ETAProofPaperSuiteAggregateReport,
    *,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    written_paths = [
        export_json_artifact(
            payload=aggregate_report.manifest,
            output_path=target_dir / "paper_suite_manifest.json",
        ),
        export_json_artifact(
            payload=aggregate_report.provenance,
            output_path=target_dir / "paper_suite_provenance.json",
        ),
        export_json_artifact(
            payload=aggregate_report.run_summaries,
            output_path=target_dir / "paper_suite_run_summaries.json",
        ),
        export_json_artifact(
            payload={
                "suite_id": aggregate_report.manifest.suite_id,
                "primary_metric_summaries": aggregate_report.primary_metric_summaries,
                "secondary_metric_summaries": aggregate_report.secondary_metric_summaries,
                "pairwise_effects": aggregate_report.pairwise_effects,
                "claim_verdicts": aggregate_report.claim_verdicts,
                "interpretation_summary": aggregate_report.interpretation_summary,
                "description": aggregate_report.description,
            },
            output_path=target_dir / "paper_suite_aggregate.json",
        ),
    ]
    if aggregate_report.reference_benchmark_report is not None:
        written_paths.append(
            export_json_artifact(
                payload=aggregate_report.reference_benchmark_report,
                output_path=target_dir / "reference_benchmark_report.json",
            )
        )
    if aggregate_report.reference_backend_report is not None:
        written_paths.append(
            export_json_artifact(
                payload=aggregate_report.reference_backend_report,
                output_path=target_dir / "reference_backend_report.json",
            )
        )
    if aggregate_report.reference_assessment is not None:
        written_paths.append(
            export_json_artifact(
                payload=aggregate_report.reference_assessment,
                output_path=target_dir / "reference_assessment.json",
            )
        )
    if aggregate_report.interpretation_summary is not None:
        written_paths.append(
            export_json_artifact(
                payload=aggregate_report.interpretation_summary,
                output_path=target_dir / "paper_suite_interpretation_summary.json",
            )
        )
    written_paths.append(
        export_json_artifact(
            payload=build_eta_paper_suite_evidence_bundle(aggregate_report),
            output_path=target_dir / "evidence_bundle.json",
        )
    )
    return tuple(written_paths)
