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
    InternalRLRewardSource,
    InternalRLSandbox,
    MiniHierarchicalEnvironment,
    ZRollout,
)
from volvence_zero.memory import Track
from volvence_zero.substrate import (
    FeatureSignal,
    LocalSubstrateRuntimeMode,
    NoOpResidualInterventionBackend,
    OpenWeightResidualRuntime,
    ResidualSequenceStep,
    SubstrateSnapshot,
    SubstrateFallbackMode,
    SurfaceKind,
    SyntheticOpenWeightResidualRuntime,
    build_transformers_runtime_with_fallback,
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
    split_detail: str = "unspecified"
    reward_profile: str = "proof-sparse-terminal-delayed"
    reward_taxonomy: tuple[InternalRLRewardSource, ...] = ()
    route_length: int = 0
    distractor_count: int = 0


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
    split_detail: str
    reward_profile: str
    reward_source_mix: tuple[tuple[str, str, float], ...]
    reward_shaping_leakage: float
    reward_sparsity: float
    route_length: int
    distractor_count: int
    mean_steps_per_abstract_action: float
    median_steps_per_abstract_action: float
    persistence_window_success_rate: float
    premature_switch_rate: float
    always_switch_rate: float
    never_switch_rate: float
    intervention_application_count: int
    mean_replacement_effect_delta: float
    residual_signal_quality: float
    first_missed_subgoal: str
    family_miss_rate: float
    credit_window_miss_rate: float
    terminal_credit_coverage: float
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


@dataclass(frozen=True)
class ETAOpenWeightRuntimeConfig:
    model_id: str = "distilgpt2"
    model_source: str | None = None
    device: str = "auto"
    layer_indices: tuple[int, ...] | None = None
    local_files_only: bool = True
    runtime_mode: LocalSubstrateRuntimeMode | str | None = LocalSubstrateRuntimeMode.STRICT_LOCAL
    fallback_mode: SubstrateFallbackMode | str | None = SubstrateFallbackMode.DENY
    builtin_model_id: str = "eta-builtin-transformers-runtime"
    max_prefix_steps: int = 6
    require_real_backend: bool = True
    description: str = "ETA open-weight runtime config for real residual capture/control."


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


def _median(values: tuple[float, ...]) -> float:
    if not values:
        return 0.0
    sorted_values = tuple(sorted(values))
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return sorted_values[midpoint]
    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2.0


def _build_eta_open_weight_runtime(config: ETAOpenWeightRuntimeConfig | None = None) -> OpenWeightResidualRuntime:
    active_config = config or ETAOpenWeightRuntimeConfig()
    return build_transformers_runtime_with_fallback(
        model_id=active_config.model_id,
        model_source=active_config.model_source,
        device=active_config.device,
        layer_indices=active_config.layer_indices,
        local_files_only=active_config.local_files_only,
        fallback_mode=active_config.fallback_mode,
        runtime_mode=active_config.runtime_mode,
        builtin_model_id=active_config.builtin_model_id,
        allow_live_substrate_mutation=False,
    )


def _validate_eta_open_weight_runtime(
    *,
    runtime: OpenWeightResidualRuntime,
    config: ETAOpenWeightRuntimeConfig | None = None,
) -> None:
    active_config = config or ETAOpenWeightRuntimeConfig()
    if not active_config.require_real_backend:
        return
    if runtime.fallback_active:
        raise RuntimeError(
            f"ETA real residual lane requires a non-fallback transformers backend, got "
            f"model_id={runtime.model_id!r} runtime_origin={runtime.runtime_origin!r}. "
            "Use a locally available transformers model or set require_real_backend=False for explicit smoke tests."
        )


def _runtime_capture_snapshot(
    *,
    runtime: OpenWeightResidualRuntime,
    case: ETAProofCase,
    source_text: str,
    step_index: int,
    total_steps: int,
) -> SubstrateSnapshot:
    capture = runtime.capture(source_text=source_text)
    runtime_origin = getattr(runtime, "runtime_origin", "unknown")
    fallback_active = 1.0 if getattr(runtime, "fallback_active", False) else 0.0
    feature_surface = capture.feature_surface + (
        FeatureSignal(
            name="eta_real_runtime_step_index",
            values=(step_index / max(total_steps - 1, 1),),
            source="eta-open-weight-step-contract",
        ),
        FeatureSignal(
            name="eta_real_runtime_prefix_fraction",
            values=(len(source_text.split()) / max(len(case.source_text.split()), 1),),
            source="eta-open-weight-step-contract",
        ),
        FeatureSignal(
            name="eta_real_runtime_capture_present",
            values=(1.0,),
            source="eta-open-weight-step-contract",
        ),
        FeatureSignal(
            name="eta_real_runtime_fallback_active",
            values=(fallback_active,),
            source="eta-open-weight-step-contract",
        ),
    )
    return SubstrateSnapshot(
        model_id=runtime.model_id,
        is_frozen=runtime.is_frozen,
        surface_kind=SurfaceKind.RESIDUAL_STREAM,
        token_logits=capture.token_logits,
        feature_surface=feature_surface,
        residual_activations=capture.residual_activations,
        residual_sequence=capture.residual_sequence,
        unavailable_fields=(),
        description=(
            f"{capture.description} ETA real residual step {step_index + 1}/{total_steps} "
            f"case={case.case_id} runtime_origin={runtime_origin} fallback_active={int(fallback_active)}."
        ),
    )


def _eta_real_residual_prefixes(
    source_text: str,
    *,
    max_prefix_steps: int,
) -> tuple[str, ...]:
    tokens = tuple(part for part in source_text.split() if part.strip())
    if not tokens:
        return (source_text or "<empty>",)
    max_steps = max(1, min(max_prefix_steps, len(tokens)))
    if max_steps == len(tokens):
        return tuple(" ".join(tokens[: index + 1]) for index in range(len(tokens)))
    positions = tuple(
        min(len(tokens), max(1, round((index + 1) * len(tokens) / max_steps)))
        for index in range(max_steps)
    )
    deduped_positions = tuple(dict.fromkeys(positions))
    return tuple(" ".join(tokens[:position]) for position in deduped_positions)


def _feature_mean(
    snapshots: tuple[SubstrateSnapshot, ...],
    *,
    feature_name: str,
) -> float:
    values = tuple(
        float(value)
        for snapshot in snapshots
        for feature in snapshot.feature_surface
        if feature.name == feature_name
        for value in feature.values
    )
    return _mean(values)


def _real_snapshot_metric_values(snapshots: tuple[SubstrateSnapshot, ...]) -> tuple[tuple[str, float], ...]:
    if not snapshots:
        return (
            ("real_open_weight_step_count", 0.0),
            ("real_open_weight_capture_rate", 0.0),
            ("real_open_weight_hook_coverage", 0.0),
            ("real_open_weight_fallback_rate", 0.0),
        )
    return (
        ("real_open_weight_step_count", float(len(snapshots))),
        (
            "real_open_weight_capture_rate",
            _feature_mean(snapshots, feature_name="eta_real_runtime_capture_present"),
        ),
        (
            "real_open_weight_hook_coverage",
            _feature_mean(snapshots, feature_name="hook_layer_coverage"),
        ),
        (
            "real_open_weight_fallback_rate",
            _feature_mean(snapshots, feature_name="eta_real_runtime_fallback_active"),
        ),
    )


def _switch_discipline_score(*, switch_sparsity: float, mean_persistence: float) -> float:
    transition_presence = max(0.0, min(1.0, (1.0 - switch_sparsity) / 0.18))
    persistence_quality = max(0.0, min(1.0, mean_persistence / 2.0))
    return max(0.0, min(1.0, transition_presence * 0.40 + persistence_quality * 0.60))


def _abstract_action_window_lengths(rollout: ZRollout) -> tuple[int, ...]:
    if not rollout.transitions:
        return ()
    lengths: list[int] = []
    current_length = 0
    for index, transition in enumerate(rollout.transitions):
        is_new_window = index == 0 or transition.controller_state.switch_gate >= 0.5
        if is_new_window and current_length > 0:
            lengths.append(current_length)
            current_length = 0
        current_length += 1
    if current_length > 0:
        lengths.append(current_length)
    return tuple(lengths)


def _abstract_action_windows(rollout: ZRollout) -> tuple[tuple[int, int], ...]:
    if not rollout.transitions:
        return ()
    windows: list[tuple[int, int]] = []
    start = 0
    for index, transition in enumerate(rollout.transitions):
        if index > 0 and transition.controller_state.switch_gate >= 0.5:
            windows.append((start, index - 1))
            start = index
    windows.append((start, len(rollout.transitions) - 1))
    return tuple(windows)


def _assignment_window_bounds(
    *,
    rollout: ZRollout,
    start_step: int,
    end_step: int,
) -> tuple[int, int]:
    if not rollout.transitions:
        return (0, -1)
    assignment_start = max(0, start_step)
    assignment_end = min(len(rollout.transitions) - 1, end_step)
    if assignment_end < assignment_start:
        return (0, -1)
    overlapping = tuple(
        (start, end)
        for start, end in _abstract_action_windows(rollout)
        if end >= assignment_start and start <= assignment_end
    )
    if not overlapping:
        return (assignment_start, assignment_end)
    return (
        min(start for start, _ in overlapping),
        max(end for _, end in overlapping),
    )


def _credit_window_miss_rate(rollout: ZRollout) -> float:
    if not rollout.delayed_credit_assignments:
        return 1.0 if rollout.completed_subgoals else 0.0
    misses = 0
    for assignment in rollout.delayed_credit_assignments:
        start, end = _assignment_window_bounds(
            rollout=rollout,
            start_step=assignment.start_step,
            end_step=assignment.end_step,
        )
        if end < start:
            misses += 1
            continue
        has_family = any(
            transition.active_family_id not in {None, "unassigned"}
            for transition in rollout.transitions[start : end + 1]
        )
        if not has_family:
            misses += 1
    return misses / len(rollout.delayed_credit_assignments)


def _persistence_metrics(rollout: ZRollout) -> tuple[float, float, float, float, float, float]:
    window_lengths = _abstract_action_window_lengths(rollout)
    if not window_lengths:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    transition_count = len(rollout.transitions)
    mean_steps = _mean(tuple(float(length) for length in window_lengths))
    median_steps = _median(tuple(float(length) for length in window_lengths))
    persistent_windows = sum(1 for length in window_lengths if length >= 2)
    one_step_windows = sum(1 for length in window_lengths if length <= 1)
    switch_count = max(len(window_lengths) - 1, 0)
    return (
        mean_steps,
        median_steps,
        persistent_windows / len(window_lengths),
        one_step_windows / len(window_lengths),
        1.0 if switch_count >= max(transition_count - 1, 0) and transition_count > 1 else 0.0,
        1.0 if switch_count == 0 and transition_count > 1 else 0.0,
    )


def _mean_control_magnitude(values: tuple[float, ...]) -> float:
    if not values:
        return 0.0
    return sum(abs(value) for value in values) / len(values)


def _temporal_family_slow_loop_metrics(policy: FullLearnedTemporalPolicy) -> tuple[tuple[str, float], ...]:
    families = policy.parameter_store.action_families
    if not families:
        return (
            ("credit_to_family_write_count", 0.0),
            ("long_horizon_payoff_coverage", 0.0),
            ("family_competition_mean", 0.0),
            ("family_payoff_mean", 0.0),
        )
    credit_write_count = sum(1 for family in families if abs(family.delayed_credit_sum) > 1e-8)
    payoff_covered_count = sum(
        1
        for family in families
        if abs(family.long_term_payoff - 0.5) > 1e-8 or abs(family.delayed_credit_sum) > 1e-8
    )
    return (
        ("credit_to_family_write_count", float(credit_write_count)),
        ("long_horizon_payoff_coverage", payoff_covered_count / len(families)),
        ("family_competition_mean", _mean(tuple(family.competition_score for family in families))),
        ("family_payoff_mean", _mean(tuple(family.long_term_payoff for family in families))),
    )


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
            split_detail="train-core",
            description="Canonical branching corridor task with alpha -> beta -> delta.",
        ),
        HierarchicalRouteSpec(
            case_id="train-zigzag-beta-gamma-alpha",
            split="train",
            source_text="careful repair planning warmth continuity zigzag support",
            waypoints=("entry", "beta", "gamma", "alpha"),
            distractor_ids=("delta",),
            split_detail="train-order-pressure",
            description="Zigzag training task that revisits familiar signatures under order pressure.",
        ),
        HierarchicalRouteSpec(
            case_id="train-loop-delta-beta-epsilon",
            split="train",
            source_text="reflective planning support loop memory return and repair",
            waypoints=("entry", "delta", "hub", "beta", "epsilon"),
            distractor_ids=("gamma",),
            split_detail="train-loop",
            description="Looped training task that pushes persistence before the final subgoal.",
        ),
        HierarchicalRouteSpec(
            case_id="eval-alpha-delta-epsilon",
            split="eval",
            source_text="steady continuity planning support reflection branch return",
            waypoints=("entry", "alpha", "delta", "epsilon"),
            distractor_ids=("beta", "gamma"),
            split_detail="eval-composition",
            description="Evaluation task that mixes known branch nodes with a new terminal demand.",
        ),
        HierarchicalRouteSpec(
            case_id="eval-gamma-beta-delta",
            split="eval",
            source_text="careful guidance through dense branch with repair continuity",
            waypoints=("entry", "beta", "gamma", "beta", "delta"),
            distractor_ids=("alpha", "epsilon"),
            split_detail="eval-distractor",
            description="Evaluation task that reuses beta/delta under a denser distractor field.",
        ),
        HierarchicalRouteSpec(
            case_id="heldout-delta-alpha-gamma-epsilon",
            split="heldout",
            source_text="reflective planning support with careful guidance through heldout branch",
            waypoints=("entry", "delta", "alpha", "gamma", "epsilon"),
            distractor_ids=("beta",),
            split_detail="heldout-composition",
            description="Held-out compositional route with four-stage reordering and a hard terminal node.",
        ),
        HierarchicalRouteSpec(
            case_id="heldout-epsilon-beta-alpha-delta",
            split="heldout",
            source_text="careful branching return support and reflection in heldout loop",
            waypoints=("entry", "hub", "epsilon", "beta", "alpha", "delta"),
            distractor_ids=("gamma",),
            split_detail="heldout-long-loop",
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
            split_detail=case.split_detail,
            reward_profile=case.proof_episode.reward_profile,
            reward_taxonomy=case.proof_episode.reward_taxonomy,
            route_length=len(case.route_signature),
            distractor_count=len(case.proof_episode.distractor_signatures),
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
        route_ids = tuple(route.case_id for route in default_eta_proof_routes())
        train_epochs = 2
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
            PaperMetricSpec(
                metric_name="mean_steps_per_abstract_action",
                role="secondary",
                direction="higher-is-better",
                description="Mean rollout steps controlled by one abstract action before switching.",
            ),
            PaperMetricSpec(
                metric_name="persistence_window_success_rate",
                role="secondary",
                direction="higher-is-better",
                description="Fraction of abstract-action windows persisting for at least two rollout steps.",
            ),
            PaperMetricSpec(
                metric_name="residual_signal_quality",
                role="secondary",
                direction="higher-is-better",
                description="Residual-control signal quality combining downstream effect and backend fidelity.",
            ),
            PaperMetricSpec(
                metric_name="credit_to_family_write_count",
                role="secondary",
                direction="higher-is-better",
                description="Number of temporal families receiving long-horizon delayed-credit writes.",
            ),
            PaperMetricSpec(
                metric_name="long_horizon_payoff_coverage",
                role="secondary",
                direction="higher-is-better",
                description="Fraction of temporal families with payoff or delayed-credit evidence.",
            ),
            PaperMetricSpec(
                metric_name="slow_to_fast_init_benefit",
                role="secondary",
                direction="higher-is-better",
                description="Fast-prior benefit retained over the no-fast-prior control.",
            ),
            PaperMetricSpec(
                metric_name="mean_reward_sparsity",
                role="secondary",
                direction="higher-is-better",
                description="Fraction of reward signal not attributable to shaping leakage.",
            ),
            PaperMetricSpec(
                metric_name="reward_shaping_leakage",
                role="secondary",
                direction="lower-is-better",
                description="Share of optimizer-visible reward coming from shaping components.",
            ),
            PaperMetricSpec(
                metric_name="heldout_family_reuse_gap_vs_no_fast_prior",
                role="secondary",
                direction="higher-is-better",
                description="Held-out family reuse retained over the no-fast-prior scaffold ablation.",
            ),
            PaperMetricSpec(
                metric_name="heldout_credit_window_miss_rate",
                role="secondary",
                direction="lower-is-better",
                description="Held-out delayed-credit assignments whose abstract-action window lacks an assigned family.",
            ),
            PaperMetricSpec(
                metric_name="heldout_terminal_credit_coverage",
                role="secondary",
                direction="higher-is-better",
                description="Held-out terminal successes backed by terminal delayed-credit assignments.",
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


def build_eta_open_weight_paper_suite_manifest(
    *,
    suite_tier: str = "ci-smoke",
) -> PaperSuiteManifest:
    base_manifest = build_eta_proof_paper_suite_manifest(suite_tier=suite_tier)
    case_groups = tuple(
        ("backend_labels", ("transformers-open-weight", "trace"))
        if name == "backend_labels"
        else (name, values)
        for name, values in base_manifest.case_groups
    )
    secondary_metrics = base_manifest.secondary_metrics + (
        PaperMetricSpec(
            metric_name="real_open_weight_capture_rate",
            role="secondary",
            direction="higher-is-better",
            description="Fraction of ETA substrate steps captured from a real open-weight runtime.",
        ),
        PaperMetricSpec(
            metric_name="real_open_weight_hook_coverage",
            role="secondary",
            direction="higher-is-better",
            description="Mean hook-layer coverage observed during real open-weight ETA capture.",
        ),
        PaperMetricSpec(
            metric_name="real_open_weight_fallback_rate",
            role="secondary",
            direction="lower-is-better",
            description="Fraction of real ETA capture served by a builtin fallback runtime.",
        ),
        PaperMetricSpec(
            metric_name="intervention_application_count",
            role="secondary",
            direction="higher-is-better",
            description="Mean residual intervention application count in ETA open-weight proof rollouts.",
        ),
        PaperMetricSpec(
            metric_name="episode_replacement_effect_delta",
            role="secondary",
            direction="higher-is-better",
            description="Mean per-episode replacement-effect delta for real residual-control rollouts.",
        ),
    )
    return replace(
        base_manifest,
        suite_id=f"eta-open-weight-{suite_tier}",
        suite_kind="eta-open-weight-residual-proof",
        version=base_manifest.version + 1,
        primary_metrics=base_manifest.primary_metrics + (
            PaperMetricSpec(
                metric_name="real_residual_policy_gap_vs_control",
                role="primary",
                direction="higher-is-better",
                description=(
                    "Composite real-residual policy/control advantage over same-backend matched controls, "
                    "excluding backend-ablation noops."
                ),
            ),
        ),
        case_groups=case_groups,
        secondary_metrics=secondary_metrics,
        artifact_expectations=base_manifest.artifact_expectations + (
            "real open-weight residual capture/control evidence",
        ),
        description=(
            f"Open-weight ETA paper suite {suite_tier} using real transformer residual capture/control "
            "as the primary backend and trace as the matched fallback control."
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
    open_weight_runtime: OpenWeightResidualRuntime | None = None,
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
    if profile.use_noop_backend:
        runtime = None
    elif backend_label == "synthetic-open-weight":
        runtime = SyntheticOpenWeightResidualRuntime(model_id=f"eta-proof:{profile.profile_label}")
    elif backend_label == "transformers-open-weight":
        runtime = open_weight_runtime or _build_eta_open_weight_runtime()
    elif backend_label == "trace":
        runtime = None
    else:
        raise ValueError(f"Unsupported backend label {backend_label!r}")
    return InternalRLSandbox(policy=policy, env=env, residual_runtime=runtime)


def _build_case_snapshots(
    case: ETAProofCase,
    *,
    open_weight_runtime: OpenWeightResidualRuntime | None = None,
    open_weight_config: ETAOpenWeightRuntimeConfig | None = None,
) -> tuple[SubstrateSnapshot, ...]:
    if open_weight_runtime is not None:
        active_config = open_weight_config or ETAOpenWeightRuntimeConfig()
        required_prefix_steps = max(
            active_config.max_prefix_steps,
            sum(max(subgoal.min_persistence, 1) for subgoal in case.proof_episode.subgoals),
        )
        prefixes = _eta_real_residual_prefixes(
            case.source_text,
            max_prefix_steps=required_prefix_steps,
        )
        return tuple(
            _runtime_capture_snapshot(
                runtime=open_weight_runtime,
                case=case,
                source_text=prefix,
                step_index=step_index,
                total_steps=len(prefixes),
            )
            for step_index, prefix in enumerate(prefixes)
        )
    trace = build_training_trace(trace_id=case.case_id, source_text=case.source_text)
    return tuple(_snapshot_from_step(trace.trace_id, step) for step in trace.steps)


def _credit_alignment_score(rollout: ZRollout) -> float:
    sandbox = InternalRLSandbox()
    return sandbox._delayed_credit_alignment((rollout,))


def _reward_source_mix(
    *,
    proof_episode: InternalRLProofEpisode,
    rollout: ZRollout,
) -> tuple[tuple[str, str, float], ...]:
    totals: dict[tuple[str, str], float] = {}
    for transition in rollout.transitions:
        for component_name, value in transition.reward_components:
            kind = proof_episode.reward_kind_for(component_name)
            key = (component_name, kind)
            totals[key] = totals.get(key, 0.0) + abs(value)
    return tuple(
        (component_name, kind, round(total, 4))
        for (component_name, kind), total in sorted(totals.items())
    )


def _reward_shaping_leakage(reward_source_mix: tuple[tuple[str, str, float], ...]) -> float:
    total = sum(value for _, _, value in reward_source_mix)
    if total <= 1e-8:
        return 0.0
    shaping = sum(value for _, kind, value in reward_source_mix if kind == "shaping")
    return round(shaping / total, 4)


def _episode_report(
    *,
    case: ETAProofCase,
    rollout: ZRollout,
    profile_label: str,
    backend_label: str,
    family_registry: dict[str, set[str]],
) -> ETAProofEpisodeReport:
    completed_pairs = tuple(zip(rollout.completed_subgoals, rollout.completed_family_ids, strict=True))
    expected_subgoal_ids = tuple(subgoal.subgoal_id for subgoal in case.proof_episode.subgoals)
    first_missed_subgoal = next(
        (subgoal_id for subgoal_id in expected_subgoal_ids if subgoal_id not in rollout.completed_subgoals),
        "none",
    )
    family_miss_rate = (
        sum(1.0 for family_id in rollout.completed_family_ids if family_id == "unassigned")
        / len(rollout.completed_family_ids)
        if rollout.completed_family_ids
        else (1.0 if rollout.completed_subgoals else 0.0)
    )
    credit_window_miss_rate = _credit_window_miss_rate(rollout)
    terminal_credit_coverage = float(
        rollout.terminal_success
        and any(assignment.reason == "terminal-success" for assignment in rollout.delayed_credit_assignments)
    )
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
    reward_source_mix = _reward_source_mix(
        proof_episode=case.proof_episode,
        rollout=rollout,
    )
    reward_shaping_leakage = _reward_shaping_leakage(reward_source_mix)
    reward_sparsity = round(1.0 - reward_shaping_leakage, 4)
    (
        mean_steps_per_abstract_action,
        median_steps_per_abstract_action,
        persistence_window_success_rate,
        premature_switch_rate,
        always_switch_rate,
        never_switch_rate,
    ) = _persistence_metrics(rollout)
    intervention_application_count = sum(
        1
        for transition in rollout.transitions
        if transition.backend_name != "noop-residual-backend"
    )
    mean_replacement_effect_delta = _mean(tuple(transition.replacement_effect_delta for transition in rollout.transitions))
    residual_signal_quality = _mean(
        tuple(
            min(
                _mean_control_magnitude(transition.downstream_effect) * 2.0
                + transition.backend_fidelity * 0.50,
                1.0,
            )
            for transition in rollout.transitions
        )
    )
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
        split_detail=case.split_detail,
        reward_profile=case.reward_profile,
        reward_source_mix=reward_source_mix,
        reward_shaping_leakage=reward_shaping_leakage,
        reward_sparsity=reward_sparsity,
        route_length=case.route_length,
        distractor_count=case.distractor_count,
        mean_steps_per_abstract_action=mean_steps_per_abstract_action,
        median_steps_per_abstract_action=median_steps_per_abstract_action,
        persistence_window_success_rate=persistence_window_success_rate,
        premature_switch_rate=premature_switch_rate,
        always_switch_rate=always_switch_rate,
        never_switch_rate=never_switch_rate,
        intervention_application_count=intervention_application_count,
        mean_replacement_effect_delta=mean_replacement_effect_delta,
        residual_signal_quality=residual_signal_quality,
        first_missed_subgoal=first_missed_subgoal,
        family_miss_rate=family_miss_rate,
        credit_window_miss_rate=credit_window_miss_rate,
        terminal_credit_coverage=terminal_credit_coverage,
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
        ("mean_reward_sparsity", _mean(tuple(report.reward_sparsity for report in episode_reports))),
        ("heldout_reward_sparsity", _mean(tuple(report.reward_sparsity for report in heldout_reports))),
        ("reward_shaping_leakage", _mean(tuple(report.reward_shaping_leakage for report in episode_reports))),
        ("heldout_reward_shaping_leakage", _mean(tuple(report.reward_shaping_leakage for report in heldout_reports))),
        ("mean_route_length", _mean(tuple(float(report.route_length) for report in episode_reports))),
        ("mean_distractor_count", _mean(tuple(float(report.distractor_count) for report in episode_reports))),
        (
            "mean_steps_per_abstract_action",
            _mean(tuple(report.mean_steps_per_abstract_action for report in episode_reports)),
        ),
        (
            "median_steps_per_abstract_action",
            _median(tuple(report.median_steps_per_abstract_action for report in episode_reports)),
        ),
        (
            "persistence_window_success_rate",
            _mean(tuple(report.persistence_window_success_rate for report in episode_reports)),
        ),
        ("premature_switch_rate", _mean(tuple(report.premature_switch_rate for report in episode_reports))),
        ("always_switch_rate", _mean(tuple(report.always_switch_rate for report in episode_reports))),
        ("never_switch_rate", _mean(tuple(report.never_switch_rate for report in episode_reports))),
        (
            "intervention_application_count",
            _mean(tuple(float(report.intervention_application_count) for report in episode_reports)),
        ),
        (
            "episode_replacement_effect_delta",
            _mean(tuple(report.mean_replacement_effect_delta for report in episode_reports)),
        ),
        ("residual_signal_quality", _mean(tuple(report.residual_signal_quality for report in episode_reports))),
        ("family_miss_rate", _mean(tuple(report.family_miss_rate for report in episode_reports))),
        ("heldout_family_miss_rate", _mean(tuple(report.family_miss_rate for report in heldout_reports))),
        ("credit_window_miss_rate", _mean(tuple(report.credit_window_miss_rate for report in episode_reports))),
        ("heldout_credit_window_miss_rate", _mean(tuple(report.credit_window_miss_rate for report in heldout_reports))),
        ("terminal_credit_coverage", _mean(tuple(report.terminal_credit_coverage for report in episode_reports))),
        ("heldout_terminal_credit_coverage", _mean(tuple(report.terminal_credit_coverage for report in heldout_reports))),
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


def _real_residual_control_score(report: ETAProofProfileReport) -> float:
    metrics = dict(report.metric_means)
    causal_replacement = 0.0 if report.profile_label == "full-no-replacement" else 1.0
    real_backend_evidence = min(metrics.get("real_open_weight_capture_rate", 0.0), 1.0) * (
        1.0 - min(metrics.get("real_open_weight_fallback_rate", 1.0), 1.0)
    )
    return round(
        metrics.get("heldout_strong_success_rate", 0.0)
        + max(0.0, report.mean_replacement_effect_delta) * 0.25
        + min(report.training_parameter_change_rate, 1.0) * 0.025
        + causal_replacement * 0.025
        + real_backend_evidence * 0.025,
        4,
    )


def _real_residual_policy_control_reports(
    *,
    report_map: dict[str, ETAProofProfileReport],
) -> tuple[ETAProofProfileReport, ...]:
    excluded_labels = {"full-internal-rl", "full-bootstrap-init", "noop-backend"}
    return tuple(
        report
        for label, report in report_map.items()
        if label not in excluded_labels
    )


def _best_real_residual_policy_control(
    *,
    report_map: dict[str, ETAProofProfileReport],
) -> ETAProofProfileReport | None:
    controls = _real_residual_policy_control_reports(report_map=report_map)
    best_report: ETAProofProfileReport | None = None
    best_score = float("-inf")
    for report in controls:
        candidate_score = _real_residual_control_score(report)
        if candidate_score > best_score:
            best_report = report
            best_score = candidate_score
    return best_report


def run_eta_internal_rl_proof_benchmark(
    *,
    cases: tuple[ETAProofCase, ...] = default_eta_proof_cases(),
    profile_labels: tuple[str, ...] = default_eta_proof_profiles(),
    baseline_label: str = "full-internal-rl",
    backend_label: str = "trace",
    train_epochs: int = 2,
    open_weight_runtime: OpenWeightResidualRuntime | None = None,
    open_weight_config: ETAOpenWeightRuntimeConfig | None = None,
    use_real_substrate_steps: bool | None = None,
) -> ETAProofBenchmarkReport:
    profile_reports: list[ETAProofProfileReport] = []
    benchmark_rollout_batch_count = 0
    active_open_weight_runtime = open_weight_runtime
    if backend_label == "transformers-open-weight" and active_open_weight_runtime is None:
        active_open_weight_runtime = _build_eta_open_weight_runtime(open_weight_config)
    if backend_label == "transformers-open-weight" and active_open_weight_runtime is not None:
        _validate_eta_open_weight_runtime(
            runtime=active_open_weight_runtime,
            config=open_weight_config,
        )
    real_steps_enabled = (
        backend_label == "transformers-open-weight"
        if use_real_substrate_steps is None
        else use_real_substrate_steps
    )
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
            open_weight_runtime=active_open_weight_runtime,
        )
        family_registry: dict[str, set[str]] = {}
        training_update_count = 0
        rollout_batch_count = 0
        training_transition_count = 0
        training_parameter_change_count = 0
        parameter_change_norms: list[float] = []
        value_losses: list[float] = []
        replacement_effect_deltas: list[float] = []
        real_substrate_snapshots: list[SubstrateSnapshot] = []
        for epoch in range(train_epochs):
            train_rollouts: list[ZRollout] = []
            for case in train_cases:
                snapshots = _build_case_snapshots(
                    case,
                    open_weight_runtime=active_open_weight_runtime if real_steps_enabled and not profile.use_noop_backend else None,
                    open_weight_config=open_weight_config,
                )
                if real_steps_enabled and not profile.use_noop_backend:
                    real_substrate_snapshots.extend(snapshots)
                if backend_label in {"synthetic-open-weight", "transformers-open-weight"} and not profile.use_noop_backend:
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
            snapshots = _build_case_snapshots(
                case,
                open_weight_runtime=active_open_weight_runtime if real_steps_enabled and not profile.use_noop_backend else None,
                open_weight_config=open_weight_config,
            )
            if real_steps_enabled and not profile.use_noop_backend:
                real_substrate_snapshots.extend(snapshots)
            if backend_label in {"synthetic-open-weight", "transformers-open-weight"} and not profile.use_noop_backend:
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
            *_temporal_family_slow_loop_metrics(sandbox.policy),
            *_real_snapshot_metric_values(tuple(real_substrate_snapshots)),
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
            (
                "slow_to_fast_init_benefit",
                baseline_metrics.get("temporal_fast_prior_strength", 0.0)
                - no_fast_prior_metrics.get("temporal_fast_prior_strength", 0.0),
            ),
            (
                "family_reuse_after_reset",
                baseline_metrics.get("heldout_family_reuse_rate", 0.0),
            ),
            (
                "heldout_gain_after_consolidation",
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


def run_eta_open_weight_residual_benchmark(
    *,
    cases: tuple[ETAProofCase, ...] = default_eta_proof_cases(),
    profile_labels: tuple[str, ...] = ("full-internal-rl", "full-no-optimize", "noop-backend"),
    baseline_label: str = "full-internal-rl",
    train_epochs: int = 1,
    runtime_config: ETAOpenWeightRuntimeConfig | None = None,
    runtime: OpenWeightResidualRuntime | None = None,
) -> ETAProofBenchmarkReport:
    active_runtime = runtime or _build_eta_open_weight_runtime(runtime_config)
    return run_eta_internal_rl_proof_benchmark(
        cases=cases,
        profile_labels=profile_labels,
        baseline_label=baseline_label,
        backend_label="transformers-open-weight",
        train_epochs=train_epochs,
        open_weight_runtime=active_runtime,
        open_weight_config=runtime_config,
        use_real_substrate_steps=True,
    )


def run_eta_internal_rl_backend_robustness_benchmark(
    *,
    cases: tuple[ETAProofCase, ...] = default_eta_proof_cases(),
    profile_label: str = "full-internal-rl",
    backend_labels: tuple[str, ...] = ("trace", "synthetic-open-weight"),
    open_weight_runtime: OpenWeightResidualRuntime | None = None,
    open_weight_config: ETAOpenWeightRuntimeConfig | None = None,
) -> ETAProofBackendComparisonReport:
    profile_reports = tuple(
        run_eta_internal_rl_proof_benchmark(
            cases=cases,
            profile_labels=(profile_label,),
            baseline_label=profile_label,
            backend_label=backend_label,
            open_weight_runtime=open_weight_runtime,
            open_weight_config=open_weight_config,
        ).profile_reports[0]
        for backend_label in backend_labels
    )
    if not profile_reports:
        raise ValueError("ETA backend robustness requires at least one backend label.")
    reference_report = profile_reports[0]
    reference_metrics = dict(reference_report.metric_means)
    comparison_report = profile_reports[-1]
    comparison_metrics = dict(comparison_report.metric_means)
    return ETAProofBackendComparisonReport(
        profile_label=profile_label,
        profile_reports=profile_reports,
        metric_deltas=tuple(
            (metric_name, comparison_metrics.get(metric_name, 0.0) - reference_metrics.get(metric_name, 0.0))
            for metric_name, _ in reference_report.metric_means
        ),
        description=(
            f"Backend robustness comparison for {profile_label} across backends={backend_labels}."
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
    best_real_residual_control = _best_real_residual_policy_control(report_map=report_map)
    real_residual_policy_control_score = _real_residual_control_score(full_report)
    best_real_residual_control_score = (
        _real_residual_control_score(best_real_residual_control)
        if best_real_residual_control is not None
        else 0.0
    )
    real_residual_policy_gap = max(
        0.0,
        real_residual_policy_control_score - best_real_residual_control_score,
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
        ("real_residual_policy_control_score", real_residual_policy_control_score),
        ("real_residual_policy_gap_vs_control", real_residual_policy_gap),
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
        ("mean_reward_sparsity", full_metrics.get("mean_reward_sparsity", 0.0)),
        ("reward_shaping_leakage", full_metrics.get("reward_shaping_leakage", 0.0)),
        (
            "heldout_family_reuse_gap_vs_no_fast_prior",
            full_metrics.get("heldout_family_reuse_gap_vs_no_fast_prior", 0.0),
        ),
        (
            "heldout_credit_alignment_gap_vs_no_fast_prior",
            full_metrics.get("heldout_credit_alignment_gap_vs_no_fast_prior", 0.0),
        ),
        (
            "heldout_strong_success_gap_vs_no_fast_prior",
            full_metrics.get("heldout_strong_success_gap_vs_no_fast_prior", 0.0),
        ),
        ("mean_steps_per_abstract_action", full_metrics.get("mean_steps_per_abstract_action", 0.0)),
        ("persistence_window_success_rate", full_metrics.get("persistence_window_success_rate", 0.0)),
        ("premature_switch_rate", full_metrics.get("premature_switch_rate", 0.0)),
        ("intervention_application_count", full_metrics.get("intervention_application_count", 0.0)),
        ("episode_replacement_effect_delta", full_metrics.get("episode_replacement_effect_delta", 0.0)),
        ("residual_signal_quality", full_metrics.get("residual_signal_quality", 0.0)),
        ("credit_to_family_write_count", full_metrics.get("credit_to_family_write_count", 0.0)),
        ("long_horizon_payoff_coverage", full_metrics.get("long_horizon_payoff_coverage", 0.0)),
        ("family_competition_mean", full_metrics.get("family_competition_mean", 0.0)),
        ("slow_to_fast_init_benefit", full_metrics.get("slow_to_fast_init_benefit", 0.0)),
        ("family_reuse_after_reset", full_metrics.get("family_reuse_after_reset", 0.0)),
        ("heldout_gain_after_consolidation", full_metrics.get("heldout_gain_after_consolidation", 0.0)),
        ("heldout_family_miss_rate", full_metrics.get("heldout_family_miss_rate", 0.0)),
        ("heldout_credit_window_miss_rate", full_metrics.get("heldout_credit_window_miss_rate", 0.0)),
        ("heldout_terminal_credit_coverage", full_metrics.get("heldout_terminal_credit_coverage", 0.0)),
        ("real_open_weight_step_count", full_metrics.get("real_open_weight_step_count", 0.0)),
        ("real_open_weight_capture_rate", full_metrics.get("real_open_weight_capture_rate", 0.0)),
        ("real_open_weight_hook_coverage", full_metrics.get("real_open_weight_hook_coverage", 0.0)),
        ("real_open_weight_fallback_rate", full_metrics.get("real_open_weight_fallback_rate", 0.0)),
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
    open_weight_runtime: OpenWeightResidualRuntime | None = None,
    open_weight_config: ETAOpenWeightRuntimeConfig | None = None,
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
    backend_labels = tuple(
        backend_label
        for name, values in active_manifest.case_groups
        if name == "backend_labels"
        for backend_label in values
    ) or ("trace", "synthetic-open-weight")
    primary_backend_label = backend_labels[0]
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
            backend_label=primary_backend_label,
            train_epochs=train_epochs,
            open_weight_runtime=open_weight_runtime,
            open_weight_config=open_weight_config,
        )
        backend_report = run_eta_internal_rl_backend_robustness_benchmark(
            cases=ordered_cases,
            profile_label=active_manifest.baseline_label,
            backend_labels=backend_labels,
            open_weight_runtime=open_weight_runtime,
            open_weight_config=open_weight_config,
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
        real_residual_control_report = _best_real_residual_policy_control(report_map=report_map)
        real_residual_gap = (
            max(
                0.0,
                _real_residual_control_score(full_report)
                - _real_residual_control_score(real_residual_control_report),
            )
            if full_report is not None and real_residual_control_report is not None
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
                real_residual_gap
                if active_manifest.suite_kind == "eta-open-weight-residual-proof"
                else max(raw_gap, mechanism_gap),
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
            "backend_mode": "+".join(backend_labels),
            "primary_backend": primary_backend_label,
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
    real_residual_policy_values = tuple(
        _eta_run_metric_map(summary).get("real_residual_policy_control_score", 0.0)
        for summary in run_summaries
    )
    real_residual_control_values = tuple(
        _eta_run_metric_map(summary).get("real_residual_policy_control_score", 0.0)
        - _eta_run_metric_map(summary).get("real_residual_policy_gap_vs_control", 0.0)
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
        build_pairwise_metric_effect(
            metric_name="real_residual_policy_control_score",
            candidate_label="full-internal-rl",
            control_label="same-backend-real-residual-control",
            candidate_values=real_residual_policy_values,
            control_values=real_residual_control_values,
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
    real_residual_policy_effect = next(
        (
            effect
            for effect in aggregate_report.pairwise_effects
            if effect.metric_name == "real_residual_policy_control_score"
        ),
        None,
    )
    statistical_gate_passed = "statistical-batch-evidence" not in blocked_gate_ids
    summary_map = {
        summary.metric_name: summary
        for summary in aggregate_report.primary_metric_summaries + aggregate_report.secondary_metric_summaries
    }
    real_capture_rate = summary_map.get("real_open_weight_capture_rate")
    real_hook_coverage = summary_map.get("real_open_weight_hook_coverage")
    real_fallback_rate = summary_map.get("real_open_weight_fallback_rate")
    residual_signal_quality = summary_map.get("residual_signal_quality")
    episode_replacement_effect_delta = summary_map.get("episode_replacement_effect_delta")
    real_residual_policy_gap = summary_map.get("real_residual_policy_gap_vs_control")
    reward_sparsity = summary_map.get("mean_reward_sparsity")
    reward_shaping_leakage = summary_map.get("reward_shaping_leakage")
    family_reuse_gap = summary_map.get("heldout_family_reuse_gap_vs_no_fast_prior")
    credit_alignment_gap = summary_map.get("heldout_credit_alignment_gap_vs_no_fast_prior")
    slow_to_fast_benefit = summary_map.get("slow_to_fast_init_benefit")
    credit_to_family_write_count = summary_map.get("credit_to_family_write_count")
    long_horizon_payoff_coverage = summary_map.get("long_horizon_payoff_coverage")
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
    verdicts = [
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
        ClaimVerdict(
            claim_id="claim_eta_internal_rl_sparse_reward_advantage",
            status=_eta_claim_status(
                retain_checks=(
                    claim_internal_rl == "retain",
                    reward_sparsity is not None and reward_sparsity.mean >= 0.75,
                    reward_shaping_leakage is not None and reward_shaping_leakage.mean <= 0.25,
                ),
                weak_checks=(
                    claim_internal_rl in {"weak", "retain"},
                    reward_sparsity is not None and reward_sparsity.mean > 0.0,
                ),
            ),
            required_gate_ids=("sparse-reward-success", "abstract-action-reuse", "credit-alignment"),
            supporting_artifacts=("paper_suite_aggregate", "reference_benchmark_report", "reference_assessment"),
            evidence=(
                ("internal_rl_advantage_status", claim_internal_rl),
                ("mean_reward_sparsity", reward_sparsity.mean if reward_sparsity is not None else 0.0),
                ("reward_shaping_leakage", reward_shaping_leakage.mean if reward_shaping_leakage is not None else 1.0),
            ),
            summary="ETA sparse-reward internal-RL claim verdict.",
            description="Checks whether the internal-RL advantage is backed by sparse or delayed reward evidence rather than shaping leakage.",
        ),
        ClaimVerdict(
            claim_id="claim_eta_scaffold_free_temporal_abstraction",
            status=_eta_claim_status(
                retain_checks=(
                    family_reuse_gap is not None and family_reuse_gap.mean >= 0.0,
                    credit_alignment_gap is not None and credit_alignment_gap.mean >= 0.0,
                    statistical_gate_passed,
                ),
                weak_checks=(
                    family_reuse_gap is not None,
                    credit_alignment_gap is not None,
                ),
            ),
            required_gate_ids=("scaffold-ablation-retention", "statistical-batch-evidence"),
            supporting_artifacts=("paper_suite_aggregate", "reference_benchmark_report"),
            evidence=(
                ("heldout_family_reuse_gap_vs_no_fast_prior", family_reuse_gap.mean if family_reuse_gap is not None else 0.0),
                (
                    "heldout_credit_alignment_gap_vs_no_fast_prior",
                    credit_alignment_gap.mean if credit_alignment_gap is not None else 0.0,
                ),
                ("statistical_batch_evidence_passed", float(statistical_gate_passed)),
            ),
            summary="ETA scaffold-ablation temporal-abstraction claim verdict.",
            description="Checks whether temporal-abstraction evidence survives matched scaffold ablations instead of relying on a fast-prior shortcut.",
        ),
        ClaimVerdict(
            claim_id="claim_nl_slow_loop_improves_eta_fast_path",
            status=_eta_claim_status(
                retain_checks=(
                    slow_to_fast_benefit is not None and slow_to_fast_benefit.mean >= 0.0,
                    credit_to_family_write_count is not None and credit_to_family_write_count.mean > 0.0,
                    long_horizon_payoff_coverage is not None and long_horizon_payoff_coverage.mean > 0.0,
                ),
                weak_checks=(
                    credit_to_family_write_count is not None and credit_to_family_write_count.mean > 0.0,
                    long_horizon_payoff_coverage is not None and long_horizon_payoff_coverage.mean > 0.0,
                ),
            ),
            required_gate_ids=("nl-slow-shapes-fast", "credit-alignment"),
            supporting_artifacts=("paper_suite_aggregate", "reference_benchmark_report"),
            evidence=(
                ("slow_to_fast_init_benefit", slow_to_fast_benefit.mean if slow_to_fast_benefit is not None else 0.0),
                (
                    "credit_to_family_write_count",
                    credit_to_family_write_count.mean if credit_to_family_write_count is not None else 0.0,
                ),
                (
                    "long_horizon_payoff_coverage",
                    long_horizon_payoff_coverage.mean if long_horizon_payoff_coverage is not None else 0.0,
                ),
            ),
            summary="NL slow-loop support for ETA fast path claim verdict.",
            description="Checks whether delayed and long-horizon NL signals reach temporal families and fast-prior evidence.",
        ),
    ]
    if aggregate_report.manifest.suite_kind == "eta-open-weight-residual-proof":
        fallback_rate = real_fallback_rate.mean if real_fallback_rate is not None else 1.0
        if fallback_rate > 0.10:
            real_claim_status = "fail"
        else:
            real_claim_status = _eta_claim_status(
                retain_checks=(
                    real_capture_rate is not None and real_capture_rate.mean >= 1.0,
                    real_hook_coverage is not None and real_hook_coverage.mean > 0.0,
                    residual_signal_quality is not None and residual_signal_quality.mean > 0.0,
                    episode_replacement_effect_delta is not None and episode_replacement_effect_delta.mean > 0.0,
                    real_residual_policy_effect is not None and real_residual_policy_effect.ci_low > 0.0,
                ),
                weak_checks=(
                    real_capture_rate is not None and real_capture_rate.mean >= 1.0,
                    real_hook_coverage is not None and real_hook_coverage.mean > 0.0,
                    residual_signal_quality is not None and residual_signal_quality.mean > 0.0,
                    episode_replacement_effect_delta is not None and episode_replacement_effect_delta.mean > 0.0,
                    real_residual_policy_effect is not None and real_residual_policy_effect.mean_delta > 0.0,
                ),
            )
        verdicts.append(
            ClaimVerdict(
                claim_id="claim_eta_real_open_weight_residual_control",
                status=real_claim_status,
                required_gate_ids=("backend-robustness",),
                supporting_artifacts=("paper_suite_aggregate", "reference_benchmark_report"),
                evidence=(
                    ("real_open_weight_capture_rate", real_capture_rate.mean if real_capture_rate is not None else 0.0),
                    ("real_open_weight_hook_coverage", real_hook_coverage.mean if real_hook_coverage is not None else 0.0),
                    ("real_open_weight_fallback_rate", real_fallback_rate.mean if real_fallback_rate is not None else 0.0),
                    ("residual_signal_quality", residual_signal_quality.mean if residual_signal_quality is not None else 0.0),
                    (
                        "episode_replacement_effect_delta",
                        episode_replacement_effect_delta.mean if episode_replacement_effect_delta is not None else 0.0,
                    ),
                    (
                        "strong_success_gap_ci_low",
                        strongest_control_effect.ci_low if strongest_control_effect is not None else 0.0,
                    ),
                    (
                        "strong_success_gap_mean_delta",
                        strongest_control_effect.mean_delta if strongest_control_effect is not None else 0.0,
                    ),
                    (
                        "real_residual_policy_gap_vs_control",
                        real_residual_policy_gap.mean if real_residual_policy_gap is not None else 0.0,
                    ),
                    (
                        "real_residual_policy_gap_ci_low",
                        real_residual_policy_effect.ci_low if real_residual_policy_effect is not None else 0.0,
                    ),
                ),
                summary="ETA real open-weight residual-control claim verdict.",
                description=(
                    "Checks whether ETA evidence uses real open-weight residual capture/control rather than "
                    "only synthetic trace proof harness evidence."
                ),
            )
        )
    return tuple(verdicts)


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
