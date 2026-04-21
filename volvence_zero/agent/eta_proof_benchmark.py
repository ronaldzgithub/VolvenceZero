from __future__ import annotations

from dataclasses import dataclass

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
from volvence_zero.temporal import FullLearnedTemporalPolicy, LearnedLiteTemporalPolicy


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
    strong_success_score: float
    description: str


@dataclass(frozen=True)
class ETAProofProfileReport:
    profile_label: str
    backend_label: str
    episode_reports: tuple[ETAProofEpisodeReport, ...]
    metric_means: tuple[tuple[str, float], ...]
    training_update_count: int
    training_parameter_change_count: int
    training_parameter_change_rate: float
    description: str


@dataclass(frozen=True)
class ETAProofBenchmarkReport:
    profile_reports: tuple[ETAProofProfileReport, ...]
    baseline_label: str
    backend_label: str
    metric_deltas_from_baseline: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]
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
class ETAInternalRLAcceptanceConfig:
    required_gate_ids: tuple[str, ...] = (
        "sparse-reward-success",
        "abstract-action-reuse",
        "heldout-composition",
        "credit-alignment",
        "policy-update-evidence",
        "backend-robustness",
    )
    min_success_delta: float = 0.02
    min_terminal_success_rate: float = 0.0
    min_reuse_rate: float = 0.50
    min_credit_alignment: float = 0.50
    min_strong_success_rate: float = 0.30
    max_backend_success_gap: float = 0.30
    min_policy_update_rate: float = 0.50
    min_passed_gate_count: int = 6


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


def _sparsity_quality(value: float) -> float:
    return max(0.0, min(1.0, 1.0 - abs(value - 0.68) / 0.16))


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
        "full-no-optimize",
        "full-no-replacement",
        "learned-lite-causal",
        "noop-backend",
    )


def _profile_config(profile_label: str) -> ETAProofProfileConfig:
    if profile_label == "full-internal-rl":
        return ETAProofProfileConfig(
            profile_label=profile_label,
            replacement_mode="causal-binary",
            optimize_after_rollout=True,
            policy_kind="full",
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


def _build_sandbox(*, profile: ETAProofProfileConfig, backend_label: str) -> InternalRLSandbox:
    if profile.policy_kind == "full":
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
    if not rollout.delayed_credit_assignments:
        return 0.0
    aligned = 0.0
    for assignment in rollout.delayed_credit_assignments:
        start = max(0, assignment.start_step)
        end = min(len(rollout.transitions) - 1, assignment.end_step)
        if end < start:
            continue
        if assignment.reason == "terminal-success":
            matched = any(transition.proof_terminal_success for transition in rollout.transitions[start : end + 1])
        else:
            matched = any(
                transition.proof_subgoal_id == assignment.subgoal_id and transition.active_family_id not in {None, "unassigned"}
                for transition in rollout.transitions[start : end + 1]
            )
        if matched:
            aligned += 1.0
    return aligned / max(len(rollout.delayed_credit_assignments), 1)


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
    sparsity_quality = _sparsity_quality(switch_sparsity)
    backend_quality = min(1.0, _mean(tuple(transition.backend_fidelity for transition in rollout.transitions)) / 0.6)
    strong_success_score = (
        subgoal_completion_rate * 0.25
        + family_reuse_rate * 0.20
        + credit_alignment * 0.15
        + sparsity_quality * 0.25
        + backend_quality * 0.15
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
        ("mean_strong_success_rate", _mean(tuple(report.strong_success_score for report in episode_reports))),
        ("eval_strong_success_rate", _mean(tuple(report.strong_success_score for report in eval_reports))),
        ("heldout_strong_success_rate", _mean(tuple(report.strong_success_score for report in heldout_reports))),
        ("mean_backend_fidelity", _mean(tuple(report.backend_fidelity for report in episode_reports))),
    )


def run_eta_internal_rl_proof_benchmark(
    *,
    cases: tuple[ETAProofCase, ...] = default_eta_proof_cases(),
    profile_labels: tuple[str, ...] = default_eta_proof_profiles(),
    baseline_label: str = "full-internal-rl",
    backend_label: str = "trace",
    train_epochs: int = 2,
) -> ETAProofBenchmarkReport:
    profile_reports: list[ETAProofProfileReport] = []
    for profile_label in profile_labels:
        profile = _profile_config(profile_label)
        sandbox = _build_sandbox(profile=profile, backend_label=backend_label)
        train_cases = tuple(case for case in cases if case.split == "train")
        eval_cases = tuple(case for case in cases if case.split != "train")
        family_registry: dict[str, set[str]] = {}
        training_update_count = 0
        training_parameter_change_count = 0
        for epoch in range(train_epochs):
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
                if profile.optimize_after_rollout:
                    optimize_report = sandbox.optimize(train_rollout)
                    training_update_count += 1
                    if optimize_report.parameters_changed:
                        training_parameter_change_count += 1
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
        profile_reports.append(
            ETAProofProfileReport(
                profile_label=profile_label,
                backend_label=backend_label,
                episode_reports=tuple(episode_reports),
                metric_means=metric_means,
                training_update_count=training_update_count,
                training_parameter_change_count=training_parameter_change_count,
                training_parameter_change_rate=(
                    training_parameter_change_count / training_update_count
                    if training_update_count > 0
                    else 0.0
                ),
                description=(
                    f"{profile_label} produced {len(episode_reports)} ETA proof episode reports "
                    f"on backend={backend_label}."
                ),
            )
        )
    baseline_report = next(report for report in profile_reports if report.profile_label == baseline_label)
    baseline_metrics = dict(baseline_report.metric_means)
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
    return ETAProofBenchmarkReport(
        profile_reports=tuple(profile_reports),
        baseline_label=baseline_label,
        backend_label=backend_label,
        metric_deltas_from_baseline=tuple(deltas),
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
    success_control_labels = ("full-no-optimize", "learned-lite-causal")
    reuse_control_labels = ("full-no-optimize", "learned-lite-causal")
    success_control_reports = tuple(
        report_map[label]
        for label in success_control_labels
        if label in report_map
    )
    reuse_control_reports = tuple(
        report_map[label]
        for label in reuse_control_labels
        if label in report_map
    )
    full_metrics = dict(full_report.metric_means)
    best_control_terminal_success = max(
        (dict(report.metric_means).get("heldout_terminal_success_rate", 0.0) for report in success_control_reports),
        default=0.0,
    )
    best_control_strong_success = max(
        (dict(report.metric_means).get("heldout_strong_success_rate", 0.0) for report in success_control_reports),
        default=0.0,
    )
    best_control_reuse = max(
        (dict(report.metric_means).get("heldout_family_reuse_rate", 0.0) for report in reuse_control_reports),
        default=0.0,
    )
    best_control_credit_alignment = max(
        (dict(report.metric_means).get("heldout_credit_alignment", 0.0) for report in reuse_control_reports),
        default=0.0,
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
                and full_metrics.get("heldout_strong_success_rate", 0.0) > best_control_strong_success
            ),
            evidence=(
                ("heldout_terminal_success_rate", full_metrics.get("heldout_terminal_success_rate", 0.0)),
                ("best_control_terminal_success", best_control_terminal_success),
                ("heldout_strong_success_rate", full_metrics.get("heldout_strong_success_rate", 0.0)),
                ("best_control_strong_success", best_control_strong_success),
                (
                    "success_delta",
                    full_metrics.get("heldout_strong_success_rate", 0.0) - best_control_strong_success,
                ),
            ),
            description="Full internal RL should beat matched controls on held-out terminal success and strong sparse-reward success.",
        ),
        ETAInternalRLAcceptanceGate(
            gate_id="abstract-action-reuse",
            passed=(
                full_metrics.get("heldout_family_reuse_rate", 0.0) >= 0.50
                and full_metrics.get("heldout_credit_alignment", 0.0) >= 0.50
                and full_metrics.get("heldout_family_reuse_rate", 0.0) >= best_control_reuse
            ),
            evidence=(
                ("heldout_family_reuse_rate", full_metrics.get("heldout_family_reuse_rate", 0.0)),
                ("best_control_reuse", best_control_reuse),
                ("heldout_credit_alignment", full_metrics.get("heldout_credit_alignment", 0.0)),
                ("best_control_credit_alignment", best_control_credit_alignment),
            ),
            description="Discovered abstract-action families should be reused across cases, not recreated every time.",
        ),
        ETAInternalRLAcceptanceGate(
            gate_id="heldout-composition",
            passed=(
                full_metrics.get("heldout_strong_success_rate", 0.0) >= 0.30
                and full_metrics.get("heldout_subgoal_completion_rate", 0.0) >= 0.25
            ),
            evidence=(
                ("heldout_strong_success_rate", full_metrics.get("heldout_strong_success_rate", 0.0)),
                ("heldout_subgoal_completion_rate", full_metrics.get("heldout_subgoal_completion_rate", 0.0)),
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
            ),
            evidence=(
                ("training_update_count", float(full_report.training_update_count)),
                ("training_parameter_change_count", float(full_report.training_parameter_change_count)),
                ("training_parameter_change_rate", full_report.training_parameter_change_rate),
            ),
            description="Internal RL should show concrete parameter-update evidence during training, not only better outcomes.",
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
    control_reports = tuple(
        report
        for report in assessment.benchmark_report.profile_reports
        if report.profile_label in {"full-no-optimize", "learned-lite-causal"}
    )
    best_control_success = max(
        (dict(report.metric_means).get("heldout_strong_success_rate", 0.0) for report in control_reports),
        default=0.0,
    )
    if full_metrics.get("heldout_terminal_success_rate", 0.0) < active_config.min_terminal_success_rate:
        reasons.append("heldout-success-below-threshold")
    if (
        full_metrics.get("heldout_strong_success_rate", 0.0) - best_control_success
        < active_config.min_success_delta
    ):
        reasons.append("success-delta-below-threshold")
    if full_metrics.get("heldout_strong_success_rate", 0.0) < active_config.min_strong_success_rate:
        reasons.append("heldout-strong-success-below-threshold")
    if full_metrics.get("heldout_family_reuse_rate", 0.0) < active_config.min_reuse_rate:
        reasons.append("family-reuse-below-threshold")
    if full_metrics.get("heldout_credit_alignment", 0.0) < active_config.min_credit_alignment:
        reasons.append("credit-alignment-below-threshold")
    if full_profile_report.training_parameter_change_rate < active_config.min_policy_update_rate:
        reasons.append("policy-update-rate-below-threshold")
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
