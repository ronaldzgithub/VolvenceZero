"""Attribution-only diagnostics for the sealed ETA Stage-3 negative result.

This evidence lane reuses the exact Stage-3 observation, residual capture,
fixed folding, scorer, and Store SSL objective. It may explain the sealed
``kill-eta`` verdict, but it cannot re-adjudicate it or change production
wiring.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
import math
import statistics
import time
from typing import Any, Protocol

from volvence_zero.agent.eta_proof_benchmark import (
    ETAOpenWeightRuntimeConfig,
    ETAProofCase,
    ETAProofCorpus,
    _build_eta_open_weight_runtime,
    _validate_eta_open_weight_runtime,
)
from volvence_zero.agent.eta_rate_distortion_evidence import (
    GATE_MODE_HARD_ST,
    OBSERVATION_PROTOCOL_V4,
    _action_options,
    _baseline_distortion,
    _build_traces,
    _rate_distortion_observation_texts,
)
from volvence_zero.substrate import (
    OpenWeightResidualRuntime,
    TrainingTrace,
    fit_linear_classification_probe,
)
from volvence_zero.temporal import MetacontrollerParameterStore
from volvence_zero.temporal.metacontroller_components import (
    POSTERIOR_PARAMETERIZATION_SMOOTH,
    RATE_GATING_SWITCH,
)
from volvence_zero.temporal.torch_store_ssl import (
    STEERED_CONTROL_CYCLIC_PERMUTED_Z,
    STEERED_CONTROL_ZERO_Z,
    STEERED_TRAINING_BIAS_ONLY,
    STEERED_TRAINING_FULL,
    StoreSSLEvaluationReport,
    StoreSSLTrainingSession,
    fold_trace_inputs_for_metacontroller,
)

ETA_STAGE3_EQUIVALENCE_SCHEMA_VERSION = "eta-stage3-equivalence-diagnostic.v1"


@dataclass(frozen=True)
class Stage3ExactEntryProbeReport:
    layer_indices: tuple[int, ...]
    activation_width: int
    folded_width: int
    accuracy: float
    chance_accuracy: float
    majority_accuracy: float
    support: int
    reference_gate2_accuracy: float
    accuracy_retention_from_gate2: float
    readable_at_two_x_chance: bool
    description: str = ""


@dataclass(frozen=True)
class Stage3SteeringControlPoint:
    seed: int
    training_mode: str
    optimizer_steps: int
    baseline_train_distortion: float
    baseline_heldout_distortion: float
    train_distortion: float
    heldout_distortion: float
    heldout_zero_z_distortion: float | None
    heldout_permuted_z_distortion: float | None
    action_boundary_f1: float
    oracle_subgoal_boundary_f1: float
    oracle_boundary_switch_probability: float
    oracle_continuation_switch_probability: float
    mean_switch_probability: float
    hard_switch_frequency: float
    final_total_loss: float
    final_grad_norm: float
    wall_seconds: float
    description: str = ""


@dataclass(frozen=True)
class Stage3EquivalenceThresholds:
    entry_chance_multiple: float = 2.0
    bias_recovery_min: float = 0.80
    zero_z_recovery_min: float = 0.80
    permuted_z_penalty_min: float = 0.02
    boundary_f1_semantic_delta_min: float = 0.10


@dataclass(frozen=True)
class Stage3EquivalenceAttribution:
    exact_entry_readable: bool
    mean_bias_only_recovery: float
    mean_zero_z_recovery: float
    mean_permuted_z_penalty: float
    mean_oracle_minus_action_boundary_f1: float
    free_bias_bypass_open: bool
    learned_z_causal: bool
    boundary_semantics_materially_different: bool
    dominant_attribution: str
    description: str = ""


@dataclass(frozen=True)
class Stage3EquivalenceDiagnosticReport:
    schema_version: str
    source_stage3_verdict: str
    claim_scope: str
    model_id: str
    model_source: str
    device: str
    layer_indices: tuple[int, ...]
    activation_width: int
    n_z: int
    alpha: float
    seed_schedule: tuple[int, ...]
    updates_per_run: int
    learning_rate: float
    switch_threshold: float
    observation_protocol: str
    posterior_parameterization: str
    rate_gating: str
    gate_mode: str
    corpus_seed: int
    train_route_count: int
    heldout_route_count: int
    train_step_count: int
    heldout_step_count: int
    injection_layer_index: int
    control_norm_cap: float
    probe_hidden_norm: float
    thresholds: Stage3EquivalenceThresholds
    exact_entry_probe: Stage3ExactEntryProbeReport
    control_points: tuple[Stage3SteeringControlPoint, ...]
    attribution: Stage3EquivalenceAttribution
    description: str = ""


class Stage3EquivalencePointCache(Protocol):
    def load_point(
        self, *, training_mode: str, seed: int
    ) -> Stage3SteeringControlPoint | None: ...

    def store_point(self, point: Stage3SteeringControlPoint) -> None: ...


def subgoal_transition_labels(
    subgoals: tuple[str | None, ...],
) -> tuple[float, ...]:
    """Oracle boundary labels; kept outside every training loss."""

    if len(subgoals) < 2:
        raise ValueError("Subgoal boundary readout requires at least two steps.")
    return tuple(
        float(current != previous)
        for previous, current in zip(
            subgoals[:-1], subgoals[1:], strict=True
        )
    )


def _subgoal_sequences(
    *,
    cases: tuple[ETAProofCase, ...],
    traces: tuple[TrainingTrace, ...],
    environment: Any,
    protocol_version: str,
) -> dict[str, tuple[str | None, ...]]:
    if len(cases) != len(traces):
        raise ValueError("Cases and traces must align one-to-one.")
    sequences: dict[str, tuple[str | None, ...]] = {}
    for case, trace in zip(cases, traces, strict=True):
        _texts, _targets, subgoals = _rate_distortion_observation_texts(
            case,
            environment=environment,
            protocol_version=protocol_version,
        )
        if len(subgoals) != len(trace.steps):
            raise RuntimeError(
                f"Subgoal sequence for {case.case_id!r} does not align with "
                "the captured Stage-3 trace."
            )
        sequences[trace.trace_id] = subgoals
    return sequences


def oracle_boundary_labels(
    subgoal_sequences: dict[str, tuple[str | None, ...]],
) -> dict[str, tuple[float, ...]]:
    return {
        trace_id: subgoal_transition_labels(subgoals)
        for trace_id, subgoals in subgoal_sequences.items()
    }


def fit_exact_stage3_entry_probe(
    *,
    torch_module: Any,
    train_traces: tuple[TrainingTrace, ...],
    heldout_traces: tuple[TrainingTrace, ...],
    train_subgoals: dict[str, tuple[str | None, ...]],
    heldout_subgoals: dict[str, tuple[str | None, ...]],
    objective_ids: tuple[str, ...],
    layer_indices: tuple[int, ...],
    activation_width: int,
    folded_width: int,
    reference_gate2_accuracy: float,
    chance_multiple: float = 2.0,
) -> Stage3ExactEntryProbeReport:
    """Closed-form probe over the exact per-step folded Store SSL input."""

    if len(objective_ids) < 2:
        raise ValueError("Exact-entry probe needs at least two objectives.")
    if not 0.0 < reference_gate2_accuracy <= 1.0:
        raise ValueError("reference_gate2_accuracy must be in (0, 1].")
    if chance_multiple <= 0.0:
        raise ValueError("chance_multiple must be positive.")
    label_index = {name: index for index, name in enumerate(objective_ids)}

    def rows(
        traces: tuple[TrainingTrace, ...],
        sequences: dict[str, tuple[str | None, ...]],
    ) -> tuple[list[tuple[float, ...]], list[int]]:
        features: list[tuple[float, ...]] = []
        labels: list[int] = []
        if set(sequences) != {trace.trace_id for trace in traces}:
            raise ValueError(
                "Subgoal sequences must provide exactly one row per trace."
            )
        for trace in traces:
            folded = fold_trace_inputs_for_metacontroller(
                trace=trace,
                n_input=folded_width,
            )
            subgoals = sequences[trace.trace_id]
            if len(folded) != len(subgoals):
                raise ValueError(
                    f"Folded inputs and subgoals disagree for {trace.trace_id!r}."
                )
            for vector, subgoal in zip(folded, subgoals, strict=True):
                if subgoal is None:
                    continue
                if subgoal not in label_index:
                    raise ValueError(f"Unknown subgoal label {subgoal!r}.")
                features.append(vector)
                labels.append(label_index[subgoal])
        return features, labels

    train_features, train_labels = rows(train_traces, train_subgoals)
    heldout_features, heldout_labels = rows(heldout_traces, heldout_subgoals)
    fit = fit_linear_classification_probe(
        torch_module=torch_module,
        train_features=torch_module.tensor(train_features, dtype=torch_module.float32),
        train_labels=torch_module.tensor(train_labels, dtype=torch_module.long),
        eval_features=torch_module.tensor(heldout_features, dtype=torch_module.float32),
        eval_labels=torch_module.tensor(heldout_labels, dtype=torch_module.long),
        layer_index=-1,
        class_count=len(objective_ids),
        alpha=1.0,
    )
    readable = fit.accuracy >= chance_multiple * fit.chance_accuracy
    return Stage3ExactEntryProbeReport(
        layer_indices=layer_indices,
        activation_width=activation_width,
        folded_width=folded_width,
        accuracy=fit.accuracy,
        chance_accuracy=fit.chance_accuracy,
        majority_accuracy=fit.majority_accuracy,
        support=fit.support,
        reference_gate2_accuracy=reference_gate2_accuracy,
        accuracy_retention_from_gate2=(
            fit.accuracy / reference_gate2_accuracy
        ),
        readable_at_two_x_chance=readable,
        description=(
            "Linear active-subgoal probe on the exact Stage-3 single-step "
            f"{folded_width}-dim folded entry: accuracy={fit.accuracy:.4f}, "
            f"chance={fit.chance_accuracy:.4f}."
        ),
    )


def _run_control_point(
    *,
    training_mode: str,
    seed: int,
    n_z: int,
    alpha: float,
    scorer: Any,
    train_traces: tuple[TrainingTrace, ...],
    heldout_traces: tuple[TrainingTrace, ...],
    heldout_oracle_labels: dict[str, tuple[float, ...]],
    updates_per_run: int,
    learning_rate: float,
    switch_threshold: float,
    progress: Callable[[str], None] | None,
) -> Stage3SteeringControlPoint:
    start = time.perf_counter()
    store = MetacontrollerParameterStore(n_z=n_z, initialization_seed=seed)
    session = StoreSSLTrainingSession(
        n_z=n_z,
        alpha=alpha,
        learning_rate=learning_rate,
        switch_rate_weight=0.0,
        switch_binary_weight=0.0,
        switch_group_weight=0.0,
        proposal_prediction_weight=0.0,
        gate_choice_weight=0.0,
        action_scorer=scorer,
        reparam_seed=seed * 1_000_003 + 17,
        posterior_parameterization=POSTERIOR_PARAMETERIZATION_SMOOTH,
        rate_gating=RATE_GATING_SWITCH,
        steered_gate_mode=GATE_MODE_HARD_ST,
        steered_training_mode=training_mode,
    )
    final_report = None
    for update_index in range(updates_per_run):
        final_report = session.train_batch(
            store=store,
            traces=train_traces,
            batch_id=(
                f"stage3-equivalence:{training_mode}:s{seed}:u{update_index}"
            ),
            switch_threshold=switch_threshold,
            write_back=False,
        )
        if not math.isfinite(final_report.total_loss):
            raise RuntimeError(
                f"Non-finite diagnostic loss mode={training_mode} seed={seed} "
                f"update={update_index}."
            )
        completed = update_index + 1
        if progress is not None and (
            completed == 1
            or completed == updates_per_run
            or completed % 5 == 0
        ):
            progress(
                f"mode={training_mode} seed={seed} "
                f"update={completed}/{updates_per_run} "
                f"loss={final_report.total_loss:.4f}"
            )
    if final_report is None:
        raise RuntimeError("updates_per_run must be at least 1.")

    baseline_train = _baseline_distortion(train_traces, scorer)
    baseline_heldout = _baseline_distortion(heldout_traces, scorer)
    train_eval = session.evaluate_batch(
        store=store,
        traces=train_traces,
        batch_id=f"stage3-equivalence:{training_mode}:s{seed}:train",
        switch_threshold=switch_threshold,
    )
    heldout_eval = session.evaluate_batch(
        store=store,
        traces=heldout_traces,
        batch_id=f"stage3-equivalence:{training_mode}:s{seed}:heldout",
        switch_threshold=switch_threshold,
    )
    oracle_eval = session.evaluate_batch(
        store=store,
        traces=heldout_traces,
        batch_id=f"stage3-equivalence:{training_mode}:s{seed}:oracle",
        switch_threshold=switch_threshold,
        boundary_labels=heldout_oracle_labels,
    )
    zero_eval: StoreSSLEvaluationReport | None = None
    permuted_eval: StoreSSLEvaluationReport | None = None
    if training_mode == STEERED_TRAINING_FULL:
        zero_eval = session.evaluate_batch(
            store=store,
            traces=heldout_traces,
            batch_id=f"stage3-equivalence:{training_mode}:s{seed}:zero-z",
            switch_threshold=switch_threshold,
            control_ablation=STEERED_CONTROL_ZERO_Z,
        )
        permuted_eval = session.evaluate_batch(
            store=store,
            traces=heldout_traces,
            batch_id=(
                f"stage3-equivalence:{training_mode}:s{seed}:permuted-z"
            ),
            switch_threshold=switch_threshold,
            control_ablation=STEERED_CONTROL_CYCLIC_PERMUTED_Z,
        )
    return Stage3SteeringControlPoint(
        seed=seed,
        training_mode=training_mode,
        optimizer_steps=session.optimizer_step,
        baseline_train_distortion=baseline_train,
        baseline_heldout_distortion=baseline_heldout,
        train_distortion=train_eval.distortion,
        heldout_distortion=heldout_eval.distortion,
        heldout_zero_z_distortion=(
            zero_eval.distortion if zero_eval is not None else None
        ),
        heldout_permuted_z_distortion=(
            permuted_eval.distortion if permuted_eval is not None else None
        ),
        action_boundary_f1=heldout_eval.boundary_f1,
        oracle_subgoal_boundary_f1=oracle_eval.boundary_f1,
        oracle_boundary_switch_probability=(
            oracle_eval.boundary_switch_probability
        ),
        oracle_continuation_switch_probability=(
            oracle_eval.continuation_switch_probability
        ),
        mean_switch_probability=heldout_eval.mean_switch_probability,
        hard_switch_frequency=heldout_eval.hard_switch_frequency,
        final_total_loss=final_report.total_loss,
        final_grad_norm=final_report.grad_norm,
        wall_seconds=time.perf_counter() - start,
        description=(
            f"Stage-3 equivalence mode={training_mode} seed={seed}: "
            f"heldout distortion={heldout_eval.distortion:.4f}."
        ),
    )


def _recovery_fraction(
    *, baseline: float, candidate: float, full: float
) -> float:
    full_improvement = baseline - full
    if full_improvement <= 1e-8:
        return 0.0
    return (baseline - candidate) / full_improvement


def assess_stage3_equivalence(
    *,
    exact_entry_probe: Stage3ExactEntryProbeReport,
    control_points: tuple[Stage3SteeringControlPoint, ...],
    thresholds: Stage3EquivalenceThresholds,
) -> Stage3EquivalenceAttribution:
    full_by_seed = {
        point.seed: point
        for point in control_points
        if point.training_mode == STEERED_TRAINING_FULL
    }
    bias_by_seed = {
        point.seed: point
        for point in control_points
        if point.training_mode == STEERED_TRAINING_BIAS_ONLY
    }
    if not full_by_seed or set(full_by_seed) != set(bias_by_seed):
        raise ValueError(
            "Attribution requires matched full and bias-only points per seed."
        )
    bias_recoveries: list[float] = []
    zero_recoveries: list[float] = []
    permuted_penalties: list[float] = []
    boundary_deltas: list[float] = []
    for seed, full in sorted(full_by_seed.items()):
        bias = bias_by_seed[seed]
        if full.heldout_zero_z_distortion is None:
            raise ValueError("Full control point lacks zero-z distortion.")
        if full.heldout_permuted_z_distortion is None:
            raise ValueError("Full control point lacks permuted-z distortion.")
        bias_recoveries.append(
            _recovery_fraction(
                baseline=full.baseline_heldout_distortion,
                candidate=bias.heldout_distortion,
                full=full.heldout_distortion,
            )
        )
        zero_recoveries.append(
            _recovery_fraction(
                baseline=full.baseline_heldout_distortion,
                candidate=full.heldout_zero_z_distortion,
                full=full.heldout_distortion,
            )
        )
        permuted_penalties.append(
            full.heldout_permuted_z_distortion - full.heldout_distortion
        )
        boundary_deltas.append(
            full.oracle_subgoal_boundary_f1 - full.action_boundary_f1
        )
    mean_bias = statistics.fmean(bias_recoveries)
    mean_zero = statistics.fmean(zero_recoveries)
    mean_permuted = statistics.fmean(permuted_penalties)
    mean_boundary_delta = statistics.fmean(boundary_deltas)
    free_bias_bypass = (
        mean_bias >= thresholds.bias_recovery_min
        or mean_zero >= thresholds.zero_z_recovery_min
    )
    learned_z_causal = (
        mean_zero < thresholds.zero_z_recovery_min
        and mean_permuted >= thresholds.permuted_z_penalty_min
    )
    boundary_mismatch = (
        abs(mean_boundary_delta)
        >= thresholds.boundary_f1_semantic_delta_min
    )
    if not exact_entry_probe.readable_at_two_x_chance:
        dominant = "information-dead-at-stage3-entry"
    elif free_bias_bypass:
        dominant = "incentive-bypass-via-free-bias"
    elif not learned_z_causal:
        dominant = "optimization-or-latent-noncausality"
    else:
        dominant = "mechanism-shape-mismatch"
    return Stage3EquivalenceAttribution(
        exact_entry_readable=exact_entry_probe.readable_at_two_x_chance,
        mean_bias_only_recovery=mean_bias,
        mean_zero_z_recovery=mean_zero,
        mean_permuted_z_penalty=mean_permuted,
        mean_oracle_minus_action_boundary_f1=mean_boundary_delta,
        free_bias_bypass_open=free_bias_bypass,
        learned_z_causal=learned_z_causal,
        boundary_semantics_materially_different=boundary_mismatch,
        dominant_attribution=dominant,
        description=(
            f"Attribution={dominant}; entry readable="
            f"{exact_entry_probe.readable_at_two_x_chance}, bias recovery="
            f"{mean_bias:.3f}, zero-z recovery={mean_zero:.3f}, permuted-z "
            f"penalty={mean_permuted:.3f}."
        ),
    )


def run_eta_stage3_equivalence_diagnostic(
    *,
    corpus: ETAProofCorpus,
    open_weight_config: ETAOpenWeightRuntimeConfig,
    source_stage3_verdict: str = "kill-eta",
    seed_schedule: tuple[int, ...] = (0, 1, 2),
    n_z: int = 16,
    alpha: float = 0.1,
    updates_per_run: int = 100,
    learning_rate: float = 0.02,
    switch_threshold: float = 0.55,
    reference_gate2_accuracy: float = 0.944,
    thresholds: Stage3EquivalenceThresholds | None = None,
    runtime: OpenWeightResidualRuntime | None = None,
    point_cache: Stage3EquivalencePointCache | None = None,
    progress: Callable[[str], None] | None = None,
) -> Stage3EquivalenceDiagnosticReport:
    if source_stage3_verdict != "kill-eta":
        raise ValueError(
            "P1 attribution is authorized only for the sealed kill-eta result."
        )
    if not seed_schedule or len(set(seed_schedule)) != len(seed_schedule):
        raise ValueError("seed_schedule must be non-empty and unique.")
    if n_z <= 3 or updates_per_run < 1:
        raise ValueError("n_z must exceed 3 and updates_per_run must be positive.")
    active_thresholds = thresholds or Stage3EquivalenceThresholds()
    config = open_weight_config
    if config.model_dtype != "float32":
        config = replace(config, model_dtype="float32")
    if config.layer_indices is None:
        raise ValueError("P1 exact-entry probe requires explicit layer indices.")
    if runtime is None:
        runtime = _build_eta_open_weight_runtime(config)
        _validate_eta_open_weight_runtime(runtime=runtime, config=config)

    environment = corpus.environment
    train_traces = _build_traces(
        corpus.train_cases,
        environment=environment,
        runtime=runtime,
        label="p1-train",
        protocol_version=OBSERVATION_PROTOCOL_V4,
    )
    heldout_traces = _build_traces(
        corpus.heldout_cases,
        environment=environment,
        runtime=runtime,
        label="p1-heldout",
        protocol_version=OBSERVATION_PROTOCOL_V4,
    )
    train_subgoals = _subgoal_sequences(
        cases=corpus.train_cases,
        traces=train_traces,
        environment=environment,
        protocol_version=OBSERVATION_PROTOCOL_V4,
    )
    heldout_subgoals = _subgoal_sequences(
        cases=corpus.heldout_cases,
        traces=heldout_traces,
        environment=environment,
        protocol_version=OBSERVATION_PROTOCOL_V4,
    )
    objective_ids = tuple(
        location.location_id for location in environment.objective_locations()
    )

    import torch

    exact_entry_probe = fit_exact_stage3_entry_probe(
        torch_module=torch,
        train_traces=train_traces,
        heldout_traces=heldout_traces,
        train_subgoals=train_subgoals,
        heldout_subgoals=heldout_subgoals,
        objective_ids=objective_ids,
        layer_indices=config.layer_indices,
        activation_width=config.activation_width,
        folded_width=n_z,
        reference_gate2_accuracy=reference_gate2_accuracy,
        chance_multiple=active_thresholds.entry_chance_multiple,
    )
    heldout_oracle = oracle_boundary_labels(heldout_subgoals)
    scorer = runtime.build_steered_action_scorer(
        action_options=_action_options(environment),
        joint_training=False,
        prefix_cache=True,
    )
    points: list[Stage3SteeringControlPoint] = []
    for seed in seed_schedule:
        for training_mode in (
            STEERED_TRAINING_FULL,
            STEERED_TRAINING_BIAS_ONLY,
        ):
            cached = (
                point_cache.load_point(
                    training_mode=training_mode,
                    seed=seed,
                )
                if point_cache is not None
                else None
            )
            if cached is not None:
                points.append(cached)
                if progress is not None:
                    progress(f"resumed mode={training_mode} seed={seed}")
                continue
            point = _run_control_point(
                training_mode=training_mode,
                seed=seed,
                n_z=n_z,
                alpha=alpha,
                scorer=scorer,
                train_traces=train_traces,
                heldout_traces=heldout_traces,
                heldout_oracle_labels=heldout_oracle,
                updates_per_run=updates_per_run,
                learning_rate=learning_rate,
                switch_threshold=switch_threshold,
                progress=progress,
            )
            if point_cache is not None:
                point_cache.store_point(point)
            points.append(point)
    ordered_points = tuple(
        sorted(points, key=lambda point: (point.seed, point.training_mode))
    )
    attribution = assess_stage3_equivalence(
        exact_entry_probe=exact_entry_probe,
        control_points=ordered_points,
        thresholds=active_thresholds,
    )
    return Stage3EquivalenceDiagnosticReport(
        schema_version=ETA_STAGE3_EQUIVALENCE_SCHEMA_VERSION,
        source_stage3_verdict=source_stage3_verdict,
        claim_scope="stage3-attribution-only-no-readjudication",
        model_id=config.model_id,
        model_source=config.model_source or config.model_id,
        device=config.device,
        layer_indices=config.layer_indices,
        activation_width=config.activation_width,
        n_z=n_z,
        alpha=alpha,
        seed_schedule=seed_schedule,
        updates_per_run=updates_per_run,
        learning_rate=learning_rate,
        switch_threshold=switch_threshold,
        observation_protocol=OBSERVATION_PROTOCOL_V4,
        posterior_parameterization=POSTERIOR_PARAMETERIZATION_SMOOTH,
        rate_gating=RATE_GATING_SWITCH,
        gate_mode=GATE_MODE_HARD_ST,
        corpus_seed=corpus.seed,
        train_route_count=corpus.train_route_count,
        heldout_route_count=corpus.heldout_route_count,
        train_step_count=sum(len(trace.steps) for trace in train_traces),
        heldout_step_count=sum(len(trace.steps) for trace in heldout_traces),
        injection_layer_index=scorer.injection_layer_index,
        control_norm_cap=scorer.control_norm_cap,
        probe_hidden_norm=scorer.probe_hidden_norm,
        thresholds=active_thresholds,
        exact_entry_probe=exact_entry_probe,
        control_points=ordered_points,
        attribution=attribution,
        description=(
            "Attribution-only P1 diagnostic for the sealed Stage-3 kill-eta "
            f"artifact; dominant attribution={attribution.dominant_attribution}."
        ),
    )
