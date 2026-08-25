"""C3 real-dialogue transfer evidence for the bounded steering gate.

The expensive SHADOW collector freezes matched steer/noop N+1 PE settlements.
This module then replays only owner-published gate observations and routes every
terminal scalar through PE -> Credit -> SteeringGateModule.  It never sees raw
dialogue text, evaluation scores, judge output, or per-step oracle labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import statistics

from volvence_zero.credit import CreditModule
from volvence_zero.canonical_json import (
    canonical_json_bytes,
    strict_json_loads,
    typed_from_json,
    typed_to_json,
)
from volvence_zero.prediction import (
    bind_steering_terminal_prediction_error_decisions,
)
from volvence_zero.runtime import WiringLevel
from volvence_zero.steering_contracts import (
    STEERING_GATE_ARTIFACT_SCHEMA_VERSION,
    SteeringGateArtifact,
    SteeringTerminalPredictionError,
)
from volvence_zero.steering_gate import SteeringGateModule


DIALOGUE_STEERING_EVIDENCE_SCHEMA_VERSION = "dialogue-steering-evidence.v1"


@dataclass(frozen=True)
class DialogueSteeringThresholds:
    min_real_trace_turns: int = 500
    action_sensitivity_abs_credit: float = 0.01
    min_action_sensitive_fraction: float = 0.10
    min_convergence_improvement: float = 0.005
    min_gain_vs_noop: float = 0.005
    min_gain_vs_always_on: float = 0.005
    min_gain_vs_random_gate: float = 0.005
    min_gate_selectivity: float = 0.10
    require_clustered_ci_lower_positive: bool = True


@dataclass(frozen=True)
class DialogueSteeringTraceRow:
    sample_id: str
    split: str
    episode_id: str
    cluster_id: str
    session_index: int
    observations: tuple[tuple[str, float], ...]
    terminal_prediction_error: SteeringTerminalPredictionError
    reader_artifact_id: str
    executor_artifact_id: str
    source_model_id: str
    source_model_weights_sha256: str
    shadow_hook_latency_ms: float
    end_to_end_latency_ms: float
    shadow_owner_chain_complete: bool
    shadow_hook_executed: bool
    free_bias_present: bool
    zero_code_strict_noop: bool
    raw_text_retained: bool
    evaluation_writeback_allowed: bool
    sensor_off_executor_artifact_id: str = ""
    control_norm: float = 0.0
    control_norm_cap: float = 0.0
    sensor_off_control_norm: float = 0.0
    sensor_off_mean_squared_error: float | None = None
    sensor_off_cosine_similarity: float | None = None

    def __post_init__(self) -> None:
        for value in (
            self.sample_id,
            self.split,
            self.episode_id,
            self.cluster_id,
            self.reader_artifact_id,
            self.executor_artifact_id,
            self.source_model_id,
        ):
            if not value.strip():
                raise ValueError("dialogue steering trace lineage must be non-empty")
        if self.session_index < 1:
            raise ValueError("dialogue steering session_index must be positive")
        names = tuple(name for name, _ in self.observations)
        if not names or len(set(names)) != len(names):
            raise ValueError("dialogue steering observations must be uniquely named")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for _, value in self.observations
        ):
            raise ValueError("dialogue steering observations must be within [0, 1]")
        settlement = self.terminal_prediction_error
        if settlement.episode_id != self.episode_id:
            raise ValueError("dialogue steering settlement episode drift")
        if settlement.sample_ids != (self.sample_id,):
            raise ValueError("dialogue steering settlement sample drift")
        if (
            settlement.target_model_id != self.source_model_id
            or settlement.target_model_weights_sha256
            != self.source_model_weights_sha256
        ):
            raise ValueError("dialogue steering context/target substrate drift")
        for value in (self.shadow_hook_latency_ms, self.end_to_end_latency_ms):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("dialogue steering latency must be non-negative")
        if self.end_to_end_latency_ms < self.shadow_hook_latency_ms:
            raise ValueError("dialogue steering hook exceeds end-to-end latency")
        if (self.sensor_off_mean_squared_error is None) != (
            self.sensor_off_cosine_similarity is None
        ):
            raise ValueError("dialogue steering sensor-off PE is partial")
        for value in (
            self.control_norm,
            self.control_norm_cap,
            self.sensor_off_control_norm,
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("dialogue steering control budget is invalid")
        if (
            self.control_norm > self.control_norm_cap + 1e-8
            or self.sensor_off_control_norm > self.control_norm_cap + 1e-8
        ):
            raise ValueError("dialogue steering control exceeds the shared cap")
        if self.sensor_off_mean_squared_error is not None:
            if (
                not self.sensor_off_executor_artifact_id.strip()
                or not math.isfinite(self.sensor_off_mean_squared_error)
                or self.sensor_off_mean_squared_error < 0.0
                or self.sensor_off_cosine_similarity is None
                or not math.isfinite(self.sensor_off_cosine_similarity)
                or not -1.0 <= self.sensor_off_cosine_similarity <= 1.0
            ):
                raise ValueError("dialogue steering sensor-off PE is invalid")
        elif self.sensor_off_executor_artifact_id or self.sensor_off_control_norm != 0.0:
            raise ValueError("dialogue steering sensor-off lineage is partial")

    @property
    def relative_steer_credit(self) -> float:
        return self.terminal_prediction_error.relative_mse_improvement

    @property
    def normalized_noop_loss(self) -> float:
        settlement = self.terminal_prediction_error
        denominator = max(
            settlement.action_mean_squared_error,
            settlement.noop_mean_squared_error,
            1e-12,
        )
        return settlement.noop_mean_squared_error / denominator

    @property
    def normalized_steer_loss(self) -> float:
        settlement = self.terminal_prediction_error
        denominator = max(
            settlement.action_mean_squared_error,
            settlement.noop_mean_squared_error,
            1e-12,
        )
        return settlement.action_mean_squared_error / denominator


@dataclass(frozen=True)
class DialogueSteeringTraceDataset:
    schema_version: str
    bundle_id: str
    prediction_head_fingerprint: str
    train_rows: tuple[DialogueSteeringTraceRow, ...]
    validation_rows: tuple[DialogueSteeringTraceRow, ...]
    raw_text_retained: bool
    evaluation_writeback_allowed: bool
    description: str

    def __post_init__(self) -> None:
        if self.schema_version != "dialogue-steering-trace-dataset.v1":
            raise ValueError("dialogue steering trace dataset schema is unsupported")
        if not self.bundle_id.strip() or not self.description.strip():
            raise ValueError("dialogue steering trace dataset lineage is incomplete")
        if (
            len(self.prediction_head_fingerprint) != 64
            or any(
                char not in "0123456789abcdef"
                for char in self.prediction_head_fingerprint
            )
        ):
            raise ValueError("dialogue steering head fingerprint is invalid")
        if not self.train_rows or not self.validation_rows:
            raise ValueError("dialogue steering trace dataset splits are empty")
        if self.raw_text_retained or self.evaluation_writeback_allowed:
            raise ValueError("dialogue steering trace dataset violates R12/privacy")
        all_rows = (*self.train_rows, *self.validation_rows)
        if any(row.split != "train" for row in self.train_rows) or any(
            row.split != "validation" for row in self.validation_rows
        ):
            raise ValueError("dialogue steering trace rows are in the wrong split")
        sample_ids = tuple(row.sample_id for row in all_rows)
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("dialogue steering trace sample ids must be unique")
        if set(row.cluster_id for row in self.train_rows) & set(
            row.cluster_id for row in self.validation_rows
        ):
            raise ValueError("dialogue steering train/validation clusters overlap")
        feature_names = tuple(name for name, _ in all_rows[0].observations)
        if any(
            tuple(name for name, _ in row.observations) != feature_names
            for row in all_rows
        ):
            raise ValueError("dialogue steering observation schema drift")
        if any(
            row.terminal_prediction_error.prediction_head_fingerprint
            != self.prediction_head_fingerprint
            for row in all_rows
        ):
            raise ValueError("dialogue steering prediction-head lineage drift")
        lineage = {
            (
                row.reader_artifact_id,
                row.executor_artifact_id,
                row.sensor_off_executor_artifact_id,
                row.source_model_id,
                row.source_model_weights_sha256,
            )
            for row in all_rows
        }
        if len(lineage) != 1:
            raise ValueError("dialogue steering artifact/substrate lineage drift")

    def to_json(self) -> str:
        return canonical_json_bytes(
            typed_to_json(self, DialogueSteeringTraceDataset)
        ).decode("utf-8")

    @classmethod
    def from_json(cls, payload: str) -> "DialogueSteeringTraceDataset":
        raw = strict_json_loads(
            payload.encode("utf-8"), max_bytes=512 * 1024 * 1024
        )
        decoded = typed_from_json(raw, DialogueSteeringTraceDataset)
        if not isinstance(decoded, DialogueSteeringTraceDataset):
            raise TypeError("decoded dialogue steering dataset has wrong type")
        return decoded


@dataclass(frozen=True)
class DialogueSteeringEffect:
    mean: float
    ci_lower: float
    ci_upper: float
    cluster_count: int
    row_count: int


@dataclass(frozen=True)
class DialogueSteeringArms:
    noop: float
    learned_gate: float
    always_on: float
    random_gate: float


@dataclass(frozen=True)
class DialogueSteeringSeedPoint:
    seed: int
    selected_restart: int
    selection_train_loss: float
    initial_train_loss: float
    final_train_loss: float
    convergence_improvement: float
    arms: DialogueSteeringArms
    steer_rate: float
    steer_rate_positive_credit: float
    steer_rate_nonpositive_credit: float
    gate_selectivity: float
    gain_vs_noop: DialogueSteeringEffect
    gain_vs_always_on: DialogueSteeringEffect
    gain_vs_random_gate: DialogueSteeringEffect
    policy_parameters_changed: bool
    selected_gate_artifact: SteeringGateArtifact


@dataclass(frozen=True)
class DialogueSteeringAggregate:
    seed_count: int
    noop_loss_mean: float
    learned_gate_loss_mean: float
    always_on_loss_mean: float
    random_gate_loss_mean: float
    convergence_improvement_mean: float
    convergence_improvement_worst_seed: float
    gate_selectivity_mean: float
    gate_selectivity_worst_seed: float
    gain_vs_noop_mean_worst_seed: float
    gain_vs_always_on_mean_worst_seed: float
    gain_vs_random_gate_mean_worst_seed: float
    gain_vs_noop_ci_lower_worst_seed: float
    gain_vs_always_on_ci_lower_worst_seed: float
    gain_vs_random_gate_ci_lower_worst_seed: float


@dataclass(frozen=True)
class DialogueSteeringAdmission:
    admitted: bool
    condition_real_trace: bool
    condition_action_sensitivity: bool
    condition_convergence: bool
    condition_gain_vs_noop: bool
    condition_gain_vs_always_on: bool
    condition_gain_vs_random_gate: bool
    condition_gate_selectivity: bool
    condition_structural_integrity: bool
    failed_conditions: tuple[str, ...]
    exit_reason: str
    description: str


@dataclass(frozen=True)
class DialogueSteeringReport:
    schema_version: str
    preregistration_sha256: str
    train_turn_count: int
    validation_turn_count: int
    train_cluster_count: int
    validation_cluster_count: int
    action_sensitive_fraction: float
    seed_schedule: tuple[int, ...]
    policy_restarts: int
    max_online_episodes: int
    eval_every: int
    learning_rate: float
    bootstrap_resamples: int
    bootstrap_confidence: float
    thresholds: DialogueSteeringThresholds
    seed_points: tuple[DialogueSteeringSeedPoint, ...]
    aggregate: DialogueSteeringAggregate
    admission: DialogueSteeringAdmission
    substrate_trainable_parameter_count: int
    reader_parameters_changed: bool
    executor_parameters_changed: bool
    policy_parameters_changed: bool
    free_bias_present: bool
    zero_code_strict_noop: bool
    raw_text_retained: bool
    evaluation_writeback_allowed: bool
    terminal_credit_source: str
    description: str

    def __post_init__(self) -> None:
        if self.schema_version != DIALOGUE_STEERING_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("dialogue steering report schema is unsupported")
        if (
            self.train_turn_count < 1
            or self.validation_turn_count < 1
            or self.train_cluster_count < 1
            or self.validation_cluster_count < 1
        ):
            raise ValueError("dialogue steering report counts must be positive")
        if (
            not self.seed_schedule
            or len(set(self.seed_schedule)) != len(self.seed_schedule)
            or tuple(point.seed for point in self.seed_points)
            != self.seed_schedule
        ):
            raise ValueError("dialogue steering report seed lineage drift")
        if not 0.0 <= self.action_sensitive_fraction <= 1.0:
            raise ValueError("dialogue steering action-sensitive fraction is invalid")
        if self.aggregate.seed_count != len(self.seed_points):
            raise ValueError("dialogue steering aggregate seed count drift")
        if self.substrate_trainable_parameter_count != 0:
            raise ValueError("dialogue steering report mutated the frozen substrate")
        if not self.terminal_credit_source.strip() or not self.description.strip():
            raise ValueError("dialogue steering report provenance is incomplete")

    def to_json(self) -> str:
        return canonical_json_bytes(
            typed_to_json(self, DialogueSteeringReport)
        ).decode("utf-8")

    @classmethod
    def from_json(cls, payload: str) -> "DialogueSteeringReport":
        raw = strict_json_loads(payload.encode("utf-8"), max_bytes=256 * 1024 * 1024)
        decoded = typed_from_json(raw, DialogueSteeringReport)
        if not isinstance(decoded, DialogueSteeringReport):
            raise TypeError("decoded dialogue steering report has wrong type")
        return decoded


def _artifact_for_restart(
    *,
    feature_names: tuple[str, ...],
    preregistration_sha256: str,
    seed: int,
    restart: int,
) -> SteeringGateArtifact:
    rng = random.Random(seed * 1_000_003 + restart * 101 + 17)
    weights = tuple(
        (rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05))
        for _ in feature_names
    )
    bias = (rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02))
    return SteeringGateArtifact(
        schema_version=STEERING_GATE_ARTIFACT_SCHEMA_VERSION,
        artifact_id=f"dialogue-gate:s{seed}:r{restart}:initial",
        source_preregistration_sha256=preregistration_sha256,
        feature_names=feature_names,
        weights=weights,
        bias=bias,
        policy_version=1,
        description="C3 preregistered stochastic gate initialization.",
    )


def _steer_for(
    artifact: SteeringGateArtifact,
    observations: tuple[tuple[str, float], ...],
) -> bool:
    if tuple(name for name, _ in observations) != artifact.feature_names:
        raise ValueError("dialogue steering replay feature schema drift")
    logits = tuple(
        artifact.bias[action]
        + sum(
            value * artifact.weights[index][action]
            for index, (_, value) in enumerate(observations)
        )
        for action in range(2)
    )
    return logits[1] > logits[0]


def _loss_rows(
    artifact: SteeringGateArtifact,
    rows: tuple[DialogueSteeringTraceRow, ...],
) -> tuple[tuple[float, ...], tuple[bool, ...]]:
    flags = tuple(_steer_for(artifact, row.observations) for row in rows)
    losses = tuple(
        row.normalized_steer_loss if flag else row.normalized_noop_loss
        for row, flag in zip(rows, flags, strict=True)
    )
    return losses, flags


def _clustered_effect(
    *,
    rows: tuple[DialogueSteeringTraceRow, ...],
    effects: tuple[float, ...],
    seed: int,
    resamples: int,
    confidence: float,
) -> DialogueSteeringEffect:
    if len(rows) != len(effects) or not rows:
        raise ValueError("clustered dialogue effect rows are invalid")
    by_cluster: dict[str, list[float]] = {}
    for row, effect in zip(rows, effects, strict=True):
        by_cluster.setdefault(row.cluster_id, []).append(effect)
    cluster_ids = tuple(sorted(by_cluster))
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(resamples):
        sampled = tuple(
            cluster_ids[rng.randrange(len(cluster_ids))]
            for _ in cluster_ids
        )
        values = tuple(
            value for cluster_id in sampled for value in by_cluster[cluster_id]
        )
        estimates.append(statistics.fmean(values))
    estimates.sort()
    tail = (1.0 - confidence) / 2.0
    lower_index = min(len(estimates) - 1, max(0, int(tail * len(estimates))))
    upper_index = min(
        len(estimates) - 1,
        max(0, int((1.0 - tail) * len(estimates)) - 1),
    )
    return DialogueSteeringEffect(
        mean=statistics.fmean(effects),
        ci_lower=estimates[lower_index],
        ci_upper=estimates[upper_index],
        cluster_count=len(cluster_ids),
        row_count=len(rows),
    )


def _train_restart(
    *,
    rows: tuple[DialogueSteeringTraceRow, ...],
    preregistration_sha256: str,
    seed: int,
    restart: int,
    max_online_episodes: int,
    eval_every: int,
    learning_rate: float,
) -> tuple[SteeringGateArtifact, float, float, bool]:
    initial = _artifact_for_restart(
        feature_names=tuple(name for name, _ in rows[0].observations),
        preregistration_sha256=preregistration_sha256,
        seed=seed,
        restart=restart,
    )
    gate = SteeringGateModule(
        artifact=initial,
        learning_rate=learning_rate,
        decision_mode="evidence-stochastic",
        exploration_seed=seed * 10_000 + restart,
        wiring_level=WiringLevel.SHADOW,
    )
    credit = CreditModule()
    rng = random.Random(seed * 1_000_003 + restart * 97 + 29)
    initial_losses, _ = _loss_rows(initial, rows)
    initial_loss = statistics.fmean(initial_losses)
    trajectory: list[float] = [initial_loss]
    for episode_index in range(max_online_episodes):
        row = rows[rng.randrange(len(rows))]
        decision = gate.replay_observations(row.observations).value
        rebound = bind_steering_terminal_prediction_error_decisions(
            row.terminal_prediction_error,
            episode_id=(
                f"{row.episode_id}:c3:s{seed}:r{restart}:e{episode_index}"
            ),
            decision_ids=(decision.decision_id,),
        )
        credit_snapshot = credit.settle_steering_terminal_prediction_errors(
            (rebound,), timestamp_ms=episode_index
        )
        gate.settle_terminal_credit(credit_snapshot)
        if (
            (episode_index + 1) % eval_every == 0
            or episode_index + 1 == max_online_episodes
        ):
            losses, _ = _loss_rows(gate.artifact, rows)
            trajectory.append(statistics.fmean(losses))
    final_window = trajectory[-min(3, len(trajectory)) :]
    final_loss = statistics.fmean(final_window)
    changed = (
        gate.artifact.weights != initial.weights
        or gate.artifact.bias != initial.bias
    )
    return gate.artifact, initial_loss, final_loss, changed


def _run_seed(
    *,
    train_rows: tuple[DialogueSteeringTraceRow, ...],
    validation_rows: tuple[DialogueSteeringTraceRow, ...],
    preregistration_sha256: str,
    seed: int,
    policy_restarts: int,
    max_online_episodes: int,
    eval_every: int,
    learning_rate: float,
    bootstrap_resamples: int,
    bootstrap_confidence: float,
) -> DialogueSteeringSeedPoint:
    candidates = tuple(
        (
            restart,
            *_train_restart(
                rows=train_rows,
                preregistration_sha256=preregistration_sha256,
                seed=seed,
                restart=restart,
                max_online_episodes=max_online_episodes,
                eval_every=eval_every,
                learning_rate=learning_rate,
            ),
        )
        for restart in range(policy_restarts)
    )
    scored = tuple(
        (
            statistics.fmean(_loss_rows(candidate[1], train_rows)[0]),
            candidate,
        )
        for candidate in candidates
    )
    selection_loss, selected = min(scored, key=lambda item: item[0])
    restart, artifact, initial_loss, final_loss, changed = selected
    learned_rows, flags = _loss_rows(artifact, validation_rows)
    noop_rows = tuple(row.normalized_noop_loss for row in validation_rows)
    always_rows = tuple(row.normalized_steer_loss for row in validation_rows)
    steer_rate = statistics.fmean(1.0 if flag else 0.0 for flag in flags)
    random_flags = [False] * len(validation_rows)
    random_indices = list(range(len(validation_rows)))
    random.Random(seed * 101 + 71).shuffle(random_indices)
    for index in random_indices[: round(steer_rate * len(validation_rows))]:
        random_flags[index] = True
    random_rows = tuple(
        always_rows[index] if random_flags[index] else noop_rows[index]
        for index in range(len(validation_rows))
    )
    positive_flags = tuple(
        flags[index]
        for index, row in enumerate(validation_rows)
        if row.relative_steer_credit > 0.0
    )
    nonpositive_flags = tuple(
        flags[index]
        for index, row in enumerate(validation_rows)
        if row.relative_steer_credit <= 0.0
    )
    positive_rate = (
        statistics.fmean(1.0 if value else 0.0 for value in positive_flags)
        if positive_flags
        else 0.0
    )
    nonpositive_rate = (
        statistics.fmean(1.0 if value else 0.0 for value in nonpositive_flags)
        if nonpositive_flags
        else 0.0
    )
    return DialogueSteeringSeedPoint(
        seed=seed,
        selected_restart=restart,
        selection_train_loss=selection_loss,
        initial_train_loss=initial_loss,
        final_train_loss=final_loss,
        convergence_improvement=initial_loss - final_loss,
        arms=DialogueSteeringArms(
            noop=statistics.fmean(noop_rows),
            learned_gate=statistics.fmean(learned_rows),
            always_on=statistics.fmean(always_rows),
            random_gate=statistics.fmean(random_rows),
        ),
        steer_rate=steer_rate,
        steer_rate_positive_credit=positive_rate,
        steer_rate_nonpositive_credit=nonpositive_rate,
        gate_selectivity=positive_rate - nonpositive_rate,
        gain_vs_noop=_clustered_effect(
            rows=validation_rows,
            effects=tuple(
                noop - learned
                for noop, learned in zip(noop_rows, learned_rows, strict=True)
            ),
            seed=seed * 101 + 3,
            resamples=bootstrap_resamples,
            confidence=bootstrap_confidence,
        ),
        gain_vs_always_on=_clustered_effect(
            rows=validation_rows,
            effects=tuple(
                always - learned
                for always, learned in zip(always_rows, learned_rows, strict=True)
            ),
            seed=seed * 101 + 5,
            resamples=bootstrap_resamples,
            confidence=bootstrap_confidence,
        ),
        gain_vs_random_gate=_clustered_effect(
            rows=validation_rows,
            effects=tuple(
                random_loss - learned
                for random_loss, learned in zip(
                    random_rows, learned_rows, strict=True
                )
            ),
            seed=seed * 101 + 7,
            resamples=bootstrap_resamples,
            confidence=bootstrap_confidence,
        ),
        policy_parameters_changed=changed,
        selected_gate_artifact=artifact,
    )


def _aggregate(
    points: tuple[DialogueSteeringSeedPoint, ...],
) -> DialogueSteeringAggregate:
    def mean(selector) -> float:
        return statistics.fmean(selector(point) for point in points)

    return DialogueSteeringAggregate(
        seed_count=len(points),
        noop_loss_mean=mean(lambda point: point.arms.noop),
        learned_gate_loss_mean=mean(lambda point: point.arms.learned_gate),
        always_on_loss_mean=mean(lambda point: point.arms.always_on),
        random_gate_loss_mean=mean(lambda point: point.arms.random_gate),
        convergence_improvement_mean=mean(
            lambda point: point.convergence_improvement
        ),
        convergence_improvement_worst_seed=min(
            point.convergence_improvement for point in points
        ),
        gate_selectivity_mean=mean(lambda point: point.gate_selectivity),
        gate_selectivity_worst_seed=min(point.gate_selectivity for point in points),
        gain_vs_noop_mean_worst_seed=min(point.gain_vs_noop.mean for point in points),
        gain_vs_always_on_mean_worst_seed=min(
            point.gain_vs_always_on.mean for point in points
        ),
        gain_vs_random_gate_mean_worst_seed=min(
            point.gain_vs_random_gate.mean for point in points
        ),
        gain_vs_noop_ci_lower_worst_seed=min(
            point.gain_vs_noop.ci_lower for point in points
        ),
        gain_vs_always_on_ci_lower_worst_seed=min(
            point.gain_vs_always_on.ci_lower for point in points
        ),
        gain_vs_random_gate_ci_lower_worst_seed=min(
            point.gain_vs_random_gate.ci_lower for point in points
        ),
    )


def run_dialogue_steering_evidence(
    *,
    train_rows: tuple[DialogueSteeringTraceRow, ...],
    validation_rows: tuple[DialogueSteeringTraceRow, ...],
    preregistration_sha256: str,
    seed_schedule: tuple[int, ...] = (0, 1, 2, 3, 4),
    policy_restarts: int = 4,
    max_online_episodes: int = 1200,
    eval_every: int = 80,
    learning_rate: float = 0.05,
    bootstrap_resamples: int = 5000,
    bootstrap_confidence: float = 0.95,
    thresholds: DialogueSteeringThresholds | None = None,
    artifact_fit_prerequisite_passed: bool,
) -> DialogueSteeringReport:
    if not train_rows or not validation_rows:
        raise ValueError("C3 requires non-empty train and validation traces")
    if not seed_schedule or len(set(seed_schedule)) != len(seed_schedule):
        raise ValueError("C3 seed schedule must be non-empty and unique")
    if policy_restarts < 2 or max_online_episodes < eval_every or eval_every < 1:
        raise ValueError("C3 multi-restart/episode schedule is invalid")
    if bootstrap_resamples < 100 or not 0.5 < bootstrap_confidence < 1.0:
        raise ValueError("C3 bootstrap configuration is invalid")
    feature_names = tuple(name for name, _ in train_rows[0].observations)
    all_rows = (*train_rows, *validation_rows)
    if any(
        tuple(name for name, _ in row.observations) != feature_names
        for row in all_rows
    ):
        raise ValueError("C3 gate observation schemas drift across rows")
    if set(row.sample_id for row in train_rows) & set(
        row.sample_id for row in validation_rows
    ):
        raise ValueError("C3 train/validation sample overlap")
    active_thresholds = thresholds or DialogueSteeringThresholds()
    points = tuple(
        _run_seed(
            train_rows=train_rows,
            validation_rows=validation_rows,
            preregistration_sha256=preregistration_sha256,
            seed=seed,
            policy_restarts=policy_restarts,
            max_online_episodes=max_online_episodes,
            eval_every=eval_every,
            learning_rate=learning_rate,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_confidence=bootstrap_confidence,
        )
        for seed in seed_schedule
    )
    aggregate = _aggregate(points)
    sensitive_fraction = statistics.fmean(
        1.0
        if abs(row.relative_steer_credit)
        >= active_thresholds.action_sensitivity_abs_credit
        else 0.0
        for row in validation_rows
    )
    lower_required = active_thresholds.require_clustered_ci_lower_positive
    structural = (
        artifact_fit_prerequisite_passed
        and all(row.shadow_owner_chain_complete for row in all_rows)
        and all(row.shadow_hook_executed for row in all_rows)
        and not any(row.free_bias_present for row in all_rows)
        and all(row.zero_code_strict_noop for row in all_rows)
        and not any(row.raw_text_retained for row in all_rows)
        and not any(row.evaluation_writeback_allowed for row in all_rows)
        and all(point.policy_parameters_changed for point in points)
    )
    conditions = {
        "real-trace": (
            len(validation_rows) >= active_thresholds.min_real_trace_turns
        ),
        "action-sensitivity": (
            sensitive_fraction >= active_thresholds.min_action_sensitive_fraction
        ),
        "convergence": (
            aggregate.convergence_improvement_worst_seed
            >= active_thresholds.min_convergence_improvement
        ),
        "gain-vs-noop": (
            aggregate.gain_vs_noop_mean_worst_seed
            >= active_thresholds.min_gain_vs_noop
            and (
                not lower_required
                or aggregate.gain_vs_noop_ci_lower_worst_seed > 0.0
            )
        ),
        "gain-vs-always-on": (
            aggregate.gain_vs_always_on_mean_worst_seed
            >= active_thresholds.min_gain_vs_always_on
            and (
                not lower_required
                or aggregate.gain_vs_always_on_ci_lower_worst_seed > 0.0
            )
        ),
        "gain-vs-random-gate": (
            aggregate.gain_vs_random_gate_mean_worst_seed
            >= active_thresholds.min_gain_vs_random_gate
            and (
                not lower_required
                or aggregate.gain_vs_random_gate_ci_lower_worst_seed > 0.0
            )
        ),
        "gate-selectivity": (
            aggregate.gate_selectivity_worst_seed
            >= active_thresholds.min_gate_selectivity
        ),
        "structural-integrity": structural,
    }
    failed = tuple(name for name, passed in conditions.items() if not passed)
    exit_reason = "admitted"
    if "action-sensitivity" in failed:
        exit_reason = "dialogue-n-plus-one-signal-insensitive-to-steering"
    elif failed:
        exit_reason = "proxy-level-when-to-steer-transfer-not-supported"
    admission = DialogueSteeringAdmission(
        admitted=not failed,
        condition_real_trace=conditions["real-trace"],
        condition_action_sensitivity=conditions["action-sensitivity"],
        condition_convergence=conditions["convergence"],
        condition_gain_vs_noop=conditions["gain-vs-noop"],
        condition_gain_vs_always_on=conditions["gain-vs-always-on"],
        condition_gain_vs_random_gate=conditions["gain-vs-random-gate"],
        condition_gate_selectivity=conditions["gate-selectivity"],
        condition_structural_integrity=conditions["structural-integrity"],
        failed_conditions=failed,
        exit_reason=exit_reason,
        description=(
            "Dialogue-domain gate learned when to steer from sparse PE-owned "
            "terminal credit."
            if not failed
            else "C3 transfer blocked without threshold or signal substitution."
        ),
    )
    return DialogueSteeringReport(
        schema_version=DIALOGUE_STEERING_EVIDENCE_SCHEMA_VERSION,
        preregistration_sha256=preregistration_sha256,
        train_turn_count=len(train_rows),
        validation_turn_count=len(validation_rows),
        train_cluster_count=len({row.cluster_id for row in train_rows}),
        validation_cluster_count=len({row.cluster_id for row in validation_rows}),
        action_sensitive_fraction=sensitive_fraction,
        seed_schedule=seed_schedule,
        policy_restarts=policy_restarts,
        max_online_episodes=max_online_episodes,
        eval_every=eval_every,
        learning_rate=learning_rate,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_confidence=bootstrap_confidence,
        thresholds=active_thresholds,
        seed_points=points,
        aggregate=aggregate,
        admission=admission,
        substrate_trainable_parameter_count=0,
        reader_parameters_changed=False,
        executor_parameters_changed=False,
        policy_parameters_changed=all(
            point.policy_parameters_changed for point in points
        ),
        free_bias_present=False,
        zero_code_strict_noop=True,
        raw_text_retained=False,
        evaluation_writeback_allowed=False,
        terminal_credit_source=(
            "substrate_n_plus_one_representation_pe->credit->steering_gate"
        ),
        description=(
            "C3 real-dialogue SHADOW transfer with multi-restart training-side "
            "selection and dyad-clustered heldout confidence intervals."
        ),
    )


__all__ = (
    "DIALOGUE_STEERING_EVIDENCE_SCHEMA_VERSION",
    "DialogueSteeringAdmission",
    "DialogueSteeringAggregate",
    "DialogueSteeringArms",
    "DialogueSteeringEffect",
    "DialogueSteeringReport",
    "DialogueSteeringSeedPoint",
    "DialogueSteeringThresholds",
    "DialogueSteeringTraceDataset",
    "DialogueSteeringTraceRow",
    "run_dialogue_steering_evidence",
)
