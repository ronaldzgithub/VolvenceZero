from __future__ import annotations

from dataclasses import dataclass, replace
import math
import os

from volvence_zero.credit.gate import (
    CreditRecord,
    CreditSnapshot,
    GateDecision,
    ModificationGate,
    SelfModificationRecord,
)
from volvence_zero.environment import EnvironmentOutcome
from volvence_zero.internal_rl.environment import (
    InternalRLDelayedCreditAssignment,
    InternalRLEnvStep,
    InternalRLEnvironment,
    InternalRLProofEpisode,
    InternalRLProofProgress,
)
from volvence_zero.memory import Track
from volvence_zero.prediction import PredictionErrorSnapshot
from volvence_zero.runtime import WiringLevel
from volvence_zero.runtime.kernel import stable_value_hash
from volvence_zero.tensor_backend import is_torch_available
from volvence_zero.substrate import OpenWeightResidualRuntime, SubstrateSnapshot
from volvence_zero.temporal import (
    ControllerState,
    FamilyOutcomeFeedback,
    FullLearnedTemporalPolicy,
    LearnedLiteTemporalPolicy,
    MetacontrollerParameterSnapshot,
    MetacontrollerParameterStore,
    MetacontrollerRuntimeState,
    TemporalAbstractionSnapshot,
)
from volvence_zero.temporal.metacontroller_components import (
    _project_to_ndim,
    residual_sequence_from_snapshot,
    summarize_residual_activations,
)

_CAUSAL_ACTION_ADVANTAGE_SCALE_FLOOR = 0.05


@dataclass(frozen=True)
class ZTransition:
    step_index: int
    track: Track
    abstract_action: str
    controller_state: ControllerState
    observation_signature: tuple[float, ...]
    policy_action: tuple[float, ...]
    latent_code: tuple[float, ...]
    decoder_output: tuple[float, ...]
    applied_control: tuple[float, ...]
    downstream_effect: tuple[float, ...]
    hidden_state: tuple[float, ...]
    policy_score: float
    log_prob: float
    reward: float
    raw_reward: float
    policy_replacement_quality: float
    backend_name: str
    backend_fidelity: float
    policy_mean: tuple[float, ...] = ()
    policy_std: tuple[float, ...] = ()
    policy_noise: tuple[float, ...] = ()
    value_estimate: float = 0.0
    return_estimate: float = 0.0
    advantage_estimate: float = 0.0
    replacement_effect_delta: float = 0.0
    reward_components: tuple[tuple[str, float], ...] = ()
    reward_mode: str = "dense"
    proof_subgoal_id: str | None = None
    proof_subgoal_score: float = 0.0
    proof_subgoal_completed: bool = False
    proof_terminal_success: bool = False
    active_family_id: str | None = None
    transition_source: str = "synthetic"
    runtime_turn_index: int = 0
    prediction_id: str = ""
    environment_outcome_id: str = ""
    runtime_base_mean: tuple[float, ...] = ()
    runtime_base_std: tuple[float, ...] = ()
    runtime_previous_code: tuple[float, ...] = ()
    runtime_beta_t: float | tuple[float, ...] = ()
    runtime_other_track_sum: tuple[float, ...] = ()
    runtime_action_head_residual: tuple[float, ...] = ()
    lineage_matched: bool = False
    runtime_segment_id: str = ""
    runtime_terminal: bool = False
    runtime_milestone: bool = False
    runtime_action_head_state: tuple[float, ...] = ()


@dataclass(frozen=True)
class ZRollout:
    rollout_id: str
    track: Track
    transitions: tuple[ZTransition, ...]
    total_reward: float
    description: str
    replacement_mode: str = "causal"
    reward_mode: str = "dense"
    proof_episode_id: str | None = None
    completed_subgoals: tuple[str, ...] = ()
    completed_family_ids: tuple[str, ...] = ()
    terminal_success: bool = False
    delayed_credit_assignments: tuple[InternalRLDelayedCreditAssignment, ...] = ()


@dataclass(frozen=True)
class CausalPolicyState:
    track: Track
    hidden_state: tuple[float, ...]
    previous_action: tuple[float, ...]
    step_index: int


@dataclass(frozen=True)
class DualTrackRollout:
    task_rollout: ZRollout
    relationship_rollout: ZRollout
    description: str


@dataclass(frozen=True)
class CausalPolicyParameters:
    track: Track
    weights: tuple[float, ...]
    persistence: float
    learning_rate: float
    update_step: int
    critic_weights: tuple[float, ...] = ()
    critic_bias: float = 0.0


class RuntimeReplayLineageError(RuntimeError):
    """A runtime outcome/PE lineage does not match the captured action."""


@dataclass(frozen=True)
class RuntimeActionCapture:
    capture_id: str
    turn_index: int
    track: Track
    prediction_id: str
    substrate_snapshot: SubstrateSnapshot
    temporal_snapshot: TemporalAbstractionSnapshot
    runtime_state: MetacontrollerRuntimeState
    previous_code: tuple[float, ...]
    observation_signature: tuple[float, ...]
    policy_action: tuple[float, ...]
    policy_mean: tuple[float, ...]
    policy_std: tuple[float, ...]
    policy_noise: tuple[float, ...]
    log_prob: float
    value_estimate: float
    runtime_base_mean: tuple[float, ...]
    runtime_base_std: tuple[float, ...]
    runtime_beta_t: tuple[float, ...]
    runtime_other_track_sum: tuple[float, ...]
    runtime_action_head_state: tuple[float, ...]
    runtime_action_head_residual: tuple[float, ...]


@dataclass(frozen=True)
class RuntimeReplaySettlement:
    capture_id: str
    rollout: ZRollout | None
    lineage_matched: bool
    environment_outcome_id: str = ""
    drop_reason: str = ""


@dataclass(frozen=True)
class RuntimeReplayCheckpoint:
    pending_capture: RuntimeActionCapture | None
    previous_code: tuple[float, ...]
    captured_count: int
    settled_count: int
    dropped_count: int
    last_drop_reason: str


@dataclass(frozen=True)
class CausalPolicyCheckpoint:
    checkpoint_id: str
    parameters_by_track: tuple[CausalPolicyParameters, ...]
    metacontroller_snapshot: MetacontrollerParameterSnapshot
    policy_optimization_fingerprint: str = ""
    temporal_learning_fingerprint: str = ""
    runtime_replay: RuntimeReplayCheckpoint | None = None


@dataclass(frozen=True)
class OptimizationReport:
    track: Track
    average_reward: float
    baseline_reward: float
    mean_advantage: float
    surrogate_objective: float
    clip_fraction: float
    kl_penalty: float
    parameter_summary: str
    epochs_executed: int = 1
    kl_early_stopped: bool = False
    parameters_changed: bool = False
    rollout_count: int = 1
    transition_count: int = 0
    mean_return: float = 0.0
    value_loss: float = 0.0
    parameter_change_norm: float = 0.0
    replacement_effect_delta: float = 0.0
    # autograd-owner-integration Phase C: real torch PPO evidence over the live
    # ZTransition batch (append-only; DISABLED leaves these at defaults).
    torch_backend: str = "disabled"
    torch_parameters_changed: int = 0
    torch_policy_loss: float = 0.0
    torch_value_loss: float = 0.0
    torch_approx_kl: float = 0.0
    torch_wrote_back: bool = False


@dataclass(frozen=True)
class TransitionBatchTargets:
    transitions: tuple[ZTransition, ...]
    normalized_advantages: tuple[float, ...]
    returns: tuple[float, ...]
    mean_return: float
    value_loss: float


@dataclass(frozen=True)
class PolicyBatchResult:
    report: OptimizationReport
    updated_rollouts: tuple[ZRollout, ...]


@dataclass(frozen=True)
class DualTrackOptimizationReport:
    task_report: OptimizationReport
    relationship_report: OptimizationReport
    description: str


@dataclass(frozen=True)
class PolicyOptimizationResult:
    optimization_report: DualTrackOptimizationReport
    modification_records: tuple[SelfModificationRecord, ...]
    policy_update_applied: bool
    total_kl_divergence: float
    total_epochs_executed: int


def _clamp(value: float) -> float:
    return max(-1.0, min(1.0, value))


def runtime_replay_policy_distribution(
    *,
    base_mean: tuple[float, ...],
    base_std: tuple[float, ...],
    previous_code: tuple[float, ...],
    beta_t: float | tuple[float, ...],
    track_weights: tuple[float, ...],
    other_track_sum: tuple[float, ...],
    modulation_strength: float,
    action_head_residual: tuple[float, ...] = (),
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Reconstruct the ndim runtime behavior distribution.

    The posterior is the stochastic source used by the real runtime forward.
    Track modulation is applied before the captured beta gate blends the
    candidate with the previous code.  Pure and torch PPO mirror this formula;
    synthetic transitions never enter it.
    """

    n = len(base_mean)
    lengths = {
        len(base_std),
        len(previous_code),
        len(track_weights),
        len(other_track_sum),
    }
    if n == 0 or lengths != {n}:
        raise ValueError(
            "runtime replay distribution requires aligned non-empty vectors: "
            f"mean={n}, std={len(base_std)}, previous={len(previous_code)}, "
            f"weights={len(track_weights)}, other_sum={len(other_track_sum)}"
        )
    beta_vector = (
        tuple(float(beta_t) for _ in range(n))
        if isinstance(beta_t, (float, int))
        else tuple(beta_t)
    )
    if len(beta_vector) != n or any(
        not 0.0 <= value <= 1.0 for value in beta_vector
    ):
        raise ValueError(
            "runtime replay beta_t must be a scalar or aligned [0, 1] vector, "
            f"got {beta_t!r}"
        )
    if modulation_strength < 0.0:
        raise ValueError(
            "runtime replay modulation_strength must be >= 0, "
            f"got {modulation_strength!r}"
        )
    if action_head_residual and len(action_head_residual) != n:
        raise ValueError(
            "runtime replay action-head residual dimension mismatch: "
            f"expected={n}, actual={len(action_head_residual)}"
        )
    aggregate_weights = tuple(
        (track_weights[index] + other_track_sum[index]) / 3.0
        for index in range(n)
    )
    gains = tuple(
        max(
            0.5,
            min(
                1.5,
                1.0
                + modulation_strength
                * (aggregate_weights[index] * n - 1.0),
            ),
        )
        for index in range(n)
    )
    modulated_mean = tuple(
        _clamp(base_mean[index] * gains[index])
        for index in range(n)
    )
    candidate_mean = tuple(
        _clamp(
            modulated_mean[index]
            + (
                action_head_residual[index]
                if action_head_residual
                else 0.0
            )
        )
        for index in range(n)
    )
    policy_mean = tuple(
        _clamp(
            beta_vector[index] * candidate_mean[index]
            + (1.0 - beta_vector[index]) * previous_code[index]
        )
        for index in range(n)
    )
    policy_std = tuple(
        max(
            0.02,
            min(
                0.5,
                abs(
                    beta_vector[index]
                    * base_std[index]
                    * gains[index]
                    * 0.5
                ),
            ),
        )
        for index in range(n)
    )
    return policy_mean, policy_std


def _surface_signature(substrate_snapshot: SubstrateSnapshot, n_z: int = 3) -> tuple[float, ...]:
    if substrate_snapshot.residual_activations:
        values = [
            sum(activation.activation) / len(activation.activation)
            for activation in substrate_snapshot.residual_activations
            if activation.activation
        ]
    else:
        values = [
            sum(feature.values) / len(feature.values)
            for feature in substrate_snapshot.feature_surface
            if feature.values
        ]
    if not values:
        return tuple(0.0 for _ in range(n_z))
    average = sum(values) / len(values)
    maximum = max(values)
    minimum = min(values)
    spread = maximum - minimum
    stability = 1.0 - spread
    leading = values[-1] if values else 0.0
    richness = tuple(
        _clamp(value)
        for value in (
            average,
            maximum,
            spread,
            stability,
            leading,
        )
    )
    if n_z <= 3:
        return richness[:n_z]
    from volvence_zero.temporal.metacontroller_components import _project_to_ndim
    return _project_to_ndim(richness, n_z)


def _sequence_observation_signature(substrate_snapshot: SubstrateSnapshot, n_z: int = 3) -> tuple[float, ...]:
    sequence = residual_sequence_from_snapshot(substrate_snapshot)
    if not sequence:
        return tuple(0.0 for _ in range(n_z))
    summary_vectors = tuple(
        summarize_residual_activations(step.residual_activations, step.feature_surface)
        for step in sequence
    )
    averaged = tuple(
        sum(vector[index] for vector in summary_vectors) / len(summary_vectors)
        for index in range(3)
    )
    peaked = tuple(max(vector[index] for vector in summary_vectors) for index in range(3))
    trended = tuple(summary_vectors[-1][index] - summary_vectors[0][index] for index in range(3))
    volatility = tuple(abs(trended[index]) for index in range(3))
    persistence = tuple(
        _clamp(1.0 - abs(peaked[index] - averaged[index]))
        for index in range(3)
    )
    raw = tuple(
        _clamp(averaged[index] * 0.35 + peaked[index] * 0.25 + trended[index] * 0.20 + persistence[index] * 0.20)
        for index in range(3)
    )
    if n_z <= 3:
        return raw[:n_z]
    enriched = raw + volatility + persistence
    return _project_to_ndim(enriched, n_z)


class CausalZPolicy:
    """Causal z-policy with bounded stochastic actions and a lightweight critic."""

    def __init__(
        self,
        *,
        parameter_store: MetacontrollerParameterStore,
        rl_backend: WiringLevel = WiringLevel.DISABLED,
        runtime_track_modulation_strength: float = 0.0,
        causal_action_head_wiring: WiringLevel = WiringLevel.DISABLED,
        causal_action_head_strength: float = 0.0,
    ) -> None:
        self._parameter_store = parameter_store
        self._rl_backend = rl_backend
        if runtime_track_modulation_strength < 0.0:
            raise ValueError(
                "runtime_track_modulation_strength must be >= 0, "
                f"got {runtime_track_modulation_strength!r}"
            )
        self._runtime_track_modulation_strength = float(
            runtime_track_modulation_strength
        )
        if not 0.0 <= causal_action_head_strength <= 1.0:
            raise ValueError(
                "causal_action_head_strength must be within [0, 1], "
                f"got {causal_action_head_strength!r}"
            )
        self._causal_action_head_wiring = causal_action_head_wiring
        self._causal_action_head_strength = float(
            causal_action_head_strength
        )
        self._value_weights: dict[Track, tuple[float, ...]] = {
            track: tuple(weight * 0.8 for weight in parameter_store.track_weights[track])
            for track in (Track.WORLD, Track.SELF, Track.SHARED)
        }
        self._value_bias: dict[Track, float] = {
            Track.WORLD: 0.05,
            Track.SELF: 0.05,
            Track.SHARED: 0.05,
        }

    @property
    def rl_backend(self) -> WiringLevel:
        return self._rl_backend

    def set_rl_backend(self, wiring_level: WiringLevel) -> None:
        """Switch the PPO backend (DISABLED <-> SHADOW <-> ACTIVE); reversible."""

        self._rl_backend = wiring_level

    @property
    def runtime_track_modulation_strength(self) -> float:
        """Owner-local evidence that sandbox and live forward share one bridge."""

        return self._runtime_track_modulation_strength

    @property
    def causal_action_head_wiring(self) -> WiringLevel:
        return self._causal_action_head_wiring

    @property
    def causal_action_head_strength(self) -> float:
        return self._causal_action_head_strength

    @property
    def n_z(self) -> int:
        return self._parameter_store.n_z

    def initial_state(self, *, track: Track) -> CausalPolicyState:
        n = self.n_z
        return CausalPolicyState(
            track=track,
            hidden_state=tuple(0.0 for _ in range(n)),
            previous_action=tuple(0.0 for _ in range(n)),
            step_index=0,
        )

    def step(
        self,
        *,
        substrate_snapshot: SubstrateSnapshot,
        state: CausalPolicyState,
        observation_mode: str = "default",
    ) -> tuple[
        CausalPolicyState,
        tuple[float, ...],
        tuple[float, ...],
        tuple[float, ...],
        tuple[float, ...],
        tuple[float, ...],
        tuple[float, ...],
        float,
        float,
        float,
    ]:
        n = self.n_z
        if observation_mode == "proof":
            surface = _sequence_observation_signature(substrate_snapshot, n)
        else:
            surface = _surface_signature(substrate_snapshot, n)
        features = self._policy_features(
            surface=surface,
            previous_action=state.previous_action,
            previous_hidden_state=state.hidden_state,
        )
        hidden_state = tuple(
            _clamp(
                previous * self._parameter_store.persistence
                + current * (1.0 - self._parameter_store.persistence)
            )
            for previous, current in zip(state.hidden_state, features, strict=True)
        )
        weights = self._project_track_weights(track=state.track, n=n)
        policy_mean = self._policy_mean(
            track=state.track,
            hidden_state=hidden_state,
            surface=surface,
            previous_action=state.previous_action,
            weights=weights,
        )
        update_pressure = 0.0
        if observation_mode == "proof":
            update_pressure = min(self._parameter_store.update_steps[state.track] / 2.0, 1.0)
            if update_pressure > 0.0:
                exploitation_mean = tuple(
                    _clamp(
                        weights[index] * 0.48
                        + surface[index] * 0.34
                        + hidden_state[index] * 0.18
                    )
                    for index in range(n)
                )
                policy_mean = tuple(
                    _clamp(
                        policy_mean[index] * (1.0 - 0.45 * update_pressure)
                        + exploitation_mean[index] * 0.45 * update_pressure
                    )
                    for index in range(n)
                )
        policy_std = self._policy_std(
            hidden_state=hidden_state,
            surface=surface,
            previous_action=state.previous_action,
            policy_mean=policy_mean,
        )
        if update_pressure > 0.0:
            policy_std = tuple(max(0.04, value * (1.0 - 0.35 * update_pressure)) for value in policy_std)
        policy_noise = self._policy_noise(
            hidden_state=hidden_state,
            surface=surface,
            step_index=state.step_index,
            track=state.track,
        )
        policy_action = self._sample_action(
            policy_mean=policy_mean,
            policy_std=policy_std,
            policy_noise=policy_noise,
        )
        next_state = CausalPolicyState(
            track=state.track,
            hidden_state=hidden_state,
            previous_action=policy_action,
            step_index=state.step_index + 1,
        )
        policy_score = self._policy_score(
            weights=weights,
            hidden_state=hidden_state,
            surface=surface,
            policy_action=policy_action,
        )
        log_prob = self._log_prob(
            policy_action=policy_action,
            policy_mean=policy_mean,
            policy_std=policy_std,
        )
        value_estimate = self._value_estimate(
            track=state.track,
            hidden_state=hidden_state,
            surface=surface,
        )
        return (
            next_state,
            surface,
            hidden_state,
            policy_action,
            policy_mean,
            policy_std,
            policy_noise,
            policy_score,
            log_prob,
            value_estimate,
        )

    def export_parameters(self) -> tuple[CausalPolicyParameters, ...]:
        return tuple(
            CausalPolicyParameters(
                track=track,
                weights=self._parameter_store.track_weights[track],
                persistence=self._parameter_store.persistence,
                learning_rate=self._parameter_store.learning_rate,
                update_step=self._parameter_store.update_steps[track],
                critic_weights=self._value_weights[track],
                critic_bias=self._value_bias[track],
            )
            for track in (Track.WORLD, Track.SELF, Track.SHARED)
        )

    def create_checkpoint(self, *, checkpoint_id: str) -> CausalPolicyCheckpoint:
        parameters = self.export_parameters()
        optimization_state = tuple(
            (
                params.track.value,
                params.update_step,
                params.critic_weights,
                params.critic_bias,
            )
            for params in parameters
        ) + (
            (
                "causal_action_heads",
                tuple(
                    self._parameter_store.causal_action_head_parameters(
                        track=track
                    )
                    for track in (
                        Track.WORLD,
                        Track.SELF,
                        Track.SHARED,
                    )
                ),
            ),
        )
        return CausalPolicyCheckpoint(
            checkpoint_id=checkpoint_id,
            parameters_by_track=parameters,
            metacontroller_snapshot=self._parameter_store.export_parameter_snapshot(),
            policy_optimization_fingerprint=stable_value_hash(
                optimization_state
            ),
            temporal_learning_fingerprint=(
                self._parameter_store.learning_parameter_fingerprint()
            ),
        )

    def restore_checkpoint(self, checkpoint: CausalPolicyCheckpoint) -> None:
        self._parameter_store.track_weights = {
            params.track: params.weights for params in checkpoint.parameters_by_track
        }
        self._value_weights = {
            params.track: (
                params.critic_weights
                if params.critic_weights
                else tuple(weight * 0.8 for weight in params.weights)
            )
            for params in checkpoint.parameters_by_track
        }
        self._value_bias = {
            params.track: params.critic_bias
            for params in checkpoint.parameters_by_track
        }
        self._parameter_store.persistence = checkpoint.parameters_by_track[0].persistence
        self._parameter_store.learning_rate = checkpoint.parameters_by_track[0].learning_rate
        self._parameter_store.update_steps = {
            params.track: params.update_step for params in checkpoint.parameters_by_track
        }
        self._parameter_store.restore_parameter_snapshot(checkpoint.metacontroller_snapshot)

    def _project_track_weights(self, *, track: Track, n: int) -> tuple[float, ...]:
        weights = self._parameter_store.track_weights[track]
        if len(weights) == n:
            return weights
        if len(weights) > n:
            return weights[:n]
        if not weights:
            return tuple(1.0 / max(n, 1) for _ in range(n))
        padded = list(weights)
        while len(padded) < n:
            padded.append(weights[len(padded) % len(weights)])
        total = max(sum(padded), 1e-6)
        return tuple(value / total for value in padded)

    def _policy_features(
        self,
        *,
        surface: tuple[float, ...],
        previous_action: tuple[float, ...],
        previous_hidden_state: tuple[float, ...],
    ) -> tuple[float, ...]:
        return tuple(
            _clamp(
                surface[index] * 0.50
                + previous_action[index] * 0.20
                + previous_hidden_state[index] * 0.30
            )
            for index in range(len(surface))
        )

    def _policy_mean(
        self,
        *,
        track: Track,
        hidden_state: tuple[float, ...],
        surface: tuple[float, ...],
        previous_action: tuple[float, ...],
        weights: tuple[float, ...],
    ) -> tuple[float, ...]:
        if self._runtime_track_modulation_strength > 0.0:
            base_candidate = tuple(
                _clamp(
                    hidden_state[index] * 0.50
                    + surface[index] * 0.30
                    + previous_action[index] * 0.20
                )
                for index in range(len(hidden_state))
            )
            candidate = (
                self._parameter_store.runtime_track_modulated_code(
                    base_candidate,
                    strength=self._runtime_track_modulation_strength,
                    track_override=(track, weights),
                )
            )
        else:
            candidate = tuple(
                _clamp(
                    hidden_state[index] * 0.40
                    + surface[index] * 0.25
                    + previous_action[index] * 0.15
                    + weights[index] * 0.20
                )
                for index in range(len(hidden_state))
            )
        if (
            self._causal_action_head_wiring is WiringLevel.ACTIVE
        ):
            residual = (
                self._parameter_store.causal_action_head_residual(
                    track=track,
                    state_features=hidden_state,
                    strength=self._causal_action_head_strength,
                )
            )
            return tuple(
                _clamp(value + delta)
                for value, delta in zip(
                    candidate,
                    residual,
                    strict=True,
                )
            )
        return candidate

    def _policy_std(
        self,
        *,
        hidden_state: tuple[float, ...],
        surface: tuple[float, ...],
        previous_action: tuple[float, ...],
        policy_mean: tuple[float, ...],
    ) -> tuple[float, ...]:
        return tuple(
            max(
                0.05,
                min(
                    0.25,
                    0.08
                    + abs(hidden_state[index] - surface[index]) * 0.08
                    + abs(policy_mean[index] - previous_action[index]) * 0.10,
                ),
            )
            for index in range(len(policy_mean))
        )

    def _policy_noise(
        self,
        *,
        hidden_state: tuple[float, ...],
        surface: tuple[float, ...],
        step_index: int,
        track: Track,
    ) -> tuple[float, ...]:
        track_factor = {
            Track.WORLD: 1.0,
            Track.SELF: 1.7,
            Track.SHARED: 2.3,
        }[track]
        return tuple(
            math.sin(
                (step_index + 1) * (index + 1) * 1.618
                + hidden_state[index] * 7.0
                + surface[index] * 11.0
                + track_factor
            )
            for index in range(len(hidden_state))
        )

    def _sample_action(
        self,
        *,
        policy_mean: tuple[float, ...],
        policy_std: tuple[float, ...],
        policy_noise: tuple[float, ...],
    ) -> tuple[float, ...]:
        return tuple(
            _clamp(policy_mean[index] + policy_std[index] * policy_noise[index] * 0.5)
            for index in range(len(policy_mean))
        )

    def _policy_score(
        self,
        *,
        weights: tuple[float, ...],
        hidden_state: tuple[float, ...],
        surface: tuple[float, ...],
        policy_action: tuple[float, ...],
    ) -> float:
        n = len(hidden_state)
        score = sum(weights[i] * hidden_state[i] for i in range(n)) / max(n, 1) * 2.0
        score += sum(weights[i] * surface[i] for i in range(n)) / max(n, 1) * 1.5
        score += sum(weights[i] * policy_action[i] for i in range(n)) / max(n, 1) * 1.0
        return _clamp(score)

    def _log_prob(
        self,
        *,
        policy_action: tuple[float, ...],
        policy_mean: tuple[float, ...],
        policy_std: tuple[float, ...],
    ) -> float:
        total = 0.0
        for action_value, mean_value, std_value in zip(policy_action, policy_mean, policy_std, strict=True):
            variance = max(std_value * std_value, 1e-6)
            total += -0.5 * (((action_value - mean_value) ** 2) / variance + math.log(2.0 * math.pi * variance))
        return total

    def _value_estimate(
        self,
        *,
        track: Track,
        hidden_state: tuple[float, ...],
        surface: tuple[float, ...],
    ) -> float:
        weights = self._project_track_weights(track=track, n=len(hidden_state))
        critic_weights = self._value_weights[track]
        score = sum(
            (hidden_state[index] * 0.55 + surface[index] * 0.45)
            * critic_weights[index]
            * (0.6 + weights[index] * 0.4)
            for index in range(len(hidden_state))
        ) / max(len(hidden_state), 1)
        score += self._value_bias[track]
        return max(-1.0, min(1.0, math.tanh(score * 2.5)))

    def _trajectory_gradient(
        self,
        *,
        transitions: tuple[ZTransition, ...],
        advantages: tuple[float, ...],
        track: Track,
    ) -> tuple[float, ...]:
        dims = len(transitions[0].observation_signature)
        accum = [0.0 for _ in range(dims)]
        track_weights = self._project_track_weights(track=track, n=dims)
        for transition, advantage in zip(transitions, advantages, strict=True):
            if transition.transition_source not in {"synthetic", "runtime-replay"}:
                raise ValueError(
                    "unsupported Internal-RL transition source "
                    f"{transition.transition_source!r}"
                )
            for index, value in enumerate(transition.observation_signature):
                variance = max(transition.policy_std[index] ** 2, 1e-6)
                score_term = (transition.policy_action[index] - transition.policy_mean[index]) / variance
                if transition.transition_source == "runtime-replay":
                    aggregate_weight = (
                        track_weights[index]
                        + transition.runtime_other_track_sum[index]
                    ) / 3.0
                    raw_gain = 1.0 + self._runtime_track_modulation_strength * (
                        aggregate_weight * dims - 1.0
                    )
                    gain = max(0.5, min(1.5, raw_gain))
                    unbounded_mean = transition.runtime_base_mean[index] * gain
                    if (
                        gain != raw_gain
                        or not -1.0 < unbounded_mean < 1.0
                    ):
                        policy_sensitivity = 0.0
                    else:
                        policy_sensitivity = (
                            (
                                transition.runtime_beta_t[index]
                                if isinstance(
                                    transition.runtime_beta_t,
                                    tuple,
                                )
                                else transition.runtime_beta_t
                            )
                            * transition.runtime_base_mean[index]
                            * self._runtime_track_modulation_strength
                            * dims
                            / 3.0
                        )
                elif self._runtime_track_modulation_strength > 0.0:
                    base_candidate = _clamp(
                        transition.hidden_state[index] * 0.50
                        + value * 0.30
                        + transition.policy_action[index] * 0.20
                    )
                    policy_sensitivity = max(
                        base_candidate
                        * self._runtime_track_modulation_strength
                        * dims
                        / 3.0,
                        1e-3,
                    )
                else:
                    policy_sensitivity = (
                        max(
                            value * 0.6
                            + transition.hidden_state[index] * 0.4,
                            1e-3,
                        )
                        * (0.55 + track_weights[index] * 0.45)
                    )
                accum[index] += (
                    score_term
                    * policy_sensitivity
                    * advantage
                )
        scale = 1.0 / max(len(transitions), 1)
        return tuple(delta * scale for delta in accum)

    def _surrogate_metrics(
        self,
        *,
        transitions: tuple[ZTransition, ...],
        advantages: tuple[float, ...],
        old_weights: tuple[float, ...],
        new_weights: tuple[float, ...],
    ) -> tuple[float, float, float, float]:
        clipped = 0
        objective_terms: list[float] = []
        kl_terms: list[float] = []
        replacement_effects: list[float] = []
        for transition, advantage in zip(transitions, advantages, strict=True):
            if transition.transition_source == "runtime-replay":
                new_mean, new_std = runtime_replay_policy_distribution(
                    base_mean=transition.runtime_base_mean,
                    base_std=transition.runtime_base_std,
                    previous_code=transition.runtime_previous_code,
                    beta_t=transition.runtime_beta_t,
                    track_weights=new_weights,
                    other_track_sum=transition.runtime_other_track_sum,
                    modulation_strength=self._runtime_track_modulation_strength,
                    action_head_residual=(
                        transition.runtime_action_head_residual
                    ),
                )
            elif transition.transition_source == "synthetic":
                new_mean = self._policy_mean(
                    track=transition.track,
                    hidden_state=transition.hidden_state,
                    surface=transition.observation_signature,
                    previous_action=transition.policy_action,
                    weights=new_weights,
                )
                new_std = self._policy_std(
                    hidden_state=transition.hidden_state,
                    surface=transition.observation_signature,
                    previous_action=transition.policy_action,
                    policy_mean=new_mean,
                )
            else:
                raise ValueError(
                    "unsupported Internal-RL transition source "
                    f"{transition.transition_source!r}"
                )
            new_log_prob = self._log_prob(
                policy_action=transition.policy_action,
                policy_mean=new_mean,
                policy_std=new_std,
            )
            ratio = 2.718281828 ** (new_log_prob - transition.log_prob)
            clipped_ratio = max(
                1.0 - self._parameter_store.clip_epsilon,
                min(1.0 + self._parameter_store.clip_epsilon, ratio),
            )
            if clipped_ratio != ratio:
                clipped += 1
            unclipped_objective = ratio * advantage
            clipped_objective = clipped_ratio * advantage
            if advantage >= 0:
                objective_terms.append(min(unclipped_objective, clipped_objective))
            else:
                objective_terms.append(max(unclipped_objective, clipped_objective))
            if transition.transition_source == "runtime-replay":
                old_mean = transition.policy_mean
                old_std = transition.policy_std
            else:
                old_mean = self._policy_mean(
                    track=transition.track,
                    hidden_state=transition.hidden_state,
                    surface=transition.observation_signature,
                    previous_action=transition.policy_action,
                    weights=old_weights,
                )
                old_std = self._policy_std(
                    hidden_state=transition.hidden_state,
                    surface=transition.observation_signature,
                    previous_action=transition.policy_action,
                    policy_mean=old_mean,
                )
            kl_terms.append(
                self._gaussian_kl(
                    old_mean=old_mean,
                    old_std=old_std,
                    new_mean=new_mean,
                    new_std=new_std,
                )
            )
            replacement_effects.append(
                self._mean_abs_delta(new_mean, old_mean) + transition.policy_replacement_quality * 0.1
            )
        clip_fraction = clipped / max(len(transitions), 1)
        surrogate_objective = sum(objective_terms) / max(len(objective_terms), 1)
        kl_penalty = sum(kl_terms) / max(len(kl_terms), 1)
        replacement_effect_delta = sum(replacement_effects) / max(len(replacement_effects), 1)
        return (surrogate_objective, clip_fraction, kl_penalty, replacement_effect_delta)

    def _estimate_rollout_targets(
        self,
        *,
        rollout: ZRollout,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> TransitionBatchTargets:
        n = len(rollout.transitions)
        returns = [0.0] * n
        if rollout.reward_mode.startswith("proof"):
            running_return = 0.0
            for index in range(n - 1, -1, -1):
                running_return = rollout.transitions[index].reward + gamma * running_return
                returns[index] = running_return
            mean_return = sum(returns) / max(n, 1)
            var_return = sum((value - mean_return) ** 2 for value in returns) / max(n, 1)
            std_return = max(var_return ** 0.5, 1e-8)
            normalized_advantages = tuple((value - mean_return) / std_return for value in returns)
        else:
            values = tuple(t.value_estimate for t in rollout.transitions)
            raw_advantages = [0.0] * n
            last_gae = 0.0
            for index in range(n - 1, -1, -1):
                transition = rollout.transitions[index]
                if index + 1 < n:
                    next_value = values[index + 1]
                elif (
                    rollout.reward_mode == "runtime-replay"
                    and not transition.runtime_terminal
                ):
                    # Runtime replay rollouts are bounded fragments, not
                    # terminal episodes.  Treating every final real tick as a
                    # terminal state subtracts the full critic estimate from
                    # tiny but correctly signed local ecology payoffs, which
                    # can reverse the actor update.  Bootstrap the fragment
                    # boundary from the next published substrate signature;
                    # terminal milestones retain the true zero continuation.
                    next_surface = tuple(
                        _clamp(value + delta)
                        for value, delta in zip(
                            transition.observation_signature,
                            transition.downstream_effect,
                            strict=True,
                        )
                    )
                    next_value = self._value_estimate(
                        track=transition.track,
                        hidden_state=transition.hidden_state,
                        surface=next_surface,
                    )
                else:
                    next_value = 0.0
                delta = rollout.transitions[index].reward + gamma * next_value - values[index]
                last_gae = delta + gamma * gae_lambda * last_gae
                raw_advantages[index] = last_gae
                returns[index] = raw_advantages[index] + values[index]
            if rollout.reward_mode == "runtime-replay":
                # Runtime evidence arrives as one real environment transition
                # per rollout. Per-rollout normalization would turn every
                # one-sample advantage into exactly zero, severing the
                # reward->policy path. Keep bounded raw GAE here; batching and
                # PPO clipping still constrain the update.
                normalized_advantages = tuple(
                    _clamp(value) for value in raw_advantages
                )
            else:
                mean_advantage = sum(raw_advantages) / max(n, 1)
                var_advantage = sum((value - mean_advantage) ** 2 for value in raw_advantages) / max(n, 1)
                std_advantage = max(var_advantage ** 0.5, 1e-8)
                normalized_advantages = tuple((value - mean_advantage) / std_advantage for value in raw_advantages)
            mean_return = sum(returns) / max(n, 1)
        value_loss = sum(
            (returns[index] - rollout.transitions[index].value_estimate) ** 2
            for index in range(n)
        ) / max(n, 1)
        updated = tuple(
            replace(
                transition,
                return_estimate=returns[index],
                advantage_estimate=normalized_advantages[index],
            )
            for index, transition in enumerate(rollout.transitions)
        )
        return TransitionBatchTargets(
            transitions=updated,
            normalized_advantages=normalized_advantages,
            returns=tuple(returns),
            mean_return=sum(returns) / max(n, 1),
            value_loss=value_loss,
        )

    def _aggregate_batch_targets(
        self,
        *,
        rollouts: tuple[ZRollout, ...],
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> tuple[tuple[ZRollout, ...], tuple[ZTransition, ...], tuple[float, ...], float, float]:
        updated_rollouts: list[ZRollout] = []
        aggregated_transitions: list[ZTransition] = []
        aggregated_advantages: list[float] = []
        mean_returns: list[float] = []
        value_losses: list[float] = []
        for rollout in rollouts:
            targets = self._estimate_rollout_targets(
                rollout=rollout,
                gamma=gamma,
                gae_lambda=gae_lambda,
            )
            updated_rollouts.append(replace(rollout, transitions=targets.transitions))
            aggregated_transitions.extend(targets.transitions)
            aggregated_advantages.extend(targets.normalized_advantages)
            mean_returns.append(targets.mean_return)
            value_losses.append(targets.value_loss)
        return (
            tuple(updated_rollouts),
            tuple(aggregated_transitions),
            tuple(aggregated_advantages),
            sum(mean_returns) / max(len(mean_returns), 1),
            sum(value_losses) / max(len(value_losses), 1),
        )

    def _compute_gae(
        self,
        *,
        rollout: ZRollout,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> tuple[float, ...]:
        return self._estimate_rollout_targets(
            rollout=rollout,
            gamma=gamma,
            gae_lambda=gae_lambda,
        ).normalized_advantages

    def _gaussian_kl(
        self,
        *,
        old_mean: tuple[float, ...],
        old_std: tuple[float, ...],
        new_mean: tuple[float, ...],
        new_std: tuple[float, ...],
    ) -> float:
        total = 0.0
        for om, os, nm, ns in zip(old_mean, old_std, new_mean, new_std, strict=True):
            old_var = max(os * os, 1e-6)
            new_var = max(ns * ns, 1e-6)
            total += math.log(max(ns, 1e-6) / max(os, 1e-6)) + (old_var + (om - nm) ** 2) / (2.0 * new_var) - 0.5
        return total / max(len(old_mean), 1)

    def _mean_abs_delta(
        self,
        left: tuple[float, ...],
        right: tuple[float, ...],
    ) -> float:
        return sum(abs(lv - rv) for lv, rv in zip(left, right, strict=True)) / max(len(left), 1)

    def _update_value_head(
        self,
        *,
        track: Track,
        transitions: tuple[ZTransition, ...],
        returns: tuple[float, ...],
    ) -> float:
        weights = list(self._value_weights[track])
        bias = self._value_bias[track]
        lr = self._parameter_store.learning_rate * 0.35
        deltas = [0.0 for _ in weights]
        bias_delta = 0.0
        for transition, return_value in zip(transitions, returns, strict=True):
            features = tuple(
                _clamp(transition.hidden_state[index] * 0.55 + transition.observation_signature[index] * 0.45)
                for index in range(len(weights))
            )
            error = return_value - transition.value_estimate
            for index, feature_value in enumerate(features):
                deltas[index] += error * feature_value
            bias_delta += error
        scale = 1.0 / max(len(transitions), 1)
        for index, delta in enumerate(deltas):
            weights[index] = _clamp(weights[index] + lr * delta * scale)
        bias = max(-1.0, min(1.0, bias + lr * bias_delta * scale * 0.25))
        self._value_weights[track] = tuple(weights)
        self._value_bias[track] = bias
        return sum(abs(delta) for delta in deltas) * scale

    def _update_causal_action_head(
        self,
        *,
        track: Track,
        transitions: tuple[ZTransition, ...],
        advantages: tuple[float, ...],
    ) -> float:
        if self._causal_action_head_wiring is WiringLevel.DISABLED:
            return 0.0
        runtime_transitions = tuple(
            (
                transition,
                advantage,
            )
            for transition, advantage in zip(
                transitions,
                advantages,
                strict=True,
            )
            if transition.transition_source == "runtime-replay"
        )
        if not runtime_transitions:
            return 0.0
        advantage_scale = max(
            math.sqrt(
                sum(
                    advantage * advantage
                    for _, advantage in runtime_transitions
                )
                / len(runtime_transitions)
            ),
            _CAUSAL_ACTION_ADVANTAGE_SCALE_FLOOR,
        )
        state_feature_batch: list[tuple[float, ...]] = []
        action_gradients: list[tuple[float, ...]] = []
        selected_advantages: list[float] = []
        for transition, advantage in runtime_transitions:
            beta_vector = (
                tuple(
                    float(transition.runtime_beta_t)
                    for _ in range(self.n_z)
                )
                if isinstance(
                    transition.runtime_beta_t,
                    (float, int),
                )
                else tuple(transition.runtime_beta_t)
            )
            if len(beta_vector) != self.n_z:
                raise ValueError(
                    "causal action head runtime beta dimension mismatch: "
                    f"expected={self.n_z}, actual={len(beta_vector)}"
                )
            gradient = tuple(
                max(
                    -4.0,
                    min(
                        4.0,
                        (
                            transition.policy_action[index]
                            - transition.policy_mean[index]
                        )
                        / max(
                            transition.policy_std[index] ** 2,
                            1e-4,
                        )
                        * beta_vector[index]
                        * self._causal_action_head_strength,
                    ),
                )
                for index in range(self.n_z)
            )
            if len(transition.runtime_action_head_state) != self.n_z:
                raise ValueError(
                    "causal action head runtime state dimension mismatch: "
                    f"expected={self.n_z}, "
                    f"actual={len(transition.runtime_action_head_state)}"
                )
            state_feature_batch.append(
                transition.runtime_action_head_state
            )
            action_gradients.append(gradient)
            # Normalize only the causal action-head signal.  Track/value
            # updates keep the physical payoff scale, while the low-rank head
            # receives a bounded directionally faithful signal.  Without this
            # floor-relative normalization, exact local progress rewards are
            # orders of magnitude below the head learning rate and produce a
            # fingerprint change with no durable motor effect.
            selected_advantages.append(
                max(-1.0, min(1.0, advantage / advantage_scale))
            )
        return self._parameter_store.update_causal_action_head(
            track=track,
            state_feature_batch=tuple(state_feature_batch),
            action_gradients=tuple(action_gradients),
            advantages=tuple(selected_advantages),
        )

    def optimize(
        self,
        *,
        rollouts: tuple[ZRollout, ...] | ZRollout | None = None,
        rollout: ZRollout | None = None,
        n_epochs: int = 3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        max_kl: float = 0.05,
    ) -> PolicyBatchResult:
        selected_rollouts = rollouts if rollouts is not None else rollout
        if selected_rollouts is None:
            selected_rollouts = ()
        normalized_rollouts = (
            selected_rollouts
            if isinstance(selected_rollouts, tuple)
            else (selected_rollouts,)
        )
        filtered_rollouts = tuple(rollout for rollout in normalized_rollouts if rollout.transitions)
        if not filtered_rollouts:
            empty_track = normalized_rollouts[0].track if normalized_rollouts else Track.SHARED
            return PolicyBatchResult(
                report=OptimizationReport(
                    track=empty_track,
                    average_reward=0.0,
                    baseline_reward=0.0,
                    mean_advantage=0.0,
                    surrogate_objective=0.0,
                    clip_fraction=0.0,
                    kl_penalty=0.0,
                    parameter_summary="no-op",
                    rollout_count=len(normalized_rollouts),
                    transition_count=0,
                ),
                updated_rollouts=normalized_rollouts,
            )
        transition_sources = {
            transition.transition_source
            for selected_rollout in filtered_rollouts
            for transition in selected_rollout.transitions
        }
        if len(transition_sources) != 1:
            raise ValueError(
                "Internal-RL batches cannot mix transition sources, got "
                f"{tuple(sorted(transition_sources))}"
            )
        transition_source = next(iter(transition_sources))
        track = filtered_rollouts[0].track
        updated_rollouts, transitions, advantages, mean_return, value_loss = self._aggregate_batch_targets(
            rollouts=filtered_rollouts,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        rewards = tuple(transition.reward for transition in transitions)
        baseline_reward = sum(rewards) / max(len(rewards), 1)
        mean_advantage = sum(advantages) / max(len(advantages), 1)
        initial_weights = self._project_track_weights(track=track, n=self.n_z)
        previous_weights = initial_weights
        best_surrogate = -1.0
        best_clip_fraction = 0.0
        best_kl = 0.0
        best_replacement_delta = 0.0
        kl_early_stopped = False
        epochs_executed = 0
        for epoch in range(n_epochs):
            gradient = self._trajectory_gradient(
                transitions=transitions,
                advantages=advantages,
                track=track,
            )
            proposed_weights = tuple(
                _clamp(weight + self._parameter_store.learning_rate * delta * 0.12)
                for weight, delta in zip(previous_weights, gradient, strict=True)
            )
            total = max(sum(proposed_weights), 1e-6)
            normalized_weights = tuple(weight / total for weight in proposed_weights)
            surrogate_objective, clip_fraction, kl_penalty, replacement_effect_delta = self._surrogate_metrics(
                transitions=transitions,
                advantages=advantages,
                old_weights=initial_weights,
                new_weights=normalized_weights,
            )
            epochs_executed = epoch + 1
            best_surrogate = surrogate_objective
            best_clip_fraction = clip_fraction
            best_kl = kl_penalty
            best_replacement_delta = replacement_effect_delta
            if (
                transition_source == "runtime-replay"
                or surrogate_objective >= -0.15
            ):
                self._parameter_store.track_weights[track] = normalized_weights
                previous_weights = normalized_weights
            if kl_penalty > max_kl:
                kl_early_stopped = True
                break
        value_change_norm = self._update_value_head(
            track=track,
            transitions=transitions,
            returns=tuple(transition.return_estimate for transition in transitions),
        )
        action_head_change_norm = self._update_causal_action_head(
            track=track,
            transitions=transitions,
            advantages=advantages,
        )
        self._parameter_store.persistence = _clamp(
            self._parameter_store.persistence
            + mean_advantage * self._parameter_store.learning_rate * 0.03
        )
        self._parameter_store.update_steps[track] += 1
        self._parameter_store.align_temporal_from_tracks()
        parameter_change_norm = self._mean_abs_delta(
            initial_weights,
            self._project_track_weights(track=track, n=self.n_z),
        ) + value_change_norm + action_head_change_norm
        torch_evidence = self._maybe_run_torch_ppo(track=track, transitions=transitions)
        return PolicyBatchResult(
            report=OptimizationReport(
                track=track,
                average_reward=sum(rollout.total_reward for rollout in updated_rollouts)
                / max(sum(len(rollout.transitions) for rollout in updated_rollouts), 1),
                baseline_reward=baseline_reward,
                mean_advantage=mean_advantage,
                surrogate_objective=best_surrogate,
                clip_fraction=best_clip_fraction,
                kl_penalty=best_kl,
                epochs_executed=epochs_executed,
                kl_early_stopped=kl_early_stopped,
                parameters_changed=parameter_change_norm > 1e-6,
                rollout_count=len(updated_rollouts),
                transition_count=len(transitions),
                mean_return=mean_return,
                value_loss=value_loss,
                parameter_change_norm=parameter_change_norm,
                replacement_effect_delta=best_replacement_delta,
                torch_backend=torch_evidence["backend"],
                torch_parameters_changed=torch_evidence["parameters_changed"],
                torch_policy_loss=torch_evidence["policy_loss"],
                torch_value_loss=torch_evidence["value_loss"],
                torch_approx_kl=torch_evidence["approx_kl"],
                torch_wrote_back=torch_evidence["wrote_back"],
                parameter_summary=(
                    f"track={track.value} rollouts={len(updated_rollouts)} transitions={len(transitions)} "
                    f"weights={self._parameter_store.track_weights[track]} "
                    f"persistence={self._parameter_store.persistence:.3f} "
                    f"causal_head_change={action_head_change_norm:.6f} "
                    f"objective={best_surrogate:.3f} value_loss={value_loss:.3f} "
                    f"epochs={epochs_executed}/{n_epochs} kl_stopped={kl_early_stopped} "
                    f"replacement_delta={best_replacement_delta:.3f}"
                ),
            ),
            updated_rollouts=updated_rollouts,
        )

    def _maybe_run_torch_ppo(self, *, track: Track, transitions: tuple[ZTransition, ...]) -> dict:
        """Phase C: real torch PPO over the live transition batch when enabled.

        DISABLED -> no-op. SHADOW -> torch evidence without write-back.
        ACTIVE -> torch refines and writes track weights + critic back into the
        live store/policy (authoritative for the final weights). Requires torch;
        otherwise returns an explicit skipped result.
        """

        disabled = {
            "backend": "disabled", "parameters_changed": 0, "policy_loss": 0.0,
            "value_loss": 0.0, "approx_kl": 0.0, "wrote_back": False,
        }
        if self._rl_backend is WiringLevel.DISABLED:
            return disabled
        if not transitions:
            return {**disabled, "backend": "skipped-no-transitions"}
        if not is_torch_available():
            return {**disabled, "backend": "skipped-no-torch"}
        force = os.environ.get("VZ_TORCH_BACKENDS_FORCE", "").strip().lower() in (
            "1", "true", "on", "yes",
        )
        if (
            not force
            and os.name == "nt"
            and os.environ.get("VZ_SUBSTRATE_DEVICE", "").startswith("cuda")
        ):
            # Mirror the SSL trainer guard (temporal/ssl.py): the float64 CPU
            # torch autograd path spawns background parallel-for worker threads
            # that intermittently trip a native 0xC0000005 on Windows CUDA
            # hosts after sustained load. Skip the torch refinement here; the
            # heuristic pure-policy update remains the live writer and substrate
            # inference stays on GPU. Bypass with VZ_TORCH_BACKENDS_FORCE=1 on
            # a stabilized lane (E-core pinned or patched microcode).
            return {**disabled, "backend": "skipped-windows-cuda"}

        from volvence_zero.internal_rl.torch_causal_ppo import torch_causal_ppo_update

        write_back = self._rl_backend is WiringLevel.ACTIVE
        report = torch_causal_ppo_update(
            parameter_store=self._parameter_store,
            value_weights=self._value_weights,
            value_bias=self._value_bias,
            track=track,
            transitions=transitions,
            n_z=self.n_z,
            write_back=write_back,
            learning_rate=self._parameter_store.learning_rate,
            clip_epsilon=self._parameter_store.clip_epsilon,
            runtime_track_modulation_strength=self._runtime_track_modulation_strength,
            causal_action_head_enabled=(
                self._causal_action_head_wiring
                is not WiringLevel.DISABLED
            ),
            causal_action_head_strength=(
                self._causal_action_head_strength
            ),
        )
        return {
            "backend": self._rl_backend.value,
            "parameters_changed": report.parameters_changed,
            "policy_loss": report.policy_loss,
            "value_loss": report.value_loss,
            "approx_kl": report.approx_kl,
            "wrote_back": report.wrote_back,
        }


class InternalRLSandbox:
    """Minimal z-space rollout sandbox for abstract-action RL experiments."""

    def __init__(
        self,
        *,
        policy: FullLearnedTemporalPolicy | LearnedLiteTemporalPolicy | None = None,
        env: InternalRLEnvironment | None = None,
        residual_runtime: OpenWeightResidualRuntime | None = None,
        rl_backend: WiringLevel = WiringLevel.DISABLED,
    ) -> None:
        self._policy = policy or FullLearnedTemporalPolicy()
        runtime_track_modulation_strength = (
            self._policy.runtime_track_modulation
            if isinstance(self._policy, FullLearnedTemporalPolicy)
            else 0.0
        )
        causal_action_head_wiring = (
            self._policy.causal_action_head_wiring
            if isinstance(self._policy, FullLearnedTemporalPolicy)
            else WiringLevel.DISABLED
        )
        causal_action_head_strength = (
            self._policy.causal_action_head_strength
            if isinstance(self._policy, FullLearnedTemporalPolicy)
            else 0.0
        )
        self._causal_policy = CausalZPolicy(
            parameter_store=self._policy.parameter_store,
            rl_backend=rl_backend,
            runtime_track_modulation_strength=runtime_track_modulation_strength,
            causal_action_head_wiring=causal_action_head_wiring,
            causal_action_head_strength=causal_action_head_strength,
        )
        self._env = env or InternalRLEnvironment()
        self._residual_runtime = residual_runtime
        # Owner-local evidence retention (same pattern as CMS
        # ``latest_cms_backend_evidence``): the most recent optimization
        # report, including torch_* PPO backend fields, stays readable so
        # evidence exporters do not re-run optimization.
        self._latest_optimization_report: (
            DualTrackOptimizationReport | OptimizationReport | None
        ) = None
        self._pending_runtime_capture: RuntimeActionCapture | None = None
        self._runtime_previous_code: tuple[float, ...] = tuple(
            0.0 for _ in range(self._causal_policy.n_z)
        )
        self._runtime_captured_count = 0
        self._runtime_settled_count = 0
        self._runtime_dropped_count = 0
        self._runtime_last_drop_reason = ""

    @property
    def latest_optimization_report(
        self,
    ) -> DualTrackOptimizationReport | OptimizationReport | None:
        """Return the most recent optimization report (evidence readout only)."""

        return self._latest_optimization_report

    def set_rl_backend(self, wiring_level: WiringLevel) -> None:
        """Switch the causal-policy PPO backend (reversible to DISABLED)."""

        self._causal_policy.set_rl_backend(wiring_level)

    @property
    def runtime_replay_checkpoint(self) -> RuntimeReplayCheckpoint:
        """Immutable owner-state readout used only by bounded checkpoints."""

        return RuntimeReplayCheckpoint(
            pending_capture=self._pending_runtime_capture,
            previous_code=self._runtime_previous_code,
            captured_count=self._runtime_captured_count,
            settled_count=self._runtime_settled_count,
            dropped_count=self._runtime_dropped_count,
            last_drop_reason=self._runtime_last_drop_reason,
        )

    def restore_runtime_replay_checkpoint(
        self,
        checkpoint: RuntimeReplayCheckpoint,
    ) -> None:
        if len(checkpoint.previous_code) != self._causal_policy.n_z:
            raise ValueError(
                "runtime replay checkpoint latent dimension mismatch: "
                f"expected={self._causal_policy.n_z}, "
                f"actual={len(checkpoint.previous_code)}"
            )
        self._pending_runtime_capture = checkpoint.pending_capture
        self._runtime_previous_code = checkpoint.previous_code
        self._runtime_captured_count = checkpoint.captured_count
        self._runtime_settled_count = checkpoint.settled_count
        self._runtime_dropped_count = checkpoint.dropped_count
        self._runtime_last_drop_reason = checkpoint.last_drop_reason

    def reset_runtime_replay_for_episode_transfer(self) -> None:
        """Clear episode-local replay while preserving learned policy state."""

        self.restore_runtime_replay_checkpoint(
            RuntimeReplayCheckpoint(
                pending_capture=None,
                previous_code=tuple(
                    0.0 for _ in range(self._causal_policy.n_z)
                ),
                captured_count=0,
                settled_count=0,
                dropped_count=0,
                last_drop_reason="",
            )
        )

    def capture_runtime_action(
        self,
        *,
        turn_index: int,
        track: Track,
        prediction_id: str,
        substrate_snapshot: SubstrateSnapshot,
        temporal_snapshot: TemporalAbstractionSnapshot,
        runtime_state: MetacontrollerRuntimeState,
    ) -> RuntimeActionCapture:
        """Capture one real runtime action before its environment settles."""

        if self._pending_runtime_capture is not None:
            raise RuntimeError(
                "runtime replay capture must be settled or explicitly dropped "
                "before the next action is captured"
            )
        if not prediction_id:
            raise RuntimeReplayLineageError(
                "runtime replay capture requires a PE-owner prediction_id"
            )
        n = self._causal_policy.n_z
        action = tuple(temporal_snapshot.controller_state.code)
        if len(action) != n:
            raise ValueError(
                "runtime replay action latent dimension mismatch: "
                f"expected={n}, actual={len(action)}"
            )
        base_mean = tuple(runtime_state.posterior_mean)
        base_std = tuple(runtime_state.posterior_std)
        policy_noise = tuple(runtime_state.posterior_sample_noise)
        hidden_state = tuple(runtime_state.posterior_hidden_state)
        sampled_candidate = tuple(runtime_state.z_tilde)
        if runtime_state.causal_action_head_wiring != (
            self._causal_policy.causal_action_head_wiring.value
        ):
            raise ValueError(
                "runtime replay causal action head wiring mismatch: "
                f"runtime={runtime_state.causal_action_head_wiring!r}, "
                "sandbox="
                f"{self._causal_policy.causal_action_head_wiring.value!r}"
            )
        action_head_residual = (
            tuple(runtime_state.causal_action_head_residual)
            if runtime_state.causal_action_head_wiring
            == WiringLevel.ACTIVE.value
            else tuple(0.0 for _ in range(n))
        )
        action_head_state = tuple(
            runtime_state.causal_action_head_state
        )
        for name, values in (
            ("posterior_mean", base_mean),
            ("posterior_std", base_std),
            ("posterior_sample_noise", policy_noise),
            ("posterior_hidden_state", hidden_state),
            ("z_tilde", sampled_candidate),
            ("causal_action_head_state", action_head_state),
            ("causal_action_head_residual", action_head_residual),
        ):
            if len(values) != n:
                raise ValueError(
                    f"runtime replay {name} dimension mismatch: "
                    f"expected={n}, actual={len(values)}"
                )
        track_parameters = dict(runtime_state.track_parameters)
        required_tracks = {member.value for member in (Track.WORLD, Track.SELF, Track.SHARED)}
        if set(track_parameters) != required_tracks:
            raise ValueError(
                "runtime replay requires world/self/shared track parameters, "
                f"got={tuple(sorted(track_parameters))}"
            )
        for track_name, values in track_parameters.items():
            if len(values) != n:
                raise ValueError(
                    f"runtime replay track {track_name!r} dimension mismatch: "
                    f"expected={n}, actual={len(values)}"
                )
        other_track_sum = tuple(
            sum(
                track_parameters[other.value][index]
                for other in (Track.WORLD, Track.SELF, Track.SHARED)
                if other is not track
            )
            for index in range(n)
        )
        track_weights = tuple(track_parameters[track.value])
        scalar_beta = float(runtime_state.latest_switch_gate)
        beta_t = tuple(
            max(
                0.0,
                min(
                    1.0,
                    (
                        (action[index] - self._runtime_previous_code[index])
                        / (
                            sampled_candidate[index]
                            - self._runtime_previous_code[index]
                        )
                    )
                    if abs(
                        sampled_candidate[index]
                        - self._runtime_previous_code[index]
                    )
                    > 1e-9
                    else scalar_beta,
                ),
            )
            for index in range(n)
        )
        policy_mean, policy_std = runtime_replay_policy_distribution(
            base_mean=base_mean,
            base_std=base_std,
            previous_code=self._runtime_previous_code,
            beta_t=beta_t,
            track_weights=track_weights,
            other_track_sum=other_track_sum,
            modulation_strength=self._causal_policy.runtime_track_modulation_strength,
            action_head_residual=action_head_residual,
        )
        observation_signature = _surface_signature(substrate_snapshot, n)
        log_prob = self._causal_policy._log_prob(
            policy_action=action,
            policy_mean=policy_mean,
            policy_std=policy_std,
        )
        value_estimate = self._causal_policy._value_estimate(
            track=track,
            hidden_state=hidden_state,
            surface=observation_signature,
        )
        capture = RuntimeActionCapture(
            capture_id=f"runtime:{track.value}:turn-{turn_index}:{prediction_id}",
            turn_index=turn_index,
            track=track,
            prediction_id=prediction_id,
            substrate_snapshot=substrate_snapshot,
            temporal_snapshot=temporal_snapshot,
            runtime_state=runtime_state,
            previous_code=self._runtime_previous_code,
            observation_signature=observation_signature,
            policy_action=action,
            policy_mean=policy_mean,
            policy_std=policy_std,
            policy_noise=policy_noise,
            log_prob=log_prob,
            value_estimate=value_estimate,
            runtime_base_mean=base_mean,
            runtime_base_std=base_std,
            runtime_beta_t=beta_t,
            runtime_other_track_sum=other_track_sum,
            runtime_action_head_state=action_head_state,
            runtime_action_head_residual=action_head_residual,
        )
        self._pending_runtime_capture = capture
        self._runtime_previous_code = action
        self._runtime_captured_count += 1
        return capture

    def settle_runtime_action(
        self,
        *,
        next_substrate_snapshot: SubstrateSnapshot,
        environment_outcome: EnvironmentOutcome | None,
        prediction_error_snapshot: PredictionErrorSnapshot | None,
        credit_snapshot: CreditSnapshot | None,
        prediction_error_reward_enabled: bool = True,
    ) -> RuntimeReplaySettlement:
        """Settle the prior action using only matching PE-owner evidence.

        ``PredictionError.signed_reward`` is a residual
        (actual-minus-predicted), not the realized utility of the action.  A
        policy trained on that residual would stop reinforcing a beneficial
        action as soon as the predictor learned to expect its payoff.  Runtime
        replay therefore optimizes the PE owner's typed
        ``ActualOutcome.action_payoff`` and retains signed PE as a diagnostic
        component.  The environment measurement is never consumed directly as
        optimizer reward here.
        """

        capture = self._pending_runtime_capture
        if capture is None:
            return RuntimeReplaySettlement(
                capture_id="",
                rollout=None,
                lineage_matched=False,
                drop_reason="no-pending-capture",
            )
        if environment_outcome is None:
            self._pending_runtime_capture = None
            self._runtime_dropped_count += 1
            self._runtime_last_drop_reason = "missing-environment-outcome"
            return RuntimeReplaySettlement(
                capture_id=capture.capture_id,
                rollout=None,
                lineage_matched=False,
                drop_reason=self._runtime_last_drop_reason,
            )
        if prediction_error_snapshot is None or credit_snapshot is None:
            raise RuntimeReplayLineageError(
                "runtime replay settlement with an EnvironmentOutcome requires "
                "matching prediction_error and credit snapshots"
            )
        evaluated_prediction = prediction_error_snapshot.evaluated_prediction
        action_context = prediction_error_snapshot.action_context
        lineage_values = {
            "environment_outcome.prediction_id": environment_outcome.prediction_id or "",
            "prediction_error.evaluated_prediction.prediction_id": (
                evaluated_prediction.prediction_id
                if evaluated_prediction is not None
                else ""
            ),
            "prediction_error.action_context.prediction_id": action_context.prediction_id,
        }
        mismatched = tuple(
            f"{name}={value!r}"
            for name, value in lineage_values.items()
            if value != capture.prediction_id
        )
        if action_context.environment_outcome_id != environment_outcome.outcome_id:
            mismatched = mismatched + (
                "prediction_error.action_context.environment_outcome_id="
                f"{action_context.environment_outcome_id!r}",
            )
        if mismatched:
            raise RuntimeReplayLineageError(
                "runtime replay lineage mismatch for "
                f"{capture.capture_id}: expected prediction_id="
                f"{capture.prediction_id!r}, outcome_id="
                f"{environment_outcome.outcome_id!r}; actual={mismatched}"
            )
        segment_records = tuple(
            record
            for record in credit_snapshot.recent_credits
            if (
                record.level == "abstract_action_segment"
                and action_context.segment_id
                and record.source_event == f"segment:{action_context.segment_id}"
            )
        )
        segment_bonus = 0.0
        if prediction_error_reward_enabled and segment_records:
            segment_bonus = _clamp(
                (
                    sum(record.credit_value for record in segment_records)
                    / len(segment_records)
                )
                * 0.1
                * (1.0 + max(0.0, prediction_error_snapshot.error.magnitude))
            )
        realized_action_payoff = (
            _clamp(prediction_error_snapshot.actual_outcome.action_payoff)
            if prediction_error_reward_enabled
            else 0.0
        )
        pe_residual = _clamp(
            prediction_error_snapshot.error.signed_reward
        )
        reward = _clamp(realized_action_payoff + segment_bonus)
        next_signature = _surface_signature(
            next_substrate_snapshot,
            self._causal_policy.n_z,
        )
        downstream_effect = tuple(
            _clamp(next_signature[index] - capture.observation_signature[index])
            for index in range(self._causal_policy.n_z)
        )
        reward_components = [
            ("realized_action_payoff", realized_action_payoff),
            ("prediction_error_residual", pe_residual),
        ]
        if abs(segment_bonus) > 1e-12:
            reward_components.append(("abstract_action_credit", segment_bonus))
        runtime_state = capture.runtime_state
        measurement = environment_outcome.measurement
        transition = ZTransition(
            step_index=0,
            track=capture.track,
            abstract_action=capture.temporal_snapshot.active_abstract_action,
            controller_state=capture.temporal_snapshot.controller_state,
            observation_signature=capture.observation_signature,
            policy_action=capture.policy_action,
            latent_code=capture.policy_action,
            decoder_output=runtime_state.decoder_control,
            applied_control=runtime_state.decoder_applied_control,
            downstream_effect=downstream_effect,
            hidden_state=runtime_state.posterior_hidden_state,
            policy_score=runtime_state.policy_replacement_score,
            log_prob=capture.log_prob,
            reward=reward,
            raw_reward=realized_action_payoff,
            policy_replacement_quality=1.0,
            backend_name="runtime-replay",
            backend_fidelity=1.0,
            policy_mean=capture.policy_mean,
            policy_std=capture.policy_std,
            policy_noise=capture.policy_noise,
            value_estimate=capture.value_estimate,
            replacement_effect_delta=self._mean_abs_tuple(
                capture.observation_signature,
                next_signature,
            ),
            reward_components=tuple(reward_components),
            reward_mode="runtime-replay",
            active_family_id=runtime_state.active_label,
            transition_source="runtime-replay",
            runtime_turn_index=capture.turn_index,
            prediction_id=capture.prediction_id,
            environment_outcome_id=environment_outcome.outcome_id,
            runtime_base_mean=capture.runtime_base_mean,
            runtime_base_std=capture.runtime_base_std,
            runtime_previous_code=capture.previous_code,
            runtime_beta_t=capture.runtime_beta_t,
            runtime_other_track_sum=capture.runtime_other_track_sum,
            runtime_action_head_residual=(
                capture.runtime_action_head_residual
            ),
            lineage_matched=True,
            runtime_segment_id=action_context.segment_id,
            runtime_terminal=(
                measurement.terminal if measurement is not None else False
            ),
            runtime_milestone=(
                measurement is not None
                and measurement.task_progress is not None
            ),
            runtime_action_head_state=(
                capture.runtime_action_head_state
            ),
        )
        rollout = ZRollout(
            rollout_id=(
                f"{capture.capture_id}:outcome:{environment_outcome.outcome_id}"
            ),
            track=capture.track,
            transitions=(transition,),
            total_reward=reward,
            description=(
                "Real runtime replay settled from next substrate dynamics and "
                "PE-owner realized utility plus typed segment credit for "
                f"prediction {capture.prediction_id}."
            ),
            replacement_mode="runtime-replay",
            reward_mode="runtime-replay",
        )
        self._pending_runtime_capture = None
        self._runtime_settled_count += 1
        self._runtime_last_drop_reason = ""
        return RuntimeReplaySettlement(
            capture_id=capture.capture_id,
            rollout=rollout,
            lineage_matched=True,
            environment_outcome_id=environment_outcome.outcome_id,
        )

    def observe_runtime_transition(
        self,
        *,
        turn_index: int,
        track: Track,
        prediction_id: str,
        substrate_snapshot: SubstrateSnapshot,
        temporal_snapshot: TemporalAbstractionSnapshot,
        runtime_state: MetacontrollerRuntimeState,
        environment_outcome: EnvironmentOutcome | None,
        prediction_error_snapshot: PredictionErrorSnapshot | None,
        credit_snapshot: CreditSnapshot | None,
        prediction_error_reward_enabled: bool = True,
    ) -> tuple[RuntimeReplaySettlement, RuntimeActionCapture]:
        """Settle the previous action, then capture the current runtime action."""

        settlement = self.settle_runtime_action(
            next_substrate_snapshot=substrate_snapshot,
            environment_outcome=environment_outcome,
            prediction_error_snapshot=prediction_error_snapshot,
            credit_snapshot=credit_snapshot,
            prediction_error_reward_enabled=prediction_error_reward_enabled,
        )
        capture = self.capture_runtime_action(
            turn_index=turn_index,
            track=track,
            prediction_id=prediction_id,
            substrate_snapshot=substrate_snapshot,
            temporal_snapshot=temporal_snapshot,
            runtime_state=runtime_state,
        )
        return settlement, capture

    @property
    def policy(self) -> FullLearnedTemporalPolicy | LearnedLiteTemporalPolicy:
        return self._policy

    @property
    def causal_policy(self) -> CausalZPolicy:
        return self._causal_policy

    @property
    def latest_reward_composition(self) -> dict[str, object] | None:
        """The environment's last-step reward composition readout (CP-07)."""

        return self._env.latest_reward_composition

    def configure_runtime_backend(self, *, source_text: str | None) -> None:
        if self._residual_runtime is None or not source_text:
            return
        self._env.use_open_weight_runtime(
            runtime=self._residual_runtime,
            source_text=source_text,
        )

    def ingest_temporal_fast_prior(
        self,
        rollouts: ZRollout | tuple[ZRollout, ...],
        *,
        enabled: bool = True,
    ) -> tuple[float, float, float, float, float]:
        normalized_rollouts = rollouts if isinstance(rollouts, tuple) else (rollouts,)
        if not enabled or not normalized_rollouts:
            self._policy.parameter_store.record_fast_prior_signals(
                strength=0.0,
                action_bias=0.0,
                family_bias=0.0,
                sequence_bias=0.0,
                switch_pressure_delta=0.0,
            )
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        self._apply_family_outcome_feedbacks(normalized_rollouts)
        credit_alignment = self._delayed_credit_alignment(normalized_rollouts)
        terminal_success_rate = sum(float(rollout.terminal_success) for rollout in normalized_rollouts) / len(normalized_rollouts)
        family_assignment_rate = sum(
            (
                sum(1.0 for family_id in rollout.completed_family_ids if family_id != "unassigned")
                / max(len(rollout.completed_family_ids), 1)
            )
            if rollout.completed_family_ids
            else 0.0
            for rollout in normalized_rollouts
        ) / len(normalized_rollouts)
        sequence_completion_rate = sum(
            (
                len(rollout.completed_subgoals) / max(len(rollout.delayed_credit_assignments), 1)
                if rollout.delayed_credit_assignments
                else float(rollout.terminal_success)
            )
            for rollout in normalized_rollouts
        ) / len(normalized_rollouts)
        strength = max(
            0.0,
            min(
                1.0,
                credit_alignment * 0.40
                + family_assignment_rate * 0.25
                + terminal_success_rate * 0.20
                + sequence_completion_rate * 0.15,
            ),
        )
        action_bias = max(
            -1.0,
            min(
                1.0,
                (family_assignment_rate - 0.5) * 0.50
                + (credit_alignment - 0.5) * 0.30
                + (terminal_success_rate - 0.5) * 0.20,
            ),
        )
        family_bias = max(
            -1.0,
            min(
                1.0,
                (credit_alignment - 0.5) * 0.45
                + (sequence_completion_rate - 0.5) * 0.35
                + (family_assignment_rate - 0.5) * 0.20,
            ),
        )
        sequence_bias = max(
            -1.0,
            min(
                1.0,
                (sequence_completion_rate - 0.5) * 0.55
                + (terminal_success_rate - 0.5) * 0.25
                + (credit_alignment - 0.5) * 0.20,
            ),
        )
        switch_pressure_delta = max(
            -0.18,
            min(
                0.18,
                -(
                    action_bias * 0.35
                    + family_bias * 0.40
                    + sequence_bias * 0.25
                )
                * max(strength, 0.2),
            ),
        )
        self._policy.parameter_store.record_fast_prior_signals(
            strength=strength,
            action_bias=action_bias,
            family_bias=family_bias,
            sequence_bias=sequence_bias,
            switch_pressure_delta=switch_pressure_delta,
        )
        return (strength, action_bias, family_bias, sequence_bias, switch_pressure_delta)

    def _apply_family_outcome_feedbacks(
        self,
        rollouts: tuple[ZRollout, ...],
    ) -> tuple[FamilyOutcomeFeedback, ...]:
        feedbacks = self._family_outcome_feedbacks_from_rollouts(rollouts)
        for feedback in feedbacks:
            self._policy.parameter_store.observe_family_outcome_feedback(feedback=feedback)
        return feedbacks

    def _family_outcome_feedbacks_from_rollouts(
        self,
        rollouts: tuple[ZRollout, ...],
    ) -> tuple[FamilyOutcomeFeedback, ...]:
        family_rewards: dict[str, list[float]] = {}
        family_session_payoffs: dict[str, list[float]] = {}
        family_delayed_credits: dict[str, list[float]] = {}
        for rollout in rollouts:
            valid_transitions = tuple(
                transition
                for transition in rollout.transitions
                if transition.active_family_id not in {None, "unassigned"}
            )
            if not valid_transitions:
                continue
            rollout_family_ids = tuple(dict.fromkeys(
                transition.active_family_id for transition in valid_transitions if transition.active_family_id is not None
            ))
            session_payoff_share = _clamp(rollout.total_reward / max(len(rollout_family_ids), 1))
            for family_id in rollout_family_ids:
                family_session_payoffs.setdefault(family_id, []).append(session_payoff_share)
            for transition in valid_transitions:
                family_rewards.setdefault(transition.active_family_id or "unassigned", []).append(
                    _clamp(transition.reward)
                )
            for assignment in rollout.delayed_credit_assignments:
                start, end = self._credit_assignment_window_bounds(
                    transitions=rollout.transitions,
                    assignment=assignment,
                )
                if end < start:
                    continue
                window_family_ids = tuple(dict.fromkeys(
                    transition.active_family_id
                    for transition in rollout.transitions[start : end + 1]
                    if transition.active_family_id not in {None, "unassigned"}
                ))
                if not window_family_ids:
                    continue
                distributed_credit = _clamp(assignment.reward / len(window_family_ids))
                for family_id in window_family_ids:
                    family_delayed_credits.setdefault(family_id, []).append(distributed_credit)
        feedbacks: list[FamilyOutcomeFeedback] = []
        for family_id in sorted(set(family_rewards) | set(family_session_payoffs) | set(family_delayed_credits)):
            rewards = family_rewards.get(family_id, [])
            session_payoffs = family_session_payoffs.get(family_id, [])
            delayed_credits = family_delayed_credits.get(family_id, [])
            outcome_value = _clamp(sum(rewards) / len(rewards)) if rewards else 0.0
            delayed_credit_delta = _clamp(sum(delayed_credits) / len(delayed_credits)) if delayed_credits else 0.0
            session_payoff_delta = (
                _clamp(sum(session_payoffs) / len(session_payoffs)) if session_payoffs else 0.0
            )
            if (
                abs(outcome_value) <= 1e-8
                and abs(delayed_credit_delta) <= 1e-8
                and abs(session_payoff_delta) <= 1e-8
            ):
                continue
            feedbacks.append(
                FamilyOutcomeFeedback(
                    family_id=family_id,
                    outcome_value=outcome_value,
                    delayed_credit_delta=delayed_credit_delta,
                    session_payoff_delta=session_payoff_delta,
                    credit_record_count=len(delayed_credits),
                    description=(
                        f"proof rollouts={len(rollouts)} family={family_id} "
                        f"reward={outcome_value:.3f} delayed={delayed_credit_delta:.3f} "
                        f"session={session_payoff_delta:.3f}"
                    ),
                )
            )
        return tuple(feedbacks)

    def _delayed_credit_alignment(self, rollouts: tuple[ZRollout, ...]) -> float:
        if not rollouts:
            return 0.0
        aligned_scores: list[float] = []
        for rollout in rollouts:
            if not rollout.delayed_credit_assignments:
                aligned_scores.append(0.0)
                continue
            aligned = 0.0
            for assignment in rollout.delayed_credit_assignments:
                start, end = self._credit_assignment_window_bounds(
                    transitions=rollout.transitions,
                    assignment=assignment,
                )
                if end < start:
                    continue
                if assignment.reason == "terminal-success":
                    matched = any(
                        transition.proof_terminal_success
                        for transition in rollout.transitions[start : end + 1]
                    )
                else:
                    matched = any(
                        transition.proof_subgoal_id == assignment.subgoal_id
                        and transition.active_family_id not in {None, "unassigned"}
                        for transition in rollout.transitions[start : end + 1]
                    )
                if matched:
                    aligned += 1.0
            aligned_scores.append(aligned / max(len(rollout.delayed_credit_assignments), 1))
        return sum(aligned_scores) / len(aligned_scores)

    def _abstract_action_windows(self, transitions: tuple[ZTransition, ...]) -> tuple[tuple[int, int], ...]:
        if not transitions:
            return ()
        windows: list[tuple[int, int]] = []
        start = 0
        for index, transition in enumerate(transitions):
            if index > 0 and transition.controller_state.switch_gate >= 0.5:
                windows.append((start, index - 1))
                start = index
        windows.append((start, len(transitions) - 1))
        return tuple(windows)

    def _credit_assignment_window_bounds(
        self,
        *,
        transitions: tuple[ZTransition, ...],
        assignment: InternalRLDelayedCreditAssignment,
    ) -> tuple[int, int]:
        if not transitions:
            return (0, -1)
        assignment_start = max(0, assignment.start_step)
        assignment_end = min(len(transitions) - 1, assignment.end_step)
        if assignment_end < assignment_start:
            return (0, -1)
        overlapping_windows = tuple(
            (start, end)
            for start, end in self._abstract_action_windows(transitions)
            if end >= assignment_start and start <= assignment_end
        )
        if not overlapping_windows:
            return (assignment_start, assignment_end)
        return (
            min(start for start, _ in overlapping_windows),
            max(end for _, end in overlapping_windows),
        )

    def _proof_optimizer_visible_reward(self, transition: ZTransition) -> float:
        excluded_components = {
            "proof_subgoal_complete",
            "proof_terminal_success",
            "proof_observation_alignment",
            "proof_intervention_effect",
            "proof_subgoal_progress",
        }
        return sum(
            value
            for component_name, value in transition.reward_components
            if component_name not in excluded_components
        )

    def _apply_delayed_credit_assignments(
        self,
        *,
        transitions: tuple[ZTransition, ...],
        assignments: tuple[InternalRLDelayedCreditAssignment, ...],
    ) -> tuple[ZTransition, ...]:
        if not transitions or not assignments:
            return transitions
        adjusted_rewards = [
            self._proof_optimizer_visible_reward(transition)
            if transition.reward_mode.startswith("proof")
            else transition.raw_reward
            for transition in transitions
        ]
        for assignment in assignments:
            start, end = self._credit_assignment_window_bounds(
                transitions=transitions,
                assignment=assignment,
            )
            if end < start:
                continue
            span = end - start + 1
            distributed_reward = assignment.reward / max(span, 1)
            for step_index in range(start, end + 1):
                adjusted_rewards[step_index] += distributed_reward
        return tuple(
            replace(transition, reward=_clamp(adjusted_rewards[index]))
            for index, transition in enumerate(transitions)
        )

    def rollout(
        self,
        *,
        rollout_id: str,
        substrate_steps: tuple[SubstrateSnapshot, ...],
        track: Track = Track.SHARED,
        replacement_mode: str = "causal",
        proof_episode: InternalRLProofEpisode | None = None,
        source_text_by_step: tuple[str, ...] = (),
    ) -> ZRollout:
        if replacement_mode in {"causal", "causal-binary"}:
            self._policy.parameter_store.require_causal_takeover_phase(
                operation=f"InternalRLSandbox.rollout[{replacement_mode}]"
            )
        previous_snapshot: TemporalAbstractionSnapshot | None = None
        transitions: list[ZTransition] = []
        policy_state = self._causal_policy.initial_state(track=track)
        proof_progress: InternalRLProofProgress | None = None
        for step_index, substrate_snapshot in enumerate(substrate_steps):
            if source_text_by_step and step_index < len(source_text_by_step):
                self.configure_runtime_backend(source_text=source_text_by_step[step_index])
            (
                policy_state,
                observation_signature,
                hidden_state,
                policy_action,
                policy_mean,
                policy_std,
                policy_noise,
                policy_score,
                log_prob,
                value_estimate,
            ) = self._causal_policy.step(
                substrate_snapshot=substrate_snapshot,
                state=policy_state,
                observation_mode="proof" if proof_episode is not None else "default",
            )
            env_step = self._env.step(
                substrate_snapshot=substrate_snapshot,
                track=track,
                policy=self._policy,
                previous_snapshot=previous_snapshot,
                policy_latent_override=policy_action if replacement_mode in {"causal", "causal-binary"} else None,
                policy_replacement_score=policy_score if replacement_mode in {"causal", "causal-binary"} else 0.0,
                binary_gate_override=replacement_mode == "causal-binary",
                step_index=step_index,
                is_terminal_step=step_index == len(substrate_steps) - 1,
                proof_episode=proof_episode,
                proof_progress=proof_progress,
            )
            transitions.append(
                ZTransition(
                    step_index=step_index,
                    track=track,
                    abstract_action=env_step.temporal_step.active_abstract_action,
                    controller_state=env_step.temporal_step.controller_state,
                    observation_signature=observation_signature,
                    policy_action=policy_action,
                    latent_code=env_step.latent_code,
                    decoder_output=env_step.decoder_output,
                    applied_control=env_step.applied_control,
                    downstream_effect=env_step.downstream_effect,
                    hidden_state=hidden_state,
                    policy_mean=policy_mean,
                    policy_std=policy_std,
                    policy_noise=policy_noise,
                    policy_score=policy_score,
                    log_prob=log_prob,
                    value_estimate=value_estimate,
                    reward=env_step.reward,
                    raw_reward=env_step.reward,
                    policy_replacement_quality=env_step.policy_replacement_quality,
                    replacement_effect_delta=self._mean_abs_tuple(
                        env_step.applied_control,
                        env_step.downstream_effect,
                    ),
                    backend_name=env_step.backend_name,
                    backend_fidelity=env_step.backend_fidelity,
                    reward_components=env_step.reward_components,
                    reward_mode=env_step.reward_mode,
                    proof_subgoal_id=env_step.proof_subgoal_id,
                    proof_subgoal_score=env_step.proof_subgoal_score,
                    proof_subgoal_completed=env_step.proof_subgoal_completed,
                    proof_terminal_success=env_step.proof_terminal_success,
                    active_family_id=env_step.active_family_id,
                )
            )
            previous_snapshot = env_step.next_previous_snapshot
            proof_progress = env_step.proof_progress
        rollout_transitions = tuple(transitions)
        if proof_progress is not None and proof_progress.delayed_credit_assignments:
            rollout_transitions = self._apply_delayed_credit_assignments(
                transitions=rollout_transitions,
                assignments=proof_progress.delayed_credit_assignments,
            )
        total_reward = sum(transition.reward for transition in rollout_transitions)
        return ZRollout(
            rollout_id=rollout_id,
            track=track,
            transitions=rollout_transitions,
            total_reward=total_reward,
            replacement_mode=replacement_mode,
            reward_mode="proof-delayed" if proof_progress is not None else "dense",
            proof_episode_id=proof_episode.episode_id if proof_episode is not None else None,
            completed_subgoals=proof_progress.completed_subgoals if proof_progress is not None else (),
            completed_family_ids=proof_progress.completed_family_ids if proof_progress is not None else (),
            terminal_success=proof_progress.terminal_success if proof_progress is not None else False,
            delayed_credit_assignments=proof_progress.delayed_credit_assignments if proof_progress is not None else (),
            description=(
                f"Internal RL rollout mode={replacement_mode} reward_mode="
                f"{'proof-delayed' if proof_progress is not None else 'dense'} "
                f"track={track.value} over {len(rollout_transitions)} abstract actions "
                f"with total reward {total_reward:.2f}."
            ),
        )

    def _mean_abs_tuple(
        self,
        left: tuple[float, ...],
        right: tuple[float, ...],
    ) -> float:
        if not left or not right:
            return 0.0
        count = min(len(left), len(right))
        return sum(abs(left[index] - right[index]) for index in range(count)) / max(count, 1)

    def rollout_dual_track(
        self,
        *,
        rollout_id: str,
        substrate_steps: tuple[SubstrateSnapshot, ...],
        proof_episode: InternalRLProofEpisode | None = None,
    ) -> DualTrackRollout:
        task_rollout = self.rollout(
            rollout_id=f"{rollout_id}:task",
            substrate_steps=substrate_steps,
            track=Track.WORLD,
            replacement_mode="causal-binary",
            proof_episode=proof_episode,
        )
        relationship_rollout = self.rollout(
            rollout_id=f"{rollout_id}:relationship",
            substrate_steps=substrate_steps,
            track=Track.SELF,
            replacement_mode="causal-binary",
            proof_episode=proof_episode,
        )
        return DualTrackRollout(
            task_rollout=task_rollout,
            relationship_rollout=relationship_rollout,
            description=(
                f"Dual-track rollout task_reward={task_rollout.total_reward:.2f}, "
                f"relationship_reward={relationship_rollout.total_reward:.2f}."
            ),
        )

    def create_checkpoint(
        self,
        *,
        checkpoint_id: str,
        include_runtime_replay: bool = False,
    ) -> CausalPolicyCheckpoint:
        checkpoint = self._causal_policy.create_checkpoint(
            checkpoint_id=checkpoint_id
        )
        if not include_runtime_replay:
            return checkpoint
        return replace(
            checkpoint,
            runtime_replay=self.runtime_replay_checkpoint,
        )

    def restore_checkpoint(self, checkpoint: CausalPolicyCheckpoint) -> None:
        self._causal_policy.restore_checkpoint(checkpoint)
        if checkpoint.runtime_replay is not None:
            self.restore_runtime_replay_checkpoint(checkpoint.runtime_replay)

    def optimize(
        self,
        rollout: ZRollout | DualTrackRollout | tuple[ZRollout, ...] | tuple[DualTrackRollout, ...],
    ) -> DualTrackOptimizationReport | OptimizationReport:
        self._policy.parameter_store.require_causal_takeover_phase(
            operation="InternalRLSandbox.optimize"
        )
        if isinstance(rollout, tuple):
            if not rollout:
                return OptimizationReport(
                    track=Track.SHARED,
                    average_reward=0.0,
                    baseline_reward=0.0,
                    mean_advantage=0.0,
                    surrogate_objective=0.0,
                    clip_fraction=0.0,
                    kl_penalty=0.0,
                    parameter_summary="no-op",
                    transition_count=0,
                    rollout_count=0,
                )
            if isinstance(rollout[0], DualTrackRollout):
                return self._optimize_dual_track(rollout).optimization_report
            single_report = self._optimize_single(rollout)
            self._latest_optimization_report = single_report
            return single_report
        if isinstance(rollout, DualTrackRollout):
            return self._optimize_dual_track(rollout).optimization_report
        single_report = self._optimize_single(rollout)
        self._latest_optimization_report = single_report
        return single_report

    def optimize_with_audit(
        self,
        rollout: DualTrackRollout | tuple[DualTrackRollout, ...],
        *,
        timestamp_ms: int = 0,
    ) -> PolicyOptimizationResult:
        self._policy.parameter_store.require_causal_takeover_phase(
            operation="InternalRLSandbox.optimize_with_audit"
        )
        return self._optimize_dual_track(rollout, timestamp_ms=timestamp_ms)

    def _optimize_dual_track(
        self,
        rollout: DualTrackRollout | tuple[DualTrackRollout, ...],
        *,
        timestamp_ms: int = 0,
    ) -> PolicyOptimizationResult:
        normalized_rollouts = rollout if isinstance(rollout, tuple) else (rollout,)
        before_hash = stable_value_hash(self._causal_policy.export_parameters())
        task_report = self._optimize_single(tuple(item.task_rollout for item in normalized_rollouts))
        relationship_report = self._optimize_single(
            tuple(item.relationship_rollout for item in normalized_rollouts)
        )
        after_hash = stable_value_hash(self._causal_policy.export_parameters())
        params_changed = before_hash != after_hash
        dual_report = DualTrackOptimizationReport(
            task_report=task_report,
            relationship_report=relationship_report,
            description=(
                f"task_adv={task_report.mean_advantage:.3f}, "
                f"rel_adv={relationship_report.mean_advantage:.3f}, "
                f"task_rollouts={task_report.rollout_count}, rel_rollouts={relationship_report.rollout_count}"
            ),
        )
        self._latest_optimization_report = dual_report
        total_kl = task_report.kl_penalty + relationship_report.kl_penalty
        total_epochs = task_report.epochs_executed + relationship_report.epochs_executed
        records: list[SelfModificationRecord] = []
        if params_changed:
            records.append(SelfModificationRecord(
                target="causal_policy.track_weights",
                gate=ModificationGate.ONLINE,
                decision=GateDecision.ALLOW,
                old_value_hash=before_hash,
                new_value_hash=after_hash,
                justification=(
                    f"RL policy update: task_obj={task_report.surrogate_objective:.3f} "
                    f"rel_obj={relationship_report.surrogate_objective:.3f} "
                    f"kl={total_kl:.3f} epochs={total_epochs}"
                ),
                timestamp_ms=timestamp_ms,
                is_reversible=True,
            ))
        return PolicyOptimizationResult(
            optimization_report=dual_report,
            modification_records=tuple(records),
            policy_update_applied=params_changed,
            total_kl_divergence=total_kl,
            total_epochs_executed=total_epochs,
        )

    def _optimize_single(self, rollout: ZRollout | tuple[ZRollout, ...]) -> OptimizationReport:
        normalized_rollouts = rollout if isinstance(rollout, tuple) else (rollout,)
        filtered_rollouts = tuple(item for item in normalized_rollouts if item.transitions)
        if not filtered_rollouts:
            return OptimizationReport(
                track=normalized_rollouts[0].track if normalized_rollouts else Track.SHARED,
                average_reward=0.0,
                baseline_reward=0.0,
                mean_advantage=0.0,
                surrogate_objective=0.0,
                clip_fraction=0.0,
                kl_penalty=0.0,
                parameter_summary="no-op",
                parameters_changed=False,
            )
        primary_rollout = filtered_rollouts[0]
        average_reward = sum(item.total_reward for item in filtered_rollouts) / max(
            sum(len(item.transitions) for item in filtered_rollouts),
            1,
        )
        reward_scale = max(average_reward, 0.05)
        before_hash = stable_value_hash(
            (
                self._causal_policy.export_parameters(),
                self._policy.export_parameters(),
            )
        )
        batch_result = self._causal_policy.optimize(rollouts=filtered_rollouts)
        causal_report = batch_result.report
        if primary_rollout.track is Track.WORLD:
            self._policy.fit_from_signals(
                residual_strength=max(0.45, reward_scale),
                memory_strength=0.20,
                reflection_strength=0.15,
            )
            after_hash = stable_value_hash(
                (
                    self._causal_policy.export_parameters(),
                    self._policy.export_parameters(),
                )
            )
            return replace(
                causal_report,
                parameters_changed=before_hash != after_hash,
            )
        if primary_rollout.track is Track.SELF:
            self._policy.fit_from_signals(
                residual_strength=0.35,
                memory_strength=0.25,
                reflection_strength=max(0.25, reward_scale),
            )
            after_hash = stable_value_hash(
                (
                    self._causal_policy.export_parameters(),
                    self._policy.export_parameters(),
                )
            )
            return replace(
                causal_report,
                parameters_changed=before_hash != after_hash,
            )
        self._policy.fit_from_signals(
            residual_strength=max(0.4, reward_scale),
            memory_strength=0.25,
            reflection_strength=0.35 if average_reward > 0.4 else 0.2,
        )
        after_hash = stable_value_hash(
            (
                self._causal_policy.export_parameters(),
                self._policy.export_parameters(),
            )
        )
        return replace(
            causal_report,
            parameters_changed=before_hash != after_hash,
        )


    def _reward_for_step(
        self,
        *,
        controller_state: ControllerState,
        hidden_state: tuple[float, ...],
        track: Track,
    ) -> float:
        n = len(controller_state.code)
        reward = sum(controller_state.code) / max(n, 1)
        reward += sum(hidden_state) / max(len(hidden_state), 1) * 0.2
        if controller_state.is_switching:
            reward += 0.1
        reward -= controller_state.steps_since_switch * 0.02
        if track is Track.WORLD and len(hidden_state) > 0:
            reward += hidden_state[0] * 0.15
        elif track is Track.SELF and len(hidden_state) > 1:
            reward += hidden_state[1] * 0.15
        return _clamp(reward)


def derive_abstract_action_credit(
    *,
    rollout: ZRollout,
    timestamp_ms: int,
) -> tuple[CreditRecord, ...]:
    records: list[CreditRecord] = []
    for transition in rollout.transitions:
        records.append(
            CreditRecord(
                record_id=f"{rollout.rollout_id}:{transition.step_index}",
                level="abstract_action",
                track=transition.track,
                source_event=transition.abstract_action,
                credit_value=_clamp(transition.reward),
                context=rollout.description,
                timestamp_ms=timestamp_ms + transition.step_index,
            )
        )
    return tuple(records)
