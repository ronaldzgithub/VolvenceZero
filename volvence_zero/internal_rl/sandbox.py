from __future__ import annotations

from dataclasses import dataclass, replace
import math

from volvence_zero.credit.gate import CreditRecord, GateDecision, ModificationGate, SelfModificationRecord
from volvence_zero.internal_rl.environment import (
    InternalRLDelayedCreditAssignment,
    InternalRLEnvStep,
    InternalRLEnvironment,
    InternalRLProofEpisode,
    InternalRLProofProgress,
)
from volvence_zero.memory import Track
from volvence_zero.runtime.kernel import stable_value_hash
from volvence_zero.substrate import OpenWeightResidualRuntime, SubstrateSnapshot
from volvence_zero.temporal import (
    ControllerState,
    FullLearnedTemporalPolicy,
    LearnedLiteTemporalPolicy,
    MetacontrollerParameterSnapshot,
    MetacontrollerParameterStore,
    TemporalAbstractionSnapshot,
)
from volvence_zero.temporal.metacontroller_components import (
    _project_to_ndim,
    residual_sequence_from_snapshot,
    summarize_residual_activations,
)


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


@dataclass(frozen=True)
class CausalPolicyCheckpoint:
    checkpoint_id: str
    parameters_by_track: tuple[CausalPolicyParameters, ...]
    metacontroller_snapshot: MetacontrollerParameterSnapshot


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

    def __init__(self, *, parameter_store: MetacontrollerParameterStore) -> None:
        self._parameter_store = parameter_store
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
            hidden_state=hidden_state,
            surface=surface,
            previous_action=state.previous_action,
            weights=weights,
        )
        policy_std = self._policy_std(
            hidden_state=hidden_state,
            surface=surface,
            previous_action=state.previous_action,
            policy_mean=policy_mean,
        )
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
        return CausalPolicyCheckpoint(
            checkpoint_id=checkpoint_id,
            parameters_by_track=self.export_parameters(),
            metacontroller_snapshot=self._parameter_store.export_parameter_snapshot(),
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
        hidden_state: tuple[float, ...],
        surface: tuple[float, ...],
        previous_action: tuple[float, ...],
        weights: tuple[float, ...],
    ) -> tuple[float, ...]:
        return tuple(
            _clamp(
                hidden_state[index] * 0.40
                + surface[index] * 0.25
                + previous_action[index] * 0.15
                + weights[index] * 0.20
            )
            for index in range(len(hidden_state))
        )

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
            for index, value in enumerate(transition.observation_signature):
                variance = max(transition.policy_std[index] ** 2, 1e-6)
                score_term = (transition.policy_action[index] - transition.policy_mean[index]) / variance
                accum[index] += (
                    score_term
                    * max(value * 0.6 + transition.hidden_state[index] * 0.4, 1e-3)
                    * (0.55 + track_weights[index] * 0.45)
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
            new_mean = self._policy_mean(
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
            old_mean = self._policy_mean(
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
                next_value = values[index + 1] if index + 1 < n else 0.0
                delta = rollout.transitions[index].reward + gamma * next_value - values[index]
                last_gae = delta + gamma * gae_lambda * last_gae
                raw_advantages[index] = last_gae
                returns[index] = raw_advantages[index] + values[index]
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
            if surrogate_objective >= -0.15:
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
        self._parameter_store.persistence = _clamp(
            self._parameter_store.persistence
            + mean_advantage * self._parameter_store.learning_rate * 0.03
        )
        self._parameter_store.update_steps[track] += 1
        self._parameter_store.align_temporal_from_tracks()
        parameter_change_norm = self._mean_abs_delta(
            initial_weights,
            self._project_track_weights(track=track, n=self.n_z),
        ) + value_change_norm
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
                parameter_summary=(
                    f"track={track.value} rollouts={len(updated_rollouts)} transitions={len(transitions)} "
                    f"weights={self._parameter_store.track_weights[track]} "
                    f"persistence={self._parameter_store.persistence:.3f} "
                    f"objective={best_surrogate:.3f} value_loss={value_loss:.3f} "
                    f"epochs={epochs_executed}/{n_epochs} kl_stopped={kl_early_stopped} "
                    f"replacement_delta={best_replacement_delta:.3f}"
                ),
            ),
            updated_rollouts=updated_rollouts,
        )


class InternalRLSandbox:
    """Minimal z-space rollout sandbox for abstract-action RL experiments."""

    def __init__(
        self,
        *,
        policy: FullLearnedTemporalPolicy | LearnedLiteTemporalPolicy | None = None,
        env: InternalRLEnvironment | None = None,
        residual_runtime: OpenWeightResidualRuntime | None = None,
    ) -> None:
        self._policy = policy or FullLearnedTemporalPolicy()
        self._causal_policy = CausalZPolicy(parameter_store=self._policy.parameter_store)
        self._env = env or InternalRLEnvironment()
        self._residual_runtime = residual_runtime

    @property
    def policy(self) -> FullLearnedTemporalPolicy | LearnedLiteTemporalPolicy:
        return self._policy

    @property
    def causal_policy(self) -> CausalZPolicy:
        return self._causal_policy

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
                start = max(0, assignment.start_step)
                end = min(len(rollout.transitions) - 1, assignment.end_step)
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

    def _apply_delayed_credit_assignments(
        self,
        *,
        transitions: tuple[ZTransition, ...],
        assignments: tuple[InternalRLDelayedCreditAssignment, ...],
    ) -> tuple[ZTransition, ...]:
        if not transitions or not assignments:
            return transitions
        adjusted_rewards = [transition.raw_reward for transition in transitions]
        for assignment in assignments:
            start = max(0, assignment.start_step)
            end = min(len(transitions) - 1, assignment.end_step)
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

    def create_checkpoint(self, *, checkpoint_id: str) -> CausalPolicyCheckpoint:
        return self._causal_policy.create_checkpoint(checkpoint_id=checkpoint_id)

    def restore_checkpoint(self, checkpoint: CausalPolicyCheckpoint) -> None:
        self._causal_policy.restore_checkpoint(checkpoint)

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
            return self._optimize_single(rollout)
        if isinstance(rollout, DualTrackRollout):
            return self._optimize_dual_track(rollout).optimization_report
        return self._optimize_single(rollout)

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
