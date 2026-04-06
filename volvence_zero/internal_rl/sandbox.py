from __future__ import annotations

from dataclasses import dataclass

from volvence_zero.credit import CreditRecord
from volvence_zero.internal_rl.environment import InternalRLEnvStep, InternalRLEnvironment
from volvence_zero.memory import Track
from volvence_zero.substrate import SubstrateSnapshot
from volvence_zero.temporal import (
    ControllerState,
    FullLearnedTemporalPolicy,
    LearnedLiteTemporalPolicy,
    MetacontrollerParameterSnapshot,
    MetacontrollerParameterStore,
    TemporalAbstractionSnapshot,
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
    policy_replacement_quality: float
    backend_name: str


@dataclass(frozen=True)
class ZRollout:
    rollout_id: str
    track: Track
    transitions: tuple[ZTransition, ...]
    total_reward: float
    description: str
    replacement_mode: str = "causal"


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


@dataclass(frozen=True)
class DualTrackOptimizationReport:
    task_report: OptimizationReport
    relationship_report: OptimizationReport
    description: str


def _clamp(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _surface_signature(substrate_snapshot: SubstrateSnapshot) -> tuple[float, ...]:
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
        return (0.0, 0.0, 0.0)
    average = sum(values) / len(values)
    maximum = max(values)
    spread = maximum - min(values)
    return (_clamp(average), _clamp(maximum), _clamp(spread))


class CausalZPolicy:
    """Small causal policy over z-space with per-track recurrent state."""

    def __init__(self, *, parameter_store: MetacontrollerParameterStore) -> None:
        self._parameter_store = parameter_store

    def initial_state(self, *, track: Track) -> CausalPolicyState:
        return CausalPolicyState(
            track=track,
            hidden_state=(0.0, 0.0, 0.0),
            previous_action=(0.0, 0.0, 0.0),
            step_index=0,
        )

    def step(
        self,
        *,
        substrate_snapshot: SubstrateSnapshot,
        state: CausalPolicyState,
    ) -> tuple[CausalPolicyState, tuple[float, ...], tuple[float, ...], tuple[float, ...], float, float]:
        surface = _surface_signature(substrate_snapshot)
        weights = self._parameter_store.track_weights[state.track]
        track_projected = tuple(
            _clamp(surface[i] * weights[i] * 1.5 + surface[i] * 0.25)
            for i in range(len(surface))
        )
        hidden_state = tuple(
            _clamp(
                previous * self._parameter_store.persistence
                + current * (1.0 - self._parameter_store.persistence)
            )
            for previous, current in zip(state.hidden_state, track_projected)
        )
        policy_action = self._policy_action(
            hidden_state=hidden_state,
            surface=surface,
            previous_action=state.previous_action,
            weights=weights,
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
        log_prob = self._log_prob(policy_score=policy_score)
        return next_state, surface, hidden_state, policy_action, policy_score, log_prob

    def export_parameters(self) -> tuple[CausalPolicyParameters, ...]:
        return tuple(
            CausalPolicyParameters(
                track=track,
                weights=self._parameter_store.track_weights[track],
                persistence=self._parameter_store.persistence,
                learning_rate=self._parameter_store.learning_rate,
                update_step=self._parameter_store.update_steps[track],
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
        self._parameter_store.persistence = checkpoint.parameters_by_track[0].persistence
        self._parameter_store.learning_rate = checkpoint.parameters_by_track[0].learning_rate
        self._parameter_store.update_steps = {
            params.track: params.update_step for params in checkpoint.parameters_by_track
        }
        self._parameter_store.restore_parameter_snapshot(checkpoint.metacontroller_snapshot)

    def _policy_score(
        self,
        *,
        weights: tuple[float, ...],
        hidden_state: tuple[float, ...],
        surface: tuple[float, ...],
        policy_action: tuple[float, ...],
    ) -> float:
        score = sum(weight * hidden for weight, hidden in zip(weights, hidden_state))
        score += sum(weight * value for weight, value in zip(weights, surface)) * 0.5
        score += sum(weight * value for weight, value in zip(weights, policy_action)) * 0.35
        return _clamp(score)

    def _policy_action(
        self,
        *,
        hidden_state: tuple[float, ...],
        surface: tuple[float, ...],
        previous_action: tuple[float, ...],
        weights: tuple[float, ...],
    ) -> tuple[float, ...]:
        proposal = tuple(
            _clamp(
                hidden_state[index] * 0.50
                + surface[index] * 0.30
                + previous_action[index] * 0.20
                + weights[index] * 0.10
            )
            for index in range(3)
        )
        return self._normalize_weights(proposal)

    def _log_prob(self, *, policy_score: float) -> float:
        centered = policy_score - 0.5
        return -(centered * centered) / 0.125

    def _normalize_weights(self, weights: tuple[float, ...]) -> tuple[float, ...]:
        total = max(sum(weights), 1e-6)
        return tuple(value / total for value in weights)

    def _trajectory_gradient(
        self,
        *,
        rollout: ZRollout,
        advantages: tuple[float, ...],
    ) -> tuple[float, ...]:
        dims = len(rollout.transitions[0].observation_signature)
        accum = [0.0 for _ in range(dims)]
        for transition, advantage in zip(rollout.transitions, advantages):
            for index, value in enumerate(transition.observation_signature):
                accum[index] += (
                    value * advantage
                    + transition.policy_action[index] * transition.policy_replacement_quality * 0.5
                )
        scale = 1.0 / max(len(rollout.transitions), 1)
        return tuple(delta * scale for delta in accum)

    def _surrogate_metrics(
        self,
        *,
        rollout: ZRollout,
        advantages: tuple[float, ...],
        old_weights: tuple[float, ...],
        new_weights: tuple[float, ...],
    ) -> tuple[float, float, float]:
        clipped = 0
        objective_terms: list[float] = []
        kl_terms: list[float] = []
        for transition, advantage in zip(rollout.transitions, advantages):
            new_score = self._policy_score(
                weights=new_weights,
                hidden_state=transition.hidden_state,
                surface=transition.observation_signature,
                policy_action=transition.policy_action,
            )
            new_log_prob = self._log_prob(policy_score=new_score)
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
            old_score = self._policy_score(
                weights=old_weights,
                hidden_state=transition.hidden_state,
                surface=transition.observation_signature,
                policy_action=transition.policy_action,
            )
            kl_terms.append(abs(new_score - old_score))
        clip_fraction = clipped / max(len(rollout.transitions), 1)
        surrogate_objective = sum(objective_terms) / max(len(objective_terms), 1)
        kl_penalty = sum(kl_terms) / max(len(kl_terms), 1)
        return (surrogate_objective, clip_fraction, kl_penalty)

    def optimize(self, *, rollout: ZRollout) -> OptimizationReport:
        if not rollout.transitions:
            return OptimizationReport(
                track=rollout.track,
                average_reward=0.0,
                baseline_reward=0.0,
                mean_advantage=0.0,
                surrogate_objective=0.0,
                clip_fraction=0.0,
                kl_penalty=0.0,
                parameter_summary="no-op",
            )
        rewards = tuple(transition.reward for transition in rollout.transitions)
        baseline_reward = sum(rewards) / len(rewards)
        advantages = tuple(reward - baseline_reward for reward in rewards)
        mean_advantage = sum(advantages) / len(advantages)
        previous_weights = self._parameter_store.track_weights[rollout.track]
        gradient = self._trajectory_gradient(rollout=rollout, advantages=advantages)
        proposed_weights = tuple(
            _clamp(weight + self._parameter_store.learning_rate * delta)
            for weight, delta in zip(previous_weights, gradient)
        )
        normalized_weights = self._normalize_weights(proposed_weights)
        surrogate_objective, clip_fraction, kl_penalty = self._surrogate_metrics(
            rollout=rollout,
            advantages=advantages,
            old_weights=previous_weights,
            new_weights=normalized_weights,
        )
        if surrogate_objective >= -0.05:
            self._parameter_store.track_weights[rollout.track] = normalized_weights
        self._parameter_store.persistence = _clamp(
            self._parameter_store.persistence
            + mean_advantage * self._parameter_store.learning_rate * 0.05
        )
        self._parameter_store.update_steps[rollout.track] += 1
        self._parameter_store.align_temporal_from_tracks()
        return OptimizationReport(
            track=rollout.track,
            average_reward=rollout.total_reward / len(rollout.transitions),
            baseline_reward=baseline_reward,
            mean_advantage=mean_advantage,
            surrogate_objective=surrogate_objective,
            clip_fraction=clip_fraction,
            kl_penalty=kl_penalty,
            parameter_summary=(
                f"track={rollout.track.value} weights={self._parameter_store.track_weights[rollout.track]} "
                f"persistence={self._parameter_store.persistence:.3f} "
                f"step={self._parameter_store.update_steps[rollout.track]} "
                f"objective={surrogate_objective:.3f} "
                f"replacement={sum(t.policy_replacement_quality for t in rollout.transitions)/len(rollout.transitions):.3f}"
            ),
        )


class InternalRLSandbox:
    """Minimal z-space rollout sandbox for abstract-action RL experiments."""

    def __init__(
        self,
        *,
        policy: FullLearnedTemporalPolicy | LearnedLiteTemporalPolicy | None = None,
        env: InternalRLEnvironment | None = None,
    ) -> None:
        self._policy = policy or FullLearnedTemporalPolicy()
        self._causal_policy = CausalZPolicy(parameter_store=self._policy.parameter_store)
        self._env = env or InternalRLEnvironment()

    @property
    def policy(self) -> FullLearnedTemporalPolicy | LearnedLiteTemporalPolicy:
        return self._policy

    @property
    def causal_policy(self) -> CausalZPolicy:
        return self._causal_policy

    def rollout(
        self,
        *,
        rollout_id: str,
        substrate_steps: tuple[SubstrateSnapshot, ...],
        track: Track = Track.SHARED,
        replacement_mode: str = "causal",
    ) -> ZRollout:
        previous_snapshot: TemporalAbstractionSnapshot | None = None
        transitions: list[ZTransition] = []
        policy_state = self._causal_policy.initial_state(track=track)
        for step_index, substrate_snapshot in enumerate(substrate_steps):
            (
                policy_state,
                observation_signature,
                hidden_state,
                policy_action,
                policy_score,
                log_prob,
            ) = self._causal_policy.step(
                substrate_snapshot=substrate_snapshot,
                state=policy_state,
            )
            env_step = self._env.step(
                substrate_snapshot=substrate_snapshot,
                track=track,
                policy=self._policy,
                previous_snapshot=previous_snapshot,
                policy_latent_override=policy_action if replacement_mode in {"causal", "causal-binary"} else None,
                policy_replacement_score=policy_score if replacement_mode in {"causal", "causal-binary"} else 0.0,
                binary_gate_override=replacement_mode == "causal-binary",
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
                    policy_score=policy_score,
                    log_prob=log_prob,
                    reward=env_step.reward,
                    policy_replacement_quality=env_step.policy_replacement_quality,
                    backend_name=env_step.backend_name,
                )
            )
            previous_snapshot = env_step.next_previous_snapshot
        total_reward = sum(transition.reward for transition in transitions)
        return ZRollout(
            rollout_id=rollout_id,
            track=track,
            transitions=tuple(transitions),
            total_reward=total_reward,
            replacement_mode=replacement_mode,
            description=(
                f"Internal RL rollout mode={replacement_mode} track={track.value} over {len(transitions)} abstract actions "
                f"with total reward {total_reward:.2f}."
            ),
        )

    def rollout_dual_track(
        self,
        *,
        rollout_id: str,
        substrate_steps: tuple[SubstrateSnapshot, ...],
    ) -> DualTrackRollout:
        task_rollout = self.rollout(
            rollout_id=f"{rollout_id}:task",
            substrate_steps=substrate_steps,
            track=Track.WORLD,
            replacement_mode="causal-binary",
        )
        relationship_rollout = self.rollout(
            rollout_id=f"{rollout_id}:relationship",
            substrate_steps=substrate_steps,
            track=Track.SELF,
            replacement_mode="causal-binary",
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

    def optimize(self, rollout: ZRollout | DualTrackRollout) -> OptimizationReport | DualTrackOptimizationReport:
        if isinstance(rollout, DualTrackRollout):
            task_report = self.optimize(rollout.task_rollout)
            relationship_report = self.optimize(rollout.relationship_rollout)
            if not isinstance(task_report, OptimizationReport):
                raise TypeError("task_report must be OptimizationReport.")
            if not isinstance(relationship_report, OptimizationReport):
                raise TypeError("relationship_report must be OptimizationReport.")
            return DualTrackOptimizationReport(
                task_report=task_report,
                relationship_report=relationship_report,
                description=(
                    f"task_adv={task_report.mean_advantage:.3f}, "
                    f"rel_adv={relationship_report.mean_advantage:.3f}"
                ),
            )
        if not rollout.transitions:
            return OptimizationReport(
                track=rollout.track,
                average_reward=0.0,
                baseline_reward=0.0,
                mean_advantage=0.0,
                surrogate_objective=0.0,
                clip_fraction=0.0,
                kl_penalty=0.0,
                parameter_summary="no-op",
            )
        average_reward = rollout.total_reward / len(rollout.transitions)
        reward_scale = max(average_reward, 0.05)
        causal_report = self._causal_policy.optimize(rollout=rollout)
        if rollout.track is Track.WORLD:
            self._policy.fit_from_signals(
                residual_strength=max(0.45, reward_scale),
                memory_strength=0.20,
                reflection_strength=0.15,
            )
            return causal_report
        if rollout.track is Track.SELF:
            self._policy.fit_from_signals(
                residual_strength=0.35,
                memory_strength=0.25,
                reflection_strength=max(0.25, reward_scale),
            )
            return causal_report
        self._policy.fit_from_signals(
            residual_strength=max(0.4, reward_scale),
            memory_strength=0.25,
            reflection_strength=0.35 if average_reward > 0.4 else 0.2,
        )
        return causal_report


    def _reward_for_step(
        self,
        *,
        controller_state: ControllerState,
        hidden_state: tuple[float, ...],
        track: Track,
    ) -> float:
        reward = sum(controller_state.code) / max(len(controller_state.code), 1)
        reward += sum(hidden_state) / max(len(hidden_state), 1) * 0.2
        if controller_state.is_switching:
            reward += 0.1
        reward -= controller_state.steps_since_switch * 0.02
        if track is Track.WORLD:
            reward += hidden_state[0] * 0.15
        elif track is Track.SELF:
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
