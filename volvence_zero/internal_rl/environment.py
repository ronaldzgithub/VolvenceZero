from __future__ import annotations

from dataclasses import dataclass, field

from volvence_zero.memory import Track
from volvence_zero.substrate import (
    OpenWeightResidualInterventionBackend,
    OpenWeightResidualRuntime,
    ResidualInterventionBackend,
    SubstrateSnapshot,
    TraceResidualInterventionBackend,
)
from volvence_zero.temporal import FullLearnedTemporalPolicy, TemporalAbstractionSnapshot, TemporalPolicy, TemporalStep
from volvence_zero.temporal.metacontroller_components import (
    residual_sequence_from_snapshot,
    summarize_residual_activations,
)


def _clamp(value: float) -> float:
    return max(-1.0, min(1.0, value))


@dataclass(frozen=True)
class InternalRLEnvStep:
    temporal_step: TemporalStep
    next_previous_snapshot: TemporalAbstractionSnapshot
    observation_signature: tuple[float, ...]
    latent_code: tuple[float, ...]
    decoder_output: tuple[float, ...]
    applied_control: tuple[float, ...]
    applied_snapshot: SubstrateSnapshot
    downstream_effect: tuple[float, ...]
    reward: float
    reward_components: tuple[tuple[str, float], ...]
    policy_replacement_quality: float
    backend_name: str
    backend_fidelity: float
    reward_mode: str
    reward_assignment_count: int
    proof_subgoal_id: str | None
    proof_subgoal_score: float
    proof_subgoal_completed: bool
    proof_terminal_success: bool
    active_family_id: str | None
    proof_progress: "InternalRLProofProgress | None"
    description: str


@dataclass(frozen=True)
class InternalRLDelayedCreditAssignment:
    start_step: int
    end_step: int
    reward: float
    reason: str
    subgoal_id: str | None = None
    alignment_score: float = 0.0
    window_length: int = 0
    reward_mode: str = "dense"


@dataclass(frozen=True)
class InternalRLProofSubgoal:
    subgoal_id: str
    target_signature: tuple[float, ...]
    completion_threshold: float = 0.72
    min_persistence: int = 1
    credit_horizon: int = 2
    observation_weight: float = 0.45
    effect_weight: float = 0.35
    control_weight: float = 0.20
    description: str = ""


@dataclass(frozen=True)
class InternalRLRewardSource:
    component_name: str
    kind: str
    optimizer_visible: bool
    description: str = ""


def default_proof_reward_taxonomy() -> tuple[InternalRLRewardSource, ...]:
    return (
        InternalRLRewardSource(
            component_name="proof_subgoal_complete",
            kind="delayed",
            optimizer_visible=True,
            description="Delayed reward assigned when an abstract-action window completes a subgoal.",
        ),
        InternalRLRewardSource(
            component_name="proof_terminal_success",
            kind="terminal",
            optimizer_visible=True,
            description="Terminal sparse reward assigned when all proof subgoals are complete.",
        ),
        InternalRLRewardSource(
            component_name="proof_terminal_failure",
            kind="terminal",
            optimizer_visible=True,
            description="Terminal penalty assigned when the episode ends before completing required subgoals.",
        ),
        InternalRLRewardSource(
            component_name="proof_distractor_penalty",
            kind="delayed",
            optimizer_visible=True,
            description="Delayed penalty for aligning with distractor signatures.",
        ),
        InternalRLRewardSource(
            component_name="proof_subgoal_progress",
            kind="shaping",
            optimizer_visible=True,
            description="Intermediate progress shaping retained for legacy proof profiles and reported as leakage.",
        ),
        InternalRLRewardSource(
            component_name="proof_observation_alignment",
            kind="diagnostic",
            optimizer_visible=False,
            description="Diagnostic readout of observation-to-subgoal alignment, excluded from optimizer-visible reward.",
        ),
        InternalRLRewardSource(
            component_name="proof_intervention_effect",
            kind="diagnostic",
            optimizer_visible=False,
            description="Diagnostic readout of actual residual intervention effect, excluded from optimizer-visible reward.",
        ),
    )


def sparse_proof_reward_taxonomy() -> tuple[InternalRLRewardSource, ...]:
    return (
        InternalRLRewardSource(
            component_name="proof_subgoal_complete",
            kind="delayed",
            optimizer_visible=True,
            description="Delayed sparse reward assigned when an abstract-action window completes a subgoal.",
        ),
        InternalRLRewardSource(
            component_name="proof_terminal_success",
            kind="terminal",
            optimizer_visible=True,
            description="Terminal sparse reward assigned when all proof subgoals are complete.",
        ),
        InternalRLRewardSource(
            component_name="proof_terminal_failure",
            kind="terminal",
            optimizer_visible=True,
            description="Terminal sparse penalty assigned when the route ends before completing required subgoals.",
        ),
        InternalRLRewardSource(
            component_name="proof_distractor_penalty",
            kind="delayed",
            optimizer_visible=True,
            description="Delayed sparse penalty for entering a distractor signature.",
        ),
        InternalRLRewardSource(
            component_name="proof_subgoal_progress",
            kind="diagnostic",
            optimizer_visible=False,
            description="Progress diagnostic excluded from the primary sparse optimizer path.",
        ),
        InternalRLRewardSource(
            component_name="proof_observation_alignment",
            kind="diagnostic",
            optimizer_visible=False,
            description="Observation/subgoal alignment diagnostic excluded from the primary sparse optimizer path.",
        ),
        InternalRLRewardSource(
            component_name="proof_intervention_effect",
            kind="diagnostic",
            optimizer_visible=False,
            description="Residual intervention-effect diagnostic excluded from the primary sparse optimizer path.",
        ),
    )


@dataclass(frozen=True)
class InternalRLProofEpisode:
    episode_id: str
    subgoals: tuple[InternalRLProofSubgoal, ...]
    distractor_signatures: tuple[tuple[float, ...], ...] = ()
    subgoal_reward: float = 0.35
    terminal_reward: float = 1.0
    distractor_penalty: float = 0.12
    failure_penalty: float = 0.25
    description: str = ""
    reward_profile: str = "proof-sparse-legacy-shaping"
    split_detail: str = "unspecified"
    reward_taxonomy: tuple[InternalRLRewardSource, ...] = field(default_factory=default_proof_reward_taxonomy)

    @property
    def sparse_only_rewards(self) -> bool:
        return self.reward_profile == "proof-sparse-terminal-delayed"

    def reward_source_for(self, component_name: str) -> InternalRLRewardSource:
        for source in self.reward_taxonomy:
            if source.component_name == component_name:
                return source
        raise ValueError(f"Unknown proof reward component {component_name!r} in episode {self.episode_id!r}.")

    def reward_kind_for(self, component_name: str) -> str:
        return self.reward_source_for(component_name).kind

    def reward_optimizer_visible(self, component_name: str) -> bool:
        return self.reward_source_for(component_name).optimizer_visible


@dataclass(frozen=True)
class InternalRLProofProgress:
    episode_id: str
    current_subgoal_index: int = 0
    completed_subgoals: tuple[str, ...] = ()
    completed_family_ids: tuple[str, ...] = ()
    delayed_credit_assignments: tuple[InternalRLDelayedCreditAssignment, ...] = ()
    distractor_hits: int = 0
    terminal_success: bool = False


class InternalRLEnvironment:
    """Trace-driven internal RL environment with explicit decoder control."""

    def __init__(
        self,
        *,
        control_backend: ResidualInterventionBackend | None = None,
        evaluation_family_signals: dict[str, float] | None = None,
        primary_prediction_error_enabled: bool = True,
    ) -> None:
        self._control_backend = control_backend or TraceResidualInterventionBackend()
        self._evaluation_family_signals = evaluation_family_signals or {}
        self._primary_prediction_error_enabled = primary_prediction_error_enabled

    def set_evaluation_signals(self, signals: dict[str, float]) -> None:
        self._evaluation_family_signals = dict(signals)

    @property
    def primary_prediction_error_enabled(self) -> bool:
        return self._primary_prediction_error_enabled

    def set_primary_prediction_error_enabled(self, enabled: bool) -> None:
        self._primary_prediction_error_enabled = enabled

    def set_control_backend(self, backend: ResidualInterventionBackend) -> None:
        self._control_backend = backend

    def use_open_weight_runtime(
        self,
        *,
        runtime: OpenWeightResidualRuntime,
        source_text: str,
    ) -> None:
        self._control_backend = OpenWeightResidualInterventionBackend(
            runtime=runtime,
            source_text=source_text,
        )

    def backend_fidelity(self) -> float:
        backend_name = self._control_backend.name
        if backend_name.startswith("open-weight:") or backend_name.startswith("transformers-open-weight:"):
            return 1.0
        if backend_name.startswith("synthetic-open-weight:"):
            return 0.75
        if backend_name == "trace-residual-backend":
            return 0.45
        if backend_name == "noop-residual-backend":
            return 0.2
        return 0.5

    def _family_delta(self, family: str) -> float:
        return self._evaluation_family_signals.get(family, 0.5) - 0.5

    def _signal_delta(self, signal_name: str) -> float:
        return self._evaluation_family_signals.get(signal_name, 0.5) - 0.5

    def _primary_prediction_error_signal(self) -> float:
        if not self._primary_prediction_error_enabled:
            return 0.0
        raw_value = _clamp(self._evaluation_family_signals.get("prediction_error_reward", 0.0))
        if raw_value == 0.0:
            return 0.0
        # Evaluation publishes a readout in [0, 1] centered at 0.5, while the
        # session owner injects the raw signed PE reward in [-1, 1]. Support
        # both without making evaluation a second owner of PE semantics.
        if 0.0 <= raw_value <= 1.0 and "predictive_accuracy" in self._evaluation_family_signals:
            return _clamp((raw_value - 0.5) * 2.0)
        return raw_value

    def _prediction_error_readout_signal(self) -> float:
        explicit_readout = self._evaluation_family_signals.get("prediction_error_reward_readout")
        if explicit_readout is not None:
            return _clamp(float(explicit_readout))
        raw_value = _clamp(self._evaluation_family_signals.get("prediction_error_reward", 0.0))
        if raw_value == 0.0:
            return 0.0
        if 0.0 <= raw_value <= 1.0 and "predictive_accuracy" in self._evaluation_family_signals:
            return _clamp((raw_value - 0.5) * 2.0)
        return raw_value

    def _compress_signature(self, values: tuple[float, ...] | list[float], *, size: int = 3) -> tuple[float, ...]:
        if not values:
            return tuple(0.0 for _ in range(size))
        if len(values) == size:
            return tuple(_clamp(float(value)) for value in values)
        chunk = max(len(values) // size, 1)
        compressed: list[float] = []
        for index in range(size):
            start = index * chunk
            end = len(values) if index == size - 1 else min((index + 1) * chunk, len(values))
            if start >= len(values):
                compressed.append(0.0)
                continue
            window = values[start:end]
            compressed.append(sum(window) / max(len(window), 1))
        return tuple(_clamp(value) for value in compressed)

    def _normalize_signature(self, values: tuple[float, ...]) -> tuple[float, ...]:
        norm = sum(value * value for value in values) ** 0.5
        if norm <= 1e-8:
            return tuple(0.0 for _ in values)
        return tuple(value / norm for value in values)

    def _alignment_score(self, left: tuple[float, ...], right: tuple[float, ...]) -> float:
        normalized_left = self._normalize_signature(left)
        normalized_right = self._normalize_signature(right)
        if not normalized_left or not normalized_right:
            return 0.0
        dot = sum(lv * rv for lv, rv in zip(normalized_left, normalized_right, strict=True))
        return max(0.0, min(1.0, (dot + 1.0) * 0.5))

    def _proof_signature(
        self,
        *,
        observation_signature: tuple[float, ...],
        downstream_effect: tuple[float, ...],
        applied_control: tuple[float, ...],
        observation_weight: float = 0.45,
        effect_weight: float = 0.35,
        control_weight: float = 0.20,
    ) -> tuple[float, ...]:
        control_signature = self._compress_signature(applied_control, size=3)
        total_weight = max(observation_weight + effect_weight + control_weight, 1e-6)
        return tuple(
            _clamp(
                observation_signature[index] * (observation_weight / total_weight)
                + downstream_effect[index] * (effect_weight / total_weight)
                + control_signature[index] * (control_weight / total_weight)
            )
            for index in range(3)
        )

    def _reward_components(
        self,
        *,
        track: Track,
        temporal_step: TemporalStep,
        downstream_effect: tuple[float, ...],
        control_energy: float,
        policy_replacement_quality: float,
    ) -> tuple[tuple[str, float], ...]:
        prediction_error_reward = self._primary_prediction_error_signal()
        prediction_error_readout = self._prediction_error_readout_signal()
        has_primary_pe = abs(prediction_error_reward) > 1e-8
        has_readout_pe = abs(prediction_error_readout) > 1e-8
        primary_weight = min(abs(prediction_error_reward), 1.0)
        components: list[tuple[str, float]] = [
            ("control_effect", sum(downstream_effect) / len(downstream_effect) * (0.10 if has_primary_pe else 1.0)),
            ("control_energy_bonus", control_energy * (0.01 if has_primary_pe else 0.05)),
            ("replacement_alignment", policy_replacement_quality * (0.015 if has_primary_pe else 0.08)),
            ("persistence_bonus", (1.0 - temporal_step.controller_state.switch_gate) * (0.008 if has_primary_pe else 0.04)),
            (
                "switch_bonus",
                (0.008 if has_primary_pe else 0.06) if temporal_step.controller_state.is_switching else 0.0,
            ),
            ("staleness_penalty", -temporal_step.controller_state.steps_since_switch * (0.0015 if has_primary_pe else 0.01)),
        ]
        if has_primary_pe:
            components.append(("primary_prediction_error", prediction_error_reward * (0.85 + primary_weight * 0.10)))
        if not self._evaluation_family_signals:
            return tuple(components)
        task_delta = self._family_delta("task")
        relationship_delta = self._family_delta("relationship")
        learning_delta = self._family_delta("learning")
        abstraction_delta = self._family_delta("abstraction")
        stability_delta = (self._family_delta("safety") + relationship_delta) / 2.0
        delayed_mix_delta = self._signal_delta("delayed_retrieval_mix_alignment")
        delayed_regime_delta = self._signal_delta("delayed_regime_alignment")
        delayed_action_delta = self._signal_delta("delayed_abstract_action_alignment")
        sequence_payoff_delta = self._signal_delta("regime_sequence_payoff")
        playbook_task_delta = self._signal_delta("experience_playbook_task_prior")
        playbook_support_delta = self._signal_delta("experience_playbook_support_prior")
        case_task_delta = self._signal_delta("experience_case_task_prior")
        case_support_delta = self._signal_delta("experience_case_support_prior")
        control_prior_strength = self._evaluation_family_signals.get("experience_control_prior_strength", 0.0)
        if track is Track.WORLD:
            task_weight = 0.02 if has_primary_pe else 0.24
            relationship_weight = 0.008 if has_primary_pe else 0.05
            stability_weight = 0.008 if has_primary_pe else 0.08
            control_prior_delta = (playbook_task_delta + case_task_delta) / 2.0
        elif track is Track.SELF:
            task_weight = 0.008 if has_primary_pe else 0.05
            relationship_weight = 0.02 if has_primary_pe else 0.24
            stability_weight = 0.010 if has_primary_pe else 0.10
            control_prior_delta = (playbook_support_delta + case_support_delta) / 2.0
        else:
            task_weight = 0.010 if has_primary_pe else 0.14
            relationship_weight = 0.010 if has_primary_pe else 0.14
            stability_weight = 0.008 if has_primary_pe else 0.09
            control_prior_delta = (
                playbook_task_delta
                + playbook_support_delta
                + case_task_delta
                + case_support_delta
            ) / 4.0
        components.extend(
            (
                ("task_outcome_delta", task_delta * task_weight),
                ("relationship_outcome_delta", relationship_delta * relationship_weight),
                ("learning_outcome_delta", learning_delta * (0.008 if has_primary_pe else 0.12)),
                ("abstraction_outcome_delta", abstraction_delta * (0.008 if has_primary_pe else 0.10)),
                ("stability_outcome_delta", stability_delta * stability_weight),
                ("experience_mix_delta", delayed_mix_delta * (0.006 if has_primary_pe else 0.07)),
                ("experience_regime_delta", delayed_regime_delta * (0.006 if has_primary_pe else 0.07)),
                ("experience_action_delta", delayed_action_delta * (0.005 if has_primary_pe else 0.06)),
                ("experience_sequence_delta", sequence_payoff_delta * (0.006 if has_primary_pe else 0.08)),
                (
                    "experience_control_prior",
                    control_prior_delta * ((0.004 if has_primary_pe else 0.05) + control_prior_strength * 0.02),
                ),
                (
                    "prediction_error_readout",
                    prediction_error_readout
                    * (
                        0.03
                        if has_primary_pe
                        else (0.06 if has_readout_pe else 0.15)
                    ),
                ),
            )
        )
        return tuple(components)

    def _proof_reward_components(
        self,
        *,
        proof_episode: InternalRLProofEpisode,
        proof_progress: InternalRLProofProgress | None,
        step_index: int,
        is_terminal_step: bool,
        observation_signature: tuple[float, ...],
        temporal_step: TemporalStep,
        downstream_effect: tuple[float, ...],
        applied_control: tuple[float, ...],
        active_family_id: str | None,
    ) -> tuple[
        tuple[tuple[str, float], ...],
        InternalRLProofProgress,
        str | None,
        float,
        bool,
        bool,
    ]:
        progress = proof_progress or InternalRLProofProgress(episode_id=proof_episode.episode_id)
        if progress.episode_id != proof_episode.episode_id:
            raise ValueError(
                f"Proof progress episode mismatch: expected {proof_episode.episode_id!r}, got {progress.episode_id!r}"
            )
        components: list[tuple[str, float]] = []
        assignments = list(progress.delayed_credit_assignments)
        completed_subgoals = progress.completed_subgoals
        completed_family_ids = progress.completed_family_ids
        current_subgoal_index = progress.current_subgoal_index
        distractor_hits = progress.distractor_hits
        subgoal_id: str | None = None
        subgoal_score = 0.0
        subgoal_completed = False
        terminal_success = progress.terminal_success
        if current_subgoal_index < len(proof_episode.subgoals):
            subgoal = proof_episode.subgoals[current_subgoal_index]
            proof_signature = self._proof_signature(
                observation_signature=observation_signature,
                downstream_effect=downstream_effect,
                applied_control=applied_control,
                observation_weight=subgoal.observation_weight,
                effect_weight=subgoal.effect_weight,
                control_weight=subgoal.control_weight,
            )
            subgoal_id = subgoal.subgoal_id
            subgoal_score = self._alignment_score(
                proof_signature,
                self._compress_signature(subgoal.target_signature, size=3),
            )
            observation_alignment = self._alignment_score(
                observation_signature,
                self._compress_signature(subgoal.target_signature, size=3),
            )
            intervention_effect = sum(abs(value) for value in downstream_effect) / max(len(downstream_effect), 1)
            components.append(("proof_observation_alignment", observation_alignment * 0.01))
            components.append(("proof_intervention_effect", intervention_effect * 0.01))
            if subgoal_score >= subgoal.completion_threshold and temporal_step.controller_state.steps_since_switch >= (
                subgoal.min_persistence - 1
            ):
                subgoal_completed = True
                components.append(("proof_subgoal_complete", proof_episode.subgoal_reward))
                assignments.append(
                    InternalRLDelayedCreditAssignment(
                        start_step=max(0, step_index - max(subgoal.credit_horizon - 1, 0)),
                        end_step=step_index,
                        reward=proof_episode.subgoal_reward,
                        reason="subgoal-complete",
                        subgoal_id=subgoal.subgoal_id,
                        alignment_score=subgoal_score,
                        window_length=max(subgoal.credit_horizon, 1),
                        reward_mode="proof-sparse",
                    )
                )
                completed_subgoals = completed_subgoals + (subgoal.subgoal_id,)
                completed_family_ids = completed_family_ids + ((active_family_id or "unassigned"),)
                current_subgoal_index += 1
            elif not proof_episode.sparse_only_rewards:
                components.append(("proof_subgoal_progress", subgoal_score * 0.05))
        else:
            proof_signature = self._proof_signature(
                observation_signature=observation_signature,
                downstream_effect=downstream_effect,
                applied_control=applied_control,
            )
        distractor_score = 0.0
        for distractor in proof_episode.distractor_signatures:
            distractor_score = max(
                distractor_score,
                self._alignment_score(proof_signature, self._compress_signature(distractor, size=3)),
            )
        if distractor_score >= 0.76:
            distractor_hits += 1
            components.append(("proof_distractor_penalty", -proof_episode.distractor_penalty))
        if is_terminal_step:
            if current_subgoal_index >= len(proof_episode.subgoals):
                terminal_success = True
                components.append(("proof_terminal_success", proof_episode.terminal_reward))
                assignments.append(
                    InternalRLDelayedCreditAssignment(
                        start_step=max(0, step_index - 2),
                        end_step=step_index,
                        reward=proof_episode.terminal_reward,
                        reason="terminal-success",
                        subgoal_id=subgoal_id,
                        alignment_score=1.0,
                        window_length=3,
                        reward_mode="proof-sparse",
                    )
                )
            else:
                components.append(("proof_terminal_failure", -proof_episode.failure_penalty))
        progress = InternalRLProofProgress(
            episode_id=progress.episode_id,
            current_subgoal_index=current_subgoal_index,
            completed_subgoals=completed_subgoals,
            completed_family_ids=completed_family_ids,
            delayed_credit_assignments=tuple(assignments),
            distractor_hits=distractor_hits,
            terminal_success=terminal_success,
        )
        return (
            tuple(components),
            progress,
            subgoal_id,
            subgoal_score,
            subgoal_completed,
            terminal_success,
        )

    def step(
        self,
        *,
        substrate_snapshot: SubstrateSnapshot,
        track: Track,
        policy: TemporalPolicy,
        previous_snapshot: TemporalAbstractionSnapshot | None,
        policy_latent_override: tuple[float, ...] | None = None,
        policy_replacement_score: float = 0.0,
        binary_gate_override: bool = False,
        step_index: int = 0,
        is_terminal_step: bool = False,
        proof_episode: InternalRLProofEpisode | None = None,
        proof_progress: InternalRLProofProgress | None = None,
    ) -> InternalRLEnvStep:
        if isinstance(policy, FullLearnedTemporalPolicy) and policy_latent_override is not None:
            temporal_step = policy.step_with_causal_override(
                substrate_snapshot=substrate_snapshot,
                previous_snapshot=previous_snapshot,
                latent_override=policy_latent_override,
                policy_replacement_score=policy_replacement_score,
                binary_gate_override=binary_gate_override,
                memory_snapshot=None,
                reflection_snapshot=None,
            )
        else:
            temporal_step = policy.step(
                substrate_snapshot=substrate_snapshot,
                previous_snapshot=previous_snapshot,
                memory_snapshot=None,
                reflection_snapshot=None,
            )
        runtime_state = policy.export_runtime_state()
        sequence = residual_sequence_from_snapshot(substrate_snapshot)
        sequence_signature = tuple(
            sum(
                summarize_residual_activations(step.residual_activations, step.feature_surface)[index]
                for step in sequence
            )
            / len(sequence)
            for index in range(3)
        )
        decoder_output = (
            runtime_state.decoder_control
            if runtime_state is not None
            else temporal_step.controller_state.code
        )
        applied_control = (
            runtime_state.decoder_applied_control
            if runtime_state is not None and runtime_state.decoder_applied_control
            else decoder_output
        )
        active_family_id = (
            runtime_state.active_family_summary.family_id
            if runtime_state is not None and runtime_state.active_family_summary is not None
            else None
        )
        track_emphasis = {
            Track.WORLD: (1.0, 0.4, 0.5),
            Track.SELF: (0.4, 1.0, 0.6),
            Track.SHARED: (0.7, 0.7, 0.7),
        }[track]
        control_application = self._control_backend.apply_control(
            substrate_snapshot=substrate_snapshot,
            applied_control=applied_control,
            track_scale=track_emphasis,
        )
        downstream_effect = control_application.downstream_effect
        policy_replacement_quality = _clamp(
            1.0
            - sum(
                abs(applied_control[index] - downstream_effect[index]) for index in range(3)
            )
            / 3.0
            + policy_replacement_score * 0.25
        )
        if proof_episode is None:
            reward_components = self._reward_components(
                track=track,
                temporal_step=temporal_step,
                downstream_effect=downstream_effect,
                control_energy=control_application.control_energy,
                policy_replacement_quality=policy_replacement_quality,
            )
            reward_mode = "dense"
            next_proof_progress = proof_progress
            proof_subgoal_id = None
            proof_subgoal_score = 0.0
            proof_subgoal_completed = False
            proof_terminal_success = False
        else:
            (
                reward_components,
                next_proof_progress,
                proof_subgoal_id,
                proof_subgoal_score,
                proof_subgoal_completed,
                proof_terminal_success,
            ) = self._proof_reward_components(
                proof_episode=proof_episode,
                proof_progress=proof_progress,
                step_index=step_index,
                is_terminal_step=is_terminal_step,
                observation_signature=sequence_signature,
                temporal_step=temporal_step,
                downstream_effect=downstream_effect,
                applied_control=applied_control,
                active_family_id=active_family_id,
            )
            reward_mode = "proof-sparse"
        reward = sum(value for _, value in reward_components)
        next_previous_snapshot = TemporalAbstractionSnapshot(
            controller_state=temporal_step.controller_state,
            active_abstract_action=temporal_step.active_abstract_action,
            controller_params_hash=temporal_step.controller_params_hash,
            description=temporal_step.description,
        )
        return InternalRLEnvStep(
            temporal_step=temporal_step,
            next_previous_snapshot=next_previous_snapshot,
            observation_signature=sequence_signature,
            latent_code=temporal_step.controller_state.code,
            decoder_output=decoder_output,
            applied_control=applied_control,
            applied_snapshot=control_application.applied_snapshot,
            downstream_effect=downstream_effect,
            reward=_clamp(reward),
            reward_components=reward_components,
            policy_replacement_quality=policy_replacement_quality,
            backend_name=control_application.backend_name,
            backend_fidelity=self.backend_fidelity(),
            reward_mode=reward_mode,
            reward_assignment_count=(
                len(next_proof_progress.delayed_credit_assignments) if next_proof_progress is not None else 0
            ),
            proof_subgoal_id=proof_subgoal_id,
            proof_subgoal_score=proof_subgoal_score,
            proof_subgoal_completed=proof_subgoal_completed,
            proof_terminal_success=proof_terminal_success,
            active_family_id=active_family_id,
            proof_progress=next_proof_progress,
            description=(
                f"track={track.value} latent={tuple(round(value, 3) for value in temporal_step.controller_state.code)} "
                f"decoder={tuple(round(value, 3) for value in decoder_output)} "
                f"applied={tuple(round(value, 3) for value in applied_control)} "
                f"backend={control_application.backend_name} "
                f"backend_fidelity={self.backend_fidelity():.2f} "
                f"reward_components={tuple((name, round(value, 3)) for name, value in reward_components)} "
                f"replacement_quality={policy_replacement_quality:.3f} "
                f"{control_application.description}"
            ),
        )
