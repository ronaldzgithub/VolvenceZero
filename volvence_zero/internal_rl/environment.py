from __future__ import annotations

from dataclasses import dataclass

from volvence_zero.memory import Track
from volvence_zero.substrate import (
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
    policy_replacement_quality: float
    backend_name: str
    description: str


class InternalRLEnvironment:
    """Trace-driven internal RL environment with explicit decoder control."""

    def __init__(
        self,
        *,
        control_backend: ResidualInterventionBackend | None = None,
        evaluation_family_signals: dict[str, float] | None = None,
    ) -> None:
        self._control_backend = control_backend or TraceResidualInterventionBackend()
        self._evaluation_family_signals = evaluation_family_signals or {}

    def set_evaluation_signals(self, signals: dict[str, float]) -> None:
        self._evaluation_family_signals = dict(signals)

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
        reward = sum(downstream_effect) / len(downstream_effect)
        reward += control_application.control_energy * 0.05
        reward += (1.0 - temporal_step.controller_state.switch_gate) * 0.05
        if temporal_step.controller_state.is_switching:
            reward += 0.08
        reward -= temporal_step.controller_state.steps_since_switch * 0.01
        if self._evaluation_family_signals:
            eval_bonus = 0.0
            if track is Track.WORLD:
                eval_bonus += self._evaluation_family_signals.get("task", 0.5) * 0.08
            elif track is Track.SELF:
                eval_bonus += self._evaluation_family_signals.get("relationship", 0.5) * 0.08
            eval_bonus += self._evaluation_family_signals.get("learning", 0.5) * 0.04
            reward += eval_bonus
        policy_replacement_quality = _clamp(
            1.0
            - sum(
                abs(applied_control[index] - downstream_effect[index]) for index in range(3)
            )
            / 3.0
            + policy_replacement_score * 0.25
        )
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
            policy_replacement_quality=policy_replacement_quality,
            backend_name=control_application.backend_name,
            description=(
                f"track={track.value} latent={tuple(round(value, 3) for value in temporal_step.controller_state.code)} "
                f"decoder={tuple(round(value, 3) for value in decoder_output)} "
                f"applied={tuple(round(value, 3) for value in applied_control)} "
                f"backend={control_application.backend_name} "
                f"replacement_quality={policy_replacement_quality:.3f} "
                f"{control_application.description}"
            ),
        )
