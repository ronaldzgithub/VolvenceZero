from __future__ import annotations

from dataclasses import dataclass

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

    def _reward_components(
        self,
        *,
        track: Track,
        temporal_step: TemporalStep,
        downstream_effect: tuple[float, ...],
        control_energy: float,
        policy_replacement_quality: float,
    ) -> tuple[tuple[str, float], ...]:
        prediction_error_reward = self._evaluation_family_signals.get("prediction_error_reward", 0.0)
        has_primary_pe = abs(prediction_error_reward) > 1e-8
        components: list[tuple[str, float]] = [
            ("control_effect", sum(downstream_effect) / len(downstream_effect) * (0.12 if has_primary_pe else 1.0)),
            ("control_energy_bonus", control_energy * (0.01 if has_primary_pe else 0.05)),
            ("replacement_alignment", policy_replacement_quality * (0.02 if has_primary_pe else 0.08)),
            ("persistence_bonus", (1.0 - temporal_step.controller_state.switch_gate) * (0.01 if has_primary_pe else 0.04)),
            (
                "switch_bonus",
                (0.01 if has_primary_pe else 0.06) if temporal_step.controller_state.is_switching else 0.0,
            ),
            ("staleness_penalty", -temporal_step.controller_state.steps_since_switch * (0.002 if has_primary_pe else 0.01)),
        ]
        if has_primary_pe:
            components.append(("primary_prediction_error", prediction_error_reward * 0.70))
        if not self._evaluation_family_signals:
            return tuple(components)
        task_delta = self._family_delta("task")
        relationship_delta = self._family_delta("relationship")
        learning_delta = self._family_delta("learning")
        abstraction_delta = self._family_delta("abstraction")
        stability_delta = (self._family_delta("safety") + relationship_delta) / 2.0
        if track is Track.WORLD:
            task_weight = 0.03 if has_primary_pe else 0.24
            relationship_weight = 0.01 if has_primary_pe else 0.05
            stability_weight = 0.01 if has_primary_pe else 0.08
        elif track is Track.SELF:
            task_weight = 0.01 if has_primary_pe else 0.05
            relationship_weight = 0.03 if has_primary_pe else 0.24
            stability_weight = 0.015 if has_primary_pe else 0.10
        else:
            task_weight = 0.015 if has_primary_pe else 0.14
            relationship_weight = 0.015 if has_primary_pe else 0.14
            stability_weight = 0.01 if has_primary_pe else 0.09
        components.extend(
            (
                ("task_outcome_delta", task_delta * task_weight),
                ("relationship_outcome_delta", relationship_delta * relationship_weight),
                ("learning_outcome_delta", learning_delta * (0.01 if has_primary_pe else 0.12)),
                ("abstraction_outcome_delta", abstraction_delta * (0.01 if has_primary_pe else 0.10)),
                ("stability_outcome_delta", stability_delta * stability_weight),
                ("prediction_error_reward", prediction_error_reward * (0.05 if has_primary_pe else 0.15)),
            )
        )
        return tuple(components)

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
        policy_replacement_quality = _clamp(
            1.0
            - sum(
                abs(applied_control[index] - downstream_effect[index]) for index in range(3)
            )
            / 3.0
            + policy_replacement_score * 0.25
        )
        reward_components = self._reward_components(
            track=track,
            temporal_step=temporal_step,
            downstream_effect=downstream_effect,
            control_energy=control_application.control_energy,
            policy_replacement_quality=policy_replacement_quality,
        )
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
