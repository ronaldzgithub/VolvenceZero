from __future__ import annotations

from dataclasses import dataclass

from volvence_zero.substrate import (
    FeatureSignal,
    ResidualActivation,
    ResidualSequenceStep,
    SubstrateSnapshot,
)


def clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def clamp_signed(value: float) -> float:
    return max(-1.0, min(1.0, value))


@dataclass(frozen=True)
class PosteriorState:
    prior_mean: tuple[float, ...]
    prior_std: tuple[float, ...]
    posterior_mean: tuple[float, ...]
    posterior_std: tuple[float, ...]
    sample_noise: tuple[float, ...]
    z_tilde: tuple[float, ...]
    hidden_state: tuple[float, ...]
    posterior_drift: float


@dataclass(frozen=True)
class EncodedSequence:
    posterior: PosteriorState
    sequence_length: int
    summary: str

    @property
    def latent_mean(self) -> tuple[float, ...]:
        return self.posterior.posterior_mean

    @property
    def latent_scale(self) -> tuple[float, ...]:
        return self.posterior.posterior_std

    @property
    def z_tilde(self) -> tuple[float, ...]:
        return self.posterior.z_tilde


@dataclass(frozen=True)
class SwitchGateDecision:
    beta_continuous: float
    beta_binary: int
    sparsity: float
    binary_switch_rate: float
    mean_persistence_window: float
    summary: str


@dataclass(frozen=True)
class DecoderControl:
    decoder_output: tuple[float, ...]
    applied_control: tuple[float, ...]
    summary: str


def residual_sequence_from_snapshot(substrate_snapshot: SubstrateSnapshot) -> tuple[ResidualSequenceStep, ...]:
    if substrate_snapshot.residual_sequence:
        return substrate_snapshot.residual_sequence
    return (
        ResidualSequenceStep(
            step=max((activation.step for activation in substrate_snapshot.residual_activations), default=0),
            token="<runtime-step>",
            feature_surface=substrate_snapshot.feature_surface,
            residual_activations=substrate_snapshot.residual_activations,
            description="Synthesized single-step residual sequence.",
        ),
    )


def summarize_feature_surface(feature_surface: tuple[FeatureSignal, ...]) -> tuple[float, float, float]:
    if not feature_surface:
        return (0.0, 0.0, 0.0)
    values = [sum(feature.values) / len(feature.values) for feature in feature_surface if feature.values]
    if not values:
        return (0.0, 0.0, 0.0)
    average = sum(values) / len(values)
    maximum = max(values)
    spread = maximum - min(values)
    return (clamp_unit(average), clamp_unit(maximum), clamp_unit(spread))


def summarize_residual_activations(
    residual_activations: tuple[ResidualActivation, ...],
    feature_surface: tuple[FeatureSignal, ...],
) -> tuple[float, float, float]:
    if not residual_activations:
        return summarize_feature_surface(feature_surface)
    values = [
        sum(activation.activation) / len(activation.activation)
        for activation in residual_activations
        if activation.activation
    ]
    if not values:
        return summarize_feature_surface(feature_surface)
    average = sum(values) / len(values)
    maximum = max(values)
    spread = maximum - min(values)
    return (clamp_unit(average), clamp_unit(maximum), clamp_unit(spread))


class SequenceEncoder:
    def encode(
        self,
        *,
        substrate_snapshot: SubstrateSnapshot,
        encoder_weights: tuple[tuple[float, ...], ...],
        recurrence_weights: tuple[tuple[float, ...], ...] = (
            (0.60, 0.20, 0.20),
            (0.20, 0.60, 0.20),
            (0.20, 0.20, 0.60),
        ),
        previous_hidden_state: tuple[float, ...] = (0.0, 0.0, 0.0),
    ) -> EncodedSequence:
        sequence = residual_sequence_from_snapshot(substrate_snapshot)
        step_vectors = tuple(
            summarize_residual_activations(step.residual_activations, step.feature_surface) for step in sequence
        )
        if not step_vectors:
            step_vectors = ((0.0, 0.0, 0.0),)
        hidden_state = previous_hidden_state
        hidden_history: list[tuple[float, ...]] = []
        for step_vector in step_vectors:
            hidden_state = tuple(
                clamp_unit(
                    sum(weight * hidden_value for weight, hidden_value in zip(row, hidden_state, strict=True)) * 0.55
                    + step_vector[index] * 0.45
                )
                for index, row in enumerate(recurrence_weights)
            )
            hidden_history.append(hidden_state)
        average_vector = tuple(
            sum(vector[index] for vector in step_vectors) / len(step_vectors) for index in range(3)
        )
        peak_vector = tuple(max(vector[index] for vector in step_vectors) for index in range(3))
        first_vector = step_vectors[0]
        last_vector = step_vectors[-1]
        trend_vector = tuple(last_vector[index] - first_vector[index] for index in range(3))
        prior_mean = tuple(
            clamp_unit(previous_hidden_state[index] * 0.35) for index in range(3)
        )
        prior_std = tuple(
            max(0.05, 1.0 - clamp_unit(abs(previous_hidden_state[index] - average_vector[index]) * 0.5))
            for index in range(3)
        )
        posterior_mean = tuple(
            clamp_unit(
                sum(weight * value for weight, value in zip(row, hidden_state, strict=True))
                + average_vector[index] * 0.20
                + trend_vector[index] * 0.10
            )
            for index, row in enumerate(encoder_weights)
        )
        posterior_std = tuple(
            clamp_unit(
                abs(peak_vector[index] - average_vector[index]) * 0.7
                + abs(trend_vector[index]) * 0.3
            )
            for index in range(3)
        )
        sample_noise = tuple(
            clamp_signed(trend_vector[index] * 1.5 + (average_vector[index] - 0.5) * 0.8)
            for index in range(3)
        )
        z_tilde = tuple(
            clamp_unit(
                posterior_mean[index]
                + posterior_std[index] * sample_noise[index] * 0.5
            )
            for index in range(3)
        )
        posterior_drift = max(
            abs(hidden_state[index] - previous_hidden_state[index]) for index in range(3)
        )
        return EncodedSequence(
            posterior=PosteriorState(
                prior_mean=prior_mean,
                prior_std=prior_std,
                posterior_mean=posterior_mean,
                posterior_std=posterior_std,
                sample_noise=sample_noise,
                z_tilde=z_tilde,
                hidden_state=hidden_state,
                posterior_drift=posterior_drift,
            ),
            sequence_length=len(sequence),
            summary=(
                f"encoded_sequence len={len(sequence)} hidden={tuple(round(value, 3) for value in hidden_state)} "
                f"prior_mu={tuple(round(value, 3) for value in prior_mean)} "
                f"mu={tuple(round(value, 3) for value in posterior_mean)} "
                f"sigma={tuple(round(value, 3) for value in posterior_std)} "
                f"eps={tuple(round(value, 3) for value in sample_noise)}"
            ),
        )


class SwitchUnit:
    def compute_decision(
        self,
        *,
        previous_code: tuple[float, ...],
        z_tilde: tuple[float, ...],
        posterior_std: tuple[float, ...],
        switch_weights: tuple[float, ...],
        switch_bias: float,
        memory_signal: float,
        reflection_signal: float,
        previous_binary: int = 0,
        previous_steps_since_switch: int = 0,
    ) -> SwitchGateDecision:
        delta = tuple(abs(current - previous) for current, previous in zip(z_tilde, previous_code, strict=True))
        raw_gate = switch_bias
        raw_gate += sum(weight * value for weight, value in zip(switch_weights, delta, strict=True))
        raw_gate += sum(posterior_std) / max(len(posterior_std), 1) * 0.25
        raw_gate += memory_signal * 0.10 + reflection_signal * 0.20
        beta_continuous = clamp_unit(raw_gate)
        beta_binary = 1 if beta_continuous >= 0.55 else 0
        binary_switch_rate = (beta_binary + previous_binary) / 2.0
        mean_persistence_window = 0.0 if beta_binary else float(previous_steps_since_switch + 1)
        return SwitchGateDecision(
            beta_continuous=beta_continuous,
            beta_binary=beta_binary,
            sparsity=1.0 - beta_continuous,
            binary_switch_rate=binary_switch_rate,
            mean_persistence_window=mean_persistence_window,
            summary=(
                f"beta={beta_continuous:.3f} binary={beta_binary} "
                f"sparsity={1.0 - beta_continuous:.3f}"
            ),
        )

    def compute_beta(
        self,
        *,
        previous_code: tuple[float, ...],
        latent_mean: tuple[float, ...],
        latent_scale: tuple[float, ...],
        switch_weights: tuple[float, ...],
        switch_bias: float,
        memory_signal: float,
        reflection_signal: float,
    ) -> float:
        return self.compute_decision(
            previous_code=previous_code,
            z_tilde=latent_mean,
            posterior_std=latent_scale,
            switch_weights=switch_weights,
            switch_bias=switch_bias,
            memory_signal=memory_signal,
            reflection_signal=reflection_signal,
        ).beta_continuous


class ResidualDecoder:
    def decode(
        self,
        *,
        latent_code: tuple[float, ...],
        decoder_matrix: tuple[tuple[float, ...], ...],
        hidden_matrix: tuple[tuple[float, ...], ...] = (
            (0.60, 0.25, 0.15),
            (0.20, 0.60, 0.20),
            (0.15, 0.25, 0.60),
        ),
    ) -> DecoderControl:
        hidden = tuple(
            clamp_unit(sum(weight * value for weight, value in zip(row, latent_code, strict=True)))
            for row in hidden_matrix
        )
        decoder_output = tuple(
            clamp_unit(sum(weight * value for weight, value in zip(row, hidden, strict=True)))
            for row in decoder_matrix
        )
        applied_control = tuple(
            clamp_unit(0.65 * latent_code[index] + 0.35 * decoder_output[index]) for index in range(3)
        )
        return DecoderControl(
            decoder_output=decoder_output,
            applied_control=applied_control,
            summary=(
                f"decoder_output={tuple(round(value, 3) for value in decoder_output)} "
                f"applied={tuple(round(value, 3) for value in applied_control)}"
            ),
        )


def classify_latent_action(
    *,
    latent_code: tuple[float, ...],
    decoder_control: tuple[float, ...],
    prototypes: tuple[tuple[str, tuple[float, ...]], ...],
) -> tuple[str, str]:
    best_label = "stabilize_controller"
    best_score = float("-inf")
    for label, prototype in prototypes:
        score = sum(value * target for value, target in zip(latent_code, prototype, strict=True))
        if score > best_score:
            best_label = label
            best_score = score
    decoder_summary = (
        f"decoder_control={tuple(round(value, 3) for value in decoder_control)} "
        f"label={best_label} score={best_score:.3f}"
    )
    return (best_label, decoder_summary)
