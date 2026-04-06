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
        cms_online_fast: tuple[float, ...] | None = None,
        cms_session_medium: tuple[float, ...] | None = None,
        cms_background_slow: tuple[float, ...] | None = None,
    ) -> EncodedSequence:
        sequence = residual_sequence_from_snapshot(substrate_snapshot)
        step_vectors = tuple(
            summarize_residual_activations(step.residual_activations, step.feature_surface) for step in sequence
        )
        if not step_vectors:
            step_vectors = ((0.0, 0.0, 0.0),)
        cms_context = self._cms_context(
            cms_online_fast=cms_online_fast,
            cms_session_medium=cms_session_medium,
            cms_background_slow=cms_background_slow,
            dim=len(step_vectors[0]),
        )
        hidden_state = previous_hidden_state
        hidden_history: list[tuple[float, ...]] = []
        for step_vector in step_vectors:
            augmented_input = tuple(
                step_vector[index] * 0.80 + cms_context[index] * 0.20
                for index in range(len(step_vector))
            )
            hidden_state = tuple(
                clamp_unit(
                    sum(weight * hidden_value for weight, hidden_value in zip(row, hidden_state, strict=True)) * 0.55
                    + augmented_input[index] * 0.45
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
        prior_mean = self._cms_informed_prior_mean(
            previous_hidden_state=previous_hidden_state,
            cms_session_medium=cms_session_medium,
            cms_background_slow=cms_background_slow,
        )
        prior_std = self._cms_informed_prior_std(
            previous_hidden_state=previous_hidden_state,
            average_vector=average_vector,
            cms_session_medium=cms_session_medium,
            cms_background_slow=cms_background_slow,
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

    def _cms_context(
        self,
        *,
        cms_online_fast: tuple[float, ...] | None,
        cms_session_medium: tuple[float, ...] | None,
        cms_background_slow: tuple[float, ...] | None,
        dim: int,
    ) -> tuple[float, ...]:
        zero = tuple(0.0 for _ in range(dim))
        fast = cms_online_fast or zero
        medium = cms_session_medium or zero
        slow = cms_background_slow or zero
        return tuple(
            clamp_unit(fast[i] * 0.50 + medium[i] * 0.30 + slow[i] * 0.20)
            for i in range(dim)
        )

    def _cms_informed_prior_mean(
        self,
        *,
        previous_hidden_state: tuple[float, ...],
        cms_session_medium: tuple[float, ...] | None,
        cms_background_slow: tuple[float, ...] | None,
    ) -> tuple[float, ...]:
        if cms_session_medium is None and cms_background_slow is None:
            return tuple(clamp_unit(previous_hidden_state[i] * 0.35) for i in range(3))
        slow = cms_background_slow or (0.0, 0.0, 0.0)
        medium = cms_session_medium or (0.0, 0.0, 0.0)
        return tuple(
            clamp_unit(
                previous_hidden_state[i] * 0.20
                + slow[i] * 0.45
                + medium[i] * 0.35
            )
            for i in range(3)
        )

    def _cms_informed_prior_std(
        self,
        *,
        previous_hidden_state: tuple[float, ...],
        average_vector: tuple[float, ...],
        cms_session_medium: tuple[float, ...] | None,
        cms_background_slow: tuple[float, ...] | None,
    ) -> tuple[float, ...]:
        base_std = tuple(
            max(0.05, 1.0 - clamp_unit(abs(previous_hidden_state[i] - average_vector[i]) * 0.5))
            for i in range(3)
        )
        if cms_session_medium is None and cms_background_slow is None:
            return base_std
        slow = cms_background_slow or (0.0, 0.0, 0.0)
        medium = cms_session_medium or (0.0, 0.0, 0.0)
        cms_evidence = sum(abs(v) for v in slow) + sum(abs(v) for v in medium)
        contraction = clamp_unit(cms_evidence / 6.0) * 0.35
        return tuple(max(0.05, s * (1.0 - contraction)) for s in base_std)

    def encoder_output_for_cms(self, encoded: EncodedSequence) -> tuple[float, ...]:
        """Signal to feed back from encoder into CMS observation."""
        return tuple(
            clamp_unit(
                encoded.posterior.posterior_mean[i] * 0.6
                + encoded.posterior.z_tilde[i] * 0.4
            )
            for i in range(len(encoded.posterior.posterior_mean))
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


# ---------------------------------------------------------------------------
#  N-dim components (P15): GRU encoder, element-wise switch, FFN decoder
# ---------------------------------------------------------------------------

from volvence_zero.temporal.tensor_ops import (
    Mat,
    Vec,
    dot,
    ffn_2layer,
    gru_cell,
    init_ffn_params,
    init_gru_params,
    vec_abs,
    vec_add,
    vec_clamp,
    vec_mean,
    vec_mul,
    vec_norm,
    vec_scale,
    vec_sigmoid,
    vec_sub,
    vec_tanh,
    zeros,
    mat_vec,
    rand_mat,
    rand_vec,
    identity_mat,
)


DEFAULT_N_Z = 16


@dataclass(frozen=True)
class NdimGRUParams:
    W_z: Mat
    U_z: Mat
    b_z: Vec
    W_r: Mat
    U_r: Mat
    b_r: Vec
    W_h: Mat
    U_h: Mat
    b_h: Vec


@dataclass(frozen=True)
class NdimFFNParams:
    W1: Mat
    b1: Vec
    W2: Mat
    b2: Vec


def _project_to_ndim(raw: tuple[float, ...], n: int) -> Vec:
    """Project an arbitrary-length raw vector to n dimensions via tiling/truncation."""
    if not raw:
        return zeros(n)
    if len(raw) >= n:
        return raw[:n]
    repeats = (n // len(raw)) + 1
    extended = raw * repeats
    return extended[:n]


def _summarize_substrate_ndim(
    substrate_snapshot: SubstrateSnapshot,
    n: int,
) -> tuple[Vec, ...]:
    """Extract n-dim step vectors from substrate."""
    sequence = residual_sequence_from_snapshot(substrate_snapshot)
    result: list[Vec] = []
    for step in sequence:
        raw_values: list[float] = []
        for act in step.residual_activations:
            raw_values.extend(act.activation)
        if not raw_values:
            for feat in step.feature_surface:
                raw_values.extend(feat.values)
        if not raw_values:
            raw_values = [0.0]
        result.append(_project_to_ndim(tuple(raw_values), n))
    if not result:
        result.append(zeros(n))
    return tuple(result)


class NdimSequenceEncoder:
    """GRU-based sequence encoder operating in n_z-dimensional latent space."""

    def __init__(self, *, n_z: int = DEFAULT_N_Z, n_input: int | None = None, seed: int = 42) -> None:
        self._n_z = n_z
        self._n_input = n_input or n_z
        params = init_gru_params(self._n_input, n_z, seed=seed)
        self._gru = NdimGRUParams(
            W_z=params["W_z"], U_z=params["U_z"], b_z=params["b_z"],
            W_r=params["W_r"], U_r=params["U_r"], b_r=params["b_r"],
            W_h=params["W_h"], U_h=params["U_h"], b_h=params["b_h"],
        )
        self._posterior_proj = rand_mat(n_z, n_z, scale=0.1, seed=seed + 10)
        self._posterior_std_proj = rand_mat(n_z, n_z, scale=0.05, seed=seed + 11)

    @property
    def n_z(self) -> int:
        return self._n_z

    def encode(
        self,
        *,
        substrate_snapshot: SubstrateSnapshot,
        previous_hidden_state: Vec | None = None,
        cms_context: Vec | None = None,
    ) -> EncodedSequence:
        h = previous_hidden_state or zeros(self._n_z)
        step_vectors = _summarize_substrate_ndim(substrate_snapshot, self._n_input)
        hidden_history: list[Vec] = []
        for step_vec in step_vectors:
            if cms_context is not None:
                aug = vec_add(vec_scale(step_vec, 0.8), vec_scale(cms_context, 0.2))
            else:
                aug = step_vec
            h = gru_cell(
                x=aug, h_prev=h,
                W_z=self._gru.W_z, U_z=self._gru.U_z, b_z=self._gru.b_z,
                W_r=self._gru.W_r, U_r=self._gru.U_r, b_r=self._gru.b_r,
                W_h=self._gru.W_h, U_h=self._gru.U_h, b_h=self._gru.b_h,
            )
            hidden_history.append(h)
        avg_hidden = tuple(
            sum(hh[i] for hh in hidden_history) / len(hidden_history)
            for i in range(self._n_z)
        )
        prior_mean = vec_scale(h if previous_hidden_state is None else previous_hidden_state, 0.35)
        prior_mean = vec_clamp(prior_mean, 0.0, 1.0)
        prior_std_raw = tuple(
            max(0.05, 1.0 - abs(prior_mean[i] - avg_hidden[i]) * 0.5)
            for i in range(self._n_z)
        )
        posterior_mean = vec_clamp(
            vec_add(
                vec_scale(mat_vec(self._posterior_proj, h), 0.7),
                vec_scale(avg_hidden, 0.3),
            ),
            0.0, 1.0,
        )
        posterior_std = vec_clamp(
            vec_abs(mat_vec(self._posterior_std_proj, h)),
            0.05, 0.95,
        )
        sample_noise = vec_clamp(
            vec_sub(avg_hidden, vec_scale(posterior_mean, 0.5)),
            -1.0, 1.0,
        )
        z_tilde = vec_clamp(
            vec_add(posterior_mean, vec_scale(vec_mul(posterior_std, sample_noise), 0.5)),
            0.0, 1.0,
        )
        drift = max(abs(h[i] - (previous_hidden_state or zeros(self._n_z))[i]) for i in range(self._n_z))
        return EncodedSequence(
            posterior=PosteriorState(
                prior_mean=prior_mean,
                prior_std=prior_std_raw,
                posterior_mean=posterior_mean,
                posterior_std=posterior_std,
                sample_noise=sample_noise,
                z_tilde=z_tilde,
                hidden_state=h,
                posterior_drift=drift,
            ),
            sequence_length=len(step_vectors),
            summary=(
                f"ndim_encoder n_z={self._n_z} len={len(step_vectors)} "
                f"drift={drift:.3f} h_norm={vec_norm(h):.3f}"
            ),
        )


class NdimSwitchUnit:
    """Element-wise switch gate β_t ∈ [0,1]^{n_z} with a learned gate network."""

    def __init__(self, *, n_z: int = DEFAULT_N_Z, seed: int = 42) -> None:
        self._n_z = n_z
        ffn_params = init_ffn_params(n_z * 2, n_z, n_z, seed=seed + 20)
        self._ffn = NdimFFNParams(
            W1=ffn_params["W1"], b1=ffn_params["b1"],
            W2=ffn_params["W2"], b2=ffn_params["b2"],
        )

    def compute(
        self,
        *,
        z_tilde: Vec,
        previous_code: Vec,
        memory_signal: float = 0.0,
        reflection_signal: float = 0.0,
    ) -> tuple[Vec, Vec, float]:
        """Returns (beta_continuous, beta_binary, scalar_beta_mean)."""
        delta = vec_abs(vec_sub(z_tilde, previous_code))
        gate_input = delta + z_tilde
        raw = ffn_2layer(x=gate_input, W1=self._ffn.W1, b1=self._ffn.b1, W2=self._ffn.W2, b2=self._ffn.b2)
        bias = memory_signal * 0.1 + reflection_signal * 0.2
        beta_continuous = vec_sigmoid(vec_add(raw, tuple(bias for _ in range(self._n_z))))
        threshold = 0.55
        beta_binary = tuple(1.0 if b >= threshold else 0.0 for b in beta_continuous)
        scalar_mean = vec_mean(beta_continuous)
        return beta_continuous, beta_binary, scalar_mean


class NdimResidualDecoder:
    """2-layer FFN decoder producing n_z-dimensional applied control."""

    def __init__(self, *, n_z: int = DEFAULT_N_Z, seed: int = 42) -> None:
        self._n_z = n_z
        ffn_params = init_ffn_params(n_z, n_z, n_z, seed=seed + 30)
        self._ffn = NdimFFNParams(
            W1=ffn_params["W1"], b1=ffn_params["b1"],
            W2=ffn_params["W2"], b2=ffn_params["b2"],
        )

    def decode(self, *, latent_code: Vec) -> DecoderControl:
        decoder_output = ffn_2layer(
            x=latent_code,
            W1=self._ffn.W1, b1=self._ffn.b1,
            W2=self._ffn.W2, b2=self._ffn.b2,
        )
        applied_control = vec_clamp(
            vec_add(vec_scale(latent_code, 0.65), vec_scale(decoder_output, 0.35)),
            0.0, 1.0,
        )
        return DecoderControl(
            decoder_output=decoder_output,
            applied_control=applied_control,
            summary=f"ndim_decoder n_z={self._n_z} ctrl_norm={vec_norm(applied_control):.3f}",
        )
