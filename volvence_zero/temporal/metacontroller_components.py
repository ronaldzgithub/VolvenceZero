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


@dataclass(frozen=True)
class DiscoveredActionFamily:
    family_id: str
    latent_centroid: tuple[float, ...]
    decoder_centroid: tuple[float, ...]
    support: int
    stability: float
    switch_bias: float
    mean_posterior_drift: float = 0.0
    mean_persistence_window: float = 0.0
    reuse_streak: int = 0
    stagnation_pressure: float = 0.0
    monopoly_pressure: float = 0.0
    competition_score: float = 0.0
    outcome_history: tuple[float, ...] = ()
    outcome_driven_score: float = 0.0
    long_term_payoff: float = 0.5
    delayed_credit_sum: float = 0.0
    summary: str = ""


@dataclass(frozen=True)
class SwitchGateStats:
    beta_histogram: tuple[int, ...]
    switch_frequency: float
    mean_persistence_steps: float
    observation_count: int


@dataclass(frozen=True)
class FamilyCompetitionState:
    ranked_families: tuple[tuple[str, float], ...]
    top1_share: float
    monopoly_alert: bool
    collapse_alert: bool


@dataclass(frozen=True)
class ActionFamilyObservation:
    latent_code: tuple[float, ...]
    decoder_control: tuple[float, ...]
    switch_gate: float
    posterior_drift: float
    persistence_window: float


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


def _cosine_similarity(lhs: tuple[float, ...], rhs: tuple[float, ...]) -> float:
    numerator = sum(left * right for left, right in zip(lhs, rhs, strict=True))
    lhs_norm = sum(value * value for value in lhs) ** 0.5
    rhs_norm = sum(value * value for value in rhs) ** 0.5
    if lhs_norm == 0.0 or rhs_norm == 0.0:
        return 0.0
    return numerator / (lhs_norm * rhs_norm)


def _blend_centroid(
    previous: tuple[float, ...],
    observation: tuple[float, ...],
    *,
    support: int,
) -> tuple[float, ...]:
    blend = 1.0 / max(support, 1)
    return tuple(
        clamp_unit(prev * (1.0 - blend) + current * blend)
        for prev, current in zip(previous, observation, strict=True)
    )


def _dominant_axis(values: tuple[float, ...]) -> str:
    if not values:
        return "unknown"
    index = max(range(len(values)), key=lambda i: values[i])
    if index == 0:
        return "world"
    if index == 1:
        return "self"
    return "shared"


def _family_match_score(
    family: DiscoveredActionFamily,
    observation: ActionFamilyObservation,
) -> float:
    latent_similarity = _cosine_similarity(observation.latent_code, family.latent_centroid)
    decoder_similarity = _cosine_similarity(observation.decoder_control, family.decoder_centroid)
    switch_alignment = 1.0 - abs(observation.switch_gate - family.switch_bias)
    base = (
        latent_similarity * 0.52
        + decoder_similarity * 0.28
        + family.stability * 0.10
        + clamp_unit(switch_alignment) * 0.06
        + clamp_unit(1.0 - abs(observation.posterior_drift - family.mean_posterior_drift)) * 0.04
        + family.competition_score * 0.04
        - family.monopoly_pressure * 0.05
        - family.stagnation_pressure * 0.03
    )
    outcome_bonus = family.outcome_driven_score * 0.15 if family.outcome_history else 0.0
    return base + outcome_bonus


def _family_summary(
    family: DiscoveredActionFamily,
    *,
    prefix: str,
) -> str:
    return (
        f"{prefix} dominant_axis={_dominant_axis(family.decoder_centroid)} "
        f"support={family.support} stability={family.stability:.3f} "
        f"drift={family.mean_posterior_drift:.3f} persistence={family.mean_persistence_window:.3f} "
        f"reuse_streak={family.reuse_streak} stagnation={family.stagnation_pressure:.3f} "
        f"monopoly={family.monopoly_pressure:.3f} competition={family.competition_score:.3f}"
    )


def _family_from_observation(
    *,
    family_id: str,
    observation: ActionFamilyObservation,
    support: int = 1,
    stability: float = 1.0,
) -> DiscoveredActionFamily:
    return DiscoveredActionFamily(
        family_id=family_id,
        latent_centroid=observation.latent_code,
        decoder_centroid=observation.decoder_control,
        support=support,
        stability=stability,
        switch_bias=observation.switch_gate,
        mean_posterior_drift=observation.posterior_drift,
        mean_persistence_window=observation.persistence_window,
        summary="",
    )


def _refresh_family_summary(family: DiscoveredActionFamily, *, prefix: str) -> DiscoveredActionFamily:
    return DiscoveredActionFamily(
        family_id=family.family_id,
        latent_centroid=family.latent_centroid,
        decoder_centroid=family.decoder_centroid,
        support=family.support,
        stability=family.stability,
        switch_bias=family.switch_bias,
        mean_posterior_drift=family.mean_posterior_drift,
        mean_persistence_window=family.mean_persistence_window,
        reuse_streak=family.reuse_streak,
        stagnation_pressure=family.stagnation_pressure,
        monopoly_pressure=family.monopoly_pressure,
        competition_score=family.competition_score,
        outcome_history=family.outcome_history,
        outcome_driven_score=family.outcome_driven_score,
        long_term_payoff=family.long_term_payoff,
        delayed_credit_sum=family.delayed_credit_sum,
        summary=_family_summary(family, prefix=prefix),
    )


def _merge_vectors(
    left: tuple[float, ...],
    right: tuple[float, ...],
    *,
    left_weight: float,
    right_weight: float,
) -> tuple[float, ...]:
    total = max(left_weight + right_weight, 1e-6)
    return tuple(
        clamp_unit((l_value * left_weight + r_value * right_weight) / total)
        for l_value, r_value in zip(left, right, strict=True)
    )


def _merge_family_pair(
    left: DiscoveredActionFamily,
    right: DiscoveredActionFamily,
) -> DiscoveredActionFamily:
    merged_history = (left.outcome_history + right.outcome_history)[-12:]
    merged = DiscoveredActionFamily(
        family_id=left.family_id,
        latent_centroid=_merge_vectors(
            left.latent_centroid,
            right.latent_centroid,
            left_weight=float(left.support),
            right_weight=float(right.support),
        ),
        decoder_centroid=_merge_vectors(
            left.decoder_centroid,
            right.decoder_centroid,
            left_weight=float(left.support),
            right_weight=float(right.support),
        ),
        support=left.support + right.support,
        stability=clamp_unit(
            (left.stability * left.support + right.stability * right.support)
            / max(left.support + right.support, 1)
        ),
        switch_bias=clamp_unit(
            (left.switch_bias * left.support + right.switch_bias * right.support)
            / max(left.support + right.support, 1)
        ),
        mean_posterior_drift=clamp_unit(
            (left.mean_posterior_drift * left.support + right.mean_posterior_drift * right.support)
            / max(left.support + right.support, 1)
        ),
        mean_persistence_window=clamp_unit(
            (left.mean_persistence_window * left.support + right.mean_persistence_window * right.support)
            / max(left.support + right.support, 1)
        ),
        reuse_streak=max(left.reuse_streak, right.reuse_streak),
        stagnation_pressure=min(left.stagnation_pressure, right.stagnation_pressure),
        monopoly_pressure=max(left.monopoly_pressure, right.monopoly_pressure),
        competition_score=clamp_unit((left.competition_score + right.competition_score) / 2.0),
        outcome_history=merged_history,
        outcome_driven_score=_outcome_driven_score(merged_history),
        long_term_payoff=clamp_unit(
            (left.long_term_payoff * left.support + right.long_term_payoff * right.support)
            / max(left.support + right.support, 1)
        ),
        delayed_credit_sum=left.delayed_credit_sum + right.delayed_credit_sum,
    )
    return _refresh_family_summary(merged, prefix=f"merged:{left.family_id}+{right.family_id}")


def _outcome_driven_score(history: tuple[float, ...]) -> float:
    if not history:
        return 0.0
    return clamp_unit(sum(history) / len(history))


def build_family_competition_state(
    action_families: tuple[DiscoveredActionFamily, ...],
) -> FamilyCompetitionState:
    if not action_families:
        return FamilyCompetitionState(
            ranked_families=(), top1_share=0.0, monopoly_alert=False, collapse_alert=False,
        )
    total_support = max(sum(f.support for f in action_families), 1)
    ranked = sorted(
        action_families,
        key=lambda f: f.long_term_payoff * 0.4 + f.competition_score * 0.3 + f.stability * 0.3,
        reverse=True,
    )
    ranked_tuples = tuple(
        (f.family_id, round(f.long_term_payoff * 0.4 + f.competition_score * 0.3 + f.stability * 0.3, 4))
        for f in ranked
    )
    top1_share = ranked[0].support / total_support if ranked else 0.0
    monopoly_alert = any(f.monopoly_pressure > 0.7 and f.reuse_streak >= 4 for f in action_families)
    collapse_alert = top1_share > 0.8 and len(action_families) > 1
    return FamilyCompetitionState(
        ranked_families=ranked_tuples,
        top1_share=round(top1_share, 4),
        monopoly_alert=monopoly_alert,
        collapse_alert=collapse_alert,
    )


def update_family_outcome_history(
    families: tuple[DiscoveredActionFamily, ...],
    *,
    family_id: str,
    outcome_value: float,
    max_history: int = 12,
) -> tuple[DiscoveredActionFamily, ...]:
    result: list[DiscoveredActionFamily] = []
    for family in families:
        if family.family_id != family_id:
            result.append(family)
            continue
        new_history = (family.outcome_history + (outcome_value,))[-max_history:]
        result.append(
            _refresh_family_summary(
                DiscoveredActionFamily(
                    family_id=family.family_id,
                    latent_centroid=family.latent_centroid,
                    decoder_centroid=family.decoder_centroid,
                    support=family.support,
                    stability=family.stability,
                    switch_bias=family.switch_bias,
                    mean_posterior_drift=family.mean_posterior_drift,
                    mean_persistence_window=family.mean_persistence_window,
                    reuse_streak=family.reuse_streak,
                    stagnation_pressure=family.stagnation_pressure,
                    monopoly_pressure=family.monopoly_pressure,
                    competition_score=family.competition_score,
                    outcome_history=new_history,
                    outcome_driven_score=_outcome_driven_score(new_history),
                    long_term_payoff=family.long_term_payoff,
                    delayed_credit_sum=family.delayed_credit_sum,
                ),
                prefix="outcome-updated",
            )
        )
    return tuple(result)


def _competition_score(family: DiscoveredActionFamily) -> float:
    support_signal = min(family.support / 4.0, 1.0)
    reuse_penalty = min(family.reuse_streak / 6.0, 1.0)
    return clamp_unit(
        family.stability * 0.28
        + (1.0 - family.monopoly_pressure) * 0.32
        + (1.0 - family.stagnation_pressure) * 0.18
        + support_signal * 0.10
        + (1.0 - reuse_penalty) * 0.12
    )


def _update_family_competition_state(
    action_families: tuple[DiscoveredActionFamily, ...],
    *,
    active_family_id: str,
) -> tuple[DiscoveredActionFamily, ...]:
    total_support = max(sum(family.support for family in action_families), 1)
    updated: list[DiscoveredActionFamily] = []
    for family in action_families:
        is_active = family.family_id == active_family_id
        support_share = family.support / total_support
        reuse_streak = family.reuse_streak + 1 if is_active else 0
        stagnation_pressure = (
            clamp_unit(family.stagnation_pressure * 0.55 + 0.08)
            if is_active
            else clamp_unit(
                family.stagnation_pressure * 0.82
                + 0.18
                + (0.08 if family.support <= 1 else 0.0)
            )
        )
        monopoly_pressure = (
            clamp_unit(
                support_share * 0.52
                + min(reuse_streak / 6.0, 1.0) * 0.48
            )
            if is_active
            else clamp_unit(family.monopoly_pressure * 0.55)
        )
        refreshed = DiscoveredActionFamily(
            family_id=family.family_id,
            latent_centroid=family.latent_centroid,
            decoder_centroid=family.decoder_centroid,
            support=family.support,
            stability=family.stability,
            switch_bias=family.switch_bias,
            mean_posterior_drift=family.mean_posterior_drift,
            mean_persistence_window=family.mean_persistence_window,
            reuse_streak=reuse_streak,
            stagnation_pressure=stagnation_pressure,
            monopoly_pressure=monopoly_pressure,
            outcome_history=family.outcome_history,
            outcome_driven_score=family.outcome_driven_score,
            long_term_payoff=family.long_term_payoff,
            delayed_credit_sum=family.delayed_credit_sum,
        )
        updated.append(
            _refresh_family_summary(
                DiscoveredActionFamily(
                    family_id=refreshed.family_id,
                    latent_centroid=refreshed.latent_centroid,
                    decoder_centroid=refreshed.decoder_centroid,
                    support=refreshed.support,
                    stability=refreshed.stability,
                    switch_bias=refreshed.switch_bias,
                    mean_posterior_drift=refreshed.mean_posterior_drift,
                    mean_persistence_window=refreshed.mean_persistence_window,
                    reuse_streak=refreshed.reuse_streak,
                    stagnation_pressure=refreshed.stagnation_pressure,
                    monopoly_pressure=refreshed.monopoly_pressure,
                    competition_score=_competition_score(refreshed),
                    outcome_history=refreshed.outcome_history,
                    outcome_driven_score=refreshed.outcome_driven_score,
                    long_term_payoff=refreshed.long_term_payoff,
                    delayed_credit_sum=refreshed.delayed_credit_sum,
                ),
                prefix="competitive" if is_active else "idle",
            )
        )
    return tuple(updated)


def _anti_collapse_topology_maintenance(
    action_families: tuple[DiscoveredActionFamily, ...],
    *,
    active_family_id: str,
    max_families: int,
) -> tuple[tuple[DiscoveredActionFamily, ...], tuple[str, ...]]:
    families = list(action_families)
    events: list[str] = []
    active_family = next((family for family in families if family.family_id == active_family_id), None)
    if active_family is not None and active_family.monopoly_pressure > 0.74 and active_family.reuse_streak >= 4:
        if len(families) < max_families:
            challenger_axis = "self" if _dominant_axis(active_family.decoder_centroid) == "world" else "shared"
            challenger_id = max(
                (
                    int(family.family_id.rsplit("_", 1)[-1])
                    for family in families
                    if family.family_id.rsplit("_", 1)[-1].isdigit()
                ),
                default=-1,
            ) + 1
            challenger = _refresh_family_summary(
                DiscoveredActionFamily(
                    family_id=f"discovered_family_{challenger_id}",
                    latent_centroid=_blend_centroid(
                        active_family.latent_centroid,
                        tuple(
                            0.85 if axis_index == (1 if challenger_axis == "self" else 2) else 0.15
                            for axis_index in range(len(active_family.latent_centroid))
                        ),
                        support=max(active_family.support // 4, 1),
                    ),
                    decoder_centroid=_blend_centroid(
                        active_family.decoder_centroid,
                        tuple(
                            0.85 if axis_index == (1 if challenger_axis == "self" else 2) else 0.15
                            for axis_index in range(len(active_family.decoder_centroid))
                        ),
                        support=max(active_family.support // 4, 1),
                    ),
                    support=max(active_family.support // 4, 1),
                    stability=clamp_unit(active_family.stability * 0.82),
                    switch_bias=active_family.switch_bias,
                    mean_posterior_drift=clamp_unit(active_family.mean_posterior_drift + 0.08),
                    mean_persistence_window=clamp_unit(active_family.mean_persistence_window * 0.8),
                    reuse_streak=0,
                    stagnation_pressure=0.0,
                    monopoly_pressure=0.0,
                    competition_score=0.5,
                ),
                prefix=f"anti-collapse-split:{active_family.family_id}",
            )
            families.append(challenger)
            events.append(f"anti-collapse-create:{challenger.family_id}")
    pruned_families: list[DiscoveredActionFamily] = []
    for family in families:
        should_prune = (
            family.stagnation_pressure > 0.78
            and family.support <= 1
            and family.family_id != active_family_id
            and len(families) - len(events) > 1
        )
        if should_prune:
            events.append(f"anti-collapse-prune:{family.family_id}")
            continue
        pruned_families.append(family)
    return (tuple(pruned_families), tuple(events))


def _merge_similar_action_families(
    action_families: tuple[DiscoveredActionFamily, ...],
    *,
    merge_threshold: float = 0.95,
) -> tuple[tuple[DiscoveredActionFamily, ...], int]:
    merged_count = 0
    families = list(action_families)
    index = 0
    while index < len(families):
        cursor = index + 1
        while cursor < len(families):
            left = families[index]
            right = families[cursor]
            pair_score = (
                _cosine_similarity(left.latent_centroid, right.latent_centroid) * 0.6
                + _cosine_similarity(left.decoder_centroid, right.decoder_centroid) * 0.4
            )
            if pair_score >= merge_threshold:
                primary, secondary = (
                    (left, right)
                    if (left.support, left.stability) >= (right.support, right.stability)
                    else (right, left)
                )
                merged_family = _merge_family_pair(primary, secondary)
                if primary.family_id == left.family_id:
                    families[index] = merged_family
                    families.pop(cursor)
                else:
                    families[index] = merged_family
                    families.pop(cursor)
                merged_count += 1
                continue
            cursor += 1
        index += 1
    return (tuple(families), merged_count)


def _prune_action_families(
    action_families: tuple[DiscoveredActionFamily, ...],
    *,
    max_families: int,
) -> tuple[tuple[DiscoveredActionFamily, ...], int]:
    if len(action_families) <= 1:
        return (action_families, 0)
    ranked = sorted(
        action_families,
        key=lambda family: (
            family.support,
            family.stability,
            1.0 - family.mean_posterior_drift,
            family.mean_persistence_window,
        ),
        reverse=True,
    )
    kept: list[DiscoveredActionFamily] = []
    pruned_count = 0
    for family in ranked:
        should_keep = (
            len(kept) == 0
            or family.support > 1
            or family.stability >= 0.6
            or family.competition_score >= 0.45
            or (
                len(kept) < max_families
                and family.support > 0
                and family.stability >= 0.35
                and family.stagnation_pressure < 0.82
            )
        )
        if should_keep:
            kept.append(family)
        else:
            pruned_count += 1
    return (tuple(kept[:max_families]), pruned_count)


def classify_latent_action(
    *,
    observation: ActionFamilyObservation,
    action_families: tuple[DiscoveredActionFamily, ...],
) -> tuple[str, str, float]:
    best_label = "unassigned_action"
    best_score = float("-inf")
    best_family: DiscoveredActionFamily | None = None
    for family in action_families:
        score = _family_match_score(family, observation)
        if score > best_score:
            best_label = family.family_id
            best_score = score
            best_family = family
    dominant_axis = (
        _dominant_axis(best_family.decoder_centroid if best_family is not None else observation.decoder_control)
    )
    decoder_summary = (
        f"decoder_control={tuple(round(value, 3) for value in observation.decoder_control)} "
        f"label={best_label} score={best_score:.3f} dominant_axis={dominant_axis}"
    )
    return (best_label, decoder_summary, best_score)


def discover_latent_action_family(
    *,
    observation: ActionFamilyObservation,
    action_families: tuple[DiscoveredActionFamily, ...],
    structure_frozen: bool,
    allow_topology_maintenance: bool = True,
    max_families: int = 6,
    similarity_threshold: float = 0.84,
    split_similarity_threshold: float = 0.93,
    split_support_threshold: int = 6,
) -> tuple[tuple[DiscoveredActionFamily, ...], str, str]:
    if not action_families:
        family = _refresh_family_summary(
            _family_from_observation(
                family_id="discovered_family_0",
                observation=observation,
                support=1,
                stability=1.0,
            ),
            prefix="created",
        )
        label, summary, _ = classify_latent_action(
            observation=observation,
            action_families=(family,),
        )
        return ((family,), label, summary)
    best_label, _, best_score = classify_latent_action(
        observation=observation,
        action_families=action_families,
    )
    updated_families = list(action_families)
    best_index = next(
        index for index, family in enumerate(updated_families) if family.family_id == best_label
    )
    maintenance_events: list[str] = []
    if not structure_frozen and best_score < similarity_threshold and len(updated_families) < max_families:
        next_id = max(
            (
                int(family.family_id.rsplit("_", 1)[-1])
                for family in updated_families
                if family.family_id.rsplit("_", 1)[-1].isdigit()
            ),
            default=-1,
        ) + 1
        created = _refresh_family_summary(
            _family_from_observation(
                family_id=f"discovered_family_{next_id}",
                observation=observation,
                support=1,
                stability=1.0,
            ),
            prefix="created",
        )
        updated_families.append(created)
        maintenance_events.append(f"create:{created.family_id}")
    elif not structure_frozen:
        current = updated_families[best_index]
        should_split = (
            allow_topology_maintenance
            and len(updated_families) < max_families
            and current.support >= split_support_threshold
            and best_score < split_similarity_threshold + current.monopoly_pressure * 0.05
            and observation.posterior_drift > max(0.16, current.mean_posterior_drift + 0.05)
            and (
                observation.persistence_window + 0.25 < max(current.mean_persistence_window, 0.75)
                or current.reuse_streak >= 4
                or current.monopoly_pressure > 0.70
            )
        )
        if should_split:
            next_id = max(
                (
                    int(family.family_id.rsplit("_", 1)[-1])
                    for family in updated_families
                    if family.family_id.rsplit("_", 1)[-1].isdigit()
                ),
                default=-1,
            ) + 1
            updated_families[best_index] = _refresh_family_summary(
                DiscoveredActionFamily(
                    family_id=current.family_id,
                    latent_centroid=current.latent_centroid,
                    decoder_centroid=current.decoder_centroid,
                    support=max(current.support - 1, 1),
                    stability=clamp_unit(current.stability * 0.92),
                    switch_bias=current.switch_bias,
                    mean_posterior_drift=current.mean_posterior_drift,
                    mean_persistence_window=current.mean_persistence_window,
                    reuse_streak=current.reuse_streak,
                    stagnation_pressure=current.stagnation_pressure,
                    monopoly_pressure=current.monopoly_pressure,
                    competition_score=current.competition_score,
                ),
                prefix=f"split-parent:{current.family_id}",
            )
            child = _refresh_family_summary(
                _family_from_observation(
                    family_id=f"discovered_family_{next_id}",
                    observation=observation,
                    support=max(1, current.support // 3),
                    stability=clamp_unit(0.72 + observation.posterior_drift * 0.15),
                ),
                prefix=f"split-child:{current.family_id}",
            )
            updated_families.append(child)
            maintenance_events.append(f"split:{current.family_id}->{child.family_id}")
        else:
            support = current.support + 1
            latent_centroid = _blend_centroid(
                current.latent_centroid,
                observation.latent_code,
                support=support,
            )
            decoder_centroid = _blend_centroid(
                current.decoder_centroid,
                observation.decoder_control,
                support=support,
            )
            updated_families[best_index] = _refresh_family_summary(
                DiscoveredActionFamily(
                    family_id=current.family_id,
                    latent_centroid=latent_centroid,
                    decoder_centroid=decoder_centroid,
                    support=support,
                    stability=clamp_unit(current.stability * 0.72 + best_score * 0.28),
                    switch_bias=clamp_unit(current.switch_bias * 0.75 + observation.switch_gate * 0.25),
                    mean_posterior_drift=clamp_unit(
                        current.mean_posterior_drift * 0.75 + observation.posterior_drift * 0.25
                    ),
                    mean_persistence_window=clamp_unit(
                        current.mean_persistence_window * 0.75 + observation.persistence_window * 0.25
                    ),
                    reuse_streak=current.reuse_streak,
                    stagnation_pressure=current.stagnation_pressure,
                    monopoly_pressure=current.monopoly_pressure,
                    competition_score=current.competition_score,
                ),
                prefix="reused",
            )
            maintenance_events.append(f"reuse:{current.family_id}")
    if allow_topology_maintenance and updated_families:
        merged_families, merged_count = _merge_similar_action_families(tuple(updated_families))
        if merged_count:
            maintenance_events.append(f"merge:{merged_count}")
        pruned_families, pruned_count = _prune_action_families(
            merged_families,
            max_families=max_families,
        )
        if pruned_count:
            maintenance_events.append(f"prune:{pruned_count}")
        updated_families = list(pruned_families)
    families_tuple = tuple(updated_families)
    label, summary, _ = classify_latent_action(
        observation=observation,
        action_families=families_tuple,
    )
    families_tuple = _update_family_competition_state(
        families_tuple,
        active_family_id=label,
    )
    families_tuple, anti_collapse_events = _anti_collapse_topology_maintenance(
        families_tuple,
        active_family_id=label,
        max_families=max_families,
    )
    if anti_collapse_events:
        maintenance_events.extend(anti_collapse_events)
        label, summary, _ = classify_latent_action(
            observation=observation,
            action_families=families_tuple,
        )
        families_tuple = _update_family_competition_state(
            families_tuple,
            active_family_id=label,
        )
    if maintenance_events:
        summary = f"{summary} lifecycle={','.join(maintenance_events)}"
    return (families_tuple, label, summary)


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
