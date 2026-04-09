from __future__ import annotations

import math
from dataclasses import dataclass

from volvence_zero.substrate import (
    ResidualSequenceStep,
    SubstrateSnapshot,
    SurfaceKind,
    TrainingTrace,
)
from volvence_zero.temporal.interface import FullLearnedTemporalPolicy
from volvence_zero.temporal.metacontroller_components import (
    DecoderControl,
    EncodedSequence,
    NdimResidualDecoder,
    NdimSequenceEncoder,
    NdimSwitchUnit,
    PosteriorState,
    ResidualDecoder,
    SequenceEncoder,
    SwitchGateStats,
    SwitchUnit,
    summarize_residual_activations,
)
from volvence_zero.temporal.noncausal_embedder import NonCausalSequenceEmbedder


from volvence_zero.temporal.m3_optimizer import M3Optimizer


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class SSLTrainingReport:
    trace_id: str
    prediction_loss: float
    kl_loss: float
    total_loss: float
    posterior_drift: float
    trained_steps: int
    m3_slow_momentum_signal: tuple[float, ...] = ()
    noncausal_kl_tightening: float = 0.0
    noncausal_information_content: float = 0.0
    switch_gate_stats: SwitchGateStats | None = None
    description: str = ""


class MetacontrollerSSLTrainer:
    """Small Eq.3-style training loop over residual traces."""

    def __init__(self, *, n_z: int = 3, alpha: float = 0.1) -> None:
        self._n_z = n_z
        self._alpha = alpha
        self._encoder = SequenceEncoder()
        self._decoder = ResidualDecoder()
        self._switch = SwitchUnit()
        self._ndim_encoder: "NdimSequenceEncoder | None" = None
        self._ndim_decoder: "NdimResidualDecoder | None" = None
        self._ndim_switch: "NdimSwitchUnit | None" = None
        if n_z > 3:
            from volvence_zero.temporal.metacontroller_components import (
                NdimResidualDecoder as _NRD,
                NdimSequenceEncoder as _NSE,
                NdimSwitchUnit as _NSU,
            )
            self._ndim_encoder = _NSE(n_z=n_z)
            self._ndim_decoder = _NRD(n_z=n_z)
            self._ndim_switch = _NSU(n_z=n_z)
        self._m3_encoder = M3Optimizer(num_groups=n_z, group_dim=n_z, slow_interval=3)
        self._m3_decoder = M3Optimizer(num_groups=n_z, group_dim=n_z, slow_interval=3)
        self._noncausal_embedder = NonCausalSequenceEmbedder(n_z=n_z)

    def optimize(
        self,
        *,
        policy: FullLearnedTemporalPolicy,
        trace: TrainingTrace,
    ) -> SSLTrainingReport:
        if len(trace.steps) < 2:
            policy.parameter_store.record_ssl_metrics(total_loss=0.0, kl_loss=0.0)
            return SSLTrainingReport(
                trace_id=trace.trace_id,
                prediction_loss=0.0,
                kl_loss=0.0,
                total_loss=0.0,
                posterior_drift=0.0,
                trained_steps=0,
                description="SSL trainer skipped because the trace is shorter than 2 steps.",
            )

        prediction_total = 0.0
        kl_total = 0.0
        trained_steps = 0
        n = self._n_z
        latest_mean = tuple(0.0 for _ in range(n))
        latest_scale = tuple(0.0 for _ in range(n))
        latest_prior_mean = tuple(0.0 for _ in range(n))
        latest_prior_std = tuple(1.0 for _ in range(n))
        latest_z_tilde = tuple(0.0 for _ in range(n))
        latest_decoder = tuple(0.0 for _ in range(n))
        latest_label = "unassigned_action"
        posterior_drift_total = 0.0
        store = policy.parameter_store
        store.set_learning_phase("ssl", structure_frozen=False)
        previous_hidden_state = store.latest_posterior_hidden_state
        previous_code = tuple(0.0 for _ in range(n))
        previous_steps = 0

        full_substrate = self._snapshot_from_prefix(trace=trace, prefix=trace.steps)
        noncausal_embedding = self._noncausal_embedder.embed(substrate_snapshot=full_substrate)
        kl_tightening_total = 0.0
        beta_values: list[float] = []
        switch_count = 0

        for next_index in range(1, len(trace.steps)):
            prefix = trace.steps[:next_index]
            next_step = trace.steps[next_index]
            substrate_snapshot = self._snapshot_from_prefix(trace=trace, prefix=prefix)
            if self._ndim_encoder is not None:
                encoded = self._ndim_encoder.encode(
                    substrate_snapshot=substrate_snapshot,
                    previous_hidden_state=previous_hidden_state,
                )
            else:
                encoded = self._encoder.encode(
                    substrate_snapshot=substrate_snapshot,
                    encoder_weights=store.encoder_weights,
                    recurrence_weights=store.encoder_recurrence,
                    previous_hidden_state=previous_hidden_state,
                )
            enrichment = self._noncausal_embedder.enrich_posterior(
                causal_mean=encoded.posterior.posterior_mean,
                causal_std=encoded.posterior.posterior_std,
                embedding=noncausal_embedding,
            )
            kl_tightening_total += enrichment.kl_tightening
            enriched_encoded = EncodedSequence(
                posterior=PosteriorState(
                    prior_mean=encoded.posterior.prior_mean,
                    prior_std=encoded.posterior.prior_std,
                    posterior_mean=enrichment.enriched_mean,
                    posterior_std=enrichment.enriched_std,
                    sample_noise=encoded.posterior.sample_noise,
                    z_tilde=encoded.posterior.z_tilde,
                    hidden_state=encoded.posterior.hidden_state,
                    posterior_drift=encoded.posterior.posterior_drift,
                ),
                sequence_length=encoded.sequence_length,
                summary=encoded.summary,
            )
            if self._ndim_decoder is not None:
                if self._ndim_switch is None:
                    raise RuntimeError("Ndim switch unit must be available when n_z > 3.")
                beta_cont, _, scalar_beta = self._ndim_switch.compute(
                    z_tilde=encoded.z_tilde,
                    previous_code=previous_code,
                    memory_signal=0.0,
                    reflection_signal=0.0,
                )
                latent_code = tuple(
                    _clamp(
                        beta_cont[index] * encoded.z_tilde[index]
                        + (1.0 - beta_cont[index]) * previous_code[index]
                    )
                    for index in range(len(encoded.z_tilde))
                )
                is_switching = scalar_beta >= store.beta_threshold
                persistence_window = 0.0 if is_switching else float(previous_steps + 1)
                decoder_control = self._ndim_decoder.decode(latent_code=latent_code)
            else:
                switch_decision = self._switch.compute_decision(
                    previous_code=previous_code,
                    z_tilde=encoded.z_tilde,
                    posterior_std=encoded.latent_scale,
                    switch_weights=store.switch_weights,
                    switch_bias=store.switch_bias,
                    memory_signal=0.0,
                    reflection_signal=0.0,
                    previous_binary=0,
                    previous_steps_since_switch=previous_steps,
                )
                scalar_beta = switch_decision.beta_continuous
                latent_code = tuple(
                    _clamp(
                        scalar_beta * current + (1.0 - scalar_beta) * previous
                    )
                    for current, previous in zip(encoded.z_tilde, previous_code, strict=True)
                )
                is_switching = bool(switch_decision.beta_binary)
                persistence_window = switch_decision.mean_persistence_window
                decoder_control = self._decoder.decode(
                    latent_code=latent_code,
                    decoder_matrix=store.decoder_matrix,
                    hidden_matrix=store.decoder_hidden,
                )
            target_action = summarize_residual_activations(next_step.residual_activations, next_step.feature_surface)
            if self._n_z > 3:
                from volvence_zero.temporal.metacontroller_components import _project_to_ndim
                target_action = _project_to_ndim(target_action, self._n_z)
            prediction_loss = self._action_prediction_loss(
                target_action=target_action,
                decoder_control=decoder_control,
            )
            kl_loss = self._kl_to_prior(encoded=enriched_encoded)
            total_loss = prediction_loss + kl_loss * self._alpha
            beta_values.append(scalar_beta)
            if is_switching:
                switch_count += 1
            prediction_total += prediction_loss
            kl_total += kl_loss
            posterior_drift_total += encoded.posterior.posterior_drift
            trained_steps += 1
            self._apply_training_step(
                policy=policy,
                target_action=target_action,
                encoded=enriched_encoded,
                decoder_control=decoder_control,
                prediction_loss=prediction_loss,
                kl_loss=kl_loss,
            )
            latest_mean = enriched_encoded.posterior.posterior_mean
            latest_scale = enriched_encoded.posterior.posterior_std
            latest_prior_mean = encoded.posterior.prior_mean
            latest_prior_std = encoded.posterior.prior_std
            latest_z_tilde = latent_code
            latest_decoder = decoder_control.applied_control
            latest_label, _ = store.discover_action_family(
                latent_code=latent_code,
                decoder_control=decoder_control.applied_control,
                switch_gate=scalar_beta,
                posterior_drift=encoded.posterior.posterior_drift,
                persistence_window=persistence_window,
            )
            previous_hidden_state = encoded.posterior.hidden_state
            previous_code = latent_code
            previous_steps = 0 if is_switching else previous_steps + 1

        avg_prediction = prediction_total / trained_steps
        avg_kl = kl_total / trained_steps
        avg_drift = posterior_drift_total / trained_steps
        total_loss = avg_prediction + avg_kl * self._alpha
        histogram = [0] * 10
        for bv in beta_values:
            bin_index = min(int(bv * 10), 9)
            histogram[bin_index] += 1
        switch_frequency = switch_count / trained_steps if trained_steps else 0.0
        persistence_steps_sum = 0.0
        persist_count = 0
        current_persist = 0
        for bv in beta_values:
            if bv >= 0.55:
                if current_persist > 0:
                    persistence_steps_sum += current_persist
                    persist_count += 1
                current_persist = 0
            else:
                current_persist += 1
        if current_persist > 0:
            persistence_steps_sum += current_persist
            persist_count += 1
        mean_persistence = persistence_steps_sum / persist_count if persist_count else 0.0
        gate_stats = SwitchGateStats(
            beta_histogram=tuple(histogram),
            switch_frequency=round(switch_frequency, 4),
            mean_persistence_steps=round(mean_persistence, 4),
            observation_count=trained_steps,
        )
        store.record_runtime_observation(
            latent_mean=latest_mean,
            latent_scale=latest_scale,
            decoder_control=latest_decoder,
            switch_gate=store.switch_bias,
            sequence_length=len(trace.steps),
            active_label=latest_label,
            prior_mean=latest_prior_mean,
            prior_std=latest_prior_std,
            posterior_mean=latest_mean,
            posterior_std=latest_scale,
            z_tilde=latest_z_tilde,
            posterior_hidden_state=previous_hidden_state,
            posterior_drift=avg_drift,
        )
        store.record_ssl_metrics(total_loss=total_loss, kl_loss=avg_kl)
        slow_signal = tuple(
            v1 + v2
            for v1, v2 in zip(
                self._m3_encoder.slow_momentum_signal(),
                self._m3_decoder.slow_momentum_signal(),
            )
        )
        slow_signal = tuple(_clamp(v / 2.0) for v in slow_signal) if slow_signal else ()
        avg_kl_tightening = kl_tightening_total / trained_steps
        return SSLTrainingReport(
            trace_id=trace.trace_id,
            prediction_loss=avg_prediction,
            kl_loss=avg_kl,
            total_loss=total_loss,
            posterior_drift=avg_drift,
            trained_steps=trained_steps,
            m3_slow_momentum_signal=slow_signal,
            noncausal_kl_tightening=avg_kl_tightening,
            noncausal_information_content=noncausal_embedding.information_content,
            switch_gate_stats=gate_stats,
            description=(
                f"SSL trainer optimized trace={trace.trace_id} over {trained_steps} steps, "
                f"prediction_loss={avg_prediction:.3f}, kl_loss={avg_kl:.3f}, drift={avg_drift:.3f}, "
                f"kl_tightening={avg_kl_tightening:.3f}, "
                f"noncausal_info={noncausal_embedding.information_content:.3f}, "
                f"m3_slow_signal={tuple(round(v, 3) for v in slow_signal)}."
            ),
        )

    def _snapshot_from_prefix(
        self,
        *,
        trace: TrainingTrace,
        prefix: tuple[ResidualSequenceStep | object, ...],
    ) -> SubstrateSnapshot:
        typed_prefix = tuple(prefix)
        latest_step = typed_prefix[-1]
        return SubstrateSnapshot(
            model_id=f"ssl-trace:{trace.trace_id}",
            is_frozen=True,
            surface_kind=SurfaceKind.RESIDUAL_STREAM,
            token_logits=tuple(
                min(sum(feature.values) / max(len(feature.values), 1), 1.0)
                for feature in latest_step.feature_surface
            ),
            feature_surface=latest_step.feature_surface,
            residual_activations=latest_step.residual_activations,
            residual_sequence=tuple(
                ResidualSequenceStep(
                    step=step.step,
                    token=step.token,
                    feature_surface=step.feature_surface,
                    residual_activations=step.residual_activations,
                    description=f"SSL prefix token '{step.token}' at step {step.step}.",
                )
                for step in typed_prefix
            ),
            unavailable_fields=(),
            description=f"SSL prefix snapshot for {trace.trace_id} len={len(typed_prefix)}.",
        )

    def _apply_training_step(
        self,
        *,
        policy: FullLearnedTemporalPolicy,
        target_action: tuple[float, ...],
        encoded: EncodedSequence,
        decoder_control: DecoderControl,
        prediction_loss: float,
        kl_loss: float,
    ) -> None:
        store = policy.parameter_store
        learning_rate = store.learning_rate * 0.10
        encoder_gradients: list[tuple[float, ...]] = []
        decoder_gradients: list[tuple[float, ...]] = []
        for row_index, row in enumerate(store.encoder_weights):
            delta = target_action[row_index] - encoded.posterior.posterior_mean[row_index]
            encoder_gradients.append(
                tuple(
                    delta * (0.35 + encoded.posterior.posterior_std[row_index] * 0.15)
                    for _ in row
                )
            )
        for row_index, row in enumerate(store.decoder_matrix):
            delta = target_action[row_index] - decoder_control.decoder_output[row_index]
            decoder_gradients.append(
                tuple(delta * encoded.z_tilde[col_index] for col_index in range(len(row)))
            )
        store.encoder_weights = self._m3_encoder.update(
            gradients=tuple(encoder_gradients),
            learning_rate=learning_rate,
            parameters=store.encoder_weights,
        )
        store.decoder_matrix = self._m3_decoder.update(
            gradients=tuple(decoder_gradients),
            learning_rate=learning_rate,
            parameters=store.decoder_matrix,
        )
        recurrence_rows: list[tuple[float, ...]] = []
        for row_index, row in enumerate(store.encoder_recurrence):
            recurrence_rows.append(
                tuple(
                    _clamp(
                        weight
                        + (
                            encoded.posterior.posterior_mean[row_index]
                            - encoded.posterior.prior_mean[row_index]
                        )
                        * 0.05
                        * learning_rate
                    )
                    for weight in row
                )
            )
        store.encoder_recurrence = tuple(recurrence_rows)
        store.switch_weights = tuple(
            _clamp(weight + (0.5 - prediction_loss) * 0.05 * learning_rate + kl_loss * 0.01)
            for weight in store.switch_weights
        )
        store.switch_bias = _clamp(store.switch_bias + (0.35 - prediction_loss) * 0.05)

    def _action_prediction_loss(
        self,
        *,
        target_action: tuple[float, ...],
        decoder_control: DecoderControl,
    ) -> float:
        n = min(len(target_action), len(decoder_control.applied_control))
        return (
            sum(
                abs(target_action[i] - decoder_control.applied_control[i])
                for i in range(n)
            )
            / max(n, 1)
        )

    def _kl_to_prior(
        self,
        *,
        encoded: EncodedSequence,
    ) -> float:
        posterior_mean = encoded.posterior.posterior_mean
        posterior_std = tuple(max(value, 0.05) for value in encoded.posterior.posterior_std)
        prior_mean = encoded.posterior.prior_mean
        prior_std = tuple(max(value, 0.05) for value in encoded.posterior.prior_std)
        n = min(len(posterior_mean), len(prior_mean))
        terms = []
        for index in range(n):
            variance_ratio = (posterior_std[index] ** 2) / (prior_std[index] ** 2)
            mean_delta = ((posterior_mean[index] - prior_mean[index]) ** 2) / (prior_std[index] ** 2)
            log_term = 2.0 * math.log(prior_std[index] / posterior_std[index])
            terms.append(0.5 * (variance_ratio + mean_delta - 1.0 + log_term))
        return sum(max(term, 0.0) for term in terms) / max(n, 1)
