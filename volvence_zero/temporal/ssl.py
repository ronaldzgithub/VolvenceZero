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
    ResidualDecoder,
    SequenceEncoder,
    classify_latent_action,
    summarize_residual_activations,
)


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
    description: str = ""


class MetacontrollerSSLTrainer:
    """Small Eq.3-style training loop over residual traces."""

    def __init__(self) -> None:
        self._encoder = SequenceEncoder()
        self._decoder = ResidualDecoder()
        self._m3_encoder = M3Optimizer(num_groups=3, group_dim=3, slow_interval=3)
        self._m3_decoder = M3Optimizer(num_groups=3, group_dim=3, slow_interval=3)

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
        latest_mean = (0.0, 0.0, 0.0)
        latest_scale = (0.0, 0.0, 0.0)
        latest_prior_mean = (0.0, 0.0, 0.0)
        latest_prior_std = (1.0, 1.0, 1.0)
        latest_z_tilde = (0.0, 0.0, 0.0)
        latest_decoder = (0.0, 0.0, 0.0)
        latest_label = "stabilize_controller"
        posterior_drift_total = 0.0
        store = policy.parameter_store
        previous_hidden_state = store.latest_posterior_hidden_state

        for next_index in range(1, len(trace.steps)):
            prefix = trace.steps[:next_index]
            next_step = trace.steps[next_index]
            substrate_snapshot = self._snapshot_from_prefix(trace=trace, prefix=prefix)
            encoded = self._encoder.encode(
                substrate_snapshot=substrate_snapshot,
                encoder_weights=store.encoder_weights,
                recurrence_weights=store.encoder_recurrence,
                previous_hidden_state=previous_hidden_state,
            )
            decoder_control = self._decoder.decode(
                latent_code=encoded.z_tilde,
                decoder_matrix=store.decoder_matrix,
                hidden_matrix=store.decoder_hidden,
            )
            target_action = summarize_residual_activations(next_step.residual_activations, next_step.feature_surface)
            prediction_loss = self._action_prediction_loss(
                target_action=target_action,
                decoder_control=decoder_control,
            )
            kl_loss = self._kl_to_prior(encoded=encoded)
            total_loss = prediction_loss + kl_loss * 0.1
            prediction_total += prediction_loss
            kl_total += kl_loss
            posterior_drift_total += encoded.posterior.posterior_drift
            trained_steps += 1
            self._apply_training_step(
                policy=policy,
                target_action=target_action,
                encoded=encoded,
                decoder_control=decoder_control,
                prediction_loss=prediction_loss,
                kl_loss=kl_loss,
            )
            latest_mean = encoded.posterior.posterior_mean
            latest_scale = encoded.posterior.posterior_std
            latest_prior_mean = encoded.posterior.prior_mean
            latest_prior_std = encoded.posterior.prior_std
            latest_z_tilde = encoded.z_tilde
            latest_decoder = decoder_control.decoder_output
            latest_label, _ = classify_latent_action(
                latent_code=encoded.z_tilde,
                decoder_control=decoder_control.applied_control,
                prototypes=store.latent_prototypes,
            )
            previous_hidden_state = encoded.posterior.hidden_state

        avg_prediction = prediction_total / trained_steps
        avg_kl = kl_total / trained_steps
        avg_drift = posterior_drift_total / trained_steps
        total_loss = avg_prediction + avg_kl * 0.1
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
        return SSLTrainingReport(
            trace_id=trace.trace_id,
            prediction_loss=avg_prediction,
            kl_loss=avg_kl,
            total_loss=total_loss,
            posterior_drift=avg_drift,
            trained_steps=trained_steps,
            m3_slow_momentum_signal=slow_signal,
            description=(
                f"SSL trainer optimized trace={trace.trace_id} over {trained_steps} steps, "
                f"prediction_loss={avg_prediction:.3f}, kl_loss={avg_kl:.3f}, drift={avg_drift:.3f}, "
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
        return (
            sum(
                abs(target_value - applied_value)
                for target_value, applied_value in zip(target_action, decoder_control.applied_control, strict=True)
            )
            / 3.0
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
        terms = []
        for index in range(3):
            variance_ratio = (posterior_std[index] ** 2) / (prior_std[index] ** 2)
            mean_delta = ((posterior_mean[index] - prior_mean[index]) ** 2) / (prior_std[index] ** 2)
            log_term = 2.0 * math.log(prior_std[index] / posterior_std[index])
            terms.append(0.5 * (variance_ratio + mean_delta - 1.0 + log_term))
        return sum(max(term, 0.0) for term in terms) / 3.0
