from __future__ import annotations

import math
import os
from dataclasses import dataclass

from volvence_zero.runtime import WiringLevel
from volvence_zero.substrate import (
    ResidualSequenceStep,
    SubstrateSnapshot,
    SurfaceKind,
    TrainingTrace,
)
from volvence_zero.tensor_backend import is_torch_available
from volvence_zero.temporal.interface import (
    FullLearnedTemporalPolicy,
    MetacontrollerParameterStore,
)
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


from volvence_zero.temporal.m3_optimizer import M3Optimizer, M3OptimizerState


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _mean_abs(values: tuple[float, ...]) -> float:
    if not values:
        return 0.0
    return sum(abs(value) for value in values) / len(values)


@dataclass(frozen=True)
class SSLTrainingReport:
    trace_id: str
    prediction_loss: float
    kl_loss: float
    total_loss: float
    posterior_drift: float
    trained_steps: int
    encoder_optimizer_state: M3OptimizerState | None = None
    decoder_optimizer_state: M3OptimizerState | None = None
    m3_slow_momentum_signal: tuple[float, ...] = ()
    noncausal_kl_tightening: float = 0.0
    noncausal_information_content: float = 0.0
    switch_gate_stats: SwitchGateStats | None = None
    # Phase A autograd-owner-integration evidence (append-only, no schema break).
    # When the torch SSL backend is SHADOW/ACTIVE these carry the real-autograd
    # pass over the live store ndim params; DISABLED leaves them at defaults.
    torch_backend: str = "disabled"
    torch_prediction_loss: float = 0.0
    torch_kl_loss: float = 0.0
    torch_switch_sparsity: float = 0.0
    torch_parameters_changed: int = 0
    torch_grad_norm: float = 0.0
    torch_wrote_back: bool = False
    description: str = ""


@dataclass(frozen=True)
class SSLBatchTrainingReport:
    batch_id: str
    batch_count: int
    trajectory_count: int
    trained_step_count: int
    prediction_loss_mean: float
    kl_loss_mean: float
    total_loss_mean: float
    switch_sparsity_mean: float
    switch_entropy: float
    posterior_drift_mean: float
    noncausal_information_content: float
    cluster_stability: float
    family_birth_count: int
    family_merge_count: int
    family_prune_count: int
    scaffold_ablation_retention: float
    alpha_schedule: tuple[float, ...]
    trace_reports: tuple[SSLTrainingReport, ...]
    description: str = ""


class MetacontrollerSSLTrainer:
    """Small Eq.3-style training loop over residual traces.

    The ``ssl_backend`` WiringLevel controls the autograd-owner-integration path
    (Phase A): DISABLED keeps the pure heuristic update as the sole live writer;
    SHADOW additionally runs a real torch autograd pass over a copy of the store
    ndim params for evidence (no write-back); ACTIVE runs the torch autograd pass
    seeded from the store and writes the refined ndim weights back into the same
    store the runtime consumes. Family discovery / telemetry always run via the
    pure path, so structure ownership is unchanged.
    """

    def __init__(
        self,
        *,
        n_z: int = 3,
        alpha: float = 0.1,
        ssl_backend: WiringLevel = WiringLevel.DISABLED,
    ) -> None:
        self._n_z = n_z
        self._alpha = alpha
        self._ssl_backend = ssl_backend
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
        # Owner-local evidence retention (same pattern as CMS
        # ``latest_cms_backend_evidence``): the most recent SSLTrainingReport,
        # including the torch_* SHADOW/ACTIVE backend fields, stays readable
        # after optimize() so evidence exporters do not re-run training.
        self._latest_report: SSLTrainingReport | None = None

    @property
    def latest_report(self) -> SSLTrainingReport | None:
        """Return the most recent SSL training report (evidence readout only)."""

        return self._latest_report

    def optimize(
        self,
        *,
        policy: FullLearnedTemporalPolicy,
        trace: TrainingTrace,
    ) -> SSLTrainingReport:
        report = self._optimize_impl(policy=policy, trace=trace)
        self._latest_report = report
        return report

    def _optimize_impl(
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
        if os.name == "nt" and os.environ.get("VZ_SUBSTRATE_DEVICE", "").startswith("cuda"):
            policy.parameter_store.record_ssl_metrics(total_loss=0.0, kl_loss=0.0)
            return SSLTrainingReport(
                trace_id=trace.trace_id,
                prediction_loss=0.0,
                kl_loss=0.0,
                total_loss=0.0,
                posterior_drift=0.0,
                trained_steps=0,
                description=(
                    "SSL trainer skipped on Windows CUDA to avoid native "
                    "0xC0000005 failures; substrate inference remains on GPU."
                ),
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
        if store.learning_phase == "runtime" and store.structure_frozen:
            store.set_learning_phase("ssl", structure_frozen=False)
        store.require_ssl_discovery_phase(operation="MetacontrollerSSLTrainer.optimize")
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
                if store.ndim_encoder_parameters is None:
                    raise RuntimeError("n_z > 3 requires ndim encoder parameters in the metacontroller store.")
                encoded = self._ndim_encoder.encode(
                    substrate_snapshot=substrate_snapshot,
                    previous_hidden_state=previous_hidden_state,
                    params=store.ndim_encoder_parameters,
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
                if store.ndim_switch_parameters is None:
                    raise RuntimeError("n_z > 3 requires ndim switch parameters in the metacontroller store.")
                if store.ndim_decoder_parameters is None:
                    raise RuntimeError("n_z > 3 requires ndim decoder parameters in the metacontroller store.")
                beta_cont, _, scalar_beta = self._ndim_switch.compute(
                    z_tilde=encoded.z_tilde,
                    previous_code=previous_code,
                    memory_signal=0.0,
                    reflection_signal=0.0,
                    params=store.ndim_switch_parameters,
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
                decoder_control = self._ndim_decoder.decode(
                    latent_code=latent_code,
                    params=store.ndim_decoder_parameters,
                )
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
                noncausal_kl_tightening=enrichment.kl_tightening,
                noncausal_information_content=noncausal_embedding.information_content,
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
        encoder_optimizer_state = self._m3_encoder.export_state()
        decoder_optimizer_state = self._m3_decoder.export_state()
        store.record_optimizer_memory_states(
            encoder_state=encoder_optimizer_state,
            decoder_state=decoder_optimizer_state,
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
        torch_evidence = self._maybe_run_torch_backend(store=store, trace=trace)
        return SSLTrainingReport(
            trace_id=trace.trace_id,
            prediction_loss=avg_prediction,
            kl_loss=avg_kl,
            total_loss=total_loss,
            posterior_drift=avg_drift,
            trained_steps=trained_steps,
            encoder_optimizer_state=encoder_optimizer_state,
            decoder_optimizer_state=decoder_optimizer_state,
            m3_slow_momentum_signal=slow_signal,
            noncausal_kl_tightening=avg_kl_tightening,
            noncausal_information_content=noncausal_embedding.information_content,
            switch_gate_stats=gate_stats,
            torch_backend=torch_evidence["backend"],
            torch_prediction_loss=torch_evidence["prediction_loss"],
            torch_kl_loss=torch_evidence["kl_loss"],
            torch_switch_sparsity=torch_evidence["switch_sparsity"],
            torch_parameters_changed=torch_evidence["parameters_changed"],
            torch_grad_norm=torch_evidence["grad_norm"],
            torch_wrote_back=torch_evidence["wrote_back"],
            description=(
                f"SSL trainer optimized trace={trace.trace_id} over {trained_steps} steps, "
                f"prediction_loss={avg_prediction:.3f}, kl_loss={avg_kl:.3f}, drift={avg_drift:.3f}, "
                f"kl_tightening={avg_kl_tightening:.3f}, "
                f"noncausal_info={noncausal_embedding.information_content:.3f}, "
                f"m3_slow_signal={tuple(round(v, 3) for v in slow_signal)}."
            ),
        )

    def _maybe_run_torch_backend(
        self,
        *,
        store: MetacontrollerParameterStore,
        trace: TrainingTrace,
    ) -> dict:
        """Phase A: run the real-autograd SSL pass when the backend is enabled.

        DISABLED -> no-op. SHADOW -> torch trains a copy (no write-back).
        ACTIVE -> torch writes refined ndim weights back into the live store.
        Requires ndim params (n_z > 3) and torch; otherwise returns a no-op
        result with an explicit reason rather than silently degrading.
        """

        disabled = {
            "backend": "disabled", "prediction_loss": 0.0, "kl_loss": 0.0,
            "switch_sparsity": 0.0, "parameters_changed": 0, "grad_norm": 0.0,
            "wrote_back": False,
        }
        if self._ssl_backend is WiringLevel.DISABLED:
            return disabled
        if self._n_z <= 3 or store.ndim_encoder_parameters is None:
            return {**disabled, "backend": "skipped-no-ndim"}
        if not is_torch_available():
            return {**disabled, "backend": "skipped-no-torch"}

        from volvence_zero.temporal.torch_store_ssl import train_store_ssl

        write_back = self._ssl_backend is WiringLevel.ACTIVE
        report = train_store_ssl(
            store=store,
            trace=trace,
            n_z=self._n_z,
            alpha=self._alpha,
            switch_threshold=store.beta_threshold,
            write_back=write_back,
        )
        return {
            "backend": self._ssl_backend.value,
            "prediction_loss": report.prediction_loss,
            "kl_loss": report.kl_loss,
            "switch_sparsity": report.switch_sparsity,
            "parameters_changed": report.parameters_changed,
            "grad_norm": report.grad_norm,
            "wrote_back": report.wrote_back,
        }

    def optimize_batch(
        self,
        *,
        policy: FullLearnedTemporalPolicy,
        traces: tuple[TrainingTrace, ...],
        batch_id: str = "ssl-batch",
        semantic_labels_enabled: bool = False,
    ) -> SSLBatchTrainingReport:
        if not traces:
            raise ValueError("MetacontrollerSSLTrainer.optimize_batch requires at least one trace.")
        family_count_before = len(policy.parameter_store.action_families)
        trace_reports: list[SSLTrainingReport] = []
        for trace in traces:
            trace_reports.append(self.optimize(policy=policy, trace=trace))
        reports = tuple(trace_reports)
        family_count_after = len(policy.parameter_store.action_families)
        trained_step_count = sum(report.trained_steps for report in reports)
        switch_sparsity_values = tuple(
            1.0 - report.switch_gate_stats.switch_frequency
            for report in reports
            if report.switch_gate_stats is not None
        )
        switch_entropy = self._switch_entropy(reports)
        cluster_stability = self._cluster_stability(policy=policy)
        scaffold_ablation_retention = cluster_stability if not semantic_labels_enabled else 1.0
        return SSLBatchTrainingReport(
            batch_id=batch_id,
            batch_count=1,
            trajectory_count=len(traces),
            trained_step_count=trained_step_count,
            prediction_loss_mean=self._mean(tuple(report.prediction_loss for report in reports)),
            kl_loss_mean=self._mean(tuple(report.kl_loss for report in reports)),
            total_loss_mean=self._mean(tuple(report.total_loss for report in reports)),
            switch_sparsity_mean=self._mean(switch_sparsity_values),
            switch_entropy=switch_entropy,
            posterior_drift_mean=self._mean(tuple(report.posterior_drift for report in reports)),
            noncausal_information_content=self._mean(
                tuple(report.noncausal_information_content for report in reports)
            ),
            cluster_stability=cluster_stability,
            family_birth_count=max(family_count_after - family_count_before, 0),
            family_merge_count=0,
            family_prune_count=0,
            scaffold_ablation_retention=round(scaffold_ablation_retention, 4),
            alpha_schedule=(self._alpha,),
            trace_reports=reports,
            description=(
                f"SSL batch {batch_id} optimized {len(traces)} residual trajectories "
                f"over {trained_step_count} trained steps; switch_entropy={switch_entropy:.3f}, "
                f"cluster_stability={cluster_stability:.3f}."
            ),
        )

    def _mean(self, values: tuple[float, ...]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _switch_entropy(self, reports: tuple[SSLTrainingReport, ...]) -> float:
        histogram = [0] * 10
        total = 0
        for report in reports:
            if report.switch_gate_stats is None:
                continue
            for index, count in enumerate(report.switch_gate_stats.beta_histogram):
                histogram[index] += count
                total += count
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in histogram:
            if count == 0:
                continue
            probability = count / total
            entropy -= probability * math.log(probability, 2)
        max_entropy = math.log(len(histogram), 2)
        return round(entropy / max_entropy, 4) if max_entropy > 0.0 else 0.0

    def _cluster_stability(self, *, policy: FullLearnedTemporalPolicy) -> float:
        families = policy.parameter_store.action_families
        if not families:
            return 0.0
        support_total = sum(max(family.support, 0) for family in families)
        if support_total <= 0:
            return 0.0
        weighted = sum(
            max(family.support, 0)
            * _clamp(
                family.stability * 0.45
                + family.competition_score * 0.25
                + family.long_term_payoff * 0.20
                + min(family.mean_persistence_window / 3.0, 1.0) * 0.10
            )
            for family in families
        )
        return round(weighted / support_total, 4)

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
        noncausal_kl_tightening: float,
        noncausal_information_content: float,
    ) -> None:
        store = policy.parameter_store
        learning_rate = store.learning_rate * 0.10
        target_delta = tuple(
            target_action[index] - decoder_control.decoder_output[index]
            for index in range(len(target_action))
        )
        posterior_delta = tuple(
            target_action[index] - encoded.posterior.posterior_mean[index]
            for index in range(len(target_action))
        )
        update_features = (
            _clamp(prediction_loss),
            _clamp(kl_loss),
            _clamp(noncausal_kl_tightening),
            _clamp(noncausal_information_content),
            _clamp(encoded.posterior.posterior_drift),
            _clamp(_mean_abs(posterior_delta)),
            _clamp(_mean_abs(target_delta)),
            _clamp(_mean_abs(encoded.posterior.hidden_state)),
            _clamp(_mean_abs(encoded.posterior.posterior_std)),
            _clamp(_mean_abs(encoded.posterior.prior_std)),
            _clamp(_mean_abs(encoded.z_tilde)),
            _clamp(1.0 - abs(prediction_loss - 0.35)),
        )
        encoder_decision = store.learned_update_rule.decide(
            target_id="temporal-encoder",
            features=update_features + (1.0, 0.0, 0.0),
        )
        decoder_decision = store.learned_update_rule.decide(
            target_id="temporal-decoder",
            features=update_features + (0.0, 1.0, 0.0),
        )
        switch_decision = store.learned_update_rule.decide(
            target_id="temporal-switch",
            features=update_features + (0.0, 0.0, 1.0),
        )
        encoder_gain = 0.10 + encoder_decision.step_scale * 0.18
        decoder_gain = 0.10 + decoder_decision.step_scale * 0.18
        switch_gain = 0.08 + switch_decision.step_scale * 0.14
        encoder_write = 0.12 + encoder_decision.write_gate * 0.20
        decoder_write = 0.12 + decoder_decision.write_gate * 0.20
        switch_write = 0.10 + switch_decision.write_gate * 0.16
        improvement = max(-1.0, min(1.0, (0.35 - prediction_loss) * 1.5 + noncausal_kl_tightening * 0.2 - kl_loss * 0.1))
        stability = _clamp(1.0 - encoded.posterior.posterior_drift)
        if (
            store.n_z > 3
            and store.ndim_encoder_parameters is not None
            and store.ndim_switch_parameters is not None
            and store.ndim_decoder_parameters is not None
        ):
            encoder_lr = learning_rate * encoder_gain * encoder_write
            decoder_lr = learning_rate * decoder_gain * decoder_write
            switch_delta = (
                switch_decision.bias_delta * 0.02
                + (0.35 - prediction_loss) * switch_gain * switch_write * 0.04
                + noncausal_kl_tightening * 0.01
                + noncausal_information_content * 0.005
            )
            store.ndim_encoder_parameters = type(store.ndim_encoder_parameters)(
                n_input=store.ndim_encoder_parameters.n_input,
                gru=type(store.ndim_encoder_parameters.gru)(
                    W_z=store.ndim_encoder_parameters.gru.W_z,
                    U_z=store.ndim_encoder_parameters.gru.U_z,
                    b_z=tuple(
                        _clamp(
                            bias
                            + posterior_delta[index] * encoder_lr * 0.10
                            + encoder_decision.bias_delta * 0.01
                        )
                        for index, bias in enumerate(store.ndim_encoder_parameters.gru.b_z)
                    ),
                    W_r=store.ndim_encoder_parameters.gru.W_r,
                    U_r=store.ndim_encoder_parameters.gru.U_r,
                    b_r=tuple(
                        _clamp(
                            bias
                            + posterior_delta[index] * encoder_lr * 0.06
                            + encoder_decision.bias_delta * 0.008
                        )
                        for index, bias in enumerate(store.ndim_encoder_parameters.gru.b_r)
                    ),
                    W_h=store.ndim_encoder_parameters.gru.W_h,
                    U_h=store.ndim_encoder_parameters.gru.U_h,
                    b_h=tuple(
                        _clamp(
                            bias
                            + posterior_delta[index] * encoder_lr * 0.08
                            + encoder_decision.bias_delta * 0.01
                        )
                        for index, bias in enumerate(store.ndim_encoder_parameters.gru.b_h)
                    ),
                ),
                posterior_proj=self._m3_encoder.update(
                    gradients=tuple(
                        tuple(
                            posterior_delta[row_index] * encoded.posterior.hidden_state[col_index]
                            for col_index in range(len(row))
                        )
                        for row_index, row in enumerate(store.ndim_encoder_parameters.posterior_proj)
                    ),
                    learning_rate=encoder_lr,
                    parameters=store.ndim_encoder_parameters.posterior_proj,
                ),
                posterior_std_proj=self._m3_encoder.update(
                    gradients=tuple(
                        tuple(
                            (0.25 - encoded.posterior.posterior_std[row_index]) * encoded.posterior.hidden_state[col_index]
                            for col_index in range(len(row))
                        )
                        for row_index, row in enumerate(store.ndim_encoder_parameters.posterior_std_proj)
                    ),
                    learning_rate=encoder_lr * (0.15 + encoder_decision.momentum_gate * 0.20),
                    parameters=store.ndim_encoder_parameters.posterior_std_proj,
                ),
            )
            store.ndim_decoder_parameters = type(store.ndim_decoder_parameters)(
                decoder_ffn=type(store.ndim_decoder_parameters.decoder_ffn)(
                    W1=self._m3_decoder.update(
                        gradients=tuple(
                            tuple(
                                target_delta[row_index] * encoded.z_tilde[col_index]
                                for col_index in range(len(row))
                            )
                            for row_index, row in enumerate(store.ndim_decoder_parameters.decoder_ffn.W1)
                        ),
                        learning_rate=decoder_lr,
                        parameters=store.ndim_decoder_parameters.decoder_ffn.W1,
                    ),
                    b1=tuple(
                        _clamp(
                            bias + target_delta[index] * decoder_lr * 0.10 + decoder_decision.bias_delta * 0.01
                        )
                        for index, bias in enumerate(store.ndim_decoder_parameters.decoder_ffn.b1)
                    ),
                    W2=self._m3_decoder.update(
                        gradients=tuple(
                            tuple(
                                target_delta[row_index] * decoder_control.decoder_output[col_index]
                                for col_index in range(len(row))
                            )
                            for row_index, row in enumerate(store.ndim_decoder_parameters.decoder_ffn.W2)
                        ),
                        learning_rate=decoder_lr,
                        parameters=store.ndim_decoder_parameters.decoder_ffn.W2,
                    ),
                    b2=tuple(
                        _clamp(
                            bias + target_delta[index] * decoder_lr * 0.10 + decoder_decision.bias_delta * 0.01
                        )
                        for index, bias in enumerate(store.ndim_decoder_parameters.decoder_ffn.b2)
                    ),
                )
            )
            store.ndim_switch_parameters = type(store.ndim_switch_parameters)(
                gate_ffn=type(store.ndim_switch_parameters.gate_ffn)(
                    W1=store.ndim_switch_parameters.gate_ffn.W1,
                    b1=tuple(_clamp(bias + switch_delta) for bias in store.ndim_switch_parameters.gate_ffn.b1),
                    W2=store.ndim_switch_parameters.gate_ffn.W2,
                    b2=tuple(_clamp(bias + switch_delta) for bias in store.ndim_switch_parameters.gate_ffn.b2),
                )
            )
            store.switch_bias = _clamp(store.switch_bias + switch_delta)
            store.learned_update_rule.learn(
                features=update_features + (1.0, 0.0, 0.0),
                decision=encoder_decision,
                improvement=improvement,
                stability=stability,
            )
            store.learned_update_rule.learn(
                features=update_features + (0.0, 1.0, 0.0),
                decision=decoder_decision,
                improvement=improvement,
                stability=stability,
            )
            store.learned_update_rule.learn(
                features=update_features + (0.0, 0.0, 1.0),
                decision=switch_decision,
                improvement=improvement,
                stability=stability,
            )
            store.record_learned_update_rule_state(state=store.learned_update_rule.export_state())
            return
        encoder_gradients: list[tuple[float, ...]] = []
        decoder_gradients: list[tuple[float, ...]] = []
        for row_index, row in enumerate(store.encoder_weights):
            delta = target_action[row_index] - encoded.posterior.posterior_mean[row_index]
            encoder_gradients.append(
                tuple(
                    delta
                    * (
                        0.08
                        + encoder_gain * 0.22
                        + encoded.posterior.posterior_std[row_index] * (0.05 + encoder_decision.confidence * 0.10)
                    )
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
            learning_rate=learning_rate * encoder_gain * encoder_write,
            parameters=store.encoder_weights,
        )
        store.decoder_matrix = self._m3_decoder.update(
            gradients=tuple(decoder_gradients),
            learning_rate=learning_rate * decoder_gain * decoder_write,
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
                        * (0.01 + (0.10 + encoder_decision.momentum_gate * 0.08))
                        * learning_rate
                        + encoder_decision.bias_delta * 0.008
                    )
                    for weight in row
                )
            )
        store.encoder_recurrence = tuple(recurrence_rows)
        store.switch_weights = tuple(
            _clamp(
                weight
                + (0.5 - prediction_loss) * switch_gain * switch_write * 0.05 * learning_rate
                + noncausal_kl_tightening * 0.01
                + noncausal_information_content * 0.004
                + switch_decision.bias_delta * 0.012
            )
            for weight in store.switch_weights
        )
        store.switch_bias = _clamp(
            store.switch_bias
            + (0.35 - prediction_loss) * switch_write * 0.05
            + switch_decision.bias_delta * 0.015
        )
        store.learned_update_rule.learn(
            features=update_features + (1.0, 0.0, 0.0),
            decision=encoder_decision,
            improvement=improvement,
            stability=stability,
        )
        store.learned_update_rule.learn(
            features=update_features + (0.0, 1.0, 0.0),
            decision=decoder_decision,
            improvement=improvement,
            stability=stability,
        )
        store.learned_update_rule.learn(
            features=update_features + (0.0, 0.0, 1.0),
            decision=switch_decision,
            improvement=improvement,
            stability=stability,
        )
        store.record_learned_update_rule_state(state=store.learned_update_rule.export_state())

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
