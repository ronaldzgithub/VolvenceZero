from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Any, Mapping

from volvence_zero.memory import MemorySnapshot, Track
from volvence_zero.reflection import ReflectionSnapshot, TemporalPriorUpdate, TemporalStructureProposal
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel
from volvence_zero.substrate import FeatureSignal, SubstrateSnapshot, SurfaceKind
from volvence_zero.temporal.metacontroller_components import (
    ActionFamilyObservation,
    DecoderControl,
    DiscoveredActionFamily,
    EncodedSequence,
    FamilyCompetitionState,
    NdimResidualDecoder,
    NdimSequenceEncoder,
    NdimSwitchUnit,
    PosteriorState,
    ResidualDecoder,
    SequenceEncoder,
    SwitchGateDecision,
    SwitchGateStats,
    SwitchUnit,
    build_family_competition_state,
    discover_latent_action_family,
    residual_sequence_from_snapshot,
    summarize_feature_surface,
    summarize_residual_activations,
)


class TemporalImplementationMode(str, Enum):
    PLACEHOLDER = "placeholder"
    HEURISTIC = "heuristic"
    LEARNED_LITE = "learned-lite"
    FULL_LEARNED = "full-learned"


@dataclass(frozen=True)
class ControllerState:
    code: tuple[float, ...]
    code_dim: int
    switch_gate: float
    is_switching: bool
    steps_since_switch: int
    track_codes: tuple[tuple[str, tuple[float, ...]], ...] = ()


@dataclass(frozen=True)
class TemporalAbstractionSnapshot:
    controller_state: ControllerState
    active_abstract_action: str
    controller_params_hash: str
    description: str
    action_family_version: int = 0
    switch_gate_stats: SwitchGateStats | None = None


@dataclass(frozen=True)
class TemporalStep:
    controller_state: ControllerState
    active_abstract_action: str
    controller_params_hash: str
    description: str
    action_family_version: int = 0


@dataclass(frozen=True)
class TemporalControllerParameters:
    residual_weight: float
    memory_weight: float
    reflection_weight: float
    switch_bias: float


@dataclass(frozen=True)
class MetacontrollerRuntimeState:
    mode: str
    temporal_parameters: TemporalControllerParameters
    track_parameters: tuple[tuple[str, tuple[float, ...]], ...]
    encoder_weights: tuple[tuple[float, ...], ...]
    switch_weights: tuple[float, ...]
    decoder_matrix: tuple[tuple[float, ...], ...]
    persistence: float
    learning_rate: float
    clip_epsilon: float
    update_steps: tuple[tuple[str, int], ...]
    latent_mean: tuple[float, ...]
    latent_scale: tuple[float, ...]
    decoder_control: tuple[float, ...]
    latest_switch_gate: float
    sequence_length: int
    latest_ssl_loss: float
    latest_ssl_kl_loss: float
    active_label: str
    encoder_recurrence: tuple[tuple[float, ...], ...] = ()
    beta_threshold: float = 0.55
    decoder_hidden: tuple[tuple[float, ...], ...] = ()
    prior_mean: tuple[float, ...] = ()
    prior_std: tuple[float, ...] = ()
    posterior_mean: tuple[float, ...] = ()
    posterior_std: tuple[float, ...] = ()
    posterior_sample_noise: tuple[float, ...] = ()
    z_tilde: tuple[float, ...] = ()
    posterior_hidden_state: tuple[float, ...] = ()
    posterior_drift: float = 0.0
    beta_binary: int = 0
    switch_sparsity: float = 0.0
    binary_switch_rate: float = 0.0
    mean_persistence_window: float = 0.0
    decoder_applied_control: tuple[float, ...] = ()
    policy_replacement_score: float = 0.0
    structure_frozen: bool = False
    learning_phase: str = "runtime"
    action_family_version: int = 0
    action_family_summaries: tuple["ActionFamilyPublicSummary", ...] = ()
    active_family_summary: "ActionFamilyPublicSummary | None" = None
    active_family_competition_score: float = 0.0
    action_family_monopoly_pressure: float = 0.0
    action_family_turnover_health: float = 0.0
    description: str = ""


@dataclass(frozen=True)
class ActionFamilyPublicSummary:
    family_id: str
    dominant_axis: str
    support: int
    stability: float
    switch_bias: float
    mean_posterior_drift: float
    mean_persistence_window: float
    reuse_streak: int
    stagnation_pressure: float
    monopoly_pressure: float
    competition_score: float
    outcome_history: tuple[float, ...] = ()
    outcome_driven_score: float = 0.0
    summary: str = ""


@dataclass(frozen=True)
class MetacontrollerParameterSnapshot:
    temporal_parameters: TemporalControllerParameters
    track_parameters: tuple[tuple[str, tuple[float, ...]], ...]
    encoder_weights: tuple[tuple[float, ...], ...]
    encoder_recurrence: tuple[tuple[float, ...], ...]
    switch_weights: tuple[float, ...]
    beta_threshold: float
    decoder_matrix: tuple[tuple[float, ...], ...]
    decoder_hidden: tuple[tuple[float, ...], ...]
    persistence: float
    learning_rate: float
    clip_epsilon: float
    update_steps: tuple[tuple[str, int], ...]
    latent_mean: tuple[float, ...]
    latent_scale: tuple[float, ...]
    decoder_control: tuple[float, ...]
    latest_switch_gate: float
    sequence_length: int
    latest_ssl_loss: float
    latest_ssl_kl_loss: float
    active_label: str
    prior_mean: tuple[float, ...]
    prior_std: tuple[float, ...]
    posterior_mean: tuple[float, ...]
    posterior_std: tuple[float, ...]
    posterior_sample_noise: tuple[float, ...]
    z_tilde: tuple[float, ...]
    posterior_hidden_state: tuple[float, ...]
    posterior_drift: float
    beta_binary: int
    switch_sparsity: float
    binary_switch_rate: float
    mean_persistence_window: float
    decoder_applied_control: tuple[float, ...]
    policy_replacement_score: float
    action_families: tuple[DiscoveredActionFamily, ...] = ()
    structure_frozen: bool = False
    learning_phase: str = "runtime"
    action_family_version: int = 0


class MetacontrollerParameterStore:
    """Shared parameter store for runtime temporal control and internal RL."""

    def __init__(self, *, n_z: int = 3) -> None:
        self._n_z = n_z
        self.temporal_weights: dict[str, float] = {
            "residual": 0.65,
            "memory": 0.2,
            "reflection": 0.15,
        }
        self.switch_bias = 0.1
        if n_z == 3:
            self.encoder_weights: tuple[tuple[float, ...], ...] = (
                (0.70, 0.20, 0.10),
                (0.25, 0.55, 0.20),
                (0.15, 0.25, 0.60),
            )
            self.encoder_recurrence: tuple[tuple[float, ...], ...] = (
                (0.60, 0.20, 0.20),
                (0.20, 0.60, 0.20),
                (0.20, 0.20, 0.60),
            )
            self.switch_weights: tuple[float, ...] = (0.45, 0.35, 0.20)
            self.decoder_matrix: tuple[tuple[float, ...], ...] = (
                (0.80, 0.15, 0.05),
                (0.20, 0.65, 0.15),
                (0.25, 0.25, 0.50),
            )
            self.decoder_hidden: tuple[tuple[float, ...], ...] = (
                (0.60, 0.25, 0.15),
                (0.20, 0.60, 0.20),
                (0.15, 0.25, 0.60),
            )
            self.action_families: tuple[DiscoveredActionFamily, ...] = _init_action_families(n_z, seed=105)
            self.track_weights: dict[Track, tuple[float, float, float]] = {
                Track.WORLD: (0.70, 0.20, 0.10),
                Track.SELF: (0.20, 0.70, 0.10),
                Track.SHARED: (0.40, 0.40, 0.20),
            }
        else:
            self.encoder_weights = _random_mat(n_z, n_z, seed=100)
            self.encoder_recurrence = _random_mat(n_z, n_z, seed=101)
            self.switch_weights = _random_vec(n_z, seed=102)
            self.decoder_matrix = _random_mat(n_z, n_z, seed=103)
            self.decoder_hidden = _random_mat(n_z, n_z, seed=104)
            self.action_families = _init_action_families(n_z, seed=105)
            self.track_weights = _init_track_weights(n_z, seed=106)
        self.beta_threshold = 0.55
        self.persistence = 0.65
        self.learning_rate = 0.08
        self.clip_epsilon = 0.2
        self.update_steps: dict[Track, int] = {
            Track.WORLD: 0,
            Track.SELF: 0,
            Track.SHARED: 0,
        }
        self.latest_latent_mean: tuple[float, ...] = _nz_zeros(n_z)
        self.latest_latent_scale: tuple[float, ...] = _nz_zeros(n_z)
        self.latest_decoder_control: tuple[float, ...] = _nz_zeros(n_z)
        self.latest_switch_gate = 0.0
        self.latest_sequence_length = 0
        self.latest_ssl_loss = 0.0
        self.latest_ssl_kl_loss = 0.0
        self.latest_active_label = "unassigned_action"
        self.latest_prior_mean: tuple[float, ...] = _nz_zeros(n_z)
        self.latest_prior_std: tuple[float, ...] = _nz_ones(n_z)
        self.latest_posterior_mean: tuple[float, ...] = _nz_zeros(n_z)
        self.latest_posterior_std: tuple[float, ...] = _nz_zeros(n_z)
        self.latest_posterior_sample_noise: tuple[float, ...] = _nz_zeros(n_z)
        self.latest_z_tilde: tuple[float, ...] = _nz_zeros(n_z)
        self.latest_posterior_hidden_state: tuple[float, ...] = _nz_zeros(n_z)
        self.latest_posterior_drift = 0.0
        self.latest_beta_binary = 0
        self.latest_switch_sparsity = 0.0
        self.latest_binary_switch_rate = 0.0
        self.latest_mean_persistence_window = 0.0
        self.latest_decoder_applied_control: tuple[float, ...] = _nz_zeros(n_z)
        self.latest_policy_replacement_score = 0.0
        self.structure_frozen = False
        self.learning_phase = "runtime"
        self._action_family_version = 0

    @property
    def n_z(self) -> int:
        return self._n_z

    @property
    def action_family_version(self) -> int:
        return self._action_family_version

    def export_temporal_parameters(self) -> TemporalControllerParameters:
        return TemporalControllerParameters(
            residual_weight=self.temporal_weights["residual"],
            memory_weight=self.temporal_weights["memory"],
            reflection_weight=self.temporal_weights["reflection"],
            switch_bias=self.switch_bias,
        )

    def export_runtime_state(self, *, mode: str) -> MetacontrollerRuntimeState:
        action_family_summaries = self._public_action_family_summaries()
        active_family_summary = next(
            (summary for summary in action_family_summaries if summary.family_id == self.latest_active_label),
            None,
        )
        active_family_competition_score = (
            active_family_summary.competition_score
            if active_family_summary is not None
            else 0.0
        )
        action_family_monopoly_pressure = (
            active_family_summary.monopoly_pressure
            if active_family_summary is not None
            else 0.0
        )
        action_family_turnover_health = self._action_family_turnover_health(action_family_summaries)
        return MetacontrollerRuntimeState(
            mode=mode,
            temporal_parameters=self.export_temporal_parameters(),
            track_parameters=tuple(
                (track.value, self.track_weights[track])
                for track in (Track.WORLD, Track.SELF, Track.SHARED)
            ),
            encoder_weights=self.encoder_weights,
            encoder_recurrence=self.encoder_recurrence,
            switch_weights=self.switch_weights,
            beta_threshold=self.beta_threshold,
            decoder_matrix=self.decoder_matrix,
            decoder_hidden=self.decoder_hidden,
            persistence=self.persistence,
            learning_rate=self.learning_rate,
            clip_epsilon=self.clip_epsilon,
            update_steps=tuple(
                (track.value, self.update_steps[track])
                for track in (Track.WORLD, Track.SELF, Track.SHARED)
            ),
            latent_mean=self.latest_latent_mean,
            latent_scale=self.latest_latent_scale,
            decoder_control=self.latest_decoder_control,
            latest_switch_gate=self.latest_switch_gate,
            sequence_length=self.latest_sequence_length,
            latest_ssl_loss=self.latest_ssl_loss,
            latest_ssl_kl_loss=self.latest_ssl_kl_loss,
            active_label=self.latest_active_label,
            prior_mean=self.latest_prior_mean,
            prior_std=self.latest_prior_std,
            posterior_mean=self.latest_posterior_mean,
            posterior_std=self.latest_posterior_std,
            posterior_sample_noise=self.latest_posterior_sample_noise,
            z_tilde=self.latest_z_tilde,
            posterior_hidden_state=self.latest_posterior_hidden_state,
            posterior_drift=self.latest_posterior_drift,
            beta_binary=self.latest_beta_binary,
            switch_sparsity=self.latest_switch_sparsity,
            binary_switch_rate=self.latest_binary_switch_rate,
            mean_persistence_window=self.latest_mean_persistence_window,
            decoder_applied_control=self.latest_decoder_applied_control,
            policy_replacement_score=self.latest_policy_replacement_score,
            structure_frozen=self.structure_frozen,
            learning_phase=self.learning_phase,
            action_family_version=self._action_family_version,
            action_family_summaries=action_family_summaries,
            active_family_summary=active_family_summary,
            active_family_competition_score=active_family_competition_score,
            action_family_monopoly_pressure=action_family_monopoly_pressure,
            action_family_turnover_health=action_family_turnover_health,
            description=(
                f"Metacontroller runtime state mode={mode}, active_label={self.latest_active_label}, "
                f"switch_bias={self.switch_bias:.2f}, persistence={self.persistence:.2f}, "
                f"beta_binary={self.latest_beta_binary}, seq_len={self.latest_sequence_length}, "
                f"family_version={self._action_family_version}, family_count={len(action_family_summaries)}, "
                f"competition={active_family_competition_score:.2f}, monopoly={action_family_monopoly_pressure:.2f}, "
                f"phase={self.learning_phase}, structure_frozen={self.structure_frozen}, "
                f"ssl_loss={self.latest_ssl_loss:.3f}."
            ),
        )

    def export_parameter_snapshot(self) -> MetacontrollerParameterSnapshot:
        return MetacontrollerParameterSnapshot(
            temporal_parameters=self.export_temporal_parameters(),
            track_parameters=tuple(
                (track.value, self.track_weights[track])
                for track in (Track.WORLD, Track.SELF, Track.SHARED)
            ),
            encoder_weights=self.encoder_weights,
            encoder_recurrence=self.encoder_recurrence,
            switch_weights=self.switch_weights,
            beta_threshold=self.beta_threshold,
            decoder_matrix=self.decoder_matrix,
            decoder_hidden=self.decoder_hidden,
            persistence=self.persistence,
            learning_rate=self.learning_rate,
            clip_epsilon=self.clip_epsilon,
            update_steps=tuple(
                (track.value, self.update_steps[track])
                for track in (Track.WORLD, Track.SELF, Track.SHARED)
            ),
            latent_mean=self.latest_latent_mean,
            latent_scale=self.latest_latent_scale,
            decoder_control=self.latest_decoder_control,
            latest_switch_gate=self.latest_switch_gate,
            sequence_length=self.latest_sequence_length,
            latest_ssl_loss=self.latest_ssl_loss,
            latest_ssl_kl_loss=self.latest_ssl_kl_loss,
            active_label=self.latest_active_label,
            prior_mean=self.latest_prior_mean,
            prior_std=self.latest_prior_std,
            posterior_mean=self.latest_posterior_mean,
            posterior_std=self.latest_posterior_std,
            posterior_sample_noise=self.latest_posterior_sample_noise,
            z_tilde=self.latest_z_tilde,
            posterior_hidden_state=self.latest_posterior_hidden_state,
            posterior_drift=self.latest_posterior_drift,
            beta_binary=self.latest_beta_binary,
            switch_sparsity=self.latest_switch_sparsity,
            binary_switch_rate=self.latest_binary_switch_rate,
            mean_persistence_window=self.latest_mean_persistence_window,
            decoder_applied_control=self.latest_decoder_applied_control,
            policy_replacement_score=self.latest_policy_replacement_score,
            action_families=self.action_families,
            structure_frozen=self.structure_frozen,
            learning_phase=self.learning_phase,
            action_family_version=self._action_family_version,
        )

    def restore_parameter_snapshot(self, snapshot: MetacontrollerParameterSnapshot) -> None:
        self.temporal_weights = {
            "residual": snapshot.temporal_parameters.residual_weight,
            "memory": snapshot.temporal_parameters.memory_weight,
            "reflection": snapshot.temporal_parameters.reflection_weight,
        }
        self.switch_bias = snapshot.temporal_parameters.switch_bias
        self.track_weights = {
            Track(track_name): weights for track_name, weights in snapshot.track_parameters
        }
        self.encoder_weights = snapshot.encoder_weights
        self.encoder_recurrence = snapshot.encoder_recurrence
        self.switch_weights = snapshot.switch_weights
        self.beta_threshold = snapshot.beta_threshold
        self.decoder_matrix = snapshot.decoder_matrix
        self.decoder_hidden = snapshot.decoder_hidden
        self.persistence = snapshot.persistence
        self.learning_rate = snapshot.learning_rate
        self.clip_epsilon = snapshot.clip_epsilon
        self.update_steps = {
            Track(track_name): step_count for track_name, step_count in snapshot.update_steps
        }
        self.latest_latent_mean = snapshot.latent_mean
        self.latest_latent_scale = snapshot.latent_scale
        self.latest_decoder_control = snapshot.decoder_control
        self.latest_switch_gate = snapshot.latest_switch_gate
        self.latest_sequence_length = snapshot.sequence_length
        self.latest_ssl_loss = snapshot.latest_ssl_loss
        self.latest_ssl_kl_loss = snapshot.latest_ssl_kl_loss
        self.latest_active_label = snapshot.active_label
        self.latest_prior_mean = snapshot.prior_mean
        self.latest_prior_std = snapshot.prior_std
        self.latest_posterior_mean = snapshot.posterior_mean
        self.latest_posterior_std = snapshot.posterior_std
        self.latest_posterior_sample_noise = snapshot.posterior_sample_noise
        self.latest_z_tilde = snapshot.z_tilde
        self.latest_posterior_hidden_state = snapshot.posterior_hidden_state
        self.latest_posterior_drift = snapshot.posterior_drift
        self.latest_beta_binary = snapshot.beta_binary
        self.latest_switch_sparsity = snapshot.switch_sparsity
        self.latest_binary_switch_rate = snapshot.binary_switch_rate
        self.latest_mean_persistence_window = snapshot.mean_persistence_window
        self.latest_decoder_applied_control = snapshot.decoder_applied_control
        self.latest_policy_replacement_score = snapshot.policy_replacement_score
        self.action_families = snapshot.action_families
        self.structure_frozen = snapshot.structure_frozen
        self.learning_phase = snapshot.learning_phase
        self._action_family_version = snapshot.action_family_version

    def record_runtime_observation(
        self,
        *,
        latent_mean: tuple[float, ...],
        latent_scale: tuple[float, ...],
        decoder_control: tuple[float, ...],
        switch_gate: float,
        sequence_length: int,
        active_label: str,
        prior_mean: tuple[float, ...] | None = None,
        prior_std: tuple[float, ...] | None = None,
        posterior_mean: tuple[float, ...] | None = None,
        posterior_std: tuple[float, ...] | None = None,
        posterior_sample_noise: tuple[float, ...] | None = None,
        z_tilde: tuple[float, ...] | None = None,
        posterior_hidden_state: tuple[float, ...] | None = None,
        posterior_drift: float | None = None,
        beta_binary: int | None = None,
        switch_sparsity: float | None = None,
        binary_switch_rate: float | None = None,
        mean_persistence_window: float | None = None,
        decoder_applied_control: tuple[float, ...] | None = None,
        policy_replacement_score: float | None = None,
    ) -> None:
        self.latest_latent_mean = latent_mean
        self.latest_latent_scale = latent_scale
        self.latest_decoder_control = decoder_control
        self.latest_switch_gate = switch_gate
        self.latest_sequence_length = sequence_length
        self.latest_active_label = active_label
        self.latest_prior_mean = prior_mean or self.latest_prior_mean
        self.latest_prior_std = prior_std or self.latest_prior_std
        self.latest_posterior_mean = posterior_mean or latent_mean
        self.latest_posterior_std = posterior_std or latent_scale
        self.latest_posterior_sample_noise = posterior_sample_noise or self.latest_posterior_sample_noise
        self.latest_z_tilde = z_tilde or latent_mean
        self.latest_posterior_hidden_state = posterior_hidden_state or self.latest_posterior_hidden_state
        self.latest_posterior_drift = posterior_drift or 0.0
        self.latest_beta_binary = beta_binary if beta_binary is not None else int(switch_gate >= self.beta_threshold)
        self.latest_switch_sparsity = switch_sparsity if switch_sparsity is not None else 1.0 - switch_gate
        self.latest_binary_switch_rate = binary_switch_rate if binary_switch_rate is not None else float(
            self.latest_beta_binary
        )
        self.latest_mean_persistence_window = (
            mean_persistence_window if mean_persistence_window is not None else self.latest_mean_persistence_window
        )
        self.latest_decoder_applied_control = decoder_applied_control or decoder_control
        self.latest_policy_replacement_score = (
            policy_replacement_score if policy_replacement_score is not None else self.latest_policy_replacement_score
        )

    def record_ssl_metrics(self, *, total_loss: float, kl_loss: float) -> None:
        self.latest_ssl_loss = total_loss
        self.latest_ssl_kl_loss = kl_loss

    def set_learning_phase(self, phase: str, *, structure_frozen: bool | None = None) -> None:
        self.learning_phase = phase
        if structure_frozen is not None:
            self.structure_frozen = structure_frozen

    def _public_action_family_summaries(self) -> tuple[ActionFamilyPublicSummary, ...]:
        return tuple(
            ActionFamilyPublicSummary(
                family_id=family.family_id,
                dominant_axis=_family_dominant_axis(family.decoder_centroid),
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
                summary=family.summary,
            )
            for family in self.action_families
        )

    def _action_family_turnover_health(
        self,
        action_family_summaries: tuple[ActionFamilyPublicSummary, ...],
    ) -> float:
        if not action_family_summaries:
            return 0.0
        average_competition = sum(summary.competition_score for summary in action_family_summaries) / len(
            action_family_summaries
        )
        average_stagnation = sum(summary.stagnation_pressure for summary in action_family_summaries) / len(
            action_family_summaries
        )
        average_monopoly = sum(summary.monopoly_pressure for summary in action_family_summaries) / len(
            action_family_summaries
        )
        diversity = _clamp(len(action_family_summaries) / 4.0)
        return _clamp(
            average_competition * 0.35
            + diversity * 0.30
            + (1.0 - average_stagnation) * 0.20
            + (1.0 - average_monopoly) * 0.15
        )

    def discover_action_family(
        self,
        *,
        latent_code: tuple[float, ...],
        decoder_control: tuple[float, ...],
        switch_gate: float,
        posterior_drift: float = 0.0,
        persistence_window: float = 0.0,
    ) -> tuple[str, str]:
        observation = ActionFamilyObservation(
            latent_code=latent_code,
            decoder_control=decoder_control,
            switch_gate=switch_gate,
            posterior_drift=posterior_drift,
            persistence_window=persistence_window,
        )
        previous_families = self.action_families
        self.action_families, active_label, family_summary = discover_latent_action_family(
            observation=observation,
            action_families=self.action_families,
            structure_frozen=self.structure_frozen,
            allow_topology_maintenance=self.learning_phase.startswith("ssl") or not self.action_families,
        )
        if self.action_families != previous_families:
            self._action_family_version += 1
        self.latest_active_label = active_label
        return (active_label, family_summary)

    def fit_temporal_from_signals(
        self,
        *,
        residual_strength: float,
        memory_strength: float,
        reflection_strength: float,
    ) -> None:
        total = max(residual_strength + memory_strength + reflection_strength, 1e-6)
        self.temporal_weights = {
            "residual": residual_strength / total,
            "memory": memory_strength / total,
            "reflection": reflection_strength / total,
        }

    def apply_reflection_prior_update(
        self,
        *,
        update: TemporalPriorUpdate,
        allowed_target_groups: tuple[str, ...] | None = None,
    ) -> tuple[str, ...]:
        active_groups = set(allowed_target_groups or update.target_groups or ("base-weights",))
        operations: list[str] = []
        if "base-weights" in active_groups:
            current = dict(self.temporal_weights)
            blended_residual = _clamp(current["residual"] * 0.75 + update.residual_strength * 0.25)
            blended_memory = _clamp(current["memory"] * 0.75 + update.memory_strength * 0.25)
            blended_reflection = _clamp(current["reflection"] * 0.75 + update.reflection_strength * 0.25)
            self.fit_temporal_from_signals(
                residual_strength=blended_residual,
                memory_strength=blended_memory,
                reflection_strength=blended_reflection,
            )
            operations.extend(
                (
                    f"temporal-prior:residual={self.temporal_weights['residual']:.3f}",
                    f"temporal-prior:memory={self.temporal_weights['memory']:.3f}",
                    f"temporal-prior:reflection={self.temporal_weights['reflection']:.3f}",
                )
            )
        if "switch" in active_groups:
            self.switch_bias = _clamp(self.switch_bias + update.switch_bias_delta)
            operations.append(f"temporal-prior:switch-bias={self.switch_bias:.3f}")
        if "persistence" in active_groups:
            self.persistence = _clamp(self.persistence + update.persistence_delta)
            operations.append(f"temporal-prior:persistence={self.persistence:.3f}")
        if "learning-rate" in active_groups:
            self.learning_rate = _clamp(self.learning_rate + update.learning_rate_delta)
            operations.append(f"temporal-prior:learning-rate={self.learning_rate:.3f}")
        if "beta-threshold" in active_groups:
            self.beta_threshold = _clamp(self.beta_threshold + update.beta_threshold_delta)
            operations.append(f"temporal-prior:beta-threshold={self.beta_threshold:.3f}")
        if "encoder" in active_groups:
            self.encoder_weights = _scale_matrix(self.encoder_weights, update.encoder_strength_delta)
            self.encoder_recurrence = _scale_matrix(self.encoder_recurrence, update.encoder_strength_delta * 0.75)
            operations.append(f"temporal-prior:encoder={update.encoder_strength_delta:.3f}")
        if "decoder" in active_groups:
            self.decoder_matrix = _scale_matrix(self.decoder_matrix, update.decoder_strength_delta)
            self.decoder_hidden = _scale_matrix(self.decoder_hidden, update.decoder_strength_delta * 0.75)
            operations.append(f"temporal-prior:decoder={update.decoder_strength_delta:.3f}")
        if "track-world" in active_groups:
            self.track_weights[Track.WORLD] = _blend_track_weights(
                self.track_weights[Track.WORLD],
                self.latest_decoder_applied_control or self.latest_latent_mean,
                delta=update.world_track_delta,
            )
            operations.append(f"temporal-prior:track-world={update.world_track_delta:.3f}")
        if "track-self" in active_groups:
            self.track_weights[Track.SELF] = _blend_track_weights(
                self.track_weights[Track.SELF],
                self.latest_posterior_mean or self.latest_latent_mean,
                delta=update.self_track_delta,
            )
            operations.append(f"temporal-prior:track-self={update.self_track_delta:.3f}")
        if "track-shared" in active_groups:
            shared_anchor = tuple(
                (
                    (self.latest_decoder_applied_control[i] if i < len(self.latest_decoder_applied_control) else 0.0)
                    + (self.latest_posterior_mean[i] if i < len(self.latest_posterior_mean) else 0.0)
                )
                / 2.0
                for i in range(self._n_z)
            )
            self.track_weights[Track.SHARED] = _blend_track_weights(
                self.track_weights[Track.SHARED],
                shared_anchor,
                delta=update.shared_track_delta,
            )
            operations.append(f"temporal-prior:track-shared={update.shared_track_delta:.3f}")
        if "action-families" in active_groups:
            self.action_families = tuple(
                DiscoveredActionFamily(
                    family_id=family.family_id,
                    latent_centroid=family.latent_centroid,
                    decoder_centroid=family.decoder_centroid,
                    support=family.support,
                    stability=_clamp(family.stability + update.family_stability_delta),
                    switch_bias=_clamp(family.switch_bias + update.beta_threshold_delta),
                    summary=family.summary,
                )
                for family in self.action_families
            )
            self._action_family_version += 1
            operations.append(f"temporal-prior:action-families={update.family_stability_delta:.3f}")
        if "action-family-structure" in active_groups and update.structure_proposals:
            before_families = self.action_families
            self.action_families, structure_ops = _apply_action_family_structure_proposals(
                action_families=self.action_families,
                proposals=update.structure_proposals,
            )
            if self.action_families != before_families:
                self._action_family_version += 1
            operations.extend(structure_ops)
        return tuple(operations)

    def align_temporal_from_tracks(self) -> None:
        world_weights = self.track_weights[Track.WORLD]
        self_weights = self.track_weights[Track.SELF]
        shared_weights = self.track_weights[Track.SHARED]
        residual_strength = _clamp((world_weights[0] + shared_weights[0]) / 2.0)
        memory_strength = _clamp((self_weights[1] + shared_weights[1]) / 2.0)
        reflection_strength = _clamp((world_weights[2] + self_weights[2] + shared_weights[2]) / 3.0)
        self.fit_temporal_from_signals(
            residual_strength=residual_strength,
            memory_strength=memory_strength,
            reflection_strength=reflection_strength,
        )
        self.switch_bias = _clamp(1.0 - self.persistence)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _scale_matrix(
    matrix: tuple[tuple[float, ...], ...],
    delta: float,
) -> tuple[tuple[float, ...], ...]:
    factor = 1.0 + delta * 0.18
    return tuple(
        tuple(max(-1.0, min(1.0, value * factor)) for value in row)
        for row in matrix
    )


def _normalize_vector(values: tuple[float, ...]) -> tuple[float, ...]:
    total = sum(max(value, 0.0) for value in values)
    if total <= 1e-9:
        return tuple(1.0 / len(values) for _ in values) if values else ()
    return tuple(max(value, 0.0) / total for value in values)


def _blend_track_weights(
    current: tuple[float, ...],
    anchor: tuple[float, ...],
    *,
    delta: float,
) -> tuple[float, ...]:
    if not current:
        return current
    normalized_anchor = _normalize_vector(
        tuple(anchor[index] if index < len(anchor) else 0.0 for index in range(len(current)))
    )
    blend = min(max(abs(delta), 0.0), 0.25)
    if delta < 0.0:
        normalized_anchor = tuple(1.0 / len(current) for _ in current)
    return _normalize_vector(
        tuple(
            current[index] * (1.0 - blend) + normalized_anchor[index] * blend
            for index in range(len(current))
        )
    )


def _family_dominant_axis(values: tuple[float, ...]) -> str:
    if not values:
        return "unknown"
    index = max(range(len(values)), key=lambda i: values[i])
    if index == 0:
        return "world"
    if index == 1:
        return "self"
    return "shared"


def _blend_family_centroid(
    left: tuple[float, ...],
    right: tuple[float, ...],
    *,
    left_weight: float,
    right_weight: float,
) -> tuple[float, ...]:
    total = max(left_weight + right_weight, 1e-6)
    return tuple(
        _clamp((l_value * left_weight + r_value * right_weight) / total)
        for l_value, r_value in zip(left, right, strict=True)
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
        summary=(
            f"{prefix} dominant_axis={_family_dominant_axis(family.decoder_centroid)} "
            f"support={family.support} stability={family.stability:.3f}"
        ),
    )


def _tilt_family_centroid(
    centroid: tuple[float, ...],
    *,
    axis: str,
    amount: float = 0.12,
) -> tuple[float, ...]:
    if not centroid:
        return centroid
    axis_index = 0 if axis == "world" else 1 if axis == "self" else 2
    updated = list(centroid)
    updated[axis_index] = _clamp(updated[axis_index] + amount)
    for index in range(len(updated)):
        if index != axis_index:
            updated[index] = _clamp(updated[index] - amount * 0.5)
    return tuple(updated)


def _next_family_id(action_families: tuple[DiscoveredActionFamily, ...]) -> str:
    next_index = max(
        (
            int(family.family_id.rsplit("_", 1)[-1])
            for family in action_families
            if family.family_id.rsplit("_", 1)[-1].isdigit()
        ),
        default=-1,
    ) + 1
    return f"discovered_family_{next_index}"


def _apply_action_family_structure_proposals(
    *,
    action_families: tuple[DiscoveredActionFamily, ...],
    proposals: tuple[TemporalStructureProposal, ...],
) -> tuple[tuple[DiscoveredActionFamily, ...], tuple[str, ...]]:
    families = list(action_families)
    operations: list[str] = []
    for proposal in proposals:
        index_by_id = {family.family_id: index for index, family in enumerate(families)}
        if proposal.proposal_type == "prune":
            index = index_by_id.get(proposal.family_id)
            if index is None or len(families) <= 1:
                continue
            families.pop(index)
            operations.append(f"temporal-prior:action-family-prune={proposal.family_id}")
            continue
        if proposal.proposal_type == "merge":
            left_index = index_by_id.get(proposal.family_id)
            right_index = index_by_id.get(proposal.related_family_id or "")
            if left_index is None or right_index is None or left_index == right_index:
                continue
            left = families[left_index]
            right = families[right_index]
            merged = _refresh_family_summary(
                DiscoveredActionFamily(
                    family_id=left.family_id,
                    latent_centroid=_blend_family_centroid(
                        left.latent_centroid,
                        right.latent_centroid,
                        left_weight=float(left.support),
                        right_weight=float(right.support),
                    ),
                    decoder_centroid=_blend_family_centroid(
                        left.decoder_centroid,
                        right.decoder_centroid,
                        left_weight=float(left.support),
                        right_weight=float(right.support),
                    ),
                    support=left.support + right.support,
                    stability=_clamp((left.stability + right.stability) / 2.0),
                    switch_bias=_clamp((left.switch_bias + right.switch_bias) / 2.0),
                    mean_posterior_drift=_clamp(
                        (left.mean_posterior_drift + right.mean_posterior_drift) / 2.0
                    ),
                    mean_persistence_window=_clamp(
                        (left.mean_persistence_window + right.mean_persistence_window) / 2.0
                    ),
                ),
                prefix=f"reflect-merge:{left.family_id}+{right.family_id}",
            )
            primary_index = min(left_index, right_index)
            secondary_index = max(left_index, right_index)
            families[primary_index] = merged
            families.pop(secondary_index)
            operations.append(
                f"temporal-prior:action-family-merge={proposal.family_id}+{proposal.related_family_id}"
            )
            continue
        if proposal.proposal_type == "split":
            index = index_by_id.get(proposal.family_id)
            if index is None:
                continue
            family = families[index]
            child_id = _next_family_id(tuple(families))
            child = _refresh_family_summary(
                DiscoveredActionFamily(
                    family_id=child_id,
                    latent_centroid=_tilt_family_centroid(
                        family.latent_centroid,
                        axis=_family_dominant_axis(family.decoder_centroid),
                    ),
                    decoder_centroid=_tilt_family_centroid(
                        family.decoder_centroid,
                        axis=_family_dominant_axis(family.decoder_centroid),
                    ),
                    support=max(1, family.support // 2),
                    stability=_clamp(family.stability * 0.85),
                    switch_bias=family.switch_bias,
                    mean_posterior_drift=_clamp(family.mean_posterior_drift + 0.08),
                    mean_persistence_window=_clamp(max(family.mean_persistence_window - 0.1, 0.0)),
                ),
                prefix=f"reflect-split:{proposal.family_id}",
            )
            families[index] = _refresh_family_summary(
                DiscoveredActionFamily(
                    family_id=family.family_id,
                    latent_centroid=family.latent_centroid,
                    decoder_centroid=family.decoder_centroid,
                    support=max(1, family.support - child.support),
                    stability=_clamp(family.stability * 0.92),
                    switch_bias=family.switch_bias,
                    mean_posterior_drift=family.mean_posterior_drift,
                    mean_persistence_window=family.mean_persistence_window,
                ),
                prefix=f"reflect-split-parent:{proposal.family_id}",
            )
            families.append(child)
            operations.append(f"temporal-prior:action-family-split={proposal.family_id}->{child_id}")
    return (tuple(families), tuple(operations))


def _random_mat(rows: int, cols: int, *, seed: int) -> tuple[tuple[float, ...], ...]:
    """Deterministic random matrix for n_z > 3 initialization."""
    import random as _rng
    r = _rng.Random(seed)
    scale = 1.0 / max(rows, 1) ** 0.5
    return tuple(
        tuple(r.gauss(0.0, scale) for _ in range(cols))
        for _ in range(rows)
    )


def _random_vec(n: int, *, seed: int) -> tuple[float, ...]:
    import random as _rng
    r = _rng.Random(seed)
    return tuple(r.gauss(0.0, 0.1) for _ in range(n))


def _init_action_families(n_z: int, *, seed: int) -> tuple[DiscoveredActionFamily, ...]:
    del n_z
    del seed
    return ()


def _init_track_weights(n_z: int, *, seed: int) -> dict:
    import random as _rng
    r = _rng.Random(seed)
    result = {}
    for track in (Track.WORLD, Track.SELF, Track.SHARED):
        raw = tuple(abs(r.gauss(0.4, 0.2)) for _ in range(n_z))
        total = max(sum(raw), 1e-6)
        result[track] = tuple(v / total for v in raw)
    return result


def _nz_zeros(n: int) -> tuple[float, ...]:
    return tuple(0.0 for _ in range(n))


def _nz_ones(n: int) -> tuple[float, ...]:
    return tuple(1.0 for _ in range(n))


def _hash_payload(payload: object) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(serialized.encode("utf-8")).hexdigest()


def _feature_signature(feature_surface: tuple[FeatureSignal, ...]) -> tuple[str, ...]:
    return tuple(feature.name for feature in feature_surface[:4])


def _residual_signature(substrate_snapshot: SubstrateSnapshot) -> tuple[float, ...]:
    sequence = residual_sequence_from_snapshot(substrate_snapshot)
    summaries = tuple(
        summarize_residual_activations(step.residual_activations, step.feature_surface) for step in sequence
    )
    if not summaries:
        return _code_from_feature_surface(substrate_snapshot.feature_surface)
    return tuple(
        _clamp(sum(summary[index] for summary in summaries) / len(summaries)) for index in range(3)
    )


def _code_from_feature_surface(feature_surface: tuple[FeatureSignal, ...]) -> tuple[float, ...]:
    return summarize_feature_surface(feature_surface)


def _abstract_action_from_code(code: tuple[float, ...], switch_gate: float) -> str:
    average, maximum, spread = code
    if switch_gate > 0.7:
        return "refresh-controller-context"
    if spread > 0.35:
        return "focus-dominant-signal"
    if average < 0.2 and maximum < 0.25:
        return "hold-low-signal-context"
    return "stabilize-current-controller"


def _memory_signal(memory_snapshot: MemorySnapshot | None) -> float:
    if memory_snapshot is None:
        return 0.0
    retrieval_pressure = min(len(memory_snapshot.retrieved_entries) / 5.0, 1.0)
    promotion_pressure = min(memory_snapshot.pending_promotions / 5.0, 1.0)
    return _clamp((retrieval_pressure + promotion_pressure) / 2.0)


def _reflection_signal(reflection_snapshot: ReflectionSnapshot | None) -> float:
    if reflection_snapshot is None:
        return 0.0
    lesson_pressure = min(len(reflection_snapshot.lessons_extracted) / 4.0, 1.0)
    tension_pressure = min(len(reflection_snapshot.tensions_identified) / 4.0, 1.0)
    return _clamp((lesson_pressure + tension_pressure) / 2.0)


def _cms_band(memory_snapshot: MemorySnapshot | None, band_name: str) -> tuple[float, ...] | None:
    if memory_snapshot is None or memory_snapshot.cms_state is None:
        return None
    band = getattr(memory_snapshot.cms_state, band_name, None)
    if band is None:
        return None
    return band.vector


class TemporalPolicy(ABC):
    """Common interface for placeholder, heuristic, and future learned policies."""

    mode: TemporalImplementationMode

    @abstractmethod
    def step(
        self,
        *,
        substrate_snapshot: SubstrateSnapshot,
        previous_snapshot: TemporalAbstractionSnapshot | None,
        memory_snapshot: MemorySnapshot | None = None,
        reflection_snapshot: ReflectionSnapshot | None = None,
    ) -> TemporalStep:
        """Produce the next temporal abstraction state."""

    def export_runtime_state(self) -> MetacontrollerRuntimeState | None:
        return None

    def apply_reflection_prior_update(
        self,
        *,
        update: TemporalPriorUpdate,
        allowed_target_groups: tuple[str, ...] | None = None,
    ) -> tuple[str, ...]:
        del update
        del allowed_target_groups
        return ()


class PlaceholderTemporalPolicy(TemporalPolicy):
    mode = TemporalImplementationMode.PLACEHOLDER

    def step(
        self,
        *,
        substrate_snapshot: SubstrateSnapshot,
        previous_snapshot: TemporalAbstractionSnapshot | None,
        memory_snapshot: MemorySnapshot | None = None,
        reflection_snapshot: ReflectionSnapshot | None = None,
    ) -> TemporalStep:
        steps_since_switch = (
            previous_snapshot.controller_state.steps_since_switch + 1
            if previous_snapshot is not None
            else 0
        )
        controller_state = ControllerState(
            code=(0.0, 0.0, 0.0),
            code_dim=3,
            switch_gate=0.0,
            is_switching=False,
            steps_since_switch=steps_since_switch,
        )
        params_hash = _hash_payload(
            {
                "mode": self.mode.value,
                "model_id": substrate_snapshot.model_id,
            }
        )
        return TemporalStep(
            controller_state=controller_state,
            active_abstract_action="placeholder-controller",
            controller_params_hash=params_hash,
            description="Placeholder temporal controller with no active switching.",
            action_family_version=0,
        )


class HeuristicTemporalPolicy(TemporalPolicy):
    mode = TemporalImplementationMode.HEURISTIC

    def __init__(self) -> None:
        self._previous_feature_signature: tuple[str, ...] = ()

    def step(
        self,
        *,
        substrate_snapshot: SubstrateSnapshot,
        previous_snapshot: TemporalAbstractionSnapshot | None,
        memory_snapshot: MemorySnapshot | None = None,
        reflection_snapshot: ReflectionSnapshot | None = None,
    ) -> TemporalStep:
        feature_signature = _feature_signature(substrate_snapshot.feature_surface)
        residual_code = _residual_signature(substrate_snapshot)
        memory_signal = _memory_signal(memory_snapshot)
        reflection_signal = _reflection_signal(reflection_snapshot)
        code = (
            _clamp(residual_code[0]),
            _clamp((residual_code[1] + memory_signal) / 2.0),
            _clamp((residual_code[2] + reflection_signal) / 2.0),
        )
        previous_steps = 0
        if previous_snapshot is not None:
            previous_steps = previous_snapshot.controller_state.steps_since_switch

        signature_changed = feature_signature != self._previous_feature_signature
        switch_gate = 0.15
        if signature_changed and feature_signature:
            switch_gate = 0.75 + memory_signal * 0.1 + reflection_signal * 0.1
        is_switching = switch_gate > 0.7
        steps_since_switch = 0 if is_switching else previous_steps + 1
        active_action = _abstract_action_from_code(code, switch_gate)
        signature_suffix = "|".join(feature_signature) if feature_signature else "no-feature-signal"
        controller_state = ControllerState(
            code=code,
            code_dim=len(code),
            switch_gate=switch_gate,
            is_switching=is_switching,
            steps_since_switch=steps_since_switch,
        )
        params_hash = _hash_payload(
            {
                "mode": self.mode.value,
                "feature_signature": feature_signature,
                "code": code,
            }
        )
        step = TemporalStep(
            controller_state=controller_state,
            active_abstract_action=f"{active_action}:{signature_suffix}",
            controller_params_hash=params_hash,
            description=(
                f"Heuristic temporal controller mode={self.mode.value}, "
                f"switch_gate={switch_gate:.2f}, feature_signature={signature_suffix}."
            ),
            action_family_version=0,
        )
        self._previous_feature_signature = feature_signature
        return step


class LearnedLiteTemporalPolicy(TemporalPolicy):
    mode = TemporalImplementationMode.LEARNED_LITE

    def __init__(self, *, parameter_store: MetacontrollerParameterStore | None = None) -> None:
        self._parameter_store = parameter_store or MetacontrollerParameterStore()
        self._previous_code = (0.0, 0.0, 0.0)

    @property
    def weights(self) -> Mapping[str, float]:
        return dict(self._parameter_store.temporal_weights)

    @property
    def parameter_store(self) -> MetacontrollerParameterStore:
        return self._parameter_store

    def export_runtime_state(self) -> MetacontrollerRuntimeState:
        return self._parameter_store.export_runtime_state(mode=self.mode.value)

    def export_parameters(self) -> TemporalControllerParameters:
        return self._parameter_store.export_temporal_parameters()

    def export_rare_heavy_snapshot(self) -> MetacontrollerParameterSnapshot:
        return self._parameter_store.export_parameter_snapshot()

    def fit_from_signals(
        self,
        *,
        residual_strength: float,
        memory_strength: float,
        reflection_strength: float,
    ) -> None:
        self._parameter_store.fit_temporal_from_signals(
            residual_strength=residual_strength,
            memory_strength=memory_strength,
            reflection_strength=reflection_strength,
        )

    def apply_reflection_prior_update(
        self,
        *,
        update: TemporalPriorUpdate,
        allowed_target_groups: tuple[str, ...] | None = None,
    ) -> tuple[str, ...]:
        return self._parameter_store.apply_reflection_prior_update(
            update=update,
            allowed_target_groups=allowed_target_groups,
        )

    def apply_rare_heavy_snapshot(self, snapshot: MetacontrollerParameterSnapshot) -> tuple[str, ...]:
        self._parameter_store.restore_parameter_snapshot(snapshot)
        return ("rare-heavy:temporal-import",)

    def align_with_internal_rl(
        self,
        *,
        world_weights: tuple[float, ...],
        self_weights: tuple[float, ...],
        shared_weights: tuple[float, ...],
        persistence: float,
    ) -> None:
        self._parameter_store.track_weights[Track.WORLD] = world_weights
        self._parameter_store.track_weights[Track.SELF] = self_weights
        self._parameter_store.track_weights[Track.SHARED] = shared_weights
        self._parameter_store.persistence = persistence
        self._parameter_store.align_temporal_from_tracks()

    def step(
        self,
        *,
        substrate_snapshot: SubstrateSnapshot,
        previous_snapshot: TemporalAbstractionSnapshot | None,
        memory_snapshot: MemorySnapshot | None = None,
        reflection_snapshot: ReflectionSnapshot | None = None,
    ) -> TemporalStep:
        residual_code = _residual_signature(substrate_snapshot)
        memory_signal = _memory_signal(memory_snapshot)
        reflection_signal = _reflection_signal(reflection_snapshot)
        code = (
            _clamp(
                residual_code[0] * self._parameter_store.temporal_weights["residual"]
                + memory_signal * self._parameter_store.temporal_weights["memory"]
            ),
            _clamp(
                residual_code[1] * self._parameter_store.temporal_weights["residual"]
                + reflection_signal * self._parameter_store.temporal_weights["reflection"]
            ),
            _clamp(
                residual_code[2] * self._parameter_store.temporal_weights["residual"]
                + (memory_signal + reflection_signal) / 2.0
            ),
        )
        delta = sum(abs(current - previous) for current, previous in zip(code, self._previous_code))
        switch_gate = _clamp(self._parameter_store.switch_bias + delta / 2.0 + reflection_signal * 0.2)
        is_switching = switch_gate >= 0.55
        previous_steps = (
            previous_snapshot.controller_state.steps_since_switch if previous_snapshot is not None else 0
        )
        steps_since_switch = 0 if is_switching else previous_steps + 1
        active_action = _abstract_action_from_code(code, switch_gate)
        params_hash = _hash_payload(
            {
                "mode": self.mode.value,
                "weights": self._parameter_store.temporal_weights,
                "switch_bias": self._parameter_store.switch_bias,
            }
        )
        description = (
            f"Learned-lite temporal controller residual={self._parameter_store.temporal_weights['residual']:.2f}, "
            f"memory={self._parameter_store.temporal_weights['memory']:.2f}, "
            f"reflection={self._parameter_store.temporal_weights['reflection']:.2f}, "
            f"switch_gate={switch_gate:.2f}."
        )
        self._parameter_store.record_runtime_observation(
            latent_mean=code,
            latent_scale=tuple(abs(current - previous) for current, previous in zip(code, self._previous_code)),
            decoder_control=code,
            switch_gate=switch_gate,
            sequence_length=len(residual_sequence_from_snapshot(substrate_snapshot)),
            active_label=f"{active_action}:learned-lite",
        )
        self._previous_code = code
        return TemporalStep(
            controller_state=ControllerState(
                code=code,
                code_dim=len(code),
                switch_gate=switch_gate,
                is_switching=is_switching,
                steps_since_switch=steps_since_switch,
            ),
            active_abstract_action=f"{active_action}:learned-lite",
            controller_params_hash=params_hash,
            description=description,
            action_family_version=0,
        )


class FullLearnedTemporalPolicy(TemporalPolicy):
    mode = TemporalImplementationMode.FULL_LEARNED

    def __init__(self, *, parameter_store: MetacontrollerParameterStore | None = None) -> None:
        self._parameter_store = parameter_store or MetacontrollerParameterStore()
        n_z = self._parameter_store.n_z
        self._encoder = SequenceEncoder()
        self._switch_unit = SwitchUnit()
        self._decoder = ResidualDecoder()
        self._ndim_encoder: NdimSequenceEncoder | None = None
        self._ndim_switch: NdimSwitchUnit | None = None
        self._ndim_decoder: NdimResidualDecoder | None = None
        if n_z > 3:
            self._ndim_encoder = NdimSequenceEncoder(n_z=n_z)
            self._ndim_switch = NdimSwitchUnit(n_z=n_z)
            self._ndim_decoder = NdimResidualDecoder(n_z=n_z)
        self._previous_code = _nz_zeros(n_z)
        self._previous_hidden_state = _nz_zeros(n_z)
        self._previous_beta_binary = 0
        self._latest_encoder_output_for_cms: tuple[float, ...] | None = None

    @property
    def parameter_store(self) -> MetacontrollerParameterStore:
        return self._parameter_store

    @property
    def latest_encoder_output_for_cms(self) -> tuple[float, ...] | None:
        return self._latest_encoder_output_for_cms

    def export_runtime_state(self) -> MetacontrollerRuntimeState:
        return self._parameter_store.export_runtime_state(mode=self.mode.value)

    def export_parameters(self) -> TemporalControllerParameters:
        return self._parameter_store.export_temporal_parameters()

    def export_rare_heavy_snapshot(self) -> MetacontrollerParameterSnapshot:
        return self._parameter_store.export_parameter_snapshot()

    def fit_from_signals(
        self,
        *,
        residual_strength: float,
        memory_strength: float,
        reflection_strength: float,
    ) -> None:
        self._parameter_store.fit_temporal_from_signals(
            residual_strength=residual_strength,
            memory_strength=memory_strength,
            reflection_strength=reflection_strength,
        )

    def apply_reflection_prior_update(
        self,
        *,
        update: TemporalPriorUpdate,
        allowed_target_groups: tuple[str, ...] | None = None,
    ) -> tuple[str, ...]:
        return self._parameter_store.apply_reflection_prior_update(
            update=update,
            allowed_target_groups=allowed_target_groups,
        )

    def apply_rare_heavy_snapshot(self, snapshot: MetacontrollerParameterSnapshot) -> tuple[str, ...]:
        self._parameter_store.restore_parameter_snapshot(snapshot)
        return ("rare-heavy:temporal-import",)

    def step_with_causal_override(
        self,
        *,
        substrate_snapshot: SubstrateSnapshot,
        previous_snapshot: TemporalAbstractionSnapshot | None,
        latent_override: tuple[float, ...],
        policy_replacement_score: float,
        binary_gate_override: bool = False,
        memory_snapshot: MemorySnapshot | None = None,
        reflection_snapshot: ReflectionSnapshot | None = None,
    ) -> TemporalStep:
        return self._step_impl(
            substrate_snapshot=substrate_snapshot,
            previous_snapshot=previous_snapshot,
            memory_snapshot=memory_snapshot,
            reflection_snapshot=reflection_snapshot,
            latent_override=latent_override,
            policy_replacement_score=policy_replacement_score,
            binary_gate_override=binary_gate_override,
        )

    def step(
        self,
        *,
        substrate_snapshot: SubstrateSnapshot,
        previous_snapshot: TemporalAbstractionSnapshot | None,
        memory_snapshot: MemorySnapshot | None = None,
        reflection_snapshot: ReflectionSnapshot | None = None,
    ) -> TemporalStep:
        return self._step_impl(
            substrate_snapshot=substrate_snapshot,
            previous_snapshot=previous_snapshot,
            memory_snapshot=memory_snapshot,
            reflection_snapshot=reflection_snapshot,
            latent_override=None,
            policy_replacement_score=0.0,
            binary_gate_override=False,
        )

    def _compute_track_codes(
        self,
        latent_code: tuple[float, ...],
    ) -> tuple[tuple[str, tuple[float, ...]], ...]:
        result: list[tuple[str, tuple[float, ...]]] = []
        for track in (Track.WORLD, Track.SELF, Track.SHARED):
            weights = self._parameter_store.track_weights[track]
            projected = tuple(
                _clamp(latent_code[i] * weights[i])
                for i in range(min(len(latent_code), len(weights)))
            )
            result.append((track.value, projected))
        return tuple(result)

    def _step_impl(
        self,
        *,
        substrate_snapshot: SubstrateSnapshot,
        previous_snapshot: TemporalAbstractionSnapshot | None,
        memory_snapshot: MemorySnapshot | None,
        reflection_snapshot: ReflectionSnapshot | None,
        latent_override: tuple[float, ...] | None,
        policy_replacement_score: float,
        binary_gate_override: bool,
    ) -> TemporalStep:
        if self._ndim_encoder is not None:
            return self._step_impl_ndim(
                substrate_snapshot=substrate_snapshot,
                previous_snapshot=previous_snapshot,
                memory_snapshot=memory_snapshot,
                reflection_snapshot=reflection_snapshot,
                latent_override=latent_override,
                policy_replacement_score=policy_replacement_score,
                binary_gate_override=binary_gate_override,
            )
        return self._step_impl_legacy(
            substrate_snapshot=substrate_snapshot,
            previous_snapshot=previous_snapshot,
            memory_snapshot=memory_snapshot,
            reflection_snapshot=reflection_snapshot,
            latent_override=latent_override,
            policy_replacement_score=policy_replacement_score,
            binary_gate_override=binary_gate_override,
        )

    def _step_impl_ndim(
        self,
        *,
        substrate_snapshot: SubstrateSnapshot,
        previous_snapshot: TemporalAbstractionSnapshot | None,
        memory_snapshot: MemorySnapshot | None,
        reflection_snapshot: ReflectionSnapshot | None,
        latent_override: tuple[float, ...] | None,
        policy_replacement_score: float,
        binary_gate_override: bool,
    ) -> TemporalStep:
        assert self._ndim_encoder is not None
        assert self._ndim_switch is not None
        assert self._ndim_decoder is not None
        n_z = self._parameter_store.n_z
        previous_code = previous_snapshot.controller_state.code if previous_snapshot is not None else self._previous_code
        previous_steps = (
            previous_snapshot.controller_state.steps_since_switch if previous_snapshot is not None else 0
        )
        cms_fast = _cms_band(memory_snapshot, "online_fast")
        cms_medium = _cms_band(memory_snapshot, "session_medium")
        cms_slow = _cms_band(memory_snapshot, "background_slow")
        from volvence_zero.temporal.metacontroller_components import _project_to_ndim
        cms_ctx: tuple[float, ...] | None = None
        if cms_fast or cms_medium or cms_slow:
            z = _nz_zeros(n_z)
            fast = _project_to_ndim(cms_fast, n_z) if cms_fast else z
            med = _project_to_ndim(cms_medium, n_z) if cms_medium else z
            slow = _project_to_ndim(cms_slow, n_z) if cms_slow else z
            from volvence_zero.temporal.tensor_ops import vec_add as _va, vec_scale as _vs, vec_clamp as _vc
            cms_ctx = _vc(_va(_va(_vs(fast, 0.5), _vs(med, 0.3)), _vs(slow, 0.2)), 0.0, 1.0)
        encoded = self._ndim_encoder.encode(
            substrate_snapshot=substrate_snapshot,
            previous_hidden_state=self._previous_hidden_state,
            cms_context=cms_ctx,
        )
        self._latest_encoder_output_for_cms = tuple(
            _clamp(encoded.posterior.posterior_mean[i] * 0.6 + encoded.posterior.z_tilde[i] * 0.4)
            for i in range(n_z)
        )
        memory_signal = _memory_signal(memory_snapshot)
        reflection_signal = _reflection_signal(reflection_snapshot)
        beta_cont, beta_bin, scalar_beta = self._ndim_switch.compute(
            z_tilde=encoded.z_tilde,
            previous_code=previous_code,
            memory_signal=memory_signal,
            reflection_signal=reflection_signal,
        )
        if binary_gate_override:
            effective_gate = beta_bin
        else:
            effective_gate = beta_cont
        z_candidate = latent_override or encoded.z_tilde
        latent_code = tuple(
            _clamp(effective_gate[i] * z_candidate[i] + (1.0 - effective_gate[i]) * previous_code[i])
            for i in range(n_z)
        )
        decoder_control = self._ndim_decoder.decode(latent_code=latent_code)
        is_switching_scalar = scalar_beta >= 0.55
        active_label, decoder_summary = self._parameter_store.discover_action_family(
            latent_code=latent_code,
            decoder_control=decoder_control.applied_control,
            switch_gate=scalar_beta,
            posterior_drift=encoded.posterior.posterior_drift,
            persistence_window=0.0 if is_switching_scalar else float(previous_steps + 1),
        )
        beta_binary_int = 1 if is_switching_scalar else 0
        steps_since_switch = 0 if is_switching_scalar else previous_steps + 1
        self._parameter_store.record_runtime_observation(
            latent_mean=encoded.latent_mean,
            latent_scale=encoded.latent_scale,
            decoder_control=decoder_control.decoder_output,
            switch_gate=scalar_beta,
            sequence_length=encoded.sequence_length,
            active_label=active_label,
            prior_mean=encoded.posterior.prior_mean,
            prior_std=encoded.posterior.prior_std,
            posterior_mean=encoded.posterior.posterior_mean,
            posterior_std=encoded.posterior.posterior_std,
            posterior_sample_noise=encoded.posterior.sample_noise,
            z_tilde=z_candidate,
            posterior_hidden_state=encoded.posterior.hidden_state,
            posterior_drift=encoded.posterior.posterior_drift,
            beta_binary=beta_binary_int,
            switch_sparsity=1.0 - scalar_beta,
            binary_switch_rate=float(beta_binary_int),
            mean_persistence_window=0.0 if is_switching_scalar else float(previous_steps + 1),
            decoder_applied_control=decoder_control.applied_control,
            policy_replacement_score=policy_replacement_score,
        )
        params_hash = _hash_payload({
            "mode": self.mode.value,
            "n_z": n_z,
            "beta_threshold": self._parameter_store.beta_threshold,
        })
        description = (
            f"Full-learned ndim metacontroller n_z={n_z}, "
            f"scalar_beta={scalar_beta:.3f}, seq_len={encoded.sequence_length}, "
            f"{encoded.summary}, {decoder_control.summary}, "
            f"replacement_score={policy_replacement_score:.3f}, {decoder_summary}."
        )
        self._previous_code = latent_code
        self._previous_hidden_state = encoded.posterior.hidden_state
        self._previous_beta_binary = beta_binary_int
        track_codes = self._compute_track_codes(latent_code)
        return TemporalStep(
            controller_state=ControllerState(
                code=latent_code,
                code_dim=len(latent_code),
                switch_gate=scalar_beta,
                is_switching=is_switching_scalar,
                steps_since_switch=steps_since_switch,
                track_codes=track_codes,
            ),
            active_abstract_action=active_label,
            controller_params_hash=params_hash,
            description=description,
            action_family_version=self._parameter_store.action_family_version,
        )

    def _step_impl_legacy(
        self,
        *,
        substrate_snapshot: SubstrateSnapshot,
        previous_snapshot: TemporalAbstractionSnapshot | None,
        memory_snapshot: MemorySnapshot | None,
        reflection_snapshot: ReflectionSnapshot | None,
        latent_override: tuple[float, ...] | None,
        policy_replacement_score: float,
        binary_gate_override: bool,
    ) -> TemporalStep:
        previous_code = previous_snapshot.controller_state.code if previous_snapshot is not None else self._previous_code
        previous_steps = (
            previous_snapshot.controller_state.steps_since_switch if previous_snapshot is not None else 0
        )
        encoded = self._encoder.encode(
            substrate_snapshot=substrate_snapshot,
            encoder_weights=self._parameter_store.encoder_weights,
            recurrence_weights=self._parameter_store.encoder_recurrence,
            previous_hidden_state=self._previous_hidden_state,
            cms_online_fast=_cms_band(memory_snapshot, "online_fast"),
            cms_session_medium=_cms_band(memory_snapshot, "session_medium"),
            cms_background_slow=_cms_band(memory_snapshot, "background_slow"),
        )
        self._latest_encoder_output_for_cms = self._encoder.encoder_output_for_cms(encoded)
        memory_signal = _memory_signal(memory_snapshot)
        reflection_signal = _reflection_signal(reflection_snapshot)
        switch_decision = self._switch_unit.compute_decision(
            previous_code=previous_code,
            z_tilde=encoded.z_tilde,
            posterior_std=encoded.latent_scale,
            switch_weights=self._parameter_store.switch_weights,
            switch_bias=self._parameter_store.switch_bias,
            memory_signal=memory_signal,
            reflection_signal=reflection_signal,
            previous_binary=self._previous_beta_binary,
            previous_steps_since_switch=previous_steps,
        )
        effective_switch_gate = (
            float(switch_decision.beta_binary) if binary_gate_override else switch_decision.beta_continuous
        )
        z_candidate = latent_override or encoded.z_tilde
        latent_code = tuple(
            _clamp(effective_switch_gate * current + (1.0 - effective_switch_gate) * previous)
            for current, previous in zip(z_candidate, previous_code, strict=True)
        )
        decoder_control = self._decoder.decode(
            latent_code=latent_code,
            decoder_matrix=self._parameter_store.decoder_matrix,
            hidden_matrix=self._parameter_store.decoder_hidden,
        )
        active_label, decoder_summary = self._parameter_store.discover_action_family(
            latent_code=latent_code,
            decoder_control=decoder_control.applied_control,
            switch_gate=effective_switch_gate,
            posterior_drift=encoded.posterior.posterior_drift,
            persistence_window=switch_decision.mean_persistence_window,
        )
        is_switching = bool(switch_decision.beta_binary)
        steps_since_switch = 0 if is_switching else previous_steps + 1
        self._parameter_store.record_runtime_observation(
            latent_mean=encoded.latent_mean,
            latent_scale=encoded.latent_scale,
            decoder_control=decoder_control.decoder_output,
            switch_gate=effective_switch_gate,
            sequence_length=encoded.sequence_length,
            active_label=active_label,
            prior_mean=encoded.posterior.prior_mean,
            prior_std=encoded.posterior.prior_std,
            posterior_mean=encoded.posterior.posterior_mean,
            posterior_std=encoded.posterior.posterior_std,
            posterior_sample_noise=encoded.posterior.sample_noise,
            z_tilde=z_candidate,
            posterior_hidden_state=encoded.posterior.hidden_state,
            posterior_drift=encoded.posterior.posterior_drift,
            beta_binary=switch_decision.beta_binary,
            switch_sparsity=switch_decision.sparsity,
            binary_switch_rate=switch_decision.binary_switch_rate,
            mean_persistence_window=switch_decision.mean_persistence_window,
            decoder_applied_control=decoder_control.applied_control,
            policy_replacement_score=policy_replacement_score,
        )
        params_hash = _hash_payload(
            {
                "mode": self.mode.value,
                "encoder_weights": self._parameter_store.encoder_weights,
                "encoder_recurrence": self._parameter_store.encoder_recurrence,
                "switch_weights": self._parameter_store.switch_weights,
                "decoder_hidden": self._parameter_store.decoder_hidden,
                "decoder_matrix": self._parameter_store.decoder_matrix,
                "track_weights": self._parameter_store.track_weights,
                "beta_threshold": self._parameter_store.beta_threshold,
            }
        )
        description = (
            f"Full-learned metacontroller {switch_decision.summary}, "
            f"effective_beta={effective_switch_gate:.3f}, seq_len={encoded.sequence_length}, "
            f"{encoded.summary}, {decoder_control.summary}, replacement_score={policy_replacement_score:.3f}, "
            f"{decoder_summary}."
        )
        self._previous_code = latent_code
        self._previous_hidden_state = encoded.posterior.hidden_state
        self._previous_beta_binary = switch_decision.beta_binary
        track_codes = self._compute_track_codes(latent_code)
        return TemporalStep(
            controller_state=ControllerState(
                code=latent_code,
                code_dim=len(latent_code),
                switch_gate=effective_switch_gate,
                is_switching=is_switching,
                steps_since_switch=steps_since_switch,
                track_codes=track_codes,
            ),
            active_abstract_action=active_label,
            controller_params_hash=params_hash,
            description=description,
            action_family_version=self._parameter_store.action_family_version,
        )


class TemporalModule(RuntimeModule[TemporalAbstractionSnapshot]):
    slot_name = "temporal_abstraction"
    owner = "TemporalModule"
    value_type = TemporalAbstractionSnapshot
    dependencies = ("substrate", "memory", "reflection")
    default_wiring_level = WiringLevel.SHADOW

    def __init__(
        self,
        *,
        policy: TemporalPolicy | None = None,
        wiring_level: WiringLevel | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._policy = policy or FullLearnedTemporalPolicy()
        self._previous_snapshot: TemporalAbstractionSnapshot | None = None

    @property
    def policy(self) -> TemporalPolicy:
        return self._policy

    def export_runtime_state(self) -> MetacontrollerRuntimeState | None:
        return self._policy.export_runtime_state()

    async def process(
        self,
        upstream: Mapping[str, Snapshot[Any]],
    ) -> Snapshot[TemporalAbstractionSnapshot]:
        substrate_snapshot = upstream["substrate"]
        memory_snapshot = upstream["memory"]
        reflection_snapshot = upstream["reflection"]
        substrate_value = substrate_snapshot.value
        memory_value = memory_snapshot.value if isinstance(memory_snapshot.value, MemorySnapshot) else None
        reflection_value = (
            reflection_snapshot.value if isinstance(reflection_snapshot.value, ReflectionSnapshot) else None
        )
        if not isinstance(substrate_value, SubstrateSnapshot):
            step = PlaceholderTemporalPolicy().step(
                substrate_snapshot=SubstrateSnapshot(
                    model_id="runtime-placeholder",
                    is_frozen=True,
                    surface_kind=SurfaceKind.PLACEHOLDER,
                    token_logits=(),
                    feature_surface=(),
                    residual_activations=(),
                    residual_sequence=(),
                    unavailable_fields=(),
                    description="Runtime placeholder substrate value.",
                ),
                previous_snapshot=self._previous_snapshot,
                memory_snapshot=memory_value,
                reflection_snapshot=reflection_value,
            )
        else:
            step = self._policy.step(
                substrate_snapshot=substrate_value,
                previous_snapshot=self._previous_snapshot,
                memory_snapshot=memory_value,
                reflection_snapshot=reflection_value,
            )

        snapshot_value = TemporalAbstractionSnapshot(
            controller_state=step.controller_state,
            active_abstract_action=step.active_abstract_action,
            controller_params_hash=step.controller_params_hash,
            description=step.description,
            action_family_version=step.action_family_version,
        )
        self._previous_snapshot = snapshot_value
        return self.publish(snapshot_value)

    async def process_standalone(self, **kwargs: Any) -> Snapshot[TemporalAbstractionSnapshot]:
        substrate_snapshot = kwargs.get("substrate_snapshot")
        if not isinstance(substrate_snapshot, SubstrateSnapshot):
            raise TypeError("substrate_snapshot must be a SubstrateSnapshot.")
        memory_snapshot = kwargs.get("memory_snapshot")
        reflection_snapshot = kwargs.get("reflection_snapshot")
        step = self._policy.step(
            substrate_snapshot=substrate_snapshot,
            previous_snapshot=self._previous_snapshot,
            memory_snapshot=memory_snapshot if isinstance(memory_snapshot, MemorySnapshot) else None,
            reflection_snapshot=reflection_snapshot
            if isinstance(reflection_snapshot, ReflectionSnapshot)
            else None,
        )
        snapshot_value = TemporalAbstractionSnapshot(
            controller_state=step.controller_state,
            active_abstract_action=step.active_abstract_action,
            controller_params_hash=step.controller_params_hash,
            description=step.description,
            action_family_version=step.action_family_version,
        )
        self._previous_snapshot = snapshot_value
        return self.publish(snapshot_value)
