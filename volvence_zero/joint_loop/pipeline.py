"""Unified SSL→RL Training Pipeline (ETA Appendix B.3 + B.5).

Orchestrates the two-phase training protocol:
    Phase 1 (SSL): Eq.3 self-supervision discovers switching structure
    Phase 2 (RL):  Causal policy trained with binary gate on learned structure

The pipeline manages transition between phases based on convergence
criteria and maintains phase-aware checkpointing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time
from uuid import uuid4

from volvence_zero.internal_rl.sandbox import (
    CausalPolicyCheckpoint,
    DualTrackRollout,
    DualTrackOptimizationReport,
    InternalRLSandbox,
)
from volvence_zero.memory import MemoryStore, MemoryStoreCheckpoint, Track, build_default_memory_store
from volvence_zero.substrate import OpenWeightResidualRuntime, SubstrateRareHeavyCheckpoint, SubstrateSnapshot, TrainingTrace
from volvence_zero.temporal import (
    DualTrackRareHeavySnapshot,
    FullLearnedTemporalPolicy,
    MetacontrollerParameterStore,
    MetacontrollerParameterSnapshot,
    MetacontrollerSSLTrainer,
    SSLTrainingReport,
)


class TrainingPhase(str, Enum):
    SSL = "ssl"
    TRANSITION = "transition"
    RL = "rl"
    COMPLETE = "complete"


@dataclass(frozen=True)
class PipelineConfig:
    n_z: int = 16
    ssl_convergence_threshold: float = 0.15
    ssl_min_steps: int = 3
    ssl_max_steps: int = 20
    transition_max_steps: int = 2
    transition_agreement_threshold: float = 0.35
    transition_family_retention_threshold: float = 0.20
    rl_max_steps: int = 10
    rl_convergence_threshold: float = 0.05
    transition_kl_threshold: float = 1.0
    binary_gate_rl: bool = True
    rl_rollouts_per_step: int = 2


@dataclass(frozen=True)
class PhaseReport:
    owner_path: str
    phase: str
    step_index: int
    ssl_loss: float
    kl_loss: float
    posterior_drift: float
    noncausal_kl_tightening: float
    total_reward: float
    task_reward: float
    relationship_reward: float
    policy_objective: float
    convergence_metric: float
    rollout_batch_count: int
    transition_agreement: float
    switch_sparsity_retention: float
    family_reuse_retention: float
    takeover_ready: bool
    description: str


@dataclass(frozen=True)
class PipelineResult:
    config: PipelineConfig
    owner_path: str
    final_phase: str
    ssl_steps_completed: int
    rl_steps_completed: int
    phase_reports: tuple[PhaseReport, ...]
    final_ssl_loss: float
    final_total_reward: float
    transition_step: int
    substrate_training_mode: str
    substrate_checkpoint_present: bool
    training_evidence: "RareHeavyTrainingEvidence"
    description: str


@dataclass(frozen=True)
class RareHeavyTrainingEvidence:
    trace_count: int
    provided_substrate_batch_count: int
    resolved_rl_batch_count: int
    aligned_batch_count: int
    alignment_ratio: float
    mean_trace_step_count: float
    mean_substrate_sequence_length: float
    mean_substrate_residual_magnitude: float
    description: str


@dataclass(frozen=True)
class RareHeavyArtifact:
    artifact_id: str
    owner_path: str
    created_at_ms: int
    temporal_snapshot: MetacontrollerParameterSnapshot | DualTrackRareHeavySnapshot
    memory_checkpoint: MemoryStoreCheckpoint | None
    substrate_checkpoint: SubstrateRareHeavyCheckpoint | None
    transition_step: int
    final_ssl_loss: float
    final_total_reward: float
    description: str
    training_evidence: "RareHeavyTrainingEvidence | None" = None


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


class SSLRLTrainingPipeline:
    """Two-phase training orchestrator.

    Phase 1 runs Eq.3 SSL to discover switching structure in the encoder,
    switch unit, and decoder. Phase 2 freezes the learned structure and
    trains the causal z-policy via dual-track RL rollouts with binary gate.
    """

    owner_path = "offline-sslrl-pipeline"

    def __init__(
        self,
        *,
        config: PipelineConfig | None = None,
        policy: FullLearnedTemporalPolicy | None = None,
        memory_store: MemoryStore | None = None,
        residual_runtime: OpenWeightResidualRuntime | None = None,
    ) -> None:
        self._config = config or PipelineConfig()
        n_z = self._config.n_z
        if policy is not None:
            self._policy = policy
        else:
            store = MetacontrollerParameterStore(n_z=n_z)
            self._policy = FullLearnedTemporalPolicy(parameter_store=store)
        self._residual_runtime = residual_runtime
        self._sandbox = InternalRLSandbox(policy=self._policy, residual_runtime=residual_runtime)
        self._ssl_trainer = MetacontrollerSSLTrainer(n_z=n_z)
        self._memory_store = memory_store or build_default_memory_store(latent_dim=n_z)
        self._phase = TrainingPhase.SSL
        self._policy.parameter_store.set_learning_phase("ssl", structure_frozen=False)
        self._ssl_loss_history: list[float] = []
        self._ssl_kl_history: list[float] = []
        self._rl_reward_history: list[float] = []
        self._phase_reports: list[PhaseReport] = []
        self._transition_step = -1
        self._ssl_checkpoint: CausalPolicyCheckpoint | None = None
        self._substrate_checkpoint: SubstrateRareHeavyCheckpoint | None = None
        self._training_evidence = RareHeavyTrainingEvidence(
            trace_count=0,
            provided_substrate_batch_count=0,
            resolved_rl_batch_count=0,
            aligned_batch_count=0,
            alignment_ratio=0.0,
            mean_trace_step_count=0.0,
            mean_substrate_sequence_length=0.0,
            mean_substrate_residual_magnitude=0.0,
            description="No rare-heavy training evidence recorded yet.",
        )

    @property
    def phase(self) -> TrainingPhase:
        return self._phase

    @property
    def policy(self) -> FullLearnedTemporalPolicy:
        return self._policy

    def run_pipeline(
        self,
        *,
        traces: tuple[TrainingTrace, ...],
        substrate_steps_per_trace: tuple[tuple[SubstrateSnapshot, ...], ...] | None = None,
    ) -> PipelineResult:
        """Run the full two-phase pipeline over the provided traces.

        Each trace is used for one SSL step. After convergence,
        substrate_steps are used for RL rollouts. If substrate_steps_per_trace
        is None, synthetic substrates are derived from traces.
        """
        cfg = self._config
        ssl_steps = 0
        rl_steps = 0
        if not traces:
            raise ValueError("SSLRLTrainingPipeline requires at least one training trace.")
        rl_batches = self._resolve_rl_batches(
            traces=traces,
            substrate_steps_per_trace=substrate_steps_per_trace,
        )
        self._training_evidence = self._build_training_evidence(
            traces=traces,
            substrate_steps_per_trace=substrate_steps_per_trace,
            rl_batches=rl_batches,
        )

        while self._phase == TrainingPhase.SSL and ssl_steps < cfg.ssl_max_steps:
            trace_index = ssl_steps % len(traces)
            report = self._run_ssl_step(step_index=ssl_steps, trace=traces[trace_index])
            self._phase_reports.append(report)
            ssl_steps += 1
            if self._should_transition_to_rl(ssl_steps):
                self._phase = TrainingPhase.TRANSITION
                self._transition_step = ssl_steps - 1
                self._ssl_checkpoint = self._sandbox.create_checkpoint(
                    checkpoint_id=f"ssl-phase-{self._transition_step}",
                )

        if self._phase == TrainingPhase.SSL:
            self._transition_step = ssl_steps - 1
            self._phase = TrainingPhase.COMPLETE

        transition_steps = 0
        while self._phase == TrainingPhase.TRANSITION and transition_steps < cfg.transition_max_steps:
            rollout_batches = self._select_rollout_batches(
                rl_batches=rl_batches,
                step_index=transition_steps,
            )
            report = self._run_transition_step(
                step_index=transition_steps,
                rollout_batches=rollout_batches,
            )
            self._phase_reports.append(report)
            transition_steps += 1
            if report.takeover_ready:
                self._phase = TrainingPhase.RL
                break
        if self._phase == TrainingPhase.TRANSITION:
            self._phase = TrainingPhase.COMPLETE

        while self._phase == TrainingPhase.RL and rl_steps < cfg.rl_max_steps:
            rollout_batches = self._select_rollout_batches(
                rl_batches=rl_batches,
                step_index=rl_steps,
            )
            report = self._run_rl_step(step_index=rl_steps, rollout_batches=rollout_batches)
            self._phase_reports.append(report)
            rl_steps += 1
            if self._should_complete(rl_steps):
                self._phase = TrainingPhase.COMPLETE

        if self._residual_runtime is not None:
            offline_runtime = self._residual_runtime.clone_for_rare_heavy()
            self._substrate_checkpoint = offline_runtime.train_rare_heavy(
                traces=traces,
                substrate_steps_per_trace=rl_batches,
                checkpoint_id=f"{self.owner_path}:substrate",
            )
            if self._substrate_checkpoint.training_mode != "adapter-delta-v2":
                raise RuntimeError(
                    f"{type(self._residual_runtime).__name__} exported unsupported substrate training mode "
                    f"{self._substrate_checkpoint.training_mode!r}; adapter-delta-v2 is required."
                )

        final_ssl = self._ssl_loss_history[-1] if self._ssl_loss_history else 0.0
        final_reward = self._rl_reward_history[-1] if self._rl_reward_history else 0.0
        substrate_training_mode = (
            self._substrate_checkpoint.training_mode
            if self._substrate_checkpoint is not None
            else "not-run"
        )

        return PipelineResult(
            config=cfg,
            owner_path=self.owner_path,
            final_phase=self._phase.value,
            ssl_steps_completed=ssl_steps,
            rl_steps_completed=rl_steps,
            phase_reports=tuple(self._phase_reports),
            final_ssl_loss=final_ssl,
            final_total_reward=final_reward,
            transition_step=self._transition_step,
            substrate_training_mode=substrate_training_mode,
            substrate_checkpoint_present=self._substrate_checkpoint is not None,
            training_evidence=self._training_evidence,
            description=(
                f"Pipeline completed: ssl={ssl_steps}, rl={rl_steps}, "
                f"owner={self.owner_path}, phase={self._phase.value}, "
                f"ssl_loss={final_ssl:.3f}, reward={final_reward:.3f}, "
                f"transition_at={self._transition_step}, substrate={substrate_training_mode}, "
                f"alignment={self._training_evidence.alignment_ratio:.2f}."
            ),
        )

    def _run_ssl_step(self, *, step_index: int, trace: TrainingTrace) -> PhaseReport:
        self._policy.parameter_store.set_learning_phase("ssl", structure_frozen=False)
        report = self._ssl_trainer.optimize(policy=self._policy, trace=trace)
        self._ssl_loss_history.append(report.total_loss)
        self._ssl_kl_history.append(report.kl_loss)
        if report.m3_slow_momentum_signal:
            self._memory_store.observe_encoder_feedback(
                encoder_signal=report.m3_slow_momentum_signal,
                timestamp_ms=step_index + 1,
            )
        convergence = self._ssl_convergence_metric()
        return PhaseReport(
            owner_path=self.owner_path,
            phase="ssl",
            step_index=step_index,
            ssl_loss=report.total_loss,
            kl_loss=report.kl_loss,
            posterior_drift=report.posterior_drift,
            noncausal_kl_tightening=report.noncausal_kl_tightening,
            total_reward=0.0,
            task_reward=0.0,
            relationship_reward=0.0,
            policy_objective=0.0,
            convergence_metric=convergence,
            rollout_batch_count=0,
            transition_agreement=0.0,
            switch_sparsity_retention=0.0,
            family_reuse_retention=0.0,
            takeover_ready=False,
            description=(
                f"SSL step {step_index}: loss={report.total_loss:.3f}, "
                f"kl={report.kl_loss:.3f}, drift={report.posterior_drift:.3f}, "
                f"kl_tightening={report.noncausal_kl_tightening:.3f}, "
                f"convergence={convergence:.3f}."
            ),
        )

    def _run_transition_step(
        self,
        *,
        step_index: int,
        rollout_batches: tuple[tuple[SubstrateSnapshot, ...], ...],
    ) -> PhaseReport:
        self._policy.parameter_store.set_learning_phase("rl", structure_frozen=True)
        dual_rollouts = tuple(
            self._sandbox.rollout_dual_track(
                rollout_id=f"pipeline-transition-{step_index}-{batch_index}",
                substrate_steps=substrates,
            )
            for batch_index, substrates in enumerate(rollout_batches)
        )
        transitions = tuple(
            transition
            for rollout in dual_rollouts
            for track_rollout in (rollout.task_rollout, rollout.relationship_rollout)
            for transition in track_rollout.transitions
        )
        if transitions:
            transition_agreement = sum(
                1.0
                - (
                    sum(
                        abs(action - latent)
                        for action, latent in zip(
                            transition.policy_action,
                            transition.latent_code,
                            strict=True,
                        )
                    )
                    / max(len(transition.policy_action), 1)
                )
                for transition in transitions
            ) / len(transitions)
            switch_sparsity_retention = sum(
                1.0 - transition.controller_state.switch_gate
                for transition in transitions
            ) / len(transitions)
            family_reuse_retention = sum(
                float(transition.active_family_id not in {None, "unassigned"})
                for transition in transitions
            ) / len(transitions)
        else:
            transition_agreement = 0.0
            switch_sparsity_retention = 0.0
            family_reuse_retention = 0.0
        takeover_ready = (
            transition_agreement >= self._config.transition_agreement_threshold
            and family_reuse_retention >= self._config.transition_family_retention_threshold
        )
        return PhaseReport(
            owner_path=self.owner_path,
            phase="transition",
            step_index=step_index,
            ssl_loss=0.0,
            kl_loss=0.0,
            posterior_drift=0.0,
            noncausal_kl_tightening=0.0,
            total_reward=sum(
                rollout.task_rollout.total_reward + rollout.relationship_rollout.total_reward
                for rollout in dual_rollouts
            ),
            task_reward=sum(rollout.task_rollout.total_reward for rollout in dual_rollouts),
            relationship_reward=sum(rollout.relationship_rollout.total_reward for rollout in dual_rollouts),
            policy_objective=0.0,
            convergence_metric=transition_agreement,
            rollout_batch_count=len(rollout_batches),
            transition_agreement=transition_agreement,
            switch_sparsity_retention=switch_sparsity_retention,
            family_reuse_retention=family_reuse_retention,
            takeover_ready=takeover_ready,
            description=(
                f"Transition step {step_index}: agreement={transition_agreement:.3f}, "
                f"sparsity_retention={switch_sparsity_retention:.3f}, "
                f"family_retention={family_reuse_retention:.3f}, "
                f"ready={takeover_ready}."
            ),
        )

    def _run_rl_step(
        self,
        *,
        step_index: int,
        rollout_batches: tuple[tuple[SubstrateSnapshot, ...], ...],
    ) -> PhaseReport:
        self._policy.parameter_store.set_learning_phase("rl", structure_frozen=True)
        replacement_mode = "causal-binary" if self._config.binary_gate_rl else "causal"
        dual_rollouts = tuple(
            DualTrackRollout(
                task_rollout=self._sandbox.rollout(
                    rollout_id=f"pipeline-rl-{step_index}-{batch_index}:task",
                    substrate_steps=substrates,
                    track=Track.WORLD,
                    replacement_mode=replacement_mode,
                ),
                relationship_rollout=self._sandbox.rollout(
                    rollout_id=f"pipeline-rl-{step_index}-{batch_index}:relationship",
                    substrate_steps=substrates,
                    track=Track.SELF,
                    replacement_mode=replacement_mode,
                ),
                description=f"pipeline batch {batch_index}",
            )
            for batch_index, substrates in enumerate(rollout_batches)
        )
        opt_report = self._sandbox.optimize(dual_rollouts)
        if not isinstance(opt_report, DualTrackOptimizationReport):
            raise TypeError("Expected DualTrackOptimizationReport")
        total_reward = (
            sum(rollout.task_rollout.total_reward for rollout in dual_rollouts)
            + sum(rollout.relationship_rollout.total_reward for rollout in dual_rollouts)
        )
        self._rl_reward_history.append(total_reward)
        convergence = self._rl_convergence_metric()
        policy_objective = (
            opt_report.task_report.surrogate_objective
            + opt_report.relationship_report.surrogate_objective
        )
        return PhaseReport(
            owner_path=self.owner_path,
            phase="rl",
            step_index=step_index,
            ssl_loss=0.0,
            kl_loss=0.0,
            posterior_drift=0.0,
            noncausal_kl_tightening=0.0,
            total_reward=total_reward,
            task_reward=sum(rollout.task_rollout.total_reward for rollout in dual_rollouts),
            relationship_reward=sum(rollout.relationship_rollout.total_reward for rollout in dual_rollouts),
            policy_objective=policy_objective,
            convergence_metric=convergence,
            rollout_batch_count=len(rollout_batches),
            transition_agreement=0.0,
            switch_sparsity_retention=0.0,
            family_reuse_retention=0.0,
            takeover_ready=True,
            description=(
                f"RL step {step_index}: reward={total_reward:.3f}, "
                f"task={sum(rollout.task_rollout.total_reward for rollout in dual_rollouts):.3f}, "
                f"rel={sum(rollout.relationship_rollout.total_reward for rollout in dual_rollouts):.3f}, "
                f"objective={policy_objective:.3f}, "
                f"batch={len(rollout_batches)}, convergence={convergence:.3f}."
            ),
        )

    def _ssl_convergence_metric(self) -> float:
        if len(self._ssl_loss_history) < 2:
            return 1.0
        recent = self._ssl_loss_history[-3:] if len(self._ssl_loss_history) >= 3 else self._ssl_loss_history
        avg_loss = sum(recent) / len(recent)
        delta = abs(self._ssl_loss_history[-1] - self._ssl_loss_history[-2])
        return _clamp(avg_loss + delta)

    def _rl_convergence_metric(self) -> float:
        if len(self._rl_reward_history) < 2:
            return 1.0
        delta = abs(self._rl_reward_history[-1] - self._rl_reward_history[-2])
        return _clamp(delta)

    def _should_transition_to_rl(self, ssl_steps: int) -> bool:
        cfg = self._config
        if ssl_steps < cfg.ssl_min_steps:
            return False
        if ssl_steps >= cfg.ssl_max_steps:
            return True
        metric = self._ssl_convergence_metric()
        latest_kl = self._ssl_kl_history[-1] if self._ssl_kl_history else 0.0
        return metric < cfg.ssl_convergence_threshold and latest_kl <= cfg.transition_kl_threshold

    def _should_complete(self, rl_steps: int) -> bool:
        cfg = self._config
        if rl_steps >= cfg.rl_max_steps:
            return True
        if len(self._rl_reward_history) < 2:
            return False
        metric = self._rl_convergence_metric()
        return metric < cfg.rl_convergence_threshold

    def _resolve_rl_batches(
        self,
        *,
        traces: tuple[TrainingTrace, ...],
        substrate_steps_per_trace: tuple[tuple[SubstrateSnapshot, ...], ...] | None,
    ) -> tuple[tuple[SubstrateSnapshot, ...], ...]:
        if substrate_steps_per_trace:
            filtered = tuple(batch for batch in substrate_steps_per_trace if batch)
            if filtered:
                return filtered
        return tuple(self._substrates_from_trace(trace) for trace in traces)

    def _select_rollout_batches(
        self,
        *,
        rl_batches: tuple[tuple[SubstrateSnapshot, ...], ...],
        step_index: int,
    ) -> tuple[tuple[SubstrateSnapshot, ...], ...]:
        if not rl_batches:
            return ()
        batch_size = max(1, min(self._config.rl_rollouts_per_step, len(rl_batches)))
        start_index = (step_index * batch_size) % len(rl_batches)
        return tuple(
            rl_batches[(start_index + offset) % len(rl_batches)]
            for offset in range(batch_size)
        )

    def _build_training_evidence(
        self,
        *,
        traces: tuple[TrainingTrace, ...],
        substrate_steps_per_trace: tuple[tuple[SubstrateSnapshot, ...], ...] | None,
        rl_batches: tuple[tuple[SubstrateSnapshot, ...], ...],
    ) -> RareHeavyTrainingEvidence:
        trace_count = len(traces)
        provided_substrate_batch_count = len(substrate_steps_per_trace or ())
        resolved_rl_batch_count = len(rl_batches)
        aligned_batch_count = min(trace_count, resolved_rl_batch_count)
        alignment_ratio = aligned_batch_count / max(trace_count, 1)
        mean_trace_step_count = (
            sum(len(trace.steps) for trace in traces) / max(trace_count, 1)
            if traces
            else 0.0
        )
        flattened = tuple(snapshot for batch in rl_batches for snapshot in batch)
        mean_substrate_sequence_length = (
            sum(max(len(snapshot.residual_sequence), 1) for snapshot in flattened) / max(len(flattened), 1)
            if flattened
            else 0.0
        )
        residual_values = tuple(
            abs(value)
            for snapshot in flattened
            for activation in snapshot.residual_activations
            for value in activation.activation
        )
        mean_substrate_residual_magnitude = (
            sum(residual_values) / len(residual_values)
            if residual_values
            else 0.0
        )
        return RareHeavyTrainingEvidence(
            trace_count=trace_count,
            provided_substrate_batch_count=provided_substrate_batch_count,
            resolved_rl_batch_count=resolved_rl_batch_count,
            aligned_batch_count=aligned_batch_count,
            alignment_ratio=alignment_ratio,
            mean_trace_step_count=mean_trace_step_count,
            mean_substrate_sequence_length=mean_substrate_sequence_length,
            mean_substrate_residual_magnitude=mean_substrate_residual_magnitude,
            description=(
                f"Rare-heavy training evidence traces={trace_count} provided_batches={provided_substrate_batch_count} "
                f"resolved_batches={resolved_rl_batch_count} alignment={alignment_ratio:.2f} "
                f"mean_trace_steps={mean_trace_step_count:.2f}."
            ),
        )

    def _substrates_from_trace(self, trace: TrainingTrace) -> tuple[SubstrateSnapshot, ...]:
        from volvence_zero.substrate import SurfaceKind
        return tuple(
            SubstrateSnapshot(
                model_id=f"pipeline-trace:{trace.trace_id}",
                is_frozen=True,
                surface_kind=SurfaceKind.RESIDUAL_STREAM,
                token_logits=tuple(
                    min(sum(f.values) / max(len(f.values), 1), 1.0)
                    for f in step.feature_surface
                ),
                feature_surface=step.feature_surface,
                residual_activations=step.residual_activations,
                residual_sequence=(),
                unavailable_fields=(),
                description=f"Pipeline substrate for {trace.trace_id} step {step.step}.",
            )
            for step in trace.steps
        )

    def rollback_to_ssl_checkpoint(self) -> bool:
        """Rollback to the SSL phase checkpoint if RL went poorly."""
        if self._ssl_checkpoint is None:
            return False
        self._sandbox.restore_checkpoint(self._ssl_checkpoint)
        self._phase = TrainingPhase.RL
        self._rl_reward_history.clear()
        return True

    def export_rare_heavy_artifact(
        self,
        *,
        artifact_id: str | None = None,
        include_memory: bool = True,
        include_substrate: bool = True,
    ) -> RareHeavyArtifact:
        final_ssl = self._ssl_loss_history[-1] if self._ssl_loss_history else 0.0
        final_reward = self._rl_reward_history[-1] if self._rl_reward_history else 0.0
        return RareHeavyArtifact(
            artifact_id=artifact_id or str(uuid4()),
            owner_path=self.owner_path,
            created_at_ms=int(time.time() * 1000),
            temporal_snapshot=self._policy.export_rare_heavy_snapshot(),
            memory_checkpoint=self._memory_store.export_rare_heavy_state() if include_memory else None,
            substrate_checkpoint=(
                self._substrate_checkpoint
                if include_substrate
                else None
            ),
            transition_step=self._transition_step,
            final_ssl_loss=final_ssl,
            final_total_reward=final_reward,
            training_evidence=self._training_evidence,
            description=(
                f"Rare-heavy artifact exported from {self.owner_path} with phase={self._phase.value}, "
                f"transition_step={self._transition_step}, ssl_loss={final_ssl:.3f}, reward={final_reward:.3f}, "
                f"substrate={'yes' if self._substrate_checkpoint is not None and include_substrate else 'no'}"
                f"/{self._substrate_checkpoint.training_mode if self._substrate_checkpoint is not None and include_substrate else 'not-run'}."
            ),
        )
