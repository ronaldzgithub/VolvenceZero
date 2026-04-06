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

from volvence_zero.internal_rl.sandbox import (
    CausalPolicyCheckpoint,
    DualTrackOptimizationReport,
    InternalRLSandbox,
)
from volvence_zero.memory import CMSMemoryCore, MemoryStore, Track
from volvence_zero.substrate import SubstrateSnapshot, TrainingTrace
from volvence_zero.temporal import (
    FullLearnedTemporalPolicy,
    MetacontrollerParameterStore,
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
    rl_max_steps: int = 10
    rl_convergence_threshold: float = 0.05
    transition_kl_threshold: float = 0.3
    binary_gate_rl: bool = True


@dataclass(frozen=True)
class PhaseReport:
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
    description: str


@dataclass(frozen=True)
class PipelineResult:
    config: PipelineConfig
    final_phase: str
    ssl_steps_completed: int
    rl_steps_completed: int
    phase_reports: tuple[PhaseReport, ...]
    final_ssl_loss: float
    final_total_reward: float
    transition_step: int
    description: str


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


class SSLRLTrainingPipeline:
    """Two-phase training orchestrator.

    Phase 1 runs Eq.3 SSL to discover switching structure in the encoder,
    switch unit, and decoder. Phase 2 freezes the learned structure and
    trains the causal z-policy via dual-track RL rollouts with binary gate.
    """

    def __init__(
        self,
        *,
        config: PipelineConfig | None = None,
        policy: FullLearnedTemporalPolicy | None = None,
        memory_store: MemoryStore | None = None,
    ) -> None:
        self._config = config or PipelineConfig()
        n_z = self._config.n_z
        if policy is not None:
            self._policy = policy
        else:
            store = MetacontrollerParameterStore(n_z=n_z)
            self._policy = FullLearnedTemporalPolicy(parameter_store=store)
        self._sandbox = InternalRLSandbox(policy=self._policy)
        self._ssl_trainer = MetacontrollerSSLTrainer(n_z=n_z)
        self._memory_store = memory_store or MemoryStore(learned_core=CMSMemoryCore(dim=n_z))
        self._phase = TrainingPhase.SSL
        self._ssl_loss_history: list[float] = []
        self._rl_reward_history: list[float] = []
        self._phase_reports: list[PhaseReport] = []
        self._transition_step = -1
        self._ssl_checkpoint: CausalPolicyCheckpoint | None = None

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

        for i, trace in enumerate(traces):
            if self._phase == TrainingPhase.COMPLETE:
                break

            if self._phase == TrainingPhase.SSL:
                report = self._run_ssl_step(step_index=i, trace=trace)
                self._phase_reports.append(report)
                ssl_steps += 1
                if self._should_transition_to_rl(ssl_steps):
                    self._phase = TrainingPhase.TRANSITION
                    self._transition_step = i
                    self._ssl_checkpoint = self._sandbox.create_checkpoint(
                        checkpoint_id=f"ssl-phase-{i}",
                    )
                    self._phase = TrainingPhase.RL

            elif self._phase == TrainingPhase.RL:
                substrates = (
                    substrate_steps_per_trace[i]
                    if substrate_steps_per_trace and i < len(substrate_steps_per_trace)
                    else self._substrates_from_trace(trace)
                )
                report = self._run_rl_step(step_index=i, substrates=substrates)
                self._phase_reports.append(report)
                rl_steps += 1
                if self._should_complete(rl_steps):
                    self._phase = TrainingPhase.COMPLETE

        if ssl_steps >= cfg.ssl_max_steps and self._phase == TrainingPhase.SSL:
            self._transition_step = ssl_steps - 1
            self._phase = TrainingPhase.COMPLETE

        final_ssl = self._ssl_loss_history[-1] if self._ssl_loss_history else 0.0
        final_reward = self._rl_reward_history[-1] if self._rl_reward_history else 0.0

        return PipelineResult(
            config=cfg,
            final_phase=self._phase.value,
            ssl_steps_completed=ssl_steps,
            rl_steps_completed=rl_steps,
            phase_reports=tuple(self._phase_reports),
            final_ssl_loss=final_ssl,
            final_total_reward=final_reward,
            transition_step=self._transition_step,
            description=(
                f"Pipeline completed: ssl={ssl_steps}, rl={rl_steps}, "
                f"phase={self._phase.value}, "
                f"ssl_loss={final_ssl:.3f}, reward={final_reward:.3f}, "
                f"transition_at={self._transition_step}."
            ),
        )

    def _run_ssl_step(self, *, step_index: int, trace: TrainingTrace) -> PhaseReport:
        report = self._ssl_trainer.optimize(policy=self._policy, trace=trace)
        self._ssl_loss_history.append(report.total_loss)
        if report.m3_slow_momentum_signal:
            self._memory_store.observe_encoder_feedback(
                encoder_signal=report.m3_slow_momentum_signal,
                timestamp_ms=step_index + 1,
            )
        convergence = self._ssl_convergence_metric()
        return PhaseReport(
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
            description=(
                f"SSL step {step_index}: loss={report.total_loss:.3f}, "
                f"kl={report.kl_loss:.3f}, drift={report.posterior_drift:.3f}, "
                f"kl_tightening={report.noncausal_kl_tightening:.3f}, "
                f"convergence={convergence:.3f}."
            ),
        )

    def _run_rl_step(
        self,
        *,
        step_index: int,
        substrates: tuple[SubstrateSnapshot, ...],
    ) -> PhaseReport:
        replacement_mode = "causal-binary" if self._config.binary_gate_rl else "causal"
        dual_rollout = self._sandbox.rollout_dual_track(
            rollout_id=f"pipeline-rl-{step_index}",
            substrate_steps=substrates,
        )
        opt_report = self._sandbox.optimize(dual_rollout)
        if not isinstance(opt_report, DualTrackOptimizationReport):
            raise TypeError("Expected DualTrackOptimizationReport")
        total_reward = (
            dual_rollout.task_rollout.total_reward
            + dual_rollout.relationship_rollout.total_reward
        )
        self._rl_reward_history.append(total_reward)
        convergence = self._rl_convergence_metric()
        policy_objective = (
            opt_report.task_report.surrogate_objective
            + opt_report.relationship_report.surrogate_objective
        )
        return PhaseReport(
            phase="rl",
            step_index=step_index,
            ssl_loss=0.0,
            kl_loss=0.0,
            posterior_drift=0.0,
            noncausal_kl_tightening=0.0,
            total_reward=total_reward,
            task_reward=dual_rollout.task_rollout.total_reward,
            relationship_reward=dual_rollout.relationship_rollout.total_reward,
            policy_objective=policy_objective,
            convergence_metric=convergence,
            description=(
                f"RL step {step_index}: reward={total_reward:.3f}, "
                f"task={dual_rollout.task_rollout.total_reward:.3f}, "
                f"rel={dual_rollout.relationship_rollout.total_reward:.3f}, "
                f"objective={policy_objective:.3f}, "
                f"convergence={convergence:.3f}."
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
        return metric < cfg.ssl_convergence_threshold

    def _should_complete(self, rl_steps: int) -> bool:
        cfg = self._config
        if rl_steps >= cfg.rl_max_steps:
            return True
        if len(self._rl_reward_history) < 2:
            return False
        metric = self._rl_convergence_metric()
        return metric < cfg.rl_convergence_threshold

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
