"""Frozen joint-loop report, schedule, and import checkpoint contracts."""

from __future__ import annotations

from dataclasses import dataclass

from volvence_zero.application.runtime import ApplicationRareHeavyCheckpoint
from volvence_zero.evaluation import EvaluationScore, EvolutionJudgement
from volvence_zero.internal_rl import CausalPolicyCheckpoint
from volvence_zero.memory import MemoryStoreCheckpoint
from volvence_zero.substrate import SubstrateOnlineFastCheckpoint, SubstrateRareHeavyCheckpoint
from volvence_zero.temporal import MetacontrollerParameterSnapshot, MetacontrollerRuntimeState


@dataclass(frozen=True)
class DefaultContinualLearningSurface:
    surface_id: str
    active: bool
    owner_path: str
    memory_regime_writeback_applied: bool
    temporal_writeback_applied: bool
    regime_evidence_applied: bool
    substrate_live_mutation_applied: bool
    substrate_review_only: bool
    rare_heavy_review_recommended: bool
    applied_operations: tuple[str, ...]
    blocked_operations: tuple[str, ...]
    rollback_applied: bool
    evolution_decision: str
    evolution_category: str
    description: str


@dataclass(frozen=True)
class JointCycleReport:
    cycle_index: int
    acceptance_passed: bool
    ssl_prediction_loss: float
    ssl_kl_loss: float
    ssl_posterior_drift: float
    total_reward: float
    mean_transition_reward: float
    task_reward: float
    relationship_reward: float
    ssl_rollback_applied: bool
    policy_rollback_applied: bool
    rollback_reasons: tuple[str, ...]
    optimization_summary: str
    policy_objective: float
    kernel_score_count: int
    kernel_scores: tuple[EvaluationScore, ...]
    backend_name: str
    backend_fidelity: float
    applied_operations: tuple[str, ...]
    metacontroller_state: MetacontrollerRuntimeState | None
    cms_description: str
    evolution_judgement: EvolutionJudgement | None
    owner_path: str
    schedule_telemetry: tuple[tuple[str, int], ...]
    description: str
    policy_update_applied: bool = False
    policy_kl_divergence: float = 0.0
    policy_epochs_executed: int = 0
    rare_heavy_review_recommended: bool = False
    rl_batch_rollout_count: int = 1
    default_continual_learning_surface: DefaultContinualLearningSurface | None = None


@dataclass(frozen=True)
class JointLoopSchedule:
    ssl_interval: int = 1
    rl_interval: int = 3
    rl_batch_max_wait_turns: int = 2
    pe_full_cycle_threshold: float = 0.6
    pe_ssl_threshold: float = 0.18
    pe_substrate_online_fast_threshold: float = 0.18
    pe_rare_heavy_threshold: float = 1.2
    latent_continuation_threshold: float = 0.52


@dataclass(frozen=True)
class ScheduledJointLoopResult:
    turn_index: int
    schedule_action: str
    cycle_report: JointCycleReport | None
    kernel_scores: tuple[EvaluationScore, ...]
    ssl_prediction_loss: float
    ssl_kl_loss: float
    metacontroller_state: MetacontrollerRuntimeState | None
    cms_description: str
    owner_path: str
    schedule_telemetry: tuple[tuple[str, int], ...]
    description: str
    substrate_online_fast_due: bool = False
    rare_heavy_review_recommended: bool = False
    default_continual_learning_surface: DefaultContinualLearningSurface | None = None


@dataclass(frozen=True)
class RareHeavyImportCheckpoint:
    artifact_id: str
    world_policy_checkpoint: CausalPolicyCheckpoint
    self_policy_checkpoint: CausalPolicyCheckpoint
    world_temporal_snapshot: MetacontrollerParameterSnapshot
    self_temporal_snapshot: MetacontrollerParameterSnapshot
    memory_checkpoint: MemoryStoreCheckpoint
    substrate_checkpoint: SubstrateRareHeavyCheckpoint | None = None
    application_checkpoint: ApplicationRareHeavyCheckpoint | None = None


@dataclass(frozen=True)
class RareHeavyImportResult:
    artifact_id: str
    applied_operations: tuple[str, ...]
    checkpoint: RareHeavyImportCheckpoint
    description: str


@dataclass(frozen=True)
class OnlineFastImportCheckpoint:
    checkpoint_id: str
    substrate_checkpoint: SubstrateOnlineFastCheckpoint | None = None


@dataclass(frozen=True)
class OnlineFastImportResult:
    applied_operations: tuple[str, ...]
    checkpoint: OnlineFastImportCheckpoint
    description: str
