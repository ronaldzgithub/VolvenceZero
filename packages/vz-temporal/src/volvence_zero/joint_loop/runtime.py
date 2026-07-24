"""ETA/NL joint learning loop runtime.

Pure contracts, cadence scheduling helpers, and artifact import/rollback
helpers live in sibling modules. This module keeps the online runtime core
and re-exports the historic public API from ``volvence_zero.joint_loop.runtime``.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import math
from typing import Any

from volvence_zero.canonical_json import (
    CanonicalJsonError,
    canonical_json_bytes,
    typed_from_json,
    typed_to_json,
)
from volvence_zero.credit.gate import (
    CreditModule,
    CreditSnapshot,
    GateDecision,
    ModificationGate,
    SelfModificationRecord,
    derive_delayed_attribution_credit_records,
    derive_metacontroller_credit_records,
    derive_runtime_adaptation_audit_records,
    extend_credit_snapshot,
    extract_abstract_action_credit_bonus,
    has_blocking_writeback,
)
from volvence_zero.dual_track import DualTrackModule
from volvence_zero.environment import EnvironmentOutcome
from volvence_zero.evaluation import (
    CrossSessionBenchmarkSuite,
    EvaluationBackbone,
    EvaluationReport,
    EvolutionDecision,
    EvolutionJudgement,
    JudgementCategory,
    EvaluationModule,
    EvaluationSnapshot,
)
from volvence_zero.internal_rl import (
    DualTrackOptimizationReport,
    DualTrackRollout,
    InternalRLSandbox,
    OptimizationReport,
    PolicyOptimizationResult,
    ZRollout,
    derive_abstract_action_credit,
)
from volvence_zero.joint_loop.artifact_imports import _JointLoopArtifactImportMixin
from volvence_zero.joint_loop.contracts import (
    DefaultContinualLearningSurface,
    JointCycleReport,
    JointLoopSchedule,
    OnlineFastImportCheckpoint,
    OnlineFastImportResult,
    RareHeavyImportCheckpoint,
    RareHeavyImportResult,
    RuntimeReplayReport,
    ScheduledJointLoopResult,
)
from volvence_zero.joint_loop.pipeline import RareHeavyArtifact
from volvence_zero.joint_loop.scheduling import _JointLoopSchedulingMixin
from volvence_zero.memory import MemoryModule
from volvence_zero.memory import MemoryStore, Track, build_default_memory_store
from volvence_zero.owner_hydration import (
    HydrationOwnerMismatchError,
    HydrationPayloadInvalidError,
    HydrationVersionMismatchError,
    OwnerPersistenceSnapshot,
)
from volvence_zero.prediction import PredictionErrorSnapshot
from volvence_zero.reflection import ReflectionEngine, ReflectionModule, WritebackMode
from volvence_zero.regime import RegimeModule
from volvence_zero.runtime import EventRecorder, SlotRegistry, Snapshot, WiringLevel, propagate
from volvence_zero.runtime.kernel import stable_value_hash
from volvence_zero.substrate import (
    OpenWeightResidualRuntime,
    ResidualSequenceStep,
    SimulatedResidualSubstrateAdapter,
    SubstrateModule,
    SubstrateSnapshot,
    SurfaceKind,
    TraceStep,
    TrainingTrace,
)
from volvence_zero.temporal import (
    FullLearnedTemporalPolicy,
    MetacontrollerParameterStore,
    MetacontrollerSSLTrainer,
    MetacontrollerRuntimeState,
    TemporalAbstractionSnapshot,
    TemporalAggregateModule,
    TrackTemporalConsolidationModule,
    TrackTemporalModule,
    build_temporal_runtime_state_aggregate,
    clone_full_learned_temporal_policy,
)


_JOINT_LEARNING_OWNER_NAME = "joint_loop.learning"
_JOINT_LEARNING_SCHEMA_VERSION = 2


class ETANLJointLoop(_JointLoopSchedulingMixin, _JointLoopArtifactImportMixin):
    """SSL-RL joint training loop over the stage-two building blocks.

    State ownership and write contract (read before changing ``run_cycle``):

    * ``_memory_store``, ``_evaluation_backbone``, ``_world_policy`` and
      ``_self_policy`` are by construction **shared** with the runtime main
      chain (see ``AgentSessionRunner.__init__``). This is intentional
      online-adaptation: joint-loop training updates must be visible to the
      next serving turn. ``_regime_module`` is joint-loop private.
    * Within a turn, the runtime main chain writes first (through
      ``propagate(modules)`` and the final-wiring session-post path).
      ``run_cycle`` is invoked afterwards. The two paths are therefore
      serialised inside a turn, not concurrent.
    * The only place ``run_cycle`` is permitted to mutate the shared
      owners is the clearly marked TRAINING WRITEBACK PHASE below. If a
      future change adds owner-state mutation outside that block, the
      joint-loop becomes a second uncontrolled orchestrator and violates
      R8. Keep the contract test
      ``tests/test_phase2_eta_nl.py::test_joint_loop_shares_owner_instances_with_runtime``
      green.

    Handoffs that DO NOT go through the training writeback phase:
    ``apply_rare_heavy_artifact`` / ``rollback_rare_heavy_import`` /
    ``apply_online_fast_substrate_checkpoint`` are bounded checkpoint
    transfers gated by ``live_substrate_mutation`` and audited via
    ``SelfModificationRecord``; they are intentionally outside the
    per-turn training writeback phase.
    """

    owner_path = "online-joint-loop"

    def __init__(
        self,
        *,
        policy: FullLearnedTemporalPolicy | None = None,
        world_policy: FullLearnedTemporalPolicy | None = None,
        self_policy: FullLearnedTemporalPolicy | None = None,
        memory_store: MemoryStore | None = None,
        residual_runtime: OpenWeightResidualRuntime | None = None,
        evaluation_backbone: EvaluationBackbone | None = None,
        primary_prediction_error_dominance_enabled: bool = True,
        rl_batch_accumulation_size: int = 1,
        ssl_backend: WiringLevel = WiringLevel.DISABLED,
        internal_rl_backend: WiringLevel = WiringLevel.DISABLED,
        internal_rl_runtime_replay: WiringLevel = WiringLevel.DISABLED,
        runtime_replay_prediction_error_enabled: bool = True,
        runtime_replay_segment_credit: WiringLevel = WiringLevel.DISABLED,
        runtime_replay_segment_max_steps: int = 24,
        prediction_error_temporal_switch: WiringLevel = WiringLevel.DISABLED,
        prediction_error_temporal_switch_strength: float = 0.35,
        prediction_error_temporal_switch_floor: float = 0.5,
    ) -> None:
        self._world_policy = world_policy or policy or FullLearnedTemporalPolicy()
        self._self_policy = self_policy or FullLearnedTemporalPolicy(
            parameter_store=MetacontrollerParameterStore(n_z=self._world_policy.parameter_store.n_z)
        )
        if self_policy is None:
            self._self_policy = clone_full_learned_temporal_policy(self._world_policy)
        world_latent_dim = self._world_policy.parameter_store.n_z
        self_latent_dim = self._self_policy.parameter_store.n_z
        if self_latent_dim != world_latent_dim:
            raise ValueError(
                f"ETANLJointLoop requires aligned world/self latent dims, got {world_latent_dim} and {self_latent_dim}."
            )
        self._world_sandbox = InternalRLSandbox(
            policy=self._world_policy, residual_runtime=residual_runtime, rl_backend=internal_rl_backend
        )
        self._self_sandbox = InternalRLSandbox(
            policy=self._self_policy, residual_runtime=residual_runtime, rl_backend=internal_rl_backend
        )
        self._residual_runtime = residual_runtime
        self._ssl_trainer = MetacontrollerSSLTrainer(n_z=world_latent_dim, ssl_backend=ssl_backend)
        default_latent_dim = world_latent_dim
        self._memory_store = memory_store or build_default_memory_store(latent_dim=default_latent_dim)
        self._evaluation_backbone = evaluation_backbone or EvaluationBackbone()
        self._regime_module = RegimeModule(wiring_level=WiringLevel.ACTIVE)
        self._previous_total_reward: float | None = None
        self._previous_metacontroller_state: MetacontrollerRuntimeState | None = None
        self._previous_family_signals: dict[str, float] = {}
        self._previous_credit_snapshot: CreditSnapshot | None = None
        self._external_learning_signals: dict[str, float] = {}
        self._prediction_error_temporal_switch = (
            prediction_error_temporal_switch
        )
        if not 0.0 <= prediction_error_temporal_switch_strength <= 1.0:
            raise ValueError(
                "prediction_error_temporal_switch_strength must be within "
                "[0, 1]"
            )
        if prediction_error_temporal_switch_floor < 0.0:
            raise ValueError(
                "prediction_error_temporal_switch_floor must be >= 0"
            )
        self._prediction_error_temporal_switch_strength = float(
            prediction_error_temporal_switch_strength
        )
        self._prediction_error_temporal_switch_floor = float(
            prediction_error_temporal_switch_floor
        )
        self._primary_prediction_error_dominance_enabled = primary_prediction_error_dominance_enabled
        self._last_schedule_action = "evidence-only"
        self._last_learning_turn_index = 0
        self._rl_batch_accumulation_size = max(1, rl_batch_accumulation_size)
        self._pending_task_rollouts: list[ZRollout] = []
        self._pending_relationship_rollouts: list[ZRollout] = []
        self._open_task_segment_rollouts: list[ZRollout] = []
        self._open_relationship_segment_rollouts: list[ZRollout] = []
        self._internal_rl_runtime_replay = internal_rl_runtime_replay
        self._runtime_replay_segment_credit = runtime_replay_segment_credit
        if runtime_replay_segment_max_steps < 1:
            raise ValueError(
                "runtime_replay_segment_max_steps must be >= 1, "
                f"got {runtime_replay_segment_max_steps!r}"
            )
        self._runtime_replay_segment_max_steps = (
            runtime_replay_segment_max_steps
        )
        self._runtime_replay_prediction_error_enabled = bool(
            runtime_replay_prediction_error_enabled
        )
        self._runtime_replay_captured_count = 0
        self._runtime_replay_settled_count = 0
        self._runtime_replay_transition_count = 0
        self._runtime_replay_lineage_match_count = 0
        self._runtime_replay_drop_reasons: list[str] = []
        self._runtime_segment_closed_count = 0
        self._runtime_longest_segment_length = 0
        self._runtime_last_segment_close_reason = ""
        self._runtime_segment_close_reason_counts: dict[str, int] = {}
        # T3 (#88): learned SSL->RL schedule gate, report-only SHADOW.
        # The rule cascade stays the live writer; the learner's shadow
        # recommendation is published in schedule telemetry and settled
        # against realized SSL improvement so promotion evidence can be
        # collected without touching the live decision.
        from volvence_zero.joint_loop.gate_learner import ScheduleGateLearner

        self._schedule_gate_learner = ScheduleGateLearner()
        self._last_ssl_prediction_loss: float | None = None
        self.set_primary_prediction_error_dominance_enabled(primary_prediction_error_dominance_enabled)

    def set_external_learning_signals(self, signals: dict[str, float]) -> None:
        self._external_learning_signals = dict(signals)
        if self._prediction_error_temporal_switch is WiringLevel.DISABLED:
            return
        magnitude = abs(float(signals.get("prediction_error_magnitude", 0.0)))
        excess = max(
            0.0,
            magnitude - self._prediction_error_temporal_switch_floor,
        )
        switch_pressure = min(
            0.18,
            self._prediction_error_temporal_switch_strength
            * math.tanh(excess),
        )
        applied = (
            self._prediction_error_temporal_switch is WiringLevel.ACTIVE
        )
        for policy in (self._world_policy, self._self_policy):
            policy.parameter_store.record_prediction_error_switch_pressure(
                switch_pressure if applied else 0.0
            )

    @property
    def internal_rl_runtime_replay(self) -> WiringLevel:
        return self._internal_rl_runtime_replay

    @property
    def runtime_replay_segment_credit(self) -> WiringLevel:
        return self._runtime_replay_segment_credit

    @property
    def latest_runtime_replay_report(self) -> RuntimeReplayReport:
        pending_capture_count = sum(
            int(sandbox.runtime_replay_checkpoint.pending_capture is not None)
            for sandbox in (self._world_sandbox, self._self_sandbox)
        )
        staged_rollout_count = (
            len(self._pending_task_rollouts)
            + len(self._pending_relationship_rollouts)
            if self._internal_rl_runtime_replay is WiringLevel.ACTIVE
            else 0
        )
        source = (
            "synthetic"
            if self._internal_rl_runtime_replay is WiringLevel.DISABLED
            else "runtime-replay"
        )
        return RuntimeReplayReport(
            wiring_level=self._internal_rl_runtime_replay.value,
            transition_source=source,
            captured_count=self._runtime_replay_captured_count,
            settled_count=self._runtime_replay_settled_count,
            transition_count=self._runtime_replay_transition_count,
            lineage_match_count=self._runtime_replay_lineage_match_count,
            pending_capture_count=pending_capture_count,
            staged_rollout_count=staged_rollout_count,
            drop_reasons=tuple(self._runtime_replay_drop_reasons[-20:]),
            description=(
                f"Internal-RL transition source={source} "
                f"wiring={self._internal_rl_runtime_replay.value} "
                f"captured={self._runtime_replay_captured_count} "
                f"settled={self._runtime_replay_settled_count} "
                f"lineage_matches={self._runtime_replay_lineage_match_count} "
                f"drops={len(self._runtime_replay_drop_reasons)} "
                f"segment_credit={self._runtime_replay_segment_credit.value} "
                f"closed_segments={self._runtime_segment_closed_count}."
            ),
            segment_credit_wiring=self._runtime_replay_segment_credit.value,
            open_segment_transition_count=len(
                self._open_task_segment_rollouts
            ),
            closed_segment_count=self._runtime_segment_closed_count,
            longest_segment_length=self._runtime_longest_segment_length,
            last_segment_close_reason=(
                self._runtime_last_segment_close_reason
            ),
            segment_close_reason_counts=tuple(
                sorted(self._runtime_segment_close_reason_counts.items())
            ),
        )

    def observe_runtime_transition(
        self,
        *,
        turn_index: int,
        substrate_snapshot: SubstrateSnapshot,
        world_temporal_snapshot: TemporalAbstractionSnapshot,
        self_temporal_snapshot: TemporalAbstractionSnapshot,
        world_runtime_state: MetacontrollerRuntimeState,
        self_runtime_state: MetacontrollerRuntimeState,
        environment_outcome: EnvironmentOutcome | None,
        prediction_error_snapshot: PredictionErrorSnapshot,
        credit_snapshot: CreditSnapshot,
    ) -> RuntimeReplayReport:
        """Owner entry point for post-environment runtime evidence."""

        if self._internal_rl_runtime_replay is WiringLevel.DISABLED:
            return self.latest_runtime_replay_report
        prediction_id = prediction_error_snapshot.next_prediction.prediction_id
        if not prediction_id:
            raise RuntimeError(
                "runtime replay requires prediction_error.next_prediction "
                "to carry the PE-owner prediction_id"
            )
        world_settlement, _ = self._world_sandbox.observe_runtime_transition(
            turn_index=turn_index,
            track=Track.WORLD,
            prediction_id=prediction_id,
            substrate_snapshot=substrate_snapshot,
            temporal_snapshot=world_temporal_snapshot,
            runtime_state=world_runtime_state,
            environment_outcome=environment_outcome,
            prediction_error_snapshot=prediction_error_snapshot,
            credit_snapshot=credit_snapshot,
            prediction_error_reward_enabled=(
                self._runtime_replay_prediction_error_enabled
            ),
        )
        self_settlement, _ = self._self_sandbox.observe_runtime_transition(
            turn_index=turn_index,
            track=Track.SELF,
            prediction_id=prediction_id,
            substrate_snapshot=substrate_snapshot,
            temporal_snapshot=self_temporal_snapshot,
            runtime_state=self_runtime_state,
            environment_outcome=environment_outcome,
            prediction_error_snapshot=prediction_error_snapshot,
            credit_snapshot=credit_snapshot,
            prediction_error_reward_enabled=(
                self._runtime_replay_prediction_error_enabled
            ),
        )
        self._runtime_replay_captured_count += 2
        for track_name, settlement in (
            ("world", world_settlement),
            ("self", self_settlement),
        ):
            if settlement.rollout is not None:
                self._runtime_replay_settled_count += 1
                self._runtime_replay_transition_count += len(
                    settlement.rollout.transitions
                )
                self._runtime_replay_lineage_match_count += int(
                    settlement.lineage_matched
                )
            elif settlement.drop_reason not in {"", "no-pending-capture"}:
                self._runtime_replay_drop_reasons.append(
                    f"{track_name}:{settlement.drop_reason}"
                )
                if len(self._runtime_replay_drop_reasons) > 20:
                    del self._runtime_replay_drop_reasons[:-20]
        if self._internal_rl_runtime_replay is WiringLevel.ACTIVE:
            if (
                world_settlement.rollout is None
            ) != (
                self_settlement.rollout is None
            ):
                raise RuntimeError(
                    "runtime replay settlement diverged between world and self tracks"
                )
            if (
                world_settlement.rollout is not None
                and self_settlement.rollout is not None
            ):
                if (
                    self._runtime_replay_segment_credit
                    is WiringLevel.ACTIVE
                ):
                    self._stage_runtime_segment_pair(
                        world_rollout=world_settlement.rollout,
                        self_rollout=self_settlement.rollout,
                    )
                else:
                    self._pending_task_rollouts.append(
                        world_settlement.rollout
                    )
                    self._pending_relationship_rollouts.append(
                        self_settlement.rollout
                    )
            if len(self._pending_task_rollouts) != len(
                self._pending_relationship_rollouts
            ):
                raise RuntimeError(
                    "runtime replay staging diverged between world and self tracks"
                )
        return self.latest_runtime_replay_report

    def _stage_runtime_segment_pair(
        self,
        *,
        world_rollout: ZRollout,
        self_rollout: ZRollout,
    ) -> None:
        if (
            len(world_rollout.transitions) != 1
            or len(self_rollout.transitions) != 1
        ):
            raise ValueError(
                "runtime segment staging requires one settled transition "
                "per track"
            )
        world_transition = world_rollout.transitions[0]
        self_transition = self_rollout.transitions[0]
        # World and self metacontrollers switch independently, so a segment
        # boundary is a beta_t switch on either track (matching the OR
        # semantics milestone/terminal closure already uses below). Segments
        # stay pairwise aligned; a divergent switch only shortens them.
        boundary = (
            world_transition.controller_state.is_switching
            or self_transition.controller_state.is_switching
        )
        if boundary and self._open_task_segment_rollouts:
            self._close_runtime_segment(reason="beta-switch")
        self._open_task_segment_rollouts.append(world_rollout)
        self._open_relationship_segment_rollouts.append(self_rollout)
        close_for_milestone = (
            world_transition.runtime_terminal
            or world_transition.runtime_milestone
            or self_transition.runtime_terminal
            or self_transition.runtime_milestone
        )
        close_for_bound = (
            len(self._open_task_segment_rollouts)
            >= self._runtime_replay_segment_max_steps
        )
        if close_for_milestone:
            self._close_runtime_segment(reason="environment-milestone")
        elif close_for_bound:
            self._close_runtime_segment(reason="bounded-horizon")

    def _close_runtime_segment(self, *, reason: str) -> None:
        if not self._open_task_segment_rollouts:
            return
        if len(self._open_task_segment_rollouts) != len(
            self._open_relationship_segment_rollouts
        ):
            raise RuntimeError(
                "open runtime segment diverged between world and self tracks"
            )
        segment_length = len(self._open_task_segment_rollouts)
        world_first = self._open_task_segment_rollouts[0].transitions[0]
        world_last = self._open_task_segment_rollouts[-1].transitions[-1]
        self_first = self._open_relationship_segment_rollouts[0].transitions[0]
        self_last = self._open_relationship_segment_rollouts[-1].transitions[-1]
        task_segment = self._merge_runtime_rollouts(
            rollout_id=(
                "runtime-segment:world:"
                f"{world_first.runtime_turn_index}-{world_last.runtime_turn_index}"
            ),
            track=Track.WORLD,
            rollouts=tuple(self._open_task_segment_rollouts),
        )
        relationship_segment = self._merge_runtime_rollouts(
            rollout_id=(
                "runtime-segment:self:"
                f"{self_first.runtime_turn_index}-{self_last.runtime_turn_index}"
            ),
            track=Track.SELF,
            rollouts=tuple(self._open_relationship_segment_rollouts),
        )
        self._pending_task_rollouts.append(
            replace(
                task_segment,
                description=(
                    f"Closed runtime beta_t segment ({reason}) with "
                    f"{segment_length} real transitions."
                ),
            )
        )
        self._pending_relationship_rollouts.append(
            replace(
                relationship_segment,
                description=(
                    f"Closed runtime beta_t segment ({reason}) with "
                    f"{segment_length} real transitions."
                ),
            )
        )
        self._runtime_segment_closed_count += 1
        self._runtime_last_segment_close_reason = reason
        self._runtime_segment_close_reason_counts[reason] = (
            self._runtime_segment_close_reason_counts.get(reason, 0) + 1
        )
        self._runtime_longest_segment_length = max(
            self._runtime_longest_segment_length,
            segment_length,
        )
        self._open_task_segment_rollouts.clear()
        self._open_relationship_segment_rollouts.clear()

    @property
    def schedule_gate_learner(self):
        """Read-only access to the SHADOW schedule gate (evidence collection)."""

        return self._schedule_gate_learner

    def _settle_schedule_gate(self, *, current_ssl_prediction_loss: float) -> None:
        """Settle the learned schedule gate against realized SSL improvement.

        Positive gain = the learning turn reduced prediction loss relative
        to the previous learning turn. First learning turn only seeds the
        baseline.
        """

        previous = self._last_ssl_prediction_loss
        self._last_ssl_prediction_loss = current_ssl_prediction_loss
        if previous is None:
            return
        self._schedule_gate_learner.observe_realized_outcome(
            realized_gain=previous - current_ssl_prediction_loss
        )

    @property
    def primary_prediction_error_dominance_enabled(self) -> bool:
        return self._primary_prediction_error_dominance_enabled

    def set_primary_prediction_error_dominance_enabled(self, enabled: bool) -> None:
        self._primary_prediction_error_dominance_enabled = enabled
        self._world_sandbox._env.set_primary_prediction_error_enabled(enabled)
        self._self_sandbox._env.set_primary_prediction_error_enabled(enabled)


    def _apply_temporal_reflection_writeback(
        self,
        *,
        temporal_module: TemporalModule,
        reflection_snapshot,
        credit_snapshot: CreditSnapshot,
        timestamp_ms: int,
        apply_enabled: bool,
    ) -> tuple[tuple[str, ...], tuple[SelfModificationRecord, ...]]:
        temporal_prior_update = reflection_snapshot.policy_consolidation.temporal_prior_update
        if temporal_prior_update is None:
            return ((), ())
        target_groups = temporal_prior_update.target_groups or ("base-weights",)
        before_hash = stable_value_hash(temporal_module.policy.export_parameters())
        if not apply_enabled:
            return (
                ("temporal-prior:writeback-mode-not-apply",),
                tuple(
                    SelfModificationRecord(
                        target=f"{temporal_prior_update.target}.{group}",
                        gate=ModificationGate.BACKGROUND,
                        decision=GateDecision.BLOCK,
                        old_value_hash=before_hash,
                        new_value_hash=before_hash,
                        justification="Joint loop skipped reflection-to-temporal writeback because apply_writeback is disabled.",
                        timestamp_ms=timestamp_ms,
                        is_reversible=True,
                    )
                    for group in target_groups
                ),
            )
        blocked_groups = tuple(
            group
            for group in target_groups
            if has_blocking_writeback(
                credit_snapshot,
                target_prefix=f"{temporal_prior_update.target}.{group}",
            )
        )
        allowed_groups = tuple(group for group in target_groups if group not in blocked_groups)
        if not allowed_groups:
            return (
                ("temporal-prior:credit-gate-block",),
                tuple(
                    SelfModificationRecord(
                        target=f"{temporal_prior_update.target}.{group}",
                        gate=ModificationGate.BACKGROUND,
                        decision=GateDecision.BLOCK,
                        old_value_hash=before_hash,
                        new_value_hash=before_hash,
                        justification="Joint loop blocked reflection-to-temporal writeback via target-specific credit gate.",
                        timestamp_ms=timestamp_ms,
                        is_reversible=True,
                    )
                    for group in blocked_groups
                ),
            )
        applied_operations = temporal_module.policy.apply_reflection_prior_update(
            update=temporal_prior_update,
            allowed_target_groups=allowed_groups,
        )
        if blocked_groups:
            applied_operations = applied_operations + ("temporal-prior:partial-credit-gate-block",)
        after_hash = stable_value_hash(temporal_module.policy.export_parameters())
        return (
            applied_operations,
            tuple(
                [
                    SelfModificationRecord(
                        target=f"{temporal_prior_update.target}.{group}",
                        gate=ModificationGate.BACKGROUND,
                        decision=GateDecision.ALLOW,
                        old_value_hash=before_hash,
                        new_value_hash=after_hash,
                        justification=temporal_prior_update.description,
                        timestamp_ms=timestamp_ms,
                        is_reversible=True,
                    )
                    for group in allowed_groups
                ]
                + [
                    SelfModificationRecord(
                        target=f"{temporal_prior_update.target}.{group}",
                        gate=ModificationGate.BACKGROUND,
                        decision=GateDecision.BLOCK,
                        old_value_hash=before_hash,
                        new_value_hash=before_hash,
                        justification="Joint loop blocked reflection-to-temporal writeback via target-specific credit gate.",
                        timestamp_ms=timestamp_ms,
                        is_reversible=True,
                    )
                    for group in blocked_groups
                ]
            ),
        )

    @property
    def memory_store(self) -> MemoryStore:
        return self._memory_store

    @property
    def temporal_policy(self) -> FullLearnedTemporalPolicy:
        return self._world_policy

    @property
    def world_temporal_policy(self) -> FullLearnedTemporalPolicy:
        return self._world_policy

    @property
    def self_temporal_policy(self) -> FullLearnedTemporalPolicy:
        return self._self_policy

    @property
    def residual_runtime(self) -> OpenWeightResidualRuntime | None:
        return self._residual_runtime

    def create_learning_checkpoint(
        self,
        *,
        checkpoint_id: str,
        include_runtime_replay: bool = True,
    ) -> RareHeavyImportCheckpoint:
        """Export owner-held online learning state for fair held-out branches."""

        return RareHeavyImportCheckpoint(
            artifact_id=checkpoint_id,
            world_policy_checkpoint=self._world_sandbox.create_checkpoint(
                checkpoint_id=f"{checkpoint_id}:world-policy",
                include_runtime_replay=include_runtime_replay,
            ),
            self_policy_checkpoint=self._self_sandbox.create_checkpoint(
                checkpoint_id=f"{checkpoint_id}:self-policy",
                include_runtime_replay=include_runtime_replay,
            ),
            world_temporal_snapshot=self._world_policy.export_rare_heavy_snapshot(),
            self_temporal_snapshot=self._self_policy.export_rare_heavy_snapshot(),
            memory_checkpoint=self._memory_store.create_checkpoint(
                checkpoint_id=f"{checkpoint_id}:memory"
            ),
            pending_task_rollouts=(
                tuple(self._pending_task_rollouts)
                if include_runtime_replay
                else ()
            ),
            pending_relationship_rollouts=tuple(
                self._pending_relationship_rollouts
            )
            if include_runtime_replay
            else (),
            runtime_replay_report=(
                self.latest_runtime_replay_report
                if include_runtime_replay
                else None
            ),
            schedule_gate_state=self._schedule_gate_learner.export_state(),
            open_task_segment_rollouts=(
                tuple(self._open_task_segment_rollouts)
                if include_runtime_replay
                else ()
            ),
            open_relationship_segment_rollouts=(
                tuple(self._open_relationship_segment_rollouts)
                if include_runtime_replay
                else ()
            ),
            runtime_segment_closed_count=(
                self._runtime_segment_closed_count
                if include_runtime_replay
                else 0
            ),
            runtime_longest_segment_length=(
                self._runtime_longest_segment_length
                if include_runtime_replay
                else 0
            ),
        )

    def export_learning_persistence_snapshot(
        self,
        *,
        include_runtime_replay: bool = False,
    ) -> OwnerPersistenceSnapshot:
        """Publish strict JSON state for out-of-turn archive orchestration."""

        checkpoint = self.create_learning_checkpoint(
            checkpoint_id="joint-loop:learning-archive",
            include_runtime_replay=include_runtime_replay,
        )
        payload = typed_to_json(checkpoint, RareHeavyImportCheckpoint)
        if not isinstance(payload, dict):
            raise RuntimeError("joint-loop checkpoint codec did not produce an object")
        return OwnerPersistenceSnapshot(
            owner_name=_JOINT_LEARNING_OWNER_NAME,
            schema_version=_JOINT_LEARNING_SCHEMA_VERSION,
            payload=payload,
            description=(
                "Joint-loop learned state "
                f"runtime_replay={'included' if include_runtime_replay else 'excluded'}"
            ),
        )

    def hydrate_learning_from_persistence(
        self,
        snapshot: OwnerPersistenceSnapshot,
    ) -> tuple[str, ...]:
        """Validate and restore an owner-authored learning archive part."""

        checkpoint = self._decode_learning_persistence_snapshot(snapshot)
        return self.restore_learning_checkpoint(checkpoint)

    def learning_persistence_fingerprints(
        self,
        snapshot: OwnerPersistenceSnapshot,
    ) -> tuple[str, str, str]:
        """Describe policy/temporal/memory substate inside the owner boundary."""

        self._decode_learning_persistence_snapshot(snapshot)
        payload = typed_to_json(
            dict(snapshot.payload),
            dict[str, Any],
        )
        if not isinstance(payload, dict):
            raise RuntimeError(
                "joint-loop persistence fingerprint codec did not produce an object"
            )
        policy_payload = {
            "world": payload["world_policy_checkpoint"],
            "self": payload["self_policy_checkpoint"],
        }
        temporal_payload = {
            "world_policy": payload["world_policy_checkpoint"],
            "self_policy": payload["self_policy_checkpoint"],
            "world_temporal": payload["world_temporal_snapshot"],
            "self_temporal": payload["self_temporal_snapshot"],
        }
        return (
            hashlib.sha256(canonical_json_bytes(policy_payload)).hexdigest(),
            hashlib.sha256(canonical_json_bytes(temporal_payload)).hexdigest(),
            hashlib.sha256(
                canonical_json_bytes(payload["memory_checkpoint"])
            ).hexdigest(),
        )

    @staticmethod
    def _decode_learning_persistence_snapshot(
        snapshot: OwnerPersistenceSnapshot,
    ) -> RareHeavyImportCheckpoint:
        if snapshot.owner_name != _JOINT_LEARNING_OWNER_NAME:
            raise HydrationOwnerMismatchError(
                "ETANLJointLoop expected owner_name="
                f"{_JOINT_LEARNING_OWNER_NAME!r}, got {snapshot.owner_name!r}"
            )
        if snapshot.schema_version != _JOINT_LEARNING_SCHEMA_VERSION:
            raise HydrationVersionMismatchError(
                "ETANLJointLoop unsupported schema_version="
                f"{snapshot.schema_version!r}; expected "
                f"{_JOINT_LEARNING_SCHEMA_VERSION}"
            )
        try:
            checkpoint = typed_from_json(
                dict(snapshot.payload),
                RareHeavyImportCheckpoint,
            )
        except CanonicalJsonError as exc:
            raise HydrationPayloadInvalidError(
                f"ETANLJointLoop learning payload is invalid: {exc}"
            ) from exc
        if not isinstance(checkpoint, RareHeavyImportCheckpoint):
            raise HydrationPayloadInvalidError(
                "ETANLJointLoop codec returned an unexpected checkpoint type"
            )
        return checkpoint

    def restore_learning_checkpoint(
        self, checkpoint: RareHeavyImportCheckpoint
    ) -> tuple[str, ...]:
        """Restore a checkpoint created by :meth:`create_learning_checkpoint`."""

        operations = self.rollback_rare_heavy_import(checkpoint)
        if checkpoint.schedule_gate_state is not None:
            self._schedule_gate_learner.restore_state(
                checkpoint.schedule_gate_state
            )
            operations = operations + ("schedule-gate:restored",)
        if checkpoint.runtime_replay_report is None:
            self._world_sandbox.reset_runtime_replay_for_episode_transfer()
            self._self_sandbox.reset_runtime_replay_for_episode_transfer()
            self._pending_task_rollouts = []
            self._pending_relationship_rollouts = []
            self._open_task_segment_rollouts = []
            self._open_relationship_segment_rollouts = []
            self._runtime_replay_captured_count = 0
            self._runtime_replay_settled_count = 0
            self._runtime_replay_transition_count = 0
            self._runtime_replay_lineage_match_count = 0
            self._runtime_replay_drop_reasons = []
            self._runtime_segment_closed_count = 0
            self._runtime_longest_segment_length = 0
            self._runtime_last_segment_close_reason = ""
            self._runtime_segment_close_reason_counts = {}
            return operations + ("runtime-replay:episode-transfer-reset",)
        return operations

    @property
    def ssl_backend(self):
        """Deploy-wiring readout: the SSL trainer's torch backend level."""

        return self._ssl_trainer.ssl_backend

    @property
    def internal_rl_backend(self):
        """Deploy-wiring readout: the internal-RL sandboxes' torch backend level.

        Both track sandboxes are constructed with the same level; fail loudly
        if they ever diverge (that would be a wiring bug, not a valid state).
        """

        world_level = self._world_sandbox.causal_policy.rl_backend
        self_level = self._self_sandbox.causal_policy.rl_backend
        if world_level is not self_level:
            raise RuntimeError(
                "internal RL backend diverged between world "
                f"({world_level.value}) and self ({self_level.value}) sandboxes"
            )
        return world_level

    @property
    def latest_ssl_report(self):
        """Most recent SSLTrainingReport from the shared trainer (evidence readout).

        Carries the ``torch_*`` backend fields for the learned-shadow evidence
        artifact. ``None`` until the schedule has run SSL at least once.
        """

        return self._ssl_trainer.latest_report

    @property
    def latest_internal_rl_report(self):
        """Most recent world-sandbox optimization report (evidence readout).

        Carries the ``torch_*`` PPO backend fields for the learned-shadow
        evidence artifact. ``None`` until a full cycle has optimized once.
        """

        return self._world_sandbox.latest_optimization_report

    @property
    def latest_reward_composition(self):
        """Most recent world-env reward composition readout (CP-07 / GAP-09).

        Includes ``pe_derived_abs_fraction`` — the share of |reward| that
        came from PE / segment-credit derived components — so the
        reward-purity claim is measurable per cycle. ``None`` until a
        rollout has stepped the environment.
        """

        return self._world_sandbox.latest_reward_composition

    @property
    def latest_torch_ssl_candidate(self):
        """Most recent torch SSL candidate checkpoint report (CP-05 / GAP-09)."""

        return self._ssl_trainer.latest_torch_ssl_candidate

    @property
    def latest_ssl_forward_parity(self):
        """Most recent SSL-lane pure/torch forward-parity report (CP-05)."""

        return self._ssl_trainer.latest_ssl_forward_parity

    def _aggregate_metacontroller_state(self) -> MetacontrollerRuntimeState:
        return build_temporal_runtime_state_aggregate(
            world_state=self._world_policy.export_runtime_state(),
            self_state=self._self_policy.export_runtime_state(),
        )

    @staticmethod
    def _empty_runtime_rollout(
        *,
        rollout_id: str,
        track: Track,
        reason: str,
    ) -> ZRollout:
        return ZRollout(
            rollout_id=rollout_id,
            track=track,
            transitions=(),
            total_reward=0.0,
            description=reason,
            replacement_mode="runtime-replay",
            reward_mode="runtime-replay",
        )

    @staticmethod
    def _merge_runtime_rollouts(
        *,
        rollout_id: str,
        track: Track,
        rollouts: tuple[ZRollout, ...],
    ) -> ZRollout:
        if any(rollout.track is not track for rollout in rollouts):
            raise ValueError(
                f"cannot merge runtime replay rollouts across tracks for {track.value}"
            )
        flattened = tuple(
            transition
            for rollout in rollouts
            for transition in rollout.transitions
        )
        transitions = tuple(
            replace(transition, step_index=index)
            for index, transition in enumerate(flattened)
        )
        return ZRollout(
            rollout_id=rollout_id,
            track=track,
            transitions=transitions,
            total_reward=sum(transition.reward for transition in transitions),
            description=(
                f"Merged {len(rollouts)} settled runtime replay rollouts "
                f"for {track.value} with {len(transitions)} transitions."
            ),
            replacement_mode="runtime-replay",
            reward_mode="runtime-replay",
        )

    @staticmethod
    def _waiting_for_runtime_replay_report(*, track: Track, count: int) -> OptimizationReport:
        return OptimizationReport(
            track=track,
            average_reward=0.0,
            baseline_reward=0.0,
            mean_advantage=0.0,
            surrogate_objective=0.0,
            clip_fraction=0.0,
            kl_penalty=0.0,
            parameter_summary="waiting-for-runtime-replay",
            rollout_count=count,
            transition_count=0,
        )

    async def run_cycle(
        self,
        *,
        cycle_index: int,
        trace: TrainingTrace,
        session_id: str | None = None,
        wave_id: str | None = None,
        prior_session_reports: tuple[EvaluationReport, ...] = (),
        apply_writeback: bool = True,
        apply_policy_optimization: bool = True,
    ) -> JointCycleReport:
        world_cycle_checkpoint = self._world_sandbox.create_checkpoint(
            checkpoint_id=f"joint-cycle-{cycle_index}:world"
        )
        self_cycle_checkpoint = self._self_sandbox.create_checkpoint(
            checkpoint_id=f"joint-cycle-{cycle_index}:self"
        )
        self._world_policy.parameter_store.set_learning_phase("ssl-online", structure_frozen=False)
        world_ssl_report = self._ssl_trainer.optimize(policy=self._world_policy, trace=trace)
        self._self_policy.parameter_store.set_learning_phase("ssl-online", structure_frozen=False)
        self_ssl_report = self._ssl_trainer.optimize(policy=self._self_policy, trace=trace)
        substrate_snapshots = tuple(self._snapshot_from_trace_step(step, trace) for step in trace.steps)
        self._world_policy.parameter_store.set_learning_phase("rl-online", structure_frozen=True)
        self._self_policy.parameter_store.set_learning_phase("rl-online", structure_frozen=True)
        # Matched no-optimize control: checkpoint AFTER SSL and BEFORE RL so the
        # arm observes the same trace, SSL schedule, rollout, PE and optimizer
        # report, but cannot persist policy/critic changes. This is separate
        # from ``apply_writeback``, which gates reflection/memory consolidation.
        world_pre_rl_checkpoint = self._world_sandbox.create_checkpoint(
            checkpoint_id=f"joint-cycle-{cycle_index}:world-pre-rl"
        )
        self_pre_rl_checkpoint = self._self_sandbox.create_checkpoint(
            checkpoint_id=f"joint-cycle-{cycle_index}:self-pre-rl"
        )
        if self._internal_rl_runtime_replay is WiringLevel.ACTIVE:
            if len(self._pending_task_rollouts) != len(
                self._pending_relationship_rollouts
            ):
                raise RuntimeError(
                    "ACTIVE runtime replay staging diverged between tracks"
                )
            batch_due = (
                len(self._pending_task_rollouts)
                >= self._rl_batch_accumulation_size
            )
            if batch_due:
                task_batch = tuple(self._pending_task_rollouts)
                relationship_batch = tuple(
                    self._pending_relationship_rollouts
                )
                task_rollout = self._merge_runtime_rollouts(
                    rollout_id=f"joint-{cycle_index}:runtime-task",
                    track=Track.WORLD,
                    rollouts=task_batch,
                )
                relationship_rollout = self._merge_runtime_rollouts(
                    rollout_id=f"joint-{cycle_index}:runtime-relationship",
                    track=Track.SELF,
                    rollouts=relationship_batch,
                )
            else:
                task_batch = ()
                relationship_batch = ()
                task_rollout = self._empty_runtime_rollout(
                    rollout_id=f"joint-{cycle_index}:runtime-task-wait",
                    track=Track.WORLD,
                    reason="waiting-for-runtime-replay",
                )
                relationship_rollout = self._empty_runtime_rollout(
                    rollout_id=f"joint-{cycle_index}:runtime-relationship-wait",
                    track=Track.SELF,
                    reason="waiting-for-runtime-replay",
                )
        else:
            self._world_sandbox.configure_runtime_backend(
                source_text=trace.source_text
            )
            self._self_sandbox.configure_runtime_backend(
                source_text=trace.source_text
            )
            eval_signals = dict(self._previous_family_signals)
            eval_signals.update(self._external_learning_signals)
            if self._previous_credit_snapshot is not None:
                credit_bonus = extract_abstract_action_credit_bonus(
                    self._previous_credit_snapshot
                )
                eval_signals.update(credit_bonus)
            if eval_signals:
                self._world_sandbox._env.set_evaluation_signals(eval_signals)
                self._self_sandbox._env.set_evaluation_signals(eval_signals)
            task_rollout = self._world_sandbox.rollout(
                rollout_id=f"joint-{cycle_index}:task",
                substrate_steps=substrate_snapshots,
                track=Track.WORLD,
                replacement_mode="causal-binary",
            )
            relationship_rollout = self._self_sandbox.rollout(
                rollout_id=f"joint-{cycle_index}:relationship",
                substrate_steps=substrate_snapshots,
                track=Track.SELF,
                replacement_mode="causal-binary",
            )
            self._pending_task_rollouts.append(task_rollout)
            self._pending_relationship_rollouts.append(relationship_rollout)
            batch_due = (
                len(self._pending_task_rollouts)
                >= self._rl_batch_accumulation_size
            )
            task_batch = (
                tuple(self._pending_task_rollouts) if batch_due else ()
            )
            relationship_batch = (
                tuple(self._pending_relationship_rollouts)
                if batch_due
                else ()
            )
        dual_track_rollout = DualTrackRollout(
            task_rollout=task_rollout,
            relationship_rollout=relationship_rollout,
            description=(
                f"Dual-track rollout task_reward={task_rollout.total_reward:.2f}, "
                f"relationship_reward={relationship_rollout.total_reward:.2f}."
            ),
        )
        world_before_hash = stable_value_hash(
            self._world_sandbox.causal_policy.export_parameters()
        )
        self_before_hash = stable_value_hash(
            self._self_sandbox.causal_policy.export_parameters()
        )
        if batch_due:
            task_report = self._world_sandbox.optimize(task_batch)
            relationship_report = self._self_sandbox.optimize(
                relationship_batch
            )
            if not apply_policy_optimization:
                self._world_sandbox.restore_checkpoint(world_pre_rl_checkpoint)
                self._self_sandbox.restore_checkpoint(self_pre_rl_checkpoint)
            self._pending_task_rollouts.clear()
            self._pending_relationship_rollouts.clear()
            rl_batch_rollout_count = len(task_batch)
        elif self._internal_rl_runtime_replay is WiringLevel.ACTIVE:
            task_report = self._waiting_for_runtime_replay_report(
                track=Track.WORLD,
                count=len(self._pending_task_rollouts),
            )
            relationship_report = self._waiting_for_runtime_replay_report(
                track=Track.SELF,
                count=len(self._pending_relationship_rollouts),
            )
            rl_batch_rollout_count = len(self._pending_task_rollouts)
        else:
            task_report = OptimizationReport(
                track=Track.WORLD,
                average_reward=0.0,
                baseline_reward=0.0,
                mean_advantage=0.0,
                surrogate_objective=0.0,
                clip_fraction=0.0,
                kl_penalty=0.0,
                parameter_summary="waiting-for-batch",
                rollout_count=len(self._pending_task_rollouts),
                transition_count=sum(
                    len(rollout.transitions)
                    for rollout in self._pending_task_rollouts
                ),
            )
            relationship_report = OptimizationReport(
                track=Track.SELF,
                average_reward=0.0,
                baseline_reward=0.0,
                mean_advantage=0.0,
                surrogate_objective=0.0,
                clip_fraction=0.0,
                kl_penalty=0.0,
                parameter_summary="waiting-for-batch",
                rollout_count=len(self._pending_relationship_rollouts),
                transition_count=sum(
                    len(rollout.transitions)
                    for rollout in self._pending_relationship_rollouts
                ),
            )
            rl_batch_rollout_count = len(self._pending_task_rollouts)
        if not isinstance(task_report, OptimizationReport):
            raise TypeError("Expected OptimizationReport for world track optimization.")
        if not isinstance(relationship_report, OptimizationReport):
            raise TypeError("Expected OptimizationReport for self track optimization.")
        world_after_hash = stable_value_hash(self._world_sandbox.causal_policy.export_parameters())
        self_after_hash = stable_value_hash(self._self_sandbox.causal_policy.export_parameters())
        modification_records: list[SelfModificationRecord] = []
        if world_before_hash != world_after_hash:
            modification_records.append(
                SelfModificationRecord(
                    target="causal_policy.world_track_weights",
                    gate=ModificationGate.ONLINE,
                    decision=GateDecision.ALLOW,
                    old_value_hash=world_before_hash,
                    new_value_hash=world_after_hash,
                    justification=(
                        f"World-track RL update task_obj={task_report.surrogate_objective:.3f} "
                        f"kl={task_report.kl_penalty:.3f} epochs={task_report.epochs_executed}"
                    ),
                    timestamp_ms=cycle_index,
                    is_reversible=True,
                )
            )
        if self_before_hash != self_after_hash:
            modification_records.append(
                SelfModificationRecord(
                    target="causal_policy.self_track_weights",
                    gate=ModificationGate.ONLINE,
                    decision=GateDecision.ALLOW,
                    old_value_hash=self_before_hash,
                    new_value_hash=self_after_hash,
                    justification=(
                        f"Self-track RL update rel_obj={relationship_report.surrogate_objective:.3f} "
                        f"kl={relationship_report.kl_penalty:.3f} epochs={relationship_report.epochs_executed}"
                    ),
                    timestamp_ms=cycle_index,
                    is_reversible=True,
                )
            )
        optimization_report = DualTrackOptimizationReport(
            task_report=task_report,
            relationship_report=relationship_report,
            description=(
                f"task_adv={task_report.mean_advantage:.3f}, "
                f"rel_adv={relationship_report.mean_advantage:.3f}"
            ),
        )
        optimization_result = PolicyOptimizationResult(
            optimization_report=optimization_report,
            modification_records=tuple(modification_records),
            policy_update_applied=bool(modification_records),
            total_kl_divergence=task_report.kl_penalty + relationship_report.kl_penalty,
            total_epochs_executed=task_report.epochs_executed + relationship_report.epochs_executed,
        )
        optimization_report = optimization_result.optimization_report
        session_id = session_id or f"joint-session-{cycle_index}"
        wave_id = wave_id or f"joint-wave-{cycle_index}"
        modules = [
            SubstrateModule(
                adapter=SimulatedResidualSubstrateAdapter(trace=trace),
                wiring_level=WiringLevel.ACTIVE,
            ),
            MemoryModule(
                store=self._memory_store,
                wiring_level=WiringLevel.ACTIVE,
                memory_feedback_signal=tuple(
                    (
                        (world_ssl_report.m3_slow_momentum_signal[index] if index < len(world_ssl_report.m3_slow_momentum_signal) else 0.0)
                        + (self_ssl_report.m3_slow_momentum_signal[index] if index < len(self_ssl_report.m3_slow_momentum_signal) else 0.0)
                    )
                    / 2.0
                    for index in range(
                        max(
                            len(world_ssl_report.m3_slow_momentum_signal),
                            len(self_ssl_report.m3_slow_momentum_signal),
                            0,
                        )
                    )
                ),
            ),
            TrackTemporalModule(
                track=Track.WORLD,
                policy=self._world_policy,
                wiring_level=WiringLevel.ACTIVE,
            ),
            TrackTemporalModule(
                track=Track.SELF,
                policy=self._self_policy,
                wiring_level=WiringLevel.ACTIVE,
            ),
            TemporalAggregateModule(wiring_level=WiringLevel.ACTIVE),
            DualTrackModule(wiring_level=WiringLevel.ACTIVE),
            EvaluationModule(
                backbone=self._evaluation_backbone,
                session_id=session_id,
                wave_id=wave_id,
                wiring_level=WiringLevel.ACTIVE,
            ),
            self._regime_module,
            CreditModule(wiring_level=WiringLevel.ACTIVE),
            ReflectionModule(
                engine=ReflectionEngine(writeback_mode=WritebackMode.APPLY),
                wiring_level=WiringLevel.ACTIVE,
            ),
            TrackTemporalConsolidationModule(
                track=Track.WORLD,
                policy=self._world_policy,
                wiring_level=WiringLevel.ACTIVE,
            ),
            TrackTemporalConsolidationModule(
                track=Track.SELF,
                policy=self._self_policy,
                wiring_level=WiringLevel.ACTIVE,
            ),
        ]
        world_temporal_module = next(
            module
            for module in modules
            if isinstance(module, TrackTemporalModule) and module.track is Track.WORLD
        )
        self_temporal_module = next(
            module
            for module in modules
            if isinstance(module, TrackTemporalModule) and module.track is Track.SELF
        )
        recorder = EventRecorder()
        active_snapshots = await propagate(
            modules,
            registry=SlotRegistry(),
            recorder=recorder,
            shadow_snapshots={},
            session_id=session_id,
            wave_id=wave_id,
        )
        enriched_credit_snapshot = self._enrich_credit_snapshot(
            active_snapshots,
            dual_track_rollout=dual_track_rollout,
        )
        if optimization_result.modification_records:
            enriched_credit_snapshot = extend_credit_snapshot(
                credit_snapshot=enriched_credit_snapshot,
                extra_modifications=optimization_result.modification_records,
            )
        enriched_credit_snapshot = extend_credit_snapshot(
            credit_snapshot=enriched_credit_snapshot,
            extra_records=derive_delayed_attribution_credit_records(
                regime_snapshot=active_snapshots["regime"].value,
                timestamp_ms=active_snapshots["credit"].timestamp_ms + 125,
            ),
        )
        total_reward = (
            dual_track_rollout.task_rollout.total_reward
            + dual_track_rollout.relationship_rollout.total_reward
        )
        all_transitions = dual_track_rollout.task_rollout.transitions + dual_track_rollout.relationship_rollout.transitions
        mean_transition_reward = total_reward / len(all_transitions) if all_transitions else 0.0
        backend_fidelity = (
            sum(transition.backend_fidelity for transition in all_transitions) / len(all_transitions)
            if all_transitions
            else 0.0
        )
        backend_names = {transition.backend_name for transition in all_transitions}
        if len(backend_names) == 1:
            backend_name = backend_names.pop()
        elif not backend_names and self._internal_rl_runtime_replay is WiringLevel.ACTIVE:
            backend_name = "waiting-for-runtime-replay"
        else:
            backend_name = "mixed"
        evaluation_snapshot = active_snapshots["evaluation"].value
        temporal_snapshot = active_snapshots.get("temporal_abstraction")
        temporal_value = (
            temporal_snapshot.value
            if temporal_snapshot is not None and isinstance(temporal_snapshot.value, TemporalAbstractionSnapshot)
            else None
        )
        # === TRAINING WRITEBACK PHASE BEGIN ====================================
        # All owner-state mutations from here to the matching `PHASE END`
        # marker happen on shared instances (memory_store / evaluation_backbone
        # / world_policy / self_policy). See class docstring for the contract.
        # Any change that writes to a shared owner MUST sit inside this block.
        # ========================================================================
        evaluation_snapshot = self._evaluation_backbone.record_temporal_public_evidence(
            session_id=session_id,
            wave_id=wave_id,
            timestamp_ms=active_snapshots["evaluation"].timestamp_ms + 1,
            base_snapshot=evaluation_snapshot,
            temporal_snapshot=temporal_value,
        )
        self._modulate_ssl_learning_rate(evaluation_snapshot)
        pre_rollback_metacontroller_state = self._aggregate_metacontroller_state()
        rollback_reasons = self._rollback_reasons(
            total_reward=total_reward,
            evaluation_snapshot=evaluation_snapshot,
            optimization_report=optimization_report,
            metacontroller_state=pre_rollback_metacontroller_state,
            optimization_evidence_available=not (
                self._internal_rl_runtime_replay is WiringLevel.ACTIVE
                and not batch_due
            ),
        )
        rollback_required = bool(rollback_reasons)
        policy_rollback_applied = False
        ssl_rollback_applied = False
        if rollback_required:
            self._world_sandbox.restore_checkpoint(world_cycle_checkpoint)
            self._self_sandbox.restore_checkpoint(self_cycle_checkpoint)
            policy_rollback_applied = True
            ssl_rollback_applied = True
        metacontroller_state = self._aggregate_metacontroller_state()
        policy_objective = (
            optimization_report.task_report.surrogate_objective
            + optimization_report.relationship_report.surrogate_objective
        )
        kernel_scores = self._evaluation_backbone.record_metacontroller_evidence(
            session_id=session_id,
            wave_id=wave_id,
            timestamp_ms=active_snapshots["evaluation"].timestamp_ms + 1,
            metacontroller_state=metacontroller_state,
            policy_objective=policy_objective,
            rollback_reasons=rollback_reasons,
        )
        evaluation_snapshot = self._evaluation_backbone.record_external_scores(
            session_id=session_id,
            wave_id=wave_id,
            timestamp_ms=active_snapshots["evaluation"].timestamp_ms + 2,
            base_snapshot=evaluation_snapshot,
            scores=kernel_scores,
            description_suffix=f"Enriched with {len(kernel_scores)} metacontroller evidence scores.",
        )
        regime_operations = self._regime_module.apply_metacontroller_evidence(
            metacontroller_state=metacontroller_state,
            rollback_reasons=rollback_reasons,
        )
        enriched_credit_snapshot = extend_credit_snapshot(
            credit_snapshot=enriched_credit_snapshot,
            extra_records=derive_metacontroller_credit_records(
                metacontroller_state=metacontroller_state,
                policy_objective=policy_objective,
                rollback_reasons=rollback_reasons,
                timestamp_ms=active_snapshots["credit"].timestamp_ms + 150,
            ),
            extra_modifications=derive_runtime_adaptation_audit_records(
                rollback_reasons=rollback_reasons,
                metacontroller_state_description=(
                    metacontroller_state.description if metacontroller_state is not None else None
                ),
                timestamp_ms=active_snapshots["credit"].timestamp_ms + 200,
                rollback_applied=policy_rollback_applied,
            ),
        )
        session_report = self._evaluation_backbone.build_session_report(
            session_id=session_id,
            timestamp_ms=active_snapshots["evaluation"].timestamp_ms + 3,
        )
        cross_session_report = None
        if prior_session_reports:
            cross_session_report = self._evaluation_backbone.run_cross_session_benchmark(
                suite=CrossSessionBenchmarkSuite(
                    session_reports=prior_session_reports + (session_report,),
                )
            )
        replay_result = self._evaluation_backbone.run_default_evolution_benchmark(
            timestamp_ms=active_snapshots["evaluation"].timestamp_ms + 4,
        )
        evolution_judgement = self._evaluation_backbone.judge_evolution_candidate(
            replay_suite_result=replay_result,
            session_report=session_report,
            cross_session_report=cross_session_report,
        )
        # Runtime replay is trained from matched PE-first environment evidence.
        # The generic evolution benchmark has no access to that transition
        # lineage, so it remains a readout and cannot become a second reward/
        # rollback owner for this lane. Typed safety, relationship, KL and drift
        # gates in ``_rollback_reasons`` remain authoritative.
        if (
            not rollback_required
            and evolution_judgement.decision is EvolutionDecision.ROLLBACK
            and self._internal_rl_runtime_replay is not WiringLevel.ACTIVE
        ):
            self._world_sandbox.restore_checkpoint(world_cycle_checkpoint)
            self._self_sandbox.restore_checkpoint(self_cycle_checkpoint)
            policy_rollback_applied = True
            ssl_rollback_applied = True
            rollback_reasons = rollback_reasons + ("evolution-judge-rollback",)
        judge_allows_structural = (
            evolution_judgement.decision is EvolutionDecision.PROMOTE
            or (
                evolution_judgement.decision is EvolutionDecision.HOLD
                and evolution_judgement.category is not JudgementCategory.UNSAFE_MUTATION
            )
        )
        owner_writeback_enabled = (
            apply_writeback
            and judge_allows_structural
            and not rollback_required
            and not policy_rollback_applied
            and not ssl_rollback_applied
        )
        reflection_snapshot = ReflectionEngine(writeback_mode=WritebackMode.APPLY).reflect(
            timestamp_ms=active_snapshots["reflection"].timestamp_ms + 1,
            memory_snapshot=active_snapshots["memory"].value,
            dual_track_snapshot=active_snapshots["dual_track"].value,
            evaluation_snapshot=evaluation_snapshot,
            credit_snapshot=enriched_credit_snapshot,
            regime_snapshot=active_snapshots["regime"].value,
        )
        temporal_writeback_operations, temporal_writeback_audits = self._apply_temporal_reflection_writeback(
            temporal_module=world_temporal_module,
            reflection_snapshot=reflection_snapshot,
            credit_snapshot=enriched_credit_snapshot,
            timestamp_ms=active_snapshots["credit"].timestamp_ms + 175,
            apply_enabled=owner_writeback_enabled,
        )
        self_track_temporal_writeback_operations, self_track_temporal_writeback_audits = self._apply_temporal_reflection_writeback(
            temporal_module=self_temporal_module,
            reflection_snapshot=reflection_snapshot,
            credit_snapshot=enriched_credit_snapshot,
            timestamp_ms=active_snapshots["credit"].timestamp_ms + 176,
            apply_enabled=owner_writeback_enabled,
        )
        if temporal_writeback_audits:
            enriched_credit_snapshot = extend_credit_snapshot(
                credit_snapshot=enriched_credit_snapshot,
                extra_modifications=temporal_writeback_audits,
            )
        if self_track_temporal_writeback_audits:
            enriched_credit_snapshot = extend_credit_snapshot(
                credit_snapshot=enriched_credit_snapshot,
                extra_modifications=self_track_temporal_writeback_audits,
            )
        applied_operations: tuple[str, ...] = regime_operations
        blocked_writeback_operations: tuple[str, ...] = ()
        memory_regime_writeback_applied = False
        if owner_writeback_enabled:
            reflection_writeback_result = ReflectionEngine(writeback_mode=WritebackMode.APPLY).apply(
                memory_store=self._memory_store,
                reflection_snapshot=reflection_snapshot,
                credit_snapshot=enriched_credit_snapshot,
                checkpoint_id=f"{session_id}:{wave_id}",
            )
            if not reflection_writeback_result.blocked_operations:
                reflection_regime_operations = self._regime_module.apply_policy_consolidation(
                    strategy_updates=reflection_snapshot.policy_consolidation.strategy_priors_updated,
                    regime_effectiveness_updates=reflection_snapshot.policy_consolidation.regime_effectiveness_updated,
                    strategy_gain=reflection_snapshot.consolidation_score.strategy_gain,
                    effectiveness_gain=reflection_snapshot.consolidation_score.regime_effectiveness_gain,
                )
                if reflection_regime_operations:
                    reflection_writeback_result = replace(
                        reflection_writeback_result,
                        applied_operations=(
                            reflection_writeback_result.applied_operations
                            + reflection_regime_operations
                        ),
                        description=(
                            f"{reflection_writeback_result.description} "
                            f"Regime owner applied {len(reflection_regime_operations)} consolidation operations."
                        ),
                    )
            applied_operations = applied_operations + reflection_writeback_result.applied_operations
            blocked_writeback_operations = reflection_writeback_result.blocked_operations
            memory_regime_writeback_applied = bool(reflection_writeback_result.applied_operations)
        elif not apply_writeback:
            blocked_writeback_operations = ("owner-writeback:disabled",)
        elif not judge_allows_structural:
            blocked_writeback_operations = ("owner-writeback:evolution-judge-block",)
        elif rollback_required or policy_rollback_applied or ssl_rollback_applied:
            blocked_writeback_operations = ("owner-writeback:rollback-protection",)
        applied_operations = (
            applied_operations
            + temporal_writeback_operations
            + self_track_temporal_writeback_operations
        )
        if ssl_rollback_applied:
            applied_operations = applied_operations + ("ssl-rollback",)
        if policy_rollback_applied:
            applied_operations = applied_operations + ("policy-rollback",)
        # === TRAINING WRITEBACK PHASE END ======================================
        # Everything below here is pure bookkeeping on joint-loop PRIVATE state
        # (previous_total_reward / previous_metacontroller_state / family
        # signals cache / previous credit snapshot). It never mutates shared
        # owner instances.
        # ========================================================================
        if not (
            self._internal_rl_runtime_replay is WiringLevel.ACTIVE
            and not batch_due
        ):
            self._previous_total_reward = total_reward
        self._previous_metacontroller_state = metacontroller_state
        self._previous_family_signals = self._evaluation_backbone.family_signals(evaluation_snapshot)
        self._previous_credit_snapshot = enriched_credit_snapshot
        rare_heavy_review_recommended = self._pe_rare_heavy_due(schedule=JointLoopSchedule())
        temporal_writeback_applied = bool(
            temporal_writeback_operations or self_track_temporal_writeback_operations
        )
        default_continual_learning_surface = DefaultContinualLearningSurface(
            surface_id=f"{self.owner_path}:cycle-{cycle_index}:default-continual",
            active=bool(memory_regime_writeback_applied or temporal_writeback_applied or regime_operations),
            owner_path=self.owner_path,
            memory_regime_writeback_applied=memory_regime_writeback_applied,
            temporal_writeback_applied=temporal_writeback_applied,
            regime_evidence_applied=bool(regime_operations),
            substrate_live_mutation_applied=False,
            substrate_review_only=True,
            rare_heavy_review_recommended=rare_heavy_review_recommended,
            applied_operations=applied_operations,
            blocked_operations=blocked_writeback_operations,
            rollback_applied=bool(policy_rollback_applied or ssl_rollback_applied),
            evolution_decision=evolution_judgement.decision.value,
            evolution_category=evolution_judgement.category.value,
            description=(
                "Default continual learner surface retained owner-side memory/temporal/regime/reflection "
                f"writeback={int(memory_regime_writeback_applied or temporal_writeback_applied)} "
                f"regime_evidence={int(bool(regime_operations))} "
                f"substrate_live_mutation=0 rare_heavy_review={int(rare_heavy_review_recommended)}."
            ),
        )
        return JointCycleReport(
            cycle_index=cycle_index,
            acceptance_passed="reflection" in active_snapshots and bool(recorder.events),
            ssl_prediction_loss=(world_ssl_report.prediction_loss + self_ssl_report.prediction_loss) / 2.0,
            ssl_kl_loss=(world_ssl_report.kl_loss + self_ssl_report.kl_loss) / 2.0,
            ssl_posterior_drift=(world_ssl_report.posterior_drift + self_ssl_report.posterior_drift) / 2.0,
            total_reward=total_reward,
            mean_transition_reward=mean_transition_reward,
            task_reward=task_report.average_reward * max(task_report.transition_count, 1),
            relationship_reward=relationship_report.average_reward * max(relationship_report.transition_count, 1),
            ssl_rollback_applied=ssl_rollback_applied,
            policy_rollback_applied=policy_rollback_applied,
            rollback_reasons=rollback_reasons,
            optimization_summary=optimization_report.description,
            policy_objective=policy_objective,
            kernel_score_count=len(kernel_scores),
            kernel_scores=kernel_scores,
            backend_name=backend_name,
            backend_fidelity=backend_fidelity,
            applied_operations=applied_operations,
            metacontroller_state=metacontroller_state,
            cms_description=self._memory_store.learned_core.snapshot().description
            if self._memory_store.learned_core is not None
            else "No CMS core attached.",
            evolution_judgement=evolution_judgement,
            owner_path=self.owner_path,
            schedule_telemetry=(
                ("cycle_index", cycle_index),
                ("ssl_interval", 1),
                ("rl_interval", 1),
                ("ssl_due", 1),
                ("rl_due", 1),
            ),
            description=(
                f"Joint ETA/NL cycle {cycle_index} owner={self.owner_path} ran ssl("
                f"world_pred={world_ssl_report.prediction_loss:.2f}, self_pred={self_ssl_report.prediction_loss:.2f}, "
                f"world_kl={world_ssl_report.kl_loss:.2f}, self_kl={self_ssl_report.kl_loss:.2f}) and dual-track rollout "
                f"task={dual_track_rollout.task_rollout.total_reward:.2f}, "
                f"relationship={dual_track_rollout.relationship_rollout.total_reward:.2f}, "
                f"mean_reward={mean_transition_reward:.2f}, "
                f"rollback={'on' if policy_rollback_applied else 'off'}, "
                f"reasons={','.join(rollback_reasons) if rollback_reasons else 'none'}, "
                f"transition_source={self.latest_runtime_replay_report.transition_source}, "
                f"backend={backend_name}, fidelity={backend_fidelity:.2f}, "
                f"controller={metacontroller_state.description if metacontroller_state is not None else 'unavailable'}, "
                f"kernel_scores={len(kernel_scores)}, with {len(applied_operations)} bounded writeback operations."
            ),
            policy_update_applied=optimization_result.policy_update_applied,
            policy_kl_divergence=optimization_result.total_kl_divergence,
            policy_epochs_executed=optimization_result.total_epochs_executed,
            rare_heavy_review_recommended=rare_heavy_review_recommended,
            rl_batch_rollout_count=rl_batch_rollout_count,
            default_continual_learning_surface=default_continual_learning_surface,
            runtime_replay_report=self.latest_runtime_replay_report,
        )

    async def run_scheduled_step(
        self,
        *,
        turn_index: int,
        trace: TrainingTrace,
        session_id: str | None = None,
        wave_id: str | None = None,
        prior_session_reports: tuple[EvaluationReport, ...] = (),
        schedule: JointLoopSchedule | None = None,
        apply_writeback: bool = True,
        apply_policy_optimization: bool = True,
        learning_enabled: bool = True,
    ) -> ScheduledJointLoopResult:
        self._world_policy.parameter_store.set_learning_phase("runtime")
        self._self_policy.parameter_store.set_learning_phase("runtime")
        active_schedule = schedule or JointLoopSchedule()
        pe_full_cycle_due = self._pe_full_cycle_due(schedule=active_schedule)
        pe_ssl_due = self._pe_ssl_due(schedule=active_schedule)
        substrate_online_fast_due = self._pe_substrate_online_fast_due(schedule=active_schedule)
        rl_due = active_schedule.rl_interval > 0 and turn_index % active_schedule.rl_interval == 0
        rl_batch_ready_due = self._rl_batch_ready_due()
        rl_batch_wait_due = self._rl_batch_wait_due(turn_index=turn_index, schedule=active_schedule)
        latent_continuation_due = self._latent_continuation_due(
            turn_index=turn_index,
            schedule=active_schedule,
        )
        schedule_telemetry = self._schedule_telemetry(
            turn_index=turn_index,
            schedule=active_schedule,
        )
        cms_description = (
            self._memory_store.learned_core.snapshot().description
            if self._memory_store.learned_core is not None
            else "No CMS core attached."
        )
        if not learning_enabled:
            metacontroller_state = self._aggregate_metacontroller_state()
            self._record_schedule_outcome(
                turn_index=turn_index,
                schedule_action="frozen-evidence-only",
                metacontroller_state=metacontroller_state,
            )
            return ScheduledJointLoopResult(
                turn_index=turn_index,
                schedule_action="frozen-evidence-only",
                cycle_report=None,
                kernel_scores=(),
                ssl_prediction_loss=0.0,
                ssl_kl_loss=0.0,
                metacontroller_state=metacontroller_state,
                cms_description=cms_description,
                owner_path=self.owner_path,
                schedule_telemetry=schedule_telemetry,
                description=(
                    f"Scheduled joint loop owner={self.owner_path} collected "
                    f"frozen evidence only at turn {turn_index}."
                ),
                substrate_online_fast_due=False,
                rare_heavy_review_recommended=False,
                runtime_replay_report=self.latest_runtime_replay_report,
            )
        rare_heavy_review_recommended = self._pe_rare_heavy_due(schedule=active_schedule)
        # T3 (#88): report-only learned schedule gate. Derive the shadow
        # from the same inputs the rule cascade consumes and publish it in
        # telemetry; the live decision below is untouched (rollback = drop
        # these telemetry entries).
        from volvence_zero.joint_loop.gate_learner import build_schedule_gate_features

        (
            gate_pe_pressure,
            gate_family_stability,
            gate_rollback_risk,
            gate_transition_pressure,
            gate_substrate_pressure,
            gate_rare_heavy_pressure,
        ) = self._joint_schedule_inputs(turn_index=turn_index, schedule=active_schedule)
        gate_shadow = self._schedule_gate_learner.derive_shadow(
            build_schedule_gate_features(
                pe_pressure=gate_pe_pressure,
                family_stability=gate_family_stability,
                rollback_risk=gate_rollback_risk,
                transition_pressure=gate_transition_pressure,
                substrate_pressure=gate_substrate_pressure,
                rare_heavy_pressure=gate_rare_heavy_pressure,
                experience_credit=self._experience_credit_signal(),
                control_prior_strength=self._experience_control_prior_signal(),
                pe_full_cycle_due=pe_full_cycle_due,
                rl_due=rl_due,
            )
        )
        schedule_telemetry = schedule_telemetry + (
            ("learned_gate_pressure_x1000", int(gate_shadow.predicted_learning_gain * 1000)),
            ("learned_gate_recommends_learning", int(gate_shadow.recommends_learning)),
            ("learned_gate_settled_count", gate_shadow.settled_count),
        )
        batch_schedule_action = self._batch_schedule_action(
            turn_index=turn_index,
            schedule=active_schedule,
            pe_full_cycle_due=pe_full_cycle_due,
            pe_ssl_due=pe_ssl_due,
            rl_due=rl_due,
            rl_batch_ready_due=rl_batch_ready_due,
            rl_batch_wait_due=rl_batch_wait_due,
            substrate_online_fast_due=substrate_online_fast_due,
            rare_heavy_review_recommended=rare_heavy_review_recommended,
        )
        should_run_cycle = (
            batch_schedule_action is not None and batch_schedule_action.startswith("full-cycle")
        ) or (
            self._effective_rl_batch_target() <= 1
            and (pe_full_cycle_due or rl_due or rl_batch_ready_due or rl_batch_wait_due)
        )
        if should_run_cycle:
            if batch_schedule_action is not None:
                schedule_action = batch_schedule_action
            elif pe_full_cycle_due:
                schedule_action = "full-cycle-pe"
            else:
                schedule_action = "full-cycle"
            cycle_report = await self.run_cycle(
                cycle_index=turn_index,
                trace=trace,
                session_id=session_id,
                wave_id=wave_id,
                prior_session_reports=prior_session_reports,
                apply_writeback=apply_writeback,
                apply_policy_optimization=apply_policy_optimization,
            )
            if cycle_report.backend_name == "waiting-for-runtime-replay":
                schedule_action = "waiting-for-runtime-replay"
            self._record_schedule_outcome(
                turn_index=turn_index,
                schedule_action=schedule_action,
                metacontroller_state=cycle_report.metacontroller_state,
            )
            self._settle_schedule_gate(current_ssl_prediction_loss=cycle_report.ssl_prediction_loss)
            return ScheduledJointLoopResult(
                turn_index=turn_index,
                schedule_action=schedule_action,
                cycle_report=cycle_report,
                kernel_scores=cycle_report.kernel_scores,
                ssl_prediction_loss=cycle_report.ssl_prediction_loss,
                ssl_kl_loss=cycle_report.ssl_kl_loss,
                metacontroller_state=cycle_report.metacontroller_state,
                cms_description=cycle_report.cms_description,
                owner_path=self.owner_path,
                schedule_telemetry=schedule_telemetry,
                description=cycle_report.description,
                substrate_online_fast_due=substrate_online_fast_due,
                rare_heavy_review_recommended=rare_heavy_review_recommended or cycle_report.rare_heavy_review_recommended,
                default_continual_learning_surface=cycle_report.default_continual_learning_surface,
                runtime_replay_report=cycle_report.runtime_replay_report,
            )
        if batch_schedule_action in {
            "ssl-only-risk-hold",
            "evidence-only-risk-hold",
            "ssl-only-rare-heavy-hold",
        }:
            metacontroller_state = self._aggregate_metacontroller_state()
            self._record_schedule_outcome(
                turn_index=turn_index,
                schedule_action=batch_schedule_action,
                metacontroller_state=metacontroller_state,
            )
            if batch_schedule_action in {"ssl-only-risk-hold", "ssl-only-rare-heavy-hold"}:
                self._world_policy.parameter_store.set_learning_phase("ssl-online", structure_frozen=False)
                self._self_policy.parameter_store.set_learning_phase("ssl-online", structure_frozen=False)
                world_ssl_report = self._ssl_trainer.optimize(policy=self._world_policy, trace=trace)
                self_ssl_report = self._ssl_trainer.optimize(policy=self._self_policy, trace=trace)
                metacontroller_state = self._aggregate_metacontroller_state()
                self._world_policy.parameter_store.set_learning_phase("runtime", structure_frozen=True)
                self._self_policy.parameter_store.set_learning_phase("runtime", structure_frozen=True)
                self._settle_schedule_gate(
                    current_ssl_prediction_loss=(
                        world_ssl_report.prediction_loss + self_ssl_report.prediction_loss
                    )
                    / 2.0
                )
                return ScheduledJointLoopResult(
                    turn_index=turn_index,
                    schedule_action=batch_schedule_action,
                    cycle_report=None,
                    kernel_scores=(),
                    ssl_prediction_loss=(world_ssl_report.prediction_loss + self_ssl_report.prediction_loss) / 2.0,
                    ssl_kl_loss=(world_ssl_report.kl_loss + self_ssl_report.kl_loss) / 2.0,
                    metacontroller_state=metacontroller_state,
                    cms_description=cms_description,
                    owner_path=self.owner_path,
                    schedule_telemetry=schedule_telemetry,
                    description=(
                        f"Scheduled joint loop owner={self.owner_path} held RL batch and ran ssl-only "
                        f"joint-risk control at turn {turn_index}."
                    ),
                    substrate_online_fast_due=substrate_online_fast_due,
                    rare_heavy_review_recommended=rare_heavy_review_recommended,
                    runtime_replay_report=self.latest_runtime_replay_report,
                )
            return ScheduledJointLoopResult(
                turn_index=turn_index,
                schedule_action=batch_schedule_action,
                cycle_report=None,
                kernel_scores=(),
                ssl_prediction_loss=0.0,
                ssl_kl_loss=0.0,
                metacontroller_state=metacontroller_state,
                cms_description=cms_description,
                owner_path=self.owner_path,
                schedule_telemetry=schedule_telemetry,
                description=(
                    f"Scheduled joint loop owner={self.owner_path} held RL batch and collected evidence only "
                    f"at turn {turn_index}."
                ),
                substrate_online_fast_due=substrate_online_fast_due,
                rare_heavy_review_recommended=rare_heavy_review_recommended,
                runtime_replay_report=self.latest_runtime_replay_report,
            )
        if (
            pe_ssl_due
            or (active_schedule.ssl_interval > 0 and turn_index % active_schedule.ssl_interval == 0)
            or latent_continuation_due
        ):
            self._world_policy.parameter_store.set_learning_phase("ssl-online", structure_frozen=False)
            self._self_policy.parameter_store.set_learning_phase("ssl-online", structure_frozen=False)
            world_ssl_report = self._ssl_trainer.optimize(policy=self._world_policy, trace=trace)
            self_ssl_report = self._ssl_trainer.optimize(policy=self._self_policy, trace=trace)
            metacontroller_state = self._aggregate_metacontroller_state()
            self._world_policy.parameter_store.set_learning_phase("runtime", structure_frozen=True)
            self._self_policy.parameter_store.set_learning_phase("runtime", structure_frozen=True)
            self._settle_schedule_gate(
                current_ssl_prediction_loss=(
                    world_ssl_report.prediction_loss + self_ssl_report.prediction_loss
                )
                / 2.0
            )
            schedule_action = (
                "ssl-only-pe"
                if pe_ssl_due and not pe_full_cycle_due
                else ("ssl-only-continuation" if latent_continuation_due else "ssl-only")
            )
            self._record_schedule_outcome(
                turn_index=turn_index,
                schedule_action=schedule_action,
                metacontroller_state=metacontroller_state,
            )
            return ScheduledJointLoopResult(
                turn_index=turn_index,
                schedule_action=schedule_action,
                cycle_report=None,
                kernel_scores=(),
                ssl_prediction_loss=(world_ssl_report.prediction_loss + self_ssl_report.prediction_loss) / 2.0,
                ssl_kl_loss=(world_ssl_report.kl_loss + self_ssl_report.kl_loss) / 2.0,
                metacontroller_state=metacontroller_state,
                cms_description=cms_description,
                owner_path=self.owner_path,
                schedule_telemetry=schedule_telemetry,
                description=(
                    f"Scheduled joint loop owner={self.owner_path} ran ssl-only at turn {turn_index} with "
                    f"world_pred={world_ssl_report.prediction_loss:.2f}, self_pred={self_ssl_report.prediction_loss:.2f}, "
                    f"world_kl={world_ssl_report.kl_loss:.2f}, self_kl={self_ssl_report.kl_loss:.2f}."
                ),
                substrate_online_fast_due=substrate_online_fast_due,
                rare_heavy_review_recommended=rare_heavy_review_recommended,
                runtime_replay_report=self.latest_runtime_replay_report,
            )
        metacontroller_state = self._aggregate_metacontroller_state()
        self._record_schedule_outcome(
            turn_index=turn_index,
            schedule_action="evidence-only",
            metacontroller_state=metacontroller_state,
        )
        return ScheduledJointLoopResult(
            turn_index=turn_index,
            schedule_action="evidence-only",
            cycle_report=None,
            kernel_scores=(),
            ssl_prediction_loss=0.0,
            ssl_kl_loss=0.0,
            metacontroller_state=metacontroller_state,
            cms_description=cms_description,
            owner_path=self.owner_path,
            schedule_telemetry=schedule_telemetry,
            description=f"Scheduled joint loop owner={self.owner_path} collected evidence only at turn {turn_index}.",
            substrate_online_fast_due=substrate_online_fast_due,
            rare_heavy_review_recommended=rare_heavy_review_recommended,
            runtime_replay_report=self.latest_runtime_replay_report,
        )

    def _enrich_credit_snapshot(
        self,
        active_snapshots: dict[str, Snapshot[object]],
        *,
        dual_track_rollout: DualTrackRollout,
    ) -> CreditSnapshot:
        credit_snapshot = active_snapshots["credit"].value
        if not isinstance(credit_snapshot, CreditSnapshot):
            raise TypeError("credit snapshot must be CreditSnapshot.")
        return extend_credit_snapshot(
            credit_snapshot=credit_snapshot,
            extra_records=(
                derive_abstract_action_credit(
                    rollout=dual_track_rollout.task_rollout,
                    timestamp_ms=active_snapshots["credit"].timestamp_ms,
                )
                + derive_abstract_action_credit(
                    rollout=dual_track_rollout.relationship_rollout,
                    timestamp_ms=active_snapshots["credit"].timestamp_ms + 100,
                )
            ),
        )

    def _rollback_reasons(
        self,
        *,
        total_reward: float,
        evaluation_snapshot: EvaluationSnapshot,
        optimization_report: DualTrackOptimizationReport,
        metacontroller_state: MetacontrollerRuntimeState | None,
        optimization_evidence_available: bool = True,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        family_signals = self._evaluation_backbone.family_signals(evaluation_snapshot)
        # Do not threshold the aggregate safety-family mean. Safety metrics are
        # heterogeneous readouts (for example, a persona drift *trend* of 0 is
        # healthy, not a 0/1 health score), so averaging them and requiring
        # >=0.85 caused every learning cycle to roll back at the neutral 2/3
        # baseline. Typed HIGH/CRITICAL structured alerts below are the
        # authoritative safety blockers.
        if family_signals.get("relationship", 1.0) < 0.3:
            reasons.append("relationship-critical")
        if any(
            alert.severity in {"HIGH", "CRITICAL"}
            for alert in evaluation_snapshot.structured_alerts
        ):
            reasons.append("evaluation-alert")
        synthetic_surrogate_gate = (
            self._internal_rl_runtime_replay is not WiringLevel.ACTIVE
        )
        if (
            optimization_evidence_available
            and synthetic_surrogate_gate
            and (
                optimization_report.task_report.surrogate_objective < -0.1
                or optimization_report.relationship_report.surrogate_objective
                < -0.1
            )
        ):
            reasons.append("negative-surrogate")
        if optimization_evidence_available and (
            optimization_report.task_report.kl_penalty > 0.4
            or optimization_report.relationship_report.kl_penalty > 0.4
        ):
            reasons.append("excessive-kl")
        surrogate_total = (
            optimization_report.task_report.surrogate_objective
            + optimization_report.relationship_report.surrogate_objective
        )
        outcome_alignment = (
            family_signals.get("task", 0.5) * 0.3
            + family_signals.get("relationship", 0.5) * 0.3
            + family_signals.get("learning", 0.5) * 0.2
            + family_signals.get("abstraction", 0.5) * 0.2
        )
        if (
            optimization_evidence_available
            and synthetic_surrogate_gate
            and surrogate_total > 0.1
            and outcome_alignment < 0.45
        ):
            reasons.append("surrogate-outcome-decoupling")
        if self._metacontroller_drift_exceeds_limit(metacontroller_state):
            reasons.append("metacontroller-drift")
        if self._previous_total_reward is None or not optimization_evidence_available:
            return tuple(reasons)
        if total_reward + 0.30 < self._previous_total_reward:
            reasons.append("reward-regression")
            if surrogate_total > 0.1:
                reasons.append("surrogate-positive-outcome-negative")
        return tuple(reasons)

    def _modulate_ssl_learning_rate(self, evaluation_snapshot: EvaluationSnapshot) -> None:
        """Adjust SSL learning rate based on evaluation quality signals."""
        family_signals = self._evaluation_backbone.family_signals(evaluation_snapshot)
        quality_signal = (
            family_signals.get("learning", 0.5) * 0.4
            + family_signals.get("abstraction", 0.5) * 0.3
            + family_signals.get("safety", 1.0) * 0.3
        )
        base_lr = 0.08
        modulated_lr = base_lr * (0.5 + quality_signal)
        adjusted_lr = max(0.01, min(0.15, modulated_lr))
        self._world_policy.parameter_store.learning_rate = adjusted_lr
        self._self_policy.parameter_store.learning_rate = adjusted_lr

    def _metacontroller_drift_exceeds_limit(
        self,
        metacontroller_state: MetacontrollerRuntimeState | None,
    ) -> bool:
        if metacontroller_state is None or self._previous_metacontroller_state is None:
            return False
        current_temporal = metacontroller_state.temporal_parameters
        previous_temporal = self._previous_metacontroller_state.temporal_parameters
        temporal_shift = max(
            abs(current_temporal.residual_weight - previous_temporal.residual_weight),
            abs(current_temporal.memory_weight - previous_temporal.memory_weight),
            abs(current_temporal.reflection_weight - previous_temporal.reflection_weight),
            abs(current_temporal.switch_bias - previous_temporal.switch_bias),
        )
        persistence_shift = abs(
            metacontroller_state.persistence - self._previous_metacontroller_state.persistence
        )
        current_tracks = dict(metacontroller_state.track_parameters)
        previous_tracks = dict(self._previous_metacontroller_state.track_parameters)
        track_shift = max(
            max(
                abs(current_value - previous_value)
                for current_value, previous_value in zip(current_tracks[track], previous_tracks[track], strict=True)
            )
            for track in current_tracks
        )
        return max(temporal_shift, persistence_shift, track_shift) > 0.35

    def _snapshot_from_trace_step(self, step: TraceStep, trace: TrainingTrace) -> SubstrateSnapshot:
        return SubstrateSnapshot(
            model_id=f"joint-trace:{trace.trace_id}",
            is_frozen=True,
            surface_kind=SurfaceKind.RESIDUAL_STREAM,
            token_logits=tuple(
                min(sum(feature.values) / max(len(feature.values), 1), 1.0)
                for feature in step.feature_surface
            ),
            feature_surface=step.feature_surface,
            residual_activations=step.residual_activations,
            residual_sequence=tuple(
                ResidualSequenceStep(
                    step=prefix_step.step,
                    token=prefix_step.token,
                    feature_surface=prefix_step.feature_surface,
                    residual_activations=prefix_step.residual_activations,
                    description=f"Joint trace token '{prefix_step.token}' at step {prefix_step.step}.",
                )
                for prefix_step in trace.steps[: step.step + 1]
            ),
            unavailable_fields=(),
            description=f"Trace step {step.step} for {trace.trace_id}.",
        )



__all__ = [
    "DefaultContinualLearningSurface",
    "ETANLJointLoop",
    "JointCycleReport",
    "JointLoopSchedule",
    "OnlineFastImportCheckpoint",
    "OnlineFastImportResult",
    "RareHeavyImportCheckpoint",
    "RareHeavyImportResult",
    "RuntimeReplayReport",
    "ScheduledJointLoopResult",
]
