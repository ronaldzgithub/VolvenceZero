"""Rare-heavy and online-fast import/rollback mixin for ETANLJointLoop."""

from __future__ import annotations

from volvence_zero.joint_loop.contracts import (
    OnlineFastImportCheckpoint,
    OnlineFastImportResult,
    RareHeavyImportCheckpoint,
    RareHeavyImportResult,
)
from volvence_zero.joint_loop.pipeline import RareHeavyArtifact
from volvence_zero.substrate import SubstrateOnlineFastCheckpoint
from volvence_zero.temporal import DualTrackRareHeavySnapshot


class _JointLoopArtifactImportMixin:
    def _require_live_substrate_mutation_enabled(self, *, operation: str) -> None:
        if self._residual_runtime is None:
            raise RuntimeError(f"{operation} requires a residual runtime.")
        self._residual_runtime.require_live_substrate_mutation(operation=operation)

    def apply_rare_heavy_artifact(
        self,
        artifact: RareHeavyArtifact,
        *,
        checkpoint_id: str | None = None,
    ) -> RareHeavyImportResult:
        self._require_live_substrate_mutation_enabled(operation="apply_rare_heavy_artifact()")
        checkpoint_label = checkpoint_id or f"rare-heavy:{artifact.artifact_id}"
        world_policy_checkpoint = self._world_sandbox.create_checkpoint(
            checkpoint_id=f"{checkpoint_label}:world-policy",
            include_runtime_replay=True,
        )
        self_policy_checkpoint = self._self_sandbox.create_checkpoint(
            checkpoint_id=f"{checkpoint_label}:self-policy",
            include_runtime_replay=True,
        )
        world_temporal_snapshot = self._world_policy.export_rare_heavy_snapshot()
        self_temporal_snapshot = self._self_policy.export_rare_heavy_snapshot()
        memory_checkpoint = self._memory_store.export_rare_heavy_state(checkpoint_id=f"{checkpoint_label}:memory")
        substrate_checkpoint = (
            self._residual_runtime.export_rare_heavy_state(checkpoint_id=f"{checkpoint_label}:substrate")
            if self._residual_runtime is not None and artifact.substrate_checkpoint is not None
            else None
        )
        if isinstance(artifact.temporal_snapshot, DualTrackRareHeavySnapshot):
            applied_operations = self._world_policy.apply_rare_heavy_snapshot(
                artifact.temporal_snapshot.world_snapshot
            ) + self._self_policy.apply_rare_heavy_snapshot(
                artifact.temporal_snapshot.self_snapshot
            )
        else:
            applied_operations = self._world_policy.apply_rare_heavy_snapshot(artifact.temporal_snapshot)
            applied_operations = applied_operations + self._self_policy.apply_rare_heavy_snapshot(
                artifact.temporal_snapshot
            )
        if artifact.memory_checkpoint is not None:
            applied_operations = applied_operations + self._memory_store.import_rare_heavy_state(
                artifact.memory_checkpoint
            )
        if artifact.substrate_checkpoint is not None and self._residual_runtime is not None:
            applied_operations = applied_operations + self._residual_runtime.import_rare_heavy_state(
                artifact.substrate_checkpoint
            )
        checkpoint = RareHeavyImportCheckpoint(
            artifact_id=artifact.artifact_id,
            world_policy_checkpoint=world_policy_checkpoint,
            self_policy_checkpoint=self_policy_checkpoint,
            world_temporal_snapshot=world_temporal_snapshot,
            self_temporal_snapshot=self_temporal_snapshot,
            memory_checkpoint=memory_checkpoint,
            substrate_checkpoint=substrate_checkpoint,
            pending_task_rollouts=tuple(self._pending_task_rollouts),
            pending_relationship_rollouts=tuple(
                self._pending_relationship_rollouts
            ),
            runtime_replay_report=self.latest_runtime_replay_report,
            open_task_segment_rollouts=tuple(
                self._open_task_segment_rollouts
            ),
            open_relationship_segment_rollouts=tuple(
                self._open_relationship_segment_rollouts
            ),
            runtime_segment_closed_count=(
                self._runtime_segment_closed_count
            ),
            runtime_longest_segment_length=(
                self._runtime_longest_segment_length
            ),
        )
        return RareHeavyImportResult(
            artifact_id=artifact.artifact_id,
            applied_operations=applied_operations,
            checkpoint=checkpoint,
            description=(
                f"Applied rare-heavy artifact {artifact.artifact_id} from {artifact.owner_path} "
                f"with {len(applied_operations)} owner-side imports."
            ),
        )

    def review_rare_heavy_artifact(
        self,
        artifact: RareHeavyArtifact,
        *,
        checkpoint_id: str | None = None,
    ) -> RareHeavyImportResult:
        checkpoint_label = checkpoint_id or f"rare-heavy-review:{artifact.artifact_id}"
        checkpoint = RareHeavyImportCheckpoint(
            artifact_id=artifact.artifact_id,
            world_policy_checkpoint=self._world_sandbox.create_checkpoint(
                checkpoint_id=f"{checkpoint_label}:world-policy",
                include_runtime_replay=True,
            ),
            self_policy_checkpoint=self._self_sandbox.create_checkpoint(
                checkpoint_id=f"{checkpoint_label}:self-policy",
                include_runtime_replay=True,
            ),
            world_temporal_snapshot=self._world_policy.export_rare_heavy_snapshot(),
            self_temporal_snapshot=self._self_policy.export_rare_heavy_snapshot(),
            memory_checkpoint=self._memory_store.export_rare_heavy_state(
                checkpoint_id=f"{checkpoint_label}:memory"
            ),
            substrate_checkpoint=None,
            pending_task_rollouts=tuple(self._pending_task_rollouts),
            pending_relationship_rollouts=tuple(
                self._pending_relationship_rollouts
            ),
            runtime_replay_report=self.latest_runtime_replay_report,
            open_task_segment_rollouts=tuple(
                self._open_task_segment_rollouts
            ),
            open_relationship_segment_rollouts=tuple(
                self._open_relationship_segment_rollouts
            ),
            runtime_segment_closed_count=(
                self._runtime_segment_closed_count
            ),
            runtime_longest_segment_length=(
                self._runtime_longest_segment_length
            ),
        )
        return RareHeavyImportResult(
            artifact_id=artifact.artifact_id,
            applied_operations=(),
            checkpoint=checkpoint,
            description=(
                f"Reviewed rare-heavy artifact {artifact.artifact_id} in review-only mode; "
                "no owner-side imports were applied because the runtime is staying under the "
                "frozen-substrate doctrine."
            ),
        )

    def rollback_rare_heavy_import(self, checkpoint: RareHeavyImportCheckpoint) -> tuple[str, ...]:
        self._world_sandbox.restore_checkpoint(checkpoint.world_policy_checkpoint)
        self._self_sandbox.restore_checkpoint(checkpoint.self_policy_checkpoint)
        self._world_policy.apply_rare_heavy_snapshot(checkpoint.world_temporal_snapshot)
        self._self_policy.apply_rare_heavy_snapshot(checkpoint.self_temporal_snapshot)
        self._memory_store.restore_checkpoint(checkpoint.memory_checkpoint)
        if checkpoint.pending_task_rollouts is not None:
            self._pending_task_rollouts = list(checkpoint.pending_task_rollouts)
        if checkpoint.pending_relationship_rollouts is not None:
            self._pending_relationship_rollouts = list(
                checkpoint.pending_relationship_rollouts
            )
        if checkpoint.open_task_segment_rollouts is not None:
            self._open_task_segment_rollouts = list(
                checkpoint.open_task_segment_rollouts
            )
        if checkpoint.open_relationship_segment_rollouts is not None:
            self._open_relationship_segment_rollouts = list(
                checkpoint.open_relationship_segment_rollouts
            )
        self._runtime_segment_closed_count = (
            checkpoint.runtime_segment_closed_count
        )
        self._runtime_longest_segment_length = (
            checkpoint.runtime_longest_segment_length
        )
        if checkpoint.runtime_replay_report is not None:
            replay_report = checkpoint.runtime_replay_report
            self._runtime_replay_captured_count = replay_report.captured_count
            self._runtime_replay_settled_count = replay_report.settled_count
            self._runtime_replay_transition_count = replay_report.transition_count
            self._runtime_replay_lineage_match_count = (
                replay_report.lineage_match_count
            )
            self._runtime_replay_drop_reasons = list(
                replay_report.drop_reasons
            )
            self._runtime_replay_outcome_lineages = list(
                replay_report.outcome_lineages
            )
            self._runtime_last_segment_close_reason = (
                replay_report.last_segment_close_reason
            )
            self._runtime_segment_close_reason_counts = dict(
                replay_report.segment_close_reason_counts
            )
        operations = [
            "rare-heavy:world-temporal-rollback",
            "rare-heavy:self-temporal-rollback",
            "rare-heavy:memory-rollback",
        ]
        if checkpoint.substrate_checkpoint is not None and self._residual_runtime is not None:
            operations.extend(self._residual_runtime.restore_rare_heavy_state(checkpoint.substrate_checkpoint))
        return tuple(operations)

    def apply_online_fast_substrate_checkpoint(
        self,
        checkpoint: SubstrateOnlineFastCheckpoint,
        *,
        checkpoint_id: str | None = None,
    ) -> OnlineFastImportResult:
        self._require_live_substrate_mutation_enabled(operation="apply_online_fast_substrate_checkpoint()")
        prior_checkpoint = (
            self._residual_runtime.export_online_fast_state(
                checkpoint_id=checkpoint_id or f"{checkpoint.checkpoint_id}:prior"
            )
            if self._residual_runtime is not None
            else None
        )
        applied_operations = (
            self._residual_runtime.apply_online_fast_state(checkpoint)
            if self._residual_runtime is not None
            else ()
        )
        return OnlineFastImportResult(
            applied_operations=applied_operations,
            checkpoint=OnlineFastImportCheckpoint(
                checkpoint_id=checkpoint_id or checkpoint.checkpoint_id,
                substrate_checkpoint=prior_checkpoint,
            ),
            description=(
                f"Applied online-fast substrate checkpoint {checkpoint.checkpoint_id} "
                f"with {len(applied_operations)} owner-side operations."
            ),
        )

    def rollback_online_fast_substrate_import(self, checkpoint: OnlineFastImportCheckpoint) -> tuple[str, ...]:
        if checkpoint.substrate_checkpoint is None or self._residual_runtime is None:
            return ()
        return self._residual_runtime.restore_online_fast_state(checkpoint.substrate_checkpoint)
