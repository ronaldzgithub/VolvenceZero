"""Directed tests: execution_result evidence enriches the FORWARD task axis.

Packet 1 diagnostic anchor: all-fail trajectories predicted no lower than
all-success ones because ``compute_prediction`` never consumed the
execution_result owner's settled actions. The enrichment is presence-gated:
lanes without settled execution evidence must take the byte-identical
pre-existing path.
"""

from __future__ import annotations

from volvence_zero.prediction.error import (
    _EXECUTION_EVIDENCE_TASK_WEIGHT,
    _build_outcome_evidence,
    _execution_success_signal,
    _OutcomeEvidence,
    _PredictionErrorHead,
    PredictionActionContext,
)
from volvence_zero.semantic_state import ExecutionResultSnapshot, SemanticRecord


def _record(record_id: str, status: str) -> SemanticRecord:
    return SemanticRecord(
        record_id=record_id,
        summary=f"action {record_id}",
        detail=f"detail {record_id}",
        confidence=0.9,
        status=status,
        source_turn=1,
        evidence="test",
    )


def _execution_snapshot(*, completed: int, failed: int) -> ExecutionResultSnapshot:
    completed_records = tuple(_record(f"c{i}", "completed") for i in range(completed))
    failed_records = tuple(_record(f"f{i}", "blocked") for i in range(failed))
    return ExecutionResultSnapshot(
        attempted_actions=(),
        completed_actions=completed_records,
        failed_actions=failed_records,
        artifact_refs=(),
        execution_grounding_score=0.5,
        control_signal=0.5,
        description="test execution snapshot",
    )


def _evidence(execution_signal: float | None) -> _OutcomeEvidence:
    return _OutcomeEvidence(
        family_signals={"task": 0.5, "learning": 0.5, "safety": 0.5},
        substrate_signals={
            "task_pull": 0.5,
            "support_pull": 0.5,
            "repair_pull": 0.5,
            "exploration_pull": 0.5,
            "directive_pull": 0.5,
        },
        previous_substrate_signals={
            "task_pull": 0.5,
            "support_pull": 0.5,
            "repair_pull": 0.5,
            "exploration_pull": 0.5,
            "directive_pull": 0.5,
        },
        substrate_delta={
            "task_shift": 0.5,
            "support_shift": 0.5,
            "residual_shift": 0.5,
            "directive_shift": 0.5,
            "length_delta": 0.5,
        },
        cross_track_tension=0.5,
        regime_stability=0.5,
        execution_signal=execution_signal,
    )


class TestExecutionSuccessSignal:
    def test_none_without_snapshot(self) -> None:
        assert _execution_success_signal(None) is None

    def test_none_without_settled_actions(self) -> None:
        assert _execution_success_signal(_execution_snapshot(completed=0, failed=0)) is None

    def test_ratio(self) -> None:
        assert _execution_success_signal(_execution_snapshot(completed=3, failed=1)) == 0.75
        assert _execution_success_signal(_execution_snapshot(completed=0, failed=4)) == 0.0


class TestForwardTaskEnrichment:
    def test_failures_lower_forward_task_prediction(self) -> None:
        head = _PredictionErrorHead()
        context = PredictionActionContext()
        baseline = head.build_prediction(
            source_turn_index=3, evidence=_evidence(None), action_context=context
        )
        all_fail = head.build_prediction(
            source_turn_index=3, evidence=_evidence(0.0), action_context=context
        )
        all_pass = head.build_prediction(
            source_turn_index=3, evidence=_evidence(1.0), action_context=context
        )
        assert all_fail.predicted_task_progress < baseline.predicted_task_progress
        assert all_pass.predicted_task_progress > baseline.predicted_task_progress
        # The blend weight is the fixed bounded constant, not a free knob.
        expected_fail = baseline.predicted_task_progress * (1.0 - _EXECUTION_EVIDENCE_TASK_WEIGHT)
        assert abs(all_fail.predicted_task_progress - expected_fail) < 1e-9

    def test_absence_is_byte_identical_to_legacy_path(self) -> None:
        head = _PredictionErrorHead()
        context = PredictionActionContext()
        without_field = head.build_prediction(
            source_turn_index=5, evidence=_evidence(None), action_context=context
        )
        assert "execution_signal" not in without_field.description

    def test_other_axes_untouched(self) -> None:
        head = _PredictionErrorHead()
        context = PredictionActionContext()
        baseline = head.build_prediction(
            source_turn_index=3, evidence=_evidence(None), action_context=context
        )
        enriched = head.build_prediction(
            source_turn_index=3, evidence=_evidence(0.0), action_context=context
        )
        assert enriched.predicted_relationship_delta == baseline.predicted_relationship_delta
        assert enriched.predicted_regime_stability == baseline.predicted_regime_stability
        assert enriched.predicted_action_payoff == baseline.predicted_action_payoff
        assert enriched.confidence == baseline.confidence


class TestEvidenceAssembly:
    def test_build_outcome_evidence_threads_execution_signal(self) -> None:
        from types import SimpleNamespace

        # _build_outcome_evidence only reads turn_scores/session_scores and
        # cross_track_tension from these snapshots; lightweight stubs keep
        # the test stable against unrelated snapshot-schema growth.
        evaluation = SimpleNamespace(turn_scores=(), session_scores=())
        dual_track = SimpleNamespace(cross_track_tension=0.4)
        evidence = _build_outcome_evidence(
            substrate_snapshot=None,
            previous_substrate_snapshot=None,
            evaluation_snapshot=evaluation,
            dual_track_snapshot=dual_track,
            regime_snapshot=None,
            execution_result_snapshot=_execution_snapshot(completed=1, failed=3),
        )
        assert evidence.execution_signal == 0.25
        legacy = _build_outcome_evidence(
            substrate_snapshot=None,
            previous_substrate_snapshot=None,
            evaluation_snapshot=evaluation,
            dual_track_snapshot=dual_track,
            regime_snapshot=None,
        )
        assert legacy.execution_signal is None
