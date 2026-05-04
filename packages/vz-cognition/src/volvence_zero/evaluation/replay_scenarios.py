"""Default replay scenarios used by the cross-session benchmark suite.

``_default_evolution_benchmark_cases`` is the fixed six-case replay
suite that :class:`EvaluationBackbone.build_cross_session_benchmark_suite`
runs against a candidate controller to extract a :class:`LongitudinalReport`.
``_feature_surface_snapshot`` is the supporting builder that produces a
deterministic :class:`SubstrateSnapshot` from a small set of pressure-axis
pulls.

Slice S.2 (2026-05-04): extracted from ``evaluation/backbone.py``.
"""

from __future__ import annotations

from volvence_zero.substrate import FeatureSignal, SubstrateSnapshot, SurfaceKind

from volvence_zero.evaluation.types import EvaluationReplayCase


def _feature_surface_snapshot(
    *,
    model_id: str,
    task_pull: float,
    support_pull: float,
    repair_pull: float,
    exploration_pull: float,
    directive_pull: float,
) -> SubstrateSnapshot:
    return SubstrateSnapshot(
        model_id=model_id,
        is_frozen=True,
        surface_kind=SurfaceKind.FEATURE_SURFACE,
        token_logits=(),
        feature_surface=(
            FeatureSignal(name="semantic_task_pull", values=(task_pull,), source="evolution-benchmark"),
            FeatureSignal(name="semantic_support_pull", values=(support_pull,), source="evolution-benchmark"),
            FeatureSignal(name="semantic_repair_pull", values=(repair_pull,), source="evolution-benchmark"),
            FeatureSignal(name="semantic_exploration_pull", values=(exploration_pull,), source="evolution-benchmark"),
            FeatureSignal(name="semantic_directive_pull", values=(directive_pull,), source="evolution-benchmark"),
            FeatureSignal(name="fallback_active", values=(0.0,), source="evolution-benchmark"),
        ),
        residual_activations=(),
        residual_sequence=(),
        unavailable_fields=(),
        description="evolution benchmark substrate snapshot",
    )


def _default_evolution_benchmark_cases() -> tuple[EvaluationReplayCase, ...]:
    return (
        EvaluationReplayCase(
            case_id="task-dominant",
            session_id="benchmark-task",
            wave_id="wave-1",
            substrate_snapshot=_feature_surface_snapshot(
                model_id="benchmark-task",
                task_pull=0.88,
                support_pull=0.24,
                repair_pull=0.20,
                exploration_pull=0.18,
                directive_pull=0.72,
            ),
            memory_snapshot=None,
            dual_track_snapshot=None,
            metric_floors=(("task_pressure", 0.45), ("contract_integrity", 0.95)),
            max_alert_count=0,
        ),
        EvaluationReplayCase(
            case_id="support-dominant",
            session_id="benchmark-support",
            wave_id="wave-1",
            substrate_snapshot=_feature_surface_snapshot(
                model_id="benchmark-support",
                task_pull=0.22,
                support_pull=0.86,
                repair_pull=0.52,
                exploration_pull=0.20,
                directive_pull=0.12,
            ),
            memory_snapshot=None,
            dual_track_snapshot=None,
            metric_floors=(("support_presence", 0.40), ("warmth", 0.35)),
            max_alert_count=0,
        ),
        EvaluationReplayCase(
            case_id="mixed-conflict",
            session_id="benchmark-mixed",
            wave_id="wave-1",
            substrate_snapshot=_feature_surface_snapshot(
                model_id="benchmark-mixed",
                task_pull=0.58,
                support_pull=0.56,
                repair_pull=0.64,
                exploration_pull=0.44,
                directive_pull=0.28,
            ),
            memory_snapshot=None,
            dual_track_snapshot=None,
            metric_floors=(("cross_track_stability", 0.30), ("contract_integrity", 0.95)),
            max_alert_count=0,
        ),
        EvaluationReplayCase(
            case_id="family-monopoly",
            session_id="benchmark-monopoly",
            wave_id="wave-1",
            substrate_snapshot=_feature_surface_snapshot(
                model_id="benchmark-monopoly",
                task_pull=0.50,
                support_pull=0.50,
                repair_pull=0.30,
                exploration_pull=0.50,
                directive_pull=0.30,
            ),
            memory_snapshot=None,
            dual_track_snapshot=None,
            metric_floors=(("cross_track_stability", 0.30), ("contract_integrity", 0.95)),
            max_alert_count=1,
        ),
        EvaluationReplayCase(
            case_id="family-collapse",
            session_id="benchmark-collapse",
            wave_id="wave-1",
            substrate_snapshot=_feature_surface_snapshot(
                model_id="benchmark-collapse",
                task_pull=0.40,
                support_pull=0.40,
                repair_pull=0.40,
                exploration_pull=0.40,
                directive_pull=0.40,
            ),
            memory_snapshot=None,
            dual_track_snapshot=None,
            metric_floors=(("cross_track_stability", 0.30), ("contract_integrity", 0.95)),
            max_alert_count=1,
        ),
        EvaluationReplayCase(
            case_id="continuity-long-horizon",
            session_id="benchmark-continuity",
            wave_id="wave-1",
            substrate_snapshot=_feature_surface_snapshot(
                model_id="benchmark-continuity",
                task_pull=0.44,
                support_pull=0.62,
                repair_pull=0.34,
                exploration_pull=0.52,
                directive_pull=0.18,
            ),
            memory_snapshot=None,
            dual_track_snapshot=None,
            metric_floors=(("info_integration", 0.10), ("warmth", 0.30)),
            max_alert_count=0,
        ),
    )
