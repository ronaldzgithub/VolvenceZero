from __future__ import annotations

from volvence_zero.memory.store import build_default_memory_store
from volvence_zero.substrate import (
    FeatureSignal,
    ResidualActivation,
    ResidualSequenceStep,
    SubstrateSnapshot,
    SurfaceKind,
)


def _snapshot(step: int, *, shift: float) -> SubstrateSnapshot:
    feature_surface = (
        FeatureSignal("hook_layer_coverage", (1.0,), "cms-ab"),
        FeatureSignal("fallback_active", (0.0,), "cms-ab"),
        FeatureSignal("semantic_residual_weight", (0.7,), "cms-ab"),
        FeatureSignal("top_logit_margin", (0.6 + shift * 0.1,), "cms-ab"),
        FeatureSignal("top_logit_entropy", (0.2,), "cms-ab"),
        FeatureSignal("semantic_task_pull", (0.5 + shift,), "cms-ab"),
        FeatureSignal("semantic_support_pull", (0.35 + shift * 0.5,), "cms-ab"),
        FeatureSignal("semantic_repair_pull", (0.2,), "cms-ab"),
    )
    residual_activations = (
        ResidualActivation(0, (0.2 + shift, 0.3 + shift, 0.4 + shift), step),
        ResidualActivation(1, (0.3 + shift, 0.2 + shift, 0.1 + shift), step),
    )
    residual_sequence = (
        ResidualSequenceStep(
            step=step,
            token=f"cms-{step}",
            feature_surface=feature_surface,
            residual_activations=residual_activations,
            description="CMS anti-forgetting A/B fixture step.",
        ),
    )
    return SubstrateSnapshot(
        model_id="cms-ab-fixture",
        is_frozen=True,
        surface_kind=SurfaceKind.RESIDUAL_STREAM,
        token_logits=(0.2 + shift, 0.1, 0.05),
        feature_surface=feature_surface,
        residual_activations=residual_activations,
        residual_sequence=residual_sequence,
        unavailable_fields=(),
        description="Deterministic residual-stream fixture for CMS A/B evidence.",
    )


def _drive_store(*, uplift: bool) -> tuple[float, float, bool, bool]:
    store = build_default_memory_store(
        latent_dim=8,
        cms_pe_features_enabled=uplift,
        cms_replay_window_size=8 if uplift else None,
    )
    shifts = (0.0, 0.02, 0.04, 0.06, 0.55, 0.58, 0.6, 0.62)
    for index, shift in enumerate(shifts, start=1):
        store.observe_substrate(
            substrate_snapshot=_snapshot(index, shift=shift),
            timestamp_ms=index,
        )
    cms_state = store.snapshot(retrieved_entries=()).cms_state
    assert cms_state is not None
    return (
        cms_state.old_knowledge_retention,
        cms_state.new_knowledge_absorption,
        cms_state.titans_pe_gate_active,
        cms_state.atlas_replay_active,
    )


def test_cms_uplift_reports_anti_forgetting_metrics_and_rollback() -> None:
    uplift_retention, uplift_absorption, pe_gate_active, replay_active = _drive_store(
        uplift=True
    )
    rollback_retention, rollback_absorption, pe_gate_rollback, replay_rollback = (
        _drive_store(uplift=False)
    )

    assert pe_gate_active is True
    assert replay_active is True
    assert pe_gate_rollback is False
    assert replay_rollback is False
    assert 0.0 <= uplift_retention <= 1.0
    assert 0.0 <= uplift_absorption <= 1.0
    assert 0.0 <= rollback_retention <= 1.0
    assert 0.0 <= rollback_absorption <= 1.0
    assert uplift_absorption > 0.0
    assert rollback_absorption > 0.0
    assert uplift_retention >= rollback_retention
