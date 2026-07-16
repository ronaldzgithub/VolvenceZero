"""M2 (#89 code side): CMS torch band SHADOW dual-run closure tests.

Covers the code-completion pieces that make the torch band backend "only
evidence away" from ACTIVE:

1. SHADOW dual-run settles a pure-vs-torch update-outcome comparison per
   band update (same pre-weights, same target) and aggregates it.
2. `cms_backend_promotion_readout()` evaluates the documented exit / kill
   conditions in code (min comparisons + parity floor + torch not worse).
3. ACTIVE write-back records a rollback point; `rollback_last_torch_writeback`
   restores the exact pre-update band params (R15 rollback drill).
4. Anti-forgetting hooks aggregate over a bounded window for the gain-curve
   evidence run.

Torch-dependent paths use ``importorskip`` so torch-free environments stay
green (they exercise the DISABLED / no-torch readout branches instead).
"""

from __future__ import annotations

import pytest

from volvence_zero.memory.cms import (
    _CMS_PROMOTION_MIN_COMPARISONS,
    CMSMemoryCore,
)
from volvence_zero.runtime import WiringLevel
from volvence_zero.substrate import (
    FeatureSignal,
    ResidualActivation,
    ResidualSequenceStep,
    SubstrateSnapshot,
    SurfaceKind,
)


def _snapshot(step: int, *, shift: float) -> SubstrateSnapshot:
    feature_surface = (
        FeatureSignal("hook_layer_coverage", (1.0,), "m2-cms"),
        FeatureSignal("fallback_active", (0.0,), "m2-cms"),
        FeatureSignal("semantic_residual_weight", (0.7,), "m2-cms"),
        FeatureSignal("top_logit_margin", (0.6 + shift * 0.1,), "m2-cms"),
        FeatureSignal("top_logit_entropy", (0.2,), "m2-cms"),
        FeatureSignal("semantic_task_pull", (0.5 + shift,), "m2-cms"),
        FeatureSignal("semantic_support_pull", (0.35 + shift * 0.5,), "m2-cms"),
        FeatureSignal("semantic_repair_pull", (0.2,), "m2-cms"),
    )
    residual_activations = (
        ResidualActivation(0, (0.2 + shift, 0.3 + shift, 0.4 + shift), step),
        ResidualActivation(1, (0.3 + shift, 0.2 + shift, 0.1 + shift), step),
    )
    residual_sequence = (
        ResidualSequenceStep(
            step=step,
            token=f"m2-{step}",
            feature_surface=feature_surface,
            residual_activations=residual_activations,
            description="M2 CMS torch closure fixture step.",
        ),
    )
    return SubstrateSnapshot(
        model_id="m2-cms-fixture",
        is_frozen=True,
        surface_kind=SurfaceKind.RESIDUAL_STREAM,
        token_logits=(0.2 + shift, 0.1, 0.05),
        feature_surface=feature_surface,
        residual_activations=residual_activations,
        residual_sequence=residual_sequence,
        unavailable_fields=(),
        description="Deterministic residual-stream fixture for M2 closure.",
    )


def _core(backend: WiringLevel) -> CMSMemoryCore:
    return CMSMemoryCore(
        mode="mlp",
        d_in=8,
        d_hidden=16,
        variant="sequential",
        pe_features_enabled=True,
        replay_window_sizes={"online-fast": 8, "session-medium": 4, "background-slow": 2},
        cms_backend=backend,
    )


def _drive(core: CMSMemoryCore, turns: int) -> None:
    for index in range(1, turns + 1):
        shift = 0.02 * (index % 13)
        core.observe_substrate(
            substrate_snapshot=_snapshot(index, shift=shift),
            timestamp_ms=index,
        )


# ---------------------------------------------------------------------------
# SHADOW dual-run settlement
# ---------------------------------------------------------------------------


def test_shadow_dual_run_settles_update_outcome_comparison() -> None:
    pytest.importorskip("torch")
    core = _core(WiringLevel.SHADOW)
    _drive(core, turns=6)

    evidence = core.latest_cms_backend_evidence
    assert evidence is not None
    assert evidence["backend"] == "shadow"
    assert evidence["wrote_back"] is False
    assert evidence["update_outcome_settled_comparisons"] >= 6
    assert evidence["update_outcome_pure_mse"] >= 0.0
    assert evidence["update_outcome_torch_mse"] >= 0.0
    # Parity fields from the forward dual-run are present alongside.
    assert "forward_parity_within_tolerance" in evidence


def test_disabled_backend_records_no_evidence_or_comparisons() -> None:
    core = _core(WiringLevel.DISABLED)
    _drive(core, turns=6)

    assert core.latest_cms_backend_evidence is None
    readout = core.cms_backend_promotion_readout()
    assert readout.settled_comparisons == 0
    assert readout.promotable is False


# ---------------------------------------------------------------------------
# Promotion readout exit / kill conditions
# ---------------------------------------------------------------------------


def test_promotion_readout_blocks_on_insufficient_comparisons() -> None:
    pytest.importorskip("torch")
    core = _core(WiringLevel.SHADOW)
    _drive(core, turns=4)

    readout = core.cms_backend_promotion_readout()
    assert readout.settled_comparisons < _CMS_PROMOTION_MIN_COMPARISONS
    assert readout.min_comparisons_met is False
    assert readout.promotable is False
    assert "promotable=False" in readout.description


def test_promotion_readout_reports_all_gate_dimensions_after_soak() -> None:
    pytest.importorskip("torch")
    core = _core(WiringLevel.SHADOW)
    # online-fast settles every turn; ~min-comparisons turns is enough.
    _drive(core, turns=_CMS_PROMOTION_MIN_COMPARISONS + 5)

    readout = core.cms_backend_promotion_readout()
    assert readout.backend == "shadow"
    assert readout.torch_available is True
    assert readout.min_comparisons_met is True
    assert readout.parity_checks >= _CMS_PROMOTION_MIN_COMPARISONS
    # The torch forward is exact-parity with the pure forward by design.
    assert readout.parity_floor_met is True
    assert readout.kill_condition_met is False
    # The verdict is whatever the MSE comparison says — assert consistency,
    # not a specific outcome (that is the evidence run's job).
    assert readout.promotable == (
        readout.torch_available
        and readout.min_comparisons_met
        and readout.parity_floor_met
        and readout.torch_not_worse
        and not readout.kill_condition_met
    )


# ---------------------------------------------------------------------------
# ACTIVE write-back rollback drill (R15)
# ---------------------------------------------------------------------------


def test_active_writeback_rollback_restores_pre_update_params() -> None:
    pytest.importorskip("torch")
    core = _core(WiringLevel.ACTIVE)
    _drive(core, turns=1)

    evidence = core.latest_cms_backend_evidence
    assert evidence is not None
    assert evidence["wrote_back"] is True

    before_rollback = core._mlp_for_band("online-fast").export_params()
    core.rollback_last_torch_writeback("online-fast")
    after_rollback = core._mlp_for_band("online-fast").export_params()
    assert after_rollback != before_rollback

    # Rolled-back weights must equal a DISABLED core's pre-update weights:
    # both start from the same deterministic init and saw zero updates on
    # this band before the first observation.
    baseline = _core(WiringLevel.DISABLED)
    assert after_rollback[2] == baseline._mlp_for_band("online-fast").export_params()[2]
    assert after_rollback[3] == baseline._mlp_for_band("online-fast").export_params()[3]


def test_rollback_without_writeback_fails_loudly() -> None:
    core = _core(WiringLevel.SHADOW)
    with pytest.raises(KeyError, match="no torch write-back recorded"):
        core.rollback_last_torch_writeback("online-fast")


def test_rollback_is_single_shot() -> None:
    pytest.importorskip("torch")
    core = _core(WiringLevel.ACTIVE)
    _drive(core, turns=1)
    core.rollback_last_torch_writeback("online-fast")
    with pytest.raises(KeyError, match="no torch write-back recorded"):
        core.rollback_last_torch_writeback("online-fast")


# ---------------------------------------------------------------------------
# Anti-forgetting window hooks
# ---------------------------------------------------------------------------


def test_anti_forgetting_window_aggregates_are_bounded_and_populated() -> None:
    core = _core(WiringLevel.DISABLED)
    _drive(core, turns=10)

    readout = core.cms_backend_promotion_readout()
    assert readout.anti_forgetting_samples == 10
    assert 0.0 <= readout.absorption_window_mean <= 1.0
    assert 0.0 <= readout.retention_window_mean <= 1.0
    # The fixture drives real signal changes; absorption must be non-zero.
    assert readout.absorption_window_mean > 0.0
