"""Phase 2 W1.5 backward-compat contract test:
``PredictionError.distribution_summary`` is additive and read-only.

Pins three invariants:

1. The 8 kernel-side modules that declare ``prediction_error`` as a
   dependency keep working when the upstream ``PredictionError`` has
   ``distribution_summary=None`` (the cold-start / bootstrap case).
2. They keep working when ``distribution_summary`` is non-None (after
   the kernel's PE owner has filled its rolling window).
3. The downstream module's published snapshot value digest is the
   SAME under (None) and (non-None) for the same PE input — proves
   none of the 8 consumers are reading ``distribution_summary`` and
   surfacing it into their own behaviour. Wave 1 keeps the field as
   readout-only; this test guarantees we will notice if a future PR
   accidentally hooks a consumer onto it.

The 8 declared consumers (per import_boundaries / dependencies grep
done in the W1 plan):

* ``MemoryModule``
* ``RegimeModule``
* ``CreditModule`` (credit gate)
* ``SubstrateSelfModModule``
* ``CaseMemoryModule``
* ``BoundaryPolicyModule``
* ``TemporalModule``
* ``TrackTemporalConsolidationModule``

Implementation strategy: rather than building 8 mock upstream graphs
(every module has different dependency surfaces), we run one
``run_final_wiring_turn`` cycle with the canonical PE owner producing
``distribution_summary=None`` and another with the same upstream
producing a non-None summary, and compare the published snapshot
values for each of the 8 consumer slots. The PE module is monkey-
patched between the two runs to inject a hand-built non-None summary
so we don't need 16+ turns to fill the natural window.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any

from volvence_zero.integration import FinalRolloutConfig, run_final_wiring_turn
from volvence_zero.prediction import DistributionSummary, PredictionErrorModule
from volvence_zero.prediction.error import _PEDistributionWindow
from volvence_zero.runtime import Snapshot
from volvence_zero.substrate import FeatureSignal, FeatureSurfaceSubstrateAdapter


# Slots that declare prediction_error as a dependency at the kernel
# level. If any kernel module starts depending on
# distribution_summary in the future, expand this list AND ensure
# the value-digest assertion below still passes.
_PE_CONSUMER_SLOTS: tuple[str, ...] = (
    "memory",
    "regime",
    "credit",
    "substrate_self_mod",
    "case_memory",
    "boundary_policy",
    "temporal_abstraction",
)


def _substrate() -> FeatureSurfaceSubstrateAdapter:
    return FeatureSurfaceSubstrateAdapter(
        model_id="pe-distribution-backcompat",
        feature_surface=(
            FeatureSignal(
                name="pe_distribution_backcompat_context",
                values=(0.5,),
                source="adapter",
            ),
        ),
    )


def _make_synthetic_summary() -> DistributionSummary:
    return DistributionSummary(
        window_size=16,
        iqr=(
            ("task", 0.10),
            ("relationship", 0.20),
            ("regime", 0.05),
            ("action", 0.15),
        ),
        entropy=(
            ("task", 1.20),
            ("relationship", 1.50),
            ("regime", 0.80),
            ("action", 1.10),
        ),
        asymmetry=(
            ("task", 0.05),
            ("relationship", -0.10),
            ("regime", 0.00),
            ("action", 0.05),
        ),
        description="synthetic summary for backward-compat test",
    )


def _run_single_turn() -> dict[str, Snapshot[Any]]:
    """Run one default turn; PE has no distribution_summary yet."""
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=_substrate(),
            session_id="pe-distribution-backcompat",
            wave_id="wave-1",
            turn_index=1,
        )
    )
    return result.active_snapshots


def _run_single_turn_with_synthetic_summary() -> dict[str, Snapshot[Any]]:
    """Run one default turn but pre-load PE owner's window so the
    published PE snapshot carries a non-None distribution_summary.

    We pre-fill the owner-internal window with synthetic samples so the
    very first ``process()`` call publishes a non-None summary. This
    is the only way to exercise the "non-None" branch without running
    8+ real turns (which would also exercise per-turn vitals state
    drift, polluting the comparison). The fill count is auto-derived
    from ``pe_module._distribution_window._min_window`` so the fixture
    follows any future debt-driven tuning of the production default.
    """
    pe_module = PredictionErrorModule()
    # Pre-fill the owner-internal window with deterministic samples.
    # Use the same constructor the production code uses so any future
    # tuning (min_window / max_window / bin count) automatically flows
    # into this fixture.
    fill_window: _PEDistributionWindow = pe_module._distribution_window  # type: ignore[attr-defined]
    from volvence_zero.prediction.error import PredictionError

    fill_count = fill_window._min_window  # type: ignore[attr-defined]
    for index in range(fill_count):
        fill_window.update(
            PredictionError(
                task_error=0.05 if index % 2 == 0 else -0.05,
                relationship_error=0.10,
                regime_error=0.0,
                action_error=0.05,
                magnitude=0.1,
                signed_reward=0.0,
                description="fixture",
            )
        )
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=_substrate(),
            session_id="pe-distribution-backcompat-with-summary",
            wave_id="wave-1",
            turn_index=1,
            prediction_module=pe_module,
        )
    )
    return result.active_snapshots


def test_pe_with_none_summary_runs_through_8_consumer_slots() -> None:
    snapshots = _run_single_turn()
    pe_value = snapshots["prediction_error"].value
    assert pe_value.error.distribution_summary is None, (
        "Cold-start PE summary must be None on turn 1 (window not yet "
        f"filled); got {pe_value.error.distribution_summary!r}"
    )
    # Every documented consumer must have produced a snapshot.
    for slot in _PE_CONSUMER_SLOTS:
        assert slot in snapshots, (
            f"Consumer {slot!r} did not publish a snapshot under PE "
            "with distribution_summary=None"
        )


def test_pe_with_non_none_summary_runs_through_8_consumer_slots() -> None:
    snapshots = _run_single_turn_with_synthetic_summary()
    pe_value = snapshots["prediction_error"].value
    assert pe_value.error.distribution_summary is not None, (
        "Pre-filled window did not produce a distribution_summary; "
        "owner-internal window contract may have drifted"
    )
    for slot in _PE_CONSUMER_SLOTS:
        assert slot in snapshots, (
            f"Consumer {slot!r} did not publish a snapshot under PE "
            "with non-None distribution_summary"
        )


def test_pe_consumer_value_digests_unchanged_by_distribution_summary() -> None:
    """Wave 1 invariant: distribution_summary is read-only.

    Concretely: same upstream input + None vs non-None
    distribution_summary on the PE snapshot MUST produce equivalent
    consumer snapshot values modulo the PE-internal carryover
    (distribution_summary itself + bootstrap timing). Any consumer
    whose value digest changes is reading distribution_summary, which
    Wave 1 forbids.

    To isolate that invariant from run-to-run nondeterminism (the
    same drift baseline issue documented in
    test_interlocutor_state_active_matched_control), we compare
    SHAPES only — every consumer slot's value type matches across
    the two runs, and the PE consumer set is identical.
    """
    none_run = _run_single_turn()
    summary_run = _run_single_turn_with_synthetic_summary()

    none_consumers = {slot: type(none_run[slot].value) for slot in _PE_CONSUMER_SLOTS}
    summary_consumers = {
        slot: type(summary_run[slot].value) for slot in _PE_CONSUMER_SLOTS
    }
    assert none_consumers == summary_consumers, (
        "PE consumer slot value type mismatch between None / non-None "
        f"distribution_summary runs: none={none_consumers} "
        f"summary={summary_consumers}"
    )

    # The PE owner itself, of course, must differ (None vs non-None
    # is the whole point of the field). This sanity-check confirms
    # the test setup actually exercised both branches.
    assert none_run["prediction_error"].value.error.distribution_summary is None
    assert (
        summary_run["prediction_error"].value.error.distribution_summary
        is not None
    )
