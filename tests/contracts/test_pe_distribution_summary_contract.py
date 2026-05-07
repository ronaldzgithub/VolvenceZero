"""Phase 2 W1.5 contract tests: PE distribution summary invariants.

Pins the four invariants documented in
``docs/specs/prediction-error-loop.md`` -> "PE Distributional Readout":

1. ``DistributionSummary`` lookup helpers correctly handle missing axes.
2. ``_PEDistributionWindow`` returns ``None`` until ``min_window``
   non-bootstrap samples have been observed (cold-start safety).
3. The window summary is deterministic: same sample sequence -> same
   summary, byte-for-byte.
4. IQR is monotonic in distribution width: a wide-uniform sample
   stream yields strictly larger IQR than a narrow-clustered one.

The window class is owner-internal; we import directly to test it
without going through the full ``PredictionErrorModule.process``
plumbing (which already has end-to-end coverage in
``tests/test_prediction_error.py``).
"""

from __future__ import annotations

import dataclasses
import random

from volvence_zero.prediction import DistributionSummary
from volvence_zero.prediction.error import _PEDistributionWindow, PredictionError


def _make_error(
    *,
    task: float,
    relationship: float,
    regime: float,
    action: float,
) -> PredictionError:
    return PredictionError(
        task_error=task,
        relationship_error=relationship,
        regime_error=regime,
        action_error=action,
        magnitude=0.1,
        signed_reward=0.0,
        description="test",
    )


def test_distribution_summary_axis_lookup_handles_missing_axis() -> None:
    summary = DistributionSummary(
        window_size=16,
        iqr=(("task", 0.1), ("relationship", 0.2)),
        entropy=(("task", 1.0), ("relationship", 1.5)),
        asymmetry=(("task", 0.0), ("relationship", -0.1)),
        description="test",
    )
    assert summary.axis_iqr("task") == 0.1
    assert summary.axis_iqr("relationship") == 0.2
    assert summary.axis_iqr("regime") is None
    assert summary.axis_entropy("regime") is None
    assert summary.axis_asymmetry("regime") is None


def test_distribution_window_returns_none_below_min_window() -> None:
    """Production default ``min_window=8`` (post debt #11 close-out)."""
    window = _PEDistributionWindow(min_window=8, max_window=64)
    assert window.summarise() is None
    for i in range(7):
        window.update(
            _make_error(task=0.1, relationship=0.0, regime=0.0, action=0.0)
        )
    assert window.summarise() is None, "still below min_window=8"


def test_distribution_window_emits_summary_at_min_window() -> None:
    """Production default ``min_window=8`` (post debt #11 close-out)."""
    window = _PEDistributionWindow(min_window=8, max_window=64)
    for i in range(8):
        window.update(
            _make_error(
                task=0.1 if i % 2 == 0 else -0.1,
                relationship=0.05,
                regime=0.0,
                action=0.0,
            )
        )
    summary = window.summarise()
    assert summary is not None
    assert summary.window_size == 8
    # All four axes present in all three statistics.
    iqr_axes = {name for name, _ in summary.iqr}
    entropy_axes = {name for name, _ in summary.entropy}
    asymmetry_axes = {name for name, _ in summary.asymmetry}
    assert iqr_axes == {"task", "relationship", "regime", "action"}
    assert entropy_axes == iqr_axes
    assert asymmetry_axes == iqr_axes


def test_distribution_window_is_deterministic() -> None:
    """Same sample stream -> byte-for-byte identical summary."""
    samples_a = []
    samples_b = []
    rng_a = random.Random(42)
    rng_b = random.Random(42)
    for _ in range(20):
        samples_a.append(
            _make_error(
                task=rng_a.uniform(-0.3, 0.3),
                relationship=rng_a.uniform(-0.5, 0.5),
                regime=rng_a.uniform(-0.2, 0.2),
                action=rng_a.uniform(-0.4, 0.4),
            )
        )
        samples_b.append(
            _make_error(
                task=rng_b.uniform(-0.3, 0.3),
                relationship=rng_b.uniform(-0.5, 0.5),
                regime=rng_b.uniform(-0.2, 0.2),
                action=rng_b.uniform(-0.4, 0.4),
            )
        )
    window_a = _PEDistributionWindow(min_window=16, max_window=64)
    window_b = _PEDistributionWindow(min_window=16, max_window=64)
    for err in samples_a:
        window_a.update(err)
    for err in samples_b:
        window_b.update(err)
    summary_a = window_a.summarise()
    summary_b = window_b.summarise()
    assert summary_a is not None
    assert summary_b is not None
    assert dataclasses.asdict(summary_a) == dataclasses.asdict(summary_b)


def test_distribution_window_iqr_is_monotonic_in_width() -> None:
    """Wide-uniform samples yield strictly larger IQR than narrow-clustered.

    Constructs two streams with the same mean (zero) but different
    spread on the ``task`` axis, then asserts the wide stream's IQR
    is at least 2x the narrow stream's IQR. Monotonicity is what
    makes IQR usable as a distribution-width signal.
    """
    narrow = _PEDistributionWindow(min_window=16, max_window=64)
    wide = _PEDistributionWindow(min_window=16, max_window=64)
    rng = random.Random(7)
    for _ in range(32):
        narrow.update(
            _make_error(
                task=rng.uniform(-0.05, 0.05),
                relationship=0.0,
                regime=0.0,
                action=0.0,
            )
        )
        wide.update(
            _make_error(
                task=rng.uniform(-0.5, 0.5),
                relationship=0.0,
                regime=0.0,
                action=0.0,
            )
        )
    narrow_summary = narrow.summarise()
    wide_summary = wide.summarise()
    assert narrow_summary is not None
    assert wide_summary is not None
    narrow_iqr = narrow_summary.axis_iqr("task")
    wide_iqr = wide_summary.axis_iqr("task")
    assert narrow_iqr is not None
    assert wide_iqr is not None
    assert wide_iqr >= 2.0 * narrow_iqr, (
        f"IQR not monotonic in width: narrow={narrow_iqr} wide={wide_iqr}"
    )


def test_distribution_window_max_window_bounds_memory() -> None:
    """Window capped at ``max_window``: extra samples drop the oldest."""
    window = _PEDistributionWindow(min_window=16, max_window=20)
    for _ in range(100):
        window.update(
            _make_error(task=0.1, relationship=0.0, regime=0.0, action=0.0)
        )
    summary = window.summarise()
    assert summary is not None
    assert summary.window_size == 20, (
        "window must cap at max_window even after many updates"
    )


def test_distribution_summary_asymmetry_clamped_to_unit_interval() -> None:
    """Even pathological one-sided streams yield ``asymmetry`` in ``[-1, 1]``."""
    window = _PEDistributionWindow(min_window=16, max_window=64)
    # All-positive task errors: skew should saturate at +1, not blow up.
    for i in range(32):
        window.update(
            _make_error(
                task=0.01 + i * 0.005,  # monotonic positive
                relationship=0.0,
                regime=0.0,
                action=0.0,
            )
        )
    summary = window.summarise()
    assert summary is not None
    asymmetry = summary.axis_asymmetry("task")
    assert asymmetry is not None
    assert -1.0 <= asymmetry <= 1.0


def test_distribution_window_iqr_stable_at_min_window_n8() -> None:
    """Phase 2 W4 (debt #11 close-out): n=8 IQR is statistically usable.

    Replays a deterministic stream of 32 samples drawn from a bounded
    uniform distribution into two windows: one capped at 8, one
    capped at 32. Asserts the per-axis IQR ratio (n=8 / n=32) stays
    within ``[0.4, 2.5]`` — i.e. the 8-sample IQR estimate is within
    roughly a factor of 2 of the 32-sample reference. The band is
    asymmetric on purpose: the IQR's sampling distribution at small n
    is right-skewed (more frequent under-estimation, occasional
    large over-estimation), and the canonical SE-of-IQR result
    ``SE(IQR) ~= 1.36 * sigma / sqrt(n)`` predicts roughly 2x more
    variability at n=8 vs n=32. The band reflects that statistical
    truth without becoming a free pass — a 5x or 10x ratio would
    still trip.

    Two windows are independent ``_PEDistributionWindow`` instances,
    not the same instance summarised at two checkpoints, so the test
    captures "8 freshly-collected samples" vs "32 freshly-collected
    samples" — the failure mode that matters operationally (a
    real session may only ever see 8 turns).

    Pins the statistical justification for the production default
    ``min_window=8`` chosen during the debt #11 close-out, evidenced
    by ``artifacts/eq_uplift/pe_window_long_form.json``.
    """
    rng_a = random.Random(99)
    rng_b = random.Random(99)
    window_n8 = _PEDistributionWindow(min_window=8, max_window=8)
    window_n32 = _PEDistributionWindow(min_window=32, max_window=32)

    # Same deterministic stream into both windows. n8 holds last 8;
    # n32 holds all 32. After 32 samples the overlap is the most
    # recent 8 — the question is "are the 8-sample stats usable".
    for _ in range(32):
        err_a = _make_error(
            task=rng_a.uniform(-0.3, 0.3),
            relationship=rng_a.uniform(-0.5, 0.5),
            regime=rng_a.uniform(-0.2, 0.2),
            action=rng_a.uniform(-0.4, 0.4),
        )
        # Identical samples via paired RNG seed (rng_b advances in
        # lockstep below). Avoids relying on PredictionError __eq__.
        err_b = _make_error(
            task=rng_b.uniform(-0.3, 0.3),
            relationship=rng_b.uniform(-0.5, 0.5),
            regime=rng_b.uniform(-0.2, 0.2),
            action=rng_b.uniform(-0.4, 0.4),
        )
        window_n8.update(err_a)
        window_n32.update(err_b)

    summary_n8 = window_n8.summarise()
    summary_n32 = window_n32.summarise()
    assert summary_n8 is not None
    assert summary_n32 is not None
    assert summary_n8.window_size == 8
    assert summary_n32.window_size == 32

    for axis in ("task", "relationship", "regime", "action"):
        iqr_n8 = summary_n8.axis_iqr(axis)
        iqr_n32 = summary_n32.axis_iqr(axis)
        assert iqr_n8 is not None
        assert iqr_n32 is not None
        # If the 32-sample IQR is essentially zero, the 8-sample IQR
        # should also be essentially zero (otherwise the n=8 window is
        # picking up artificial spread). Use absolute tolerance.
        if iqr_n32 < 1e-3:
            assert iqr_n8 < 5e-3, (
                f"axis={axis} n8 IQR={iqr_n8:.4f} should be near-zero "
                f"when n32 IQR={iqr_n32:.4f}"
            )
            continue
        ratio = iqr_n8 / iqr_n32
        assert 0.4 <= ratio <= 2.5, (
            f"axis={axis} iqr_n8/iqr_n32={ratio:.3f} (n8={iqr_n8:.4f} "
            f"n32={iqr_n32:.4f}) outside stability band [0.4, 2.5]; "
            "8-sample IQR is too noisy to back the production "
            "min_window=8 default"
        )
