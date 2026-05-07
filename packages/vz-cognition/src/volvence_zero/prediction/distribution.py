"""Phase 2 W1.1 (DM-1) — PE distributional readout contract.

Adds a thin distributional shape descriptor to the prediction-error
chain. Keeps the existing 4-axis scalar fields on
:class:`PredictionError` exactly as before; the new ``distribution_summary``
slot is purely additive and optional.

Design constraints (from ``docs/specs/prediction-error-loop.md``):

* **None-safe cold start** — ``DistributionSummary`` is published only
  after a minimum window of PE samples has been observed by the PE
  owner. Callers MUST treat ``distribution_summary is None`` as the
  default and not synthesise stand-in values.
* **Read-only** — downstream consumers (``credit`` gate, ``regime``
  scoring, ``ModificationGate``, ``memory`` retrieval signal) MUST NOT
  consume ``distribution_summary`` to drive control decisions in
  Wave 1. Wave 1's only legitimate downstream is the ``vitals`` slot
  (``VitalsSnapshot.distributional_drift_axes``); audit / evaluation
  surfaces may read it as readout.
* **Owner-internal constants** — window size, bin count, IQR
  percentiles are owner-internal. Other modules MUST NOT depend on
  the specific window/bin layout; the public contract is the three
  per-axis statistics (IQR / entropy / asymmetry) plus the
  ``window_size`` provenance hint.

This module follows the Botvinick et al. 2025 *Depression as a
disorder of distributional coding* line of reasoning: scalar PE
collapses to a single mean even when the underlying distribution
shape carries the relevant signal (depression-like collapse,
rupture-aware widening, asymmetric drift toward negative outcomes).
By making the per-axis distribution shape an explicit part of the
PE owner snapshot, downstream evaluators can detect those failure
modes without re-reading raw samples.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DistributionSummary:
    """Per-axis distribution shape descriptors over a recent PE window.

    Three robust statistics per axis (``task`` / ``relationship`` /
    ``regime`` / ``action``). All values are bounded; tuples preserve
    insertion order so consumers can grep by axis name.

    * ``iqr`` — inter-quartile range of |axis_error| over the window.
      Signal: distribution width. Near-zero IQR after a healthy run
      indicates a *collapsing* distribution (the distributional-coding
      analogue of mode collapse / rumination).
    * ``entropy`` — discretised entropy of (sign, magnitude-bin) pairs
      over the window. Signal: distribution evenness. Low entropy
      with non-zero IQR indicates the system is locking into a single
      response mode without the variance shrinking, i.e. early
      one-sided drift before total collapse.
    * ``asymmetry`` — signed skew proxy ``(mean - median) / (iqr + eps)``.
      Signal: directional drift. Positive = right-tailed (rare large
      errors); negative = left-tailed (rare unusually-positive
      outcomes). The PE owner MUST clamp to ``[-1, 1]`` to keep
      consumers' downstream readouts bounded.
    """

    window_size: int
    iqr: tuple[tuple[str, float], ...]
    entropy: tuple[tuple[str, float], ...]
    asymmetry: tuple[tuple[str, float], ...]
    description: str = ""

    def axis_iqr(self, axis: str) -> float | None:
        """Return the IQR for ``axis`` or ``None`` if not present."""
        return _lookup(self.iqr, axis)

    def axis_entropy(self, axis: str) -> float | None:
        """Return the entropy for ``axis`` or ``None`` if not present."""
        return _lookup(self.entropy, axis)

    def axis_asymmetry(self, axis: str) -> float | None:
        """Return the signed asymmetry for ``axis`` or ``None``."""
        return _lookup(self.asymmetry, axis)


def _lookup(pairs: tuple[tuple[str, float], ...], axis: str) -> float | None:
    """Linear scan over per-axis tuples.

    Per-axis lists are tiny (4 axes); keeping a tuple-of-tuples
    representation keeps :class:`DistributionSummary` frozen and
    hashable while still allowing ergonomic name-based lookup.
    """
    for name, value in pairs:
        if name == axis:
            return value
    return None


__all__ = [
    "DistributionSummary",
]
