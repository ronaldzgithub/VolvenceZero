"""Lifeform vitals layer \u2014 internal drives and slow-scale prediction error.

This module is what makes the lifeform an "always-on organism" rather than a
turn-driven assistant. Without it, the system only produces prediction error
when the user speaks; with it, drives accumulate deviation between turns,
publishing a continuous PE source that downstream consumers (FollowupManager,
PromptPlanner, regime calibration) can read.

R-#:

* **R-PE (Prediction error is the primitive learning signal)** \u2014 vitals
  produce a slow-scale PE distinct from the kernel's per-turn PE. Drives
  define expected internal state ("bond_warmth target=0.7"); deviation from
  the homeostatic band IS the surprise signal at the metabolic timescale.
* **R1 (Multi-timescale learning)** \u2014 the kernel runs at the per-turn /
  online-fast scale; vitals run at the SYSTEM-tick / session-medium scale.
  They share NL's frequency-ordering doctrine: different update cadences,
  different parameter blocks, no shared gradient flow.
* **R8 (Snapshot-first)** \u2014 ``VitalsSnapshot`` is the only public surface;
  consumers do not reach into ``VitalsModule._levels``.
* **R11 (Internal state must be nameable and publishable)** \u2014 every drive
  has a stable name, its level is in [0, 1], and its PE contribution is
  computable from public fields alone. Enough surface for reflection,
  evaluation, rollback.

Verticals (``lifeform-domain-*``) ship a ``VitalsBootstrap`` describing
their drive set; the kernel does not know which drives matter. This keeps
the kernel vertical-agnostic while still letting each product imprint its
own "what does this lifeform care about" signature on the behavior loop.

Decay / recharge math (deliberately simple, no learned smoothing for v1):

* Each ``SYSTEM`` tick subtracts ``decay_per_tick`` from every drive level,
  clamped at 0.
* On each completed turn (``on_turn``) drives are recharged by
  ``recharge_per_turn`` (baseline) plus
  ``recharge_per_regime[active_regime]`` (regime-specific bonus). Levels
  clamp at 1.
* Deviation = ``|level - target|``. PE contribution is
  ``pe_weight * deviation`` ONLY when the level falls outside the
  homeostatic band; staying inside the band contributes 0 (homeostasis).
* Total slow-scale PE is the sum across drives. When it crosses
  ``proactive_pe_threshold``, the session may surface a proactive
  ``FollowupItem`` (subject to ``proactive_cooldown_ticks``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lifeform_core.types import TickEvent, TickKind
from volvence_zero.prediction import DistributionSummary


# ---------------------------------------------------------------------------
# Configuration types (vertical-shipped)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DriveSpec:
    """One drive channel.

    Drives are deliberately scalar in [0, 1] to keep the math interpretable.
    A drive with ``decay_per_tick=0`` and ``recharge_per_turn=0`` is a
    constant baseline that contributes 0 PE \u2014 useful as a "presence
    indicator" without behavioral pressure. A drive with strong decay and
    weak recharge produces escalating proactive pressure if the lifeform
    sits idle, which is the canonical "always-on organism" effect.
    """

    name: str
    target: float
    homeostatic_band: tuple[float, float]
    decay_per_tick: float
    pe_weight: float
    initial_level: float = 0.5
    recharge_per_turn: float = 0.0
    recharge_per_regime: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.target <= 1.0:
            raise ValueError(f"DriveSpec.target must be in [0,1], got {self.target!r}")
        low, high = self.homeostatic_band
        if not (0.0 <= low <= high <= 1.0):
            raise ValueError(
                f"DriveSpec.homeostatic_band must be 0\u2264low\u2264high\u22641, got {self.homeostatic_band!r}"
            )
        if self.decay_per_tick < 0.0:
            raise ValueError(f"DriveSpec.decay_per_tick must be \u22650, got {self.decay_per_tick!r}")
        if self.pe_weight < 0.0:
            raise ValueError(f"DriveSpec.pe_weight must be \u22650, got {self.pe_weight!r}")
        if not 0.0 <= self.initial_level <= 1.0:
            raise ValueError(
                f"DriveSpec.initial_level must be in [0,1], got {self.initial_level!r}"
            )


@dataclass(frozen=True)
class VitalsBootstrap:
    """Complete vitals configuration for one vertical.

    Verticals construct one of these (typically as a module-level factory
    function so it stays inspectable / version-controlled / diff-friendly,
    rather than pickled). The set of drives is the vertical's signature
    \u2014 what the lifeform "cares about" between turns.
    """

    schema_version: int = 1
    drives: tuple[DriveSpec, ...] = ()
    proactive_pe_threshold: float = 1.0
    proactive_followup_priority: float = 0.55
    proactive_cooldown_ticks: int = 60

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                f"Unsupported VitalsBootstrap schema_version {self.schema_version}; "
                f"this code only knows version 1"
            )
        names = [d.name for d in self.drives]
        if len(set(names)) != len(names):
            raise ValueError(f"VitalsBootstrap drive names must be unique, got {names!r}")
        if self.proactive_pe_threshold < 0.0:
            raise ValueError(
                f"VitalsBootstrap.proactive_pe_threshold must be \u22650, got "
                f"{self.proactive_pe_threshold!r}"
            )


# ---------------------------------------------------------------------------
# Public snapshot (R8)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DriveLevel:
    """Public read-out of one drive's current state."""

    name: str
    level: float
    target: float
    deviation: float
    out_of_band: bool
    pe_contribution: float


@dataclass(frozen=True)
class VitalsSnapshot:
    """Public snapshot of the entire vitals layer.

    All fields are read-only. The lifeform session re-publishes this on
    every tick advancement and every turn so consumers can subscribe to
    "current organism state" without reaching into the owner module.

    ``distributional_drift_axes`` (Phase 2 W1.3 / DM-1) is a per-axis
    signed drift readout derived from the kernel's
    :class:`PredictionError.distribution_summary`. Each entry pairs an
    axis name (``task`` / ``relationship`` / ``regime`` / ``action``)
    with a clamped drift value in ``[-1, 1]`` representing
    ``(current_iqr - baseline_iqr) / max(baseline_iqr, eps)``. The
    tuple is empty until the vitals owner has observed enough PE
    distribution summaries for the baseline EMA to stabilise (see
    ``VitalsModule._BASELINE_WARMUP_OBSERVATIONS``).
    """

    schema_version: int = 1
    tick_index: int = 0
    drive_levels: tuple[DriveLevel, ...] = ()
    total_pe: float = 0.0
    above_proactive_threshold: bool = False
    last_proactive_at_tick: int | None = None
    distributional_drift_axes: tuple[tuple[str, float], ...] = ()


# ---------------------------------------------------------------------------
# Owner module
# ---------------------------------------------------------------------------


class VitalsModule:
    """Owner of drive levels (R8: single primary owner).

    Lifecycle:

    * Constructed once per ``LifeformSession`` from a ``VitalsBootstrap``.
    * Receives ``on_tick(event)`` from the session's tick advancement loop;
      decays drives only on ``SYSTEM`` ticks.
    * Receives ``on_turn(regime=..., user_input_present=...)`` after every
      completed turn; recharges drives based on what happened.
    * Publishes ``VitalsSnapshot`` via ``current_snapshot()`` \u2014 the only
      public read path.
    * ``consider_proactive_followup(current_tick=...)`` is the side-effecting
      "should we wake the user" check; it tracks cooldown internally so
      consumers can call it on every tick without flooding follow-ups.
    """

    # Phase 2 W1.3 (DM-1) tuning constants. Owner-internal only —
    # downstream consumers MUST read the published
    # ``distributional_drift_axes`` rather than re-compute drift.
    #
    # Baseline semantics: the first ``_BASELINE_WARMUP_OBSERVATIONS``
    # non-None PE distribution summaries fed via
    # ``observe_pe_distribution`` are AVERAGED per axis to form a
    # frozen baseline. Subsequent observations do NOT update the
    # baseline; ``current_snapshot()`` reports drift relative to that
    # frozen reference. This gives drift = "how far current IQR has
    # moved from the early-phase reference" semantics, which is
    # what the Botvinick 2025 distributional-coding analogue cares
    # about. EMA semantics would create the "tracker follows current"
    # bug where drift trends toward zero by construction.
    _BASELINE_WARMUP_OBSERVATIONS: int = 5
    _DRIFT_CLAMP: float = 1.0
    _DRIFT_EPSILON: float = 1e-4

    def __init__(self, bootstrap: VitalsBootstrap) -> None:
        self._bootstrap = bootstrap
        self._levels: dict[str, float] = {
            d.name: d.initial_level for d in bootstrap.drives
        }
        self._tick_index = 0
        self._last_proactive_at: int | None = None
        self._turn_count = 0
        # Gap 2 apprentice override: when active, drive deviation
        # contributes 0 to slow-scale PE so operator-supplied teaching
        # / bulk ingestion turns do NOT look like "user is ignoring me"
        # for follow-up scheduling. Decay (on_tick) and recharge
        # (on_turn) are unaffected \u2014 the override only gates
        # published PE contributions.
        self._apprentice_override_active = False
        # Phase 2 W1.3 (DM-1): baseline IQR per PE axis, observed via
        # ``observe_pe_distribution``. ``_iqr_baseline_accum`` carries
        # running sums during warmup; on the
        # ``_BASELINE_WARMUP_OBSERVATIONS``-th observation we promote
        # the average into ``_iqr_baseline`` (frozen). Drift is computed
        # on read in ``current_snapshot()`` against that frozen baseline.
        # Both maps stay empty until warmup completes; the latest
        # summary is kept so snapshots computed between observations
        # still see fresh drift values.
        self._iqr_baseline: dict[str, float] = {}
        self._iqr_baseline_accum: dict[str, float] = {}
        self._baseline_observation_count: int = 0
        self._last_distribution_summary: DistributionSummary | None = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @property
    def bootstrap(self) -> VitalsBootstrap:
        return self._bootstrap

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def apprentice_override_active(self) -> bool:
        """Whether the apprentice override is currently gating PE to zero.

        Public read-only accessor so tests and observability surfaces
        can assert the override is correctly restored after each turn
        (leak-free invariant). The setter is ``set_apprentice_override``.
        """
        return self._apprentice_override_active

    def set_apprentice_override(self, enabled: bool) -> None:
        """Toggle the apprentice override (Gap 2).

        When ``enabled=True``:

        * ``current_snapshot()`` publishes ``total_pe=0.0`` and every
          drive's ``pe_contribution=0.0`` \u2014 deviation / out_of_band
          fields remain truthful so observability still works.
        * ``above_proactive_threshold`` becomes False, so
          ``consider_proactive_followup`` never fires during an
          apprentice / ingestion turn.
        * Decay (``on_tick``) and recharge (``on_turn``) are NOT
          affected; drives still evolve, we just suppress PE
          publication.

        The override is a simple boolean rather than a stack: the
        lifeform session invokes ``set_apprentice_override(True)`` at
        the start of an apprentice turn and restores it to the
        previous value in a ``finally`` block. Nesting would be a
        lifeform-layer orchestration concern, not a vitals concern.
        """
        self._apprentice_override_active = bool(enabled)

    # ------------------------------------------------------------------
    # Mutation entry points
    # ------------------------------------------------------------------

    def on_tick(self, event: TickEvent) -> VitalsSnapshot:
        """Apply one tick worth of decay; advance the published tick index.

        ENERGY and CONTEXT ticks are recorded for ``tick_index`` accuracy
        but do not change drive levels \u2014 only ``SYSTEM`` ticks consume the
        organism's metabolic budget. This matches NL's frequency-ordered
        levels: not every level needs to update at every tick.
        """
        self._tick_index = event.tick_index
        if event.kind != TickKind.SYSTEM:
            return self.current_snapshot()
        for drive in self._bootstrap.drives:
            self._levels[drive.name] = max(
                0.0, self._levels[drive.name] - drive.decay_per_tick
            )
        return self.current_snapshot()

    def on_turn(
        self,
        *,
        regime: str | None = None,
        user_input_present: bool = True,
    ) -> VitalsSnapshot:
        """Apply one turn's worth of recharge.

        ``user_input_present=False`` is reserved for system-initiated turns
        that should NOT count as the user replenishing engagement \u2014 in
        that case only regime-specific recharges apply, not the per-turn
        baseline.

        Negative ``recharge_per_regime`` values are honoured: a vertical
        can declare that "regime X drains drive Y" (e.g. exploration
        drains direction-certainty). Levels are always clamped to
        ``[0, 1]`` so deeply negative charges do not push state out of
        the public range.
        """
        self._turn_count += 1
        for drive in self._bootstrap.drives:
            charge = drive.recharge_per_turn if user_input_present else 0.0
            if regime is not None and regime in drive.recharge_per_regime:
                charge += drive.recharge_per_regime[regime]
            if charge != 0.0:
                new_level = self._levels[drive.name] + charge
                self._levels[drive.name] = max(0.0, min(1.0, new_level))
        return self.current_snapshot()

    def observe_pe_distribution(
        self, summary: DistributionSummary | None
    ) -> None:
        """Phase 2 W1.3 (DM-1): observe a kernel PE distribution summary.

        Called by the lifeform session after each completed turn with
        ``prediction_error.value.error.distribution_summary``. ``None``
        is a no-op (cold-start before the kernel's PE window has filled
        / bootstrap turns).

        Warmup phase (``_baseline_observation_count <
        _BASELINE_WARMUP_OBSERVATIONS``): per-axis IQR is accumulated
        into a running sum; ``current_snapshot()`` continues to report
        ``distributional_drift_axes=()``. On the warmup-completing
        observation, the per-axis sums are averaged into a frozen
        ``_iqr_baseline``. From then on, observations only update
        ``_last_distribution_summary`` so drift = current vs frozen
        baseline.
        """
        if summary is None:
            return
        self._last_distribution_summary = summary
        if self._baseline_observation_count < self._BASELINE_WARMUP_OBSERVATIONS:
            for axis, iqr_value in summary.iqr:
                self._iqr_baseline_accum[axis] = (
                    self._iqr_baseline_accum.get(axis, 0.0) + float(iqr_value)
                )
            self._baseline_observation_count += 1
            if self._baseline_observation_count >= self._BASELINE_WARMUP_OBSERVATIONS:
                # Promote the running sums into the frozen baseline.
                divisor = float(self._BASELINE_WARMUP_OBSERVATIONS)
                self._iqr_baseline = {
                    axis: total / divisor
                    for axis, total in self._iqr_baseline_accum.items()
                }
                self._iqr_baseline_accum = {}

    def consider_proactive_followup(self, *, current_tick: int) -> bool:
        """Return True if a proactive ``FollowupItem`` should surface now.

        Side-effecting: when the predicate fires it stamps
        ``last_proactive_at_tick`` so subsequent calls within the cooldown
        return False. Callers can therefore poll this on every tick without
        special-casing.
        """
        snap = self.current_snapshot()
        if not snap.above_proactive_threshold:
            return False
        if (
            self._last_proactive_at is not None
            and current_tick - self._last_proactive_at
            < self._bootstrap.proactive_cooldown_ticks
        ):
            return False
        self._last_proactive_at = current_tick
        return True

    # ------------------------------------------------------------------
    # Public read
    # ------------------------------------------------------------------

    def current_snapshot(self) -> VitalsSnapshot:
        """Publish the current vitals snapshot.

        During ``apprentice_override_active``, ``pe_contribution`` is
        forced to 0 for every drive and ``total_pe`` is 0. Deviation
        / ``out_of_band`` fields remain truthful so a proactive-minded
        consumer can still tell whether the drive IS out of band \u2014
        the override only suppresses the PE *publication*, not the
        underlying state.
        """
        levels: list[DriveLevel] = []
        total_pe = 0.0
        override_active = self._apprentice_override_active
        for drive in self._bootstrap.drives:
            level = self._levels[drive.name]
            deviation = abs(level - drive.target)
            low, high = drive.homeostatic_band
            out_of_band = level < low or level > high
            # PE contribution is suppressed under apprentice override;
            # deviation + out_of_band still report reality.
            if override_active:
                pe = 0.0
            else:
                pe = drive.pe_weight * deviation if out_of_band else 0.0
            total_pe += pe
            levels.append(
                DriveLevel(
                    name=drive.name,
                    level=level,
                    target=drive.target,
                    deviation=deviation,
                    out_of_band=out_of_band,
                    pe_contribution=pe,
                )
            )
        # Phase 2 W1.3 (DM-1) — distributional drift readout. Empty
        # until the baseline EMA has warmed up; after that, drift is a
        # signed ratio in ``[-1, 1]`` per axis. Apprentice override
        # also suppresses the drift readout (apprentice / ingestion
        # turns should not surface as "user emotional drift").
        drift_axes: tuple[tuple[str, float], ...] = ()
        if (
            not override_active
            and self._last_distribution_summary is not None
            and self._baseline_observation_count >= self._BASELINE_WARMUP_OBSERVATIONS
        ):
            drift_pairs: list[tuple[str, float]] = []
            for axis, current_iqr in self._last_distribution_summary.iqr:
                baseline = self._iqr_baseline.get(axis)
                if baseline is None:
                    continue
                denom = max(abs(baseline), self._DRIFT_EPSILON)
                drift = (float(current_iqr) - baseline) / denom
                clamped = max(
                    -self._DRIFT_CLAMP, min(self._DRIFT_CLAMP, drift)
                )
                drift_pairs.append((axis, round(clamped, 4)))
            drift_axes = tuple(drift_pairs)
        return VitalsSnapshot(
            tick_index=self._tick_index,
            drive_levels=tuple(levels),
            total_pe=total_pe,
            # Under override, above_proactive_threshold is false by
            # construction (total_pe == 0 < any positive threshold),
            # so consider_proactive_followup() returns False without
            # special-casing.
            above_proactive_threshold=(
                total_pe >= self._bootstrap.proactive_pe_threshold
            ),
            last_proactive_at_tick=self._last_proactive_at,
            distributional_drift_axes=drift_axes,
        )


__all__ = [
    "DriveSpec",
    "VitalsBootstrap",
    "DriveLevel",
    "VitalsSnapshot",
    "VitalsModule",
]
