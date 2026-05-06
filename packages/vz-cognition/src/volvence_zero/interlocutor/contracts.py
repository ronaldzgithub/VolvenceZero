"""Interlocutor contract types.

Pure data module. Defines the 12-axis :class:`InterlocutorState`,
the feature bundle :class:`InterlocutorReadoutContext`, the
:class:`InterlocutorStateSnapshot` published by the SHADOW owner,
and the single-source-of-truth :class:`InterlocutorThresholds`
constants that classify a state into named zones.

Design contract:

* Threshold values live HERE, ONCE. ``prompt_planner`` and
  ``response_synthesizer`` consume the published zone booleans
  rather than re-applying numeric thresholds. That is the
  SSOT cleanup target for Wave 2 of ``ssot-cleanup-p0-p4``.
* Zones are computed ONCE in :func:`compute_zones` and attached to
  the snapshot. Adding a new zone requires landing a member here
  and a new threshold constant if needed; consumers read by name.
* ``InterlocutorStateSnapshot`` is the SHADOW owner publication
  surface; ``InterlocutorState`` is the older read-time view that
  the lifeform layer keeps for backwards compat (``LifeformSession.
  interlocutor_state``). The two share the 12 axes; zone bools
  live on both for symmetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar


# ---------------------------------------------------------------------------
# Single source of truth for thresholds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InterlocutorThresholds:
    """Canonical thresholds for interlocutor zone classification.

    Each named zone in :class:`InterlocutorStateSnapshot` /
    :class:`InterlocutorState` is computed against these constants.
    Consumers MUST NOT re-define numeric thresholds for the same
    axes; they MUST read the zone booleans.
    """

    # Confidence floor below which all zones are False (cold start).
    min_confidence: ClassVar[float] = 0.30

    # Acknowledge-pressure zone components. The planner adds
    # ACKNOWLEDGE_PRESSURE if any of these fire, so each is exposed
    # as its own zone bool plus a composite ``acknowledge_pressure_zone``.
    emotional_high: ClassVar[float] = 0.55
    resistance_high: ClassVar[float] = 0.50
    trust_negative: ClassVar[float] = -0.10

    # Synthesizer "should I render the repair variant?" check.
    # Looser than acknowledge_pressure (resistance >= 0.30, not 0.50).
    repair_resistance: ClassVar[float] = 0.30
    repair_trust: ClassVar[float] = 0.05

    # Direct-task render zone: all-of.
    direct_task_focus: ClassVar[float] = 0.685
    direct_directness: ClassVar[float] = 0.58
    direct_emotional_max: ClassVar[float] = 0.58

    # Emotional render zone: stricter than acknowledge_pressure.
    emotional_renderer: ClassVar[float] = 0.56
    emotional_self_disclosure: ClassVar[float] = 0.65

    # Pace pressure (planner drops meta sections, caps Qs).
    pace_high: ClassVar[float] = 0.65

    # Directness floor (planner caps Qs).
    directness_low: ClassVar[float] = 0.40

    # Cold rapport (planner adds CONTINUITY_NOTE).
    rapport_low: ClassVar[float] = 0.40
    engagement_floor: ClassVar[float] = 0.30


# ---------------------------------------------------------------------------
# 12-axis state + zone readouts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InterlocutorState:
    """Twelve continuous axes describing the current interlocutor.

    All fields in ``[0, 1]`` except ``trust_signal`` which is
    signed ``[-1, 1]``. Default is a neutral mid-point value that
    corresponds to "we have no runtime signal" (consumers that
    don't check ``readout_confidence`` see a safe neutral hint).

    The ``*_zone`` booleans are computed ONCE by :func:`compute_zones`
    and attached here. Consumers read these instead of re-applying
    numeric thresholds.
    """

    engagement_intensity: float = 0.5
    self_disclosure_level: float = 0.5
    task_focus_level: float = 0.5
    emotional_weight: float = 0.5
    cognitive_engagement: float = 0.5
    resistance_level: float = 0.5
    openness_to_guidance: float = 0.5
    directness: float = 0.5
    trust_signal: float = 0.0
    stability: float = 0.5
    rapport_warmth: float = 0.5
    pace_pressure: float = 0.5
    readout_confidence: float = 0.1
    rationale: str = ""

    # Single-source-of-truth zone classifications (Wave 2). Default
    # all-False so a default-constructed neutral state with low
    # confidence does not trigger any modulation.
    acknowledge_pressure_zone: bool = False
    emotional_high_zone: bool = False
    resistance_high_zone: bool = False
    trust_negative_zone: bool = False
    repair_zone: bool = False
    direct_task_zone: bool = False
    emotional_render_zone: bool = False
    pace_pressure_zone: bool = False
    low_directness_zone: bool = False
    cold_rapport_zone: bool = False

    def __post_init__(self) -> None:
        for name in (
            "engagement_intensity",
            "self_disclosure_level",
            "task_focus_level",
            "emotional_weight",
            "cognitive_engagement",
            "resistance_level",
            "openness_to_guidance",
            "directness",
            "stability",
            "rapport_warmth",
            "pace_pressure",
            "readout_confidence",
        ):
            value = getattr(self, name)
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(
                    f"InterlocutorState.{name} must be in [0,1], "
                    f"got {value!r}"
                )
        if not -1.0 <= float(self.trust_signal) <= 1.0:
            raise ValueError(
                f"InterlocutorState.trust_signal must be in [-1,1], "
                f"got {self.trust_signal!r}"
            )
        # Single source of truth: zone booleans are always computed
        # from the axis values + the canonical thresholds. If a
        # caller passes zone bools explicitly, they are overwritten -
        # the snapshot's axes are authoritative. This guarantees
        # consumers always see a self-consistent state and cannot
        # drift the zone classification by hand.
        zones = compute_zones(self)
        for name, value in zones.items():
            object.__setattr__(self, name, value)


# ---------------------------------------------------------------------------
# Feature bundle (consumed by the readout)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InterlocutorReadoutContext:
    """Scalar feature bundle consumed by ``readout_interlocutor_state``.

    All fields are plain floats / ints / strings / bools so the
    readout has no external dependencies. The duck-typed builder
    (``build_interlocutor_readout_context_from_snapshots``)
    translates real kernel snapshots into this shape.

    Availability flags drive the ``evidence`` multiplier in the
    readout - cold-start contexts produce a low-confidence
    published state near the neutral baseline.
    """

    active_regime_id: str = ""
    turns_in_current_regime: int = 0

    # Availability flags.
    has_dual_track: bool = False
    has_evaluation: bool = False
    has_prediction_error: bool = False
    has_memory: bool = False
    has_commitment: bool = False

    # Tension axes (from dual_track).
    cross_track_tension: float = 0.0
    world_tension: float = 0.0
    self_tension: float = 0.0

    # Controller drives (from dual_track.controller_code).
    world_drive: float = 0.0
    self_drive: float = 0.0
    shared_drive: float = 0.0
    switch_pressure: float = 0.0

    # Abstract-action biases (from dual_track.abstract_action_hint
    # frequency across tracks).
    repair_bias: float = 0.0
    task_bias: float = 0.0
    exploration_bias: float = 0.0
    stabilize_bias: float = 0.0

    # Evaluation readouts.
    warmth: float = 0.5
    support_presence: float = 0.5
    task_pressure: float = 0.5
    cross_track_stability: float = 0.5
    info_integration: float = 0.5

    # Prediction error.
    pe_magnitude: float = 0.0
    pe_signed_reward: float = 0.0
    pe_relationship_error: float = 0.0

    # Memory presence.
    world_presence: float = 0.0
    self_presence: float = 0.0

    # Commitment alignment trend (signed, derived from lifecycle).
    commitment_alignment_trend: float = 0.0

    def evidence_score(self) -> float:
        """Heuristic ``[0, 1]`` for "how much signal did we see".

        Used to scale the readout's movement from the neutral
        base. Cold start (no snapshots) -> ~0.10; full signal with
        commitment evidence -> ~0.95.
        """
        score = 0.10
        if self.has_dual_track:
            score += 0.30
        if self.has_evaluation:
            score += 0.20
        if self.has_prediction_error:
            score += 0.15
        if self.has_memory:
            score += 0.10
        if self.has_commitment:
            score += 0.10
        return round(min(score, 1.0), 4)


# ---------------------------------------------------------------------------
# SHADOW owner snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InterlocutorStateSnapshot:
    """Owner-published interlocutor view (Wave 2).

    Wraps the 12-axis :class:`InterlocutorState` and exposes the
    same zone booleans alongside metadata about the snapshot
    provenance. ``description`` is the owner-authored summary;
    consumers (planner, synthesizer, evaluation) MUST NOT
    reconstruct interlocutor state from upstream snapshots; they
    MUST read this snapshot.
    """

    state: InterlocutorState
    description: str = ""

    @property
    def readout_confidence(self) -> float:
        return self.state.readout_confidence

    @property
    def rationale(self) -> str:
        return self.state.rationale


# ---------------------------------------------------------------------------
# Zone computation - SINGLE source of truth
# ---------------------------------------------------------------------------


def compute_zones(state: InterlocutorState) -> dict[str, bool]:
    """Compute zone booleans from a 12-axis :class:`InterlocutorState`.

    Returns a fresh dict (not the state) so callers can build a new
    ``InterlocutorState`` via ``dataclasses.replace``. All zones
    return ``False`` when ``readout_confidence`` is below
    :attr:`InterlocutorThresholds.min_confidence`.

    This function is the SSOT for "what does X axis value mean".
    Consumers MUST NOT re-apply the underlying numeric thresholds.
    """

    th = InterlocutorThresholds
    if state.readout_confidence < th.min_confidence:
        return {
            "acknowledge_pressure_zone": False,
            "emotional_high_zone": False,
            "resistance_high_zone": False,
            "trust_negative_zone": False,
            "repair_zone": False,
            "direct_task_zone": False,
            "emotional_render_zone": False,
            "pace_pressure_zone": False,
            "low_directness_zone": False,
            "cold_rapport_zone": False,
        }

    emotional_high = state.emotional_weight >= th.emotional_high
    resistance_high = state.resistance_level >= th.resistance_high
    trust_negative = state.trust_signal <= th.trust_negative

    return {
        # Composite "should we add ACKNOWLEDGE_PRESSURE?" gate (planner).
        "acknowledge_pressure_zone": (
            emotional_high or resistance_high or trust_negative
        ),
        "emotional_high_zone": emotional_high,
        "resistance_high_zone": resistance_high,
        "trust_negative_zone": trust_negative,
        # Synthesizer "render the repair variant" gate.
        "repair_zone": (
            state.resistance_level >= th.repair_resistance
            or state.trust_signal <= th.repair_trust
        ),
        "direct_task_zone": (
            state.task_focus_level >= th.direct_task_focus
            and state.directness >= th.direct_directness
            and state.emotional_weight <= th.direct_emotional_max
        ),
        "emotional_render_zone": (
            state.emotional_weight >= th.emotional_renderer
            and state.self_disclosure_level >= th.emotional_self_disclosure
        ),
        "pace_pressure_zone": state.pace_pressure >= th.pace_high,
        "low_directness_zone": state.directness <= th.directness_low,
        "cold_rapport_zone": (
            state.rapport_warmth <= th.rapport_low
            and state.engagement_intensity >= th.engagement_floor
        ),
    }


def with_zones(state: InterlocutorState) -> InterlocutorState:
    """Return a copy of ``state`` with zone booleans recomputed.

    Useful in tests or when an :class:`InterlocutorState` is built
    by hand without going through :func:`readout_interlocutor_state`.
    """

    from dataclasses import replace

    return replace(state, **compute_zones(state))


__all__ = [
    "InterlocutorReadoutContext",
    "InterlocutorState",
    "InterlocutorStateSnapshot",
    "InterlocutorThresholds",
    "compute_zones",
    "with_zones",
]
