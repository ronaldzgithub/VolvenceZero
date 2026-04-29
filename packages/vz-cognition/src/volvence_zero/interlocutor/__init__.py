"""Gap 9 slice 1: InterlocutorState perception readout.

EmoGPT v4.0 \u00a7 13 introduces a "pressure-driven AI with a
temper": differentiated experience via a 12-dim InterlocutorState,
a Resistance / Coaxing coefficient lookup table, and a
RelationshipStage gradual-unlock ladder.

VolvenceZero does NOT copy the lookup table. Instead:

* The 12 dimensions are continuous features read from existing
  kernel signals (``dual_track`` / ``evaluation`` / ``memory`` /
  ``prediction_error`` / ``commitment`` / ``regime``) \u2014 no
  keyword matching on user text, no ``dict[str, coefficient]``
  classification.
* ``resistance_level`` and ``openness_to_guidance`` ARE two of
  the 12 axes, but each is a smooth function of the runtime
  features, not a step-function across EmoGPT's five zones.
* Gradual unlock is handled by the existing
  ``vitals.recharge_per_regime`` schedule keyed on
  ``relationship_state.stage`` \u2014 not re-implemented here.
* This module is KERNEL-side and purely data-transformation; no
  owner mutation, no side effects. The same "scorer" pattern as
  Gap 1 slice 3 / Gap 8 slice 2: pure function of a frozen
  context, with a duck-typed builder that pulls features from
  whichever snapshots the caller has available.

Public API:

* ``InterlocutorState`` \u2014 frozen dataclass with 12 named axes
* ``InterlocutorReadoutContext`` \u2014 feature bundle
* ``readout_interlocutor_state(ctx)`` \u2014 pure function
* ``build_interlocutor_readout_context_from_snapshots(...)`` \u2014
  duck-typed builder

See ``docs/specs/interlocutor-state.md`` and Gap 9 in
``docs/implementation/13_emogpt_prd_alignment_upgrade.md``.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# InterlocutorState \u2014 the published 12-axis view
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InterlocutorState:
    """Twelve continuous axes describing the current interlocutor.

    All fields in ``[0, 1]`` except ``trust_signal`` which is
    signed ``[-1, 1]``. Default is a neutral mid-point value that
    corresponds to "we have no runtime signal" (consumers that
    don't check ``readout_confidence`` see a safe neutral hint).

    Semantic axes:

    * ``engagement_intensity`` \u2014 how actively the interlocutor
      is engaged in this exchange (vs passive / minimal input).
    * ``self_disclosure_level`` \u2014 how much the interlocutor is
      revealing about themselves.
    * ``task_focus_level`` \u2014 how focused on a concrete task.
    * ``emotional_weight`` \u2014 felt emotional pressure in the
      turn; high when self_tension + low warmth coincide.
    * ``cognitive_engagement`` \u2014 deliberate thinking load; high
      for exploration / structured reasoning turns.
    * ``resistance_level`` \u2014 pushback / friction signal. This
      replaces EmoGPT's Resistance zone lookup with a continuous
      axis.
    * ``openness_to_guidance`` \u2014 willingness to follow the
      assistant's suggestions; the inverse complement of
      resistance, but computed independently so they are not
      tied by an algebraic identity.
    * ``directness`` \u2014 communicative directness (high = literal /
      task-focused; low = indirect / emotional / hinting).
    * ``trust_signal`` \u2014 signed trust delta in ``[-1, 1]``;
      positive = trust rising this turn, negative = rupture.
    * ``stability`` \u2014 turn-to-turn consistency of the
      interlocutor; low when context is churning.
    * ``rapport_warmth`` \u2014 felt warmth / closeness.
    * ``pace_pressure`` \u2014 how fast the interlocutor wants the
      exchange to move (urgency / impatience signal).

    ``readout_confidence`` in ``[0, 1]`` scales how much downstream
    consumers should trust the 12 axes. Low when the context was
    built cold (no dual_track, no evaluation, no PE); high when
    all feeds supplied data.

    ``rationale`` is a short human-readable tag listing the
    dominant features, same convention as Gap 8 slice 2 hint
    readout and Gap 1 slice 3 affordance scorer. Format:
    ``readout.v1.interlocutor:regime_id:top_feature=+0.xx,...``.
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


# ---------------------------------------------------------------------------
# Feature bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InterlocutorReadoutContext:
    """Scalar feature bundle consumed by ``readout_interlocutor_state``.

    All fields are plain floats / ints / strings / bools so the
    readout has no external dependencies. The duck-typed builder
    (``build_interlocutor_readout_context_from_snapshots``)
    translates real kernel snapshots into this shape.

    Availability flags drive the ``evidence`` multiplier in the
    readout \u2014 cold-start contexts produce a low-confidence
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

    # Prediction error (slice 9.1 does not use raw ``signed_reward``
    # sign magnitude separately; magnitude + relationship_error
    # cover the trust + resistance signals we need).
    pe_magnitude: float = 0.0
    pe_signed_reward: float = 0.0
    pe_relationship_error: float = 0.0

    # Memory presence.
    world_presence: float = 0.0
    self_presence: float = 0.0

    # Commitment alignment: transitions from commitment lifecycle.
    # Mostly 0 on steady turns; +1 when the most recent alignment
    # transition was accept / agree; -1 on reject / block.
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
# Helpers
# ---------------------------------------------------------------------------


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clamp_signed(value: float) -> float:
    return max(-1.0, min(1.0, value))


# ---------------------------------------------------------------------------
# Readout
# ---------------------------------------------------------------------------


_NEUTRAL: float = 0.5


def _engagement_intensity(ctx: InterlocutorReadoutContext) -> float:
    pull = (
        0.32 * ctx.cross_track_tension
        + 0.22 * max(ctx.world_drive, ctx.self_drive)
        + 0.18 * ctx.switch_pressure
        + 0.12 * (ctx.task_bias + ctx.repair_bias + ctx.exploration_bias) / 3.0
    )
    push = 0.30 * ctx.stabilize_bias
    return _clamp01(_NEUTRAL + ctx.evidence_score() * (pull - push))


def _self_disclosure_level(ctx: InterlocutorReadoutContext) -> float:
    pull = (
        0.35 * ctx.self_presence
        + 0.22 * ctx.self_tension
        + 0.15 * ctx.warmth
        + 0.10 * ctx.support_presence
        + 0.12 * ctx.self_drive
    )
    push = 0.20 * ctx.task_bias + 0.10 * ctx.world_drive
    return _clamp01(_NEUTRAL + ctx.evidence_score() * (pull - push))


def _task_focus_level(ctx: InterlocutorReadoutContext) -> float:
    pull = (
        0.28 * ctx.task_bias
        + 0.25 * ctx.world_drive
        + 0.20 * ctx.task_pressure
        + 0.15 * ctx.world_presence
        + 0.10 * ctx.exploration_bias
    )
    push = (
        0.20 * ctx.repair_bias
        + 0.15 * ctx.self_tension
        + 0.12 * ctx.stabilize_bias
    )
    return _clamp01(_NEUTRAL + ctx.evidence_score() * (pull - push))


def _emotional_weight(ctx: InterlocutorReadoutContext) -> float:
    pull = (
        0.30 * ctx.self_tension
        + 0.20 * ctx.repair_bias
        + 0.18 * ctx.cross_track_tension
        + 0.15 * ctx.pe_magnitude
        + 0.12 * ctx.support_presence
    )
    push = 0.25 * ctx.warmth + 0.15 * ctx.task_bias
    return _clamp01(_NEUTRAL + ctx.evidence_score() * (pull - push))


def _cognitive_engagement(ctx: InterlocutorReadoutContext) -> float:
    pull = (
        0.28 * ctx.exploration_bias
        + 0.22 * ctx.shared_drive
        + 0.18 * ctx.info_integration
        + 0.15 * ctx.switch_pressure
        + 0.12 * ctx.task_bias
    )
    push = 0.22 * ctx.stabilize_bias + 0.10 * ctx.repair_bias
    return _clamp01(_NEUTRAL + ctx.evidence_score() * (pull - push))


def _resistance_level(ctx: InterlocutorReadoutContext) -> float:
    # Resistance rises with PE magnitude (prediction-of-interlocutor
    # failures), cross-track tension, reject-ward commitment trend,
    # and low stability. It drops with trust signals + warmth.
    pull = (
        0.28 * ctx.pe_magnitude
        + 0.22 * ctx.cross_track_tension
        + 0.20 * max(-ctx.commitment_alignment_trend, 0.0)
        + 0.18 * (1.0 - ctx.cross_track_stability)
        + 0.12 * ctx.pe_relationship_error  # signed; negative pulls less
    )
    push = (
        0.22 * ctx.warmth
        + 0.18 * ctx.cross_track_stability
        + 0.15 * max(ctx.commitment_alignment_trend, 0.0)
        + 0.10 * max(ctx.pe_signed_reward, 0.0)
    )
    return _clamp01(_NEUTRAL + ctx.evidence_score() * (pull - push))


def _openness_to_guidance(ctx: InterlocutorReadoutContext) -> float:
    # Openness rises with warmth, stability, positive commitment
    # trend, low resistance factors, and positive signed_reward.
    # Computed independently of ``resistance_level`` so the two
    # axes can co-evolve under the same feature bundle but with
    # different weightings (downstream consumers should read
    # BOTH, not assume ``openness = 1 - resistance``).
    pull = (
        0.28 * ctx.warmth
        + 0.22 * ctx.cross_track_stability
        + 0.18 * max(ctx.commitment_alignment_trend, 0.0)
        + 0.15 * ctx.support_presence
        + 0.12 * max(ctx.pe_signed_reward, 0.0)
    )
    push = (
        0.20 * ctx.pe_magnitude
        + 0.18 * ctx.cross_track_tension
        + 0.15 * max(-ctx.commitment_alignment_trend, 0.0)
    )
    return _clamp01(_NEUTRAL + ctx.evidence_score() * (pull - push))


def _directness(ctx: InterlocutorReadoutContext) -> float:
    # Directness is high in task/world-driven turns with clear
    # abstract-action dominance; low in emotional / warm /
    # indirect repair turns.
    pull = (
        0.30 * ctx.task_bias
        + 0.25 * ctx.world_drive
        + 0.15 * ctx.task_pressure
        + 0.10 * ctx.world_presence
    )
    push = (
        0.22 * ctx.repair_bias
        + 0.18 * ctx.warmth
        + 0.15 * ctx.self_tension
        + 0.10 * ctx.stabilize_bias
    )
    return _clamp01(_NEUTRAL + ctx.evidence_score() * (pull - push))


def _trust_signal(ctx: InterlocutorReadoutContext) -> float:
    # Signed in [-1, 1]: positive = rising, negative = rupture.
    positive = (
        0.35 * max(ctx.pe_signed_reward, 0.0)
        + 0.30 * max(ctx.commitment_alignment_trend, 0.0)
        + 0.15 * ctx.warmth
        + 0.10 * ctx.cross_track_stability
    )
    negative = (
        0.35 * max(-ctx.pe_signed_reward, 0.0)
        + 0.30 * max(-ctx.commitment_alignment_trend, 0.0)
        + 0.15 * ctx.cross_track_tension
        + 0.12 * ctx.pe_relationship_error  # signed; negative is rupture
    )
    return _clamp_signed(ctx.evidence_score() * (positive - negative))


def _stability(ctx: InterlocutorReadoutContext) -> float:
    pull = (
        0.35 * ctx.cross_track_stability
        + 0.20 * ctx.warmth
        + 0.15 * ctx.info_integration
        + 0.12 * ctx.stabilize_bias
    )
    push = (
        0.25 * ctx.switch_pressure
        + 0.20 * ctx.cross_track_tension
        + 0.15 * ctx.pe_magnitude
    )
    return _clamp01(_NEUTRAL + ctx.evidence_score() * (pull - push))


def _rapport_warmth(ctx: InterlocutorReadoutContext) -> float:
    pull = (
        0.35 * ctx.warmth
        + 0.22 * ctx.support_presence
        + 0.15 * ctx.cross_track_stability
        + 0.10 * max(ctx.pe_signed_reward, 0.0)
    )
    push = 0.22 * ctx.cross_track_tension + 0.15 * ctx.pe_magnitude
    return _clamp01(_NEUTRAL + ctx.evidence_score() * (pull - push))


def _pace_pressure(ctx: InterlocutorReadoutContext) -> float:
    pull = (
        0.30 * ctx.task_pressure
        + 0.22 * ctx.switch_pressure
        + 0.20 * ctx.world_drive
        + 0.12 * ctx.task_bias
    )
    push = 0.28 * ctx.stabilize_bias + 0.18 * ctx.warmth
    return _clamp01(_NEUTRAL + ctx.evidence_score() * (pull - push))


def _build_rationale(
    *, regime_id: str, axes: "tuple[tuple[str, float], ...]"
) -> str:
    """Short audit tag listing the 3 dominant axes by signed magnitude."""
    ranked = sorted(axes, key=lambda kv: abs(kv[1] - _NEUTRAL), reverse=True)[:3]
    pieces = [f"{name}={value:+.2f}" for name, value in ranked]
    return "readout.v1.interlocutor:" + regime_id + ":" + ",".join(pieces)


def readout_interlocutor_state(
    ctx: InterlocutorReadoutContext,
) -> InterlocutorState:
    """Compute an ``InterlocutorState`` from the context.

    Pure function. With ``evidence_score() == 0`` the output is
    exactly the neutral default (0.5 across 11 axes, 0.0 for
    trust_signal, 0.1 readout_confidence). With full evidence
    the axes move up to ~0.9 or down to ~0.1 depending on how
    sharply features lean.

    ``readout_confidence`` is ``0.20 + 0.70 * evidence_score``
    so downstream consumers scale their trust in the readout
    accordingly. This matches Gap 8 slice 2 / Gap 1 slice 3
    confidence conventions.
    """
    ev = ctx.evidence_score()
    engagement = _engagement_intensity(ctx)
    disclosure = _self_disclosure_level(ctx)
    focus = _task_focus_level(ctx)
    emotional = _emotional_weight(ctx)
    cognitive = _cognitive_engagement(ctx)
    resistance = _resistance_level(ctx)
    openness = _openness_to_guidance(ctx)
    directness = _directness(ctx)
    trust = _trust_signal(ctx)
    stability = _stability(ctx)
    rapport = _rapport_warmth(ctx)
    pace = _pace_pressure(ctx)
    axes = (
        ("engagement", engagement),
        ("disclosure", disclosure),
        ("focus", focus),
        ("emotional", emotional),
        ("cognitive", cognitive),
        ("resistance", resistance),
        ("openness", openness),
        ("directness", directness),
        ("trust", (trust + 1.0) / 2.0),  # re-centre for rationale magnitude
        ("stability", stability),
        ("rapport", rapport),
        ("pace", pace),
    )
    rationale = _build_rationale(regime_id=ctx.active_regime_id, axes=axes)
    return InterlocutorState(
        engagement_intensity=round(engagement, 4),
        self_disclosure_level=round(disclosure, 4),
        task_focus_level=round(focus, 4),
        emotional_weight=round(emotional, 4),
        cognitive_engagement=round(cognitive, 4),
        resistance_level=round(resistance, 4),
        openness_to_guidance=round(openness, 4),
        directness=round(directness, 4),
        trust_signal=round(trust, 4),
        stability=round(stability, 4),
        rapport_warmth=round(rapport, 4),
        pace_pressure=round(pace, 4),
        readout_confidence=round(0.20 + 0.70 * ev, 4),
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# Duck-typed builder
# ---------------------------------------------------------------------------


def _safe_get(obj: object | None, name: str, default: object) -> object:
    if obj is None:
        return default
    return getattr(obj, name, default)


def _snapshot_value(snapshot: object | None) -> object | None:
    if snapshot is None:
        return None
    value = getattr(snapshot, "value", None)
    return value if value is not None else snapshot


def _commitment_alignment_trend(commitment_value: object | None) -> float:
    """Return a signed ``[-1, 1]`` summary of recent alignment transitions.

    Reads ``lifecycle_entries[*].last_alignment`` (from Gap 7
    AAC lifecycle). Maps ``agree`` -> +1, ``reject`` -> -1,
    ``defer`` -> 0, missing -> 0. Averages over the lifecycle
    entries present in the snapshot. If there are no entries,
    returns 0.

    This keeps the readout sensitive to the MOST RECENT burst
    of acceptance or rejection \u2014 the commitment snapshot
    already rotates old entries out of view.
    """
    entries = _safe_get(commitment_value, "lifecycle_entries", None)
    if not entries:
        return 0.0
    total = 0.0
    count = 0
    for entry in entries:
        alignment = _safe_get(entry, "last_alignment", None)
        alignment_value = str(getattr(alignment, "value", alignment or ""))
        if alignment_value == "agree":
            total += 1.0
            count += 1
        elif alignment_value == "reject":
            total -= 1.0
            count += 1
        elif alignment_value == "defer":
            count += 1
    if count == 0:
        return 0.0
    return max(-1.0, min(1.0, total / count))


def _evaluation_metric(
    evaluation_value: object | None, name: str, default: float
) -> float:
    if evaluation_value is None:
        return default
    scores = getattr(evaluation_value, "turn_scores", None) or ()
    for score in scores:
        if getattr(score, "metric_name", None) == name:
            return float(getattr(score, "value", default))
    return default


def _memory_presence(memory_value: object | None) -> tuple[float, float]:
    if memory_value is None:
        return (0.0, 0.0)
    entries = getattr(memory_value, "retrieved_entries", None) or ()
    # ``Track`` enum values: "world" / "self" / "shared" (string enum).
    world_count = 0
    self_count = 0
    for entry in entries:
        track = getattr(entry, "track", None)
        track_value = str(getattr(track, "value", track))
        if track_value == "world":
            world_count += 1
        elif track_value == "self":
            self_count += 1
    return (_clamp01(world_count / 3.0), _clamp01(self_count / 3.0))


def build_interlocutor_readout_context_from_snapshots(
    *,
    regime_snapshot: object | None = None,
    dual_track_snapshot: object | None = None,
    evaluation_snapshot: object | None = None,
    prediction_error_snapshot: object | None = None,
    memory_snapshot: object | None = None,
    commitment_snapshot: object | None = None,
) -> InterlocutorReadoutContext:
    """Build a context from kernel snapshots.

    Duck-typed so this module does NOT have to import the
    kernel snapshot classes (matches the Gap 1 slice 3 affordance
    scorer pattern). Missing inputs collapse to neutral defaults
    and the corresponding ``has_*`` flag stays ``False``.

    Accepted shapes (each ``.value`` payload or the raw dataclass):

    * ``RegimeSnapshot`` \u2014 reads ``active_regime.regime_id`` and
      ``turns_in_current_regime``.
    * ``DualTrackSnapshot`` \u2014 reads tension / controller_code /
      abstract_action_hint for both tracks + cross_track_tension.
    * ``EvaluationSnapshot`` \u2014 reads 5 metric readouts by name.
    * ``PredictionErrorSnapshot`` \u2014 reads ``error.magnitude``,
      ``error.signed_reward``, ``error.relationship_error``.
    * ``MemorySnapshot`` \u2014 counts ``retrieved_entries`` by
      ``track``.
    * ``CommitmentSnapshot`` \u2014 averages ``last_alignment`` across
      ``lifecycle_entries``.
    """
    regime_value = _snapshot_value(regime_snapshot)
    dual_track_value = _snapshot_value(dual_track_snapshot)
    evaluation_value = _snapshot_value(evaluation_snapshot)
    pe_value = _snapshot_value(prediction_error_snapshot)
    memory_value = _snapshot_value(memory_snapshot)
    commitment_value = _snapshot_value(commitment_snapshot)

    active_regime = _safe_get(regime_value, "active_regime", None)
    regime_id = str(_safe_get(active_regime, "regime_id", ""))
    turns = int(_safe_get(regime_value, "turns_in_current_regime", 0))

    cross = 0.0
    world_t = 0.0
    self_t = 0.0
    world_drive = 0.0
    self_drive = 0.0
    shared_drive = 0.0
    switch_p = 0.0
    repair_b = 0.0
    task_b = 0.0
    explor_b = 0.0
    stab_b = 0.0
    if dual_track_value is not None:
        cross = _clamp01(float(getattr(dual_track_value, "cross_track_tension", 0.0)))
        world_track = _safe_get(dual_track_value, "world_track", None)
        self_track = _safe_get(dual_track_value, "self_track", None)
        world_t = _clamp01(float(_safe_get(world_track, "tension_level", 0.0)))
        self_t = _clamp01(float(_safe_get(self_track, "tension_level", 0.0)))
        w_code = tuple(_safe_get(world_track, "controller_code", ()) or ())
        s_code = tuple(_safe_get(self_track, "controller_code", ()) or ())
        world_drive = _clamp01(w_code[0] if len(w_code) > 0 else 0.0)
        self_drive = _clamp01(s_code[0] if len(s_code) > 0 else 0.0)
        shared_drive = _clamp01(
            ((w_code[1] if len(w_code) > 1 else 0.0)
             + (s_code[1] if len(s_code) > 1 else 0.0))
            / 2.0
        )
        switch_p = _clamp01(
            max(
                w_code[2] if len(w_code) > 2 else 0.0,
                s_code[2] if len(s_code) > 2 else 0.0,
            )
        )
        hints = tuple(
            h
            for h in (
                _safe_get(world_track, "abstract_action_hint", None),
                _safe_get(self_track, "abstract_action_hint", None),
            )
            if h is not None
        )
        if hints:
            repair_b = _clamp01(
                sum(1.0 for h in hints if h == "repair_controller") / len(hints)
            )
            task_b = _clamp01(
                sum(1.0 for h in hints if h == "task_controller") / len(hints)
            )
            explor_b = _clamp01(
                sum(1.0 for h in hints if h == "exploration_controller") / len(hints)
            )
            stab_b = _clamp01(
                sum(1.0 for h in hints if h == "stabilize_controller") / len(hints)
            )

    pe_mag = 0.0
    pe_signed = 0.0
    pe_rel = 0.0
    if pe_value is not None:
        error = getattr(pe_value, "error", None)
        if error is not None:
            pe_mag = float(getattr(error, "magnitude", 0.0))
            pe_signed = float(getattr(error, "signed_reward", 0.0))
            pe_rel = float(getattr(error, "relationship_error", 0.0))

    world_p, self_p = _memory_presence(memory_value)
    warmth = _evaluation_metric(evaluation_value, "warmth", default=0.5)
    support = _evaluation_metric(evaluation_value, "support_presence", default=warmth)
    task_pressure = _evaluation_metric(evaluation_value, "task_pressure", default=0.5)
    stability_eval = _evaluation_metric(
        evaluation_value, "cross_track_stability", default=0.5
    )
    info = _evaluation_metric(evaluation_value, "info_integration", default=0.5)
    alignment_trend = _commitment_alignment_trend(commitment_value)

    return InterlocutorReadoutContext(
        active_regime_id=regime_id,
        turns_in_current_regime=max(0, turns),
        has_dual_track=dual_track_value is not None,
        has_evaluation=evaluation_value is not None,
        has_prediction_error=pe_value is not None,
        has_memory=memory_value is not None,
        has_commitment=commitment_value is not None,
        cross_track_tension=cross,
        world_tension=world_t,
        self_tension=self_t,
        world_drive=world_drive,
        self_drive=self_drive,
        shared_drive=shared_drive,
        switch_pressure=switch_p,
        repair_bias=repair_b,
        task_bias=task_b,
        exploration_bias=explor_b,
        stabilize_bias=stab_b,
        warmth=warmth,
        support_presence=support,
        task_pressure=task_pressure,
        cross_track_stability=stability_eval,
        info_integration=info,
        pe_magnitude=pe_mag,
        pe_signed_reward=pe_signed,
        pe_relationship_error=pe_rel,
        world_presence=world_p,
        self_presence=self_p,
        commitment_alignment_trend=alignment_trend,
    )


__all__ = [
    "InterlocutorReadoutContext",
    "InterlocutorState",
    "build_interlocutor_readout_context_from_snapshots",
    "readout_interlocutor_state",
]
