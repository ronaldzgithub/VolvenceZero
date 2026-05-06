"""Pure 12-axis readout for the interlocutor module.

Computes :class:`InterlocutorState` from a feature-bundle context.
Same input -> same output, no side effects, no global state.

The duck-typed builder
``build_interlocutor_readout_context_from_snapshots`` adapts kernel
snapshot dataclasses into the feature bundle without depending on
the snapshot types directly (they live in different wheels). The
readout itself is wheel-independent; it is the SHADOW owner
(``volvence_zero.interlocutor.owner``) that references the kernel
snapshot types when registered into the runtime.
"""

from __future__ import annotations

from typing import Any

from volvence_zero.interlocutor.contracts import (
    InterlocutorReadoutContext,
    InterlocutorState,
    compute_zones,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clamp_signed(value: float) -> float:
    return max(-1.0, min(1.0, value))


_NEUTRAL: float = 0.5


# ---------------------------------------------------------------------------
# Per-axis readouts
# ---------------------------------------------------------------------------


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
    pull = (
        0.28 * ctx.pe_magnitude
        + 0.22 * ctx.cross_track_tension
        + 0.20 * max(-ctx.commitment_alignment_trend, 0.0)
        + 0.18 * (1.0 - ctx.cross_track_stability)
        + 0.12 * ctx.pe_relationship_error
    )
    push = (
        0.22 * ctx.warmth
        + 0.18 * ctx.cross_track_stability
        + 0.15 * max(ctx.commitment_alignment_trend, 0.0)
        + 0.10 * max(ctx.pe_signed_reward, 0.0)
    )
    return _clamp01(_NEUTRAL + ctx.evidence_score() * (pull - push))


def _openness_to_guidance(ctx: InterlocutorReadoutContext) -> float:
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
        + 0.12 * ctx.pe_relationship_error
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
    *, regime_id: str, axes: tuple[tuple[str, float], ...]
) -> str:
    """Short audit tag listing the 3 dominant axes by signed magnitude."""
    ranked = sorted(axes, key=lambda kv: abs(kv[1] - _NEUTRAL), reverse=True)[:3]
    pieces = [f"{name}={value:+.2f}" for name, value in ranked]
    return "readout.v1.interlocutor:" + regime_id + ":" + ",".join(pieces)


def readout_interlocutor_state(
    ctx: InterlocutorReadoutContext,
) -> InterlocutorState:
    """Compute an :class:`InterlocutorState` from the context.

    Pure function. With ``evidence_score() == 0`` the output is
    exactly the neutral default (0.5 across 11 axes, 0.0 for
    trust_signal, 0.1 readout_confidence). Zone booleans are
    computed from the same thresholds defined in
    :class:`InterlocutorThresholds` so consumers always see a
    self-consistent state.
    """
    from dataclasses import replace

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
        ("trust", (trust + 1.0) / 2.0),
        ("stability", stability),
        ("rapport", rapport),
        ("pace", pace),
    )
    rationale = _build_rationale(regime_id=ctx.active_regime_id, axes=axes)
    state = InterlocutorState(
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
    return replace(state, **compute_zones(state))


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

    Reads ``lifecycle_entries[*].last_alignment`` (from Gap 7 AAC
    lifecycle). Maps ``agree`` -> +1, ``reject`` -> -1, ``defer`` -> 0,
    missing -> 0. Averages over the lifecycle entries present in the
    snapshot. If there are no entries, returns 0.

    This keeps the readout sensitive to the MOST RECENT burst of
    acceptance or rejection - the commitment snapshot already rotates
    old entries out of view.
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
    world_count = 0
    self_count = 0
    for entry in entries:
        track = getattr(entry, "track", None)
        track_value = str(getattr(track, "value", track))
        if track_value == "world":
            world_count += 1
        elif track_value == "self":
            self_count += 1
    return (
        max(0.0, min(1.0, world_count / 3.0)),
        max(0.0, min(1.0, self_count / 3.0)),
    )


def build_interlocutor_readout_context_from_snapshots(
    *,
    regime_snapshot: Any | None = None,
    dual_track_snapshot: Any | None = None,
    evaluation_snapshot: Any | None = None,
    prediction_error_snapshot: Any | None = None,
    memory_snapshot: Any | None = None,
    commitment_snapshot: Any | None = None,
) -> InterlocutorReadoutContext:
    """Build a context from kernel snapshots.

    Duck-typed so this module does NOT have to import the kernel
    snapshot classes (matches the Gap 1 slice 3 affordance scorer
    pattern). Missing inputs collapse to neutral defaults and the
    corresponding ``has_*`` flag stays ``False``.
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
            (
                (w_code[1] if len(w_code) > 1 else 0.0)
                + (s_code[1] if len(s_code) > 1 else 0.0)
            )
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
    "build_interlocutor_readout_context_from_snapshots",
    "readout_interlocutor_state",
]
