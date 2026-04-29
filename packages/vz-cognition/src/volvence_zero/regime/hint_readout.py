"""Gap 8 slice 2: participation/depth hint readout from continuous features.

Slice 1 derived hints via a static ``dict[regime_id -> hint]``. That
is a categorical lookup keyed off a canonical label, but it is still
a "lookup table" shaped decision, and (more importantly) it fails to
reflect runtime conditions: a ``problem_solving`` turn under catastrophic
cross-track tension should probably drop ``task_level`` BRIEF even
though the scaffold says STRUCTURED.

Slice 2 replaces that with a **readout over continuous features**:

1. ``HintReadoutContext`` captures the measurable state: tension
   components, controller-code drives, abstract-action hint biases,
   PE magnitude, candidate-regime sharpness, turns_in_current_regime.
2. ``readout_participation_hint(context)`` computes three scalar
   scores (panorama / method / task) as weighted sums of context
   features, discretises each into the 3-tier enum, and picks a
   ``ParticipationFlowKind`` from the feature profile.
3. ``readout_cognitive_depth_hint(context)`` picks one of five
   depth tiers from a cost-vs-tension scoring surface.
4. Both return a fresh ``ParticipationHint`` / ``CognitiveDepthHint``
   with a ``rationale`` string listing the top contributing
   features, so product code and family report can audit
   WHY the hint landed on a given level.

Design notes:

* **No string matching.** ``regime_id`` is one of the feature
  signals (via template embedding distances), but no branch
  keys off the ``regime_id`` value itself. The scoring is
  defined over the feature axes the regime module already
  computes; it does not care what the regime is called.
* **Deterministic + stateless.** Pure functions of the context.
  A future learned-weights layer is a drop-in replacement: the
  same signature, the same discretisation boundaries, the same
  context features \u2014 only the weights change.
* **Confidence reflects evidence.** ``confidence`` is low when
  the context says "cold start" (no dual_track / evaluation
  data); high when drives are distinctive + candidate regime
  distribution is sharp. This replaces slice 1's flat 0.4.
* **Back-compat fallback.** When the context has no usable
  runtime data (``has_dual_track=False`` AND ``has_evaluation=False``),
  the readout falls back to the slice-1 scaffold. That keeps
  cold-boot deployments behaving exactly as before.

See ``docs/specs/cognition-regime.md`` and Gap 8 in
``docs/implementation/13_emogpt_prd_alignment_upgrade.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from volvence_zero.regime.identity import (
    CognitiveDepth,
    CognitiveDepthHint,
    ParticipationFlowKind,
    ParticipationHint,
    ParticipationLevel,
    derive_cognitive_depth_hint,
    derive_participation_hint,
)

if TYPE_CHECKING:
    from volvence_zero.dual_track import DualTrackSnapshot
    from volvence_zero.evaluation.backbone import EvaluationSnapshot
    from volvence_zero.memory import MemorySnapshot
    from volvence_zero.prediction.error import PredictionErrorSnapshot


# ---------------------------------------------------------------------------
# Discretisation boundaries
# ---------------------------------------------------------------------------

# Per-section score -> ParticipationLevel. The three-tier cut:
#
# * score < 0.30  -> SILENT  (drop the section entirely)
# * 0.30 <= s < 0.65  -> BRIEF (minimal posture / placeholder)
# * s >= 0.65  -> STRUCTURED (full rendering)
#
# These numbers come from the spec: a downstream consumer that
# misinterprets BRIEF as SILENT loses at most one section; a
# downstream that misinterprets BRIEF as STRUCTURED adds verbosity
# but doesn't break correctness.
_PARTICIPATION_THRESHOLD_SILENT: float = 0.30
_PARTICIPATION_THRESHOLD_STRUCTURED: float = 0.65

# Cognitive-depth thresholds over the ``total_pressure`` scalar.
_DEPTH_THRESHOLD_REFLEXIVE: float = 0.15
_DEPTH_THRESHOLD_SHALLOW: float = 0.35
_DEPTH_THRESHOLD_FOCUSED: float = 0.60
_DEPTH_THRESHOLD_ALERT: float = 0.80


# ---------------------------------------------------------------------------
# Context dataclass \u2014 the feature surface the readout sees
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HintReadoutContext:
    """Immutable feature bundle passed to the hint readout.

    Deliberately flat + all scalar (not references to snapshot
    objects) so the readout is provably stateless and the context
    is trivial to log / unit-test. Every field has a documented
    range / meaning and a safe default for the cold-start case.
    """

    # Regime metadata \u2014 the categorical anchor. Used ONLY for
    # populating the returned rationale string (so product code can
    # read "readout saw regime=problem_solving") and for the
    # scaffold fallback when runtime data is absent. NOT used as
    # a primary decision branch.
    regime_id: str = ""
    turns_in_current_regime: int = 0
    candidate_sharpness: float = 0.0  # top1 - top2; in [0,1]

    # Availability flags (so the readout can lower confidence when
    # runtime signals are missing).
    has_dual_track: bool = False
    has_evaluation: bool = False
    has_prediction_error: bool = False
    has_memory: bool = False

    # Tension axes (clamped to [0,1])
    cross_track_tension: float = 0.0
    world_tension: float = 0.0
    self_tension: float = 0.0

    # Controller-code drives (clamped to [0,1])
    world_drive: float = 0.0
    self_drive: float = 0.0
    shared_drive: float = 0.0
    switch_pressure: float = 0.0

    # Abstract-action biases (clamped to [0,1], NOT a distribution;
    # each is computed independently from hints present on the two
    # tracks).
    repair_bias: float = 0.0
    task_bias: float = 0.0
    exploration_bias: float = 0.0
    stabilize_bias: float = 0.0

    # Evaluation readouts (each in [0,1]).
    warmth: float = 0.5
    support_presence: float = 0.5
    task_pressure: float = 0.5
    cross_track_stability: float = 0.5
    info_integration: float = 0.5

    # Prediction error (magnitude in [0,+inf), others signed).
    pe_magnitude: float = 0.0
    pe_relationship_error: float = 0.0
    pe_task_error: float = 0.0

    # Memory presence (normalised by a generous ceiling so a full
    # stack of 3+ world entries saturates to 1.0).
    world_presence: float = 0.0
    self_presence: float = 0.0

    def evidence_score(self) -> float:
        """Heuristic [0,1] score for "how much data did we see".

        Used to scale returned ``confidence``. Cold start without
        dual_track / evaluation snaps to ~0.1; all signals
        available + sharp candidate distribution saturates near
        1.0.
        """
        score = 0.1
        if self.has_dual_track:
            score += 0.30
        if self.has_evaluation:
            score += 0.20
        if self.has_prediction_error:
            score += 0.15
        if self.has_memory:
            score += 0.10
        # Sharp candidate distribution = we're more confident in
        # the active regime, which implies more confident hints.
        score += 0.15 * min(max(self.candidate_sharpness, 0.0), 1.0)
        return round(min(score, 1.0), 4)


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _controller_drive_components(
    dual_track: "DualTrackSnapshot | None",
) -> tuple[float, float, float, float]:
    """Extract (world_drive, self_drive, shared_drive, switch_pressure).

    Matches the existing ``_controller_profile`` helper in
    ``regime.identity`` so the feature semantics are identical; we
    re-implement here to keep the readout module self-contained.
    """
    if dual_track is None:
        return (0.0, 0.0, 0.0, 0.0)
    world_code = dual_track.world_track.controller_code
    self_code = dual_track.self_track.controller_code
    world_drive = world_code[0] if len(world_code) > 0 else 0.0
    self_drive = self_code[0] if len(self_code) > 0 else 0.0
    shared_drive = (
        ((world_code[1] if len(world_code) > 1 else 0.0)
         + (self_code[1] if len(self_code) > 1 else 0.0))
        / 2.0
    )
    switch_pressure = max(
        world_code[2] if len(world_code) > 2 else 0.0,
        self_code[2] if len(self_code) > 2 else 0.0,
    )
    return (
        _clamp01(world_drive),
        _clamp01(self_drive),
        _clamp01(shared_drive),
        _clamp01(switch_pressure),
    )


def _abstract_action_bias_components(
    dual_track: "DualTrackSnapshot | None",
) -> tuple[float, float, float, float]:
    """Extract (repair_bias, task_bias, exploration_bias, stabilize_bias).

    Proportion of the two tracks' non-None abstract_action_hint
    values that match each controller kind. Per-bias in [0,1];
    the four biases sum to <= 1 only when at least one hint is set.
    """
    if dual_track is None:
        return (0.0, 0.0, 0.0, 0.0)
    hints = tuple(
        hint
        for hint in (
            dual_track.world_track.abstract_action_hint,
            dual_track.self_track.abstract_action_hint,
        )
        if hint is not None
    )
    if not hints:
        return (0.0, 0.0, 0.0, 0.0)
    repair_bias = sum(1.0 for h in hints if h == "repair_controller") / len(hints)
    task_bias = sum(1.0 for h in hints if h == "task_controller") / len(hints)
    exploration_bias = sum(1.0 for h in hints if h == "exploration_controller") / len(hints)
    stabilize_bias = sum(1.0 for h in hints if h == "stabilize_controller") / len(hints)
    return (
        _clamp01(repair_bias),
        _clamp01(task_bias),
        _clamp01(exploration_bias),
        _clamp01(stabilize_bias),
    )


def _evaluation_metric(
    evaluation: "EvaluationSnapshot | None", metric_name: str, default: float
) -> float:
    if evaluation is None:
        return default
    for score in evaluation.turn_scores:
        if score.metric_name == metric_name:
            return float(score.value)
    return default


def _memory_presence(
    memory: "MemorySnapshot | None",
) -> tuple[float, float]:
    """Return (world_presence, self_presence) normalised to [0,1] with 3 as saturation."""
    if memory is None:
        return (0.0, 0.0)
    from volvence_zero.memory import Track
    world_count = sum(
        1 for entry in memory.retrieved_entries if entry.track is Track.WORLD
    )
    self_count = sum(
        1 for entry in memory.retrieved_entries if entry.track is Track.SELF
    )
    return (_clamp01(world_count / 3.0), _clamp01(self_count / 3.0))


def _candidate_sharpness(candidates: tuple[tuple[str, float], ...]) -> float:
    """top1 - top2 score gap, clamped to [0,1].

    Sharp = 1.0 means top1 is dominant; 0.0 means the top two are tied.
    Single-candidate cases return 1.0 (fully sharp).
    """
    if len(candidates) == 0:
        return 0.0
    if len(candidates) == 1:
        return 1.0
    return _clamp01(candidates[0][1] - candidates[1][1])


def build_hint_readout_context(
    *,
    regime_id: str,
    turns_in_current_regime: int,
    candidates: tuple[tuple[str, float], ...],
    memory: "MemorySnapshot | None",
    dual_track: "DualTrackSnapshot | None",
    evaluation: "EvaluationSnapshot | None",
    prediction_error: "PredictionErrorSnapshot | None",
) -> HintReadoutContext:
    """Collect runtime signals into a ``HintReadoutContext``.

    All arguments are optional snapshots; missing ones flip the
    corresponding ``has_*`` flag to ``False`` and the feature
    defaults take over. A cold-start context (everything None)
    yields low evidence_score so the readout can either fall
    back to the scaffold or publish a low-confidence hint.
    """
    cross = 0.0
    world_t = 0.0
    self_t = 0.0
    if dual_track is not None:
        cross = _clamp01(dual_track.cross_track_tension)
        world_t = _clamp01(dual_track.world_track.tension_level)
        self_t = _clamp01(dual_track.self_track.tension_level)
    world_d, self_d, shared_d, switch_p = _controller_drive_components(dual_track)
    repair_b, task_b, explor_b, stab_b = _abstract_action_bias_components(dual_track)
    pe_mag = 0.0
    pe_rel = 0.0
    pe_task_err = 0.0
    if prediction_error is not None:
        pe_mag = float(prediction_error.error.magnitude)
        pe_rel = float(prediction_error.error.relationship_error)
        pe_task_err = float(prediction_error.error.task_error)
    world_p, self_p = _memory_presence(memory)
    warmth = _evaluation_metric(evaluation, "warmth", default=0.5)
    support = _evaluation_metric(evaluation, "support_presence", default=warmth)
    task_pressure = _evaluation_metric(evaluation, "task_pressure", default=0.5)
    stability = _evaluation_metric(evaluation, "cross_track_stability", default=0.5)
    info = _evaluation_metric(evaluation, "info_integration", default=0.5)
    return HintReadoutContext(
        regime_id=regime_id,
        turns_in_current_regime=turns_in_current_regime,
        candidate_sharpness=_candidate_sharpness(candidates),
        has_dual_track=dual_track is not None,
        has_evaluation=evaluation is not None,
        has_prediction_error=prediction_error is not None,
        has_memory=memory is not None,
        cross_track_tension=cross,
        world_tension=world_t,
        self_tension=self_t,
        world_drive=world_d,
        self_drive=self_d,
        shared_drive=shared_d,
        switch_pressure=switch_p,
        repair_bias=repair_b,
        task_bias=task_b,
        exploration_bias=explor_b,
        stabilize_bias=stab_b,
        warmth=warmth,
        support_presence=support,
        task_pressure=task_pressure,
        cross_track_stability=stability,
        info_integration=info,
        pe_magnitude=pe_mag,
        pe_relationship_error=pe_rel,
        pe_task_error=pe_task_err,
        world_presence=world_p,
        self_presence=self_p,
    )


# ---------------------------------------------------------------------------
# Readout functions
# ---------------------------------------------------------------------------


def _score_to_level(score: float) -> ParticipationLevel:
    if score < _PARTICIPATION_THRESHOLD_SILENT:
        return ParticipationLevel.SILENT
    if score < _PARTICIPATION_THRESHOLD_STRUCTURED:
        return ParticipationLevel.BRIEF
    return ParticipationLevel.STRUCTURED


def _select_flow_kind(ctx: HintReadoutContext) -> ParticipationFlowKind:
    """Select flow kind from feature profile.

    Each kind has a "strength" score; argmax wins. Ties prefer
    the milder tier (SOCIAL/INFO beats PROBLEM) to match the
    spec's "round down" disposition in ambiguous cases.
    """
    # SOCIAL: low tension + modest warmth + low drives + stabilize bias dominant.
    social = (
        0.30 * (1.0 - max(ctx.cross_track_tension, ctx.world_tension, ctx.self_tension))
        + 0.25 * ctx.warmth
        + 0.15 * ctx.stabilize_bias
        - 0.20 * ctx.task_bias
        - 0.20 * ctx.repair_bias
        - 0.10 * ctx.switch_pressure
    )
    # ACQUAINTANCE: moderate self presence / drive, low pressure from either side.
    acquaintance = (
        0.30 * ctx.self_presence
        + 0.20 * ctx.self_drive
        + 0.15 * ctx.warmth
        + 0.10 * ctx.cross_track_stability
        - 0.15 * ctx.task_bias
        - 0.10 * ctx.cross_track_tension
    )
    # PROBLEM: world drive + task bias + task pressure, low relationship cost.
    problem = (
        0.30 * ctx.world_drive
        + 0.25 * ctx.task_bias
        + 0.20 * ctx.task_pressure
        + 0.10 * ctx.world_presence
        - 0.10 * ctx.self_tension
    )
    # TASK: extreme task dominance + world-side presence. Stricter than PROBLEM.
    task_dominance = max(ctx.task_pressure - ctx.support_presence, 0.0)
    task = (
        0.35 * ctx.world_drive
        + 0.30 * ctx.task_bias
        + 0.25 * task_dominance
        + 0.10 * ctx.world_presence
        - 0.15 * ctx.repair_bias
    )
    # INFO: default middle ground when no axis dominates.
    info = 0.30 + 0.10 * ctx.info_integration

    buckets: tuple[tuple[ParticipationFlowKind, float], ...] = (
        (ParticipationFlowKind.SOCIAL, social),
        (ParticipationFlowKind.ACQUAINTANCE, acquaintance),
        (ParticipationFlowKind.INFO, info),
        (ParticipationFlowKind.PROBLEM, problem),
        (ParticipationFlowKind.TASK, task),
    )
    # Stable argmax with preference for order (tie -> earlier -> milder).
    best_kind, best_score = buckets[0]
    for kind, score in buckets[1:]:
        if score > best_score:
            best_kind = kind
            best_score = score
    return best_kind


def _panorama_score(ctx: HintReadoutContext) -> float:
    # Panorama ("current frame") wants to be present when the
    # lifeform needs situational awareness: on a fresh regime switch,
    # when world presence is high, when evaluation is unstable.
    # It should be quiet when the scene is chitchat or pure-dyad
    # emotional support.
    pull = (
        0.30 * ctx.world_presence
        + 0.25 * ctx.switch_pressure
        + 0.18 * ctx.task_bias
        + 0.12 * ctx.world_drive
        + 0.15 * (1.0 - ctx.cross_track_stability)
        + 0.10 * ctx.exploration_bias
    )
    push = (
        0.20 * ctx.stabilize_bias
        + 0.25 * (1.0 - max(ctx.world_drive, ctx.task_bias))
        + 0.10 * ctx.self_tension
        + 0.20 * max(ctx.repair_bias, 0.0)  # repair focus narrows panorama
    )
    baseline = 0.40 + 0.10 * ctx.info_integration
    return _clamp01(baseline + 0.6 * pull - 0.7 * push)


def _method_score(ctx: HintReadoutContext) -> float:
    # Method ("how we're approaching this") wants to render when
    # there is a recognisable controller bias (task/exploration/
    # repair); it fades when we're in stabilize / low drive mode.
    pull = (
        0.22 * ctx.task_bias
        + 0.20 * ctx.exploration_bias
        + 0.20 * ctx.repair_bias
        + 0.15 * ctx.world_drive
        + 0.12 * ctx.self_drive
        + 0.12 * ctx.shared_drive
        + 0.15 * ctx.switch_pressure
    )
    push = (
        0.28 * ctx.stabilize_bias
        + 0.18 * (1.0 - max(ctx.world_drive, ctx.self_drive, ctx.shared_drive))
        + 0.10 * ctx.warmth  # warm chitchat downweights method
    )
    baseline = 0.45
    return _clamp01(baseline + 0.5 * pull - 0.55 * push)


def _task_score(ctx: HintReadoutContext) -> float:
    # Task ("next step") wants to render when there's a concrete
    # forward motion: world drive + task bias; it should fade under
    # relationship crisis (high cross_track_tension, repair bias),
    # under emotional-support self-tension, and under stabilize
    # chitchat (high stabilize_bias + low drives).
    pull = (
        0.30 * ctx.task_bias
        + 0.25 * ctx.world_drive
        + 0.20 * ctx.task_pressure
        + 0.15 * ctx.world_presence
        + 0.10 * ctx.exploration_bias
    )
    push = (
        0.28 * ctx.repair_bias
        + 0.26 * ctx.cross_track_tension
        + 0.20 * ctx.self_tension
        + 0.16 * (1.0 - ctx.cross_track_stability)
        + 0.22 * ctx.stabilize_bias
    )
    baseline = 0.40 + 0.10 * ctx.info_integration
    return _clamp01(baseline + 0.6 * pull - 0.7 * push)


def _build_rationale(
    *,
    regime_id: str,
    feature_contributions: tuple[tuple[str, float], ...],
    tag: str,
) -> str:
    """Build an auditable rationale string.

    Lists the top 4 contributing features (by absolute value)
    plus the regime name, tagged with the readout version so a
    future slice-3 learned-weights readout can be distinguished
    in logs.
    """
    top = sorted(
        feature_contributions, key=lambda kv: abs(kv[1]), reverse=True
    )[:4]
    pieces = [f"{name}={value:+.2f}" for name, value in top]
    return f"{tag}:{regime_id}:" + ",".join(pieces)


def readout_participation_hint(ctx: HintReadoutContext) -> ParticipationHint:
    """Compute a ``ParticipationHint`` from the context.

    Uses ``_panorama_score`` / ``_method_score`` / ``_task_score``
    and discretises each to the 3-tier level. The rationale lists
    the top contributing features so downstream audit + family
    report can see WHY the hint landed this way.

    Fallback: when neither dual_track nor evaluation is available,
    the readout has nothing actionable to work with. We defer to
    the slice-1 scaffold; this preserves the cold-start behaviour
    the existing tests expect.
    """
    if not (ctx.has_dual_track or ctx.has_evaluation):
        fallback = derive_participation_hint(ctx.regime_id)
        # Re-tag rationale to make it clear this is a fallback, not
        # a learned readout; and lower confidence because we had no
        # runtime signal.
        return ParticipationHint(
            flow_kind=fallback.flow_kind,
            panorama_level=fallback.panorama_level,
            method_level=fallback.method_level,
            task_level=fallback.task_level,
            confidence=round(min(fallback.confidence, 0.30), 4),
            rationale=f"readout:cold-fallback:{fallback.rationale}",
        )
    panorama = _panorama_score(ctx)
    method = _method_score(ctx)
    task = _task_score(ctx)
    flow_kind = _select_flow_kind(ctx)
    contributions = (
        ("panorama", panorama),
        ("method", method),
        ("task", task),
        ("cross_track_tension", -ctx.cross_track_tension),
        ("world_drive", ctx.world_drive),
        ("task_bias", ctx.task_bias),
        ("repair_bias", -ctx.repair_bias),
        ("stabilize_bias", -ctx.stabilize_bias),
        ("switch_pressure", ctx.switch_pressure),
    )
    rationale = _build_rationale(
        regime_id=ctx.regime_id,
        feature_contributions=contributions,
        tag="readout.v1",
    )
    return ParticipationHint(
        flow_kind=flow_kind,
        panorama_level=_score_to_level(panorama),
        method_level=_score_to_level(method),
        task_level=_score_to_level(task),
        confidence=round(0.25 + 0.70 * ctx.evidence_score(), 4),
        rationale=rationale,
    )


def _depth_pressure(ctx: HintReadoutContext) -> float:
    """Scalar "how hard should this turn think" score in [0,1]."""
    pull = (
        0.22 * ctx.task_bias
        + 0.18 * ctx.exploration_bias
        + 0.24 * ctx.cross_track_tension
        + 0.16 * ctx.switch_pressure
        + 0.18 * ctx.repair_bias
        + 0.14 * max(ctx.world_drive, ctx.self_drive)
        + 0.10 * ctx.task_pressure
        + 0.10 * ctx.pe_magnitude
    )
    push = (
        0.28 * ctx.stabilize_bias
        + 0.16 * ctx.warmth
        + 0.18 * (1.0 - max(ctx.task_bias, ctx.repair_bias, ctx.exploration_bias))
    )
    baseline = 0.35
    return _clamp01(baseline + 0.70 * pull - 0.50 * push)


def _pressure_to_depth(pressure: float) -> CognitiveDepth:
    if pressure < _DEPTH_THRESHOLD_REFLEXIVE:
        return CognitiveDepth.REFLEXIVE
    if pressure < _DEPTH_THRESHOLD_SHALLOW:
        return CognitiveDepth.SHALLOW
    if pressure < _DEPTH_THRESHOLD_FOCUSED:
        return CognitiveDepth.FOCUSED
    if pressure < _DEPTH_THRESHOLD_ALERT:
        return CognitiveDepth.ALERT
    return CognitiveDepth.DEEP


def readout_cognitive_depth_hint(ctx: HintReadoutContext) -> CognitiveDepthHint:
    """Compute a ``CognitiveDepthHint`` from the context.

    Maps a scalar "pressure" score onto the 5-tier depth ladder.
    Uses the same cold-start fallback semantics as
    ``readout_participation_hint``.
    """
    if not (ctx.has_dual_track or ctx.has_evaluation):
        fallback = derive_cognitive_depth_hint(ctx.regime_id)
        return CognitiveDepthHint(
            depth=fallback.depth,
            rationale=f"readout:cold-fallback:{fallback.rationale}",
            confidence=round(min(fallback.confidence, 0.30), 4),
        )
    pressure = _depth_pressure(ctx)
    depth = _pressure_to_depth(pressure)
    contributions = (
        ("pressure", pressure),
        ("cross_track_tension", ctx.cross_track_tension),
        ("task_bias", ctx.task_bias),
        ("repair_bias", ctx.repair_bias),
        ("stabilize_bias", -ctx.stabilize_bias),
        ("switch_pressure", ctx.switch_pressure),
        ("pe_magnitude", ctx.pe_magnitude),
    )
    rationale = _build_rationale(
        regime_id=ctx.regime_id,
        feature_contributions=contributions,
        tag="readout.v1.depth",
    )
    return CognitiveDepthHint(
        depth=depth,
        rationale=rationale,
        confidence=round(0.25 + 0.70 * ctx.evidence_score(), 4),
    )


__all__ = [
    "HintReadoutContext",
    "build_hint_readout_context",
    "readout_cognitive_depth_hint",
    "readout_participation_hint",
]
