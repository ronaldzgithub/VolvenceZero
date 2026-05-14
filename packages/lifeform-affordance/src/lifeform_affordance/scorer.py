"""Gap 1 slice 3: metacontroller-driven affordance scoring.

Replaces the slice-1 ``build_neutral_snapshot`` (all candidates at
score=0.5) with a **continuous-feature scorer**. Same rule the Gap 8
slice 2 hint readout follows: decisions flow from scalar features
(controller drives, abstract-action biases, regime + cognitive
depth hints, cross-track tension) rather than from ``dict[name ->
score]`` lookups.

Four hard invariants:

1. **No branch keys off ``descriptor.name``.** The name shows up
   only in audit rationale strings. Scoring consumes categorical
   fields the descriptor already declares (``kind`` / ``affordance_tags``
   / ``cost_model.latency_class`` / ``safety_model.*``) plus the
   runtime context's continuous features.
2. **``lifeform-affordance`` does NOT import ``vz-cognition``.**
   The context is a plain frozen dataclass with scalar fields; the
   caller (a lifeform-side adapter) builds it from whichever
   snapshots are available. This preserves the wheel-boundary from
   ``pyproject.toml`` (we depend on ``vz-contracts + lifeform-core``
   only).
3. **Snapshot-level ``blocked_reason`` = regime block ONLY.**
   Consent grants are host-level / tenant-level. The invoker
   enforces them at call time (see slice 2a ``DescriptorDerivedBoundaryPolicy``);
   surfacing "consent missing" in the snapshot would leak host
   policy into an advisory surface.
4. **Deterministic + stateless.** ``score_affordance`` is a pure
   function. A future slice-4 learned-weights scorer is a drop-in
   replacement: same signature, same ``AffordanceCandidate`` output,
   only the weights change.

See ``docs/specs/affordance.md`` \u00a7 "slice 3: metacontroller scoring"
and Gap 1 in ``docs/implementation/13_emogpt_prd_alignment_upgrade.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

from volvence_zero.affordance import (
    AffordanceDescriptor,
    AffordanceKind,
    AffordanceLatencyClass,
)

from lifeform_affordance.registry import AffordanceRegistry
from lifeform_affordance.snapshot import AffordanceCandidate, AffordanceSnapshot


# Mirror of ``CognitiveDepth.REFLEXIVE`` / ``CognitiveDepth.SHALLOW``
# string values from ``vz-cognition.regime.hints``. Kept as string
# literals here so ``lifeform-affordance`` stays framework-agnostic
# per the ``AffordanceScoringContext`` docstring (no import of the
# ``vz-cognition.regime`` enum, keeping the wheel graph thin). If
# ``CognitiveDepth`` gains a new shallow-side member, mirror it here
# AND keep the comment pointing back to the kernel enum so a
# repo-wide grep stays meaningful.
_LOW_DEPTH_VALUES: frozenset[str] = frozenset({"reflexive", "shallow"})


# ---------------------------------------------------------------------------
# Scoring context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AffordanceScoringContext:
    """Runtime feature bundle consumed by ``score_affordance``.

    Every field is a plain string / float / int / bool so the
    scorer doesn't need to import kernel snapshot types. The
    lifeform-side builder (``build_scoring_context_from_session``
    in a companion adapter module) translates real kernel
    ``RegimeSnapshot`` / ``DualTrackSnapshot`` fields into this
    shape.

    ``flow_kind`` / ``cognitive_depth`` are deliberately strings,
    not enums, so the scorer remains framework-agnostic. The set
    of valid values matches ``ParticipationFlowKind`` /
    ``CognitiveDepth`` from ``vz-cognition.regime`` \u2014 but we
    don't import those to keep the wheel graph thin.

    ``evidence`` is a [0,1] confidence multiplier: low when the
    builder had to synthesise defaults (cold start) and the
    caller should trust the scorer less.
    """

    active_regime_id: str = ""
    flow_kind: str = "info"
    cognitive_depth: str = "focused"
    turns_in_current_regime: int = 0

    # Controller drives (from dual_track.controller_code).
    world_drive: float = 0.0
    self_drive: float = 0.0
    shared_drive: float = 0.0
    switch_pressure: float = 0.0

    # Abstract-action biases (from dual_track.abstract_action_hint).
    task_bias: float = 0.0
    repair_bias: float = 0.0
    exploration_bias: float = 0.0
    stabilize_bias: float = 0.0

    # Cross-track tension (for high-pressure dampening).
    cross_track_tension: float = 0.0

    # Evidence multiplier: scales the final score pull/push so
    # cold-start snapshots stay close to neutral.
    evidence: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "world_drive",
            "self_drive",
            "shared_drive",
            "switch_pressure",
            "task_bias",
            "repair_bias",
            "exploration_bias",
            "stabilize_bias",
            "cross_track_tension",
            "evidence",
        ):
            value = getattr(self, name)
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(
                    f"AffordanceScoringContext.{name} must be in [0,1], "
                    f"got {value!r}"
                )
        if self.turns_in_current_regime < 0:
            raise ValueError(
                f"AffordanceScoringContext.turns_in_current_regime must be "
                f">= 0, got {self.turns_in_current_regime!r}"
            )


# ---------------------------------------------------------------------------
# Scoring function
# ---------------------------------------------------------------------------


_NEUTRAL_BASE: float = 0.40


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _kind_affinity(
    kind: AffordanceKind, ctx: AffordanceScoringContext
) -> float:
    """Per-kind pull score from the context's drive / bias axes.

    TOOL is the workhorse: boosted by task / exploration biases
    and by world drive. ACTION (internal ops like "clarify" /
    "commit") is boosted by shared / switch pressure because
    those are the moments that need conscious routing. ORGAN
    (composed capability) needs exploration bias. SHELL is
    deployment-scoped, neutral.
    """
    if kind is AffordanceKind.TOOL:
        return (
            0.25 * ctx.task_bias
            + 0.20 * ctx.exploration_bias
            + 0.18 * ctx.world_drive
            - 0.18 * ctx.stabilize_bias
        )
    if kind is AffordanceKind.ACTION:
        return (
            0.18 * ctx.shared_drive
            + 0.15 * ctx.switch_pressure
            + 0.12 * ctx.repair_bias
        )
    if kind is AffordanceKind.ORGAN:
        return (
            0.22 * ctx.exploration_bias
            + 0.12 * ctx.shared_drive
        )
    # SHELL: neutral (deployment-layer capability).
    return 0.0


def _latency_penalty(
    cost_latency: AffordanceLatencyClass, cognitive_depth: str
) -> float:
    """Push score DOWN when a slow tool is proposed at shallow depth.

    REFLEXIVE / SHALLOW cognitive depth means "spend little
    compute"; recommending a SLOW / VERY_SLOW tool contradicts
    the depth hint. The sizes are small enough not to outright
    block; the selected filter can still pick a SLOW tool if
    the kind + task fit is strong.
    """
    if cognitive_depth in _LOW_DEPTH_VALUES:
        if cost_latency is AffordanceLatencyClass.VERY_SLOW:
            return 0.35
        if cost_latency is AffordanceLatencyClass.SLOW:
            return 0.18
    return 0.0


def _safety_penalty(
    descriptor: AffordanceDescriptor, ctx: AffordanceScoringContext
) -> float:
    """Score penalty for high-risk descriptors under low-confidence conditions.

    Irreversible + confirmation-required descriptors should NOT
    score high when evidence is low, cross-track tension is high,
    or we've only been in this regime for one turn. This prevents
    the scorer from recommending write_file / delete-type ops on
    the second turn of a fresh scene before the lifeform has
    enough context.
    """
    safety = descriptor.safety_model
    if not (safety.irreversible or safety.requires_user_confirmation):
        return 0.0
    # Multiplier: high under risky conditions, low otherwise.
    risk_multiplier = (
        0.40 * (1.0 - ctx.evidence)
        + 0.30 * ctx.cross_track_tension
        + 0.30 * (1.0 if ctx.turns_in_current_regime < 2 else 0.0)
    )
    return 0.20 * _clamp01(risk_multiplier)


def _tag_affinity(descriptor: AffordanceDescriptor, ctx: AffordanceScoringContext) -> float:
    """Tag-driven adjustments.

    Coarse and intentionally small; the heavy lift is in
    ``_kind_affinity``. Tags that signal "code / filesystem /
    test / search" align with task_bias / exploration_bias;
    "social" tags would flip sign (none today but future
    ACTION / SHELL affordances may have them).
    """
    tags = set(descriptor.affordance_tags)
    delta = 0.0
    if {"code", "filesystem", "read", "list", "search"} & tags:
        delta += 0.06 * ctx.task_bias + 0.06 * ctx.exploration_bias
    if {"write", "execute", "test"} & tags:
        # Heavier commit: only valuable when engaged + task-heavy.
        delta += 0.08 * ctx.task_bias + 0.06 * ctx.world_drive
    if {"social", "rapport"} & tags:
        # Penalise surfacing chitchat tools under task pressure.
        delta -= 0.10 * ctx.task_bias + 0.08 * ctx.world_drive
    return delta


def _regime_blocked(
    descriptor: AffordanceDescriptor, ctx: AffordanceScoringContext
) -> str:
    """Return a non-empty reason string iff the current regime
    blocks this descriptor; empty string means allowed.

    Mirrors ``DescriptorDerivedBoundaryPolicy.check`` minus the
    consent / confirmation axes (those are host-level). A
    snapshot consumer reads ``blocked_reason`` to decide whether
    to render the candidate at all.
    """
    if descriptor.excluded_from_runtime_selection:
        return "descriptor_excluded"
    if not ctx.active_regime_id:
        return ""
    if ctx.active_regime_id in descriptor.safety_model.blocked_in_regimes:
        return (
            f"regime_blocked:{ctx.active_regime_id} "
            f"in blocked_in_regimes={descriptor.safety_model.blocked_in_regimes!r}"
        )
    return ""


def _build_rationale(
    descriptor: AffordanceDescriptor,
    *,
    base: float,
    kind_pull: float,
    tag_delta: float,
    latency_push: float,
    safety_push: float,
    final_score: float,
) -> str:
    """Audit-friendly rationale string.

    Includes version tag so a future ``scorer.v2`` can be
    distinguished in logs. Components ordered from largest
    contribution to smallest for operator scanning.
    """
    parts: list[tuple[str, float]] = [
        ("kind_pull", kind_pull),
        ("tag_delta", tag_delta),
        ("latency_push", -latency_push),
        ("safety_push", -safety_push),
    ]
    parts.sort(key=lambda kv: abs(kv[1]), reverse=True)
    pieces = [f"{name}={value:+.2f}" for name, value in parts]
    return (
        f"scorer.v1:{descriptor.kind.value}:{descriptor.name}:"
        f"base={base:.2f},final={final_score:.2f},"
        + ",".join(pieces)
    )


def score_affordance(
    descriptor: AffordanceDescriptor,
    ctx: AffordanceScoringContext,
) -> AffordanceCandidate:
    """Compute one ``AffordanceCandidate`` from a descriptor + context.

    Pure function, no side effects. If ``_regime_blocked`` fires,
    ``score=0.0`` and ``blocked_reason`` is populated; else
    score = clamp(base + evidence * (kind_pull + tag_delta) -
    evidence * (latency_push + safety_push), 0, 1).

    Evidence multiplier gates how far the scorer moves away from
    the neutral base when context is cold; with ``evidence=0`` the
    score stays at exactly 0.40 (slightly under the ~0.5 neutral
    so cold-start doesn't look confidently selected).
    """
    blocked_reason = _regime_blocked(descriptor, ctx)
    if blocked_reason:
        return AffordanceCandidate(
            descriptor_name=descriptor.name,
            score=0.0,
            rationale=f"scorer.v1:blocked:{blocked_reason}",
            expected_cost=descriptor.cost_model,
            blocked_reason=blocked_reason,
        )
    kind_pull = _kind_affinity(descriptor.kind, ctx)
    tag_delta = _tag_affinity(descriptor, ctx)
    latency_push = _latency_penalty(
        descriptor.cost_model.latency_class, ctx.cognitive_depth
    )
    safety_push = _safety_penalty(descriptor, ctx)
    final = _clamp01(
        _NEUTRAL_BASE
        + ctx.evidence * (kind_pull + tag_delta)
        - ctx.evidence * (latency_push + safety_push)
    )
    return AffordanceCandidate(
        descriptor_name=descriptor.name,
        score=round(final, 4),
        rationale=_build_rationale(
            descriptor,
            base=_NEUTRAL_BASE,
            kind_pull=kind_pull,
            tag_delta=tag_delta,
            latency_push=latency_push,
            safety_push=safety_push,
            final_score=final,
        ),
        expected_cost=descriptor.cost_model,
    )


# ---------------------------------------------------------------------------
# Selection threshold for ``selected``
# ---------------------------------------------------------------------------


_SELECTION_MIN_SCORE: float = 0.50
_SELECTION_MIN_MARGIN: float = 0.06
"""Pick the top scorer as ``selected`` only when
(a) its score >= 0.50 AND
(b) it beats the second-best by >= 0.06.

Both thresholds prevent the scorer from "confidently selecting"
when the differences are noise. An un-selected snapshot is NOT a
failure \u2014 it simply says "no affordance stood out this turn;
don't proactively invoke".
"""


def _pick_selected(
    candidates: tuple[AffordanceCandidate, ...],
) -> AffordanceCandidate | None:
    unblocked = [c for c in candidates if not c.is_blocked]
    if not unblocked:
        return None
    unblocked.sort(key=lambda c: c.score, reverse=True)
    top = unblocked[0]
    if top.score < _SELECTION_MIN_SCORE:
        return None
    if len(unblocked) >= 2:
        runner_up = unblocked[1]
        if (top.score - runner_up.score) < _SELECTION_MIN_MARGIN:
            return None
    return top


# ---------------------------------------------------------------------------
# Snapshot builder
# ---------------------------------------------------------------------------


def build_scored_snapshot(
    registry: AffordanceRegistry,
    ctx: AffordanceScoringContext,
    *,
    include_excluded_from_runtime_selection: bool = False,
) -> AffordanceSnapshot:
    """Replace the slice-1 ``build_neutral_snapshot`` scaffold.

    Every registered descriptor is scored via ``score_affordance``
    and the top-scoring candidate (subject to a hard min score
    + min margin) is placed in ``selected``. Blocked candidates
    are still present in ``candidates_for_turn`` (with score=0)
    so the snapshot is a faithful audit of what the scorer saw.

    ``include_excluded_from_runtime_selection`` mirrors the old
    builder's flag: testing / family-report paths that want to
    see EVERY descriptor including ones marked for runtime exclusion
    can opt in. Default is False.
    """
    available: list[AffordanceDescriptor] = []
    candidates: list[AffordanceCandidate] = []
    for d in registry.all_descriptors():
        if (
            not include_excluded_from_runtime_selection
            and d.excluded_from_runtime_selection
        ):
            continue
        available.append(d)
        candidates.append(score_affordance(d, ctx))
    candidate_tuple = tuple(candidates)
    selected = _pick_selected(candidate_tuple)
    unblocked_count = sum(1 for c in candidate_tuple if not c.is_blocked)
    description = (
        f"Scored affordance snapshot (scorer.v1); "
        f"{len(available)} available / {unblocked_count} unblocked / "
        f"selected={selected.descriptor_name if selected else None!r}. "
        f"regime={ctx.active_regime_id!r} flow={ctx.flow_kind!r} "
        f"depth={ctx.cognitive_depth!r} evidence={ctx.evidence:.2f}"
    )
    return AffordanceSnapshot(
        available=tuple(available),
        candidates_for_turn=candidate_tuple,
        selected=selected,
        description=description,
    )


# ---------------------------------------------------------------------------
# Duck-typed context builder (optional convenience)
# ---------------------------------------------------------------------------


def _safe_get(obj: object, name: str, default: object) -> object:
    """Getattr on a possibly-None holder, with a default.

    Used so the builder doesn't crash when a caller passes in
    ``None`` for a snapshot (cold start) or a minimal stub in
    tests. Every field has a neutral default in
    ``AffordanceScoringContext``.
    """
    if obj is None:
        return default
    return getattr(obj, name, default)


def _snapshot_value(snapshot: object | None) -> object | None:
    """Unwrap a kernel ``Snapshot`` wrapper into its ``value`` payload.

    When ``snapshot`` is already the raw value (tests passing a
    ``RegimeSnapshot`` instance directly) we return it unchanged;
    when it's a ``Snapshot[X]`` we extract ``.value``. Missing
    inputs pass through as ``None``.
    """
    if snapshot is None:
        return None
    value = getattr(snapshot, "value", None)
    return value if value is not None else snapshot


def build_scoring_context_from_snapshots(
    *,
    regime_snapshot: object | None,
    dual_track_snapshot: object | None,
) -> AffordanceScoringContext:
    """Build an ``AffordanceScoringContext`` from kernel snapshots.

    Duck-typed on purpose so ``lifeform-affordance`` doesn't
    acquire a hard import dependency on ``vz-cognition``. The
    caller passes in the raw snapshot payloads (or ``Snapshot``
    wrappers) and this function extracts the scalar fields. Any
    missing field falls back to the context's default.

    Accepted shapes:

    * ``regime_snapshot.value`` \u2192 ``RegimeSnapshot`` with
      ``active_regime.regime_id`` / ``participation_hint.flow_kind`` /
      ``depth_hint.depth`` / ``turns_in_current_regime``.
    * ``dual_track_snapshot.value`` \u2192 ``DualTrackSnapshot`` with
      ``world_track.controller_code`` / ``world_track.abstract_action_hint`` /
      same for ``self_track`` and ``cross_track_tension``.

    When both are ``None``, the returned context is the
    all-neutral default with ``evidence=0.1``, which makes
    ``build_scored_snapshot`` produce an essentially-flat
    snapshot (all candidates hover near the 0.40 base).
    """
    regime_value = _snapshot_value(regime_snapshot)
    dual_track_value = _snapshot_value(dual_track_snapshot)

    active_regime = _safe_get(regime_value, "active_regime", None)
    regime_id = str(_safe_get(active_regime, "regime_id", ""))
    turns = int(_safe_get(regime_value, "turns_in_current_regime", 0))
    participation = _safe_get(regime_value, "participation_hint", None)
    flow_kind_raw = _safe_get(participation, "flow_kind", None)
    flow_kind = (
        str(getattr(flow_kind_raw, "value", flow_kind_raw))
        if flow_kind_raw is not None
        else "info"
    )
    depth_raw = _safe_get(
        _safe_get(regime_value, "depth_hint", None), "depth", None
    )
    cognitive_depth = (
        str(getattr(depth_raw, "value", depth_raw))
        if depth_raw is not None
        else "focused"
    )

    world_drive = 0.0
    self_drive = 0.0
    shared_drive = 0.0
    switch_pressure = 0.0
    task_bias = 0.0
    repair_bias = 0.0
    exploration_bias = 0.0
    stabilize_bias = 0.0
    cross_tension = 0.0
    if dual_track_value is not None:
        world_track = _safe_get(dual_track_value, "world_track", None)
        self_track = _safe_get(dual_track_value, "self_track", None)
        world_code = tuple(_safe_get(world_track, "controller_code", ()) or ())
        self_code = tuple(_safe_get(self_track, "controller_code", ()) or ())
        world_drive = _clamp01(world_code[0] if len(world_code) > 0 else 0.0)
        self_drive = _clamp01(self_code[0] if len(self_code) > 0 else 0.0)
        shared_drive = _clamp01(
            ((world_code[1] if len(world_code) > 1 else 0.0)
             + (self_code[1] if len(self_code) > 1 else 0.0))
            / 2.0
        )
        switch_pressure = _clamp01(
            max(
                world_code[2] if len(world_code) > 2 else 0.0,
                self_code[2] if len(self_code) > 2 else 0.0,
            )
        )
        cross_tension = _clamp01(
            float(_safe_get(dual_track_value, "cross_track_tension", 0.0))
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
            repair_bias = _clamp01(
                sum(1.0 for h in hints if h == "repair_controller") / len(hints)
            )
            task_bias = _clamp01(
                sum(1.0 for h in hints if h == "task_controller") / len(hints)
            )
            exploration_bias = _clamp01(
                sum(1.0 for h in hints if h == "exploration_controller") / len(hints)
            )
            stabilize_bias = _clamp01(
                sum(1.0 for h in hints if h == "stabilize_controller") / len(hints)
            )

    # Evidence: scale by how much signal we actually got.
    # Both snapshots present -> 0.80; one present -> 0.45; neither -> 0.10.
    if regime_value is not None and dual_track_value is not None:
        evidence = 0.80
    elif regime_value is not None or dual_track_value is not None:
        evidence = 0.45
    else:
        evidence = 0.10

    return AffordanceScoringContext(
        active_regime_id=regime_id,
        flow_kind=flow_kind,
        cognitive_depth=cognitive_depth,
        turns_in_current_regime=max(0, turns),
        world_drive=world_drive,
        self_drive=self_drive,
        shared_drive=shared_drive,
        switch_pressure=switch_pressure,
        task_bias=task_bias,
        repair_bias=repair_bias,
        exploration_bias=exploration_bias,
        stabilize_bias=stabilize_bias,
        cross_track_tension=cross_tension,
        evidence=evidence,
    )


__all__ = [
    "AffordanceScoringContext",
    "build_scored_snapshot",
    "build_scoring_context_from_snapshots",
    "score_affordance",
]
