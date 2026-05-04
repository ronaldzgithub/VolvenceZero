"""Regime scoring helpers and factory functions.

Hosts the per-turn scorer (``score_regimes``), the pure identity
factory (``build_regime_identity``), and the private feature-extraction
helpers they both share. Kept separate from ``RegimeModule`` so the
classifier logic can be unit-tested without instantiating the module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

from volvence_zero.dual_track import DualTrackSnapshot
from volvence_zero.evaluation.types import EvaluationSnapshot
from volvence_zero.memory import MemorySnapshot, Track
from volvence_zero.regime.contracts import RegimeIdentity
from volvence_zero.regime.templates import REGIME_TEMPLATES

if TYPE_CHECKING:
    from volvence_zero.temporal.interface import MetacontrollerRuntimeState


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _metric(evaluation_snapshot: EvaluationSnapshot | None, metric_name: str, default: float = 0.0) -> float:
    if evaluation_snapshot is None:
        return default
    for score in evaluation_snapshot.turn_scores:
        if score.metric_name == metric_name:
            return score.value
    return default


def _track_counts(memory_snapshot: MemorySnapshot | None) -> tuple[int, int]:
    if memory_snapshot is None:
        return (0, 0)
    world_count = sum(1 for entry in memory_snapshot.retrieved_entries if entry.track is Track.WORLD)
    self_count = sum(1 for entry in memory_snapshot.retrieved_entries if entry.track is Track.SELF)
    return (world_count, self_count)


def _controller_profile(dual_track_snapshot: DualTrackSnapshot | None) -> tuple[float, float, float, float]:
    if dual_track_snapshot is None:
        return (0.0, 0.0, 0.0, 0.0)
    world_code = dual_track_snapshot.world_track.controller_code
    self_code = dual_track_snapshot.self_track.controller_code
    world_drive = world_code[0] if len(world_code) > 0 else 0.0
    self_drive = self_code[0] if len(self_code) > 0 else 0.0
    shared_drive = (
        ((world_code[1] if len(world_code) > 1 else 0.0) + (self_code[1] if len(self_code) > 1 else 0.0)) / 2.0
    )
    switch_pressure = (
        max(
            world_code[2] if len(world_code) > 2 else 0.0,
            self_code[2] if len(self_code) > 2 else 0.0,
        )
        if dual_track_snapshot is not None
        else 0.0
    )
    return (
        _clamp(world_drive),
        _clamp(self_drive),
        _clamp(shared_drive),
        _clamp(switch_pressure),
    )


def _abstract_action_profile(dual_track_snapshot: DualTrackSnapshot | None) -> tuple[float, float, float, float]:
    if dual_track_snapshot is None:
        return (0.0, 0.0, 0.0, 0.0)
    hints = tuple(
        hint
        for hint in (
            dual_track_snapshot.world_track.abstract_action_hint,
            dual_track_snapshot.self_track.abstract_action_hint,
        )
        if hint is not None
    )
    if not hints:
        return (0.0, 0.0, 0.0, 0.0)
    repair_bias = sum(1.0 for hint in hints if hint == "repair_controller") / len(hints)
    task_bias = sum(1.0 for hint in hints if hint == "task_controller") / len(hints)
    exploration_bias = sum(1.0 for hint in hints if hint == "exploration_controller") / len(hints)
    stabilize_bias = sum(1.0 for hint in hints if hint == "stabilize_controller") / len(hints)
    return (
        _clamp(repair_bias),
        _clamp(task_bias),
        _clamp(exploration_bias),
        _clamp(stabilize_bias),
    )


def _metacontroller_action_profile(metacontroller_state: "MetacontrollerRuntimeState") -> tuple[float, float, float]:
    track_parameters = {track_name: values for track_name, values in metacontroller_state.track_parameters}
    world_track = track_parameters.get("world", ())
    self_track = track_parameters.get("self", ())
    shared_track = track_parameters.get("shared", ())
    decoder = (
        metacontroller_state.decoder_applied_control
        if metacontroller_state.decoder_applied_control
        else metacontroller_state.decoder_control
    )
    latent = metacontroller_state.latent_mean
    world_bias = (
        (sum(world_track) / len(world_track) if world_track else 0.0) * 0.35
        + (decoder[0] if len(decoder) > 0 else 0.0) * 0.40
        + (latent[0] if len(latent) > 0 else 0.0) * 0.25
    )
    self_bias = (
        (sum(self_track) / len(self_track) if self_track else 0.0) * 0.35
        + (decoder[1] if len(decoder) > 1 else 0.0) * 0.40
        + (latent[1] if len(latent) > 1 else 0.0) * 0.25
    )
    shared_bias = (
        (sum(shared_track) / len(shared_track) if shared_track else 0.0) * 0.35
        + (decoder[2] if len(decoder) > 2 else 0.0) * 0.40
        + (latent[2] if len(latent) > 2 else 0.0) * 0.25
    )
    return (world_bias, self_bias, shared_bias)


def _dominant_abstract_action_context(
    dual_track_snapshot: DualTrackSnapshot | None,
) -> tuple[str | None, int]:
    if dual_track_snapshot is None:
        return (None, 0)
    hints = tuple(
        hint
        for hint in (
            dual_track_snapshot.world_track.abstract_action_hint,
            dual_track_snapshot.self_track.abstract_action_hint,
        )
        if hint is not None
    )
    versions = tuple(
        version
        for version in (
            dual_track_snapshot.world_track.action_family_version_hint,
            dual_track_snapshot.self_track.action_family_version_hint,
        )
        if version > 0
    )
    dominant_hint = hints[0] if hints else None
    dominant_version = max(versions) if versions else 0
    return (dominant_hint, dominant_version)


def _alert_pressure(evaluation_snapshot: EvaluationSnapshot | None) -> float:
    if evaluation_snapshot is None:
        return 0.0
    relevant_alerts = tuple(
        alert
        for alert in evaluation_snapshot.structured_alerts
        if alert.code in {"cross_track_stability_degraded", "rollback_pressure_elevated"}
    )
    if any(alert.severity == "CRITICAL" for alert in relevant_alerts):
        return 1.0
    if any(alert.severity == "HIGH" for alert in relevant_alerts):
        return 0.8
    if relevant_alerts:
        return 0.5
    return 0.0


def score_regimes(
    *,
    memory_snapshot: MemorySnapshot | None,
    dual_track_snapshot: DualTrackSnapshot | None,
    evaluation_snapshot: EvaluationSnapshot | None,
    prediction_error_snapshot: PredictionErrorSnapshot | None = None,
    historical_effectiveness: Mapping[str, float],
    strategy_priors: Mapping[str, float] | None = None,
    selection_weights: Mapping[str, float] | None = None,
    feature_weights: Mapping[str, Mapping[str, float]] | None = None,
    experience_regime_biases: Mapping[str, float] | None = None,
) -> tuple[tuple[str, float], ...]:
    regime_priors = strategy_priors or {}
    fast_regime_biases = experience_regime_biases or {}
    pe_task_shortfall = max(-prediction_error_snapshot.error.task_error, 0.0) if prediction_error_snapshot is not None and not prediction_error_snapshot.bootstrap else 0.0
    pe_relationship_shortfall = max(-prediction_error_snapshot.error.relationship_error, 0.0) if prediction_error_snapshot is not None and not prediction_error_snapshot.bootstrap else 0.0
    pe_regime_shortfall = max(-prediction_error_snapshot.error.regime_error, 0.0) if prediction_error_snapshot is not None and not prediction_error_snapshot.bootstrap else 0.0
    pe_action_shortfall = max(-prediction_error_snapshot.error.action_error, 0.0) if prediction_error_snapshot is not None and not prediction_error_snapshot.bootstrap else 0.0
    if dual_track_snapshot is None:
        base = 0.1
        cold_scores = {
            "casual_social": _clamp(base + historical_effectiveness.get("casual_social", 0.0) * 0.1 + regime_priors.get("casual_social", 0.0) + pe_relationship_shortfall * 0.03),
            "acquaintance_building": _clamp(base + historical_effectiveness.get("acquaintance_building", 0.0) * 0.1 + regime_priors.get("acquaintance_building", 0.0) + pe_relationship_shortfall * 0.04),
            "emotional_support": _clamp(base + historical_effectiveness.get("emotional_support", 0.0) * 0.1 + regime_priors.get("emotional_support", 0.0) + pe_relationship_shortfall * 0.20),
            "guided_exploration": _clamp(base + historical_effectiveness.get("guided_exploration", 0.0) * 0.1 + regime_priors.get("guided_exploration", 0.0) + pe_action_shortfall * 0.10),
            "problem_solving": _clamp(base + historical_effectiveness.get("problem_solving", 0.0) * 0.1 + regime_priors.get("problem_solving", 0.0) + pe_task_shortfall * 0.18),
            "repair_and_deescalation": _clamp(base + historical_effectiveness.get("repair_and_deescalation", 0.0) * 0.1 + regime_priors.get("repair_and_deescalation", 0.0) + pe_relationship_shortfall * 0.22 + pe_regime_shortfall * 0.08),
        }
        ranked = tuple(sorted(cold_scores.items(), key=lambda item: item[1], reverse=True))
        return ranked

    world_tension = dual_track_snapshot.world_track.tension_level
    self_tension = dual_track_snapshot.self_track.tension_level
    cross_tension = dual_track_snapshot.cross_track_tension
    # NOTE: defaults below are intentionally left at the historical 0.4 so
    # the rest of the classifier weights stay coherent with the existing
    # test calibration. The lifeform benchmark surfaces a real bias here
    # ("every casual / repair / emotional input routes to problem_solving
    # + clarify-first") but the right fix is NOT a one-off default tweak
    # — it requires real learning over collected traces (see
    # lifeform-evolution.TraceCollector). Tracked as todo #14b.
    task_score = _metric(evaluation_snapshot, "info_integration", default=0.4)
    task_pressure = _metric(evaluation_snapshot, "task_pressure", default=task_score)
    repair_pressure = _metric(evaluation_snapshot, "repair_pressure", default=0.0)
    social_pressure = _metric(evaluation_snapshot, "social_pressure", default=0.0)
    decision_delegation_pressure = _metric(
        evaluation_snapshot, "decision_delegation_pressure", default=0.0
    )
    semantic_surface_active = _metric(evaluation_snapshot, "semantic_surface_active", default=0.0)
    warmth = _metric(evaluation_snapshot, "warmth", default=0.4)
    support_presence = _metric(evaluation_snapshot, "support_presence", default=warmth)
    relationship_stability = _metric(evaluation_snapshot, "cross_track_stability", default=0.4)
    alert_pressure = _alert_pressure(evaluation_snapshot)
    world_count, self_count = _track_counts(memory_snapshot)
    balance = _clamp(1.0 - abs(world_tension - self_tension))
    world_presence = _clamp(world_count / 3.0)
    self_presence = _clamp(self_count / 3.0)
    task_dominance = _clamp(max(task_pressure - support_presence, 0.0) / 0.35)
    support_dominance = _clamp(max(support_presence - task_pressure, 0.0) / 0.35)
    support_before_decision_pressure = _metric(
        evaluation_snapshot,
        "support_before_decision_pressure",
        default=0.0,
    )
    emotional_decision_pressure = _clamp(
        support_before_decision_pressure * 0.40
        + min(support_presence, max(task_pressure, decision_delegation_pressure)) * 0.36
        + decision_delegation_pressure * 0.14
        + repair_pressure * 0.10
    )
    mixed_support_task_pressure = _clamp(
        support_presence
        * max(task_pressure, decision_delegation_pressure)
        * (1.0 - task_dominance * 0.55)
    )
    low_pressure = _clamp(1.0 - max(task_pressure, support_presence, repair_pressure))
    semantic_low_pressure = low_pressure * semantic_surface_active
    world_drive, self_drive, shared_drive, switch_pressure = _controller_profile(dual_track_snapshot)
    repair_bias, task_bias, exploration_bias, stabilize_bias = _abstract_action_profile(dual_track_snapshot)
    memory_task_planning_bias = (
        task_dominance
        if dual_track_snapshot.world_track.controller_source == "memory"
        and dual_track_snapshot.self_track.controller_source == "memory"
        else 0.0
    )
    feature_values = {
        "task_pressure": task_pressure,
        "support_presence": support_presence,
        "repair_pressure": repair_pressure,
        "social_pressure": social_pressure,
        "decision_delegation_pressure": decision_delegation_pressure,
        "support_before_decision_pressure": support_before_decision_pressure,
        "emotional_decision_pressure": emotional_decision_pressure,
        "mixed_support_task_pressure": mixed_support_task_pressure,
        "task_dominance": task_dominance,
        "support_dominance": support_dominance,
        "low_pressure": low_pressure,
        "world_tension": world_tension,
        "self_tension": self_tension,
        "cross_tension": cross_tension,
        "world_presence": world_presence,
        "self_presence": self_presence,
        "world_drive": world_drive,
        "self_drive": self_drive,
        "shared_drive": shared_drive,
        "switch_pressure": switch_pressure,
        "repair_bias": repair_bias,
        "task_bias": task_bias,
        "exploration_bias": exploration_bias,
        "stabilize_bias": stabilize_bias,
    }
    scores = {
        "casual_social": _clamp(
            0.16 * (1.0 - max(world_tension, self_tension))
            + 0.24 * warmth
            + 0.20 * relationship_stability
            + 0.12 * balance
            + 0.25 * semantic_low_pressure
            + 0.10 * social_pressure
            - 0.20 * decision_delegation_pressure
            - 0.22 * repair_pressure
            - 0.22 * switch_pressure
            + 0.10 * stabilize_bias
            - 0.14 * repair_bias
            - 0.08 * task_bias
            - 0.12 * exploration_bias
            - 0.10 * task_pressure
            - 0.40 * task_dominance
            + pe_relationship_shortfall * 0.05
        ),
        "acquaintance_building": _clamp(
            0.26 * self_presence
            + 0.16 * warmth
            + 0.10 * support_presence
            + 0.18 * relationship_stability
            + 0.55 * semantic_low_pressure
            + 0.35 * social_pressure
            - 0.28 * decision_delegation_pressure
            - 0.28 * repair_pressure
            + 0.16 * self_drive
            + 0.12 * shared_drive
            + 0.08 * balance
            - 0.08 * world_tension
            + 0.06 * stabilize_bias
            + pe_relationship_shortfall * 0.06
        ),
        # Emotional support must not be triggered by generic synthetic
        # tension alone. Companion fallback runs often have high
        # self_tension / self_drive on every prompt, so this regime is
        # anchored to *support dominance* and relationship shortfall.
        "emotional_support": _clamp(
            0.12 * self_tension
            + 0.10 * self_presence
            + 0.08 * self_drive
            + 0.08 * warmth
            + 0.18 * support_presence
            + 0.40 * support_presence * semantic_surface_active
            + 0.34 * support_dominance
            + 0.12 * repair_pressure
            + 0.08 * relationship_stability
            + 0.06 * shared_drive
            + 0.08 * switch_pressure
            + 0.12 * repair_bias
            + 0.08 * decision_delegation_pressure
            + 0.28 * emotional_decision_pressure
            + 0.26 * mixed_support_task_pressure
            + pe_relationship_shortfall * 0.18
            + pe_regime_shortfall * 0.08
        ),
        # Phase 1.8 deliberately leaves guided_exploration alone:
        # under synthetic substrate it correctly differentiates a
        # range of scenarios via task_score / task_pressure, and
        # cutting those (as we did for problem_solving) breaks the
        # calibrator's ability to lift match rate during super_loop.
        "guided_exploration": _clamp(
            0.24 * balance
            + 0.12 * task_score
            + 0.10 * task_pressure
            + 0.18 * shared_drive
            + 0.18 * switch_pressure
            + 0.12 * self_presence
            + 0.10 * world_presence
            + 0.26 * exploration_bias
            + 0.22 * decision_delegation_pressure
            + 0.12 * emotional_decision_pressure
            + 0.18 * mixed_support_task_pressure
            - 0.18 * low_pressure
            + pe_action_shortfall * 0.12
        ),
        # Phase 1.8 rebalance: ``task_score`` (info_integration) and
        # ``task_pressure`` collapse to a near-constant ~0.6-0.78 across
        # all prompts under real Qwen 0.5B substrate (probe data). They
        # gave problem_solving a structural ~0.18 free lift on every
        # input, swamping per-content signal. Reduce their carry, and
        # require ``task_dominance`` (the genuinely-discriminative
        # task-vs-support delta) to do more of the work.
        "problem_solving": _clamp(
            0.25 * world_tension
            + 0.05 * task_score        # was 0.16 - info_integration is too constant
            + 0.10 * task_pressure     # was 0.16 - same reason
            + 0.22 * task_dominance    # was 0.16 - this one DOES move with content
            + 0.20 * memory_task_planning_bias
            + 0.08 * relationship_stability
            + 0.20 * world_presence
            + 0.20 * world_drive
            + 0.11 * switch_pressure
            + 0.22 * task_bias
            + 0.08 * decision_delegation_pressure
            - 0.42 * emotional_decision_pressure
            - 0.22 * mixed_support_task_pressure
            + pe_task_shortfall * 0.18
            + pe_action_shortfall * 0.08
        ),
        # Phase 1.8 leaves repair_and_deescalation unchanged: under
        # synthetic substrate ``cross_track_tension`` and
        # ``relationship_stability`` properly fire on rupture
        # scenarios; touching them breaks
        # ``test_regime_module_prefers_repair_when_cross_track_tension_is_high``.
        # Under Qwen these features collapse to ~0 / 1.0 so repair
        # is structurally suppressed there \u2014 acknowledged limitation
        # of the formula on small open-weight LLMs; tracked as a
        # follow-up for substrate-specific recalibration.
        "repair_and_deescalation": _clamp(
            0.34 * cross_tension
            + 0.24 * (1.0 - relationship_stability)
            + 0.12 * self_tension
            + 0.56 * repair_pressure
            + 0.16 * alert_pressure
            + 0.12 * shared_drive
            + 0.12 * switch_pressure
            + 0.30 * repair_bias
            + 0.20 * decision_delegation_pressure
            + 0.18 * emotional_decision_pressure
            - 0.08 * task_dominance
            - 0.10 * support_dominance
            + pe_relationship_shortfall * 0.20
            + pe_regime_shortfall * 0.10
        ),
    }

    learned_weights = selection_weights or {}
    learned_feature_weights = feature_weights or {}
    ranked: list[tuple[str, float]] = []
    for template in REGIME_TEMPLATES:
        historical = historical_effectiveness.get(template.regime_id, 0.5)
        feature_adjustment = sum(
            feature_values.get(feature_name, 0.0) * weight
            for feature_name, weight in learned_feature_weights.get(template.regime_id, {}).items()
        )
        base_score = _clamp(
            scores[template.regime_id] * 0.80
            + historical * 0.15
            + regime_priors.get(template.regime_id, 0.0)
            + feature_adjustment
        )
        weight = learned_weights.get(template.regime_id, 1.0)
        blended = _clamp(base_score * weight + fast_regime_biases.get(template.regime_id, 0.0) * 0.45)
        ranked.append((template.regime_id, round(blended, 4)))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return tuple(ranked)


def build_regime_identity(
    *,
    regime_id: str,
    historical_effectiveness: Mapping[str, float],
) -> RegimeIdentity:
    for template in REGIME_TEMPLATES:
        if template.regime_id == regime_id:
            return RegimeIdentity(
                regime_id=template.regime_id,
                name=template.name,
                embedding=template.embedding,
                entry_conditions=template.entry_conditions,
                exit_conditions=template.exit_conditions,
                historical_effectiveness=historical_effectiveness.get(regime_id, 0.5),
            )
    raise KeyError(f"Unknown regime_id: {regime_id}")
