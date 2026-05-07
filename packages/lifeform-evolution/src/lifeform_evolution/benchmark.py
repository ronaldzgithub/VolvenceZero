"""Scripted multi-turn benchmark harness for the lifeform.

Drives a ``Lifeform`` through a fixed dialogue script, then reports compact
evidence about whether the kernel surfaces aligned with the script:

* Did the regime track the expected trajectory?
* Did prediction-error magnitude rise on adversarial turns?
* Did open-loops accumulate / clear as expected?
* Did the lifeform produce a coherent (non-empty) response per turn?

This is a **read-only** evidence harness — it does not mutate kernel state
beyond the obvious side-effect of running turns. It does NOT claim that
the kernel is "good" in some absolute sense; it claims that on this fixed
script, the surfaces are consistent with the design (R12).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from lifeform_core import Lifeform, LifeformConfig, LifeformSession
from lifeform_domain_emogpt import build_companion_package


@dataclass(frozen=True)
class ScriptedTurn:
    user_input: str
    expected_regime_in: tuple[str, ...] = ()
    expected_min_pe_magnitude: float | None = None


@dataclass(frozen=True)
class ScriptedScenario:
    scenario_id: str
    description: str
    turns: tuple[ScriptedTurn, ...]


@dataclass(frozen=True)
class TurnReport:
    turn_index: int
    user_input: str
    response_text: str
    active_regime: str | None
    active_abstract_action: str | None
    expression_intent: str | None
    pe_magnitude: float
    open_loop_count: int
    regime_match: bool
    pe_threshold_met: bool
    # Optional extras populated by ``run_benchmark_async`` for downstream
    # family-grouped evaluation. Defaults preserve backward compat: callers
    # that constructed ``TurnReport(...)`` before these fields existed still
    # work and just see flat zero-state.
    temporal_switch_gate: float = 0.0
    continuum_target_position: float | None = None
    refer_out_required: bool = False
    response_length: int = 0
    evaluation_metrics: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class BenchmarkReport:
    scenario_id: str
    turn_reports: tuple[TurnReport, ...]
    regime_match_rate: float
    pe_threshold_match_rate: float
    response_non_empty_rate: float
    closed_scene_count: int
    # End-of-scenario lifeform state used by family-grouped evaluators.
    # Defaults preserve backward compat for callers building reports
    # synthetically (e.g. older tests).
    final_vitals_total_pe: float = 0.0
    final_vitals_drive_levels: tuple[tuple[str, float], ...] = ()
    pending_followup_count: int = 0
    proactive_followup_count: int = 0
    # Phase 2 W2.0b (debt #10A closure): end-of-scenario interlocutor
    # state axes used by the longitudinal F3 aggregator. Empty tuple
    # when the lifeform did not expose an interlocutor_state (e.g.
    # legacy / non-companion verticals). Each entry is
    # ``(axis_name, value)`` where axis_name is one of
    # ``il_trust`` / ``il_rapport`` / ``il_conf`` /
    # ``il_pace_pressure`` / ``il_emotional`` / ``il_resistance``.
    final_interlocutor_axes: tuple[tuple[str, float], ...] = ()

    def passed(self, *, min_regime_match_rate: float = 0.5, min_response_non_empty: float = 1.0) -> bool:
        return (
            self.regime_match_rate >= min_regime_match_rate
            and self.response_non_empty_rate >= min_response_non_empty
        )


# ---------------------------------------------------------------------------
# Built-in scenarios
# ---------------------------------------------------------------------------


def low_mood_disclosure_scenario() -> ScriptedScenario:
    return ScriptedScenario(
        scenario_id="low-mood-disclosure",
        description=(
            "User opens with low-mood disclosure, then under pressure asks for help. "
            "Expected trajectory: emotional support \u2192 guided exploration."
        ),
        turns=(
            ScriptedTurn(
                user_input="I have been feeling really stuck lately and I do not know why.",
                expected_regime_in=("emotional_support", "acquaintance_building", "casual_social"),
            ),
            ScriptedTurn(
                user_input="It is mostly that work feels heavy and home is also tense.",
                expected_regime_in=("emotional_support", "guided_exploration"),
            ),
            ScriptedTurn(
                user_input="Can you help me figure out what to even do first?",
                expected_regime_in=("guided_exploration", "problem_solving", "emotional_support"),
            ),
            ScriptedTurn(
                user_input="Honestly that almost made it worse, I just wanted to be heard.",
                expected_regime_in=("repair_and_deescalation", "emotional_support"),
                expected_min_pe_magnitude=0.0,
            ),
        ),
    )


def trust_rupture_repair_scenario() -> ScriptedScenario:
    """Scenario specifically designed to exercise the repair-first intent.

    Mid-scene the user signals a misread — the lifeform should drop into a
    repair frame, name the rupture, and offer one concrete way back.
    """
    return ScriptedScenario(
        scenario_id="trust-rupture-repair",
        description=(
            "User signals that an earlier reply landed badly. Expected behaviour: "
            "the lifeform names the rupture, holds steady tone, and offers one repair move."
        ),
        turns=(
            ScriptedTurn(
                user_input="Can you help me think through whether to leave my job?",
                expected_regime_in=(
                    "guided_exploration",
                    "problem_solving",
                    "acquaintance_building",
                ),
            ),
            ScriptedTurn(
                user_input=(
                    "Wait, that response felt completely cold and procedural. "
                    "I am not asking you to optimise me."
                ),
                expected_regime_in=(
                    "repair_and_deescalation",
                    "emotional_support",
                ),
            ),
            ScriptedTurn(
                user_input="Can we just back up and start that over?",
                expected_regime_in=(
                    "repair_and_deescalation",
                    "emotional_support",
                ),
            ),
        ),
    )


def emotional_decision_support_scenario() -> ScriptedScenario:
    """Mixed support + decision pressure where solving too early is failure."""

    return ScriptedScenario(
        scenario_id="emotional-decision-support",
        description=(
            "User is overwhelmed but still needs a decision frame. Expected behaviour: "
            "support-first stabilization before guided exploration or bounded problem solving."
        ),
        turns=(
            ScriptedTurn(
                user_input=(
                    "I need help deciding what to do next, but I am overwhelmed and I do not want "
                    "to be optimized past how I feel."
                ),
                expected_regime_in=("emotional_support", "repair_and_deescalation", "guided_exploration"),
            ),
            ScriptedTurn(
                user_input="Please do not decide for me. Help me get steady enough to see the choice clearly.",
                expected_regime_in=("emotional_support", "guided_exploration"),
            ),
            ScriptedTurn(
                user_input=(
                    "The goal is shifting from maximum output to recovery, so the decision has to respect "
                    "that emotional constraint."
                ),
                expected_regime_in=("emotional_support", "guided_exploration", "problem_solving"),
            ),
            ScriptedTurn(
                user_input="Now help me name the tradeoff and choose one reversible next step.",
                expected_regime_in=("guided_exploration", "problem_solving", "emotional_support"),
            ),
        ),
    )


def casual_social_checkin_scenario() -> ScriptedScenario:
    """Low-pressure social check-in, no decision pressure, no rupture.

    Expected behaviour: ``casual_social`` or ``acquaintance_building`` regime
    with ``direct-answer`` / ``warmth-first`` expression intent. This proves
    the lifeform doesn't escalate every turn into solving mode.
    """
    return ScriptedScenario(
        scenario_id="casual-social-checkin",
        description=(
            "Light conversational check-in. No problem to solve; the test is "
            "whether the lifeform stays low-pressure rather than over-formalising."
        ),
        turns=(
            ScriptedTurn(
                user_input="Just saying hi, hope your day is going alright.",
                expected_regime_in=(
                    "casual_social",
                    "acquaintance_building",
                ),
            ),
            ScriptedTurn(
                user_input="Anything interesting on your mind today?",
                expected_regime_in=(
                    "casual_social",
                    "acquaintance_building",
                    "guided_exploration",
                ),
            ),
            ScriptedTurn(
                user_input="Cool, that is enough catching up — talk later.",
                expected_regime_in=(
                    "casual_social",
                    "acquaintance_building",
                ),
            ),
        ),
    )


def all_built_in_scenarios() -> tuple[ScriptedScenario, ...]:
    """Convenience: return every built-in scenario for batch running."""
    return (
        low_mood_disclosure_scenario(),
        trust_rupture_repair_scenario(),
        emotional_decision_support_scenario(),
        casual_social_checkin_scenario(),
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_benchmark(
    *,
    scenario: ScriptedScenario | None = None,
    config: LifeformConfig | None = None,
    temporal_bootstrap: object | None = None,
    regime_bootstrap: object | None = None,
    lifeform: Lifeform | None = None,
) -> BenchmarkReport:
    """Run one scenario synchronously. Wraps ``run_benchmark_async``."""
    return asyncio.run(
        run_benchmark_async(
            scenario=scenario,
            config=config,
            temporal_bootstrap=temporal_bootstrap,
            regime_bootstrap=regime_bootstrap,
            lifeform=lifeform,
        )
    )


async def run_benchmark_async(
    *,
    scenario: ScriptedScenario | None = None,
    config: LifeformConfig | None = None,
    temporal_bootstrap: object | None = None,
    regime_bootstrap: object | None = None,
    lifeform: Lifeform | None = None,
) -> BenchmarkReport:
    """Run one scripted scenario through a Lifeform.

    If ``lifeform`` is supplied, the runner uses it as-is (allowing the
    caller to pre-configure vitals, custom synthesizers, etc. via e.g.
    ``build_companion_lifeform()``). Otherwise it builds a minimal default
    Lifeform from ``config`` plus the companion ``DomainExperiencePackage``.
    The default path stays vertical-agnostic and does NOT wire vitals
    or pre-trained bootstraps \u2014 callers wanting those should supply
    ``lifeform`` directly.
    """
    chosen = scenario or low_mood_disclosure_scenario()
    if lifeform is None:
        base_config = config or LifeformConfig()
        base_config = base_config.with_domain_experience((build_companion_package(),))
        lifeform = Lifeform(
            base_config,
            temporal_bootstrap=temporal_bootstrap,
            regime_bootstrap=regime_bootstrap,
        )
    session = lifeform.create_session(session_id=f"benchmark-{chosen.scenario_id}")

    turn_reports: list[TurnReport] = []
    for index, turn in enumerate(chosen.turns, start=1):
        result = await session.run_turn(turn.user_input)

        regime_match = bool(
            not turn.expected_regime_in
            or (result.active_regime in turn.expected_regime_in)
        )
        pe_magnitude = 0.0
        pe_snapshot = result.active_snapshots.get("prediction_error")
        if pe_snapshot is not None:
            error = getattr(pe_snapshot.value, "error", None)
            if error is not None:
                pe_magnitude = float(getattr(error, "magnitude", 0.0))
        pe_threshold_met = (
            turn.expected_min_pe_magnitude is None
            or pe_magnitude >= turn.expected_min_pe_magnitude
        )

        open_loop_count = 0
        open_loop_snapshot = result.active_snapshots.get("open_loop")
        if open_loop_snapshot is not None:
            open_loop_count = len(
                getattr(open_loop_snapshot.value, "unresolved_loops", ()) or ()
            )

        expression_intent: str | None = None
        continuum_position: float | None = None
        refer_out_required = False
        assembly_snapshot = result.active_snapshots.get("response_assembly")
        if assembly_snapshot is not None:
            expression_intent = getattr(assembly_snapshot.value, "expression_intent", None)
            cp = getattr(assembly_snapshot.value, "continuum_target_position", None)
            if cp is not None:
                continuum_position = float(cp)
            refer_out_required = bool(
                getattr(assembly_snapshot.value, "refer_out_required", False)
            )

        switch_gate = 0.0
        temporal_snapshot = result.active_snapshots.get("temporal_abstraction")
        if temporal_snapshot is not None:
            sg = getattr(temporal_snapshot.value, "switch_gate", None)
            if sg is not None:
                switch_gate = float(sg)
        evaluation_metrics: tuple[tuple[str, float], ...] = ()
        evaluation_snapshot = result.active_snapshots.get("evaluation")
        if evaluation_snapshot is not None:
            scores = getattr(evaluation_snapshot.value, "turn_scores", ()) or ()
            evaluation_metrics = tuple(
                (str(getattr(score, "metric_name")), float(getattr(score, "value")))
                for score in scores
                if getattr(score, "metric_name", None) is not None
                and isinstance(getattr(score, "value", None), int | float)
            )

        turn_reports.append(
            TurnReport(
                turn_index=index,
                user_input=turn.user_input,
                response_text=result.response.text,
                active_regime=result.active_regime,
                active_abstract_action=result.active_abstract_action,
                expression_intent=expression_intent,
                pe_magnitude=pe_magnitude,
                open_loop_count=open_loop_count,
                regime_match=regime_match,
                pe_threshold_met=pe_threshold_met,
                temporal_switch_gate=switch_gate,
                continuum_target_position=continuum_position,
                refer_out_required=refer_out_required,
                response_length=len(result.response.text),
                evaluation_metrics=evaluation_metrics,
            )
        )

    # Pull lifeform-side end-of-run state for family-grouped evaluation
    # BEFORE we close the scene \u2014 closing fires slow-loop drain which can
    # mutate snapshots, and we want the state as seen by the caller.
    final_vitals_total_pe = 0.0
    final_vitals_drive_levels: tuple[tuple[str, float], ...] = ()
    if session.vitals_snapshot is not None:
        final_vitals_total_pe = session.vitals_snapshot.total_pe
        final_vitals_drive_levels = tuple(
            (d.name, d.level) for d in session.vitals_snapshot.drive_levels
        )
    # Phase 2 W2.0b (debt #10A closure): capture the end-of-scenario
    # interlocutor 12-axis readout so the longitudinal F3 aggregator
    # can track ``il_trust`` / ``il_rapport`` cross-round trends.
    # ``LifeformSession.interlocutor_state`` is the SHADOW-aware
    # accessor: under explicit SHADOW wiring it falls back to the
    # legacy duck-typed builder; under ACTIVE wiring it reads the
    # owner snapshot. Either way we get an ``InterlocutorState``
    # dataclass with the typed axes.
    final_interlocutor_axes: tuple[tuple[str, float], ...] = ()
    interlocutor_state = session.interlocutor_state
    if interlocutor_state is not None:
        final_interlocutor_axes = (
            ("il_trust", float(interlocutor_state.trust_signal)),
            ("il_rapport", float(interlocutor_state.rapport_warmth)),
            ("il_conf", float(interlocutor_state.readout_confidence)),
            ("il_pace_pressure", float(interlocutor_state.pace_pressure)),
            ("il_emotional", float(interlocutor_state.emotional_weight)),
            ("il_resistance", float(interlocutor_state.resistance_level)),
        )
    pending = session.followup_manager.pending
    pending_followup_count = len(pending)
    proactive_followup_count = sum(
        1 for item in pending if item.source == "vitals"
    )

    closed = await session.end_scene(reason="benchmark-end", drain_slow_loop=True)

    n = len(turn_reports) or 1
    return BenchmarkReport(
        scenario_id=chosen.scenario_id,
        turn_reports=tuple(turn_reports),
        regime_match_rate=sum(1 for r in turn_reports if r.regime_match) / n,
        pe_threshold_match_rate=sum(1 for r in turn_reports if r.pe_threshold_met) / n,
        response_non_empty_rate=sum(1 for r in turn_reports if r.response_text.strip()) / n,
        closed_scene_count=1 if closed is not None else 0,
        final_vitals_total_pe=final_vitals_total_pe,
        final_vitals_drive_levels=final_vitals_drive_levels,
        pending_followup_count=pending_followup_count,
        proactive_followup_count=proactive_followup_count,
        final_interlocutor_axes=final_interlocutor_axes,
    )


def format_report(report: BenchmarkReport) -> str:
    lines: list[str] = []
    lines.append(f"== Lifeform benchmark: {report.scenario_id} ==")
    lines.append(
        f"   regime match rate: {report.regime_match_rate:.0%}    "
        f"PE threshold rate: {report.pe_threshold_match_rate:.0%}    "
        f"non-empty rate: {report.response_non_empty_rate:.0%}    "
        f"closed scenes: {report.closed_scene_count}"
    )
    for tr in report.turn_reports:
        flag = "OK" if tr.regime_match and tr.response_text.strip() else "??"
        lines.append(
            f"   [{tr.turn_index}] {flag} regime={tr.active_regime or '-'} "
            f"intent={tr.expression_intent or '-'} "
            f"action={tr.active_abstract_action or '-'} "
            f"pe={tr.pe_magnitude:.3f} loops={tr.open_loop_count}"
        )
        lines.append(f"        user: {tr.user_input}")
        lines.append(f"        ai  : {tr.response_text[:160]}")

    intents = sorted({tr.expression_intent for tr in report.turn_reports if tr.expression_intent})
    if intents:
        lines.append(f"   distinct intents: {len(intents)} ({', '.join(intents)})")
    return "\n".join(lines)
