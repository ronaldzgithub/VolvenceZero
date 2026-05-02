"""Companion-specific evidence gates (C1-C4).

This module is deliberately product/evidence layer code. It does not
mutate kernel state beyond running normal turns through a Lifeform, and it
does not become a learning source. Its job is to answer: "what can we
prove about companionship behaviour today?"
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from lifeform_domain_emogpt import build_companion_lifeform, scenarios_dir
from lifeform_evolution.scenario_pack import load_scenarios
from lifeform_expression import GroundedResponseSynthesizer, PromptPlanner
from volvence_zero.memory import build_default_memory_store
from volvence_zero.social_cognition import (
    PRIMARY_INTERLOCUTOR_ID,
    SELF_INTERLOCUTOR_ID,
    SocialPredictionErrorSnapshot,
    SocialPredictionSnapshot,
)


@dataclass(frozen=True)
class CompanionEvidenceGate:
    gate_id: str
    name: str
    passed: bool
    summary: str
    metrics: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class CompanionEvidenceTurn:
    user_input: str
    response_text: str
    active_regime: str | None
    expression_intent: str | None
    active_speaker_id: str = PRIMARY_INTERLOCUTOR_ID
    addressee_ids: tuple[str, ...] = (SELF_INTERLOCUTOR_ID,)
    subject_ids: tuple[str, ...] = (PRIMARY_INTERLOCUTOR_ID,)
    audience_ids: tuple[str, ...] = (SELF_INTERLOCUTOR_ID,)
    social_prediction_count: int = 0
    social_prediction_error_count: int = 0


@dataclass(frozen=True)
class CompanionEvidenceTranscript:
    transcript_id: str
    condition: str
    turns: tuple[CompanionEvidenceTurn, ...]


@dataclass(frozen=True)
class CompanionEvidenceReport:
    gates: tuple[CompanionEvidenceGate, ...]
    description: str
    composite_score: float
    transcripts: tuple[CompanionEvidenceTranscript, ...] = ()

    @property
    def passed(self) -> bool:
        return all(gate.passed for gate in self.gates)


async def run_companion_evidence_async() -> CompanionEvidenceReport:
    gates = (
        await _state_sensitivity_gate(),
        await _within_session_adaptation_gate(),
        await _cross_session_retention_gate(),
        _default_isolation_gate(),
        await _social_scope_default_gate(),
    )
    transcripts = await _collect_widening_transcripts()
    transcript_score = _transcript_diversity_score(transcripts)
    passed_count = sum(1 for gate in gates if gate.passed)
    gate_score = passed_count / max(len(gates), 1)
    composite_score = gate_score * 0.8 + transcript_score * 0.2
    return CompanionEvidenceReport(
        gates=gates,
        composite_score=composite_score,
        transcripts=transcripts,
        description=(
            f"Companion evidence gates passed {passed_count}/{len(gates)}; "
            f"widening transcript diversity score={transcript_score:.2f}."
        ),
    )


def run_companion_evidence() -> CompanionEvidenceReport:
    return asyncio.run(run_companion_evidence_async())


async def _state_sensitivity_gate() -> CompanionEvidenceGate:
    lifeform = _build_evidence_lifeform()
    task_session = lifeform.create_session(session_id="evidence-state-task")
    emotional_session = lifeform.create_session(session_id="evidence-state-emotional")

    await task_session.run_turn(
        "Can you help me draft a concise email declining a meeting invite and suggesting async instead?"
    )
    await emotional_session.run_turn(
        "I have been feeling really stuck and heavy lately, and I mostly need to feel heard first."
    )

    task_state = task_session.interlocutor_state
    emotional_state = emotional_session.interlocutor_state
    metrics = (
        ("task_focus_delta", task_state.task_focus_level - emotional_state.task_focus_level),
        ("directness_delta", task_state.directness - emotional_state.directness),
        ("rapport_delta", emotional_state.rapport_warmth - task_state.rapport_warmth),
        ("task_confidence", task_state.readout_confidence),
        ("emotional_confidence", emotional_state.readout_confidence),
    )
    passed = (
        task_state.readout_confidence >= 0.6
        and emotional_state.readout_confidence >= 0.6
        and task_state.task_focus_level > emotional_state.task_focus_level
        and task_state.directness > emotional_state.directness
        and emotional_state.rapport_warmth > task_state.rapport_warmth
    )
    return CompanionEvidenceGate(
        gate_id="C1",
        name="state sensitivity",
        passed=passed,
        metrics=metrics,
        summary="Same companion produces different interlocutor readouts for task vs emotional contexts.",
    )


async def _within_session_adaptation_gate() -> CompanionEvidenceGate:
    scenario = next(
        sc
        for sc in load_scenarios(scenarios_dir())
        if sc.scenario_id == "low-mood-disclosure"
    )
    session = _build_evidence_lifeform().create_session(session_id="evidence-low-mood")

    regimes: list[str | None] = []
    intents: list[str | None] = []
    pe_values: list[float] = []
    for turn in scenario.turns:
        result = await session.run_turn(turn.user_input)
        regimes.append(result.active_regime)
        assembly = result.active_snapshots["response_assembly"].value
        intents.append(assembly.expression_intent)
        pe_values.append(result.prediction_error.magnitude if result.prediction_error else 0.0)

    distinct_intents = len({intent for intent in intents if intent is not None})
    pe_span = max(pe_values, default=0.0) - min(pe_values, default=0.0)
    passed = (
        "emotional_support" in regimes
        and distinct_intents >= 2
        and pe_span > 0.0
        and session.interlocutor_state.readout_confidence >= 0.6
    )
    return CompanionEvidenceGate(
        gate_id="C2",
        name="within-session adaptation",
        passed=passed,
        metrics=(
            ("distinct_intents", float(distinct_intents)),
            ("pe_span", pe_span),
            ("readout_confidence", session.interlocutor_state.readout_confidence),
        ),
        summary="A low-mood episode changes regime, intent, and PE within one session.",
    )


async def _cross_session_retention_gate() -> CompanionEvidenceGate:
    shared_memory = build_default_memory_store()
    lifeform = _build_evidence_lifeform(memory_store=shared_memory)
    first = lifeform.create_session(session_id="evidence-retention-a")
    await first.run_turn(
        "When I am overwhelmed, please do not jump straight into steps; help me sort the feeling first."
    )
    await first.end_scene(reason="preference-captured")

    second = lifeform.create_session(session_id="evidence-retention-b")
    result = await second.run_turn("I feel overwhelmed about work; what should I do first?")
    memory = result.active_snapshots["memory"].value
    retrieved_text = "\n".join(entry.content for entry in memory.retrieved_entries)
    retained = (
        "do not jump straight into steps" in retrieved_text
        and "help me sort the feeling first" in retrieved_text
    )
    return CompanionEvidenceGate(
        gate_id="C3",
        name="cross-session preference retention",
        passed=retained,
        metrics=(("retrieved_entry_count", float(len(memory.retrieved_entries))),),
        summary="Explicit shared MemoryStore lets a later session retrieve a learned preference.",
    )


def _default_isolation_gate() -> CompanionEvidenceGate:
    lifeform = _build_evidence_lifeform()
    first = lifeform.create_session(session_id="evidence-isolated-a")
    second = lifeform.create_session(session_id="evidence-isolated-b")
    isolated = first.brain_session.runner.memory_store is not second.brain_session.runner.memory_store
    return CompanionEvidenceGate(
        gate_id="C4",
        name="default memory isolation",
        passed=isolated,
        summary="Default sessions keep MemoryStore isolated unless a shared store is injected.",
    )


async def _social_scope_default_gate() -> CompanionEvidenceGate:
    session = _build_evidence_lifeform().create_session(session_id="evidence-social-scope")
    result = await session.run_turn("Help me think through this gently.")

    identity_snapshot = result.shadow_snapshots.get("multi_party_identity")
    prediction_snapshot = result.shadow_snapshots.get("social_prediction")
    social_pe_snapshot = result.shadow_snapshots.get("social_prediction_error")
    prediction_count = (
        len(prediction_snapshot.value.predictions)
        if prediction_snapshot is not None
        and isinstance(prediction_snapshot.value, SocialPredictionSnapshot)
        else -1
    )
    social_pe_count = (
        len(social_pe_snapshot.value.errors)
        if social_pe_snapshot is not None
        and isinstance(social_pe_snapshot.value, SocialPredictionErrorSnapshot)
        else -1
    )
    passed = (
        result.active_speaker_id == PRIMARY_INTERLOCUTOR_ID
        and result.subject_ids == (PRIMARY_INTERLOCUTOR_ID,)
        and result.addressee_ids == (SELF_INTERLOCUTOR_ID,)
        and result.audience_ids == (SELF_INTERLOCUTOR_ID,)
        and identity_snapshot is not None
        and prediction_count == 0
        and social_pe_count == 0
    )
    return CompanionEvidenceGate(
        gate_id="C5",
        name="default social scope isolation",
        passed=passed,
        metrics=(
            ("social_prediction_count", float(prediction_count)),
            ("social_prediction_error_count", float(social_pe_count)),
        ),
        summary=(
            "Single-party companion evidence stays scoped to primary/self while "
            "R16 social prediction and social PE slots publish empty SHADOW readouts."
        ),
    )


_WIDENING_MICRO_SCENARIOS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "paraphrase-low-mood",
        (
            "I am in a weird rut today. Nothing is catastrophic, but everything feels heavier than it should.",
        ),
    ),
    (
        "tone-shift-repair",
        (
            "Can you help me think through a decision?",
            "That sounded a bit like you were optimizing me. Can we slow down?",
        ),
    ),
    (
        "delayed-return",
        (
            "I am back after thinking about yesterday. I still prefer sorting feelings before steps.",
        ),
    ),
    (
        "preference-conflict",
        (
            "I know I usually ask you not to jump into steps, but this time give me exactly one concrete next step.",
        ),
    ),
)


async def _collect_widening_transcripts() -> tuple[CompanionEvidenceTranscript, ...]:
    lifeform = _build_evidence_lifeform()
    transcripts: list[CompanionEvidenceTranscript] = []
    for condition, turns in _WIDENING_MICRO_SCENARIOS:
        session = lifeform.create_session(session_id=f"evidence-v2-{condition}")
        transcript_turns: list[CompanionEvidenceTurn] = []
        for user_input in turns:
            result = await session.run_turn(user_input)
            assembly = result.active_snapshots["response_assembly"].value
            social_prediction = result.shadow_snapshots.get("social_prediction")
            social_prediction_error = result.shadow_snapshots.get("social_prediction_error")
            transcript_turns.append(
                CompanionEvidenceTurn(
                    user_input=user_input,
                    response_text=result.response.text,
                    active_regime=result.active_regime,
                    expression_intent=assembly.expression_intent,
                    active_speaker_id=result.active_speaker_id,
                    addressee_ids=result.addressee_ids,
                    subject_ids=result.subject_ids,
                    audience_ids=result.audience_ids,
                    social_prediction_count=(
                        len(social_prediction.value.predictions)
                        if social_prediction is not None
                        and isinstance(social_prediction.value, SocialPredictionSnapshot)
                        else 0
                    ),
                    social_prediction_error_count=(
                        len(social_prediction_error.value.errors)
                        if social_prediction_error is not None
                        and isinstance(
                            social_prediction_error.value,
                            SocialPredictionErrorSnapshot,
                        )
                        else 0
                    ),
                )
            )
        transcripts.append(
            CompanionEvidenceTranscript(
                transcript_id=f"companion-v2-{condition}",
                condition=condition,
                turns=tuple(transcript_turns),
            )
        )
    return tuple(transcripts)


def _transcript_diversity_score(
    transcripts: tuple[CompanionEvidenceTranscript, ...],
) -> float:
    turns = tuple(turn for transcript in transcripts for turn in transcript.turns)
    if not turns:
        return 0.0
    regime_count = len({turn.active_regime for turn in turns if turn.active_regime is not None})
    intent_count = len({turn.expression_intent for turn in turns if turn.expression_intent is not None})
    response_archetype_count = len({_response_archetype(turn.response_text) for turn in turns})
    return min((regime_count + intent_count + response_archetype_count) / 9.0, 1.0)


def _response_archetype(text: str) -> str:
    compact = " ".join(text.lower().split())
    if "one concrete next step" in compact or "smallest reversible action" in compact:
        return "bounded-direct"
    if "sort the feeling" in compact or "heaviest part" in compact:
        return "emotional-holding"
    if "reset the pace" in compact or "felt off" in compact:
        return "repair-reset"
    if "explore this with you" in compact:
        return "guided-explore"
    return compact[:80]


def _build_evidence_lifeform(*, memory_store: object | None = None):
    return build_companion_lifeform(
        memory_store=memory_store,
        response_synthesizer=GroundedResponseSynthesizer(planner=PromptPlanner()),
    )


def companion_evidence_report_to_dict(report: CompanionEvidenceReport) -> dict[str, object]:
    return {
        "passed": report.passed,
        "composite_score": report.composite_score,
        "description": report.description,
        "gates": [
            {
                "gate_id": gate.gate_id,
                "name": gate.name,
                "passed": gate.passed,
                "summary": gate.summary,
                "metrics": dict(gate.metrics),
            }
            for gate in report.gates
        ],
        "transcripts": [
            {
                "transcript_id": transcript.transcript_id,
                "condition": transcript.condition,
                "turns": [
                    {
                        "user_input": turn.user_input,
                        "response_text": turn.response_text,
                        "active_regime": turn.active_regime,
                        "expression_intent": turn.expression_intent,
                        "active_speaker_id": turn.active_speaker_id,
                        "addressee_ids": list(turn.addressee_ids),
                        "subject_ids": list(turn.subject_ids),
                        "audience_ids": list(turn.audience_ids),
                        "social_prediction_count": turn.social_prediction_count,
                        "social_prediction_error_count": turn.social_prediction_error_count,
                    }
                    for turn in transcript.turns
                ],
            }
            for transcript in report.transcripts
        ],
    }


def format_companion_evidence_report(report: CompanionEvidenceReport) -> str:
    lines = [
        "== Companion evidence report ==",
        f"overall: {'PASS' if report.passed else 'FAIL'}",
        f"composite_score: {report.composite_score:.3f}",
        f"description: {report.description}",
    ]
    for gate in report.gates:
        lines.append(
            f"   [{gate.gate_id}] {'PASS' if gate.passed else 'FAIL'} {gate.name}: {gate.summary}"
        )
        if gate.metrics:
            lines.append(
                "        metrics: "
                + ", ".join(f"{name}={value:.3f}" for name, value in gate.metrics)
            )
    if report.transcripts:
        lines.append(f"widening_transcripts: {len(report.transcripts)}")
    return "\n".join(lines)


__all__ = (
    "CompanionEvidenceGate",
    "CompanionEvidenceReport",
    "CompanionEvidenceTranscript",
    "CompanionEvidenceTurn",
    "companion_evidence_report_to_dict",
    "format_companion_evidence_report",
    "run_companion_evidence",
    "run_companion_evidence_async",
)
