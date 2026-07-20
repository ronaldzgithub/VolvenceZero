"""Deterministic world/FSM compiler for synthetic experience trajectories."""

from __future__ import annotations

import hashlib
import random
from dataclasses import replace

from .canonical import canonical_json, stable_hash
from .contracts import (
    SCHEMA_VERSION,
    AnnotationRecord,
    AnnotationSource,
    ExperienceSession,
    ExperienceTrajectory,
    ExperienceTurn,
    GenerationTier,
    KeyValue,
    LatentTruthFrame,
    ProvenanceRecord,
    QualityRecord,
    QualitySeverity,
    ScenarioBlueprint,
    TrainingUse,
    TurnRole,
)

GENERATOR_VERSION = "unified-world-v1"

FAMILY_TARGET_OWNERS: dict[str, str] = {
    "relationship_continuity": "relationship_state",
    "rupture_repair": "relationship_state",
    "preference_personalization": "user_model",
    "absence_reengagement": "relationship_state",
    "boundary_consent_autonomy": "boundary_consent",
    "goal_value_drift": "goal_value",
    "plan_commitment_open_loop": "commitment",
    "task_tool_execution": "execution_result",
    "belief_uncertainty_verification": "belief_assumption",
    "emotional_support_regime": "regime",
    "multi_party_identity_privacy": "identity",
    "tom_common_ground_group": "common_ground",
    "memory_timescale_reflection": "memory",
    "environment_delayed_credit": "prediction_error",
    "apprenticeship_ingestion_teaching": "apprenticeship_alignment",
    "safety_adversarial_resilience": "boundary_consent",
}

FAMILY_TARGET_OWNER_SETS: dict[str, tuple[str, ...]] = {
    "relationship_continuity": ("relationship_state",),
    "rupture_repair": ("relationship_state",),
    "preference_personalization": ("user_model",),
    "absence_reengagement": ("relationship_state", "open_loop"),
    "boundary_consent_autonomy": ("boundary_consent",),
    "goal_value_drift": ("goal_value",),
    "plan_commitment_open_loop": ("plan_intent", "commitment", "open_loop"),
    "task_tool_execution": ("execution_result",),
    "belief_uncertainty_verification": ("belief_assumption",),
    "emotional_support_regime": ("relationship_state",),
    "multi_party_identity_privacy": ("conversational_role",),
    "tom_common_ground_group": (
        "belief_about_other",
        "intent_about_other",
        "common_ground",
        "groups",
    ),
    "memory_timescale_reflection": ("memory",),
    "environment_delayed_credit": ("environment_outcome",),
    "apprenticeship_ingestion_teaching": ("apprenticeship_alignment",),
    "safety_adversarial_resilience": ("boundary_consent",),
}

_PHASES = ("opening", "development", "outcome", "reflection")


def compile_structural_trajectory(
    blueprint: ScenarioBlueprint,
    *,
    replicate_index: int,
    seed: int,
    run_id: str,
    created_at: str,
    git_sha: str,
    generator_version: str = GENERATOR_VERSION,
) -> ExperienceTrajectory:
    """Compile one immutable trajectory without any model-produced labels."""

    if replicate_index < 0:
        raise ValueError("replicate_index must be non-negative")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    target_owners = FAMILY_TARGET_OWNER_SETS[blueprint.family]
    scenario_hash = stable_hash(blueprint)
    trajectory_id = f"trajectory:{blueprint.scenario_id}:{replicate_index:05d}:{seed:010d}"
    rng = random.Random(_derived_seed(blueprint.scenario_id, replicate_index, seed))

    annotations: list[AnnotationRecord] = []
    truth_frames: list[LatentTruthFrame] = []
    sessions: list[ExperienceSession] = []
    user_turn_ordinal = 0
    total_user_turns = sum((turn_count + 1) // 2 for turn_count in blueprint.turns_per_session)

    for session_index, turn_count in enumerate(blueprint.turns_per_session):
        turns: list[ExperienceTurn] = []
        for turn_index in range(turn_count):
            turn_id = f"{trajectory_id}:session:{session_index}:turn:{turn_index}"
            if turn_index % 2 == 0:
                phase = _phase_for(user_turn_ordinal, total_user_turns)
                fact = blueprint.observable_facts[0]
                context_facts = (
                    blueprint.observable_facts[1:]
                    if len(blueprint.observable_facts) > 1
                    else blueprint.observable_facts
                )
                context_fact = _select(
                    context_facts,
                    user_turn_ordinal + rng.randrange(len(context_facts)),
                )
                private_fact = _select(
                    blueprint.private_truth,
                    user_turn_ordinal + rng.randrange(len(blueprint.private_truth)),
                )
                event_id = f"{trajectory_id}:event:{session_index}:{turn_index}"
                frame_id = f"{trajectory_id}:truth:{session_index}:{turn_index}"
                frame_annotation_ids: list[str] = []
                for target_owner in target_owners:
                    annotation_id = f"{trajectory_id}:annotation:{session_index}:{turn_index}:{target_owner}"
                    label_value = canonical_json(
                        {
                            "family": blueprint.family,
                            "phase": phase,
                            "path_id": blueprint.path_id,
                            "arc_spec_id": blueprint.arc_spec_id,
                            "expected_owner": target_owner,
                        }
                    )
                    annotation = AnnotationRecord(
                        annotation_id=annotation_id,
                        target_ref=turn_id,
                        ontology="volvence.synthetic.owner_transition",
                        ontology_version="1.0.0",
                        label_key="expected_owner_transition",
                        label_value_json=label_value,
                        source=AnnotationSource.GENERATOR_TRUTH,
                        training_use=TrainingUse.TARGET,
                        confidence=1.0,
                        evidence_refs=(frame_id,),
                        target_owner=target_owner,
                        track=blueprint.track,
                        timescale=blueprint.timescale,
                        scope_ids=(
                            f"persona:{blueprint.persona_id}",
                            f"trajectory:{trajectory_id}",
                        ),
                    )
                    annotations.append(annotation)
                    frame_annotation_ids.append(annotation_id)
                truth_frame = LatentTruthFrame(
                    frame_id=frame_id,
                    turn_ref=turn_id,
                    phase_id=phase,
                    event_kind="world_event",
                    observable_facts=(
                        KeyValue(key="fact", value=fact),
                        KeyValue(key="context_constraint", value=context_fact),
                        KeyValue(
                            key="event_ordinal",
                            value=str(user_turn_ordinal),
                        ),
                        KeyValue(key="session_index", value=str(session_index)),
                    ),
                    private_facts=(
                        KeyValue(key="latent_truth", value=private_fact),
                        KeyValue(key="latent_arc_id", value=blueprint.latent_arc_id),
                    ),
                    response_contract=blueprint.response_contract,
                    annotation_refs=tuple(frame_annotation_ids),
                )
                truth_frames.append(truth_frame)
                text = _render_structural_user_text(
                    blueprint=blueprint,
                    phase=phase,
                    fact=fact,
                    ordinal=user_turn_ordinal,
                )
                turns.append(
                    ExperienceTurn(
                        turn_id=turn_id,
                        session_index=session_index,
                        turn_index=turn_index,
                        role=TurnRole.USER,
                        text=text,
                        event_id=event_id,
                        latent_frame_ref=frame_id,
                        metadata=(
                            KeyValue(key="text_slot", value="user"),
                            KeyValue(key="phase", value=phase),
                        ),
                    )
                )
                user_turn_ordinal += 1
            else:
                text = _render_structural_assistant_text(
                    blueprint=blueprint,
                    ordinal=user_turn_ordinal - 1,
                )
                turns.append(
                    ExperienceTurn(
                        turn_id=turn_id,
                        session_index=session_index,
                        turn_index=turn_index,
                        role=TurnRole.ASSISTANT,
                        text=text,
                        event_id=None,
                        latent_frame_ref=None,
                        metadata=(
                            KeyValue(key="text_slot", value="assistant"),
                            KeyValue(
                                key="response_contract",
                                value=canonical_json(blueprint.response_contract),
                            ),
                        ),
                    )
                )
        sessions.append(
            ExperienceSession(
                session_id=f"{trajectory_id}:session:{session_index}",
                session_index=session_index,
                gap_days_before=0 if session_index == 0 else 1 + session_index * 2,
                turns=tuple(turns),
            )
        )

    quality = (
        QualityRecord(
            quality_id=f"{trajectory_id}:quality:structural",
            check_kind="deterministic_world_compilation",
            passed=True,
            severity=QualitySeverity.INFO,
            score=1.0,
            evidence_refs=(scenario_hash,),
            description=("Events, transitions, labels, and response contracts were compiled without model inference."),
        ),
    )
    return ExperienceTrajectory(
        schema_version=SCHEMA_VERSION,
        trajectory_id=trajectory_id,
        scenario_ref=blueprint.scenario_id,
        scenario_hash=scenario_hash,
        split=blueprint.split,
        family=blueprint.family,
        language=blueprint.language,
        generation_tier=GenerationTier.STRUCTURAL,
        sessions=tuple(sessions),
        truth_frames=tuple(truth_frames),
        snapshot_frames=(),
        annotations=tuple(annotations),
        artifacts=(),
        quality=quality,
        provenance=ProvenanceRecord(
            run_id=run_id,
            source_kind="synthetic_world_fsm",
            generator_version=generator_version,
            seed=seed,
            scenario_hash=scenario_hash,
            git_sha=git_sha,
            model_id=None,
            prompt_hash=None,
            created_at=created_at,
            license_id="Proprietary-Synthetic-v1",
            consent_basis="fully_synthetic",
        ),
        metadata=(
            KeyValue(key="replicate_index", value=str(replicate_index)),
            KeyValue(key="persona_id", value=blueprint.persona_id),
            KeyValue(key="latent_arc_id", value=blueprint.latent_arc_id),
            KeyValue(key="risk_level", value=blueprint.risk_level),
        ),
    )


def replace_rendered_text(
    trajectory: ExperienceTrajectory,
    *,
    rendered_slots: tuple[tuple[str, str], ...],
    model_id: str,
    prompt_hash: str,
) -> ExperienceTrajectory:
    """Replace only text fields after validating a complete stable-ID mapping."""

    if trajectory.generation_tier is not GenerationTier.STRUCTURAL:
        raise ValueError("only structural trajectories can be rendered")
    if not model_id.strip():
        raise ValueError("model_id must be non-empty")
    if len(prompt_hash) != 64:
        raise ValueError("prompt_hash must be a SHA-256 digest")
    slot_map = dict(rendered_slots)
    if len(slot_map) != len(rendered_slots):
        raise ValueError("rendered_slots contains duplicate turn ids")
    expected_ids = {turn.turn_id for session in trajectory.sessions for turn in session.turns}
    if set(slot_map) != expected_ids:
        missing = sorted(expected_ids - set(slot_map))
        unknown = sorted(set(slot_map) - expected_ids)
        raise ValueError(f"rendered slot ids must match trajectory turns; missing={missing}, unknown={unknown}")
    if any(not text.strip() for text in slot_map.values()):
        raise ValueError("rendered slot text must be non-empty")

    rendered_sessions = tuple(
        replace(
            session,
            turns=tuple(replace(turn, text=slot_map[turn.turn_id]) for turn in session.turns),
        )
        for session in trajectory.sessions
    )
    rendered = replace(
        trajectory,
        generation_tier=GenerationTier.RENDERED,
        sessions=rendered_sessions,
        provenance=replace(
            trajectory.provenance,
            source_kind="synthetic_world_fsm_plus_text_render",
            model_id=model_id,
            prompt_hash=prompt_hash,
        ),
        quality=trajectory.quality
        + (
            QualityRecord(
                quality_id=f"{trajectory.trajectory_id}:quality:render_contract",
                check_kind="stable_text_slot_only",
                passed=True,
                severity=QualitySeverity.INFO,
                score=1.0,
                evidence_refs=(prompt_hash,),
                description=(
                    "Renderer replaced the complete set of stable-ID text slots without changing generator truth."
                ),
            ),
        ),
    )
    _assert_truth_unchanged(trajectory, rendered)
    return rendered


def text_slot_request(trajectory: ExperienceTrajectory) -> tuple[dict[str, object], ...]:
    """Build the renderer input without granting write access to truth fields."""

    truth_by_id = {frame.frame_id: frame for frame in trajectory.truth_frames}
    slots: list[dict[str, object]] = []
    previous_contract: tuple[str, ...] = ()
    for session in trajectory.sessions:
        for turn in session.turns:
            if turn.latent_frame_ref is not None:
                frame = truth_by_id[turn.latent_frame_ref]
                previous_contract = frame.response_contract
                slots.append(
                    {
                        "turn_id": turn.turn_id,
                        "role": turn.role.value,
                        "language": trajectory.language,
                        "phase": frame.phase_id,
                        "observable_facts": [{"key": item.key, "value": item.value} for item in frame.observable_facts],
                        "private_motivation": [{"key": item.key, "value": item.value} for item in frame.private_facts],
                        "response_contract": [],
                    }
                )
            else:
                slots.append(
                    {
                        "turn_id": turn.turn_id,
                        "role": turn.role.value,
                        "language": trajectory.language,
                        "phase": "response",
                        "observable_facts": [],
                        "private_motivation": [],
                        "response_contract": list(previous_contract),
                    }
                )
    return tuple(slots)


def _assert_truth_unchanged(
    structural: ExperienceTrajectory,
    rendered: ExperienceTrajectory,
) -> None:
    if structural.truth_frames != rendered.truth_frames:
        raise AssertionError("renderer changed latent truth frames")
    if structural.annotations != rendered.annotations:
        raise AssertionError("renderer changed annotations")
    if structural.snapshot_frames != rendered.snapshot_frames:
        raise AssertionError("renderer changed runtime observations")
    structural_turn_shape = tuple(
        (
            turn.turn_id,
            turn.session_index,
            turn.turn_index,
            turn.role,
            turn.event_id,
            turn.latent_frame_ref,
            turn.snapshot_refs,
            turn.artifact_refs,
            turn.metadata,
        )
        for session in structural.sessions
        for turn in session.turns
    )
    rendered_turn_shape = tuple(
        (
            turn.turn_id,
            turn.session_index,
            turn.turn_index,
            turn.role,
            turn.event_id,
            turn.latent_frame_ref,
            turn.snapshot_refs,
            turn.artifact_refs,
            turn.metadata,
        )
        for session in rendered.sessions
        for turn in session.turns
    )
    if structural_turn_shape != rendered_turn_shape:
        raise AssertionError("renderer changed non-text turn fields")


def _derived_seed(scenario_id: str, replicate_index: int, seed: int) -> int:
    payload = f"{scenario_id}\0{replicate_index}\0{seed}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _phase_for(ordinal: int, total: int) -> str:
    phase_index = min((ordinal * len(_PHASES)) // max(total, 1), len(_PHASES) - 1)
    return _PHASES[phase_index]


def _select(values: tuple[str, ...], index: int) -> str:
    return values[index % len(values)]


def _render_structural_user_text(
    *,
    blueprint: ScenarioBlueprint,
    phase: str,
    fact: str,
    ordinal: int,
) -> str:
    language = "zh" if blueprint.language == "zh" or (blueprint.language == "bilingual" and ordinal % 2 == 0) else "en"
    if language == "zh":
        return f"【结构槽 {blueprint.scenario_id}/{phase}】用户陈述可观察事实：{fact}"
    return f"[STRUCTURAL SLOT {blueprint.scenario_id}/{phase}] The user states the observable fact: {fact}"


def _render_structural_assistant_text(
    *,
    blueprint: ScenarioBlueprint,
    ordinal: int,
) -> str:
    contract = blueprint.response_contract[ordinal % len(blueprint.response_contract)]
    language = "zh" if blueprint.language == "zh" or (blueprint.language == "bilingual" and ordinal % 2 == 0) else "en"
    if language == "zh":
        return f"【待渲染助手槽】响应契约：{contract}"
    return f"[ASSISTANT SLOT TO RENDER] Response contract: {contract}"


__all__ = [
    "FAMILY_TARGET_OWNERS",
    "FAMILY_TARGET_OWNER_SETS",
    "GENERATOR_VERSION",
    "compile_structural_trajectory",
    "replace_rendered_text",
    "text_slot_request",
]
