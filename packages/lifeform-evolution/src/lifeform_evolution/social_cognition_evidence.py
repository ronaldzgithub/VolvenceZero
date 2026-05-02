"""Social cognition evidence gates (R16-R20).

This module is evidence-layer code. It runs typed probes through normal
contract/runtime surfaces and does not become a learning source.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from volvence_zero.credit import derive_social_prediction_error_credit_records
from volvence_zero.environment import (
    EnvironmentActorRef,
    EnvironmentEvent,
    EnvironmentEventKind,
    EnvironmentFrame,
)
from volvence_zero.integration import FinalRolloutConfig, run_final_wiring_turn
from volvence_zero.memory import MemoryStore, Track
from volvence_zero.runtime import WiringLevel
from volvence_zero.semantic_state import (
    SemanticProposal,
    SemanticProposalBatch,
    SemanticProposalOperation,
    SemanticProposalRuntime,
)
from volvence_zero.social_cognition import (
    BeliefAboutOtherSnapshot,
    CommonGroundAtom,
    ConversationalRoleSnapshot,
    FeelingAboutOtherSnapshot,
    GroupIdentity,
    MultiPartyIdentitySnapshot,
    OtherMindRecord,
    OtherMindRecordKind,
    OtherMindRecordStatus,
    PreferenceAboutOtherSnapshot,
    SocialPredictionError,
    SocialPredictionKind,
    SocialPredictionOutcome,
    SocialScopeKind,
)
from volvence_zero.social_common_ground_runtime import LLMCommonGroundProposalRuntime
from volvence_zero.social_tom_runtime import LLMToMProposalRuntime
from volvence_zero.substrate import FeatureSignal, FeatureSurfaceSubstrateAdapter


@dataclass(frozen=True)
class SocialCognitionEvidenceGate:
    gate_id: str
    name: str
    passed: bool
    summary: str
    metrics: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class SocialCognitionEvidenceReport:
    gates: tuple[SocialCognitionEvidenceGate, ...]
    description: str

    @property
    def passed(self) -> bool:
        return all(gate.passed for gate in self.gates)


class _ExplicitToMRuntime(SemanticProposalRuntime):
    runtime_id = "explicit-tom-evidence"

    def propose(
        self,
        *,
        target_slot: str,
        user_input: str | None,
        substrate_snapshot: object | None,
        memory_snapshot: object | None,
        previous_snapshot: object | None,
        turn_index: int,
    ) -> SemanticProposalBatch:
        del substrate_snapshot, memory_snapshot, previous_snapshot
        return SemanticProposalBatch(
            proposals=(
                SemanticProposal(
                    proposal_id=f"{target_slot}:evidence:{turn_index}",
                    target_slot=target_slot,
                    operation=SemanticProposalOperation.OBSERVE,
                    summary=f"{target_slot}:evidence-record",
                    detail=user_input or "explicit ToM evidence detail",
                    confidence=0.83,
                    evidence="explicit social cognition evidence proposal",
                    control_signal=0.41,
                ),
            ),
            runtime_id=self.runtime_id,
            schema_version=1,
            description=f"explicit ToM evidence proposal for {target_slot}",
        )


class _BeliefOnlyToMRuntime(_ExplicitToMRuntime):
    runtime_id = "belief-only-tom-evidence"

    def propose(
        self,
        *,
        target_slot: str,
        user_input: str | None,
        substrate_snapshot: object | None,
        memory_snapshot: object | None,
        previous_snapshot: object | None,
        turn_index: int,
    ) -> SemanticProposalBatch:
        if target_slot != "belief_about_other":
            return SemanticProposalBatch(
                proposals=(),
                runtime_id=self.runtime_id,
                schema_version=1,
                description=f"belief-only runtime skipped {target_slot}",
            )
        return super().propose(
            target_slot=target_slot,
            user_input=user_input,
            substrate_snapshot=substrate_snapshot,
            memory_snapshot=memory_snapshot,
            previous_snapshot=previous_snapshot,
            turn_index=turn_index,
        )


class _ScriptedToMProvider:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int = 384,
        temperature: float = 0.0,
    ) -> str:
        del prompt, max_new_tokens, temperature
        return self.response


async def run_social_cognition_evidence_async() -> SocialCognitionEvidenceReport:
    gates = (
        await _active_identity_memory_scope_gate(),
        _tom_owner_contract_gate(),
        await _explicit_tom_proposal_path_gate(),
        await _false_belief_preference_separation_gate(),
        await _structured_tom_runtime_path_gate(),
        await _affect_preference_separation_gate(),
        _wrong_addressee_social_pe_credit_gate(),
        await _role_prediction_diagnostic_gate(),
        await _active_role_frame_diagnostic_gate(),
        await _common_ground_diagnostic_gate(),
        await _structured_common_ground_runtime_gate(),
        await _reference_repair_common_ground_gate(),
        await _group_diagnostic_gate(),
    )
    passed_count = sum(1 for gate in gates if gate.passed)
    return SocialCognitionEvidenceReport(
        gates=gates,
        description=(
            f"Social cognition evidence gates passed {passed_count}/{len(gates)} "
            "for ToM owner separation and role PE credit."
        ),
    )


def run_social_cognition_evidence() -> SocialCognitionEvidenceReport:
    return asyncio.run(run_social_cognition_evidence_async())


async def _active_identity_memory_scope_gate() -> SocialCognitionEvidenceGate:
    result = await run_final_wiring_turn(
        config=FinalRolloutConfig(),
        substrate_adapter=_adapter(),
        user_input="Alice asks for careful planning help.",
        memory_store=MemoryStore(),
        environment_event=EnvironmentEvent(
            event_id="r16a-active-identity-evidence",
            event_kind=EnvironmentEventKind.USER_INPUT,
            trigger_kind="user_input",
            frame=EnvironmentFrame(
                actor=EnvironmentActorRef(actor_id="alice"),
                active_speaker_id="alice",
                addressee_ids=("self",),
                subject_ids=("alice",),
                audience_ids=("self", "alice"),
            ),
            scene_id="r16a-scene",
            timestamp_ms=1,
            provenance="social-cognition-evidence",
            payload_summary="Alice asks for careful planning help.",
        ),
        session_id="r16a-active-identity-session",
        wave_id="r16a-active-identity-wave",
    )
    identity = result.active_snapshots.get("multi_party_identity")
    memory = result.active_snapshots["memory"].value
    scoped_entries = tuple(
        entry
        for entry in memory.retrieved_entries
        if entry.subject_ids == ("alice",)
        and entry.audience_ids == ("self", "alice")
    )
    passed = (
        identity is not None
        and isinstance(identity.value, MultiPartyIdentitySnapshot)
        and identity.value.active_speaker_id == "alice"
        and identity.value.subject_ids == ("alice",)
        and bool(scoped_entries)
    )
    return SocialCognitionEvidenceGate(
        gate_id="R16A",
        name="active identity memory scope",
        passed=passed,
        metrics=(
            ("scoped_memory_entries", float(len(scoped_entries))),
            ("identity_active", 1.0 if identity is not None else 0.0),
        ),
        summary="ACTIVE multi-party identity scope drives memory write subject/audience without renderer inference.",
    )


def _tom_owner_contract_gate() -> SocialCognitionEvidenceGate:
    belief_record = OtherMindRecord(
        record_id="tom:belief:contract",
        interlocutor_id="alice",
        kind=OtherMindRecordKind.BELIEF,
        summary="Alice believes the meeting is tomorrow.",
        detail="Explicit contract probe.",
        confidence=0.8,
        status=OtherMindRecordStatus.ACTIVE,
        source_turn=1,
        prediction_error_refs=(),
        evidence="contract probe",
    )
    wrong_kind_rejected = False
    try:
        BeliefAboutOtherSnapshot(
            records=(
                OtherMindRecord(
                    record_id="tom:intent:wrong-kind",
                    interlocutor_id="alice",
                    kind=OtherMindRecordKind.INTENT,
                    summary="Alice intends to leave.",
                    detail="Wrong-kind contract probe.",
                    confidence=0.8,
                    status=OtherMindRecordStatus.ACTIVE,
                    source_turn=1,
                    prediction_error_refs=(),
                    evidence="contract probe",
                ),
            ),
            active_predictions=(),
            control_signal=0.0,
            description="wrong-kind probe",
        )
    except ValueError:
        wrong_kind_rejected = True

    valid_snapshot = BeliefAboutOtherSnapshot(
        records=(belief_record,),
        active_predictions=(),
        control_signal=0.2,
        description="valid belief contract probe",
    )
    passed = wrong_kind_rejected and valid_snapshot.records[0].kind is OtherMindRecordKind.BELIEF
    return SocialCognitionEvidenceGate(
        gate_id="T1",
        name="tom owner contract",
        passed=passed,
        metrics=(("belief_record_count", float(len(valid_snapshot.records))),),
        summary="ToM contract rejects wrong-kind records and accepts typed belief records.",
    )


async def _explicit_tom_proposal_path_gate() -> SocialCognitionEvidenceGate:
    result = await run_final_wiring_turn(
        config=FinalRolloutConfig(belief_about_other=WiringLevel.ACTIVE),
        substrate_adapter=_adapter(),
        user_input="Alice believes the meeting is tomorrow.",
        tom_proposal_runtime=_BeliefOnlyToMRuntime(),
        session_id="tom-evidence-proposal",
        wave_id="tom-evidence-wave",
        turn_index=2,
    )
    belief = result.active_snapshots["belief_about_other"].value
    default_preference = result.shadow_snapshots["preference_about_other"].value
    passed = (
        isinstance(belief, BeliefAboutOtherSnapshot)
        and len(belief.records) == 1
        and belief.records[0].kind is OtherMindRecordKind.BELIEF
        and isinstance(default_preference, PreferenceAboutOtherSnapshot)
        and default_preference.records == ()
    )
    return SocialCognitionEvidenceGate(
        gate_id="T2",
        name="explicit tom proposal path",
        passed=passed,
        metrics=(("belief_records", float(len(belief.records))),),
        summary="Explicit proposal runtime can populate one ToM owner without default classifier behavior.",
    )


async def _false_belief_preference_separation_gate() -> SocialCognitionEvidenceGate:
    result = await run_final_wiring_turn(
        config=FinalRolloutConfig(
            belief_about_other=WiringLevel.ACTIVE,
            preference_about_other=WiringLevel.ACTIVE,
        ),
        substrate_adapter=_adapter(),
        user_input="Alice believes tomorrow is meeting day and prefers slow planning.",
        tom_proposal_runtime=_ExplicitToMRuntime(),
        session_id="tom-evidence-separation",
        wave_id="tom-evidence-separation-wave",
        turn_index=3,
    )
    response_assembly = result.active_snapshots["response_assembly"].value
    counts = dict(response_assembly.semantic_record_counts)
    belief = result.active_snapshots["belief_about_other"].value
    preference = result.active_snapshots["preference_about_other"].value
    passed = (
        isinstance(belief, BeliefAboutOtherSnapshot)
        and isinstance(preference, PreferenceAboutOtherSnapshot)
        and len(belief.records) == 1
        and len(preference.records) == 1
        and belief.records[0].kind is OtherMindRecordKind.BELIEF
        and preference.records[0].kind is OtherMindRecordKind.PREFERENCE
        and counts.get("belief_about_other") == 1
        and counts.get("preference_about_other") == 1
    )
    return SocialCognitionEvidenceGate(
        gate_id="T3",
        name="false-belief preference separation",
        passed=passed,
        metrics=(
            ("belief_records", float(len(belief.records))),
            ("preference_records", float(len(preference.records))),
            ("assembly_belief_count", float(counts.get("belief_about_other", -1))),
            ("assembly_preference_count", float(counts.get("preference_about_other", -1))),
        ),
        summary="False-belief and preference-conflict probes remain in separate ToM owners.",
    )


async def _structured_tom_runtime_path_gate() -> SocialCognitionEvidenceGate:
    runtime = LLMToMProposalRuntime(
        provider=_ScriptedToMProvider(
            """
            [
              {
                "target_slot": "belief_about_other",
                "summary": "Alice believes the meeting is tomorrow.",
                "detail": "Alice explicitly says the meeting is tomorrow.",
                "evidence": "meeting is tomorrow",
                "confidence": 0.84,
                "control_signal": 0.33
              }
            ]
            """
        )
    )
    result = await run_final_wiring_turn(
        config=FinalRolloutConfig(belief_about_other=WiringLevel.ACTIVE),
        substrate_adapter=_adapter(),
        user_input="Alice believes the meeting is tomorrow.",
        tom_proposal_runtime=runtime,
        session_id="tom-structured-evidence",
        wave_id="tom-structured-evidence-wave",
        turn_index=8,
    )
    belief = result.active_snapshots["belief_about_other"].value
    passed = (
        isinstance(belief, BeliefAboutOtherSnapshot)
        and len(belief.records) == 1
        and belief.records[0].kind is OtherMindRecordKind.BELIEF
        and belief.records[0].evidence == "meeting is tomorrow"
    )
    return SocialCognitionEvidenceGate(
        gate_id="T4",
        name="structured tom runtime path",
        passed=passed,
        metrics=(("belief_records", float(len(belief.records))),),
        summary="Structured LLM ToM runtime can populate a targeted belief owner.",
    )


async def _affect_preference_separation_gate() -> SocialCognitionEvidenceGate:
    runtime = LLMToMProposalRuntime(
        provider=_ScriptedToMProvider(
            """
            [
              {
                "target_slot": "feeling_about_other",
                "summary": "Alice feels overwhelmed.",
                "detail": "Alice says work feels heavy right now.",
                "evidence": "work feels heavy",
                "confidence": 0.83,
                "control_signal": 0.45
              },
              {
                "target_slot": "preference_about_other",
                "summary": "Alice prefers gentle pacing.",
                "detail": "Alice asks to slow down before steps.",
                "evidence": "slow down before steps",
                "confidence": 0.82,
                "control_signal": 0.31
              }
            ]
            """
        )
    )
    result = await run_final_wiring_turn(
        config=FinalRolloutConfig(
            feeling_about_other=WiringLevel.ACTIVE,
            preference_about_other=WiringLevel.ACTIVE,
        ),
        substrate_adapter=_adapter(),
        user_input="Work feels heavy; please slow down before steps.",
        tom_proposal_runtime=runtime,
        session_id="tom-affect-preference-evidence",
        wave_id="tom-affect-preference-wave",
        turn_index=9,
    )
    feeling = result.active_snapshots["feeling_about_other"].value
    preference = result.active_snapshots["preference_about_other"].value
    counts = dict(result.active_snapshots["response_assembly"].value.semantic_record_counts)
    passed = (
        isinstance(feeling, FeelingAboutOtherSnapshot)
        and isinstance(preference, PreferenceAboutOtherSnapshot)
        and len(feeling.records) == 1
        and len(preference.records) == 1
        and feeling.records[0].kind is OtherMindRecordKind.FEELING
        and preference.records[0].kind is OtherMindRecordKind.PREFERENCE
        and counts.get("feeling_about_other") == 1
        and counts.get("preference_about_other") == 1
    )
    return SocialCognitionEvidenceGate(
        gate_id="T5",
        name="affect preference separation",
        passed=passed,
        metrics=(
            ("feeling_records", float(len(feeling.records))),
            ("preference_records", float(len(preference.records))),
        ),
        summary="Structured ToM runtime keeps transient feeling and durable preference separate.",
    )


def social_cognition_evidence_report_to_dict(
    report: SocialCognitionEvidenceReport,
) -> dict[str, object]:
    return {
        "passed": report.passed,
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
    }


def format_social_cognition_evidence_report(
    report: SocialCognitionEvidenceReport,
) -> str:
    lines = [
        "== Social cognition evidence report ==",
        f"overall: {'PASS' if report.passed else 'FAIL'}",
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
    return "\n".join(lines)


def _wrong_addressee_social_pe_credit_gate() -> SocialCognitionEvidenceGate:
    social_error = SocialPredictionError(
        error_id="social-pe:role:wrong-addressee:evidence",
        prediction_id="role-env-frame-1:role-assignment",
        kind=SocialPredictionKind.ROLE_ASSIGNMENT,
        outcome=SocialPredictionOutcome.DISCONFIRMED,
        magnitude=0.71,
        owner="ConversationalRoleModule",
        scope_kind=SocialScopeKind.INTERLOCUTOR,
        scope_id="alice",
        evidence=("Role prediction addressed Bob, but Carol was the intended addressee.",),
    )
    records = derive_social_prediction_error_credit_records(
        social_errors=(social_error,),
        timestamp_ms=44,
    )
    passed = (
        len(records) == 1
        and records[0].level == "social_prediction_error"
        and records[0].track is Track.SHARED
        and records[0].source_event == "social_pe:role_assignment"
        and records[0].credit_value == -0.71
        and "owner=ConversationalRoleModule" in records[0].context
    )
    return SocialCognitionEvidenceGate(
        gate_id="R1",
        name="wrong-addressee role pe credit",
        passed=passed,
        metrics=(("role_pe_credit", records[0].credit_value if records else 0.0),),
        summary="A typed wrong-addressee role PE can enter shared credit without renderer logic.",
    )


async def _role_prediction_diagnostic_gate() -> SocialCognitionEvidenceGate:
    result = await run_final_wiring_turn(
        config=FinalRolloutConfig(conversational_role=WiringLevel.ACTIVE),
        substrate_adapter=_adapter(),
        user_input="Alice tells Bob about Carol.",
        environment_event=EnvironmentEvent(
            event_id="role-evidence-frame-1",
            event_kind=EnvironmentEventKind.USER_INPUT,
            trigger_kind="user_input",
            frame=EnvironmentFrame(
                actor=EnvironmentActorRef(actor_id="alice"),
                active_speaker_id="alice",
                addressee_ids=("bob",),
                subject_ids=("carol",),
                audience_ids=("bob", "alice"),
            ),
            scene_id="role-evidence-scene",
            timestamp_ms=1,
            provenance="social-cognition-evidence",
            payload_summary="Alice tells Bob about Carol.",
        ),
        session_id="role-evidence-session",
        wave_id="role-evidence-wave",
    )
    role = result.active_snapshots["conversational_role"].value
    response_assembly = result.active_snapshots["response_assembly"].value
    counts = dict(response_assembly.semantic_record_counts)
    passed = (
        isinstance(role, ConversationalRoleSnapshot)
        and role.active_speaker_id == "alice"
        and role.addressee_ids == ("bob",)
        and role.subject_ids == ("carol",)
        and len(role.active_predictions) == 1
        and role.active_predictions[0].kind is SocialPredictionKind.ROLE_ASSIGNMENT
        and counts.get("conversational_role") == 1
        and "conversational_role" not in response_assembly.semantic_residue_summary
    )
    return SocialCognitionEvidenceGate(
        gate_id="R2",
        name="role prediction diagnostic visibility",
        passed=passed,
        metrics=(
            ("role_prediction_count", float(len(role.active_predictions))),
            ("assembly_role_count", float(counts.get("conversational_role", -1))),
        ),
        summary="Role assignment prediction is visible in response assembly diagnostics without renderer consumption.",
    )


async def _active_role_frame_diagnostic_gate() -> SocialCognitionEvidenceGate:
    result = await run_final_wiring_turn(
        config=FinalRolloutConfig(),
        substrate_adapter=_adapter(),
        user_input="Alice tells Bob about Carol.",
        environment_event=EnvironmentEvent(
            event_id="role-active-frame-1",
            event_kind=EnvironmentEventKind.USER_INPUT,
            trigger_kind="user_input",
            frame=EnvironmentFrame(
                actor=EnvironmentActorRef(actor_id="alice"),
                active_speaker_id="alice",
                addressee_ids=("bob",),
                subject_ids=("carol",),
                audience_ids=("bob", "alice"),
            ),
            scene_id="role-active-scene",
            timestamp_ms=1,
            provenance="social-cognition-evidence",
            payload_summary="Alice tells Bob about Carol.",
        ),
        session_id="role-active-evidence-session",
        wave_id="role-active-evidence-wave",
    )
    role_snapshot = result.active_snapshots.get("conversational_role")
    role = role_snapshot.value if role_snapshot is not None else None
    response_assembly = result.active_snapshots["response_assembly"].value
    counts = dict(response_assembly.semantic_record_counts)
    passed = (
        isinstance(role, ConversationalRoleSnapshot)
        and role.active_speaker_id == "alice"
        and role.addressee_ids == ("bob",)
        and role.subject_ids == ("carol",)
        and len(role.active_predictions) == 1
        and counts.get("conversational_role") == 1
        and "conversational_role" not in response_assembly.semantic_residue_summary
    )
    return SocialCognitionEvidenceGate(
        gate_id="R18A",
        name="active role frame diagnostics",
        passed=passed,
        metrics=(
            ("role_prediction_count", float(len(role.active_predictions) if isinstance(role, ConversationalRoleSnapshot) else -1)),
            ("assembly_role_count", float(counts.get("conversational_role", -1))),
        ),
        summary="Default ACTIVE conversational role consumes EnvironmentEvent frame as diagnostics without renderer leakage.",
    )


async def _common_ground_diagnostic_gate() -> SocialCognitionEvidenceGate:
    dyad = CommonGroundAtom(
        atom_id="cg:evidence:dyad:alice-bob",
        scope_id="alice:bob",
        scope_kind=SocialScopeKind.DYAD,
        summary="Alice and Bob both know the plan changed.",
        recursion_depth=2,
        confidence=0.74,
        accepted_by_ids=("alice", "bob"),
        evidence=("both confirmed the change",),
    )
    group = CommonGroundAtom(
        atom_id="cg:evidence:group:launch",
        scope_id="team:launch",
        scope_kind=SocialScopeKind.GROUP,
        summary="The launch team knows the deadline moved.",
        recursion_depth=1,
        confidence=0.69,
        accepted_by_ids=("alice", "bob", "carol"),
        evidence=("team acknowledgement",),
    )
    result = await run_final_wiring_turn(
        config=FinalRolloutConfig(common_ground=WiringLevel.ACTIVE),
        substrate_adapter=_adapter(),
        common_ground_dyad_atoms=(dyad,),
        common_ground_group_atoms=(group,),
        session_id="common-ground-evidence-session",
        wave_id="common-ground-evidence-wave",
    )
    common_ground = result.active_snapshots["common_ground"].value
    response_assembly = result.active_snapshots["response_assembly"].value
    counts = dict(response_assembly.semantic_record_counts)
    atom_count = len(common_ground.dyad_atoms) + len(common_ground.group_atoms)
    passed = (
        atom_count == 2
        and counts.get("common_ground") == 2
        and "common_ground" not in response_assembly.semantic_residue_summary
    )
    return SocialCognitionEvidenceGate(
        gate_id="G1",
        name="common-ground diagnostic visibility",
        passed=passed,
        metrics=(
            ("common_ground_atom_count", float(atom_count)),
            ("assembly_common_ground_count", float(counts.get("common_ground", -1))),
        ),
        summary="Explicit dyad/group common-ground atoms are visible in response assembly diagnostics only.",
    )


async def _structured_common_ground_runtime_gate() -> SocialCognitionEvidenceGate:
    runtime = LLMCommonGroundProposalRuntime(
        provider=_ScriptedToMProvider(
            """
            [
              {
                "scope_kind": "dyad",
                "scope_id": "self:alice",
                "summary": "Self and Alice both know the plan should slow down.",
                "accepted_by_ids": ["self", "alice"],
                "evidence": "slow down like we agreed",
                "confidence": 0.82,
                "recursion_depth": 2,
                "control_signal": 0.40
              },
              {
                "scope_kind": "group",
                "scope_id": "team:launch",
                "summary": "The launch team accepted the new deadline.",
                "accepted_by_ids": ["alice", "bob", "carol"],
                "evidence": "all confirmed",
                "confidence": 0.78,
                "recursion_depth": 1,
                "control_signal": 0.35
              }
            ]
            """
        )
    )
    result = await run_final_wiring_turn(
        config=FinalRolloutConfig(common_ground=WiringLevel.ACTIVE),
        substrate_adapter=_adapter(),
        user_input="Let's slow down like we agreed; all confirmed the deadline.",
        common_ground_proposal_runtime=runtime,
        session_id="common-ground-structured-evidence",
        wave_id="common-ground-structured-wave",
        turn_index=10,
    )
    common_ground = result.active_snapshots["common_ground"].value
    counts = dict(result.active_snapshots["response_assembly"].value.semantic_record_counts)
    passed = (
        len(common_ground.dyad_atoms) == 1
        and len(common_ground.group_atoms) == 1
        and counts.get("common_ground") == 2
    )
    return SocialCognitionEvidenceGate(
        gate_id="G2",
        name="structured common-ground runtime path",
        passed=passed,
        metrics=(
            ("dyad_atoms", float(len(common_ground.dyad_atoms))),
            ("group_atoms", float(len(common_ground.group_atoms))),
            ("assembly_common_ground_count", float(counts.get("common_ground", -1))),
        ),
        summary="Structured common-ground runtime can populate dyad and group atoms.",
    )


async def _reference_repair_common_ground_gate() -> SocialCognitionEvidenceGate:
    runtime = LLMCommonGroundProposalRuntime(
        provider=_ScriptedToMProvider(
            """
            [
              {
                "scope_kind": "dyad",
                "scope_id": "self:alice",
                "summary": "Self and Alice repaired the reference to the work plan.",
                "accepted_by_ids": ["self", "alice"],
                "evidence": "I meant the work plan from earlier, not the home thing",
                "confidence": 0.84,
                "recursion_depth": 2,
                "control_signal": 0.46
              }
            ]
            """
        )
    )
    result = await run_final_wiring_turn(
        config=FinalRolloutConfig(common_ground=WiringLevel.ACTIVE),
        substrate_adapter=_adapter(),
        user_input="I meant the work plan from earlier, not the home thing.",
        common_ground_proposal_runtime=runtime,
        session_id="common-ground-repair-evidence",
        wave_id="common-ground-repair-wave",
        turn_index=11,
    )
    common_ground = result.active_snapshots["common_ground"].value
    atom = common_ground.dyad_atoms[0] if common_ground.dyad_atoms else None
    passed = (
        atom is not None
        and atom.scope_kind is SocialScopeKind.DYAD
        and "work plan" in atom.summary
        and "work plan from earlier" in atom.evidence[0]
    )
    return SocialCognitionEvidenceGate(
        gate_id="G3",
        name="reference repair common-ground probe",
        passed=passed,
        metrics=(("repair_atoms", float(len(common_ground.dyad_atoms))),),
        summary="A repair/clarification signal can enter common-ground owner as a dyad atom.",
    )


async def _group_diagnostic_gate() -> SocialCognitionEvidenceGate:
    group = GroupIdentity(
        group_id="group:launch",
        member_ids=("alice", "bob", "carol"),
        display_name="Launch group",
        confidence=0.82,
        evidence=("host membership",),
    )
    result = await run_final_wiring_turn(
        config=FinalRolloutConfig(groups=WiringLevel.ACTIVE),
        substrate_adapter=_adapter(),
        group_identities=(group,),
        active_group_id="group:launch",
        group_joint_attention=("launch-plan",),
        group_joint_commitments=("commitment:ship",),
        group_regime_id="problem_solving",
        session_id="group-evidence-session",
        wave_id="group-evidence-wave",
    )
    groups = result.active_snapshots["groups"].value
    response_assembly = result.active_snapshots["response_assembly"].value
    counts = dict(response_assembly.semantic_record_counts)
    passed = (
        len(groups.groups) == 1
        and groups.active_group_id == "group:launch"
        and len(groups.joint_commitments) == 1
        and counts.get("groups") == 1
        and counts.get("group_joint_commitments") == 1
        and "groups" not in response_assembly.semantic_residue_summary
    )
    return SocialCognitionEvidenceGate(
        gate_id="GROUP1",
        name="group diagnostic visibility",
        passed=passed,
        metrics=(
            ("group_count", float(len(groups.groups))),
            ("joint_commitment_count", float(len(groups.joint_commitments))),
            ("assembly_group_count", float(counts.get("groups", -1))),
            (
                "assembly_group_joint_commitment_count",
                float(counts.get("group_joint_commitments", -1)),
            ),
        ),
        summary="Explicit group state is visible in response assembly diagnostics without renderer consumption.",
    )


def _adapter() -> FeatureSurfaceSubstrateAdapter:
    return FeatureSurfaceSubstrateAdapter(
        model_id="social-cognition-evidence-model",
        feature_surface=(
            FeatureSignal(name="social_cognition_evidence", values=(0.5,), source="evidence"),
        ),
    )


__all__ = (
    "SocialCognitionEvidenceGate",
    "SocialCognitionEvidenceReport",
    "format_social_cognition_evidence_report",
    "run_social_cognition_evidence",
    "run_social_cognition_evidence_async",
    "social_cognition_evidence_report_to_dict",
)
