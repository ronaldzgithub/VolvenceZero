"""Social cognition evidence gates (R16-R20).

This module is evidence-layer code. It runs typed probes through normal
contract/runtime surfaces and does not become a learning source.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from volvence_zero.integration import FinalRolloutConfig, run_final_wiring_turn
from volvence_zero.runtime import WiringLevel
from volvence_zero.semantic_state import (
    SemanticProposal,
    SemanticProposalBatch,
    SemanticProposalOperation,
    SemanticProposalRuntime,
)
from volvence_zero.social_cognition import (
    BeliefAboutOtherSnapshot,
    OtherMindRecord,
    OtherMindRecordKind,
    OtherMindRecordStatus,
    PreferenceAboutOtherSnapshot,
)
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


async def run_social_cognition_evidence_async() -> SocialCognitionEvidenceReport:
    gates = (
        _tom_owner_contract_gate(),
        await _explicit_tom_proposal_path_gate(),
        await _false_belief_preference_separation_gate(),
    )
    passed_count = sum(1 for gate in gates if gate.passed)
    return SocialCognitionEvidenceReport(
        gates=gates,
        description=(
            f"Social cognition evidence gates passed {passed_count}/{len(gates)} "
            "for R17 ToM owner separation."
        ),
    )


def run_social_cognition_evidence() -> SocialCognitionEvidenceReport:
    return asyncio.run(run_social_cognition_evidence_async())


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
