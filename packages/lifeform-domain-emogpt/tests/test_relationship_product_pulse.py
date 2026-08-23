from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from dataclasses import fields, replace

import pytest

from lifeform_domain_emogpt.lab.relationship_product_pulse import (
    RelationshipProductOnboardingInput,
    RelationshipProductPreActionRequest,
    RelationshipProductPulseAuthorization,
    RelationshipProductSettlementInput,
    append_relationship_product_onboarding,
    prepare_relationship_product_preaction,
    settle_relationship_product_pulse,
)
from lifeform_domain_emogpt.relationship_action_contracts import (
    RELATIONSHIP_ACTIONS,
    RELATIONSHIP_OUTCOMES,
    RelationshipAction,
)
from lifeform_domain_emogpt.relationship_action_gate import (
    RelationshipActionGate,
    RelationshipActionGateCheckpoint,
    RelationshipActionGateDecision,
    RelationshipActionGateMode,
)
from lifeform_domain_emogpt.relationship_condition_reader import (
    PrototypeRelationshipPreferenceForecastRuntime,
    RelationshipConditionPrototype,
    RelationshipConditionReaderArtifact,
)
from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidence,
    DialogueExternalOutcomeEvidenceSource,
    DialogueExternalOutcomeKind,
)
from volvence_zero.social import (
    PreferenceActionForecastRequest,
)
from volvence_zero.social_cognition import PreferenceActionOutcomeEvidence
from volvence_zero.substrate import SubstrateSnapshot, SurfaceKind


_ACTION_IDS = tuple(action.value for action in RELATIONSHIP_ACTIONS)
_OUTCOME_IDS = tuple(outcome.value for outcome in RELATIONSHIP_OUTCOMES)
_SUBJECT_SCOPE = hashlib.sha256(b"relationship-product-pulse-subject").hexdigest()


class _DeterministicFakeEmbedder:
    def embed(self, text: str) -> tuple[float, ...]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return tuple((value + 1) / 256.0 for value in digest[:16])


class _DeterministicFakeWorld:
    """Typed transition test double with no preferred-action/evaluator label."""

    def settle(
        self,
        *,
        preaction,
    ) -> RelationshipProductSettlementInput:
        selected = preaction.gate_decision.selected_action_id
        kind = (
            DialogueExternalOutcomeKind.MISSED
            if selected == RelationshipAction.NEUTRAL_NOOP.value
            else DialogueExternalOutcomeKind.FELT_HEARD
        )
        evidence_ref = hashlib.sha256(
            f"{preaction.forecast.decision_id}:{selected}:{kind.value}".encode(
                "utf-8"
            )
        ).hexdigest()
        external = DialogueExternalOutcomeEvidence(
            evidence_id=f"fake-world:{evidence_ref}",
            turn_index=preaction.request.outcome_turn_index,
            kind=kind,
            source=DialogueExternalOutcomeEvidenceSource.ENVIRONMENT,
            confidence=1.0,
            evidence_ref=evidence_ref,
            description="Deterministic typed product-pulse outcome.",
            session_scope=preaction.request.forecast_request.session_scope,
            action_turn_index=preaction.request.forecast_request.turn_index,
            forecast_id=preaction.forecast.forecast_id,
            decision_id=preaction.forecast.decision_id,
            action_id=selected,
        )
        owner_evidence = PreferenceActionOutcomeEvidence(
            evidence_id=f"owner-outcome:{preaction.forecast.decision_id}",
            interlocutor_id=preaction.forecast.interlocutor_id,
            observation_summary=(
                preaction.request.forecast_request.current_observation
            ),
            action_id=selected,
            observed_outcome_id=kind.value,
            reaction_summary=f"Typed fake-world reaction: {kind.value}.",
            source_turn=preaction.request.outcome_turn_index,
            evidence_refs=(evidence_ref,),
        )
        return RelationshipProductSettlementInput(
            external_outcome=external,
            owner_outcome_evidence=owner_evidence,
            credit_timestamp_ms=preaction.request.outcome_turn_index * 1000,
            apply_credit_to_gate=True,
        )


def _reader() -> PrototypeRelationshipPreferenceForecastRuntime:
    artifact = RelationshipConditionReaderArtifact(
        embedding_model_id="deterministic-fake-bge",
        embedding_weights_sha256=hashlib.sha256(b"fake-bge").hexdigest(),
        prototypes=(
            RelationshipConditionPrototype(
                label="closeness_request",
                summary="The person needs calm presence without pressure.",
            ),
            RelationshipConditionPrototype(
                label="space_request",
                summary="The person needs space with a reliable return option.",
            ),
        ),
    )
    return PrototypeRelationshipPreferenceForecastRuntime(
        artifact=artifact,
        embedder=_DeterministicFakeEmbedder(),
    )


def _placeholder_substrate() -> SubstrateSnapshot:
    return SubstrateSnapshot(
        model_id="relationship-product-pulse-test-placeholder",
        is_frozen=True,
        surface_kind=SurfaceKind.PLACEHOLDER,
        token_logits=(),
        feature_surface=(),
        residual_activations=(),
        residual_sequence=(),
        unavailable_fields=(),
        description="Typed product-pulse test; no model or CUDA.",
    )


def _request(*, decision_index: int) -> RelationshipProductPreActionRequest:
    action_turn = 1 + decision_index * 2
    return RelationshipProductPreActionRequest(
        session_id=f"product-session-{decision_index}",
        forecast_request=PreferenceActionForecastRequest(
            decision_id=f"product-decision-{decision_index}",
            interlocutor_id="primary",
            current_observation=(
                f"Public relationship observation {decision_index}; no hidden truth."
            ),
            observation_ref=f"public-observation:{decision_index}",
            candidate_action_ids=_ACTION_IDS,
            outcome_ids=_OUTCOME_IDS,
            turn_index=action_turn,
            session_scope=_SUBJECT_SCOPE,
        ),
        outcome_turn_index=action_turn + 1,
    )


def _authorization() -> RelationshipProductPulseAuthorization:
    checkpoint = RelationshipActionGate().export_checkpoint()
    return RelationshipProductPulseAuthorization(
        authorization_id=hashlib.sha256(b"pulse-authorization").hexdigest(),
        allowed_policy_artifact_id=checkpoint.artifact_id,
        allowed_policy_artifact_version=checkpoint.artifact_version,
    )


async def _seed_owner_state():
    snapshot = await append_relationship_product_onboarding(
        owner_persistence_snapshot=None,
        onboarding=RelationshipProductOnboardingInput(
            session_id="seed-onboarding-session",
            session_index=0,
            turn_index=0,
            public_observation="A prior public request for calm presence.",
            action_id=RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value,
            observed_outcome_id=DialogueExternalOutcomeKind.FELT_HEARD.value,
            reaction_summary="The prior typed outcome was felt_heard.",
            evidence_ref="seed-owner-evidence-ref",
        ),
    )
    return snapshot.owner_persistence_snapshot


def test_product_onboarding_is_fresh_child_safe_and_snapshot_only() -> None:
    assert {field.name for field in fields(RelationshipProductOnboardingInput)} == {
        "session_id",
        "session_index",
        "turn_index",
        "public_observation",
        "action_id",
        "observed_outcome_id",
        "reaction_summary",
        "evidence_ref",
    }
    first_input = RelationshipProductOnboardingInput(
        session_id="onboarding-session-0",
        session_index=0,
        turn_index=0,
        public_observation="First public onboarding observation.",
        action_id=RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value,
        observed_outcome_id=DialogueExternalOutcomeKind.FELT_HEARD.value,
        reaction_summary="First typed public reaction.",
        evidence_ref="onboarding-evidence-ref-0",
    )
    first = asyncio.run(
        append_relationship_product_onboarding(
            owner_persistence_snapshot=None,
            onboarding=first_input,
        )
    )
    second = asyncio.run(
        append_relationship_product_onboarding(
            owner_persistence_snapshot=first.owner_persistence_snapshot,
            onboarding=RelationshipProductOnboardingInput(
                session_id="onboarding-session-1",
                session_index=1,
                turn_index=1,
                public_observation="Second public onboarding observation.",
                action_id=(
                    RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION.value
                ),
                observed_outcome_id=DialogueExternalOutcomeKind.HELPED.value,
                reaction_summary="Second typed public reaction.",
                evidence_ref="onboarding-evidence-ref-1",
            ),
        )
    )

    assert first.preference_snapshot.value.action_outcome_evidence[0].evidence_id == (
        first_input.evidence_id
    )
    assert len(second.preference_snapshot.value.action_outcome_evidence) == 2
    assert second.owner_persistence_snapshot != first.owner_persistence_snapshot


def test_gate_checkpoint_codec_is_canonical_strict_and_legacy_compatible() -> None:
    owner_state = asyncio.run(_seed_owner_state())
    preaction = asyncio.run(
        prepare_relationship_product_preaction(
            request=_request(decision_index=0),
            owner_persistence_snapshot=owner_state,
            gate_checkpoint=None,
            forecast_runtime=_reader(),
            gate_mode=RelationshipActionGateMode.LEARNED,
            authorization=_authorization(),
            substrate_snapshot=_placeholder_substrate(),
        )
    )
    checkpoint = preaction.gate_checkpoint_after
    payload = checkpoint.to_payload()

    assert set(payload) == {
        "schema_version",
        "artifact_id",
        "artifact_version",
        "weights",
        "bias",
        "update_count",
        "processed_credit_ids",
        "pending_decisions",
    }
    assert payload["pending_decisions"] == [
        preaction.gate_decision.to_payload()
    ]
    assert RelationshipActionGateCheckpoint.from_payload(
        json.loads(json.dumps(payload, sort_keys=True))
    ) == checkpoint
    assert RelationshipActionGateDecision.from_payload(
        preaction.gate_decision.to_payload()
    ) == preaction.gate_decision

    legacy = dict(payload)
    legacy["content_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert RelationshipActionGateCheckpoint.from_payload(legacy) == checkpoint

    extra = {**payload, "unexpected": True}
    with pytest.raises(ValueError, match="fields do not match schema"):
        RelationshipActionGateCheckpoint.from_payload(extra)
    tampered_legacy = {**legacy, "bias": 0.25}
    with pytest.raises(ValueError, match="content hash mismatch"):
        RelationshipActionGateCheckpoint.from_payload(tampered_legacy)
    decision_payload = copy.deepcopy(preaction.gate_decision.to_payload())
    decision_payload["artifact_version"] = True
    with pytest.raises(ValueError, match="must be an integer"):
        RelationshipActionGateDecision.from_payload(decision_payload)


def test_product_pulse_freezes_preaction_then_settles_only_from_pe_credit() -> None:
    preaction_field_names = {field.name for field in fields(RelationshipProductPreActionRequest)}
    assert preaction_field_names == {
        "session_id",
        "forecast_request",
        "outcome_turn_index",
    }
    assert not preaction_field_names.intersection(
        {"preferred_action", "expected_action", "evaluator", "judge", "outcome"}
    )

    owner_state = asyncio.run(_seed_owner_state())
    preaction = asyncio.run(
        prepare_relationship_product_preaction(
            request=_request(decision_index=0),
            owner_persistence_snapshot=owner_state,
            gate_checkpoint=None,
            forecast_runtime=_reader(),
            gate_mode=RelationshipActionGateMode.LEARNED,
            authorization=_authorization(),
            substrate_snapshot=_placeholder_substrate(),
        )
    )

    assert preaction.forecast.condition_readout is not None
    assert preaction.gate_decision.selected_action_id == "neutral_noop"
    assert preaction.temporal_snapshot.value.active_abstract_action == "neutral_noop"
    assert len(preaction.gate_checkpoint_after.pending_decisions) == 1
    assert preaction.gate_checkpoint_before.update_count == 0

    world = _DeterministicFakeWorld()
    settlement_input = world.settle(preaction=preaction)
    settlement = asyncio.run(
        settle_relationship_product_pulse(
            preaction=preaction,
            settlement_input=settlement_input,
        )
    )

    assert settlement.settlement.forecast_id == preaction.forecast.forecast_id
    assert settlement.credit.prediction_id == preaction.forecast.forecast_id
    assert settlement.credit.level == "relationship_action_prediction_error"
    assert settlement.credit.source_event.startswith("social_pe:social-pe:")
    assert settlement.gate_update is not None
    assert settlement.gate_checkpoint.update_count == 1
    assert not settlement.gate_checkpoint.pending_decisions
    assert settlement.owner_persistence_snapshot != owner_state

    resumed = asyncio.run(
        prepare_relationship_product_preaction(
            request=_request(decision_index=1),
            owner_persistence_snapshot=settlement.owner_persistence_snapshot,
            gate_checkpoint=RelationshipActionGateCheckpoint.from_payload(
                settlement.gate_checkpoint.to_payload()
            ),
            forecast_runtime=_reader(),
            gate_mode=RelationshipActionGateMode.LEARNED,
            authorization=_authorization(),
            substrate_snapshot=_placeholder_substrate(),
        )
    )
    assert resumed.gate_checkpoint_before.update_count == 1
    assert resumed.gate_decision.steer_probability > 0.5
    assert resumed.gate_decision.selected_action_id != "neutral_noop"


def test_product_pulse_rejects_postaction_lineage_drift_and_non_placeholder() -> None:
    owner_state = asyncio.run(_seed_owner_state())
    arguments = {
        "request": _request(decision_index=0),
        "owner_persistence_snapshot": owner_state,
        "gate_checkpoint": None,
        "forecast_runtime": _reader(),
        "gate_mode": RelationshipActionGateMode.LEARNED,
        "authorization": _authorization(),
    }
    residual_surface = replace(
        _placeholder_substrate(),
        surface_kind=SurfaceKind.RESIDUAL_STREAM,
    )
    with pytest.raises(ValueError, match="typed simulation only"):
        asyncio.run(
            prepare_relationship_product_preaction(
                **arguments,
                substrate_snapshot=residual_surface,
            )
        )

    preaction = asyncio.run(
        prepare_relationship_product_preaction(
            **arguments,
            substrate_snapshot=_placeholder_substrate(),
        )
    )
    settlement_input = _DeterministicFakeWorld().settle(preaction=preaction)
    drifted_external = replace(
        settlement_input.external_outcome,
        action_id=RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value,
    )
    with pytest.raises(ValueError, match="external action lineage mismatch"):
        asyncio.run(
            settle_relationship_product_pulse(
                preaction=preaction,
                settlement_input=replace(
                    settlement_input,
                    external_outcome=drifted_external,
                ),
            )
        )
