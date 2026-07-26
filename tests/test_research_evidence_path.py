"""Contract tests for the research-tool evidence path.

The claim under test is narrow and load-bearing: **a research result
only becomes a belief if it can say where it came from, when it was
true, and what it applies to.** Miss any one and it is an assertion that
happens to be adjacent to a tool call.

What makes this worth testing rather than trusting: a search summary and
a fabricated sentence arrive at the owner in exactly the same shape.
Nothing about the text distinguishes them. The provenance fields are the
only thing that can, so the rule has to bite on structure — and it has
to bite by adjusting an honest confidence rather than by routing
unprovenanced claims somewhere special, because a special route is a
thing that can be forgotten at the next call site.

The consequence chain is the interesting part and is tested end to end:
an unprovenanced claim lands in ``verification_needs`` → becomes an open
unknown → raises ``unknown_dominance`` → which the panorama gate reads →
which value-of-information then targets. Nowhere is anything told to
chase unverified claims; it falls out.
"""

from __future__ import annotations

import pytest

from volvence_zero.regime.decision_structure import (
    derive_decision_structure_signals,
)
from volvence_zero.semantic_state import (
    BELIEF_VERIFICATION_CONFIDENCE_THRESHOLD,
    EvidenceProvenance,
    ToolResultSemanticEvent,
)
from volvence_zero.semantic_state.proposal_runtime import (
    ToolResultSemanticAdapter,
    semantic_events_from_tool_result,
)


def _claim(**overrides: object) -> EvidenceProvenance:
    fields: dict[str, object] = {
        "claim_id": "c1",
        "source": "sec-filing:0001",
        "as_of": "2025-11-14",
        "scope": "entity:acme-inc",
        "confidence": 0.85,
    }
    fields.update(overrides)
    return EvidenceProvenance(**fields)  # type: ignore[arg-type]


def _event(*claims: EvidenceProvenance) -> ToolResultSemanticEvent:
    return ToolResultSemanticEvent(
        event_id="evt-1",
        tool_name="research_public_company",
        action_id="act-1",
        status="succeeded",
        summary="findings",
        detail="",
        provenance=claims,
    )


def _adapt(*claims: EvidenceProvenance):
    return ToolResultSemanticAdapter().adapt(
        event=_event(*claims), target_slot="belief_assumption", turn_index=1
    )


# ---------------------------------------------------------------------------
# What counts as evidence
# ---------------------------------------------------------------------------


def test_complete_provenance_is_admitted_as_belief() -> None:
    (proposal,) = _adapt(_claim())
    assert proposal.confidence >= BELIEF_VERIFICATION_CONFIDENCE_THRESHOLD


@pytest.mark.parametrize("missing", ["source", "as_of", "scope"])
def test_any_missing_provenance_field_blocks_belief(missing: str) -> None:
    """All three are required, not two out of three.

    A source with no date is a claim about an unknown moment; a source
    with no scope is a claim about an unknown subject. Both read as
    facts once they are in a sentence.
    """
    (proposal,) = _adapt(_claim(**{missing: ""}))
    assert proposal.confidence < BELIEF_VERIFICATION_CONFIDENCE_THRESHOLD


def test_a_confident_source_cannot_vouch_for_itself() -> None:
    """The cap is not negotiable by the claim's own confidence.

    A backend asserting 0.99 on an unsourced figure is precisely the
    failure mode: fluent text carries high confidence for free.
    """
    (proposal,) = _adapt(_claim(source="", confidence=0.99))
    assert proposal.confidence < BELIEF_VERIFICATION_CONFIDENCE_THRESHOLD


def test_claims_are_admitted_separately() -> None:
    """One good claim must not carry a bad one across the threshold.

    A research call routinely returns a checkable filing date next to a
    guessed valuation. Merged into one record they share one confidence,
    and the checkable part drags the guess over the line.
    """
    good, bad = _adapt(
        _claim(claim_id="filing-date"),
        _claim(claim_id="valuation", source="", as_of=""),
    )
    assert good.confidence >= BELIEF_VERIFICATION_CONFIDENCE_THRESHOLD
    assert bad.confidence < BELIEF_VERIFICATION_CONFIDENCE_THRESHOLD


def test_audit_trail_names_what_was_missing() -> None:
    """"Low confidence" is not a diagnosis; "no as_of" is."""
    (proposal,) = _adapt(_claim(as_of=""))
    assert "incomplete_provenance=as_of" in proposal.evidence


def test_provenance_appears_in_the_evidence_string() -> None:
    (proposal,) = _adapt(_claim())
    for fragment in ("sec-filing:0001", "2025-11-14", "entity:acme-inc"):
        assert fragment in proposal.evidence


def test_action_shaped_tools_are_unaffected() -> None:
    """Writing a file asserts nothing about the world.

    The legacy single-observation path must stay byte-identical for
    every tool that returns no claims.
    """
    event = ToolResultSemanticEvent(
        event_id="evt-2",
        tool_name="write_file",
        action_id="act-2",
        status="succeeded",
        summary="wrote 4 lines",
        detail="",
        confidence=0.9,
    )
    (proposal,) = ToolResultSemanticAdapter().adapt(
        event=event, target_slot="belief_assumption", turn_index=1
    )
    assert proposal.confidence == pytest.approx(0.9)
    assert proposal.summary == "tool-evidence:write_file"


def test_event_builder_threads_provenance() -> None:
    batch = semantic_events_from_tool_result(
        event_id="evt-3",
        tool_name="research_public_company",
        action_id="act-3",
        status="succeeded",
        summary="findings",
        detail="",
        provenance=(_claim(),),
    )
    assert batch.events[0].provenance == (_claim(),)


# ---------------------------------------------------------------------------
# The consequence: an unverified claim becomes something to chase
# ---------------------------------------------------------------------------


def test_unverified_research_raises_unknown_dominance() -> None:
    """The chain that makes this design self-correcting.

    Nothing tells the system to pursue its own unsourced claims. An
    unprovenanced finding lands in ``verification_needs``, which is what
    ``unknown_dominance`` measures, which is one of the four signals the
    panorama gate conjoins — and the same quantity value-of-information
    ranks when picking the next question.
    """
    from companion_standard.semantic_state import (
        BeliefAssumptionSnapshot,
        SemanticRecord,
    )

    def belief_snapshot(*, verification_count: int) -> BeliefAssumptionSnapshot:
        needs = tuple(
            SemanticRecord(
                record_id=f"verify-{index}",
                summary="",
                detail="",
                confidence=0.5,
                status="open",
                source_turn=1,
                evidence="",
            )
            for index in range(verification_count)
        )
        return BeliefAssumptionSnapshot(
            beliefs=(),
            assumptions=needs,
            verification_needs=needs,
            contradiction_refs=(),
            mean_confidence=0.5 if needs else 0.9,
            control_signal=0.0,
            description="",
        )

    without = derive_decision_structure_signals(
        belief_assumption=belief_snapshot(verification_count=0)
    )
    with_unverified = derive_decision_structure_signals(
        belief_assumption=belief_snapshot(verification_count=3)
    )
    assert with_unverified.unknown_dominance > without.unknown_dominance


# ---------------------------------------------------------------------------
# The descriptor
# ---------------------------------------------------------------------------


def test_research_descriptor_demands_provenance_in_its_output_schema() -> None:
    """A backend that cannot cite must fail at the boundary.

    Requiring the fields in the schema means an uncitable result is a
    contract error, not authoritative-sounding prose.
    """
    from lifeform_domain_growth_advisor.research_affordances import (
        RESEARCH_PUBLIC_COMPANY,
    )

    claim_schema = RESEARCH_PUBLIC_COMPANY.output_schema["properties"]["claims"][
        "items"
    ]
    for field in ("source", "as_of", "scope"):
        assert field in claim_schema["required"]


def test_research_descriptor_is_consent_gated_and_regime_blocked() -> None:
    from lifeform_domain_growth_advisor.research_affordances import (
        CONSENT_PUBLIC_RESEARCH,
        RESEARCH_PUBLIC_COMPANY,
    )

    safety = RESEARCH_PUBLIC_COMPANY.safety_model
    assert CONSENT_PUBLIC_RESEARCH in safety.requires_consent_grant
    assert "emotional_support" in safety.blocked_in_regimes
    assert "repair_and_deescalation" in safety.blocked_in_regimes
    # Read-only, but it reaches outside the conversation about a subject
    # adjacent to the user's private life.
    assert safety.audit_required is True


def test_research_descriptor_takes_a_company_not_a_person() -> None:
    """The privacy boundary, enforced by the parameter surface.

    A "help me decide about my marriage" conversation invites looking up
    a specific private individual. The tool has no parameter for one.
    """
    from lifeform_domain_growth_advisor.research_affordances import (
        RESEARCH_PUBLIC_COMPANY,
    )

    properties = RESEARCH_PUBLIC_COMPANY.parameters_schema["properties"]
    assert set(properties) == {"company", "questions"}
    assert RESEARCH_PUBLIC_COMPANY.parameters_schema["additionalProperties"] is False
    assert "private individual" in RESEARCH_PUBLIC_COMPANY.when_not_to_use


def test_research_descriptor_reports_unanswered_questions() -> None:
    """An unanswerable question must not vanish.

    Dropping it would quietly shrink the unknown set, which is the one
    thing that keeps the system honest about what it does not know.
    """
    from lifeform_domain_growth_advisor.research_affordances import (
        RESEARCH_PUBLIC_COMPANY,
    )

    assert "unanswered" in RESEARCH_PUBLIC_COMPANY.output_schema["required"]


# ---------------------------------------------------------------------------
# Invoker extraction
# ---------------------------------------------------------------------------


def test_invoker_extracts_claims_by_payload_shape_not_tool_name() -> None:
    from lifeform_affordance.invoker import (
        PROVENANCE_PAYLOAD_KEY,
        _extract_provenance,
    )

    payload = {
        PROVENANCE_PAYLOAD_KEY: [
            {
                "claim_id": "c1",
                "source": "s",
                "as_of": "2025-01-01",
                "scope": "entity:x",
                "confidence": 0.8,
            }
        ]
    }
    (claim,) = _extract_provenance(payload)
    assert claim.is_complete is True
    assert claim.confidence == pytest.approx(0.8)


def test_invoker_keeps_the_good_claims_when_one_entry_is_malformed() -> None:
    from lifeform_affordance.invoker import (
        PROVENANCE_PAYLOAD_KEY,
        _extract_provenance,
    )

    payload = {
        PROVENANCE_PAYLOAD_KEY: [
            "not-a-mapping",
            {"claim_id": "c2", "source": "s", "as_of": "d", "scope": "x"},
        ]
    }
    claims = _extract_provenance(payload)
    assert [claim.claim_id for claim in claims] == ["c2"]


def test_invoker_marks_missing_fields_rather_than_inventing_them() -> None:
    from lifeform_affordance.invoker import (
        PROVENANCE_PAYLOAD_KEY,
        _extract_provenance,
    )

    payload = {PROVENANCE_PAYLOAD_KEY: [{"claim_id": "c3", "source": "s"}]}
    (claim,) = _extract_provenance(payload)
    assert claim.is_complete is False
    assert set(claim.missing_fields()) == {"as_of", "scope"}


def test_payload_without_claims_yields_no_provenance() -> None:
    from lifeform_affordance.invoker import _extract_provenance

    assert _extract_provenance({"content": "file body"}) == ()
    assert _extract_provenance(None) == ()
