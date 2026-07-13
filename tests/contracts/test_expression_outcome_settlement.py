"""CP-13 expression settlement: external feedback settles the prior prediction.

The single legal channel for external dialogue outcomes
(``submit_dialogue_outcome``) must land on the SAME PE snapshot that
evaluates the previously published (owner-issued, CP-10) prediction — so an
expression outcome is traceable to the exact prediction it settled:

    turn N   PE publishes next_prediction with owner-issued prediction_id
    (user reacts; host submits typed external outcome evidence)
    turn N+1 PE evaluates that prediction (evaluated_prediction.prediction_id
             == turn-N id) AND actual_outcome.external_outcome_refs carries
             the evidence id that biased the realized outcome.
"""

from __future__ import annotations

from volvence_zero.agent.session import AgentSessionRunner
from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidenceSource,
    DialogueExternalOutcomeKind,
)


async def test_external_feedback_settles_prior_owner_issued_prediction() -> None:
    runner = AgentSessionRunner(rare_heavy_enabled=False)

    first = await runner.run_turn("I keep postponing the harbor paperwork.")
    issued = first.active_snapshots["prediction_error"].value.next_prediction
    assert issued.prediction_id, "PE owner must issue the pre-action prediction id"

    evidence = runner.submit_dialogue_outcome(
        kind=DialogueExternalOutcomeKind.FELT_HEARD,
        source=DialogueExternalOutcomeEvidenceSource.USER_EXPLICIT,
        confidence=0.9,
        description="User explicitly said they felt heard.",
    )

    second = await runner.run_turn("Thanks, that actually helped me start.")
    pe_value = second.active_snapshots["prediction_error"].value

    # (a) The settled prediction is exactly the one the owner issued at turn N.
    assert pe_value.evaluated_prediction is not None
    assert pe_value.evaluated_prediction.prediction_id == issued.prediction_id
    # (b) The realized outcome carries the external evidence lineage.
    assert evidence.evidence_id in pe_value.actual_outcome.external_outcome_refs
    # (c) The settlement is auditable from the snapshot alone (no raw text).
    assert "external-outcome bias applied" in pe_value.actual_outcome.description
