from __future__ import annotations

from dataclasses import replace

from dlaas_platform_contracts import (
    FeedbackPayload,
    InteractionEnvelope,
    InteractionType,
    OutputContract,
    ReportSceneEndRequest,
)


def _envelope() -> InteractionEnvelope:
    return InteractionEnvelope(
        contract_id="contract-1",
        session_id="session-1",
        end_user_ref="player-1",
        interaction_type=InteractionType.REPORT,
        protocol_version="dlaas.v1",
        human_brief="close changed scene",
        structured_context={"scene_id": "gate", "event_count": 3},
        feedback=FeedbackPayload(valence="positive", evidence="world settled"),
        target_person_ids=("qiao", "alan"),
        lang="zh-CN",
    )


def test_canonical_hash_binds_semantics_but_not_stream_or_internal_transport() -> None:
    envelope = _envelope()
    baseline = ReportSceneEndRequest.from_envelope(ai_id="ai-1", envelope=envelope)
    assert baseline.request_sha256 == ReportSceneEndRequest.from_envelope(
        ai_id="ai-1",
        envelope=replace(
            envelope,
            output_contract=OutputContract(stream=True),
        ),
    ).request_sha256
    semantic_variants = (
        ("ai-2", envelope),
        ("ai-1", replace(envelope, contract_id="contract-2")),
        ("ai-1", replace(envelope, session_id="session-2")),
        ("ai-1", replace(envelope, end_user_ref="player-2")),
        ("ai-1", replace(envelope, structured_context={"scene_id": "dock"})),
        ("ai-1", replace(envelope, protocol_version="dlaas.v2")),
        ("ai-1", replace(envelope, human_brief="different close")),
        ("ai-1", replace(envelope, target_person_ids=("qiao",))),
        ("ai-1", replace(envelope, lang="en")),
        ("ai-1", replace(envelope, feedback=None)),
    )
    for ai_id, variant in semantic_variants:
        assert ReportSceneEndRequest.from_envelope(
            ai_id=ai_id, envelope=variant
        ).request_sha256 != baseline.request_sha256


def test_public_envelope_ignores_internal_persistence_control_field() -> None:
    parsed = InteractionEnvelope.from_json(
        {**_envelope().to_json(), "report_scene_end_persistence_required": True}
    )
    assert "report_scene_end_persistence_required" not in parsed.to_json()
