from __future__ import annotations

from dataclasses import replace

import pytest

from dlaas_platform_contracts import (
    CognitiveTurnRequest,
    InteractionEnvelope,
    OutputContract,
)


def _payload() -> dict[str, object]:
    return {
        "contract_id": "contract-1",
        "session_id": "lifeform-qiao",
        "end_user_ref": "player-1",
        "interaction_type": "cognitive_turn",
        "perceived_event": {
            "action_id": "action-17",
            "perception": "I saw the traveller cut the mooring line.",
            "frame": {
                "actor": {
                    "actor_id": "player-1",
                    "actor_kind": "player_character",
                    "display_name": "Traveller",
                },
                "active_speaker_id": "player-1",
                "addressee_ids": ["qiao"],
                "subject_ids": ["player-1"],
                "audience_ids": ["qiao"],
            },
            "provenance": "novel-worlds:observer:qiao:action-17",
        },
        "cognition_task": {
            "kind": "choose_observable_action",
            "required_readouts": ["response_action_realization"],
        },
        "expression_contract": {
            "schema_name": "lifeform_intent",
            "schema": {
                "type": "object",
                "properties": {"intended_action": {"type": "string"}},
                "required": ["intended_action"],
            },
            "strict": True,
            "exact_bindings": [
                {
                    "json_pointer": "/intended_action",
                    "source": "response_action_realization.action_statement",
                }
            ],
        },
        "template_binding": {
            "template_id": "template-qiao",
            "template_uri": f"novel-worlds/blobs/{'a' * 64}.json",
            "template_bundle_sha256": "a" * 64,
            "template_source_sha256": "b" * 64,
        },
    }


def _request(payload: dict[str, object] | None = None) -> CognitiveTurnRequest:
    return CognitiveTurnRequest.from_envelope(
        ai_id="ai-qiao",
        envelope=InteractionEnvelope.from_json(payload or _payload()),
    )


def test_hash_binds_all_native_cognitive_semantics() -> None:
    request = _request()
    baseline = request.request_sha256

    template = dict(request.template_binding or {})
    template["template_source_sha256"] = "c" * 64
    perceived = dict(request.perceived_event)
    perceived["perception"] = "I heard the rope snap."
    task = dict(request.cognition_task)
    task["required_readouts"] = []
    expression = dict(request.expression_contract)
    expression["schema_name"] = "other_intent"

    mutations = (
        replace(request, template_binding=template),
        replace(request, perceived_event=perceived),
        replace(request, cognition_task=task),
        replace(request, expression_contract=expression),
    )
    assert all(mutated.request_sha256 != baseline for mutated in mutations)


def test_hash_excludes_transport_only_output_contract() -> None:
    envelope = InteractionEnvelope.from_json(_payload())
    baseline = CognitiveTurnRequest.from_envelope(ai_id="ai-qiao", envelope=envelope)
    different_transport = CognitiveTurnRequest.from_envelope(
        ai_id="ai-qiao",
        envelope=replace(
            envelope,
            output_contract=OutputContract(delivery_channel="wechat", format="markdown", stream=True),
        ),
    )
    assert different_transport.request_sha256 == baseline.request_sha256
    assert "output_contract" not in baseline.canonical_payload()


def test_nested_source_mutation_cannot_change_frozen_request_identity() -> None:
    source = {
        "action_id": "action-17",
        "frame": {"audience_ids": ["qiao"]},
        "perception": "I saw it.",
        "provenance": "world:action-17",
    }
    expression_source = {
        "schema_name": "intent",
        "schema": {"required": ["intended_action"]},
        "strict": True,
        "exact_bindings": [],
    }
    request = replace(
        _request(),
        perceived_event=source,
        expression_contract=expression_source,
    )
    digest = request.request_sha256

    source["frame"]["audience_ids"].append("future-observer")
    expression_source["schema"]["required"].append("future_field")

    assert request.request_sha256 == digest
    frame = request.perceived_event["frame"]
    with pytest.raises(TypeError):
        frame["audience_ids"] = ("mutated",)
    assert frame["audience_ids"] == ("qiao",)
    schema = request.expression_contract["schema"]
    assert schema["required"] == ("intended_action",)
