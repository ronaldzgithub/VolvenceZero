from __future__ import annotations

from copy import deepcopy

import pytest

from dlaas_platform_contracts import InteractionEnvelope, InteractionType
from volvence_zero.cognition_task import CognitionTaskKind
from volvence_zero.environment import EnvironmentEventKind
from volvence_zero.expression_output import ExpressionBindingSource


def _payload() -> dict[str, object]:
    digest = "a" * 64
    return {
        "contract_id": "ctr-world-action",
        "session_id": "lifeform-qiao",
        "end_user_ref": "player-1",
        "interaction_type": "cognitive_turn",
        "mode": "live",
        "human_brief": "",
        "structured_context": {},
        "perceived_event": {
            "action_id": "action-17",
            "perception": "我看见旅人割断系船绳。",
            "frame": {
                "actor": {
                    "actor_id": "player-1",
                    "actor_kind": "player_character",
                    "display_name": "旅人",
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
                "additionalProperties": False,
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
            "template_id": "nwtpl_scene_v1_qiao",
            "template_uri": f"novel-worlds/blobs/{digest}.json",
            "template_bundle_sha256": digest,
            "template_source_sha256": "b" * 64,
        },
    }


def test_cognitive_turn_round_trip_preserves_all_first_class_fields() -> None:
    payload = _payload()
    envelope = InteractionEnvelope.from_json(payload)

    assert envelope.interaction_type is InteractionType.COGNITIVE_TURN
    assert envelope.perceived_event is not None
    assert envelope.perceived_event.action_id == "action-17"
    assert envelope.perceived_event.perception == "我看见旅人割断系船绳。"
    assert envelope.cognition_task is not None
    assert envelope.cognition_task.kind is CognitionTaskKind.CHOOSE_OBSERVABLE_ACTION
    assert envelope.expression_contract is not None
    assert (
        envelope.expression_contract.exact_bindings[0].source
        is ExpressionBindingSource.RESPONSE_ACTION_REALIZATION_ACTION_STATEMENT
    )
    assert envelope.template_binding is not None
    assert envelope.template_binding.template_bundle_sha256 == "a" * 64

    reparsed = InteractionEnvelope.from_json(envelope.to_json())
    assert reparsed == envelope
    assert reparsed.to_json() == envelope.to_json()


@pytest.mark.parametrize(
    ("field", "detail"),
    (
        ("perceived_event", "requires perceived_event"),
        ("cognition_task", "requires perceived_event"),
        ("expression_contract", "requires perceived_event"),
    ),
)
def test_cognitive_turn_requires_all_three_native_contracts(
    field: str, detail: str
) -> None:
    payload = _payload()
    del payload[field]
    with pytest.raises(ValueError, match=detail):
        InteractionEnvelope.from_json(payload)


def test_cognitive_turn_rejects_external_event_owner_fields() -> None:
    payload = _payload()
    perceived = payload["perceived_event"]
    assert isinstance(perceived, dict)
    perceived["scene_id"] = "client-injected-scene"
    perceived["timestamp_ms"] = 999999
    perceived["event_id"] = "client-event"

    with pytest.raises(ValueError, match="unknown=.*event_id.*scene_id.*timestamp_ms"):
        InteractionEnvelope.from_json(payload)


@pytest.mark.parametrize(
    ("path", "value", "detail"),
    (
        (("cognition_task", "kind"), "invent_action", "kind must be one of"),
        (
            ("expression_contract", "exact_bindings", 0, "source"),
            "prompt.authored.action",
            "source must be one of",
        ),
        (("mode",), "story", "mode must be one of"),
    ),
)
def test_cognitive_turn_rejects_unknown_kind_source_and_mode(
    path: tuple[object, ...], value: object, detail: str
) -> None:
    payload = deepcopy(_payload())
    target: object = payload
    for part in path[:-1]:
        target = target[part]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]

    with pytest.raises(ValueError, match=detail):
        InteractionEnvelope.from_json(payload)


def test_cognitive_turn_rejects_meta_prompt_transport_fields() -> None:
    for field, value in (
        ("human_brief", "Decide what action the character should take"),
        ("structured_context", {"task_prompt": "choose an action"}),
    ):
        payload = _payload()
        payload[field] = value
        with pytest.raises(ValueError, match=field):
            InteractionEnvelope.from_json(payload)


def test_legacy_interaction_remains_unchanged() -> None:
    envelope = InteractionEnvelope.from_json(
        {
            "contract_id": "legacy",
            "session_id": "session",
            "end_user_ref": "user",
            "interaction_type": "chat",
            "human_brief": "hello",
        }
    )
    assert envelope.interaction_type is InteractionType.CHAT
    assert envelope.perceived_event is None
    assert envelope.cognition_task is None
    assert envelope.expression_contract is None
    assert envelope.template_binding is None


def test_scene_event_is_canonical_environment_kind() -> None:
    assert EnvironmentEventKind.SCENE_EVENT.value == "scene_event"
