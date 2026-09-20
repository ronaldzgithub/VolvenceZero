from __future__ import annotations

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest

from dlaas_platform_api.app import attach_dlaas_routes
from dlaas_platform_api.dispatch import dispatch_envelope
from dlaas_platform_contracts import InteractionEnvelope
from lifeform_core import TurnTriggerKind
from lifeform_core.types import environment_event_kind_for_trigger
from lifeform_service import (
    ContentAddressedTemplateBinding,
    SessionNotFoundError,
)
from volvence_zero.cognition_task import CognitionTaskKind
from volvence_zero.environment import EnvironmentEventKind


def _payload(*, digest: str = "a" * 64) -> dict[str, object]:
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


class _Response:
    text = '{"intended_action":"乔慎拉住船头，先问旅人缘由。"}'
    rationale_tags = ("owner-action",)


class _Result:
    response = _Response()
    active_regime = "cautious"
    active_abstract_action = "clarify"


class _Session:
    latest_active_snapshots: dict[str, object] = {}

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def run_turn(self, perception: str, **kwargs: object) -> _Result:
        self.calls.append((perception, kwargs))
        return _Result()


class _Manager:
    def __init__(self) -> None:
        self.session: _Session | None = None
        self.binding: ContentAddressedTemplateBinding | None = None
        self.user_id: str | None = None
        self.create_calls: list[ContentAddressedTemplateBinding | None] = []

    async def get_session(self, session_id: str) -> _Session:
        if self.session is None:
            raise SessionNotFoundError(session_id)
        return self.session

    async def create_session(
        self,
        *,
        session_id: str,
        user_id: str | None = None,
        template_binding: ContentAddressedTemplateBinding | None = None,
    ) -> _Session:
        self.create_calls.append(template_binding)
        self.binding = template_binding
        self.user_id = user_id
        self.session = _Session()
        return self.session

    def session_end_user(self, session_id: str) -> str | None:  # noqa: ARG002
        return self.user_id

    def template_binding_for(
        self, session_id: str  # noqa: ARG002
    ) -> ContentAddressedTemplateBinding | None:
        return self.binding


async def _post(manager: _Manager, payload: dict[str, object]):
    app = web.Application()
    app["session_manager"] = manager
    attach_dlaas_routes(app)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            "/dlaas/v1/instances/ai-qiao/interactions", json=payload
        )
        return response.status, await response.json()
    finally:
        await client.close()


async def test_dispatch_binds_scene_event_task_and_expression_exactly() -> None:
    envelope = InteractionEnvelope.from_json(_payload())
    session = _Session()

    body = await dispatch_envelope(
        envelope=envelope,
        session=session,
        ai_id="ai-qiao",
    )

    assert body["output_acts"][0]["payload"]["content"] == _Response.text
    assert body["cognitive_turn"] == {
        "external_action_id": "action-17",
        "event_kind": EnvironmentEventKind.SCENE_EVENT.value,
        "provenance": "novel-worlds:observer:qiao:action-17",
        "cognition_task_kind": CognitionTaskKind.CHOOSE_OBSERVABLE_ACTION.value,
        "required_readouts": ["response_action_realization"],
        "expression_schema_name": "lifeform_intent",
        "exact_bindings": [
            {
                "json_pointer": "/intended_action",
                "source": "response_action_realization.action_statement",
            }
        ],
    }
    assert len(session.calls) == 1
    perception, kwargs = session.calls[0]
    assert perception == "我看见旅人割断系船绳。"
    assert kwargs["trigger_kind"] is TurnTriggerKind.SCENE_EVENT
    assert kwargs["environment_provenance"] == (
        "novel-worlds:observer:qiao:action-17"
    )
    assert kwargs["environment_frame"] == envelope.perceived_event.frame
    assert kwargs["cognition_task_contract"] is envelope.cognition_task
    assert kwargs["expression_output_contract"] is envelope.expression_contract
    assert "environment_event" not in kwargs
    assert (
        environment_event_kind_for_trigger(TurnTriggerKind.SCENE_EVENT)
        is EnvironmentEventKind.SCENE_EVENT
    )


async def test_route_creates_first_session_from_exact_template_binding() -> None:
    manager = _Manager()

    status, body = await _post(manager, _payload())

    assert status == 200
    assert body["status"] == "ok"
    assert manager.create_calls == [
        ContentAddressedTemplateBinding(
            template_id="nwtpl_scene_v1_qiao",
            template_uri=f"novel-worlds/blobs/{'a' * 64}.json",
            template_bundle_sha256="a" * 64,
            template_source_sha256="b" * 64,
        )
    ]
    assert manager.session is not None
    assert len(manager.session.calls) == 1


async def test_route_rejects_unknown_cognition_kind_with_400() -> None:
    manager = _Manager()
    payload = _payload()
    cognition_task = payload["cognition_task"]
    assert isinstance(cognition_task, dict)
    cognition_task["kind"] = "infer-from-human-brief"

    status, body = await _post(manager, payload)

    assert status == 400
    assert body["error"] == "invalid_envelope"
    assert "cognition_task.kind" in body["detail"]
    assert manager.create_calls == []


async def test_route_rejects_sticky_template_identity_change() -> None:
    manager = _Manager()
    first_status, _first_body = await _post(manager, _payload(digest="a" * 64))
    assert first_status == 200

    status, body = await _post(manager, _payload(digest="c" * 64))

    assert status == 409
    assert body["error"] == "session_template_binding_mismatch"
    assert len(manager.create_calls) == 1
    assert manager.session is not None
    assert len(manager.session.calls) == 1


@pytest.mark.parametrize(
    "field",
    ("template_uri", "template_bundle_sha256", "template_source_sha256"),
)
async def test_route_rejects_partial_template_binding(field: str) -> None:
    manager = _Manager()
    payload = _payload()
    binding = payload["template_binding"]
    assert isinstance(binding, dict)
    del binding[field]

    status, body = await _post(manager, payload)

    assert status == 400
    assert body["error"] == "invalid_envelope"
    assert "template_binding field mismatch" in body["detail"]


async def test_route_uses_service_owner_template_validation() -> None:
    manager = _Manager()
    payload = _payload()
    binding = payload["template_binding"]
    assert isinstance(binding, dict)
    binding["template_bundle_sha256"] = "not-a-digest"

    status, body = await _post(manager, payload)

    assert status == 400
    assert body["error"] == "invalid_template_binding"
    assert "64 lowercase hexadecimal" in body["detail"]
    assert manager.create_calls == []
