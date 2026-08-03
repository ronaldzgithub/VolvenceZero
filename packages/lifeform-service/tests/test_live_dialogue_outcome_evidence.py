from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lifeform_core import TurnTriggerKind
from lifeform_core.types import TurnSummary
from lifeform_service import app as service_app
from lifeform_service.alpha import AlphaIdentityProvider, AlphaServiceConfig
from lifeform_service.live_dialogue_outcome_evidence import (
    build_live_dialogue_outcome_artifact,
    write_live_dialogue_outcome_artifact,
)
from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidence,
    DialogueExternalOutcomeEvidenceSource,
    DialogueExternalOutcomeKind,
)


def _turn_summary() -> TurnSummary:
    return TurnSummary(
        turn_index=1,
        scene_id="scene-secret",
        user_input="private user text",
        response_text="private model text",
        active_regime="repair",
        active_abstract_action="clarify",
        open_loop_count=2,
        commitment_count=1,
        pe_magnitude=0.75,
        elapsed_at_tick=3,
        trigger_kind=TurnTriggerKind.USER_INPUT,
    )


def _outcome() -> DialogueExternalOutcomeEvidence:
    return DialogueExternalOutcomeEvidence(
        evidence_id="external:user_explicit:missed:2:private-evidence-ref",
        turn_index=2,
        kind=DialogueExternalOutcomeKind.MISSED,
        source=DialogueExternalOutcomeEvidenceSource.USER_EXPLICIT,
        confidence=0.95,
        evidence_ref="private-evidence-ref",
        description="free-form private feedback",
        session_scope="session-private",
        action_turn_index=1,
    )


def test_artifact_is_typed_deidentified_and_content_verified(tmp_path: Path) -> None:
    artifact = build_live_dialogue_outcome_artifact(
        subject_scope="tenant:alice",
        session_id="session-private",
        evidence=_outcome(),
        turn_summaries=(_turn_summary(),),
        service_version="service-v1",
        policy_version="policy-v1",
        recorded_at_iso="2026-08-03T12:00:00+00:00",
    )

    path = write_live_dialogue_outcome_artifact(
        evidence_root=tmp_path,
        artifact=artifact,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = path.read_text(encoding="utf-8")

    assert payload["schema_version"] == "lifeform-live-dialogue-outcome.v1"
    assert payload["outcome_kind"] == "missed"
    assert payload["action_context"]["active_regime"] == "repair"
    assert payload["action_context"]["prediction_error_magnitude"] == 0.75
    assert len(payload["content_sha256"]) == 64
    for private_value in (
        "tenant:alice",
        "session-private",
        "scene-secret",
        "private user text",
        "private model text",
        "private-evidence-ref",
        "free-form private feedback",
    ):
        assert private_value not in raw


def test_artifact_write_is_idempotent_and_rejects_tampering(tmp_path: Path) -> None:
    artifact = build_live_dialogue_outcome_artifact(
        subject_scope="tenant:alice",
        session_id="session-private",
        evidence=_outcome(),
        turn_summaries=(_turn_summary(),),
        service_version="service-v1",
        policy_version="policy-v1",
        recorded_at_iso="2026-08-03T12:00:00+00:00",
    )
    path = write_live_dialogue_outcome_artifact(
        evidence_root=tmp_path,
        artifact=artifact,
    )
    assert write_live_dialogue_outcome_artifact(
        evidence_root=tmp_path,
        artifact=artifact,
    ) == path

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["confidence"] = 0.1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="content_sha256 validation"):
        write_live_dialogue_outcome_artifact(
            evidence_root=tmp_path,
            artifact=artifact,
        )


class _Request:
    def __init__(self, *, app, payload: dict[str, object]) -> None:
        self.app = app
        self.match_info = {"session_id": "session-private"}
        self.body_exists = True
        self._payload = payload

    async def text(self) -> str:
        return json.dumps(self._payload)


class _Session:
    def __init__(self) -> None:
        self.turn_summaries = (_turn_summary(),)

    def submit_dialogue_outcome(self, **kwargs) -> DialogueExternalOutcomeEvidence:
        return DialogueExternalOutcomeEvidence(
            evidence_id="owner-issued-evidence-id",
            turn_index=kwargs["turn_index"],
            kind=kwargs["kind"],
            source=kwargs["source"],
            confidence=kwargs["confidence"],
            evidence_ref=kwargs["evidence_ref"],
            description=kwargs["description"],
            session_scope="session-private",
            action_turn_index=kwargs["action_turn_index"],
        )


async def test_http_outcome_handler_exports_only_when_alpha_evidence_is_enabled(
    tmp_path: Path,
) -> None:
    session = _Session()
    identity_provider = AlphaIdentityProvider(allowed_users=frozenset({"alice"}))
    identity_provider.bind_session(session_id="session-private", end_user_id="alice")
    request = _Request(
        app={
            "session_manager": SimpleNamespace(
                get_session=lambda _session_id: _async_value(session)
            ),
            "alpha_config": AlphaServiceConfig(
                enabled=True,
                evidence_root_dir=str(tmp_path),
                alpha_users=frozenset({"alice"}),
            ),
            "alpha_provider": identity_provider,
        },
        payload={
            "kind": "missed",
            "confidence": 0.95,
            "description": "private feedback",
        },
    )

    response = await service_app._handle_dialogue_outcome(request)  # type: ignore[arg-type]
    body = json.loads(response.body)

    assert response.status == 201
    artifact_ref = body["evidence_artifact_ref"]
    assert isinstance(artifact_ref, str)
    assert Path(artifact_ref).is_file()
    assert "private feedback" not in Path(artifact_ref).read_text(encoding="utf-8")


async def test_http_outcome_handler_is_noop_without_alpha_evidence(
    tmp_path: Path,
) -> None:
    session = _Session()
    request = _Request(
        app={
            "session_manager": SimpleNamespace(
                get_session=lambda _session_id: _async_value(session)
            ),
            "alpha_config": AlphaServiceConfig(enabled=False),
            "alpha_provider": None,
        },
        payload={"kind": "missed"},
    )

    response = await service_app._handle_dialogue_outcome(request)  # type: ignore[arg-type]
    body = json.loads(response.body)

    assert response.status == 201
    assert body["evidence_artifact_ref"] is None
    assert not (tmp_path / "live_dialogue_outcomes").exists()


async def _async_value(value):
    return value
