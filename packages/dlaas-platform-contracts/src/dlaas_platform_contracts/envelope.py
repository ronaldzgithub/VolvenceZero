"""Typed runtime envelope and OutputAct shapes.

Mirrors the wire format documented in EmoGPT's ``DLAAS_README.md`` so an
external integrator can write against the same JSON shape regardless of
which backend (EmoGPT / VZ) hosts the lifeform.

Design rules carried from `docs/specs/dlaas-platform.md`:

* `interaction_type` is a typed enum. ``dlaas-platform-api`` MUST switch
  on this enum and translate to the right ``LifeformSession.*`` /
  ``BrainSession.submit_*`` call. It MUST NOT inspect ``human_brief`` to
  guess the type (R8 + the no-keyword-matching rule).
* All shapes are frozen dataclasses with explicit ``from_json`` /
  ``to_json``. The platform never reaches into raw HTTP dicts past the
  router boundary.
* ``OutputAct`` is the only structured response shape returned by the
  platform. When the host shell does not declare a capability, the
  platform degrades the act to ``act_type='text'`` and records the
  original capability in ``original_capability``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

DEFAULT_PROTOCOL_VERSION = "dlaas/v1"


class InteractionType(str, Enum):
    """Typed interaction kinds. Source: DLaaS public API §"Runtime Interaction Types"."""

    CHAT = "chat"
    OBSERVE = "observe"
    REPORT = "report"
    FEEDBACK = "feedback"
    TEACH = "teach"
    TASK = "task"
    COMMAND = "command"


class InteractionMode(str, Enum):
    """Interaction mode controls vitals apprentice override + reward semantics."""

    LIVE = "live"
    APPRENTICE = "apprentice"


@dataclass(frozen=True)
class FeedbackPayload:
    """Structured feedback envelope (used when ``interaction_type == FEEDBACK``).

    Mirrors DLaaS feedback object. Only ``valence`` is required by the
    platform; the rest are optional and pass through to the kernel via
    ``LifeformSession.submit_dialogue_outcome``.
    """

    valence: str
    target_response_id: str = ""
    intensity: float = 0.9
    scope: str = "response"
    evidence: str = ""

    @classmethod
    def from_json(cls, data: Mapping[str, Any] | None) -> "FeedbackPayload | None":
        if data is None:
            return None
        valence = data.get("valence")
        if not isinstance(valence, str) or not valence.strip():
            raise ValueError("FeedbackPayload.valence must be a non-empty string")
        intensity_raw = data.get("intensity", 0.9)
        if not isinstance(intensity_raw, int | float):
            raise ValueError("FeedbackPayload.intensity must be numeric")
        return cls(
            valence=valence,
            target_response_id=str(data.get("target_response_id", "") or ""),
            intensity=float(intensity_raw),
            scope=str(data.get("scope", "response") or "response"),
            evidence=str(data.get("evidence", "") or ""),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "valence": self.valence,
            "target_response_id": self.target_response_id,
            "intensity": self.intensity,
            "scope": self.scope,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class OutputContract:
    """Caller-declared output preferences. The platform respects these on a
    best-effort basis; any unsupported combination silently degrades."""

    delivery_channel: str = "dlaas"
    format: str = "text"
    stream: bool = False

    @classmethod
    def from_json(cls, data: Mapping[str, Any] | None) -> "OutputContract":
        if data is None:
            return cls()
        return cls(
            delivery_channel=str(data.get("delivery_channel", "dlaas") or "dlaas"),
            format=str(data.get("format", "text") or "text"),
            stream=bool(data.get("stream", False)),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "delivery_channel": self.delivery_channel,
            "format": self.format,
            "stream": self.stream,
        }


@dataclass(frozen=True)
class InteractionEnvelope:
    """Top-level typed interaction envelope.

    Required: ``contract_id`` / ``session_id`` / ``end_user_ref`` /
    ``interaction_type``. Everything else has a typed default so the
    platform layer never needs ``getattr(..., default)`` defensive lookups.
    """

    contract_id: str
    session_id: str
    end_user_ref: str
    interaction_type: InteractionType
    mode: InteractionMode = InteractionMode.LIVE
    protocol_version: str = DEFAULT_PROTOCOL_VERSION
    human_brief: str = ""
    structured_context: Mapping[str, Any] = field(default_factory=dict)
    output_contract: OutputContract = field(default_factory=OutputContract)
    feedback: FeedbackPayload | None = None
    target_person_ids: tuple[str, ...] = ()
    lang: str = "cn"

    def __post_init__(self) -> None:
        if not self.contract_id.strip():
            raise ValueError("InteractionEnvelope.contract_id must be non-empty")
        if not self.session_id.strip():
            raise ValueError("InteractionEnvelope.session_id must be non-empty")
        if not self.end_user_ref.strip():
            raise ValueError("InteractionEnvelope.end_user_ref must be non-empty")
        # InteractionType / InteractionMode validation already enforced by Enum.
        # Ensure structured_context is a Mapping (frozen / read-only friendly).
        if not isinstance(self.structured_context, Mapping):
            raise ValueError(
                "InteractionEnvelope.structured_context must be a Mapping"
            )
        if not isinstance(self.target_person_ids, tuple):
            raise ValueError(
                "InteractionEnvelope.target_person_ids must be a tuple of str"
            )
        for pid in self.target_person_ids:
            if not isinstance(pid, str) or not pid.strip():
                raise ValueError(
                    "InteractionEnvelope.target_person_ids entries must be non-empty str"
                )

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "InteractionEnvelope":
        """Parse a JSON dict into a typed envelope.

        Raises ``ValueError`` on missing required fields, unknown enum
        values, or wrong types. The platform-api handler catches
        ``ValueError`` and returns ``400 invalid_envelope``.
        """
        if not isinstance(data, Mapping):
            raise ValueError("InteractionEnvelope payload must be a JSON object")
        try:
            interaction_type = InteractionType(str(data["interaction_type"]).lower())
        except KeyError as exc:
            raise ValueError("InteractionEnvelope.interaction_type is required") from exc
        except ValueError as exc:
            allowed = ", ".join(t.value for t in InteractionType)
            raise ValueError(
                f"InteractionEnvelope.interaction_type must be one of: {allowed}"
            ) from exc

        mode_raw = str(data.get("mode", "live")).lower()
        try:
            mode = InteractionMode(mode_raw)
        except ValueError as exc:
            allowed = ", ".join(m.value for m in InteractionMode)
            raise ValueError(
                f"InteractionEnvelope.mode must be one of: {allowed}"
            ) from exc

        target_persons_raw = data.get("target_person_ids")
        if target_persons_raw is None:
            structured = data.get("structured_context") or {}
            if isinstance(structured, Mapping):
                target_persons_raw = structured.get("target_person_ids", ())
            else:
                target_persons_raw = ()
        if isinstance(target_persons_raw, str):
            raise ValueError(
                "InteractionEnvelope.target_person_ids must be a list, not a string"
            )
        target_person_ids = tuple(str(p) for p in (target_persons_raw or ()))

        contract_id = str(data.get("contract_id", "") or "")
        session_id = str(data.get("session_id", "") or "")
        end_user_ref = str(data.get("end_user_ref", "") or "")

        return cls(
            contract_id=contract_id,
            session_id=session_id,
            end_user_ref=end_user_ref,
            interaction_type=interaction_type,
            mode=mode,
            protocol_version=str(
                data.get("protocol_version", DEFAULT_PROTOCOL_VERSION)
                or DEFAULT_PROTOCOL_VERSION
            ),
            human_brief=str(data.get("human_brief", "") or ""),
            structured_context=dict(data.get("structured_context") or {}),
            output_contract=OutputContract.from_json(data.get("output_contract")),
            feedback=FeedbackPayload.from_json(data.get("feedback")),
            target_person_ids=target_person_ids,
            lang=str(data.get("lang", "cn") or "cn"),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "session_id": self.session_id,
            "end_user_ref": self.end_user_ref,
            "interaction_type": self.interaction_type.value,
            "mode": self.mode.value,
            "protocol_version": self.protocol_version,
            "human_brief": self.human_brief,
            "structured_context": dict(self.structured_context),
            "output_contract": self.output_contract.to_json(),
            "feedback": self.feedback.to_json() if self.feedback is not None else None,
            "target_person_ids": list(self.target_person_ids),
            "lang": self.lang,
        }


@dataclass(frozen=True)
class OutputAct:
    """Wire-format structured output unit.

    Mirrors DLaaS ``OutputAct``. The platform packages
    ``volvence_zero.agent.response.AgentResponse`` + ``rationale_tags``
    into one or more ``OutputAct`` entries. When the host shell did not
    declare ``capability`` in its embodiment, the platform degrades to
    ``act_type='text'`` and records the original capability for audit.
    """

    act_type: str
    capability: str
    payload: Mapping[str, Any]
    degraded: bool = False
    original_capability: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "act_type": self.act_type,
            "capability": self.capability,
            "payload": dict(self.payload),
            "degraded": self.degraded,
            "original_capability": self.original_capability,
        }
