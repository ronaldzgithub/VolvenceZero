"""Wire contracts for one native, owner-bounded cognitive turn.

The caller supplies only what an external world adapter owns: the stable
action reference, one character's perception, its social frame, and
provenance.  Event identity, scene identity, and time deliberately do not
appear on this wire type; the Lifeform owner creates those values when the
turn starts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from typing import Any

from volvence_zero.cognition_task import (
    CognitionRequiredReadout,
    CognitionTaskContract,
    CognitionTaskKind,
)
from volvence_zero.environment import EnvironmentActorRef, EnvironmentFrame
from volvence_zero.expression_output import (
    ExpressionBindingSource,
    ExpressionExactBinding,
    ExpressionOutputContract,
)


@dataclass(frozen=True)
class CognitiveTurnPerceivedEvent:
    """External portion of a scene event, before Lifeform ownership fields."""

    action_id: str
    perception: str
    frame: EnvironmentFrame
    provenance: str

    def __post_init__(self) -> None:
        _require_non_empty("perceived_event.action_id", self.action_id)
        _require_non_empty("perceived_event.perception", self.perception)
        if not isinstance(self.frame, EnvironmentFrame):
            raise TypeError("perceived_event.frame must be an EnvironmentFrame")
        _require_non_empty("perceived_event.provenance", self.provenance)

    @classmethod
    def from_json(cls, data: object) -> "CognitiveTurnPerceivedEvent":
        payload = _require_object("perceived_event", data)
        _require_exact_fields(
            "perceived_event",
            payload,
            {"action_id", "perception", "frame", "provenance"},
        )
        return cls(
            action_id=_require_string("perceived_event.action_id", payload["action_id"]),
            perception=_require_string(
                "perceived_event.perception", payload["perception"]
            ),
            frame=_parse_environment_frame(payload["frame"]),
            provenance=_require_string(
                "perceived_event.provenance", payload["provenance"]
            ),
        )

    def to_json(self) -> dict[str, Any]:
        actor = self.frame.actor
        return {
            "action_id": self.action_id,
            "perception": self.perception,
            "frame": {
                "actor": {
                    "actor_id": actor.actor_id,
                    "actor_kind": actor.actor_kind,
                    "display_name": actor.display_name,
                },
                "active_speaker_id": self.frame.active_speaker_id,
                "addressee_ids": list(self.frame.addressee_ids),
                "subject_ids": list(self.frame.subject_ids),
                "audience_ids": list(self.frame.audience_ids),
            },
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class CognitiveTurnTemplateBinding:
    """Wire attestation for one immutable baked character template.

    Semantic validation (URI namespace, digest format and equality) remains
    owned by ``lifeform-service.ContentAddressedTemplateBinding``.  This
    platform contract only enforces an all-or-nothing four-field transport.
    """

    template_id: str
    template_uri: str
    template_bundle_sha256: str
    template_source_sha256: str

    def __post_init__(self) -> None:
        for name, value in (
            ("template_id", self.template_id),
            ("template_uri", self.template_uri),
            ("template_bundle_sha256", self.template_bundle_sha256),
            ("template_source_sha256", self.template_source_sha256),
        ):
            _require_non_empty(f"template_binding.{name}", value)

    @classmethod
    def from_json(cls, data: object) -> "CognitiveTurnTemplateBinding":
        payload = _require_object("template_binding", data)
        fields = {
            "template_id",
            "template_uri",
            "template_bundle_sha256",
            "template_source_sha256",
        }
        _require_exact_fields("template_binding", payload, fields)
        return cls(
            template_id=_require_string(
                "template_binding.template_id", payload["template_id"]
            ),
            template_uri=_require_string(
                "template_binding.template_uri", payload["template_uri"]
            ),
            template_bundle_sha256=_require_string(
                "template_binding.template_bundle_sha256",
                payload["template_bundle_sha256"],
            ),
            template_source_sha256=_require_string(
                "template_binding.template_source_sha256",
                payload["template_source_sha256"],
            ),
        )

    def to_json(self) -> dict[str, str]:
        return {
            "template_id": self.template_id,
            "template_uri": self.template_uri,
            "template_bundle_sha256": self.template_bundle_sha256,
            "template_source_sha256": self.template_source_sha256,
        }


def cognition_task_from_json(data: object) -> CognitionTaskContract:
    payload = _require_object("cognition_task", data)
    _require_exact_fields(
        "cognition_task", payload, {"kind", "required_readouts"}
    )
    kind_raw = _require_string("cognition_task.kind", payload["kind"])
    try:
        kind = CognitionTaskKind(kind_raw)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in CognitionTaskKind)
        raise ValueError(f"cognition_task.kind must be one of: {allowed}") from exc

    readouts_raw = _require_array(
        "cognition_task.required_readouts", payload["required_readouts"]
    )
    readouts: list[CognitionRequiredReadout] = []
    for index, raw in enumerate(readouts_raw):
        value = _require_string(
            f"cognition_task.required_readouts[{index}]", raw
        )
        try:
            readouts.append(CognitionRequiredReadout(value))
        except ValueError as exc:
            allowed = ", ".join(item.value for item in CognitionRequiredReadout)
            raise ValueError(
                "cognition_task.required_readouts entries must be one of: "
                f"{allowed}"
            ) from exc
    try:
        return CognitionTaskContract(kind=kind, required_readouts=tuple(readouts))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid cognition_task: {exc}") from exc


def cognition_task_to_json(contract: CognitionTaskContract) -> dict[str, Any]:
    return {
        "kind": contract.kind.value,
        "required_readouts": [item.value for item in contract.required_readouts],
    }


def expression_contract_from_json(data: object) -> ExpressionOutputContract:
    payload = _require_object("expression_contract", data)
    _require_exact_fields(
        "expression_contract",
        payload,
        {"schema_name", "schema", "strict", "exact_bindings"},
    )
    schema = _require_object("expression_contract.schema", payload["schema"])
    strict = payload["strict"]
    if type(strict) is not bool:
        raise ValueError("expression_contract.strict must be a boolean")
    raw_bindings = _require_array(
        "expression_contract.exact_bindings", payload["exact_bindings"]
    )
    bindings: list[ExpressionExactBinding] = []
    for index, raw in enumerate(raw_bindings):
        binding = _require_object(
            f"expression_contract.exact_bindings[{index}]", raw
        )
        _require_exact_fields(
            f"expression_contract.exact_bindings[{index}]",
            binding,
            {"json_pointer", "source"},
        )
        source_raw = _require_string(
            f"expression_contract.exact_bindings[{index}].source",
            binding["source"],
        )
        try:
            source = ExpressionBindingSource(source_raw)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in ExpressionBindingSource)
            raise ValueError(
                "expression_contract.exact_bindings source must be one of: "
                f"{allowed}"
            ) from exc
        try:
            bindings.append(
                ExpressionExactBinding(
                    json_pointer=_require_string(
                        f"expression_contract.exact_bindings[{index}].json_pointer",
                        binding["json_pointer"],
                    ),
                    source=source,
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid expression_contract.exact_bindings[{index}]: {exc}"
            ) from exc

    try:
        schema_json = json.dumps(
            dict(schema),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return ExpressionOutputContract(
            schema_name=_require_string(
                "expression_contract.schema_name", payload["schema_name"]
            ),
            schema_json=schema_json,
            strict=strict,
            exact_bindings=tuple(bindings),
        )
    except (RecursionError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid expression_contract: {exc}") from exc


def expression_contract_to_json(
    contract: ExpressionOutputContract,
) -> dict[str, Any]:
    return {
        "schema_name": contract.schema_name,
        "schema": contract.schema,
        "strict": contract.strict,
        "exact_bindings": [
            {
                "json_pointer": binding.json_pointer,
                "source": binding.source.value,
            }
            for binding in contract.exact_bindings
        ],
    }


def _parse_environment_frame(data: object) -> EnvironmentFrame:
    payload = _require_object("perceived_event.frame", data)
    _require_exact_fields(
        "perceived_event.frame",
        payload,
        {
            "actor",
            "active_speaker_id",
            "addressee_ids",
            "subject_ids",
            "audience_ids",
        },
    )
    actor_raw = _require_object("perceived_event.frame.actor", payload["actor"])
    _require_exact_fields(
        "perceived_event.frame.actor",
        actor_raw,
        {"actor_id", "actor_kind", "display_name"},
    )
    display_name = actor_raw["display_name"]
    if display_name is not None:
        display_name = _require_string(
            "perceived_event.frame.actor.display_name", display_name
        )
    try:
        return EnvironmentFrame(
            actor=EnvironmentActorRef(
                actor_id=_require_string(
                    "perceived_event.frame.actor.actor_id", actor_raw["actor_id"]
                ),
                actor_kind=_require_string(
                    "perceived_event.frame.actor.actor_kind",
                    actor_raw["actor_kind"],
                ),
                display_name=display_name,
            ),
            active_speaker_id=_require_string(
                "perceived_event.frame.active_speaker_id",
                payload["active_speaker_id"],
            ),
            addressee_ids=_require_string_tuple(
                "perceived_event.frame.addressee_ids", payload["addressee_ids"]
            ),
            subject_ids=_require_string_tuple(
                "perceived_event.frame.subject_ids", payload["subject_ids"]
            ),
            audience_ids=_require_string_tuple(
                "perceived_event.frame.audience_ids", payload["audience_ids"]
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid perceived_event.frame: {exc}") from exc


def _require_object(field: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} keys must be strings")
    return value


def _require_exact_fields(
    field: str,
    value: Mapping[str, Any],
    expected: set[str],
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"{field} field mismatch: missing={missing}, unknown={unknown}"
        )


def _require_string(field: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_non_empty(field: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def _require_array(field: str, value: object) -> Sequence[object]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be an array")
    return value


def _require_string_tuple(field: str, value: object) -> tuple[str, ...]:
    items = _require_array(field, value)
    return tuple(
        _require_string(f"{field}[{index}]", item)
        for index, item in enumerate(items)
    )


__all__ = [
    "CognitiveTurnPerceivedEvent",
    "CognitiveTurnTemplateBinding",
    "cognition_task_from_json",
    "cognition_task_to_json",
    "expression_contract_from_json",
    "expression_contract_to_json",
]
