"""DLaaS debug and analysis contracts.

These contracts describe app-owned debug facts that DLaaS may store and
analyze. They intentionally stay at the platform boundary: apps publish
typed facts and field meanings, while DLaaS governs access, redaction,
querying, and analysis artifacts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DebugFieldType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    JSON = "json"
    ENUM = "enum"


class DebugFieldOwner(str, Enum):
    APP = "app"
    DLAAS = "dlaas"
    USER = "user"
    EXTERNAL_SYSTEM = "external_system"


class DebugPrivacyLevel(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    SENSITIVE = "sensitive"
    SECRET = "secret"


@dataclass(frozen=True)
class DebugFieldDefinition:
    name: str
    type: DebugFieldType
    meaning: str
    owner: DebugFieldOwner = DebugFieldOwner.APP
    privacy_level: DebugPrivacyLevel = DebugPrivacyLevel.INTERNAL
    required: bool = False
    enum_values: tuple[str, ...] = ()
    example: Any = None

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "DebugFieldDefinition":
        name = str(data.get("name", "") or "").strip()
        meaning = str(data.get("meaning", "") or "").strip()
        if not name:
            raise ValueError("debug field name is required")
        if not meaning:
            raise ValueError(f"debug field {name!r} meaning is required")
        field_type = DebugFieldType(str(data.get("type", "") or "string"))
        enum_values = tuple(str(v) for v in data.get("enum_values", ()) or ())
        if field_type is DebugFieldType.ENUM and not enum_values:
            raise ValueError(f"debug field {name!r} enum_values are required")
        return cls(
            name=name,
            type=field_type,
            meaning=meaning,
            owner=DebugFieldOwner(str(data.get("owner", "") or "app")),
            privacy_level=DebugPrivacyLevel(
                str(data.get("privacy_level", "") or "internal")
            ),
            required=bool(data.get("required", False)),
            enum_values=enum_values,
            example=data.get("example"),
        )

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "type": self.type.value,
            "meaning": self.meaning,
            "owner": self.owner.value,
            "privacy_level": self.privacy_level.value,
            "required": self.required,
        }
        if self.enum_values:
            payload["enum_values"] = list(self.enum_values)
        if self.example is not None:
            payload["example"] = self.example
        return payload


@dataclass(frozen=True)
class DebugAppRegistration:
    app_id: str
    display_name: str
    tenant_id: str = ""
    allowed_ai_ids: tuple[str, ...] = ()
    allowed_event_types: tuple[str, ...] = ()
    default_retention_days: int = 30
    created_at_ms: int = 0

    @classmethod
    def from_json(
        cls,
        data: Mapping[str, Any],
        *,
        created_at_ms: int,
    ) -> "DebugAppRegistration":
        app_id = str(data.get("app_id", "") or "").strip()
        display_name = str(data.get("display_name", "") or app_id).strip()
        if not app_id:
            raise ValueError("app_id is required")
        if not display_name:
            raise ValueError("display_name is required")
        return cls(
            app_id=app_id,
            display_name=display_name,
            tenant_id=str(data.get("tenant_id", "") or ""),
            allowed_ai_ids=tuple(str(v) for v in data.get("allowed_ai_ids", ()) or ()),
            allowed_event_types=tuple(
                str(v) for v in data.get("allowed_event_types", ()) or ()
            ),
            default_retention_days=int(data.get("default_retention_days", 30) or 30),
            created_at_ms=created_at_ms,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "app_id": self.app_id,
            "display_name": self.display_name,
            "tenant_id": self.tenant_id,
            "allowed_ai_ids": list(self.allowed_ai_ids),
            "allowed_event_types": list(self.allowed_event_types),
            "default_retention_days": self.default_retention_days,
            "created_at_ms": self.created_at_ms,
        }


@dataclass(frozen=True)
class DebugSchema:
    app_id: str
    schema_version: str
    fields: tuple[DebugFieldDefinition, ...]
    event_types: tuple[str, ...] = ()
    allow_extra_fields: bool = False
    created_at_ms: int = 0

    @classmethod
    def from_json(
        cls,
        data: Mapping[str, Any],
        *,
        app_id: str,
        created_at_ms: int,
    ) -> "DebugSchema":
        schema_version = str(data.get("schema_version", "") or "").strip()
        if not schema_version:
            raise ValueError("schema_version is required")
        raw_fields = data.get("fields", ())
        if not isinstance(raw_fields, Sequence) or isinstance(raw_fields, (str, bytes)):
            raise ValueError("fields must be a list")
        fields = tuple(DebugFieldDefinition.from_json(item) for item in raw_fields)
        if not fields:
            raise ValueError("at least one debug field is required")
        names = [field.name for field in fields]
        if len(set(names)) != len(names):
            raise ValueError("debug field names must be unique")
        return cls(
            app_id=app_id,
            schema_version=schema_version,
            fields=fields,
            event_types=tuple(str(v) for v in data.get("event_types", ()) or ()),
            allow_extra_fields=bool(data.get("allow_extra_fields", False)),
            created_at_ms=created_at_ms,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "app_id": self.app_id,
            "schema_version": self.schema_version,
            "fields": [field.to_json() for field in self.fields],
            "event_types": list(self.event_types),
            "allow_extra_fields": self.allow_extra_fields,
            "created_at_ms": self.created_at_ms,
        }


@dataclass(frozen=True)
class DebugEventEnvelope:
    debug_event_id: str
    app_id: str
    schema_version: str
    event_type: str
    stage: str
    fields: Mapping[str, Any] = field(default_factory=dict)
    ai_id: str = ""
    tenant_id: str = ""
    session_id: str = ""
    end_user_ref: str = ""
    response_id: str = ""
    interaction_id: str = ""
    occurred_at: str = ""
    created_at_ms: int = 0
    retention_expires_at_ms: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "debug_event_id": self.debug_event_id,
            "app_id": self.app_id,
            "schema_version": self.schema_version,
            "ai_id": self.ai_id,
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "end_user_ref": self.end_user_ref,
            "response_id": self.response_id,
            "interaction_id": self.interaction_id,
            "event_type": self.event_type,
            "stage": self.stage,
            "fields": dict(self.fields),
            "occurred_at": self.occurred_at,
            "created_at_ms": self.created_at_ms,
            "retention_expires_at_ms": self.retention_expires_at_ms,
        }


@dataclass(frozen=True)
class DebugAnalysisRequest:
    prompt: str
    app_id: str = ""
    ai_id: str = ""
    tenant_id: str = ""
    session_id: str = ""
    end_user_ref: str = ""
    event_types: tuple[str, ...] = ()
    include_readouts: bool = True
    include_explain: bool = True
    include_audit: bool = True
    include_snapshots: bool = False

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "DebugAnalysisRequest":
        prompt = str(data.get("prompt", "") or "").strip()
        if not prompt:
            raise ValueError("prompt is required")
        selectors = data.get("selectors", {}) or {}
        if not isinstance(selectors, Mapping):
            raise ValueError("selectors must be an object")
        return cls(
            prompt=prompt,
            app_id=str(selectors.get("app_id", "") or data.get("app_id", "") or ""),
            ai_id=str(selectors.get("ai_id", "") or data.get("ai_id", "") or ""),
            tenant_id=str(
                selectors.get("tenant_id", "") or data.get("tenant_id", "") or ""
            ),
            session_id=str(
                selectors.get("session_id", "") or data.get("session_id", "") or ""
            ),
            end_user_ref=str(
                selectors.get("end_user_ref", "")
                or data.get("end_user_ref", "")
                or ""
            ),
            event_types=tuple(
                str(v)
                for v in (
                    selectors.get("event_types", ())
                    or data.get("event_types", ())
                    or ()
                )
            ),
            include_readouts=bool(data.get("include_readouts", True)),
            include_explain=bool(data.get("include_explain", True)),
            include_audit=bool(data.get("include_audit", True)),
            include_snapshots=bool(data.get("include_snapshots", False)),
        )

    def selectors_json(self) -> dict[str, Any]:
        return {
            "app_id": self.app_id,
            "ai_id": self.ai_id,
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "end_user_ref": self.end_user_ref,
            "event_types": list(self.event_types),
        }


@dataclass(frozen=True)
class DebugVersionSuggestion:
    suggestion_type: str = "debug_version_suggestion"
    issue_area: str = "unknown"
    evidence_refs: tuple[str, ...] = ()
    recommended_owner: str = "operator"
    confidence: float = 0.0
    proposed_next_test: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "suggestion_type": self.suggestion_type,
            "issue_area": self.issue_area,
            "evidence_refs": list(self.evidence_refs),
            "recommended_owner": self.recommended_owner,
            "confidence": self.confidence,
            "proposed_next_test": self.proposed_next_test,
        }


@dataclass(frozen=True)
class DebugAnalysisReport:
    analysis_id: str
    prompt: str
    selectors: Mapping[str, Any]
    evidence: Mapping[str, Any] = field(default_factory=dict)
    recommendations: tuple[str, ...] = ()
    version_suggestions: tuple[Mapping[str, Any], ...] = ()
    analysis_mode: str = "deterministic_fallback"
    analyzer_id: str = ""
    analyzer_version: str = ""
    intent_tags: tuple[str, ...] = ()
    prompt_template: str = ""
    prompt_preview: str = ""
    artifact_id: str = ""
    retention_expires_at_ms: int = 0
    created_at_ms: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "prompt": self.prompt,
            "selectors": dict(self.selectors),
            "evidence": dict(self.evidence),
            "recommendations": list(self.recommendations),
            "version_suggestions": [dict(item) for item in self.version_suggestions],
            "analysis_mode": self.analysis_mode,
            "analyzer_id": self.analyzer_id,
            "analyzer_version": self.analyzer_version,
            "intent_tags": list(self.intent_tags),
            "prompt_template": self.prompt_template,
            "prompt_preview": self.prompt_preview,
            "artifact_id": self.artifact_id,
            "retention_expires_at_ms": self.retention_expires_at_ms,
            "created_at_ms": self.created_at_ms,
        }


__all__ = [
    "DebugAnalysisReport",
    "DebugAnalysisRequest",
    "DebugAppRegistration",
    "DebugEventEnvelope",
    "DebugFieldDefinition",
    "DebugFieldOwner",
    "DebugFieldType",
    "DebugPrivacyLevel",
    "DebugSchema",
    "DebugVersionSuggestion",
]
