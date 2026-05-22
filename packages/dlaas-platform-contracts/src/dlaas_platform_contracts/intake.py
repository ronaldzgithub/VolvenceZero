"""DLaaS asset intake contracts.

These contracts describe platform-level file/book/image intake decisions.
They do not own cognitive state; execution routes through platform asset
metadata, lifeform-ingestion, or training jobs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AssetIntakeIntent(str, Enum):
    STORAGE_ONLY = "storage_only"
    SIMPLE_INGEST = "simple_ingest"
    DEEP_READ = "deep_read"
    TRAINING_CANDIDATE = "training_candidate"
    IMAGE_INTAKE = "image_intake"
    AUTO = "auto"


class AssetMediaKind(str, Enum):
    TEXT = "text"
    MARKDOWN = "markdown"
    PDF = "pdf"
    DOCX = "docx"
    IMAGE = "image"
    JSON = "json"
    BINARY = "binary"


@dataclass(frozen=True)
class AssetIntakeRequest:
    ai_id: str
    contract_id: str
    session_id: str
    end_user_ref: str
    intent: AssetIntakeIntent
    media_kind: AssetMediaKind
    title: str = ""
    source_ref: str = ""
    mime_type: str = ""
    text: str = ""
    content_base64: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(
        cls,
        data: Mapping[str, Any],
        *,
        ai_id: str,
    ) -> "AssetIntakeRequest":
        if not isinstance(data, Mapping):
            raise ValueError("AssetIntakeRequest payload must be a JSON object")
        contract_id = str(data.get("contract_id", "") or "")
        if not contract_id.strip():
            raise ValueError("AssetIntakeRequest.contract_id must be non-empty")
        try:
            intent = AssetIntakeIntent(str(data.get("intake_intent", "auto") or "auto"))
        except ValueError as exc:
            allowed = ", ".join(intent.value for intent in AssetIntakeIntent)
            raise ValueError(
                f"AssetIntakeRequest.intake_intent must be one of: {allowed}"
            ) from exc
        try:
            media_kind = AssetMediaKind(str(data.get("media_kind", "") or ""))
        except ValueError as exc:
            allowed = ", ".join(kind.value for kind in AssetMediaKind)
            raise ValueError(
                f"AssetIntakeRequest.media_kind must be one of: {allowed}"
            ) from exc
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise ValueError("AssetIntakeRequest.metadata must be an object")
        return cls(
            ai_id=ai_id,
            contract_id=contract_id,
            session_id=str(data.get("session_id", f"intake-{ai_id}") or f"intake-{ai_id}"),
            end_user_ref=str(data.get("end_user_ref", "intake") or "intake"),
            intent=intent,
            media_kind=media_kind,
            title=str(data.get("title", "") or ""),
            source_ref=str(data.get("source_ref", "") or ""),
            mime_type=str(data.get("mime_type", "") or ""),
            text=str(data.get("text", "") or ""),
            content_base64=str(data.get("content_base64", "") or ""),
            metadata=dict(metadata),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "ai_id": self.ai_id,
            "contract_id": self.contract_id,
            "session_id": self.session_id,
            "end_user_ref": self.end_user_ref,
            "intake_intent": self.intent.value,
            "media_kind": self.media_kind.value,
            "title": self.title,
            "source_ref": self.source_ref,
            "mime_type": self.mime_type,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AssetIntakeDecision:
    intent: AssetIntakeIntent
    rationale: str = ""

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "AssetIntakeDecision":
        if not isinstance(data, Mapping):
            raise ValueError("AssetIntakeDecision payload must be a JSON object")
        try:
            intent = AssetIntakeIntent(str(data["intent"]))
        except (KeyError, ValueError) as exc:
            allowed = ", ".join(item.value for item in AssetIntakeIntent if item is not AssetIntakeIntent.AUTO)
            raise ValueError(f"AssetIntakeDecision.intent must be one of: {allowed}") from exc
        if intent is AssetIntakeIntent.AUTO:
            raise ValueError("AssetIntakeDecision.intent cannot be auto")
        return cls(intent=intent, rationale=str(data.get("rationale", "") or ""))


__all__ = [
    "AssetIntakeDecision",
    "AssetIntakeIntent",
    "AssetIntakeRequest",
    "AssetMediaKind",
]
