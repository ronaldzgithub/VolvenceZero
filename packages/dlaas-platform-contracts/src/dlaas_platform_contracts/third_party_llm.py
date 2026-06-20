"""Third-party LLM contracts for DLaaS platform compile-time work.

This surface is deliberately separate from the VZ / lifeform
``/v1/chat/completions`` runtime. It is used for operator/internal
compile-time tasks such as multi-angle bake profile extraction, where a
deployment may call an OpenAI-compatible external provider without
turning that provider into DLaaS cognition or runtime state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ThirdPartyLlmProvider(str, Enum):
    """Provider preset identifiers understood by deployment config."""

    OPENAI = "openai"
    QWEN = "qwen"
    DASHSCOPE = "dashscope"
    VLLM = "vllm"
    CUSTOM = "custom"


class ThirdPartyLlmErrorCode(str, Enum):
    UNCONFIGURED = "third_party_llm_unconfigured"
    INVALID_REQUEST = "third_party_llm_invalid_request"
    UPSTREAM_ERROR = "third_party_llm_upstream_error"
    INVALID_JSON = "third_party_llm_invalid_json"
    SCHEMA_VALIDATION_FAILED = "third_party_llm_schema_validation_failed"


@dataclass(frozen=True)
class ThirdPartyLlmJsonRequest:
    """Schema-enforced JSON-mode request for compile-time LLM work."""

    system_prompt: str
    user_prompt: str
    schema: Mapping[str, Any] = field(default_factory=dict)
    schema_name: str = "third_party_llm_json"
    model: str = ""
    temperature: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "ThirdPartyLlmJsonRequest":
        if not isinstance(data, Mapping):
            raise ValueError("ThirdPartyLlmJsonRequest payload must be an object")
        system_prompt = str(data.get("system_prompt", "") or "")
        user_prompt = str(data.get("user_prompt", "") or "")
        if not system_prompt.strip():
            raise ValueError("system_prompt must be non-empty")
        if not user_prompt.strip():
            raise ValueError("user_prompt must be non-empty")
        schema = data.get("schema") or {}
        if not isinstance(schema, Mapping):
            raise ValueError("schema must be an object")
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be an object")
        raw_temperature = data.get("temperature", 0.0)
        try:
            temperature = float(raw_temperature)
        except (TypeError, ValueError) as exc:
            raise ValueError("temperature must be numeric") from exc
        return cls(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=dict(schema),
            schema_name=str(data.get("schema_name", "third_party_llm_json") or "third_party_llm_json"),
            model=str(data.get("model", "") or ""),
            temperature=max(0.0, min(2.0, temperature)),
            metadata=dict(metadata),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "schema": dict(self.schema),
            "schema_name": self.schema_name,
            "model": self.model,
            "temperature": self.temperature,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ThirdPartyLlmJsonResponse:
    """Structured JSON response plus provider metadata."""

    content: Mapping[str, Any]
    provider: str = ""
    model: str = ""
    response_id: str = ""
    usage: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "content": dict(self.content),
            "provider": self.provider,
            "model": self.model,
            "response_id": self.response_id,
            "usage": dict(self.usage),
        }


@dataclass(frozen=True)
class ThirdPartyLlmStatus:
    provider: str
    base_url: str
    model: str
    api_key_present: bool
    configured: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "api_key_present": self.api_key_present,
            "configured": self.configured,
        }


@dataclass(frozen=True)
class ThirdPartyLlmError:
    code: ThirdPartyLlmErrorCode
    detail: str

    def to_json(self) -> dict[str, Any]:
        return {"code": self.code.value, "detail": self.detail}


__all__ = [
    "ThirdPartyLlmError",
    "ThirdPartyLlmErrorCode",
    "ThirdPartyLlmJsonRequest",
    "ThirdPartyLlmJsonResponse",
    "ThirdPartyLlmProvider",
    "ThirdPartyLlmStatus",
]
