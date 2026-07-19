"""Third-party LLM plane for compile-time DLaaS work.

This is NOT the VZ / lifeform runtime chat path. It exists so DLaaS
platform jobs (multi-angle bake, future protocol/document compile jobs)
can call an operator-configured OpenAI-compatible provider without
pretending that provider owns cognition, memory, or runtime sessions.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from aiohttp import ClientTimeout, web

from dlaas_platform_contracts import (
    ThirdPartyLlmErrorCode,
    ThirdPartyLlmJsonRequest,
    ThirdPartyLlmJsonResponse,
    ThirdPartyLlmStatus,
)
from dlaas_platform_registry import (
    REGISTRY_APP_KEY,
    Registry,
    require_control_plane_or_service,
)

THIRD_PARTY_LLM_APP_KEY = "dlaas_third_party_llm"


@dataclass(frozen=True)
class _ProviderPreset:
    base_url: str
    default_model: str
    notes: str = ""


PROVIDER_PRESETS: Mapping[str, _ProviderPreset] = {
    "openrouter": _ProviderPreset(
        base_url="https://openrouter.ai/api/v1",
        default_model="openai/gpt-4o-mini",
        notes="OpenRouter OpenAI-compatible mode",
    ),
    "openai": _ProviderPreset(
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        notes="OpenAI Chat Completions JSON mode",
    ),
    "qwen": _ProviderPreset(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model="qwen-plus",
        notes="Aliyun DashScope OpenAI-compatible mode",
    ),
    "dashscope": _ProviderPreset(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model="qwen-plus",
        notes="Alias for qwen",
    ),
    "vllm": _ProviderPreset(
        base_url="http://localhost:8000/v1",
        default_model="Qwen/Qwen2.5-7B-Instruct",
        notes="Local vLLM OpenAI-compatible server",
    ),
}


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


class ThirdPartyLlmConfig:
    """Deployment-owned OpenAI-compatible provider config."""

    __slots__ = ("provider", "base_url", "api_key", "model", "timeout_seconds")

    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float,
    ) -> None:
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    def status(self) -> ThirdPartyLlmStatus:
        return ThirdPartyLlmStatus(
            provider=self.provider,
            base_url=self.base_url,
            model=self.model,
            api_key_present=bool(self.api_key),
            configured=self.configured,
        )


def build_third_party_llm_config_from_env() -> ThirdPartyLlmConfig:
    """Build provider config from ``THIRD_PARTY_LLM_*`` env vars.

    ``PROTOCOL_LLM_*`` fallback is allowed only when
    ``THIRD_PARTY_LLM_ALLOW_PROTOCOL_FALLBACK=1``. This keeps bake-facing
    LLM credentials explicit while preserving an operator-controlled
    rollback path for environments that already configured protocol uptake.
    """

    allow_protocol_fallback = _env("THIRD_PARTY_LLM_ALLOW_PROTOCOL_FALLBACK") == "1"
    prefix = "THIRD_PARTY_LLM"
    if allow_protocol_fallback and not _env("THIRD_PARTY_LLM_API_KEY"):
        prefix = "PROTOCOL_LLM"

    provider = _env(f"{prefix}_PROVIDER", "openrouter").lower()
    preset = PROVIDER_PRESETS.get(provider)
    base_url = _env(f"{prefix}_BASE_URL")
    if not base_url and preset is not None:
        base_url = preset.base_url
    model = _env(f"{prefix}_MODEL")
    if not model and preset is not None:
        model = preset.default_model
    api_key = _env(f"{prefix}_API_KEY")
    timeout_raw = _env(f"{prefix}_TIMEOUT_SECONDS")
    try:
        timeout_seconds = float(timeout_raw) if timeout_raw else 60.0
    except ValueError:
        timeout_seconds = 60.0
    return ThirdPartyLlmConfig(
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
    )


class ThirdPartyLlmBundle:
    __slots__ = ("config",)

    def __init__(self, *, config: ThirdPartyLlmConfig | None = None) -> None:
        self.config = config or build_third_party_llm_config_from_env()


async def complete_json_with_config(
    *,
    config: ThirdPartyLlmConfig,
    llm_request: ThirdPartyLlmJsonRequest,
) -> ThirdPartyLlmJsonResponse:
    """Programmatic JSON completion helper used by platform jobs.

    Route handlers and in-process runners share this implementation so
    the bake runner does not need to call the platform over HTTP from
    inside the same process. Raises ``RuntimeError`` / ``ValueError`` on
    failures; callers convert those into typed angle or HTTP errors.
    """

    if not config.configured:
        raise RuntimeError("third-party LLM is not configured")
    model = llm_request.model or config.model
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": llm_request.system_prompt},
            {"role": "user", "content": llm_request.user_prompt},
        ],
        "temperature": llm_request.temperature,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": llm_request.schema_name,
                "strict": True,
                "schema": dict(llm_request.schema),
            },
        },
        "metadata": dict(llm_request.metadata),
    }
    upstream = await _post_openai_compat(config, body)
    if int(upstream["status"]) >= 400:
        raise RuntimeError(f"third-party LLM upstream error: {upstream['body']}")
    parsed = _extract_json_content(upstream["body"])
    validation_error = _validate_required_fields(parsed, llm_request.schema)
    if validation_error:
        raise ValueError(validation_error)
    return ThirdPartyLlmJsonResponse(
        content=parsed,
        provider=config.provider,
        model=model,
        response_id=str(upstream["body"].get("id", "") or ""),
        usage=upstream["body"].get("usage") if isinstance(upstream["body"].get("usage"), Mapping) else {},
    )


def attach_third_party_llm_routes(
    app: web.Application,
    *,
    registry: Registry,
    config: ThirdPartyLlmConfig | None = None,
) -> web.Application:
    if REGISTRY_APP_KEY not in app:
        raise ValueError(
            "attach_third_party_llm_routes requires app[REGISTRY_APP_KEY] "
            "(dlaas_platform_api.build_dlaas_app handles this)."
        )
    app[THIRD_PARTY_LLM_APP_KEY] = ThirdPartyLlmBundle(config=config)
    R = app.router
    R.add_get("/dlaas/v1/third-party-llm/status", _handle_status)
    R.add_post(
        "/dlaas/v1/third-party-llm/chat/completions",
        _handle_chat_completions,
    )
    R.add_post("/dlaas/v1/third-party-llm/json", _handle_json)
    return app


async def _handle_status(request: web.Request) -> web.Response:
    require_control_plane_or_service(request)
    return web.json_response({"status": "ok", **_bundle(request).config.status().to_json()})


async def _handle_chat_completions(request: web.Request) -> web.Response:
    require_control_plane_or_service(request)
    data = await _read_json(request)
    cfg = _bundle(request).config
    if not cfg.configured:
        return _error(503, ThirdPartyLlmErrorCode.UNCONFIGURED.value, "third-party LLM is not configured")
    body = dict(data)
    body.setdefault("model", cfg.model)
    upstream = await _post_openai_compat(cfg, body)
    return web.json_response(upstream["body"], status=int(upstream["status"]))


async def _handle_json(request: web.Request) -> web.Response:
    require_control_plane_or_service(request)
    cfg = _bundle(request).config
    if not cfg.configured:
        return _error(503, ThirdPartyLlmErrorCode.UNCONFIGURED.value, "third-party LLM is not configured")
    try:
        llm_request = ThirdPartyLlmJsonRequest.from_json(await _read_json(request))
    except ValueError as exc:
        return _error(400, ThirdPartyLlmErrorCode.INVALID_REQUEST.value, str(exc))
    try:
        response = await complete_json_with_config(
            config=cfg, llm_request=llm_request
        )
    except RuntimeError as exc:
        return _error(502, ThirdPartyLlmErrorCode.UPSTREAM_ERROR.value, str(exc))
    except ValueError as exc:
        return _error(502, ThirdPartyLlmErrorCode.INVALID_JSON.value, str(exc))
    return web.json_response({"status": "ok", **response.to_json()})


async def _post_openai_compat(
    cfg: ThirdPartyLlmConfig, body: Mapping[str, Any]
) -> dict[str, Any]:
    import aiohttp

    async with aiohttp.ClientSession(
        timeout=ClientTimeout(total=cfg.timeout_seconds)
    ) as session:
        async with session.post(
            cfg.base_url.rstrip("/") + "/chat/completions",
            json=dict(body),
            headers={
                "Authorization": f"Bearer {cfg.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        ) as resp:
            text = await resp.text()
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = {"raw": text[:1000]}
            return {"status": resp.status, "body": payload}


def _extract_json_content(envelope: Mapping[str, Any]) -> dict[str, Any]:
    choices = envelope.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("upstream response missing choices")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise ValueError("upstream choice is not an object")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("upstream choice missing message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("upstream message content is empty")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        stripped = _strip_markdown_fence(content)
        if stripped is None:
            raise
        parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("upstream JSON content is not an object")
    return parsed


def _strip_markdown_fence(text: str) -> str | None:
    candidate = text.strip()
    if candidate.startswith("```"):
        first_newline = candidate.find("\n")
        if first_newline == -1:
            return None
        candidate = candidate[first_newline + 1 :]
    if candidate.endswith("```"):
        candidate = candidate[:-3].rstrip()
    return candidate or None


def _validate_required_fields(
    payload: Mapping[str, Any], schema: Mapping[str, Any]
) -> str:
    required = schema.get("required")
    if not isinstance(required, list):
        return ""
    missing = [str(k) for k in required if not payload.get(str(k))]
    if missing:
        return f"LLM JSON missing required fields: {missing!r}"
    return ""


def _bundle(request: web.Request) -> ThirdPartyLlmBundle:
    return request.app[THIRD_PARTY_LLM_APP_KEY]


async def _read_json(request: web.Request) -> Mapping[str, Any]:
    if not request.body_exists:
        raise _bad_request("missing_body", "Body required")
    text = await request.text()
    if not text.strip():
        raise _bad_request("missing_body", "Empty body")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _bad_request("invalid_json", f"Body is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise _bad_request("invalid_envelope", "Top-level body must be a JSON object")
    return data


def _bad_request(code: str, detail: str) -> web.HTTPBadRequest:
    return web.HTTPBadRequest(
        text=json.dumps({"status": "error", "error": code, "detail": detail}),
        content_type="application/json",
    )


def _error(status: int, code: str, detail: Any) -> web.Response:
    return web.json_response(
        {"status": "error", "error": code, "detail": detail}, status=status
    )


__all__ = [
    "THIRD_PARTY_LLM_APP_KEY",
    "PROVIDER_PRESETS",
    "ThirdPartyLlmBundle",
    "ThirdPartyLlmConfig",
    "attach_third_party_llm_routes",
    "build_third_party_llm_config_from_env",
    "complete_json_with_config",
]
