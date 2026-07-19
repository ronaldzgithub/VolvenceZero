"""Service-layer factory for the protocol uptake LLM client.

The actual HTTP implementation lives in
:mod:`lifeform_protocol_runtime.llm_clients`. This module is the
*deployment-specific* wrapper that:

* Reads env vars (``PROTOCOL_LLM_*``) and translates them into an
  :class:`OpenAiCompatConfig`.
* Knows about a small table of **provider presets** (OpenRouter,
  OpenAI, Qwen DashScope, vLLM-localhost, lifeform-openai-compat) so the
  operator only sets ``PROTOCOL_LLM_PROVIDER=openrouter`` +
  ``PROTOCOL_LLM_API_KEY=...`` and gets the right base URL +
  default model automatically.

Why the split (lifeform-protocol-runtime vs here)
=================================================

* The HTTP client is generic — any consumer (CLI tools, batch
  jobs, integration tests) reuses it. Lives next to the duck-typed
  ``LlmJsonClient`` protocol.
* The env-var contract and provider-preset choices are
  deployment concerns. They live where deployment code lives
  (``lifeform-service``).
* Secrets (API keys) belong in shell env / vault / non-committed
  ``.local/*.env`` files. Never in source.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from lifeform_core import (
    OpenAiCompatConfig,
    OpenAiCompatJsonClient,
)


# ---------------------------------------------------------------------------
# Provider presets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ProviderPreset:
    base_url: str
    default_model: str
    notes: str = ""


# Operator chooses one via ``PROTOCOL_LLM_PROVIDER``. Setting the
# provider auto-fills the base_url and a sensible default model;
# explicit PROTOCOL_LLM_BASE_URL / PROTOCOL_LLM_MODEL still
# override.
PROVIDER_PRESETS: Mapping[str, _ProviderPreset] = {
    "openrouter": _ProviderPreset(
        base_url="https://openrouter.ai/api/v1",
        default_model="openai/gpt-4o-mini",
        notes=(
            "OpenRouter OpenAI-compatible Chat Completions. "
            "Override PROTOCOL_LLM_MODEL for provider-specific routing."
        ),
    ),
    "openai": _ProviderPreset(
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        notes="OpenAI Chat Completions JSON-mode",
    ),
    "qwen": _ProviderPreset(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model="qwen-plus",
        notes=(
            "Aliyun DashScope OpenAI-compat mode. Supported models "
            "include qwen-turbo / qwen-plus / qwen-max / qwen-long "
            "and qwen3-* variants."
        ),
    ),
    "dashscope": _ProviderPreset(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model="qwen-plus",
        notes="Alias for qwen.",
    ),
    "vllm": _ProviderPreset(
        base_url="http://localhost:8000/v1",
        default_model="Qwen/Qwen2.5-7B-Instruct",
        notes="Local vLLM server defaults.",
    ),
    "lifeform-openai-compat": _ProviderPreset(
        base_url="http://localhost:8765/v1",
        default_model="lifeform",
        notes=(
            "Volvence Zero's own OpenAI-compat shim "
            "(packages/lifeform-openai-compat)."
        ),
    ),
}


_DEFAULT_PROVIDER = "openrouter"


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def build_client_from_env() -> OpenAiCompatJsonClient | None:
    """Construct an :class:`OpenAiCompatJsonClient` from env, or None.

    Algorithm:

    1. ``PROTOCOL_LLM_API_KEY`` is required. Without it: return None
       (callers branch on this to mount but disable LLM-backed routes).
    2. Resolve a preset from ``PROTOCOL_LLM_PROVIDER`` (default
       ``openai``). Unknown provider → preset is ``custom`` (no
       defaults; ``PROTOCOL_LLM_BASE_URL`` becomes mandatory).
    3. Compute ``base_url`` =
       ``PROTOCOL_LLM_BASE_URL`` if set, else preset.base_url.
       (Required if no preset.)
    4. Compute ``model`` =
       ``PROTOCOL_LLM_MODEL`` if set, else preset.default_model.
    5. Compute ``timeout`` = float(``PROTOCOL_LLM_TIMEOUT_SECONDS``)
       or 60.

    Returns:
        Configured client, or ``None`` if the API key is unset.

    Raises:
        ValueError: when provider=custom but ``PROTOCOL_LLM_BASE_URL``
            is not set.
    """

    api_key = _env("PROTOCOL_LLM_API_KEY")
    if not api_key:
        return None

    provider = _env("PROTOCOL_LLM_PROVIDER", _DEFAULT_PROVIDER).lower()
    preset = PROVIDER_PRESETS.get(provider)

    base_url = _env("PROTOCOL_LLM_BASE_URL")
    if not base_url:
        if preset is None:
            raise ValueError(
                f"PROTOCOL_LLM_PROVIDER={provider!r} is not a known "
                f"preset; set PROTOCOL_LLM_BASE_URL explicitly. "
                f"Known providers: {sorted(PROVIDER_PRESETS)}"
            )
        base_url = preset.base_url

    model = _env("PROTOCOL_LLM_MODEL")
    if not model:
        model = preset.default_model if preset is not None else ""
        if not model:
            raise ValueError(
                "PROTOCOL_LLM_MODEL must be set when no preset "
                "supplies a default."
            )

    timeout_raw = _env("PROTOCOL_LLM_TIMEOUT_SECONDS")
    try:
        timeout = float(timeout_raw) if timeout_raw else 60.0
    except ValueError:
        timeout = 60.0

    return OpenAiCompatJsonClient(
        OpenAiCompatConfig(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=timeout,
        )
    )


def describe_active_provider() -> dict[str, str]:
    """Operator-friendly snapshot of which provider would be picked.

    Returns a dict with ``provider`` / ``base_url`` / ``model`` /
    ``api_key_present`` (True/False, never the value). Used by
    startup-log lines so operators can confirm the routing
    without dumping secrets.
    """

    api_key = _env("PROTOCOL_LLM_API_KEY")
    provider = _env("PROTOCOL_LLM_PROVIDER", _DEFAULT_PROVIDER).lower()
    preset = PROVIDER_PRESETS.get(provider)
    base_url = _env("PROTOCOL_LLM_BASE_URL") or (
        preset.base_url if preset is not None else ""
    )
    model = _env("PROTOCOL_LLM_MODEL") or (
        preset.default_model if preset is not None else ""
    )
    return {
        "provider": provider,
        "base_url": base_url,
        "model": model,
        "api_key_present": "yes" if api_key else "no",
    }


def get_shared_external_llm_client(app) -> OpenAiCompatJsonClient | None:
    """Helper for route handlers / vertical adapters that want the
    shared LLM client.

    Reads ``app["external_llm_client"]``; returns ``None`` when the
    service was started without an LLM (callers respond 503 / fall
    back to the substrate runtime as appropriate).

    Same instance protocol uptake uses, so any vertical-specific
    LLM consumer shares the same API key / quota / provider
    config — operators control external-LLM spend in one place.
    """

    return app.get("external_llm_client")


__all__ = [
    "PROVIDER_PRESETS",
    "build_client_from_env",
    "describe_active_provider",
    "get_shared_external_llm_client",
]
