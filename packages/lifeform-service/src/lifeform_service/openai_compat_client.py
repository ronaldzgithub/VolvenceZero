"""Minimal OpenAI-compatible JSON-mode HTTP client.

Used by the protocol uptake routes to call out to an
OpenAI-compatible endpoint (OpenAI proper, vLLM,
oollama, lifeform-openai-compat itself, etc.) for
``LlmJsonClient.complete_json``.

Stdlib-only (urllib.request) to avoid adding ``requests`` /
``httpx`` deps to the service wheel. Synchronous on purpose
— matches the duck-typed :class:`lifeform_protocol_runtime.document_uptake.LlmJsonClient`
contract. Service routes call this from a thread pool when
they need to keep the aiohttp event loop free.

Env-driven config:

* ``PROTOCOL_LLM_BASE_URL`` (e.g. ``https://api.openai.com/v1``)
* ``PROTOCOL_LLM_API_KEY``
* ``PROTOCOL_LLM_MODEL`` (default ``gpt-4o-mini``)
* ``PROTOCOL_LLM_TIMEOUT_SECONDS`` (default 60)

If the base URL is unset the client raises on construction —
callers should fall back to disabling extraction routes when
no LLM is configured.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


_DEFAULT_TIMEOUT_SECONDS = 60.0
_DEFAULT_MODEL = "gpt-4o-mini"


@dataclass(frozen=True)
class OpenAiCompatConfig:
    base_url: str
    api_key: str
    model: str = _DEFAULT_MODEL
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS


class OpenAiCompatJsonClient:
    """Stdlib OpenAI-compatible Chat Completions client (JSON mode).

    Implements :class:`lifeform_protocol_runtime.document_uptake.LlmJsonClient`
    duck-typed protocol: ``complete_json(system_prompt, user_prompt) -> dict``.
    """

    def __init__(self, config: OpenAiCompatConfig) -> None:
        if not config.base_url.strip():
            raise ValueError(
                "OpenAiCompatJsonClient: base_url must be non-empty "
                "(set PROTOCOL_LLM_BASE_URL or pass explicit config)"
            )
        if not config.api_key.strip():
            raise ValueError(
                "OpenAiCompatJsonClient: api_key must be non-empty "
                "(set PROTOCOL_LLM_API_KEY or pass explicit config)"
            )
        self._config = config

    @property
    def model(self) -> str:
        return self._config.model

    @property
    def base_url(self) -> str:
        return self._config.base_url

    def complete_json(
        self, *, system_prompt: str, user_prompt: str
    ) -> dict[str, Any]:
        payload = {
            "model": self._config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }
        body = json.dumps(payload).encode("utf-8")
        url = self._config.base_url.rstrip("/") + "/chat/completions"
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._config.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self._config.timeout_seconds
            ) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"OpenAI-compat call failed with HTTP "
                f"{exc.code} {exc.reason}: {exc.read().decode('utf-8', errors='replace')[:500]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"OpenAI-compat call failed: {exc.reason}"
            ) from exc

        try:
            envelope = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"OpenAI-compat returned non-JSON response: {response_body[:500]}"
            ) from exc

        choices = envelope.get("choices") or []
        if not choices:
            raise RuntimeError(
                "OpenAI-compat response missing 'choices'"
            )
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(
                "OpenAI-compat response missing 'content' string"
            )
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"OpenAI-compat content is not valid JSON: {content[:500]}"
            ) from exc


def build_client_from_env() -> OpenAiCompatJsonClient | None:
    """Construct from env vars; return None if not configured.

    Callers branch on the return value: ``None`` means extraction
    routes should respond 503 ("LLM not configured") rather than
    raise on every request.
    """

    base_url = (os.environ.get("PROTOCOL_LLM_BASE_URL") or "").strip()
    api_key = (os.environ.get("PROTOCOL_LLM_API_KEY") or "").strip()
    if not base_url or not api_key:
        return None
    model = (os.environ.get("PROTOCOL_LLM_MODEL") or _DEFAULT_MODEL).strip() or _DEFAULT_MODEL
    timeout_raw = (os.environ.get("PROTOCOL_LLM_TIMEOUT_SECONDS") or "").strip()
    try:
        timeout = float(timeout_raw) if timeout_raw else _DEFAULT_TIMEOUT_SECONDS
    except ValueError:
        timeout = _DEFAULT_TIMEOUT_SECONDS
    return OpenAiCompatJsonClient(
        OpenAiCompatConfig(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=timeout,
        )
    )


__all__ = [
    "OpenAiCompatConfig",
    "OpenAiCompatJsonClient",
    "build_client_from_env",
]
