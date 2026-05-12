"""External LLM client infrastructure (HTTP, OpenAI-compat).

This is the **lifeform-side infra layer** for calling out to an
external LLM. Any wheel that depends on ``lifeform-core``
(i.e. all ``lifeform-*`` wheels) can use the client without
pulling in unrelated subsystems like protocol uptake.

Distinction from "substrate" runtime
====================================

* **Substrate runtime** = the in-process model the kernel uses
  for response synthesis (e.g. Qwen2.5-7B loaded by
  ``lifeform-service``). Lives across sessions per R2 / per-GPU
  invariants. Owned by ``vz-substrate`` + ``vz-runtime``.
* **External LLM client** (this module) = stateless HTTP calls
  to a remote OpenAI-compatible endpoint, used for batch /
  off-turn / structured-extraction tasks where the in-process
  substrate is the wrong tool. Examples:
    * Protocol uptake: extract ``BehaviorProtocol`` from a PDF.
    * Background reflection: summarize a session log.
    * Vertical-specific tools: any domain that wants to call
      out for high-quality JSON output.

Why ``lifeform-core`` and not a new wheel
=========================================

1. Single-file, stdlib-only — adding a wheel just for one class
   is overkill.
2. ``lifeform-core`` is the lowest tier every ``lifeform-*``
   already depends on. Putting the client here means
   ``lifeform-domain-*``, ``lifeform-expression``,
   ``lifeform-protocol-runtime``, ``lifeform-service`` all
   share one implementation without circular deps.
3. The kernel (``vz-*``) MUST NOT import from this module —
   ``vz-* ↛ lifeform-*`` is enforced by import-boundary tests.
   External LLM access is intentionally a lifeform-side concern.

Configuration
=============

This module **does not read environment variables**.
Construct the client with explicit
:class:`OpenAiCompatConfig`. The deployment layer
(``lifeform-service``) reads env / vault / config files and
constructs the shared instance — see
``lifeform_service.openai_compat_client.build_client_from_env``.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


_DEFAULT_TIMEOUT_SECONDS = 60.0


class LlmJsonClient(Protocol):
    """Duck-typed protocol for any LLM that returns JSON-mode output.

    Mirrors the protocol declared in
    ``lifeform_protocol_runtime.document_uptake.extraction.LlmJsonClient``;
    re-declared here so consumers that don't depend on the
    protocol-runtime wheel (e.g. a domain vertical) can still
    satisfy the type.
    """

    def complete_json(
        self, *, system_prompt: str, user_prompt: str
    ) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class OpenAiCompatConfig:
    """Construct-time configuration for :class:`OpenAiCompatJsonClient`.

    No env reading, no defaults that hide secrets — secrets must
    be passed explicitly so the deployment layer is the single
    point of provenance for credentials.
    """

    base_url: str
    api_key: str
    model: str
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS


class OpenAiCompatJsonClient:
    """Stdlib OpenAI-compatible Chat Completions client (JSON mode).

    Sync on purpose: matches the duck-typed
    :class:`LlmJsonClient.complete_json` signature; async wrappers
    can hand this off to a thread pool.

    Wire format: ``POST {base_url}/chat/completions`` with
    ``response_format={"type": "json_object"}``. Endpoints that
    don't natively support JSON-mode usually still produce JSON
    when the prompts ask for it explicitly; we also strip
    leading/trailing markdown fences as a recovery step.
    """

    def __init__(self, config: OpenAiCompatConfig) -> None:
        if not config.base_url.strip():
            raise ValueError(
                "OpenAiCompatJsonClient: base_url must be non-empty"
            )
        if not config.api_key.strip():
            raise ValueError(
                "OpenAiCompatJsonClient: api_key must be non-empty"
            )
        if not config.model.strip():
            raise ValueError(
                "OpenAiCompatJsonClient: model must be non-empty"
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
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"external LLM call failed with HTTP "
                f"{exc.code} {exc.reason}: {error_body[:500]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"external LLM call failed: {exc.reason}"
            ) from exc

        try:
            envelope = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"external LLM returned non-JSON response: "
                f"{response_body[:500]}"
            ) from exc

        choices = envelope.get("choices") or []
        if not choices:
            raise RuntimeError(
                "external LLM response missing 'choices'"
            )
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(
                "external LLM response missing 'content' string"
            )
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            stripped = _strip_markdown_fence(content)
            if stripped is not None:
                try:
                    return json.loads(stripped)
                except json.JSONDecodeError:
                    pass
            raise RuntimeError(
                f"external LLM content is not valid JSON: "
                f"{content[:500]}"
            ) from exc


def _strip_markdown_fence(text: str) -> str | None:
    """Best-effort strip of ```json ... ``` fences."""
    candidate = text.strip()
    if candidate.startswith("```"):
        first_newline = candidate.find("\n")
        if first_newline == -1:
            return None
        candidate = candidate[first_newline + 1 :]
    if candidate.endswith("```"):
        candidate = candidate[: -3].rstrip()
    return candidate or None


__all__ = [
    "LlmJsonClient",
    "OpenAiCompatConfig",
    "OpenAiCompatJsonClient",
]
