"""OpenAI Chat Completions compatible facade over lifeform-service.

This wheel exists to let external chat-style benchmark harnesses (EQ-Bench 3,
EmpathyBench, OpenRouter / Chatbot Arena adapters, the OpenAI Python client,
etc.) talk to a Volvence Zero lifeform without each adapter re-implementing
our session-stateful POST /v1/sessions/{id}/turns envelope.

Design contract (enforced by tests/contracts/test_openai_adapter_*.py):

* This wheel only imports from :mod:`lifeform_service`, the standard
  library, and ``aiohttp``. It MUST NOT import any ``volvence_zero.*``
  kernel sub-package or any ``lifeform_domain_*`` vertical wheel
  directly. The kernel is reached transitively through the
  :class:`lifeform_service.SessionManager` public facade.
* This wheel is read-only with respect to kernel state. The OpenAI
  request → lifeform call mapping never invokes a method whose name
  starts with ``_`` on a SessionManager / LifeformSession, never
  mutates owner snapshots, and never bypasses the SessionManager to
  reach a Lifeform's private modules.
* The OpenAI response shape produced here is byte-compatible with
  ``openai-python``'s ``ChatCompletion`` model so any harness can hit
  it without bespoke client code.

Public surfaces (added incrementally across packets):

Packet 1 (this commit):
  * :class:`ChatCompletionRequest` / :class:`ChatCompletionResponse`
    + supporting DTOs.

Packet 2: raw substrate passthrough mode (mode=raw query param).
Packet 3: stateless + sticky session bridge.
Packet 4: ``add_openai_routes`` aiohttp wiring + three-mode dispatch.
"""

from __future__ import annotations

from lifeform_openai_compat.dto import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    GenerationConfig,
)

__all__ = (
    "ChatCompletionChoice",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatCompletionUsage",
    "ChatMessage",
    "GenerationConfig",
)
