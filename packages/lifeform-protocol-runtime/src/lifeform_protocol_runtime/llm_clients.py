"""Re-exports of the lifeform-core external-LLM client.

The actual implementation moved to
:mod:`lifeform_core.external_llm` (see that module's docstring
for the layering rationale: any ``lifeform-*`` wheel can use the
client without depending on the protocol-runtime wheel).

This module remains here as a thin shim so existing imports
(``from lifeform_protocol_runtime import OpenAiCompatJsonClient``)
keep working. New code should import directly from
``lifeform_core``.
"""

from __future__ import annotations

from lifeform_core.external_llm import (
    OpenAiCompatConfig,
    OpenAiCompatJsonClient,
)

__all__ = [
    "OpenAiCompatConfig",
    "OpenAiCompatJsonClient",
]
