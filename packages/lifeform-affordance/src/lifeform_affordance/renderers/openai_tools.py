"""OpenAI-tools renderer.

Produces a list of JSON objects shaped like OpenAI's
``chat.completions`` ``tools`` parameter:

    [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "... when_to_use + when_not_to_use ...",
                "parameters": { ... JSON Schema ... }
            }
        },
        ...
    ]

Only ``AffordanceKind.TOOL`` descriptors are surfaced by default
\u2014 the other kinds (ACTION / ORGAN / SHELL) are not directly
callable through the OpenAI tools interface. Callers that want
every kind (e.g. for a non-OpenAI adapter) can override
``only_kinds``.

The ``description`` field concatenates ``when_to_use`` +
``when_not_to_use`` because OpenAI's tool-selection prompt uses
the description for picking which tool to call; giving it both
positive and negative guidance is empirically the highest-signal
approach.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from volvence_zero.affordance import AffordanceDescriptor, AffordanceKind


def render_openai_tools(
    descriptors: Iterable[AffordanceDescriptor],
    *,
    only_kinds: frozenset[AffordanceKind] = frozenset({AffordanceKind.TOOL}),
    exclude_excluded_from_runtime_selection: bool = True,
) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for d in descriptors:
        if exclude_excluded_from_runtime_selection and d.excluded_from_runtime_selection:
            continue
        if d.kind not in only_kinds:
            continue
        description = (
            f"{d.description.strip()}\n\n"
            f"When to use: {d.when_to_use.strip()}\n\n"
            f"When NOT to use: {d.when_not_to_use.strip()}"
        )
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": d.name,
                    "description": description,
                    "parameters": dict(d.parameters_schema),
                },
            }
        )
    return tools


__all__ = ["render_openai_tools"]
