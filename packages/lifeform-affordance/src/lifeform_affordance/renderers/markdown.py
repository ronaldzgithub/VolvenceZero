"""Markdown renderer \u2014 LLM-facing prose.

The output format:

    # Available affordances

    ## read_file (tool, v0.1.0)

    Read a UTF-8 text file from the workspace and return its content.

    **When to use**: ... (full text) ...

    **When NOT to use**: ... (full text) ...

    **Parameters**: `{ "type": "object", ... }`

    **Examples**:
    - read_file(path='src/main.py')

Deliberately minimal: no tables, no HTML. LLMs read this as
context in a system prompt; the structure matters more than the
prettiness. Deterministic ordering + stable Markdown means cached
system prompts survive across turns.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

from volvence_zero.affordance import AffordanceDescriptor


def render_markdown(
    descriptors: Iterable[AffordanceDescriptor],
    *,
    header: str = "# Available affordances",
    exclude_excluded_from_runtime_selection: bool = True,
) -> str:
    """Return a Markdown document describing every descriptor.

    ``exclude_excluded_from_runtime_selection=True`` filters out
    descriptors that are registered for administrative / telemetry
    purposes but should not appear in the LLM's prompt. Set to
    ``False`` for management UIs that want the full catalog.
    """
    lines: list[str] = [header, ""]
    rendered_any = False
    for d in descriptors:
        if exclude_excluded_from_runtime_selection and d.excluded_from_runtime_selection:
            continue
        rendered_any = True
        lines.append(f"## {d.name} ({d.kind.value}, v{d.version})")
        lines.append("")
        lines.append(d.description.strip())
        lines.append("")
        lines.append(f"**When to use**: {d.when_to_use.strip()}")
        lines.append("")
        lines.append(f"**When NOT to use**: {d.when_not_to_use.strip()}")
        lines.append("")
        lines.append(
            "**Parameters**: `"
            + json.dumps(dict(d.parameters_schema), sort_keys=True)
            + "`"
        )
        if d.examples:
            lines.append("")
            lines.append("**Examples**:")
            for example in d.examples:
                lines.append(f"- {example}")
        lines.append("")
    if not rendered_any:
        # Don't lie: say explicitly when no affordances are available.
        lines.append("_(no affordances registered for runtime selection)_")
    return "\n".join(lines).rstrip() + "\n"


__all__ = ["render_markdown"]
