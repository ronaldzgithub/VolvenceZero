"""Compact list renderer.

Emits ``[(name, display_name), ...]`` pairs \u2014 the most terse
summary for inline reference (e.g. in a prompt's "you have these
tools available: read_file, grep, run_test" single line or a
dashboard tooltip).

Deterministic order preserved from ``descriptors``.
"""

from __future__ import annotations

from collections.abc import Iterable

from volvence_zero.affordance import AffordanceDescriptor


def render_compact_list(
    descriptors: Iterable[AffordanceDescriptor],
    *,
    exclude_excluded_from_runtime_selection: bool = True,
) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for d in descriptors:
        if exclude_excluded_from_runtime_selection and d.excluded_from_runtime_selection:
            continue
        pairs.append((d.name, d.display_name))
    return tuple(pairs)


__all__ = ["render_compact_list"]
