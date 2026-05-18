# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Prompt-blend policy — the SINGLE entry point that splices component
state into the request that flows upstream.

Why one entry point: each component (summary / embed / user_model /
episodic) owns its own storage table and exposes a typed snapshot
function. ``HarnessPolicy`` is the only place that aggregates these
snapshots into the outgoing prompt. No component reads another
component's raw fields; no module outside ``policy.py`` constructs
prompt prefixes.

This is the operational instantiation of R8 (snapshot-first
isolation) inside the harness.

The prompt fragments are intentionally short and use a fixed
``[ref-harness · <component>]`` tag so downstream analyses can grep
for the harness contribution in collected transcripts. The
:data:`SYSTEM_PREFIX_HEADER` / :data:`SYSTEM_PREFIX_FOOTER` constants
are public so the analysis scripts can match them exactly.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Iterable

from companion_ref_harness.session_summary import SessionSummary


# ---------------------------------------------------------------------------
# Component set
# ---------------------------------------------------------------------------


class HarnessComponent(str, enum.Enum):
    """One of the four toggleable wrapper components.

    The string values are exactly what the CLI ``--components`` flag
    accepts (comma-separated).
    """

    SUMMARY = "summary"
    EMBED = "embed"
    USER_MODEL = "user_model"
    EPISODIC = "episodic"


@dataclasses.dataclass(frozen=True)
class ComponentSet:
    """Immutable set of enabled components.

    Constructed via :func:`parse_component_set` so the CLI / server /
    tests all share one parsing path (SSOT for the toggle).
    """

    components: frozenset[HarnessComponent]

    def __contains__(self, item: object) -> bool:  # pragma: no cover - trivial
        return item in self.components

    def is_empty(self) -> bool:
        return not self.components

    def has(self, component: HarnessComponent) -> bool:
        return component in self.components

    def to_csv(self) -> str:
        """Canonical CSV representation (sorted, no spaces)."""
        return ",".join(c.value for c in sorted(self.components, key=lambda c: c.value))


def parse_component_set(raw: str | None) -> ComponentSet:
    """Parse the ``--components`` flag value.

    Args:
        raw: Comma-separated component names, or ``None`` / empty
            string for the passthrough (no components) mode.

    Returns:
        Immutable :class:`ComponentSet`.

    Raises:
        ValueError: if any name is unknown. Fail-loud per the
            no-swallow-errors rule.
    """

    if raw is None:
        return ComponentSet(frozenset())
    stripped = raw.strip()
    if not stripped:
        return ComponentSet(frozenset())
    pieces = [p.strip() for p in stripped.split(",") if p.strip()]
    valid: set[HarnessComponent] = set()
    for piece in pieces:
        try:
            valid.add(HarnessComponent(piece))
        except ValueError as exc:
            allowed = sorted(c.value for c in HarnessComponent)
            raise ValueError(
                f"unknown component {piece!r}; allowed: {allowed}"
            ) from exc
    return ComponentSet(frozenset(valid))


# ---------------------------------------------------------------------------
# Prompt fragments
# ---------------------------------------------------------------------------


SYSTEM_PREFIX_HEADER: str = "[ref-harness · cross-session memory ↓]"
SYSTEM_PREFIX_FOOTER: str = "[ref-harness · end memory]"

# Maximum number of prior session summaries we list in the system prefix.
# Public constant; changing this changes the cross-session memory budget
# and is part of the reproducibility contract.
MAX_PRIOR_SUMMARIES: int = 5


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class BlendedMessages:
    """The result of applying the policy to an incoming request.

    Distinct from the raw incoming ``messages`` so callers can log
    which messages they actually sent upstream, separately from
    what arrived at the harness.
    """

    messages: list[dict[str, str]]
    # Whether any component actually contributed to the prompt; useful for
    # smoke tests and for tagging response telemetry (if we ever decide
    # to surface "you got a blended prompt" — we do NOT do so today,
    # because the response shape must stay vendor-neutral per H-A
    # subtask 2).
    blended: bool


class HarnessPolicy:
    """Applies the enabled component set to an incoming request.

    The policy is stateless across requests — it reads from the
    store per call. State lives entirely in the store, which is the
    SSOT for component-owned data.

    H-A scope: only :class:`HarnessComponent.SUMMARY` is wired. The
    remaining components are accepted in :class:`ComponentSet` but
    raise :class:`NotImplementedError` when applied (so H-B / H-C
    development cannot accidentally pretend to be active).
    """

    def __init__(self, components: ComponentSet) -> None:
        self._components = components
        for c in components.components:
            if c is not HarnessComponent.SUMMARY:
                raise NotImplementedError(
                    f"component {c.value!r} is reserved for H-B / H-C; "
                    "H-A wires only 'summary'. Adjust --components or upgrade "
                    "to the H-B/H-C wheel when available."
                )

    @property
    def components(self) -> ComponentSet:
        return self._components

    def blend(
        self,
        *,
        scope_key: str,
        session_id: str,
        messages: list[dict[str, str]],
        prior_summaries: Iterable[SessionSummary],
    ) -> BlendedMessages:
        """Return a new ``messages`` list with component prefixes spliced in.

        The input ``messages`` is **not** mutated; a fresh list is
        returned. The original request can be logged unmodified.
        """

        if self._components.is_empty():
            return BlendedMessages(messages=list(messages), blended=False)

        prefix_block: str | None = None
        if self._components.has(HarnessComponent.SUMMARY):
            prefix_block = _build_summary_prefix(prior_summaries)
        if prefix_block is None:
            return BlendedMessages(messages=list(messages), blended=False)

        spliced = _splice_system_prefix(messages, prefix_block)
        return BlendedMessages(messages=spliced, blended=True)


# ---------------------------------------------------------------------------
# Helpers (private)
# ---------------------------------------------------------------------------


def _build_summary_prefix(
    prior_summaries: Iterable[SessionSummary],
) -> str | None:
    """Render the prior-session summaries block, or ``None`` if empty."""

    materialized = list(prior_summaries)[-MAX_PRIOR_SUMMARIES:]
    if not materialized:
        return None
    lines: list[str] = [SYSTEM_PREFIX_HEADER, ""]
    for s in materialized:
        lines.append(f"### session {s.session_id} (extracted {s.extracted_at})")
        lines.append(s.to_prompt_block())
        lines.append("")
    lines.append(SYSTEM_PREFIX_FOOTER)
    return "\n".join(lines)


def _splice_system_prefix(
    messages: list[dict[str, str]],
    prefix_block: str,
) -> list[dict[str, str]]:
    """Insert ``prefix_block`` into the system message (or create one).

    If the first message has ``role == 'system'``, prepend the block
    to its content so the user-supplied persona instructions remain
    intact. Otherwise, insert a new system message at index 0.
    """

    out: list[dict[str, str]] = [dict(m) for m in messages]
    if out and out[0].get("role") == "system":
        existing_content = out[0].get("content", "")
        out[0] = {
            "role": "system",
            "content": f"{prefix_block}\n\n{existing_content}".rstrip(),
        }
    else:
        out.insert(0, {"role": "system", "content": prefix_block})
    return out
