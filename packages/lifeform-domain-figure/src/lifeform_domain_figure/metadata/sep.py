"""Stanford Encyclopedia of Philosophy (SEP) metadata adapter.

The SEP (https://plato.stanford.edu/) provides reviewed encyclopedia
entries on philosophical topics. The figure vertical uses SEP entry
outlines / topic structure as in-domain coverage hints — they
complement OpenAlex's purely-citation-derived concept tags with
philosopher-curated topic decompositions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from lifeform_domain_figure.metadata.records import (
    DomainCoverageHint,
    MetadataSource,
)


@dataclass(frozen=True)
class SEPEntryPayload:
    """Pre-downloaded SEP entry payload."""

    entry_slug: str  # canonical SEP slug, e.g., "einstein-philscience"
    title: str
    section_titles: tuple[str, ...]
    summary: str

    def __post_init__(self) -> None:
        for name in ("entry_slug", "title", "summary"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"SEPEntryPayload.{name} must be non-empty for "
                    f"entry_slug={self.entry_slug!r}"
                )
        if not self.section_titles:
            raise ValueError(
                f"SEPEntryPayload.section_titles must be non-empty for "
                f"entry_slug={self.entry_slug!r}; SEP entries always "
                f"have a sectioned outline."
            )


def sep_to_domain_hints(
    payload: SEPEntryPayload,
    *,
    confidence: float = 0.7,
) -> tuple[DomainCoverageHint, ...]:
    """Lift SEP section titles into :class:`DomainCoverageHint` records.

    Each section title becomes one in-domain coverage hint
    (``is_out_of_scope=False``). The reviewer can downstream invert
    individual sections to out-of-scope before they reach the
    coverage map.
    """

    hints: list[DomainCoverageHint] = []
    seen: set[str] = set()
    for section in payload.section_titles:
        norm = section.strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        hints.append(
            DomainCoverageHint(
                label=norm,
                description=(
                    f"SEP section title from entry {payload.entry_slug!r}"
                ),
                is_out_of_scope=False,
                source=MetadataSource.SEP,
                source_id=payload.entry_slug,
                confidence=confidence,
            )
        )
    return tuple(hints)


class SEPClient(Protocol):
    """Forward-declared Protocol for a live SEP HTTP client."""

    def fetch_entry(self, *, slug: str) -> SEPEntryPayload: ...


class _OfflineSEPClient:
    """V1 stub: every fetch raises ``NotImplementedError``."""

    def fetch_entry(self, *, slug: str) -> SEPEntryPayload:
        raise NotImplementedError(
            "V1 of the figure vertical has no live SEP client. "
            "Construct SEPEntryPayload instances directly from "
            f"pre-downloaded outline data. Refused fetch for slug={slug!r}."
        )


def offline_sep_client() -> SEPClient:
    """Return the V1 offline stub SEP client."""
    return _OfflineSEPClient()


__all__ = [
    "SEPClient",
    "SEPEntryPayload",
    "offline_sep_client",
    "sep_to_domain_hints",
]
