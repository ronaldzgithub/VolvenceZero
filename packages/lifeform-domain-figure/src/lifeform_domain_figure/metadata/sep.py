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


# ---------------------------------------------------------------------------
# V2 live client (debt #26 closure)
# ---------------------------------------------------------------------------

SEP_PROVIDER = "sep"
SEP_ENTRY_BASE = "https://plato.stanford.edu/entries"


def _parse_sep_html(html: str, *, slug: str) -> SEPEntryPayload:
    """Extract title / section titles / summary from an SEP entry HTML page.

    Heuristics (matched against SEP's stable page layout circa 2026):

    * Title: first ``<h1>`` (or ``<title>`` minus the trailing
      ``" (Stanford Encyclopedia of Philosophy)"`` suffix).
    * Section titles: every ``<h2>`` inside ``#main-text`` (or the
      whole body when ``#main-text`` is absent), text-only.
    * Summary: paragraph immediately following the
      ``id="preamble"`` block, or the first non-empty ``<p>`` in the
      body if no preamble exists.

    The parser tolerates missing pieces by falling back rather than
    raising; only fully-empty body raises ``ValueError``.
    """

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    main = soup.find(id="main-text") or soup.body or soup
    title_tag = soup.find("h1")
    if title_tag is None:
        title_tag = soup.find("title")
    title_text = title_tag.get_text(strip=True) if title_tag is not None else ""
    if title_text.endswith(" (Stanford Encyclopedia of Philosophy)"):
        title_text = title_text[: -len(" (Stanford Encyclopedia of Philosophy)")].strip()
    if not title_text:
        title_text = slug
    section_titles: list[str] = []
    for h2 in main.find_all("h2"):
        text = h2.get_text(strip=True)
        if text and text not in section_titles:
            section_titles.append(text)
    if not section_titles:
        section_titles = ["(no sections detected)"]
    preamble = soup.find(id="preamble")
    summary = ""
    if preamble is not None:
        first_p = preamble.find("p")
        if first_p is not None:
            summary = first_p.get_text(strip=True)
    if not summary:
        first_p = main.find("p")
        if first_p is not None:
            summary = first_p.get_text(strip=True)
    if not summary:
        summary = f"(no summary detected for slug={slug!r})"
    return SEPEntryPayload(
        entry_slug=slug,
        title=title_text,
        section_titles=tuple(section_titles),
        summary=summary,
    )


class _LiveSEPClient:
    """Live SEP client backed by HTML scraping + cache."""

    def __init__(
        self,
        *,
        http_client: "MetadataHTTPClient",
        cache: "MetadataCache | None" = None,
    ) -> None:
        self._http = http_client
        self._cache = cache

    def fetch_entry(self, *, slug: str) -> SEPEntryPayload:
        if not isinstance(slug, str) or not slug.strip():
            raise ValueError(
                "LiveSEPClient.fetch_entry: slug must be non-empty"
            )
        normalised = slug.strip().strip("/")
        cache_key = f"entry:{normalised}"
        if self._cache is not None:
            cached = self._cache.get(SEP_PROVIDER, cache_key)
            if cached is not None:
                return _parse_sep_html(cached.text(), slug=normalised)
        url = f"{SEP_ENTRY_BASE}/{normalised}/"
        response = self._http.get(url, accept="text/html")
        if self._cache is not None:
            self._cache.put(SEP_PROVIDER, cache_key, response)
        return _parse_sep_html(response.text(), slug=normalised)


def live_sep_client(
    *,
    http_client: "MetadataHTTPClient | None" = None,
    cache: "MetadataCache | None" = None,
) -> SEPClient:
    """Return a V2 :class:`SEPClient` backed by the metadata HTTP stack."""

    from lifeform_domain_figure.metadata.http_client import MetadataHTTPClient

    return _LiveSEPClient(
        http_client=http_client or MetadataHTTPClient(),
        cache=cache,
    )


if False:  # pragma: no cover
    from lifeform_domain_figure.metadata.http_client import (  # noqa: F401
        MetadataCache,
        MetadataHTTPClient,
    )


__all__ = [
    "SEP_ENTRY_BASE",
    "SEP_PROVIDER",
    "SEPClient",
    "SEPEntryPayload",
    "live_sep_client",
    "offline_sep_client",
    "sep_to_domain_hints",
]
