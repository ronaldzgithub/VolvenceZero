"""Crossref metadata adapter.

Crossref (https://api.crossref.org/) provides DOI-resolved publication
metadata. The figure vertical uses Crossref records to enrich
authored-work metadata that complements OpenAlex (Crossref tends to
have more complete venue / volume / issue strings).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from lifeform_domain_figure.metadata.records import (
    AuthoredWorkSummary,
    MetadataSource,
)


@dataclass(frozen=True)
class CrossrefWorkPayload:
    """Pre-downloaded Crossref ``Work`` record (one DOI)."""

    doi: str  # canonical DOI, e.g., "10.1002/andp.19053221004"
    title: str
    publication_year: int | None
    container_title: str
    language: str
    subject_tags: tuple[str, ...] = ()
    issue: str = ""
    volume: str = ""

    def __post_init__(self) -> None:
        for name in ("doi", "title", "container_title", "language"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"CrossrefWorkPayload.{name} must be non-empty for "
                    f"doi={self.doi!r}"
                )
        if self.publication_year is not None and not (
            -3000 <= self.publication_year <= 9999
        ):
            raise ValueError(
                f"CrossrefWorkPayload.publication_year out of plausible "
                f"range: {self.publication_year!r}"
            )


def crossref_to_authored_work(
    payload: CrossrefWorkPayload,
    *,
    figure_id: str,
) -> AuthoredWorkSummary:
    """Translate a Crossref work payload into an :class:`AuthoredWorkSummary`."""

    venue = payload.container_title
    if payload.volume:
        venue = f"{venue} (vol={payload.volume})"
    if payload.issue:
        venue = f"{venue} (issue={payload.issue})"
    return AuthoredWorkSummary(
        work_id=f"crossref:{payload.doi}",
        figure_id=figure_id,
        title=payload.title,
        year=payload.publication_year,
        venue=venue,
        language=payload.language,
        topic_tags=payload.subject_tags,
        source=MetadataSource.CROSSREF,
        source_id=payload.doi,
    )


class CrossrefClient(Protocol):
    """Forward-declared Protocol for a live Crossref HTTP client."""

    def fetch_work(self, *, doi: str) -> CrossrefWorkPayload: ...


class _OfflineCrossrefClient:
    """V1 stub: every fetch raises ``NotImplementedError``."""

    def fetch_work(self, *, doi: str) -> CrossrefWorkPayload:
        raise NotImplementedError(
            "V1 of the figure vertical has no live Crossref client. "
            "Construct CrossrefWorkPayload instances directly from "
            f"pre-fetched JSON. Refused fetch for doi={doi!r}."
        )


def offline_crossref_client() -> CrossrefClient:
    """Return the V1 offline stub Crossref client."""
    return _OfflineCrossrefClient()


# ---------------------------------------------------------------------------
# V2 live client (debt #26 closure)
# ---------------------------------------------------------------------------

CROSSREF_PROVIDER = "crossref"
CROSSREF_API_BASE = "https://api.crossref.org/works"


def _coerce_first_str(value: object, default: str = "") -> str:
    if isinstance(value, list) and value:
        head = value[0]
        if isinstance(head, str) and head.strip():
            return head.strip()
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _crossref_year(record: dict) -> int | None:
    for field in ("published-print", "published-online", "issued", "created"):
        item = record.get(field)
        if isinstance(item, dict):
            parts = item.get("date-parts")
            if isinstance(parts, list) and parts:
                head = parts[0]
                if isinstance(head, list) and head and isinstance(head[0], int):
                    return head[0]
    return None


def _parse_crossref_record(record: dict) -> CrossrefWorkPayload:
    """Translate one Crossref /works/{doi} ``message`` payload into a typed payload."""

    doi = _coerce_first_str(record.get("DOI"))
    if not doi:
        raise ValueError("LiveCrossrefClient: response missing 'DOI'")
    title = _coerce_first_str(record.get("title"), default="(untitled)")
    container = _coerce_first_str(
        record.get("container-title"), default="(no venue)"
    )
    language = _coerce_first_str(record.get("language"), default="und")
    subjects = record.get("subject")
    subject_tags: tuple[str, ...]
    if isinstance(subjects, list):
        subject_tags = tuple(s.strip() for s in subjects if isinstance(s, str) and s.strip())
    else:
        subject_tags = ()
    issue = _coerce_first_str(record.get("issue"))
    volume = _coerce_first_str(record.get("volume"))
    return CrossrefWorkPayload(
        doi=doi,
        title=title,
        publication_year=_crossref_year(record),
        container_title=container,
        language=language,
        subject_tags=subject_tags,
        issue=issue,
        volume=volume,
    )


def crossref_relations(record: dict) -> dict[str, tuple[str, ...]]:
    """Extract Crossref ``relation`` map (relation_kind -> tuple[doi, ...]).

    Used by the L2 ``VERSION_RECONCILIATION`` verifier to detect
    ``is-version-of`` / ``replaces`` / ``replaced-by`` chains. Returns
    an empty mapping when no relation field is present.
    """

    relation_payload = record.get("relation")
    if not isinstance(relation_payload, dict):
        return {}
    out: dict[str, tuple[str, ...]] = {}
    for kind, items in relation_payload.items():
        if not isinstance(items, list):
            continue
        dois: list[str] = []
        for item in items:
            if isinstance(item, dict):
                identifier = item.get("id") or item.get("DOI")
                if isinstance(identifier, str) and identifier.strip():
                    dois.append(identifier.strip())
        if dois:
            out[str(kind)] = tuple(dois)
    return out


def crossref_translator_names(record: dict) -> tuple[str, ...]:
    """Extract Crossref ``translator`` author names (used by translation_lineage)."""

    translators = record.get("translator")
    if not isinstance(translators, list):
        return ()
    out: list[str] = []
    for entry in translators:
        if isinstance(entry, dict):
            name_parts: list[str] = []
            given = entry.get("given")
            family = entry.get("family")
            if isinstance(given, str) and given.strip():
                name_parts.append(given.strip())
            if isinstance(family, str) and family.strip():
                name_parts.append(family.strip())
            if name_parts:
                out.append(" ".join(name_parts))
    return tuple(out)


class _LiveCrossrefClient:
    """Live Crossref client backed by the metadata HTTP wrapper + cache.

    Caches the **raw** Crossref ``message`` JSON next to the typed
    payload so verifiers needing extra fields (relation map,
    translator names) can re-decode it without a second fetch.
    """

    def __init__(
        self,
        *,
        http_client: "MetadataHTTPClient",
        cache: "MetadataCache | None" = None,
    ) -> None:
        self._http = http_client
        self._cache = cache

    def fetch_work(self, *, doi: str) -> CrossrefWorkPayload:
        if not isinstance(doi, str) or not doi.strip():
            raise ValueError(
                "LiveCrossrefClient.fetch_work: doi must be non-empty"
            )
        normalised = doi.strip()
        cache_key = f"work:{normalised}"
        if self._cache is not None:
            cached = self._cache.get(CROSSREF_PROVIDER, cache_key)
            if cached is not None:
                payload = cached.json()
                if isinstance(payload, dict) and isinstance(payload.get("message"), dict):
                    return _parse_crossref_record(payload["message"])
        url = f"{CROSSREF_API_BASE}/{normalised}"
        response = self._http.get(url, accept="application/json")
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(
                f"LiveCrossrefClient: unexpected top-level shape for doi={normalised!r}"
            )
        message = payload.get("message")
        if not isinstance(message, dict):
            raise ValueError(
                f"LiveCrossrefClient: 'message' missing for doi={normalised!r}"
            )
        record = _parse_crossref_record(message)
        if self._cache is not None:
            self._cache.put(CROSSREF_PROVIDER, cache_key, response)
        return record

    def fetch_raw_message(self, *, doi: str) -> dict:
        """Return the raw Crossref ``message`` payload (used by verifiers)."""

        normalised = doi.strip()
        cache_key = f"work:{normalised}"
        if self._cache is not None:
            cached = self._cache.get(CROSSREF_PROVIDER, cache_key)
            if cached is not None:
                payload = cached.json()
                if isinstance(payload, dict) and isinstance(payload.get("message"), dict):
                    return payload["message"]
        url = f"{CROSSREF_API_BASE}/{normalised}"
        response = self._http.get(url, accept="application/json")
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("message"), dict):
            raise ValueError(
                f"LiveCrossrefClient.fetch_raw_message: unexpected shape for "
                f"doi={normalised!r}"
            )
        if self._cache is not None:
            self._cache.put(CROSSREF_PROVIDER, cache_key, response)
        return payload["message"]


def live_crossref_client(
    *,
    http_client: "MetadataHTTPClient | None" = None,
    cache: "MetadataCache | None" = None,
) -> CrossrefClient:
    """Return a V2 :class:`CrossrefClient` backed by the metadata HTTP stack."""

    from lifeform_domain_figure.metadata.http_client import MetadataHTTPClient

    return _LiveCrossrefClient(
        http_client=http_client or MetadataHTTPClient(),
        cache=cache,
    )


if False:  # pragma: no cover
    from lifeform_domain_figure.metadata.http_client import (  # noqa: F401
        MetadataCache,
        MetadataHTTPClient,
    )


__all__ = [
    "CROSSREF_API_BASE",
    "CROSSREF_PROVIDER",
    "CrossrefClient",
    "CrossrefWorkPayload",
    "crossref_relations",
    "crossref_to_authored_work",
    "crossref_translator_names",
    "live_crossref_client",
    "offline_crossref_client",
]
