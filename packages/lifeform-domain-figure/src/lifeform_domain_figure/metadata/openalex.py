"""OpenAlex metadata adapter.

OpenAlex (https://api.openalex.org/) is a public scholarly graph
covering papers, authors, venues, and topics. The figure vertical
uses two slices of OpenAlex:

* Author works listing → :class:`AuthoredWorkSummary` records that
  widen the profile's coverage seed before centroid building.
* Concept / topic tags → :class:`DomainCoverageHint` records that
  surface in-domain topic labels for the L4 coverage map.

V1 takes a pre-downloaded :class:`OpenAlexWorkPayload`. V2 will add
an HTTP client behind the :class:`OpenAlexClient` Protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from lifeform_domain_figure.metadata.records import (
    AuthoredWorkSummary,
    DomainCoverageHint,
    MetadataSource,
)


@dataclass(frozen=True)
class OpenAlexWorkPayload:
    """Pre-downloaded OpenAlex ``Work`` record (a single paper / book)."""

    openalex_id: str  # canonical OpenAlex id, e.g., "W4205692301"
    title: str
    publication_year: int | None
    venue: str
    language: str
    concept_labels: tuple[str, ...]
    primary_topic: str = ""
    cited_by_count: int = 0

    def __post_init__(self) -> None:
        for name in ("openalex_id", "title", "venue", "language"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"OpenAlexWorkPayload.{name} must be non-empty for "
                    f"openalex_id={self.openalex_id!r}"
                )
        if self.publication_year is not None and not (
            -3000 <= self.publication_year <= 9999
        ):
            raise ValueError(
                f"OpenAlexWorkPayload.publication_year out of plausible "
                f"range: {self.publication_year!r}"
            )
        if self.cited_by_count < 0:
            raise ValueError(
                f"OpenAlexWorkPayload.cited_by_count must be >= 0, got "
                f"{self.cited_by_count!r}"
            )


def openalex_to_authored_work(
    payload: OpenAlexWorkPayload,
    *,
    figure_id: str,
) -> AuthoredWorkSummary:
    """Translate an OpenAlex work payload into an :class:`AuthoredWorkSummary`."""

    return AuthoredWorkSummary(
        work_id=f"openalex:{payload.openalex_id}",
        figure_id=figure_id,
        title=payload.title,
        year=payload.publication_year,
        venue=payload.venue,
        language=payload.language,
        topic_tags=payload.concept_labels,
        source=MetadataSource.OPENALEX,
        source_id=payload.openalex_id,
    )


def openalex_to_domain_hints(
    payload: OpenAlexWorkPayload,
    *,
    confidence: float = 0.6,
) -> tuple[DomainCoverageHint, ...]:
    """Lift OpenAlex concept labels into :class:`DomainCoverageHint` records.

    Each concept label becomes one in-domain coverage hint
    (``is_out_of_scope=False``). Reviewers may downstream invert
    individual labels to out-of-scope before feeding them into the
    coverage map.
    """

    hints: list[DomainCoverageHint] = []
    seen: set[str] = set()
    for label in payload.concept_labels:
        norm = label.strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        hints.append(
            DomainCoverageHint(
                label=norm,
                description=f"OpenAlex concept tag from work {payload.openalex_id}",
                is_out_of_scope=False,
                source=MetadataSource.OPENALEX,
                source_id=payload.openalex_id,
                confidence=confidence,
            )
        )
    return tuple(hints)


class OpenAlexClient(Protocol):
    """Forward-declared Protocol for a live OpenAlex HTTP client."""

    def fetch_author_works(
        self, *, openalex_author_id: str
    ) -> tuple[OpenAlexWorkPayload, ...]: ...


class _OfflineOpenAlexClient:
    """V1 stub: every fetch raises ``NotImplementedError``."""

    def fetch_author_works(
        self, *, openalex_author_id: str
    ) -> tuple[OpenAlexWorkPayload, ...]:
        raise NotImplementedError(
            "V1 of the figure vertical has no live OpenAlex client. "
            "Construct OpenAlexWorkPayload instances directly from "
            f"pre-fetched JSON. Refused fetch for "
            f"openalex_author_id={openalex_author_id!r}."
        )


def offline_openalex_client() -> OpenAlexClient:
    """Return the V1 offline stub OpenAlex client."""
    return _OfflineOpenAlexClient()


# ---------------------------------------------------------------------------
# V2 live client (debt #26 closure)
# ---------------------------------------------------------------------------

OPENALEX_PROVIDER = "openalex"
OPENALEX_API_BASE = "https://api.openalex.org"


def _coerce_concept_labels(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    out: list[str] = []
    seen: set[str] = set()
    for entry in value:
        if isinstance(entry, dict):
            label = entry.get("display_name") or entry.get("name")
        else:
            label = entry
        if not isinstance(label, str):
            continue
        norm = label.strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return tuple(out)


def _coerce_year(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _coerce_str(value: object, default: str = "") -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _parse_openalex_work(record: dict) -> OpenAlexWorkPayload:
    """Translate one OpenAlex /works record JSON into a typed payload."""

    openalex_id = _coerce_str(record.get("id"))
    if openalex_id.startswith("https://openalex.org/"):
        openalex_id = openalex_id.rsplit("/", 1)[-1]
    title = _coerce_str(
        record.get("title") or record.get("display_name"), default="(untitled)"
    )
    venue_payload = record.get("primary_location") or {}
    if isinstance(venue_payload, dict):
        source_payload = venue_payload.get("source") or {}
        venue = _coerce_str(
            source_payload.get("display_name") if isinstance(source_payload, dict) else "",
            default="(no venue)",
        )
    else:
        venue = "(no venue)"
    language = _coerce_str(record.get("language"), default="und")
    primary_topic_payload = record.get("primary_topic") or {}
    primary_topic = ""
    if isinstance(primary_topic_payload, dict):
        primary_topic = _coerce_str(primary_topic_payload.get("display_name"))
    concept_labels = _coerce_concept_labels(record.get("concepts"))
    cited_by_count = record.get("cited_by_count")
    if not isinstance(cited_by_count, int) or cited_by_count < 0:
        cited_by_count = 0
    return OpenAlexWorkPayload(
        openalex_id=openalex_id,
        title=title,
        publication_year=_coerce_year(record.get("publication_year")),
        venue=venue,
        language=language,
        concept_labels=concept_labels,
        primary_topic=primary_topic,
        cited_by_count=cited_by_count,
    )


class _LiveOpenAlexClient:
    """Live OpenAlex client backed by the metadata HTTP wrapper + cache."""

    def __init__(
        self,
        *,
        http_client: "MetadataHTTPClient",
        cache: "MetadataCache | None" = None,
        per_page: int = 200,
        max_pages: int = 25,
    ) -> None:
        if per_page <= 0 or per_page > 200:
            raise ValueError("OpenAlex per_page must be in (0, 200]")
        if max_pages <= 0:
            raise ValueError("OpenAlex max_pages must be > 0")
        self._http = http_client
        self._cache = cache
        self._per_page = per_page
        self._max_pages = max_pages

    def fetch_author_works(
        self, *, openalex_author_id: str
    ) -> tuple[OpenAlexWorkPayload, ...]:
        if not isinstance(openalex_author_id, str) or not openalex_author_id.strip():
            raise ValueError(
                "LiveOpenAlexClient.fetch_author_works: openalex_author_id "
                "must be a non-empty string (e.g., 'A5023888391')"
            )
        author_id = openalex_author_id.strip()
        if author_id.startswith("https://openalex.org/"):
            author_id = author_id.rsplit("/", 1)[-1]
        cache_key = f"works:author:{author_id}"
        if self._cache is not None:
            cached = self._cache.get(OPENALEX_PROVIDER, cache_key)
            if cached is not None:
                payload = cached.json()
                if isinstance(payload, list):
                    return tuple(_parse_openalex_work(r) for r in payload if isinstance(r, dict))
        works: list[OpenAlexWorkPayload] = []
        cursor = "*"
        for _page in range(self._max_pages):
            url = (
                f"{OPENALEX_API_BASE}/works"
                f"?filter=author.id:{author_id}"
                f"&per-page={self._per_page}"
                f"&cursor={cursor}"
            )
            response = self._http.get(url, accept="application/json")
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError(
                    f"LiveOpenAlexClient: unexpected top-level shape "
                    f"{type(payload).__name__} for author_id={author_id!r}"
                )
            results = payload.get("results")
            if not isinstance(results, list):
                raise ValueError(
                    f"LiveOpenAlexClient: 'results' missing/invalid for "
                    f"author_id={author_id!r}"
                )
            for record in results:
                if isinstance(record, dict):
                    works.append(_parse_openalex_work(record))
            meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
            next_cursor = meta.get("next_cursor")
            if not isinstance(next_cursor, str) or not next_cursor:
                break
            cursor = next_cursor
        if self._cache is not None:
            from datetime import datetime, timezone

            from lifeform_domain_figure.metadata.http_client import MetadataResponse
            import json as _json

            serialised = _json.dumps(
                [
                    {
                        "id": w.openalex_id,
                        "title": w.title,
                        "publication_year": w.publication_year,
                        "venue": w.venue,
                        "language": w.language,
                        "concept_labels": list(w.concept_labels),
                        "primary_topic": w.primary_topic,
                        "cited_by_count": w.cited_by_count,
                    }
                    for w in works
                ],
                ensure_ascii=False,
            ).encode("utf-8")
            self._cache.put(
                OPENALEX_PROVIDER,
                cache_key,
                MetadataResponse(
                    body=serialised,
                    content_type="application/json",
                    fetched_at_iso=datetime.now(timezone.utc).isoformat(),
                ),
            )
        return tuple(works)


def live_openalex_client(
    *,
    http_client: "MetadataHTTPClient | None" = None,
    cache: "MetadataCache | None" = None,
    per_page: int = 200,
    max_pages: int = 25,
) -> OpenAlexClient:
    """Return a V2 :class:`OpenAlexClient` backed by the metadata HTTP stack."""

    from lifeform_domain_figure.metadata.http_client import MetadataHTTPClient

    return _LiveOpenAlexClient(
        http_client=http_client or MetadataHTTPClient(),
        cache=cache,
        per_page=per_page,
        max_pages=max_pages,
    )


# Forward-declared for type hints in the live client signature without
# pulling http_client into module top-level imports (tests can stub).
if False:  # pragma: no cover - typing-only block
    from lifeform_domain_figure.metadata.http_client import (  # noqa: F401
        MetadataCache,
        MetadataHTTPClient,
    )


__all__ = [
    "OPENALEX_API_BASE",
    "OPENALEX_PROVIDER",
    "OpenAlexClient",
    "OpenAlexWorkPayload",
    "live_openalex_client",
    "offline_openalex_client",
    "openalex_to_authored_work",
    "openalex_to_domain_hints",
]
