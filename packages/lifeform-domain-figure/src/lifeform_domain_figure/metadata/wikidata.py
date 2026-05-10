"""Wikidata metadata adapter.

Wikidata (https://www.wikidata.org/) holds biographical metadata for
historical figures. The figure vertical uses three slices:

* ``date_of_birth`` / ``date_of_death`` → :class:`FigureLifespan`.
  The lifespan feeds the L4 not-known refusal contract: any query
  about events after ``death_year`` is automatically out-of-scope.
* ``occupation`` / ``field_of_work`` labels → optional in-domain
  coverage hints (low confidence; reviewer review required).
* Reviewer-declared time-window hints (e.g., "early-career-1900-1925")
  derived from the lifespan plus reviewer policy → produced by
  :func:`wikidata_to_time_window_hints`.

V1 takes a pre-downloaded :class:`WikidataPersonPayload`. V2 will
add the live SPARQL / WDQS Protocol implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from lifeform_domain_figure.metadata.records import (
    FigureLifespan,
    MetadataSource,
    TimeWindowHint,
)
from lifeform_domain_figure.metadata.time_window_builder import (
    build_time_window_hints_from_lifespan,
)


@dataclass(frozen=True)
class WikidataPersonPayload:
    """Pre-downloaded Wikidata person record (a single Q-id)."""

    qid: str  # canonical Wikidata id, e.g., "Q937"
    label: str  # "Albert Einstein"
    birth_year: int
    death_year: int | None
    occupation_labels: tuple[str, ...] = ()
    field_of_work_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("qid", "label"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"WikidataPersonPayload.{name} must be non-empty for "
                    f"qid={self.qid!r}"
                )
        if self.death_year is not None and self.death_year < self.birth_year:
            raise ValueError(
                f"WikidataPersonPayload: death_year ({self.death_year}) "
                f"must be >= birth_year ({self.birth_year}) for "
                f"qid={self.qid!r}"
            )


def wikidata_to_lifespan(
    payload: WikidataPersonPayload,
    *,
    figure_id: str,
    confidence: float = 0.95,
) -> FigureLifespan:
    """Translate a Wikidata person payload into a :class:`FigureLifespan`."""

    return FigureLifespan(
        figure_id=figure_id,
        birth_year=payload.birth_year,
        death_year=payload.death_year,
        source=MetadataSource.WIKIDATA,
        source_id=payload.qid,
        confidence=confidence,
    )


def wikidata_to_time_window_hints(
    payload: WikidataPersonPayload,
    *,
    figure_id: str,
    splits_at_years: tuple[int, ...] = (),
) -> tuple[TimeWindowHint, ...]:
    """Build a tuple of :class:`TimeWindowHint` records from a Wikidata payload.

    ``splits_at_years`` lets the curator declare reviewed inflection
    years (e.g., ``(1925,)`` for an early-vs-late Einstein split). If
    empty, the helper returns one window covering the full lifespan.
    """

    lifespan = wikidata_to_lifespan(payload, figure_id=figure_id)
    return build_time_window_hints_from_lifespan(
        lifespan,
        splits_at_years=splits_at_years,
        source=MetadataSource.WIKIDATA,
        source_id=payload.qid,
    )


class WikidataClient(Protocol):
    """Forward-declared Protocol for a live Wikidata SPARQL / WDQS client."""

    def fetch_person(self, *, qid: str) -> WikidataPersonPayload: ...


class _OfflineWikidataClient:
    """V1 stub: every fetch raises ``NotImplementedError``."""

    def fetch_person(self, *, qid: str) -> WikidataPersonPayload:
        raise NotImplementedError(
            "V1 of the figure vertical has no live Wikidata client. "
            "Construct WikidataPersonPayload instances directly from "
            f"pre-fetched SPARQL JSON. Refused fetch for qid={qid!r}."
        )


def offline_wikidata_client() -> WikidataClient:
    """Return the V1 offline stub Wikidata client."""
    return _OfflineWikidataClient()


# ---------------------------------------------------------------------------
# V2 live client (debt #26 closure)
# ---------------------------------------------------------------------------

WIKIDATA_PROVIDER = "wikidata"
WIKIDATA_ENTITY_BASE = "https://www.wikidata.org/wiki/Special:EntityData"


def _claim_value(claims: dict, prop: str, *, key: str = "id") -> object | None:
    items = claims.get(prop)
    if not isinstance(items, list) or not items:
        return None
    mainsnak = items[0].get("mainsnak") if isinstance(items[0], dict) else None
    if not isinstance(mainsnak, dict):
        return None
    datavalue = mainsnak.get("datavalue")
    if not isinstance(datavalue, dict):
        return None
    value = datavalue.get("value")
    if isinstance(value, dict):
        return value.get(key)
    return value


def _claim_year(claims: dict, prop: str) -> int | None:
    value = _claim_value(claims, prop, key="time")
    if not isinstance(value, str):
        return None
    payload = value.lstrip("+")
    head = payload.split("-", 1)[0]
    sign = 1
    if value.startswith("-"):
        sign = -1
    if head.isdigit():
        return sign * int(head)
    return None


def _claim_string_set(
    claims: dict, prop: str, label_lookup: dict[str, str]
) -> tuple[str, ...]:
    items = claims.get(prop)
    if not isinstance(items, list):
        return ()
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        mainsnak = item.get("mainsnak")
        if not isinstance(mainsnak, dict):
            continue
        datavalue = mainsnak.get("datavalue")
        if not isinstance(datavalue, dict):
            continue
        value = datavalue.get("value")
        if isinstance(value, dict):
            qid = value.get("id")
            if isinstance(qid, str):
                label = label_lookup.get(qid, qid)
                if label not in seen:
                    seen.add(label)
                    out.append(label)
    return tuple(out)


def _label_lookup_for(qids: tuple[str, ...]) -> dict[str, str]:
    """Return identity mapping qid -> qid (no extra fetch).

    A real Wikidata client may issue additional EntityData calls to
    resolve the human-readable label for each qid; the V2 client
    keeps it minimal (qid as fallback label) and lets verifiers /
    callers join against their own label dicts.
    """

    return {qid: qid for qid in qids}


def _parse_wikidata_entity(payload: dict, *, qid: str) -> WikidataPersonPayload:
    entities = payload.get("entities")
    if not isinstance(entities, dict):
        raise ValueError(
            f"LiveWikidataClient: 'entities' missing for qid={qid!r}"
        )
    entity = entities.get(qid)
    if not isinstance(entity, dict):
        raise ValueError(
            f"LiveWikidataClient: entity {qid!r} not in response"
        )
    labels = entity.get("labels") or {}
    label = ""
    if isinstance(labels, dict):
        for lang in ("en", "de", "fr", "zh"):
            entry = labels.get(lang)
            if isinstance(entry, dict) and isinstance(entry.get("value"), str):
                label = entry["value"]
                break
        if not label:
            for entry in labels.values():
                if isinstance(entry, dict) and isinstance(entry.get("value"), str):
                    label = entry["value"]
                    break
    if not label:
        label = qid
    claims = entity.get("claims")
    if not isinstance(claims, dict):
        raise ValueError(
            f"LiveWikidataClient: 'claims' missing for qid={qid!r}"
        )
    birth_year = _claim_year(claims, "P569")
    death_year = _claim_year(claims, "P570")
    if birth_year is None:
        raise ValueError(
            f"LiveWikidataClient: birth_year (P569) missing for qid={qid!r}"
        )
    occupation_qids: list[str] = []
    field_qids: list[str] = []
    for prop_id, sink in (("P106", occupation_qids), ("P101", field_qids)):
        items = claims.get(prop_id) or []
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                mainsnak = item.get("mainsnak")
                if not isinstance(mainsnak, dict):
                    continue
                dv = mainsnak.get("datavalue")
                if not isinstance(dv, dict):
                    continue
                value = dv.get("value")
                if isinstance(value, dict) and isinstance(value.get("id"), str):
                    sink.append(value["id"])
    occupation_lookup = _label_lookup_for(tuple(occupation_qids))
    field_lookup = _label_lookup_for(tuple(field_qids))
    return WikidataPersonPayload(
        qid=qid,
        label=label,
        birth_year=birth_year,
        death_year=death_year,
        occupation_labels=tuple(occupation_lookup[q] for q in occupation_qids),
        field_of_work_labels=tuple(field_lookup[q] for q in field_qids),
    )


class _LiveWikidataClient:
    """Live Wikidata client backed by Special:EntityData JSON + cache."""

    def __init__(
        self,
        *,
        http_client: "MetadataHTTPClient",
        cache: "MetadataCache | None" = None,
    ) -> None:
        self._http = http_client
        self._cache = cache

    def fetch_person(self, *, qid: str) -> WikidataPersonPayload:
        if not isinstance(qid, str) or not qid.strip().startswith("Q"):
            raise ValueError(
                "LiveWikidataClient.fetch_person: qid must be a Wikidata "
                f"Q-id like 'Q937'; got {qid!r}"
            )
        normalised = qid.strip()
        cache_key = f"person:{normalised}"
        if self._cache is not None:
            cached = self._cache.get(WIKIDATA_PROVIDER, cache_key)
            if cached is not None:
                payload = cached.json()
                if isinstance(payload, dict):
                    return _parse_wikidata_entity(payload, qid=normalised)
        url = f"{WIKIDATA_ENTITY_BASE}/{normalised}.json"
        response = self._http.get(url, accept="application/json")
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(
                f"LiveWikidataClient: unexpected response shape for qid={normalised!r}"
            )
        person = _parse_wikidata_entity(payload, qid=normalised)
        if self._cache is not None:
            self._cache.put(WIKIDATA_PROVIDER, cache_key, response)
        return person


def live_wikidata_client(
    *,
    http_client: "MetadataHTTPClient | None" = None,
    cache: "MetadataCache | None" = None,
) -> WikidataClient:
    """Return a V2 :class:`WikidataClient` backed by the metadata HTTP stack."""

    from lifeform_domain_figure.metadata.http_client import MetadataHTTPClient

    return _LiveWikidataClient(
        http_client=http_client or MetadataHTTPClient(),
        cache=cache,
    )


if False:  # pragma: no cover
    from lifeform_domain_figure.metadata.http_client import (  # noqa: F401
        MetadataCache,
        MetadataHTTPClient,
    )


__all__ = [
    "WIKIDATA_ENTITY_BASE",
    "WIKIDATA_PROVIDER",
    "WikidataClient",
    "WikidataPersonPayload",
    "live_wikidata_client",
    "offline_wikidata_client",
    "wikidata_to_lifespan",
    "wikidata_to_time_window_hints",
]
