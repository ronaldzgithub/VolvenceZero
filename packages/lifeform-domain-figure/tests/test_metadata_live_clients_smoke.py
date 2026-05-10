"""Smoke tests for the 4 live metadata clients (debt #26)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lifeform_domain_figure.crawl.http_client import BaseHTTPClient
from lifeform_domain_figure.metadata.crossref import (
    crossref_relations,
    crossref_translator_names,
    live_crossref_client,
)
from lifeform_domain_figure.metadata.http_client import (
    MetadataCache,
    MetadataHTTPClient,
)
from lifeform_domain_figure.metadata.openalex import live_openalex_client
from lifeform_domain_figure.metadata.sep import live_sep_client
from lifeform_domain_figure.metadata.wikidata import live_wikidata_client
from crawl_mocks import FakeSession, make_response


def _http(handler) -> MetadataHTTPClient:
    from lifeform_domain_figure.crawl.scope_policy import default_metadata_scope_policy

    return MetadataHTTPClient(
        http_client=BaseHTTPClient(
            scope=default_metadata_scope_policy("test/1"),
            session=FakeSession(handler),
        )
    )


# ---- OpenAlex ---------------------------------------------------------------


def test_live_openalex_returns_typed_works(tmp_path: Path) -> None:
    payload = {
        "results": [
            {
                "id": "https://openalex.org/W4205692301",
                "title": "On the electrodynamics of moving bodies",
                "publication_year": 1905,
                "primary_location": {"source": {"display_name": "Annalen der Physik"}},
                "language": "de",
                "concepts": [
                    {"display_name": "special relativity"},
                    {"display_name": "electromagnetism"},
                ],
                "primary_topic": {"display_name": "Special Relativity"},
                "cited_by_count": 12345,
            }
        ],
        "meta": {"next_cursor": None},
    }

    def handler(url, headers):
        assert "filter=author.id:" in url
        return make_response(
            status_code=200,
            body=json.dumps(payload).encode("utf-8"),
            content_type="application/json",
            url=url,
        )

    client = live_openalex_client(http_client=_http(handler))
    works = client.fetch_author_works(openalex_author_id="A5023888391")
    assert len(works) == 1
    assert works[0].openalex_id == "W4205692301"
    assert works[0].publication_year == 1905
    assert "special relativity" in works[0].concept_labels


def test_live_openalex_caches_results(tmp_path: Path) -> None:
    fetch_count = {"n": 0}

    payload = {
        "results": [
            {
                "id": "W1",
                "title": "T",
                "publication_year": 1900,
                "primary_location": {"source": {"display_name": "Nature"}},
                "language": "en",
                "concepts": [],
            }
        ],
        "meta": {"next_cursor": None},
    }

    def handler(url, headers):
        fetch_count["n"] += 1
        return make_response(
            status_code=200,
            body=json.dumps(payload).encode("utf-8"),
            content_type="application/json",
            url=url,
        )

    cache = MetadataCache(root=tmp_path)
    client = live_openalex_client(http_client=_http(handler), cache=cache)
    client.fetch_author_works(openalex_author_id="A1")
    client.fetch_author_works(openalex_author_id="A1")
    assert fetch_count["n"] == 1


# ---- Wikidata ---------------------------------------------------------------


def test_live_wikidata_parses_entity_data() -> None:
    payload = {
        "entities": {
            "Q937": {
                "labels": {"en": {"language": "en", "value": "Albert Einstein"}},
                "claims": {
                    "P569": [
                        {
                            "mainsnak": {
                                "datavalue": {"value": {"time": "+1879-03-14T00:00:00Z"}}
                            }
                        }
                    ],
                    "P570": [
                        {
                            "mainsnak": {
                                "datavalue": {"value": {"time": "+1955-04-18T00:00:00Z"}}
                            }
                        }
                    ],
                    "P106": [
                        {
                            "mainsnak": {
                                "datavalue": {"value": {"id": "Q169470"}}
                            }
                        }
                    ],
                },
            }
        }
    }

    def handler(url, headers):
        assert url.endswith("/Q937.json")
        return make_response(
            status_code=200,
            body=json.dumps(payload).encode("utf-8"),
            content_type="application/json",
            url=url,
        )

    client = live_wikidata_client(http_client=_http(handler))
    person = client.fetch_person(qid="Q937")
    assert person.qid == "Q937"
    assert person.label == "Albert Einstein"
    assert person.birth_year == 1879
    assert person.death_year == 1955
    assert "Q169470" in person.occupation_labels


def test_live_wikidata_rejects_bad_qid() -> None:
    def handler(u, h):
        return make_response(status_code=200, body=b"{}", url=u)

    client = live_wikidata_client(http_client=_http(handler))
    with pytest.raises(ValueError, match="Q-id"):
        client.fetch_person(qid="not-a-qid")


# ---- Crossref ---------------------------------------------------------------


def test_live_crossref_parses_work() -> None:
    payload = {
        "message": {
            "DOI": "10.1002/andp.19053221004",
            "title": ["On the Electrodynamics of Moving Bodies"],
            "container-title": ["Annalen der Physik"],
            "language": "de",
            "subject": ["Physics"],
            "issue": "10",
            "volume": "322",
            "issued": {"date-parts": [[1905, 6]]},
        }
    }

    def handler(url, headers):
        assert "10.1002/andp.19053221004" in url
        return make_response(
            status_code=200,
            body=json.dumps(payload).encode("utf-8"),
            content_type="application/json",
            url=url,
        )

    client = live_crossref_client(http_client=_http(handler))
    work = client.fetch_work(doi="10.1002/andp.19053221004")
    assert work.publication_year == 1905
    assert work.container_title == "Annalen der Physik"
    assert work.volume == "322"


def test_live_crossref_relations_and_translators_from_raw_message() -> None:
    payload = {
        "message": {
            "DOI": "10.1234/example",
            "title": ["Sample Paper"],
            "container-title": ["Sample Journal"],
            "language": "en",
            "issued": {"date-parts": [[1925]]},
            "relation": {
                "is-version-of": [{"id": "10.5678/preprint", "id-type": "doi"}],
                "is-translation-of": [{"DOI": "10.9999/german-original"}],
            },
            "translator": [
                {"given": "Anna", "family": "Smith"},
                {"given": "Bob", "family": "Jones"},
            ],
        }
    }

    def handler(u, h):
        return make_response(
            status_code=200,
            body=json.dumps(payload).encode("utf-8"),
            content_type="application/json",
            url=u,
        )

    client = live_crossref_client(http_client=_http(handler))
    message = client.fetch_raw_message(doi="10.1234/example")
    rel = crossref_relations(message)
    assert "is-version-of" in rel
    assert "10.5678/preprint" in rel["is-version-of"]
    translators = crossref_translator_names(message)
    assert "Anna Smith" in translators
    assert "Bob Jones" in translators


# ---- SEP --------------------------------------------------------------------


def test_live_sep_parses_html_entry() -> None:
    html = b"""<!DOCTYPE html>
<html><head><title>Albert Einstein (Stanford Encyclopedia of Philosophy)</title></head>
<body>
<h1>Albert Einstein</h1>
<div id="preamble"><p>An overview of Einstein's philosophical contributions.</p></div>
<div id="main-text">
  <h2>1. Life</h2>
  <p>Born in 1879...</p>
  <h2>2. The Special Theory of Relativity</h2>
  <h2>3. The General Theory</h2>
</div>
</body></html>
"""

    def handler(url, headers):
        assert "/entries/einstein-philscience" in url
        return make_response(
            status_code=200, body=html, content_type="text/html", url=url
        )

    client = live_sep_client(http_client=_http(handler))
    entry = client.fetch_entry(slug="einstein-philscience")
    assert entry.title == "Albert Einstein"
    assert "1. Life" in entry.section_titles
    assert "Einstein" in entry.summary


def test_metadata_cache_round_trip(tmp_path: Path) -> None:
    from lifeform_domain_figure.metadata.http_client import MetadataResponse

    cache = MetadataCache(root=tmp_path)
    response = MetadataResponse(
        body=b'{"x": 1}',
        content_type="application/json",
        fetched_at_iso="2026-05-10T12:00:00+00:00",
    )
    cache.put("openalex", "key1", response)
    fetched = cache.get("openalex", "key1")
    assert fetched is not None
    assert fetched.body == b'{"x": 1}'
    assert fetched.from_cache is True
    assert cache.get("openalex", "missing-key") is None


def test_metadata_http_client_rejects_corpus_url() -> None:
    from lifeform_domain_figure.crawl.http_client import ScopeRejection
    from lifeform_domain_figure.crawl.scope_policy import default_metadata_scope_policy

    def handler(u, h):
        return make_response(status_code=200, url=u)

    client = MetadataHTTPClient(
        http_client=BaseHTTPClient(
            scope=default_metadata_scope_policy("test/1"),
            session=FakeSession(handler),
        )
    )
    with pytest.raises(ScopeRejection):
        client.get("https://en.wikisource.org/wiki/Foo")
