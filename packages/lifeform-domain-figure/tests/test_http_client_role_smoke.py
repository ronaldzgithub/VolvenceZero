"""Smoke tests for BaseHTTPClient role-aware get (debt #26)."""

from __future__ import annotations

import pytest

from lifeform_domain_figure.crawl.http_client import BaseHTTPClient, ScopeRejection
from lifeform_domain_figure.crawl.scope_policy import (
    ScopePolicy,
    ScopeRole,
    default_combined_scope_policy,
)
from crawl_mocks import FakeSession, make_response


def _scope_with_roles() -> ScopePolicy:
    return ScopePolicy(
        allowed_hosts=frozenset({"corpus.example", "metadata.example"}),
        user_agent="role-test/1",
        host_roles={
            "corpus.example": frozenset({ScopeRole.CORPUS_FETCH}),
            "metadata.example": frozenset({ScopeRole.METADATA_FETCH}),
        },
    )


def test_corpus_fetch_with_corpus_role_succeeds() -> None:
    def handler(u, h):
        return make_response(
            status_code=200, body=b"x", content_type="text/plain", url=u
        )

    client = BaseHTTPClient(scope=_scope_with_roles(), session=FakeSession(handler))
    response = client.get(
        "https://corpus.example/x",
        required_role=ScopeRole.CORPUS_FETCH,
    )
    assert response.body == b"x"


def test_corpus_url_with_metadata_role_rejected() -> None:
    def handler(u, h):
        return make_response(status_code=200, url=u)

    client = BaseHTTPClient(scope=_scope_with_roles(), session=FakeSession(handler))
    with pytest.raises(ScopeRejection, match="missing required_role"):
        client.get(
            "https://corpus.example/x",
            required_role=ScopeRole.METADATA_FETCH,
        )


def test_metadata_url_with_corpus_role_rejected() -> None:
    def handler(u, h):
        return make_response(status_code=200, url=u)

    client = BaseHTTPClient(scope=_scope_with_roles(), session=FakeSession(handler))
    with pytest.raises(ScopeRejection, match="missing required_role"):
        client.get(
            "https://metadata.example/x",
            required_role=ScopeRole.CORPUS_FETCH,
        )


def test_no_role_arg_preserves_legacy_behaviour() -> None:
    def handler(u, h):
        return make_response(
            status_code=200, body=b"y", content_type="text/plain", url=u
        )

    client = BaseHTTPClient(scope=_scope_with_roles(), session=FakeSession(handler))
    response = client.get("https://corpus.example/x")
    assert response.body == b"y"


def test_role_propagated_through_redirect() -> None:
    state = {"first": True}

    def handler(url, headers):
        if state["first"]:
            state["first"] = False
            return make_response(
                status_code=301,
                location="https://metadata.example/y",
                url=url,
            )
        raise AssertionError("redirect to wrong-role host should not be followed")

    client = BaseHTTPClient(scope=_scope_with_roles(), session=FakeSession(handler))
    with pytest.raises(ScopeRejection, match="redirect target"):
        client.get(
            "https://corpus.example/x",
            required_role=ScopeRole.CORPUS_FETCH,
        )


def test_combined_scope_supports_both_roles_independently() -> None:
    def handler(u, h):
        return make_response(
            status_code=200, body=b"z", content_type="application/json", url=u
        )

    client = BaseHTTPClient(
        scope=default_combined_scope_policy("test/1"),
        session=FakeSession(handler),
    )
    r1 = client.get(
        "https://en.wikisource.org/wiki/Foo",
        required_role=ScopeRole.CORPUS_FETCH,
    )
    r2 = client.get(
        "https://api.openalex.org/works",
        required_role=ScopeRole.METADATA_FETCH,
    )
    assert r1.body == b"z"
    assert r2.body == b"z"
