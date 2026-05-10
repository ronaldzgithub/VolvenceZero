"""Smoke tests for L0 BaseHTTPClient (SSRF gates + 304 + body cap)."""

from __future__ import annotations

import pytest

from lifeform_domain_figure.crawl.http_client import (
    BaseHTTPClient,
    BodyTooLarge,
    FetchError,
    HTTPResponse,
    NOT_MODIFIED,
    ScopeRejection,
)
from lifeform_domain_figure.crawl.scope_policy import ScopePolicy
from crawl_mocks import FakeSession, make_response


def _scope_for(host: str = "example.org", *, max_body: int = 1024 * 1024) -> ScopePolicy:
    return ScopePolicy(
        allowed_hosts=frozenset({host}),
        user_agent="test-agent/1",
        max_body_bytes=max_body,
    )


def test_get_returns_http_response_for_200() -> None:
    def handler(url, headers):
        assert headers["User-Agent"] == "test-agent/1"
        return make_response(
            status_code=200,
            body=b"hello",
            content_type="text/plain",
            etag='"v1"',
            url=url,
        )

    scope = _scope_for()
    client = BaseHTTPClient(scope=scope, session=FakeSession(handler))
    result = client.get("https://example.org/x")
    assert isinstance(result, HTTPResponse)
    assert result.body == b"hello"
    assert result.content_type == "text/plain"
    assert result.etag == '"v1"'


def test_scheme_gate_rejects_file_url() -> None:
    scope = _scope_for()
    client = BaseHTTPClient(scope=scope, session=FakeSession(lambda u, h: make_response()))
    with pytest.raises(ScopeRejection):
        client.get("file:///etc/passwd")


def test_host_gate_rejects_localhost() -> None:
    scope = _scope_for()
    client = BaseHTTPClient(scope=scope, session=FakeSession(lambda u, h: make_response()))
    with pytest.raises(ScopeRejection):
        client.get("https://localhost/")


def test_redirect_to_off_scope_host_rejected() -> None:
    state = {"first": True}

    def handler(url, headers):
        if state["first"]:
            state["first"] = False
            return make_response(
                status_code=302,
                location="https://evil.example.com/x",
                url=url,
            )
        raise AssertionError("should not reach second hop")

    scope = _scope_for("example.org")
    client = BaseHTTPClient(scope=scope, session=FakeSession(handler))
    with pytest.raises(ScopeRejection, match="redirect target"):
        client.get("https://example.org/x")


def test_redirect_to_in_scope_host_followed_one_hop() -> None:
    state = {"first": True}

    def handler(url, headers):
        if state["first"]:
            state["first"] = False
            return make_response(
                status_code=301,
                location="/y",
                url=url,
            )
        return make_response(
            status_code=200,
            body=b"final",
            content_type="text/plain",
            url=url,
        )

    scope = _scope_for("example.org")
    client = BaseHTTPClient(scope=scope, session=FakeSession(handler))
    result = client.get("https://example.org/x")
    assert isinstance(result, HTTPResponse)
    assert result.body == b"final"


def test_304_returns_sentinel() -> None:
    def handler(url, headers):
        assert headers.get("If-None-Match") == '"v1"'
        return make_response(status_code=304, url=url)

    scope = _scope_for()
    client = BaseHTTPClient(scope=scope, session=FakeSession(handler))
    result = client.get("https://example.org/x", etag='"v1"')
    assert result is NOT_MODIFIED


def test_body_cap_raises() -> None:
    big_body = b"a" * 2048

    def handler(url, headers):
        return make_response(status_code=200, body=big_body, url=url)

    scope = _scope_for(max_body=1024)
    client = BaseHTTPClient(scope=scope, session=FakeSession(handler))
    with pytest.raises(BodyTooLarge):
        client.get("https://example.org/x")


def test_4xx_raises_fetch_error() -> None:
    def handler(url, headers):
        return make_response(status_code=404, url=url)

    scope = _scope_for()
    client = BaseHTTPClient(scope=scope, session=FakeSession(handler))
    with pytest.raises(FetchError, match="http_status=404"):
        client.get("https://example.org/missing")
