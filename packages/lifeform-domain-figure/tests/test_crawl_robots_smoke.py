"""Smoke tests for the L0 RobotsRegistry."""

from __future__ import annotations

from lifeform_domain_figure.crawl.http_client import BaseHTTPClient
from lifeform_domain_figure.crawl.robots import RobotsRegistry
from lifeform_domain_figure.crawl.scope_policy import ScopePolicy
from crawl_mocks import FakeSession, make_response


def _scope() -> ScopePolicy:
    return ScopePolicy(
        allowed_hosts=frozenset({"example.org"}),
        user_agent="test-agent/1",
    )


def test_disallow_all_blocks_url() -> None:
    robots_text = b"User-agent: *\nDisallow: /\n"

    def handler(url, headers):
        if url.endswith("/robots.txt"):
            return make_response(status_code=200, body=robots_text, content_type="text/plain", url=url)
        raise AssertionError(f"unexpected url {url}")

    client = BaseHTTPClient(scope=_scope(), session=FakeSession(handler))
    registry = RobotsRegistry(http_client=client)
    allowed, reason = registry.is_allowed("https://example.org/x")
    assert not allowed
    assert "robots.txt disallow" in reason


def test_allow_all_returns_true() -> None:
    robots_text = b"User-agent: *\nAllow: /\n"

    def handler(url, headers):
        if url.endswith("/robots.txt"):
            return make_response(status_code=200, body=robots_text, content_type="text/plain", url=url)
        raise AssertionError("unexpected url")

    client = BaseHTTPClient(scope=_scope(), session=FakeSession(handler))
    registry = RobotsRegistry(http_client=client)
    allowed, reason = registry.is_allowed("https://example.org/x")
    assert allowed
    assert reason == ""


def test_404_robots_treated_as_allow_all() -> None:
    def handler(url, headers):
        if url.endswith("/robots.txt"):
            return make_response(status_code=404, url=url)
        raise AssertionError("unexpected url")

    client = BaseHTTPClient(scope=_scope(), session=FakeSession(handler))
    registry = RobotsRegistry(http_client=client)
    allowed, _ = registry.is_allowed("https://example.org/x")
    assert allowed


def test_fetch_failure_fail_closed() -> None:
    def handler(url, headers):
        if url.endswith("/robots.txt"):
            return make_response(status_code=500, url=url)
        raise AssertionError("unexpected url")

    client = BaseHTTPClient(scope=_scope(), session=FakeSession(handler), retries=0)
    registry = RobotsRegistry(http_client=client)
    allowed, reason = registry.is_allowed("https://example.org/x")
    assert not allowed
    assert "robots.txt fetch failed" in reason


def test_cache_hit_avoids_second_fetch() -> None:
    fetch_count = {"count": 0}

    def handler(url, headers):
        if url.endswith("/robots.txt"):
            fetch_count["count"] += 1
            return make_response(
                status_code=200,
                body=b"User-agent: *\nAllow: /\n",
                content_type="text/plain",
                url=url,
            )
        raise AssertionError("unexpected url")

    client = BaseHTTPClient(scope=_scope(), session=FakeSession(handler))
    registry = RobotsRegistry(http_client=client)
    registry.is_allowed("https://example.org/x")
    registry.is_allowed("https://example.org/y")
    assert fetch_count["count"] == 1
