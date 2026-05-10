"""Smoke tests for L0 ScopePolicy (debt #28)."""

from __future__ import annotations

import pytest

from lifeform_domain_figure.crawl.scope_policy import (
    DEFAULT_HOSTS,
    ScopePolicy,
    default_scope_policy,
)


def test_default_scope_policy_has_5_archive_host_families() -> None:
    scope = default_scope_policy()
    assert "einsteinpapers.press.princeton.edu" in scope.allowed_hosts
    assert "en.wikisource.org" in scope.allowed_hosts
    assert "www.gutenberg.org" in scope.allowed_hosts
    assert "archive.org" in scope.allowed_hosts
    assert "ctext.org" in scope.allowed_hosts


def test_scope_policy_rejects_empty_hosts() -> None:
    with pytest.raises(ValueError, match="must be non-empty"):
        ScopePolicy(allowed_hosts=frozenset(), user_agent="agent/1")


def test_is_in_scope_allows_known_host() -> None:
    scope = default_scope_policy()
    assert scope.is_in_scope("https://en.wikisource.org/wiki/Foo")


def test_is_in_scope_rejects_unknown_host() -> None:
    scope = default_scope_policy()
    assert not scope.is_in_scope("https://evil.example.com/")
    assert "not in scope.allowed_hosts" in scope.reason_out_of_scope(
        "https://evil.example.com/"
    )


def test_is_in_scope_rejects_non_http_scheme() -> None:
    scope = default_scope_policy()
    assert not scope.is_in_scope("file:///etc/passwd")
    assert "scheme=" in scope.reason_out_of_scope("file:///etc/passwd")


def test_is_in_scope_path_prefix_filter() -> None:
    scope = ScopePolicy(
        allowed_hosts=frozenset({"www.gutenberg.org"}),
        user_agent="agent/1",
        allowed_path_prefixes={"www.gutenberg.org": ("/files/", "/ebooks/")},
    )
    assert scope.is_in_scope("https://www.gutenberg.org/files/12345/12345-0.txt")
    assert scope.is_in_scope("https://www.gutenberg.org/ebooks/12345")
    assert not scope.is_in_scope("https://www.gutenberg.org/wiki/Main_Page")


def test_default_hosts_constant_includes_subdomains() -> None:
    assert "ia801.us.archive.org" in DEFAULT_HOSTS
    assert "ia902.us.archive.org" in DEFAULT_HOSTS
    assert "de.wikisource.org" in DEFAULT_HOSTS
