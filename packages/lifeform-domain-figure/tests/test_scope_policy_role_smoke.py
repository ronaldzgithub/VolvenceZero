"""Smoke tests for ScopePolicy role-tag extension (debt #26)."""

from __future__ import annotations

import pytest

from lifeform_domain_figure.crawl.scope_policy import (
    DEFAULT_CORPUS_HOSTS,
    DEFAULT_HOSTS,
    DEFAULT_METADATA_HOSTS,
    ScopePolicy,
    ScopeRole,
    default_combined_scope_policy,
    default_metadata_scope_policy,
    default_scope_policy,
)


def test_corpus_and_metadata_hosts_disjoint() -> None:
    assert DEFAULT_CORPUS_HOSTS.isdisjoint(DEFAULT_METADATA_HOSTS)
    assert DEFAULT_HOSTS == DEFAULT_CORPUS_HOSTS | DEFAULT_METADATA_HOSTS


def test_default_corpus_scope_tags_each_host_with_corpus_role() -> None:
    scope = default_scope_policy()
    for host in DEFAULT_CORPUS_HOSTS:
        assert scope.host_roles.get(host) == frozenset({ScopeRole.CORPUS_FETCH})


def test_default_metadata_scope_tags_each_host_with_metadata_role() -> None:
    scope = default_metadata_scope_policy()
    for host in DEFAULT_METADATA_HOSTS:
        assert scope.host_roles.get(host) == frozenset({ScopeRole.METADATA_FETCH})


def test_default_combined_scope_carries_both_roles_per_role_set() -> None:
    scope = default_combined_scope_policy()
    for host in DEFAULT_CORPUS_HOSTS:
        assert scope.host_roles.get(host) == frozenset({ScopeRole.CORPUS_FETCH})
    for host in DEFAULT_METADATA_HOSTS:
        assert scope.host_roles.get(host) == frozenset({ScopeRole.METADATA_FETCH})


def test_role_check_passes_when_required_role_matches() -> None:
    scope = default_scope_policy()
    assert scope.is_in_scope(
        "https://en.wikisource.org/wiki/Foo",
        required_role=ScopeRole.CORPUS_FETCH,
    )


def test_role_check_rejects_when_role_missing_on_host() -> None:
    scope = default_scope_policy()
    assert not scope.is_in_scope(
        "https://en.wikisource.org/wiki/Foo",
        required_role=ScopeRole.METADATA_FETCH,
    )
    reason = scope.reason_out_of_scope(
        "https://en.wikisource.org/wiki/Foo",
        required_role=ScopeRole.METADATA_FETCH,
    )
    assert "missing required_role" in reason
    assert "metadata_fetch" in reason


def test_role_check_passes_for_metadata_host_with_metadata_role() -> None:
    scope = default_metadata_scope_policy()
    assert scope.is_in_scope(
        "https://api.openalex.org/works",
        required_role=ScopeRole.METADATA_FETCH,
    )


def test_role_check_rejects_metadata_host_with_corpus_role() -> None:
    scope = default_metadata_scope_policy()
    assert not scope.is_in_scope(
        "https://api.openalex.org/works",
        required_role=ScopeRole.CORPUS_FETCH,
    )


def test_no_role_check_preserves_legacy_behaviour() -> None:
    scope = default_scope_policy()
    assert scope.is_in_scope("https://en.wikisource.org/wiki/Foo")


def test_host_roles_validation_rejects_unknown_host() -> None:
    with pytest.raises(ValueError, match="not present in allowed_hosts"):
        ScopePolicy(
            allowed_hosts=frozenset({"a.example"}),
            user_agent="x/1",
            host_roles={"b.example": frozenset({ScopeRole.CORPUS_FETCH})},
        )


def test_host_roles_validation_rejects_empty_role_set() -> None:
    with pytest.raises(ValueError, match="non-empty frozenset"):
        ScopePolicy(
            allowed_hosts=frozenset({"a.example"}),
            user_agent="x/1",
            host_roles={"a.example": frozenset()},
        )
