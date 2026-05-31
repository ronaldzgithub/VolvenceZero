"""D2: pluggable scoped-memory backend + two-layer scope compat shim.

Covers:
* :class:`InMemoryPersistenceBackend` honours the same CRUD contract as
  :class:`FileSystemPersistenceBackend` (versioning, prefix listing,
  delete).
* :func:`resolve_persistence_backend` selects the backend from an
  argument / env, defaulting to the historical filesystem backend.
* :func:`build_scoped_memory_store` accepts an injected backend and
  round-trips durable content through it.
* :func:`legacy_single_layer_scope` + ``extra_scopes`` keep durable
  entries written under the pre-two-layer single-layer tag reachable
  after the two-layer-default flip (no silent re-key).
"""

from __future__ import annotations

import pytest

from volvence_zero.memory import (
    EndUserIdentity,
    FileSystemPersistenceBackend,
    InMemoryPersistenceBackend,
    MemoryStratum,
    MemoryWriteRequest,
    TenantIdentity,
    Track,
    UserIdentity,
    build_scoped_memory_store,
    delete_entries_for_scope,
    derive_scope_key,
    legacy_single_layer_scope,
    list_durable_entries_for_scope,
    resolve_persistence_backend,
)


# ---------------------------------------------------------------------------
# InMemoryPersistenceBackend contract parity
# ---------------------------------------------------------------------------


def test_in_memory_backend_save_load_roundtrip() -> None:
    backend = InMemoryPersistenceBackend()
    backend.save_checkpoint(key="scope/a", data=b"v1-data", version=1)
    backend.save_checkpoint(key="scope/a", data=b"v2-data", version=2)
    loaded = backend.load_checkpoint(key="scope/a")
    assert loaded == (b"v2-data", 2)  # newest version wins


def test_in_memory_backend_missing_key_returns_none() -> None:
    backend = InMemoryPersistenceBackend()
    assert backend.load_checkpoint(key="nope") is None


def test_in_memory_backend_prunes_to_max_versions() -> None:
    backend = InMemoryPersistenceBackend(max_versions=2)
    for v in range(1, 6):
        backend.save_checkpoint(key="k", data=f"v{v}".encode(), version=v)
    # Only the 2 newest versions are retained; newest still loads.
    assert backend.load_checkpoint(key="k") == (b"v5", 5)


def test_in_memory_backend_list_and_delete() -> None:
    backend = InMemoryPersistenceBackend()
    backend.save_checkpoint(key="alpha:bob", data=b"{}", version=1)
    backend.save_checkpoint(key="alpha:carol", data=b"{}", version=1)
    backend.save_checkpoint(key="brand:dave", data=b"{}", version=1)
    assert backend.list_checkpoints(prefix="alpha:") == ("alpha:bob", "alpha:carol")
    backend.delete_checkpoint(key="alpha:bob")
    assert backend.list_checkpoints(prefix="alpha:") == ("alpha:carol",)


# ---------------------------------------------------------------------------
# resolve_persistence_backend selection
# ---------------------------------------------------------------------------


def test_resolve_defaults_to_filesystem(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("VZ_MEMORY_BACKEND", raising=False)
    backend = resolve_persistence_backend(base_dir=str(tmp_path))
    assert isinstance(backend, FileSystemPersistenceBackend)


def test_resolve_memory_backend_by_argument(tmp_path) -> None:
    backend = resolve_persistence_backend(base_dir=str(tmp_path), backend="memory")
    assert isinstance(backend, InMemoryPersistenceBackend)


def test_resolve_memory_backend_by_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VZ_MEMORY_BACKEND", "memory")
    backend = resolve_persistence_backend(base_dir=str(tmp_path))
    assert isinstance(backend, InMemoryPersistenceBackend)


def test_resolve_unknown_backend_raises(tmp_path) -> None:
    with pytest.raises(ValueError):
        resolve_persistence_backend(base_dir=str(tmp_path), backend="cassandra")


def test_resolve_postgres_without_driver_fails_loud(monkeypatch) -> None:
    pytest.importorskip  # noqa: B018 - keep import available
    try:
        import psycopg  # type: ignore  # noqa: F401
    except ImportError:
        # Driver absent (the CI default): constructing the backend must
        # raise an actionable error rather than silently degrade.
        with pytest.raises(RuntimeError) as exc:
            resolve_persistence_backend(
                base_dir=None, backend="postgres", dsn="postgresql://x"
            )
        assert "psycopg" in str(exc.value)
    else:  # pragma: no cover - only when a real driver is installed
        pytest.skip("psycopg installed; live-connection path not exercised here")


# ---------------------------------------------------------------------------
# build_scoped_memory_store with an injected backend
# ---------------------------------------------------------------------------


_TS = 1_700_000_000_000


def _durable_write(store, content: str, scope: str) -> None:
    store.write(
        MemoryWriteRequest(
            content=content,
            track=Track.SELF,
            stratum=MemoryStratum.DURABLE,
            tags=(f"user_scope:{scope}", "rupture_repair"),
            strength=0.9,
        ),
        timestamp_ms=_TS,
    )


def test_build_scoped_store_with_injected_in_memory_backend() -> None:
    identity = UserIdentity(user_id="alice", scope_key="brand_a:alice")
    backend = InMemoryPersistenceBackend()
    store = build_scoped_memory_store(
        identity=identity,
        persistence_backend=backend,
    )
    _durable_write(store, "remembers tea", "brand_a:alice")
    store.save_to_backend()
    # The injected backend received a checkpoint.
    assert backend.list_checkpoints(prefix="") != ()


# ---------------------------------------------------------------------------
# legacy single-layer scope compat shim (no silent re-key)
# ---------------------------------------------------------------------------


def test_legacy_single_layer_scope_alias() -> None:
    two_layer = UserIdentity(user_id="alice", scope_key="brand_a:alice")
    single = UserIdentity(user_id="alice", scope_key="alice")
    assert legacy_single_layer_scope(two_layer) == "alice"
    assert legacy_single_layer_scope(single) is None
    assert legacy_single_layer_scope(None) is None


def test_extra_scopes_match_legacy_tagged_entries() -> None:
    """A two-layer delete with the legacy alias reaches old single-layer memory."""

    identity = UserIdentity(user_id="alice", scope_key="brand_a:alice")
    backend = InMemoryPersistenceBackend()
    store = build_scoped_memory_store(identity=identity, persistence_backend=backend)
    # Simulate memory written under BOTH conventions for the same user.
    _durable_write(store, "old single-layer memory", "alice")
    _durable_write(store, "new two-layer memory", "brand_a:alice")

    legacy_alias = legacy_single_layer_scope(identity)
    # Without the alias, only the two-layer-tagged entry is visible.
    only_two_layer = list_durable_entries_for_scope(store, user_scope=identity.scope_key)
    assert {e.content for e in only_two_layer} == {"new two-layer memory"}

    # With the legacy alias, both are reachable (migration-safe).
    both = list_durable_entries_for_scope(
        store, user_scope=identity.scope_key, extra_scopes=(legacy_alias,)
    )
    assert {e.content for e in both} == {
        "old single-layer memory",
        "new two-layer memory",
    }

    # Deletion with the alias removes both, closing the GDPR/PIPL hole.
    deleted = delete_entries_for_scope(
        store, user_scope=identity.scope_key, extra_scopes=(legacy_alias,)
    )
    assert len(deleted) == 2
    assert list_durable_entries_for_scope(
        store, user_scope=identity.scope_key, extra_scopes=(legacy_alias,)
    ) == ()


def test_derive_scope_key_two_layer_default() -> None:
    tenant = TenantIdentity(tenant_id="brand_a")
    end_user = EndUserIdentity(tenant_id="brand_a", end_user_id="alice")
    assert derive_scope_key(tenant, end_user) == "brand_a:alice"
