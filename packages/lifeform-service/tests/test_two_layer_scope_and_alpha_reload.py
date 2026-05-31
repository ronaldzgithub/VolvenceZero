"""D3/D22 + D6: two-layer-scope default, tenant binding, alpha hot-reload.

* AlphaIdentityProvider binds two-layer ``{tenant}:{end_user}`` scope by
  default (debt #46), with an explicit single-layer legacy alias.
* SessionManager._two_layer_scope_enabled is ON by default for the
  ``tenant_ai_end_user`` strategy, OFF for the empty (closed-alpha)
  strategy, and opt-out-able via env (no silent re-key).
* AlphaIdentityProvider allow-list is hot-reloadable (D6) without
  dropping live sessions; AlphaServiceConfig.with_alpha_users keeps the
  frozen config in sync.
"""

from __future__ import annotations

import json

import pytest

from lifeform_service.alpha import (
    DEFAULT_ALPHA_TENANT_ID,
    AlphaIdentityProvider,
    AlphaServiceConfig,
    load_alpha_users,
)
from lifeform_service.session_manager import (
    SessionManager,
    _legacy_single_layer_scope_opt_out,
)


# ---------------------------------------------------------------------------
# AlphaIdentityProvider: two-layer default + legacy alias
# ---------------------------------------------------------------------------


def test_bind_session_is_two_layer_by_default() -> None:
    provider = AlphaIdentityProvider()
    identity = provider.bind_session(session_id="s1", end_user_id="alice")
    assert identity.scope_key == f"{DEFAULT_ALPHA_TENANT_ID}:alice"
    assert identity.tenant_identity is not None
    assert identity.end_user_identity is not None
    assert identity.user_id == "alice"


def test_bind_session_two_layer_with_explicit_tenant() -> None:
    provider = AlphaIdentityProvider()
    identity = provider.bind_session(
        session_id="s2", end_user_id="alice", tenant_id="brand_a"
    )
    assert identity.scope_key == "brand_a:alice"


def test_bind_session_legacy_alias_is_single_layer() -> None:
    provider = AlphaIdentityProvider()
    identity = provider.bind_session_legacy_alias(session_id="s3", user_id="alice")
    assert identity.scope_key == "alice"
    assert identity.tenant_identity is None


def test_bind_session_accepts_legacy_positional_user_id() -> None:
    # Back-compat: callers that pass user_id= still get two-layer scope.
    provider = AlphaIdentityProvider()
    identity = provider.bind_session(session_id="s4", user_id="bob")
    assert identity.scope_key == f"{DEFAULT_ALPHA_TENANT_ID}:bob"


def test_bind_session_enforces_allow_list() -> None:
    provider = AlphaIdentityProvider(allowed_users=frozenset({"alice"}))
    provider.bind_session(session_id="ok", end_user_id="alice")
    with pytest.raises(PermissionError):
        provider.bind_session(session_id="no", end_user_id="mallory")


# ---------------------------------------------------------------------------
# SessionManager two-layer gate (default ON, opt-out)
# ---------------------------------------------------------------------------


def _gate(scope_strategy: str, tenant_id: str = "") -> bool:
    # Exercise the gate logic without full SessionManager construction.
    mgr = SessionManager.__new__(SessionManager)
    mgr._scope_strategy = scope_strategy  # noqa: SLF001
    mgr._tenant_id = tenant_id  # noqa: SLF001
    return mgr._two_layer_scope_enabled()  # noqa: SLF001


def test_two_layer_default_on_for_tenant_strategy(monkeypatch) -> None:
    monkeypatch.delenv("VZ_TWO_LAYER_SCOPE", raising=False)
    monkeypatch.delenv("VZ_LEGACY_SINGLE_LAYER_SCOPE", raising=False)
    assert _gate("tenant_ai_end_user") is True


def test_two_layer_off_for_empty_strategy(monkeypatch) -> None:
    # The standalone closed-alpha service uses no scope_strategy and must
    # stay single-layer so its on-disk memory is never re-keyed.
    monkeypatch.delenv("VZ_TWO_LAYER_SCOPE", raising=False)
    monkeypatch.delenv("VZ_LEGACY_SINGLE_LAYER_SCOPE", raising=False)
    assert _gate("") is False


def test_two_layer_opt_out_via_legacy_env(monkeypatch) -> None:
    monkeypatch.delenv("VZ_TWO_LAYER_SCOPE", raising=False)
    monkeypatch.setenv("VZ_LEGACY_SINGLE_LAYER_SCOPE", "1")
    assert _gate("tenant_ai_end_user") is False


def test_two_layer_opt_out_via_negated_flag(monkeypatch) -> None:
    monkeypatch.delenv("VZ_LEGACY_SINGLE_LAYER_SCOPE", raising=False)
    monkeypatch.setenv("VZ_TWO_LAYER_SCOPE", "0")
    assert _gate("tenant_ai_end_user") is False


def test_bare_two_layer_flag_does_not_opt_out(monkeypatch) -> None:
    # A truthy/legacy VZ_TWO_LAYER_SCOPE must NOT be required and must NOT
    # opt out — two-layer is the default now.
    monkeypatch.delenv("VZ_LEGACY_SINGLE_LAYER_SCOPE", raising=False)
    monkeypatch.setenv("VZ_TWO_LAYER_SCOPE", "1")
    assert _legacy_single_layer_scope_opt_out() is False
    assert _gate("tenant_ai_end_user") is True


# ---------------------------------------------------------------------------
# D6: alpha allow-list hot-reload
# ---------------------------------------------------------------------------


def _write_users(path, users) -> None:
    path.write_text(json.dumps(list(users)), encoding="utf-8")


def test_set_allowed_users_swaps_in_place() -> None:
    provider = AlphaIdentityProvider(allowed_users=frozenset({"alice"}))
    assert provider.allowed_users == frozenset({"alice"})
    provider.set_allowed_users(frozenset({"alice", "bob"}))
    assert provider.allowed_users == frozenset({"alice", "bob"})
    # New user now binds without error.
    provider.bind_session(session_id="s", end_user_id="bob")


def test_reload_from_file_picks_up_new_signins(tmp_path) -> None:
    users_file = tmp_path / "alpha_users.json"
    _write_users(users_file, ["alice"])
    provider = AlphaIdentityProvider(allowed_users=load_alpha_users(str(users_file)))
    with pytest.raises(PermissionError):
        provider.bind_session(session_id="pre", end_user_id="carol")

    # Ops appends a new sign-in and reloads — no restart.
    _write_users(users_file, ["alice", "carol"])
    refreshed = provider.reload_allowed_users_from_file(str(users_file))
    assert refreshed == frozenset({"alice", "carol"})
    provider.bind_session(session_id="post", end_user_id="carol")  # now allowed


def test_reload_from_none_path_is_noop() -> None:
    provider = AlphaIdentityProvider(allowed_users=frozenset({"alice"}))
    assert provider.reload_allowed_users_from_file(None) == frozenset({"alice"})


def test_reload_does_not_drop_live_sessions(tmp_path) -> None:
    users_file = tmp_path / "u.json"
    _write_users(users_file, ["alice", "bob"])
    provider = AlphaIdentityProvider(allowed_users=load_alpha_users(str(users_file)))
    provider.bind_session(session_id="live", end_user_id="bob")
    # bob removed from the allow-list, then reload.
    _write_users(users_file, ["alice"])
    provider.reload_allowed_users_from_file(str(users_file))
    # The already-bound session is intentionally preserved.
    assert provider.resolve("live") is not None


def test_config_with_alpha_users_returns_synced_copy() -> None:
    config = AlphaServiceConfig(enabled=True, alpha_users=frozenset({"alice"}))
    updated = config.with_alpha_users(frozenset({"alice", "bob"}))
    assert updated.alpha_users == frozenset({"alice", "bob"})
    assert updated.enabled is True
    # Original frozen instance is unchanged.
    assert config.alpha_users == frozenset({"alice"})
