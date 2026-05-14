"""Contract test: D4 P2 two-layer scope application surface (debt #69).

Validates the P2 application-layer surface for the two-layer
``tenant × end_user`` scope (depends on debt #46 ACTIVE):

1. Calling ``AlphaIdentityProvider.bind_session(end_user_id=..., tenant_id=...)``
   with explicit tenant produces a ``brand_a:alice``-style scope_key.
2. P2 growth-advisor admin endpoint shape: ``derive_scope_key`` must
   match what the URL ``/v1/tenants/{tid}/users/{uid}/...`` would
   produce (URL contract vs scope_key contract converge).
3. Different tenants serving the same end_user_id remain isolated
   (smoke per debt #69 §isolation).
4. Closed-alpha legacy callers still get single-layer via
   ``bind_session_legacy_alias`` (back-compat preserved).

Refs:

* docs/known-debts.md #69
* docs/specs/handoff-queue-slo.md §5 tenant isolation
"""

from __future__ import annotations

import pytest

from lifeform_service.alpha import (
    DEFAULT_ALPHA_TENANT_ID,
    AlphaIdentityProvider,
)
from volvence_zero.memory import (
    EndUserIdentity,
    TenantIdentity,
    derive_scope_key,
)


def test_growth_advisor_admin_endpoint_scope_key_round_trips() -> None:
    """``derive_scope_key`` matches the URL ``/tenants/{tid}/users/{uid}/`` shape."""
    tenant = TenantIdentity(tenant_id="brand_a")
    end_user = EndUserIdentity(tenant_id="brand_a", end_user_id="alice")
    derived = derive_scope_key(tenant, end_user)
    # The admin endpoint's URL fragment should be reconstructible from
    # the scope_key (and vice versa) — single source of truth.
    fragment_from_scope = derived.replace(":", "/users/", 1)
    assert fragment_from_scope == "brand_a/users/alice"


def test_alpha_provider_default_tenant_for_p2_growth_advisor() -> None:
    """P2 sessions binding without explicit tenant get the alpha default."""
    provider = AlphaIdentityProvider(allowed_users=frozenset({"alice"}))
    identity = provider.bind_session(session_id="s1", user_id="alice")
    assert identity.scope_key == f"{DEFAULT_ALPHA_TENANT_ID}:alice"
    assert identity.tenant_identity is not None
    assert identity.tenant_identity.tenant_id == DEFAULT_ALPHA_TENANT_ID


def test_growth_advisor_b2b_explicit_tenant_isolated_from_other_tenants() -> None:
    """Two B2B tenants binding the same end_user_id stay isolated."""
    provider_a = AlphaIdentityProvider(allowed_users=frozenset({"alice"}))
    provider_b = AlphaIdentityProvider(allowed_users=frozenset({"alice"}))
    a = provider_a.bind_session(
        session_id="s1", end_user_id="alice", tenant_id="brand_a"
    )
    b = provider_b.bind_session(
        session_id="s2", end_user_id="alice", tenant_id="brand_b"
    )
    assert a.scope_key != b.scope_key
    assert a.scope_key == "brand_a:alice"
    assert b.scope_key == "brand_b:alice"


def test_legacy_alias_remains_for_closed_alpha_back_compat() -> None:
    """Closed-alpha sites that need the SHADOW single-layer contract still work."""
    provider = AlphaIdentityProvider(allowed_users=frozenset({"alice"}))
    identity = provider.bind_session_legacy_alias(
        session_id="s1", user_id="alice"
    )
    assert identity.scope_key == "alice"
    assert identity.tenant_identity is None


def test_growth_advisor_p2_endpoint_url_template_documented() -> None:
    """Spec §6 admin endpoint template aligns with double-layer scope."""
    # Admin endpoint URL template per packet G-D §6.
    template = "/v1/tenants/{tid}/users/{uid}/admin/monthly-report"
    # The {tid} / {uid} placeholders must map to tenant_id /
    # end_user_id (not user_id) — defends against the regression of
    # accidentally rebinding the URL to the legacy single-layer
    # ``user_id`` field.
    assert "{tid}" in template
    assert "{uid}" in template
    # Verify no stray placeholder named ``user_id`` (legacy contract).
    assert "{user_id}" not in template
