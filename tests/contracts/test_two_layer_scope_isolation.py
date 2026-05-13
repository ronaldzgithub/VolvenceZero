"""Contract tests for the two-layer scope schema (debt #46).

Validates the SHADOW surface land in
:mod:`volvence_zero.memory.identity` + ``lifeform_service.alpha``:

1. ``TenantIdentity`` / ``EndUserIdentity`` reject empty fields
2. ``derive_scope_key`` produces ``"<tenant>:<end_user>"`` and is the
   SSOT (consumers must not re-derive)
3. Same ``end_user_id`` under different tenants → distinct scope_key
4. Mismatched ``tenant_identity`` / ``end_user_identity`` raises
5. ``UserIdentity`` two-layer fields stay backward-compatible (None
   defaults; existing ``user_id == scope_key`` callers unaffected)
6. ``AlphaIdentityProvider.bind_session`` legacy single-layer
   behaviour preserved
7. ``AlphaIdentityProvider.bind_session_two_layer`` opt-in surface
   yields properly two-layer-derived ``UserIdentity``

See:

* ``docs/moving forward/cross-cutting-foundation-packet.md`` §2.2
* ``docs/known-debts.md`` #46 / #69
"""

from __future__ import annotations

import pytest

from lifeform_service.alpha import AlphaIdentityProvider
from volvence_zero.memory import (
    EndUserIdentity,
    TenantIdentity,
    UserIdentity,
    derive_scope_key,
)


def test_tenant_identity_rejects_empty() -> None:
    with pytest.raises(ValueError, match="tenant_id must be non-empty"):
        TenantIdentity(tenant_id="")
    with pytest.raises(ValueError, match="tenant_id must be non-empty"):
        TenantIdentity(tenant_id="   ")


def test_end_user_identity_rejects_empty() -> None:
    with pytest.raises(ValueError, match="tenant_id must be non-empty"):
        EndUserIdentity(tenant_id="", end_user_id="alice")
    with pytest.raises(ValueError, match="end_user_id must be non-empty"):
        EndUserIdentity(tenant_id="brand_a", end_user_id="")


def test_derive_scope_key_two_layer_format() -> None:
    tenant = TenantIdentity(tenant_id="brand_a")
    end_user = EndUserIdentity(tenant_id="brand_a", end_user_id="alice")
    assert derive_scope_key(tenant, end_user) == "brand_a:alice"


def test_derive_scope_key_same_end_user_different_tenants() -> None:
    """Same end_user_id under different tenants → distinct scope_key."""
    a = derive_scope_key(
        TenantIdentity(tenant_id="brand_a"),
        EndUserIdentity(tenant_id="brand_a", end_user_id="alice"),
    )
    b = derive_scope_key(
        TenantIdentity(tenant_id="brand_b"),
        EndUserIdentity(tenant_id="brand_b", end_user_id="alice"),
    )
    assert a != b
    assert a == "brand_a:alice"
    assert b == "brand_b:alice"


def test_derive_scope_key_mismatched_tenant_raises() -> None:
    with pytest.raises(ValueError, match="does not match"):
        derive_scope_key(
            TenantIdentity(tenant_id="brand_a"),
            EndUserIdentity(tenant_id="brand_b", end_user_id="alice"),
        )


def test_user_identity_two_layer_fields_default_none() -> None:
    """Backward compat: existing single-layer callers see ``None`` defaults."""
    identity = UserIdentity(user_id="alice", scope_key="alice")
    assert identity.tenant_identity is None
    assert identity.end_user_identity is None


def test_user_identity_two_layer_consistency_check() -> None:
    """When both layers supplied, scope_key must equal derive_scope_key()."""
    tenant = TenantIdentity(tenant_id="brand_a")
    end_user = EndUserIdentity(tenant_id="brand_a", end_user_id="alice")
    # Correct derivation
    UserIdentity(
        user_id="alice",
        scope_key="brand_a:alice",
        tenant_identity=tenant,
        end_user_identity=end_user,
    )
    # Mismatched scope_key fails
    with pytest.raises(ValueError, match="does not match derive_scope_key"):
        UserIdentity(
            user_id="alice",
            scope_key="alice",  # legacy format, not consistent with derived
            tenant_identity=tenant,
            end_user_identity=end_user,
        )


def test_alpha_provider_legacy_single_layer_unchanged() -> None:
    """Closed-alpha legacy ``bind_session`` keeps ``scope_key == user_id``."""
    provider = AlphaIdentityProvider(allowed_users=frozenset({"alice"}))
    identity = provider.bind_session(session_id="s1", user_id="alice")
    assert identity.user_id == "alice"
    assert identity.scope_key == "alice"
    assert identity.tenant_identity is None
    assert identity.end_user_identity is None


def test_alpha_provider_two_layer_opt_in() -> None:
    """Opt-in two-layer derives scope_key + populates typed identities."""
    provider = AlphaIdentityProvider(allowed_users=frozenset({"alice"}))
    identity = provider.bind_session_two_layer(
        session_id="s2",
        end_user_id="alice",
    )
    assert identity.user_id == "alice"
    assert identity.scope_key == "alpha:alice"
    assert identity.tenant_identity is not None
    assert identity.tenant_identity.tenant_id == "alpha"
    assert identity.end_user_identity is not None
    assert identity.end_user_identity.end_user_id == "alice"


def test_alpha_provider_two_layer_custom_tenant() -> None:
    """B2B tenant_id flows through correctly."""
    provider = AlphaIdentityProvider(allowed_users=frozenset({"alice"}))
    identity = provider.bind_session_two_layer(
        session_id="s3",
        end_user_id="alice",
        tenant_id="brand_a",
    )
    assert identity.scope_key == "brand_a:alice"
    assert identity.tenant_identity is not None
    assert identity.tenant_identity.tenant_id == "brand_a"
