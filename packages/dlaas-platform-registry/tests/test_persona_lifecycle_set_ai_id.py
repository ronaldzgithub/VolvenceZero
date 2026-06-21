"""`PersonaLifecycleStore.set_ai_id` links an adopted ai_id onto the lifecycle.

This is the template->ai_id join the soul console relies on: bake creates
the lifecycle with an empty ai_id; adopt/wake mirror the live ai_id back so
the (operator-listable) lifecycle can attach a soul to its instance/health.
"""

from __future__ import annotations

import pytest

from dlaas_platform_registry import (
    PersonaLifecycleStore,
    Registry,
    TemplateStore,
    TenantStore,
)


async def _seed(registry: Registry) -> str:
    tenant = await TenantStore(registry).create(
        tenant_name="t", contact_email="t@example.test", tenant_id="tenant_x"
    )
    template = await TemplateStore(registry).create(
        tenant_id=tenant.tenant_id,
        template_name="Soul",
        template_id="tpl_author_cao_abc",
    )
    await PersonaLifecycleStore(registry).create(
        template_id=template.template_id, tenant_id=tenant.tenant_id
    )
    return template.template_id


@pytest.mark.asyncio
async def test_set_ai_id_links_and_is_idempotent() -> None:
    registry = Registry(db_path=":memory:")
    template_id = await _seed(registry)
    store = PersonaLifecycleStore(registry)

    # Before: empty ai_id.
    before = await store.get_by_template(template_id)
    assert before.ai_id == ""

    updated = await store.set_ai_id(template_id=template_id, ai_id="ai_cao")
    assert updated is True
    after = await store.get_by_template(template_id)
    assert after.ai_id == "ai_cao"

    # Idempotent re-link (latest wins).
    assert await store.set_ai_id(template_id=template_id, ai_id="ai_cao2") is True
    assert (await store.get_by_template(template_id)).ai_id == "ai_cao2"


@pytest.mark.asyncio
async def test_set_ai_id_no_row_is_noop() -> None:
    registry = Registry(db_path=":memory:")
    store = PersonaLifecycleStore(registry)
    # No lifecycle for this template -> fail-soft no-op, returns False.
    assert await store.set_ai_id(template_id="tpl_missing", ai_id="ai_x") is False


@pytest.mark.asyncio
async def test_set_ai_id_empty_is_noop() -> None:
    registry = Registry(db_path=":memory:")
    template_id = await _seed(registry)
    store = PersonaLifecycleStore(registry)
    assert await store.set_ai_id(template_id=template_id, ai_id="") is False
    assert (await store.get_by_template(template_id)).ai_id == ""
