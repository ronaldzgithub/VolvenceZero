from __future__ import annotations

import pytest


def _test_spec():
    from lifeform_core import Lifeform, LifeformConfig
    from lifeform_service.verticals import VerticalSpec

    return VerticalSpec(
        name="companion",
        factory=lambda _runtime: Lifeform(LifeformConfig()),
        has_temporal_bootstrap=False,
        has_regime_bootstrap=False,
    )


async def test_instance_manager_attaches_default_mcp_bundle_when_enabled(
    monkeypatch,
) -> None:
    from dlaas_platform_launcher.instance_manager import InstanceManager

    calls = []

    def fake_attach(lifeform):
        calls.append(lifeform)
        return lifeform

    monkeypatch.setattr(
        "lifeform_service.session_manager.with_default_mcp_bundle",
        fake_attach,
    )

    manager = InstanceManager(
        vertical_resolver=lambda _template: _test_spec(),
        attach_default_mcp_bundle=True,
    )
    await manager.wake(
        ai_id="de_twin_company_member",
        runtime_template_id="digital-employee.twin.v0",
        reason="test",
    )
    session_manager = manager.get("de_twin_company_member")
    await session_manager.create_session(session_id="de-session")

    assert len(calls) == 1


def test_default_resolver_aliases_digital_employee_templates() -> None:
    from dlaas_platform_launcher.instance_manager import default_vertical_resolver

    resolver = default_vertical_resolver()
    org = resolver("digital-employee.org.v0")
    twin = resolver("digital-employee.twin.v0")

    if org is None or twin is None:
        pytest.skip("companion vertical not installed in this checkout")

    assert org.name == "companion"
    assert twin.name == "companion"

