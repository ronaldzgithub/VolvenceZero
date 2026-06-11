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


def test_default_resolver_resolves_digital_employee_first_class() -> None:
    """D18: the launcher resolves org/twin by exact first-class name.

    The historical companion alias in ``default_vertical_resolver`` is
    back-compat only; healthy installs must hit the registered
    ``digital-employee.{org,twin}.v0`` VerticalSpec, not the alias.
    """

    from dlaas_platform_launcher.instance_manager import default_vertical_resolver

    resolver = default_vertical_resolver()
    org = resolver("digital-employee.org.v0")
    twin = resolver("digital-employee.twin.v0")

    if org is None or twin is None:
        pytest.skip("companion vertical not installed in this checkout")

    assert org.name == "digital-employee.org.v0"
    assert twin.name == "digital-employee.twin.v0"


def test_digital_employee_verticals_use_dedicated_domain_packs() -> None:
    """D18: org/twin factories compile the role-specialised packs.

    With ``lifeform-domain-digital-employee`` installed the org / twin
    verticals must NOT be the companion factory in disguise — each
    role's lifeform config carries its dedicated data-only
    ``DomainExperiencePackage``.
    """

    pytest.importorskip("lifeform_domain_digital_employee")
    from lifeform_service.verticals import discover_verticals

    verticals = discover_verticals()
    expected = {
        "digital-employee.org.v0": "lifeform-digital-employee-org-v0",
        "digital-employee.twin.v0": "lifeform-digital-employee-twin-v0",
    }
    for vertical_name, package_id in expected.items():
        spec = verticals.get(vertical_name)
        if spec is None:
            pytest.skip("companion vertical not installed in this checkout")
        life = spec.factory(None)
        package_ids = {
            pkg.manifest.package_id
            for pkg in life.config.brain_config.domain_experience_packages
        }
        assert package_id in package_ids, (
            f"{vertical_name} must compile {package_id}; got {package_ids}"
        )


def test_force_companion_env_pins_digital_employee_fallback(
    monkeypatch,
) -> None:
    """D18 rollback: VZ_DIGITAL_EMPLOYEE_FORCE_COMPANION pins companion.

    The pin must restore companion behaviour (no digital-employee
    domain pack) even with the specialised wheel installed.
    """

    pytest.importorskip("lifeform_domain_digital_employee")
    from lifeform_service.verticals import discover_verticals

    monkeypatch.setenv("VZ_DIGITAL_EMPLOYEE_FORCE_COMPANION", "1")
    verticals = discover_verticals()
    spec = verticals.get("digital-employee.org.v0")
    if spec is None:
        pytest.skip("companion vertical not installed in this checkout")
    life = spec.factory(None)
    package_ids = {
        pkg.manifest.package_id
        for pkg in life.config.brain_config.domain_experience_packages
    }
    assert "lifeform-digital-employee-org-v0" not in package_ids

