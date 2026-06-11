"""Discovery contract tests for the digital-employee verticals (D18).

Validates that ``discover_verticals()`` enumerates the two runtime
template names the digital-employee BFF adopts against —
``digital-employee.org.v0`` and ``digital-employee.twin.v0`` — and that
the resolution breadcrumb correctly reflects:

* ``dedicated`` when ``lifeform-domain-digital-employee`` is installed;
* ``companion-fallback`` when the operator pins the documented rollback
  env ``VZ_DIGITAL_EMPLOYEE_FORCE_COMPANION=1``.

Discovery-only assertions — no factory invocation, no torch.
"""

from __future__ import annotations

from lifeform_service import discover_verticals

_DIGITAL_EMPLOYEE_NAMES = (
    "digital-employee.org.v0",
    "digital-employee.twin.v0",
)


def test_digital_employee_verticals_are_discovered(monkeypatch) -> None:
    monkeypatch.delenv("VZ_DIGITAL_EMPLOYEE_FORCE_COMPANION", raising=False)
    specs = discover_verticals()
    for name in _DIGITAL_EMPLOYEE_NAMES:
        assert name in specs, f"missing digital-employee vertical {name!r}"
        assert specs[name].name == name
        assert callable(specs[name].factory)
        assert specs[name].alpha_factory is not None


def test_digital_employee_resolves_dedicated_when_wheel_installed(
    monkeypatch, capfd
) -> None:
    # The wheel is installed in this workspace, so the breadcrumb must
    # report the dedicated factory for both roles.
    import lifeform_domain_digital_employee  # noqa: F401

    monkeypatch.delenv("VZ_DIGITAL_EMPLOYEE_FORCE_COMPANION", raising=False)
    discover_verticals()
    err = capfd.readouterr().err
    for name in _DIGITAL_EMPLOYEE_NAMES:
        assert f"{name} resolution=dedicated reason=wheel_installed" in err


def test_force_companion_env_pin_falls_back_loudly(
    monkeypatch, capfd
) -> None:
    monkeypatch.setenv("VZ_DIGITAL_EMPLOYEE_FORCE_COMPANION", "1")
    specs = discover_verticals()
    err = capfd.readouterr().err
    for name in _DIGITAL_EMPLOYEE_NAMES:
        # The vertical names stay resolvable (rolling-upgrade safe) …
        assert name in specs
        # … but the rollback pin is never silent.
        assert f"{name} resolution=companion-fallback reason=forced_by_env" in err
