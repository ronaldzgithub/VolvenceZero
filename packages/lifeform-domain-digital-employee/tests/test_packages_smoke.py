"""Smoke tests locking the digital-employee domain-package contract.

These run without a substrate runtime (data-only assertions), so they
execute in any environment that has the vz-application contracts wheel,
no torch / GPU required.
"""

from __future__ import annotations

from lifeform_domain_digital_employee import (
    build_digital_employee_org_package,
    build_digital_employee_twin_package,
)


def test_org_and_twin_packages_are_distinct_and_well_formed() -> None:
    org = build_digital_employee_org_package()
    twin = build_digital_employee_twin_package()

    # Distinct identities — org must not be byte-equal to twin (the whole
    # point of closing D18 is that the two roles differ in data).
    assert org.manifest.package_id != twin.manifest.package_id
    assert org.manifest.package_id == "lifeform-digital-employee-org-v0"
    assert twin.manifest.package_id == "lifeform-digital-employee-twin-v0"

    for pkg in (org, twin):
        assert pkg.knowledge_records, "package must ship knowledge records"
        assert pkg.case_records, "package must ship case records"
        assert pkg.playbook_rules, "package must ship playbook rules"
        assert pkg.boundary_hints, "package must ship boundary hints"
        assert pkg.manifest.owner == "lifeform-domain-digital-employee"


def test_role_regimes_reflect_org_vs_twin_specialisation() -> None:
    org = build_digital_employee_org_package()
    twin = build_digital_employee_twin_package()

    org_regimes = {r.recommended_regime for r in org.playbook_rules}
    twin_regimes = {r.recommended_regime for r in twin.playbook_rules}

    # Org leans coordination / compliance; twin leans execution / escalation.
    assert "compliance_guard" in org_regimes
    assert "work_intake_triage" in org_regimes
    assert "task_execution" in twin_regimes
    assert "escalation_to_human" in twin_regimes

    # The two role priors must not be identical regime sets.
    assert org_regimes != twin_regimes


def test_org_requires_human_gate_on_irreversible_actions() -> None:
    org = build_digital_employee_org_package()
    gate = next(
        (h for h in org.boundary_hints if h.regime_id == "compliance_guard"),
        None,
    )
    assert gate is not None
    assert gate.refer_out_required is True
    assert "human-approval-required" in gate.required_disclaimers


def test_twin_escalates_beyond_authority() -> None:
    twin = build_digital_employee_twin_package()
    gate = next(
        (h for h in twin.boundary_hints if h.regime_id == "escalation_to_human"),
        None,
    )
    assert gate is not None
    assert gate.refer_out_required is True


def test_base_packages_compile_to_existing_owners() -> None:
    from volvence_zero.application import compile_domain_experience_package

    for pkg in (
        build_digital_employee_org_package(),
        build_digital_employee_twin_package(),
    ):
        compiled = compile_domain_experience_package(pkg)
        assert compiled.validation_report.valid
        update = compiled.application_prior_update
        assert update.domain_knowledge_updates
        assert update.case_memory_updates
        assert update.strategy_playbook_updates
        assert update.boundary_policy_updates
