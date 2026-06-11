"""Industry-overlay contract tests for the digital-employee vertical.

Locks the W2 convergence-packet invariants:

* every built-in industry profile composes onto both roles and compiles
  through ``vz-application``'s canonical compiler;
* the compiled package maps to exactly the four existing application
  owners (domain_knowledge / case_memory / strategy_playbook /
  boundary_policy) — no new owner, no new slot;
* the overlay is strictly additive (base records, including the human
  escalation gates and the finance-tax refusal, all survive);
* id collisions and unknown industry ids fail loudly.

Data-only assertions — no substrate runtime, no torch.
"""

from __future__ import annotations

import dataclasses

import pytest

from volvence_zero.application import compile_domain_experience_package

from lifeform_domain_digital_employee import (
    IndustryProfile,
    build_digital_employee_org_package,
    build_digital_employee_twin_package,
    build_industry_package,
    builtin_industry_profiles,
    industry_profile_by_id,
)

_ROLES = ("org", "twin")
_BUILTIN_INDUSTRY_IDS = ("sales-sdr", "customer-support", "content-editor")


def _base_package(role: str):
    if role == "org":
        return build_digital_employee_org_package()
    return build_digital_employee_twin_package()


def test_builtin_profile_registry_is_complete() -> None:
    profiles = builtin_industry_profiles()
    assert tuple(p.industry_id for p in profiles) == _BUILTIN_INDUSTRY_IDS
    for industry_id in _BUILTIN_INDUSTRY_IDS:
        assert industry_profile_by_id(industry_id).industry_id == industry_id


def test_unknown_industry_id_fails_loudly() -> None:
    with pytest.raises(KeyError, match="unknown digital-employee industry_id"):
        industry_profile_by_id("does-not-exist")


@pytest.mark.parametrize("role", _ROLES)
@pytest.mark.parametrize("industry_id", _BUILTIN_INDUSTRY_IDS)
def test_industry_packages_compile_to_existing_owners(
    role: str, industry_id: str
) -> None:
    profile = industry_profile_by_id(industry_id)
    package = build_industry_package(profile, role=role)
    compiled = compile_domain_experience_package(package)
    assert compiled.validation_report.valid

    update = compiled.application_prior_update
    # The four existing application owners — and only those — receive data.
    assert update.domain_knowledge_updates
    assert update.case_memory_updates
    assert update.strategy_playbook_updates
    assert update.boundary_policy_updates
    package_id = package.manifest.package_id
    for item in update.domain_knowledge_updates:
        assert f"domain_experience.{package_id}.domain_knowledge." in item.target
    for item in update.case_memory_updates:
        assert f"domain_experience.{package_id}.case_memory." in item.target
    for item in update.strategy_playbook_updates:
        assert f"domain_experience.{package_id}.strategy_playbook." in item.target
    for item in update.boundary_policy_updates:
        assert f"domain_experience.{package_id}.boundary_policy." in item.target


@pytest.mark.parametrize("role", _ROLES)
@pytest.mark.parametrize("industry_id", _BUILTIN_INDUSTRY_IDS)
def test_industry_overlay_is_strictly_additive(
    role: str, industry_id: str
) -> None:
    base = _base_package(role)
    profile = industry_profile_by_id(industry_id)
    package = build_industry_package(profile, role=role)

    assert package.manifest.package_id == (
        f"{base.manifest.package_id}+{industry_id}"
    )
    # Base records all survive, overlay records are appended.
    assert set(r.record_id for r in base.knowledge_records) <= set(
        r.record_id for r in package.knowledge_records
    )
    assert set(r.case_id for r in base.case_records) <= set(
        r.case_id for r in package.case_records
    )
    assert set(r.rule_id for r in base.playbook_rules) <= set(
        r.rule_id for r in package.playbook_rules
    )
    assert set(h.hint_id for h in base.boundary_hints) <= set(
        h.hint_id for h in package.boundary_hints
    )
    assert set(r.record_id for r in profile.knowledge_records) <= set(
        r.record_id for r in package.knowledge_records
    )
    assert set(r.rule_id for r in profile.playbook_rules) <= set(
        r.rule_id for r in package.playbook_rules
    )
    # Industry domains are appended to the manifest.
    for domain_id in profile.domain_ids:
        assert domain_id in package.manifest.domain_ids


@pytest.mark.parametrize("role", _ROLES)
@pytest.mark.parametrize("industry_id", _BUILTIN_INDUSTRY_IDS)
def test_escalation_and_refusal_gates_survive_overlay(
    role: str, industry_id: str
) -> None:
    package = build_industry_package(
        industry_profile_by_id(industry_id), role=role
    )
    hints_by_id = {h.hint_id: h for h in package.boundary_hints}

    # Base human gate on irreversible / external-spend / external-publish.
    gate_id = (
        "rid-de-org:boundary:irreversible-needs-human"
        if role == "org"
        else "rid-de-twin:boundary:authority-limit"
    )
    gate = hints_by_id[gate_id]
    assert gate.refer_out_required is True
    assert "external-publish" in gate.trigger_reasons

    # Base finance / tax refusal.
    refusal_id = f"rid-de-{role}:boundary:finance-tax-refusal"
    refusal = hints_by_id[refusal_id]
    assert refusal.refer_out_required is True
    assert "tax-filing-advice" in refusal.blocked_topics
    assert "refer-to-licensed-professional" in refusal.required_disclaimers

    # Every built-in industry overlay ships at least one escalation hint
    # of its own (refer_out_required gate scoped to the industry).
    industry_hints = [
        h for h in package.boundary_hints if h.hint_id.startswith("rid-de-ind-")
    ]
    assert industry_hints
    assert any(h.refer_out_required for h in industry_hints)


def test_industry_playbooks_carry_industry_scope_tags() -> None:
    for profile in builtin_industry_profiles():
        scope_tag = f"industry:{profile.industry_id}"
        assert profile.playbook_rules
        for rule in profile.playbook_rules:
            assert scope_tag in rule.applicability_scope, (
                f"playbook rule {rule.rule_id} must carry {scope_tag}; "
                "industry routing is scope-tag data, not keyword matching"
            )


def test_id_collision_with_base_fails_loudly() -> None:
    base = build_digital_employee_twin_package()
    colliding_record = dataclasses.replace(
        builtin_industry_profiles()[0].knowledge_records[0],
        record_id=base.knowledge_records[0].record_id,
    )
    profile = IndustryProfile(
        industry_id="collision-test",
        display_name="Collision Test",
        description="overlay that collides with a base record id",
        domain_ids=("collision_domain",),
        knowledge_records=(colliding_record,),
    )
    with pytest.raises(ValueError, match="collides with"):
        build_industry_package(profile, role="twin")


def test_empty_industry_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="carries no records"):
        IndustryProfile(
            industry_id="empty-test",
            display_name="Empty Test",
            description="no records",
            domain_ids=("some_domain",),
        )
