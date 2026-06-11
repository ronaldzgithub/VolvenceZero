"""Industry parameterisation for the digital-employee vertical.

The generic org / twin packs (:mod:`org_pack` / :mod:`twin_pack`) carry
the role persona that every digital employee shares. Industry
specialisation (sales SDR, customer support, content editing, …) is an
**additive data overlay** on top of that base:

* an :class:`IndustryProfile` is a frozen, reviewed bundle of extra
  ``vz-application`` records (knowledge / cases / playbook / boundary);
* :func:`build_industry_package` composes ``base + overlay`` into a new
  ``DomainExperiencePackage`` that still compiles into the same four
  kernel application owners — no new owner, no new snapshot slot;
* industry differences are expressed exclusively through record *data*
  (``applicability_scope`` tags such as ``industry:sales-sdr``, regime
  ids, ordering priors). There is no keyword→behaviour map and no text
  matching anywhere on this path (``no-keyword-matching-hacks.mdc``).

Adding a new industry therefore means adding one data module under
:mod:`lifeform_domain_digital_employee.profiles` — never a code branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from volvence_zero.application import (
    BoundaryPriorHint,
    CaseMemoryRecord,
    DomainExperienceManifest,
    DomainExperiencePackage,
    DomainKnowledgeRecord,
    PlaybookRule,
)

from lifeform_domain_digital_employee.org_pack import (
    build_digital_employee_org_package,
)
from lifeform_domain_digital_employee.twin_pack import (
    build_digital_employee_twin_package,
)

Role = Literal["org", "twin"]


@dataclass(frozen=True)
class IndustryProfile:
    """Reviewed industry overlay for the digital-employee vertical.

    Everything in here is data compiled into the existing application
    owners; the profile never carries behaviour. Records apply to both
    the org and the twin role — role differentiation already lives in
    the base packs, industry differentiation lives here, and the two
    axes compose in :func:`build_industry_package`.
    """

    industry_id: str
    display_name: str
    description: str
    domain_ids: tuple[str, ...]
    target_contexts: tuple[str, ...] = ()
    knowledge_records: tuple[DomainKnowledgeRecord, ...] = ()
    case_records: tuple[CaseMemoryRecord, ...] = ()
    playbook_rules: tuple[PlaybookRule, ...] = ()
    boundary_hints: tuple[BoundaryPriorHint, ...] = ()

    def __post_init__(self) -> None:
        if not self.industry_id.strip():
            raise ValueError("IndustryProfile.industry_id must be non-empty")
        if not self.display_name.strip():
            raise ValueError("IndustryProfile.display_name must be non-empty")
        if not self.domain_ids:
            raise ValueError(
                f"IndustryProfile {self.industry_id!r} must declare at least "
                "one domain_id"
            )
        if not (
            self.knowledge_records
            or self.case_records
            or self.playbook_rules
            or self.boundary_hints
        ):
            raise ValueError(
                f"IndustryProfile {self.industry_id!r} carries no records; "
                "an empty overlay is a contract violation, use the base "
                "package instead"
            )
        _check_unique(
            f"IndustryProfile {self.industry_id!r} knowledge record_id",
            tuple(record.record_id for record in self.knowledge_records),
        )
        _check_unique(
            f"IndustryProfile {self.industry_id!r} case_id",
            tuple(record.case_id for record in self.case_records),
        )
        _check_unique(
            f"IndustryProfile {self.industry_id!r} rule_id",
            tuple(rule.rule_id for rule in self.playbook_rules),
        )
        _check_unique(
            f"IndustryProfile {self.industry_id!r} hint_id",
            tuple(hint.hint_id for hint in self.boundary_hints),
        )


def build_industry_package(
    profile: IndustryProfile,
    *,
    role: Role,
) -> DomainExperiencePackage:
    """Compose ``base(role) + industry overlay`` into one package.

    The overlay is strictly additive: every base record (including the
    human-gate / finance-tax boundary hints) survives unchanged, and an
    id collision between overlay and base fails loudly instead of
    silently shadowing a reviewed base record.
    """

    if role == "org":
        base = build_digital_employee_org_package()
    elif role == "twin":
        base = build_digital_employee_twin_package()
    else:  # pragma: no cover - typed Literal guards this at call sites
        raise ValueError(f"unknown digital-employee role: {role!r}")

    _require_disjoint(
        profile,
        "knowledge record_id",
        tuple(r.record_id for r in base.knowledge_records),
        tuple(r.record_id for r in profile.knowledge_records),
    )
    _require_disjoint(
        profile,
        "case_id",
        tuple(r.case_id for r in base.case_records),
        tuple(r.case_id for r in profile.case_records),
    )
    _require_disjoint(
        profile,
        "rule_id",
        tuple(r.rule_id for r in base.playbook_rules),
        tuple(r.rule_id for r in profile.playbook_rules),
    )
    _require_disjoint(
        profile,
        "hint_id",
        tuple(r.hint_id for r in base.boundary_hints),
        tuple(r.hint_id for r in profile.boundary_hints),
    )

    base_manifest = base.manifest
    package_id = f"{base_manifest.package_id}+{profile.industry_id}"
    return DomainExperiencePackage(
        manifest=DomainExperienceManifest(
            package_id=package_id,
            version=base_manifest.version,
            display_name=(
                f"{base_manifest.display_name} ({profile.display_name})"
            ),
            domain_ids=_merge_unique(base_manifest.domain_ids, profile.domain_ids),
            target_contexts=_merge_unique(
                base_manifest.target_contexts, profile.target_contexts
            ),
            evidence_level=base_manifest.evidence_level,
            owner=base_manifest.owner,
            description=(
                f"{base_manifest.description} Industry overlay "
                f"{profile.industry_id!r}: {profile.description}"
            ),
        ),
        knowledge_records=base.knowledge_records + profile.knowledge_records,
        case_records=base.case_records + profile.case_records,
        playbook_rules=base.playbook_rules + profile.playbook_rules,
        boundary_hints=base.boundary_hints + profile.boundary_hints,
    )


def _merge_unique(
    base: tuple[str, ...], extra: tuple[str, ...]
) -> tuple[str, ...]:
    merged = list(base)
    for value in extra:
        if value not in merged:
            merged.append(value)
    return tuple(merged)


def _check_unique(label: str, values: tuple[str, ...]) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{label} values must be unique, got {values!r}")


def _require_disjoint(
    profile: IndustryProfile,
    label: str,
    base_ids: tuple[str, ...],
    overlay_ids: tuple[str, ...],
) -> None:
    collisions = sorted(set(base_ids) & set(overlay_ids))
    if collisions:
        raise ValueError(
            f"IndustryProfile {profile.industry_id!r} {label} collides with "
            f"the base digital-employee package: {collisions!r}; industry "
            "overlays must be additive, never shadow reviewed base records"
        )


__all__ = [
    "IndustryProfile",
    "build_industry_package",
]
