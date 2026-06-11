# lifeform-domain-digital-employee

Vertical for the **B2B digital employee** product: one DLaaS tenant per
company, a company-level **OrgAgent** plus per-member **EmployeeTwins**.

A vertical is **data + light glue** that compiles into the kernel's
existing owners — this wheel adds no kernel owner and no snapshot slot:

| Asset | Compiles into kernel surface |
|---|---|
| Org / twin `DomainExperiencePackage` (knowledge / cases / playbooks / boundary hints) | `vz-application.domain_knowledge / case_memory / strategy_playbook / boundary_policy` |
| `IndustryProfile` overlays (`profiles/`) | same four owners, composed additively onto the role base |
| Calibration (v0) | reuses the companion bootstraps (`lifeform-domain-emogpt`) as the warm start |

## The two roles

| Runtime template | Role | Posture |
|---|---|---|
| `digital-employee.org.v0` | company OrgAgent | coordination-first: SOP grounding, intake triage, delegation briefs, compliance guard |
| `digital-employee.twin.v0` | per-member EmployeeTwin | execution-first: brief-aligned task execution, drafting, bounded clarification, escalation to a human |

Both names are registered through `lifeform-service.verticals`
(`_try_digital_employee_org` / `_try_digital_employee_twin`). When this
wheel is absent — or the operator pins
`VZ_DIGITAL_EMPLOYEE_FORCE_COMPANION=1` (the documented D18 rollback) —
the names keep resolving via the companion factory, and either branch
logs a `[verticals] … resolution=…` stderr breadcrumb so the wiring is
never silent.

## Public API

```python
from lifeform_domain_digital_employee import (
    build_digital_employee_org_package,   # OrgAgent DomainExperiencePackage
    build_digital_employee_twin_package,  # EmployeeTwin DomainExperiencePackage
    IndustryProfile,                      # frozen industry overlay schema
    build_industry_package,               # base(role) + industry overlay
    builtin_industry_profiles,            # shipped overlays (data registry)
    industry_profile_by_id,               # exact-id lookup, fail-loud
    build_digital_employee_lifeform,      # ready-to-run Lifeform (org/twin)
)
```

## Industry parameterisation (data-driven)

Role差异 (org vs twin) 和行业差异 (SDR vs 客服 vs 内容编辑) 是两条正交的数据轴：

```python
from lifeform_domain_digital_employee import (
    build_digital_employee_lifeform,
    industry_profile_by_id,
)

life = build_digital_employee_lifeform(
    role="twin",
    industry_profile=industry_profile_by_id("sales-sdr"),
)
```

Built-in profiles: `sales-sdr`, `customer-support`, `content-editor`.
All industry differences are expressed as records — `applicability_scope`
tags like `industry:sales-sdr`, regime ids, intervention orderings —
consumed by the existing playbook / boundary owners. There is no
keyword→behaviour map anywhere on this path; adding an industry means
adding one data module under `profiles/` and listing its builder in
`BUILTIN_INDUSTRY_PROFILE_BUILDERS`.

Overlays are **strictly additive**: every base record survives, id
collisions with the base fail loudly, and the base boundary gates
(human approval for irreversible / external-spend / external-publish
actions; refusal + licensed-professional referral for finance/tax
advice) cannot be removed by an industry profile.

## Boundary posture

* OrgAgent: propose-don't-execute for irreversible work, never absorbs a
  person's twin-scoped memory, flags SOP gaps instead of inventing policy.
* EmployeeTwin: escalation at the edge of authority or confidence is a
  success path; tool side effects are verified before and after; scoped
  memory never leaks across employees.

## Tenant data

The wheel ships **no tenant data**. Per-company SOP / brand / playbook
corpora arrive at runtime via the BFF's `observe` envelopes; a member's
working habits live in their `membership_id`-scoped memory (R14).

## Calibration (v0)

`build_digital_employee_lifeform` reuses the companion calibration basin
(vitals + temporal + regime bootstraps from `lifeform-domain-emogpt`) as
the warm start. Once a dedicated digital-employee super-loop produces
org/twin-specific bootstraps, only the `_load_*_bootstrap` calls in
`builder.py` change.
