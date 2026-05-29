"""Vertical: B2B digital employee (org agent + employee twin).

Public API:

* ``build_digital_employee_org_package`` — company-level OrgAgent
  ``DomainExperiencePackage`` (policy grounding / triage / delegation /
  compliance-guard regime priors).
* ``build_digital_employee_twin_package`` — per-employee EmployeeTwin
  ``DomainExperiencePackage`` (task execution / drafting / clarification /
  escalation regime priors).
* ``build_digital_employee_lifeform`` — convenience factory returning a
  ready-to-run ``Lifeform`` for either role.

Design posture (see also the repo rule ``first-principles-not-patches``):

The two roles differ only in *data* — their domain experience packages.
Both reuse the kernel's regime/credit/dual-track machinery and (v0) the
companion calibration basin. There are no prompt strings and no
keyword->behaviour maps here; regime names are runtime-state priors fed to
the playbook / boundary owners, not labels matched against text. This is
the documented per-vertical seam, so a new role or a future dedicated
super-loop bootstrap only changes this wheel, never the kernel.
"""

from __future__ import annotations

from lifeform_domain_digital_employee.builder import (
    build_digital_employee_lifeform,
)
from lifeform_domain_digital_employee.org_pack import (
    build_digital_employee_org_package,
)
from lifeform_domain_digital_employee.twin_pack import (
    build_digital_employee_twin_package,
)

__all__ = (
    "build_digital_employee_lifeform",
    "build_digital_employee_org_package",
    "build_digital_employee_twin_package",
)
