"""Coding vertical affordances (Gap 1 slice 2b).

Public API:

* ``CODING_AFFORDANCE_DESCRIPTORS`` \u2014 tuple of
  ``AffordanceDescriptor`` ready to register
* ``build_coding_affordance_registry(sandbox_root)`` \u2014 returns a
  sealed ``AffordanceRegistry`` with the coding descriptors
* ``build_coding_affordance_invoker(sandbox_root)`` \u2014 returns a
  ready-to-invoke ``AffordanceInvoker`` whose backends are bound
  to the supplied sandbox root

All backends are **read-only** in slice 2b and scoped to the
``sandbox_root`` supplied at construction time. Writing / running
tests / spawning subprocesses lands in slice 2c.

See ``docs/specs/affordance.md`` and Gap 1 slice 2b in
``docs/implementation/13_emogpt_prd_alignment_upgrade.md`` for the
spec + rollout plan.
"""

from lifeform_domain_coding.coding_affordances.backends import (
    SandboxPathError,
    build_coding_affordance_backends,
    resolve_sandbox_path,
)
from lifeform_domain_coding.coding_affordances.descriptors import (
    CODING_AFFORDANCE_DESCRIPTORS,
    CONSENT_FILESYSTEM_READ,
)
from lifeform_domain_coding.coding_affordances.factory import (
    build_coding_affordance_invoker,
    build_coding_affordance_registry,
)

__all__ = [
    "CODING_AFFORDANCE_DESCRIPTORS",
    "CONSENT_FILESYSTEM_READ",
    "SandboxPathError",
    "build_coding_affordance_backends",
    "build_coding_affordance_invoker",
    "build_coding_affordance_registry",
    "resolve_sandbox_path",
]
