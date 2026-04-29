"""Lifeform-side affordance layer (Gap 1 slice 1).

Public API:

* ``AffordanceRegistry`` \u2014 startup atomic write, runtime O(1) read
* ``AffordanceCandidate`` / ``AffordanceSnapshot`` \u2014 lifeform-side
  snapshot shape consumed by the prompt planner
* ``build_neutral_snapshot`` \u2014 slice-1 scaffold synthesiser; slice
  2 replaces it with a metacontroller-driven scorer
* 4 renderers: ``render_markdown`` / ``render_openai_tools`` /
  ``render_catalog_json`` / ``render_compact_list``

Immutable descriptor types (``AffordanceDescriptor`` /
``AffordanceKind`` / ``AffordanceCost`` / ``AffordanceSafety`` /
``AffordanceLatencyClass`` / ``AffordanceMonetaryClass``) are
re-exported from ``vz-contracts`` for convenience; they are the
cross-wheel contract surface.

See ``docs/specs/affordance.md`` for the spec and Gap 1 of
``docs/implementation/13_emogpt_prd_alignment_upgrade.md`` for the
phased rollout plan.
"""

from volvence_zero.affordance import (
    MIN_SELECTION_HINT_CHARS,
    AffordanceCost,
    AffordanceDescriptor,
    AffordanceKind,
    AffordanceLatencyClass,
    AffordanceMonetaryClass,
    AffordanceSafety,
)

from lifeform_affordance.invoker import (
    AffordanceBackend,
    AffordanceInvocationError,
    AffordanceInvocationResult,
    AffordanceInvocationStatus,
    AffordanceInvoker,
    BoundaryCheckContext,
    BoundaryDenial,
    BoundaryPolicy,
    DescriptorDerivedBoundaryPolicy,
    validate_parameters,
)
from lifeform_affordance.registry import (
    AffordanceAlreadyRegisteredError,
    AffordanceLintWarning,
    AffordanceRegistry,
    AffordanceRegistryError,
    AffordanceRegistrySealedError,
)
from lifeform_affordance.renderers import (
    render_catalog_json,
    render_compact_list,
    render_markdown,
    render_openai_tools,
)
from lifeform_affordance.snapshot import (
    AffordanceCandidate,
    AffordanceSnapshot,
    build_neutral_snapshot,
)

__all__ = [
    "MIN_SELECTION_HINT_CHARS",
    "AffordanceAlreadyRegisteredError",
    "AffordanceBackend",
    "AffordanceCandidate",
    "AffordanceCost",
    "AffordanceDescriptor",
    "AffordanceInvocationError",
    "AffordanceInvocationResult",
    "AffordanceInvocationStatus",
    "AffordanceInvoker",
    "AffordanceKind",
    "AffordanceLatencyClass",
    "AffordanceLintWarning",
    "AffordanceMonetaryClass",
    "AffordanceRegistry",
    "AffordanceRegistryError",
    "AffordanceRegistrySealedError",
    "AffordanceSafety",
    "AffordanceSnapshot",
    "BoundaryCheckContext",
    "BoundaryDenial",
    "BoundaryPolicy",
    "DescriptorDerivedBoundaryPolicy",
    "build_neutral_snapshot",
    "render_catalog_json",
    "render_compact_list",
    "render_markdown",
    "render_openai_tools",
    "validate_parameters",
]
