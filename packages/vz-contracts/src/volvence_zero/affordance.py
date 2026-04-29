"""Affordance descriptors \u2014 cross-wheel immutable contract surface.

Lives in ``vz-contracts`` because BOTH kernel owners (that may one
day consume an affordance selection state) AND the lifeform-side
scheduler / registry need the same descriptor shape. Putting it in
``lifeform-affordance`` would reverse the kernel \u2192 lifeform
direction, breaking ``tests/contracts/test_import_boundaries.py``.

The contract is *data only*: no registry, no invoker, no scheduler.
Those live in ``lifeform-affordance``. See
``docs/specs/affordance.md`` for the full design and Gap 1 in
``docs/implementation/13_emogpt_prd_alignment_upgrade.md`` for the
phased rollout plan.

Key invariants enforced at construction time (post_init):

* ``when_to_use`` and ``when_not_to_use`` are each \u2265 50 characters.
  This is the single biggest knob for LLM selection quality;
  authors who skimp on the description end up with affordances
  that don't get picked for the right turn. Fail loudly.
* ``parameters_schema`` is a JSON-Schema-shaped dict with at minimum
  a ``type`` key.
* ``safety_model.requires_consent_grant`` names must be non-empty
  strings; an empty consent grant name is a bug.
* ``affordance_tags`` must be unique (no duplicate tags).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class AffordanceKind(str, Enum):
    """Four kinds of affordances the lifeform can be offered.

    * ``TOOL`` \u2014 a function / API call (e.g. ``read_file``,
      ``run_test``). Executed synchronously or asynchronously, result
      flows back through ``BrainSession.submit_tool_result``.
    * ``ACTION`` \u2014 an internal action such as ``clarify``,
      ``commit``, ``plan``. Executed inside the kernel.
    * ``ORGAN`` \u2014 a composed capability (multi-step internal flow),
      e.g. a research sub-loop.
    * ``SHELL`` \u2014 a deployment-side capability (text streaming,
      voice, image) gated by the host environment.
    """

    TOOL = "tool"
    ACTION = "action"
    ORGAN = "organ"
    SHELL = "shell"


class AffordanceLatencyClass(str, Enum):
    INSTANT = "instant"       # < 50ms typical
    FAST = "fast"             # 50ms - 500ms
    SLOW = "slow"             # 500ms - 5s
    VERY_SLOW = "very_slow"   # > 5s


class AffordanceMonetaryClass(str, Enum):
    FREE = "free"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class AffordanceCost:
    """Cost envelope the scheduler / policy can read for gating.

    ``rate_limit_per_minute`` is optional. When ``None`` there is no
    rate limit declared; callers that enforce rate limiting treat
    that as "unlimited, but prudent use still applies".
    """

    latency_class: AffordanceLatencyClass
    monetary_class: AffordanceMonetaryClass = AffordanceMonetaryClass.FREE
    rate_limit_per_minute: int | None = None

    def __post_init__(self) -> None:
        if self.rate_limit_per_minute is not None and self.rate_limit_per_minute <= 0:
            raise ValueError(
                f"AffordanceCost.rate_limit_per_minute must be > 0 when set, "
                f"got {self.rate_limit_per_minute!r}"
            )


@dataclass(frozen=True)
class AffordanceSafety:
    """Safety envelope. Drives ModificationGate + boundary_consent.

    * ``requires_user_confirmation`` \u2014 every invocation needs an
      explicit human gate (even in autonomous modes). Use for
      irreversible or high-impact actions.
    * ``irreversible`` \u2014 the action cannot be undone programmatically.
      Tools that write files should set True.
    * ``requires_consent_grant`` \u2014 tuple of named consent grants
      that must be present in the ``boundary_consent`` snapshot for
      this affordance to be invocable.
    * ``blocked_in_regimes`` \u2014 the affordance should NOT be offered
      when the active regime is one of these. Prevents (e.g.)
      ``run_test`` being proposed during ``emotional_support``.
    * ``audit_required`` \u2014 every invocation gets a durable audit
      record even for low-impact actions.
    """

    requires_user_confirmation: bool = False
    irreversible: bool = False
    requires_consent_grant: tuple[str, ...] = ()
    blocked_in_regimes: tuple[str, ...] = ()
    audit_required: bool = False

    def __post_init__(self) -> None:
        for grant in self.requires_consent_grant:
            if not grant.strip():
                raise ValueError(
                    f"AffordanceSafety.requires_consent_grant entries "
                    f"must be non-empty; got {self.requires_consent_grant!r}"
                )
        if len(set(self.requires_consent_grant)) != len(self.requires_consent_grant):
            raise ValueError(
                f"AffordanceSafety.requires_consent_grant must be unique, "
                f"got {self.requires_consent_grant!r}"
            )
        for regime in self.blocked_in_regimes:
            if not regime.strip():
                raise ValueError(
                    f"AffordanceSafety.blocked_in_regimes entries must be "
                    f"non-empty; got {self.blocked_in_regimes!r}"
                )


# Minimum length for ``when_to_use`` / ``when_not_to_use``. The spec
# picks 50 because that's roughly the threshold at which an LLM
# selection prompt can actually distinguish between two candidates.
# Anything shorter is "label + hand-wave" and degrades selection
# quality meaningfully.
MIN_SELECTION_HINT_CHARS: int = 50


@dataclass(frozen=True)
class AffordanceDescriptor:
    """The canonical immutable descriptor for one affordance.

    Authors write one of these per affordance (typically YAML-sourced
    in the owning vertical) and register it with
    ``AffordanceRegistry`` at lifeform startup. Consumers read it at
    selection time and at render time via the four renderers.
    """

    name: str
    kind: AffordanceKind
    version: str
    display_name: str
    description: str
    when_to_use: str
    when_not_to_use: str
    parameters_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any]
    cost_model: AffordanceCost
    safety_model: AffordanceSafety
    preconditions: tuple[str, ...] = ()
    affordance_tags: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    source_path: str = ""
    excluded_from_runtime_selection: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("AffordanceDescriptor.name must be non-empty")
        if not self.version.strip():
            raise ValueError(
                f"AffordanceDescriptor.version must be non-empty "
                f"for {self.name!r}"
            )
        if not self.display_name.strip():
            raise ValueError(
                f"AffordanceDescriptor.display_name must be non-empty "
                f"for {self.name!r}"
            )
        if len(self.when_to_use) < MIN_SELECTION_HINT_CHARS:
            raise ValueError(
                f"AffordanceDescriptor.when_to_use must be \u2265 "
                f"{MIN_SELECTION_HINT_CHARS} chars for {self.name!r}, "
                f"got len={len(self.when_to_use)}."
            )
        if len(self.when_not_to_use) < MIN_SELECTION_HINT_CHARS:
            raise ValueError(
                f"AffordanceDescriptor.when_not_to_use must be \u2265 "
                f"{MIN_SELECTION_HINT_CHARS} chars for {self.name!r}, "
                f"got len={len(self.when_not_to_use)}."
            )
        if not isinstance(self.parameters_schema, Mapping):
            raise TypeError(
                f"AffordanceDescriptor.parameters_schema must be a "
                f"Mapping (JSON Schema-shaped dict) for {self.name!r}, "
                f"got {type(self.parameters_schema).__name__}"
            )
        if "type" not in self.parameters_schema:
            raise ValueError(
                f"AffordanceDescriptor.parameters_schema must have a "
                f"'type' key (JSON Schema) for {self.name!r}"
            )
        if not isinstance(self.output_schema, Mapping):
            raise TypeError(
                f"AffordanceDescriptor.output_schema must be a Mapping "
                f"for {self.name!r}, got {type(self.output_schema).__name__}"
            )
        if len(set(self.affordance_tags)) != len(self.affordance_tags):
            raise ValueError(
                f"AffordanceDescriptor.affordance_tags must be unique "
                f"for {self.name!r}, got {self.affordance_tags!r}"
            )


__all__ = [
    "MIN_SELECTION_HINT_CHARS",
    "AffordanceCost",
    "AffordanceDescriptor",
    "AffordanceKind",
    "AffordanceLatencyClass",
    "AffordanceMonetaryClass",
    "AffordanceSafety",
]
