"""Behavior Protocol Runtime — application-tier owner subpackage (vz-application).

This subpackage hosts the ``ProtocolRegistryModule`` owner and its
supporting helpers. The cross-wheel schema lives in
``vz-contracts.behavior_protocol``; per-vertical FixtureUptake
adapters live in each ``lifeform-domain-*`` wheel.

Wheel placement (packet 1.2 decision):

This module lives in ``vz-application`` (not ``vz-cognition``) because
the packet 1.2 compile path produces application-tier artifacts —
``BoundaryPriorHint`` records pushed into ``boundary_policy``'s
``ApplicationRareHeavyState``. By tier order ``vz-cognition`` cannot
import ``vz-application``, so the owner naturally belongs in the
upper application tier even though most of its inputs (PE / regime /
interlocutor / Self / rupture) are cognitive. See packet 1.2 change
log in ``docs/specs/protocol-runtime.md``.

See ``docs/specs/protocol-runtime.md`` for the full design.

Packet 1.0 / 1.0.1 / 1.2 status (SHADOW):

* ``ProtocolRegistryModule`` publishes ``ActiveMixtureSnapshot`` each
  turn into ``shadow_snapshots`` only.
* No consumer reads the slot directly; canonical content (boundary
  hints) flows through the existing ``boundary_policy`` execution
  path because protocol load compiles entries into the same
  ``ApplicationRareHeavyState`` that domain experience packages
  populate.
* ``ActivationController`` is the equal-weight fallback (packet 1.0);
  PE feedback / context match light up in packet 1.5+. Drive coupling
  is permanently deferred to a future kernel-side adapter (packet
  1.0.1, see spec §调度).
* ``identity_gate`` is hard-coded to 1.0; real R7-Self / R14-regime
  cross-checks light up in packet 1.3+.
* ACTIVE wiring is fail-loud blocked by ``FallbackActivationActiveError``
  while the activation controller is in fallback (packet 1.0.1).
"""

from __future__ import annotations

from volvence_zero.protocol_runtime.activation import (
    compute_active_mixture,
    is_fallback_mode,
)
from volvence_zero.protocol_runtime.compiler import (
    ProtocolApplicationArtifacts,
    compile_protocol_to_application_artifacts,
)
from volvence_zero.protocol_runtime.owner import (
    FallbackActivationActiveError,
    ProtocolRegistryModule,
)
from volvence_zero.protocol_runtime.registry import ProtocolRegistry
from volvence_zero.protocol_runtime.revision_queue import (
    ApprovalDecision,
    ApprovalOutcome,
    RevisionQueue,
    evaluate_protocol_revision,
)

__all__ = [
    "ApprovalDecision",
    "ApprovalOutcome",
    "FallbackActivationActiveError",
    "ProtocolApplicationArtifacts",
    "ProtocolRegistry",
    "ProtocolRegistryModule",
    "RevisionQueue",
    "compile_protocol_to_application_artifacts",
    "compute_active_mixture",
    "evaluate_protocol_revision",
    "is_fallback_mode",
]
