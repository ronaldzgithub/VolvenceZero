"""Contract: ``active_mixture`` slot has exactly one owner.

Pins the R8 single-owner invariant for the new
``active_mixture`` slot introduced in
``docs/specs/protocol-runtime.md`` (packet 1.0). Adding a second
publisher to this slot must be a deliberate decision (and would
also fail kernel ``OwnershipGuard`` validation at runtime).

Independently: ensures the owner has the expected SHADOW default
wiring level so promotion to ACTIVE is an explicit
``FinalRolloutConfig`` flip.
"""

from __future__ import annotations

from volvence_zero.behavior_protocol import ActiveMixtureSnapshot
from volvence_zero.integration.final_wiring import (
    FinalRolloutConfig,
    build_final_runtime_modules,
)
from volvence_zero.protocol_runtime import ProtocolRegistryModule
from volvence_zero.runtime import WiringLevel
from volvence_zero.substrate.adapter import PlaceholderSubstrateAdapter


_MODEL_ID = "active-mixture-uniqueness-test-model"


def _build_modules() -> list:
    return build_final_runtime_modules(
        config=FinalRolloutConfig(),
        substrate_adapter=PlaceholderSubstrateAdapter(model_id=_MODEL_ID),
    )


def test_active_mixture_slot_is_registered_exactly_once() -> None:
    modules = _build_modules()
    publishers = [m for m in modules if m.slot_name == "active_mixture"]
    assert len(publishers) == 1, (
        f"R8 single-owner invariant violated: 'active_mixture' has "
        f"{len(publishers)} publishers; expected exactly 1. Adding a "
        "second owner requires a contract review."
    )


def test_active_mixture_owner_is_protocol_registry_module() -> None:
    modules = _build_modules()
    publishers = [m for m in modules if m.slot_name == "active_mixture"]
    assert isinstance(publishers[0], ProtocolRegistryModule)


def test_active_mixture_owner_default_wiring_is_active() -> None:
    """Packet 4.0: default rollout now wires ``active_mixture`` ACTIVE.

    All SHADOW → ACTIVE checklist conditions closed by packet
    1.5a' (identity_gate, α/β learning, typed context_match,
    PE-driven arbitration, matched-control consumer test). The
    default flip in
    ``packages/vz-runtime/.../integration/final_wiring.py`` makes
    every production lifeform see protocol-driven behaviour
    automatically. Reverting requires both flipping the default
    back AND re-enabling fallback mode.
    """

    modules = _build_modules()
    owner = next(m for m in modules if m.slot_name == "active_mixture")
    assert owner.wiring_level is WiringLevel.ACTIVE


def test_active_mixture_owner_declares_expected_dependencies() -> None:
    """Pin the upstream surface of ``ProtocolRegistryModule``.

    Packet 1.0: ``dependencies = ()``.
    Packet 1.3a: adds ``("dual_track", "regime")`` for the real
    identity-gate evaluation (R7 Self trait gate + R14 regime
    compatibility cross-check).
    Packet 1.5a: adds ``("interlocutor_state", "rupture_state",
    "boundary_policy")`` for typed context_match scoring (3
    kernel-side detectors). DRIVE detectors stay deferred (vitals
    not in kernel propagate graph; packet 1.0.1 design).
    Packet 1.5b: adds ``("prediction_error",)`` for owner-side
    rolling pe_utility EMA.
    Packet 1.5a': adds ``("retrieval_policy",)`` for the
    ``RETRIEVAL_HITS_PRESENT`` detector.
    Packet 5.0: adds ``("protocol_phase",)`` so the owner can
    consume PE-driven phase pointers from
    ``ProtocolPhaseModule`` to populate
    ``ActiveProtocolEntry.current_phase_id``.
    Packet 7.0: adds ``("commitment",)`` for the
    ``COMMITMENT_FULFILLED`` / ``COMMITMENT_BROKEN`` detectors.

    Future packets (e.g. metacontroller-driven α/β with a
    dedicated upstream slot) are expected to grow this tuple —
    and update this test in the same PR. A change without test
    update is a contract drift.
    """

    modules = _build_modules()
    owner = next(m for m in modules if m.slot_name == "active_mixture")
    assert owner.dependencies == (
        "dual_track",
        "regime",
        "interlocutor_state",
        "rupture_state",
        "boundary_policy",
        "prediction_error",
        "retrieval_policy",
        "protocol_phase",
        "commitment",
    ), owner.dependencies


def test_active_mixture_value_type_is_active_mixture_snapshot() -> None:
    modules = _build_modules()
    owner = next(m for m in modules if m.slot_name == "active_mixture")
    assert owner.value_type is ActiveMixtureSnapshot


def test_active_mixture_owner_class_var_owner_string_stable() -> None:
    """Pin the owner string so debug events / logs stay stable."""
    modules = _build_modules()
    owner = next(m for m in modules if m.slot_name == "active_mixture")
    assert owner.owner == "ProtocolRegistryModule"
