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


def test_active_mixture_owner_default_wiring_is_shadow() -> None:
    """Default rollout keeps ``active_mixture`` at SHADOW.

    Promoting to ACTIVE is gated on packet 1.2+ (boundary_policy
    becomes the first consumer) and must be done by flipping
    ``FinalRolloutConfig.protocol_runtime`` explicitly. This test
    pins the default so an accidental flip would fail CI.
    """

    modules = _build_modules()
    owner = next(m for m in modules if m.slot_name == "active_mixture")
    assert owner.wiring_level is WiringLevel.SHADOW


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
    rolling pe_utility EMA. The owner attributes per-turn
    ``signed_reward`` to last-turn's active mixture and feeds the
    rolling EMA back into the next mixture's softmax (β·pe_utility
    term).

    Future packets (1.5c α/β learning) are expected to keep this
    tuple stable but may add a metacontroller-style upstream
    (e.g. ``activation_controller_state``). Change requires
    updating this test in the same PR — a change without test
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
