"""ACTIVE-gate guard contract (packet 1.0.1).

Asserts that ``ProtocolRegistryModule`` refuses to be constructed at
``WiringLevel.ACTIVE`` while ``activation.is_fallback_mode()`` is
True. This pins the SHADOW → ACTIVE upgrade checklist as an
*executable* contract rather than a docstring promise.

The guard is part of the packet 1.0.1 consolidation (external review
follow-up). Future packets that land real activation machinery (real
identity_gate at packet 1.3+, learned α/β + typed context_match + PE
arbitration at packet 1.5+, first ACTIVE consumer at packet 1.2+)
are expected to flip
``volvence_zero.protocol_runtime.activation._ACTIVATION_CONTROLLER_FALLBACK_MODE``
to False in the same landing PR; once that lands, the guard becomes
unreachable and this test will need to be reframed (or deleted).

What this test pins:

* SHADOW construction always works (default + explicit).
* ACTIVE construction fails with a typed
  :class:`FallbackActivationActiveError` while fallback mode is on.
* The error message references the spec checklist so on-call gets a
  pointer instead of a generic stack trace.
* The exception type is a subclass of ``ContractViolationError``
  (the canonical kernel contract error) so any framework that
  bundles contract violations catches it without special-casing.
"""

from __future__ import annotations

import pytest

from volvence_zero.protocol_runtime import (
    FallbackActivationActiveError,
    ProtocolRegistryModule,
    is_fallback_mode,
)
from volvence_zero.runtime import WiringLevel
from volvence_zero.runtime.kernel import ContractViolationError


# ---------------------------------------------------------------------------
# Sanity: packet 1.0.1 ships fallback mode = True
# ---------------------------------------------------------------------------


def test_activation_controller_is_in_fallback_mode_packet_1_0() -> None:
    """Packet 1.0 ships fallback mode = True.

    When packet 1.5 lands the real activation machinery, this assert
    will need to be inverted (or deleted with this test file). That
    is intentional: failing this test is the signal that the
    activation upgrade is being landed and the guard semantics need
    revisiting.
    """

    assert is_fallback_mode() is True


# ---------------------------------------------------------------------------
# SHADOW construction is unaffected
# ---------------------------------------------------------------------------


def test_shadow_default_construction_works() -> None:
    module = ProtocolRegistryModule()
    assert module.wiring_level is WiringLevel.SHADOW


def test_shadow_explicit_construction_works() -> None:
    module = ProtocolRegistryModule(wiring_level=WiringLevel.SHADOW)
    assert module.wiring_level is WiringLevel.SHADOW


def test_disabled_explicit_construction_works() -> None:
    """DISABLED is also fine — the slot is simply not published.

    Only ACTIVE triggers the fallback guard.
    """

    module = ProtocolRegistryModule(wiring_level=WiringLevel.DISABLED)
    assert module.wiring_level is WiringLevel.DISABLED


# ---------------------------------------------------------------------------
# ACTIVE construction during fallback fails fast
# ---------------------------------------------------------------------------


def test_active_construction_during_fallback_raises() -> None:
    with pytest.raises(FallbackActivationActiveError):
        ProtocolRegistryModule(wiring_level=WiringLevel.ACTIVE)


def test_fallback_active_error_inherits_contract_violation_error() -> None:
    """Generic contract-violation handlers should catch this error.

    The kernel uses ``ContractViolationError`` as the umbrella type
    for all contract violations (ownership / dependency / schema /
    immutability). Any framework that already catches
    ``ContractViolationError`` (e.g. propagate / recorder /
    promotion gates) should catch this without special-casing.
    """

    assert issubclass(FallbackActivationActiveError, ContractViolationError)


def test_fallback_active_error_message_points_to_spec() -> None:
    """The error message must surface the spec checklist as a pointer.

    On-call should get a route to the canonical upgrade list, not
    just a stack trace.
    """

    with pytest.raises(FallbackActivationActiveError) as exc_info:
        ProtocolRegistryModule(wiring_level=WiringLevel.ACTIVE)

    message = str(exc_info.value)
    assert "protocol-runtime.md" in message, message
    assert "SHADOW" in message and "ACTIVE" in message, message
    # Sanity: identifies which controller is in fallback
    assert "fallback" in message.lower(), message


# ---------------------------------------------------------------------------
# Final wiring respects the guard
# ---------------------------------------------------------------------------


def test_final_wiring_default_protocol_runtime_is_shadow() -> None:
    """The default rollout config keeps ``protocol_runtime`` at SHADOW.

    If the default were flipped to ACTIVE without flipping the
    fallback flag, this test would fire.
    """

    from volvence_zero.integration.final_wiring import FinalRolloutConfig

    config = FinalRolloutConfig()
    assert config.protocol_runtime is WiringLevel.SHADOW


def test_final_wiring_with_active_protocol_runtime_fails_construction() -> None:
    """``FinalRolloutConfig(protocol_runtime=ACTIVE)`` + default
    activation must fail at module construction time during
    ``build_final_runtime_modules``, not silently.
    """

    from volvence_zero.integration.final_wiring import (
        FinalRolloutConfig,
        build_final_runtime_modules,
    )
    from volvence_zero.substrate.adapter import PlaceholderSubstrateAdapter

    config = FinalRolloutConfig(protocol_runtime=WiringLevel.ACTIVE)
    adapter = PlaceholderSubstrateAdapter(model_id="active-gate-guard-test")
    with pytest.raises(FallbackActivationActiveError):
        build_final_runtime_modules(
            config=config,
            substrate_adapter=adapter,
        )
