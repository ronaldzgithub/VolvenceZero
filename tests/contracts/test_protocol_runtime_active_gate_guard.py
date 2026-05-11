"""ACTIVE-gate guard contract (packet 1.0.1 → flipped at 1.5a').

Asserts the SHADOW → ACTIVE upgrade pathway as an *executable*
contract. The original packet 1.0.1 invariant was:

    fallback mode True → ACTIVE construction must fail fast
    with ``FallbackActivationActiveError``.

After packet 1.5a' lands all condition-3 detectors
(``USER_DROPOUT_OBSERVED`` / ``REGIME_TRANSITION_RECENT`` /
``RETRIEVAL_HITS_PRESENT``), the activation controller is no
longer in fallback mode and the contract inverts:

    fallback mode False → ACTIVE construction succeeds.

This file pins both directions:

* ``is_fallback_mode()`` returns False (current state).
* ACTIVE construction succeeds (the guard does not fire).
* SHADOW / DISABLED constructions still work.
* The ``FallbackActivationActiveError`` class shape is preserved
  — it remains a ``ContractViolationError`` subclass and would
  fire if anyone reverted the flag (defence-in-depth in case a
  future packet legitimately needs to revert).

Reverting the flag back to True (e.g. discovering a regression in
a 1.5* mechanism) would break the "ACTIVE construction succeeds"
tests; that's the intended signal — anyone reverting must update
this contract in the same PR and document why.
"""

from __future__ import annotations

import pytest

from volvence_zero.protocol_runtime import (
    FallbackActivationActiveError,
    ProtocolRegistryModule,
    is_fallback_mode,
)
from volvence_zero.protocol_runtime import activation as _activation
from volvence_zero.runtime import WiringLevel
from volvence_zero.runtime.kernel import ContractViolationError


# ---------------------------------------------------------------------------
# Current state: fallback mode False (post packet 1.5a')
# ---------------------------------------------------------------------------


def test_activation_controller_is_no_longer_in_fallback_mode() -> None:
    """Packet 1.5a' flipped the flag.

    Reverting this test (or the underlying flag) requires a
    deliberate decision — see module docstring.
    """

    assert is_fallback_mode() is False


# ---------------------------------------------------------------------------
# SHADOW / DISABLED construction is always fine
# ---------------------------------------------------------------------------


def test_shadow_default_construction_works() -> None:
    module = ProtocolRegistryModule()
    assert module.wiring_level is WiringLevel.SHADOW


def test_shadow_explicit_construction_works() -> None:
    module = ProtocolRegistryModule(wiring_level=WiringLevel.SHADOW)
    assert module.wiring_level is WiringLevel.SHADOW


def test_disabled_explicit_construction_works() -> None:
    module = ProtocolRegistryModule(wiring_level=WiringLevel.DISABLED)
    assert module.wiring_level is WiringLevel.DISABLED


# ---------------------------------------------------------------------------
# ACTIVE construction now succeeds
# ---------------------------------------------------------------------------


def test_active_construction_now_succeeds() -> None:
    """Post packet 1.5a': ACTIVE wiring is legal.

    The construction returns a normal owner with
    ``wiring_level=ACTIVE`` and no exception is raised. This is
    the executable proof that the entire SHADOW → ACTIVE
    checklist closed.
    """

    module = ProtocolRegistryModule(wiring_level=WiringLevel.ACTIVE)
    assert module.wiring_level is WiringLevel.ACTIVE


# ---------------------------------------------------------------------------
# The guard class shape is preserved (defence-in-depth)
# ---------------------------------------------------------------------------


def test_fallback_active_error_class_still_exists() -> None:
    """The error class is preserved across the flag flip.

    If a future packet legitimately needs to revert the flag
    (e.g. a regression in α/β learning is discovered and we want
    to gate ACTIVE again until it's fixed), the class is still
    importable and ready to fire.
    """

    assert issubclass(FallbackActivationActiveError, ContractViolationError)


def test_guard_fires_when_flag_is_temporarily_reverted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hard-revert the flag in-process and assert the guard still fires.

    This is the defence-in-depth path: if a real reversion
    scenario lands in the future, the guard is still wired and
    will fail fast at construction. Uses ``monkeypatch`` so the
    rest of the test session doesn't see the revert.
    """

    monkeypatch.setattr(
        _activation,
        "_ACTIVATION_CONTROLLER_FALLBACK_MODE",
        True,
    )
    assert is_fallback_mode() is True

    with pytest.raises(FallbackActivationActiveError) as exc_info:
        ProtocolRegistryModule(wiring_level=WiringLevel.ACTIVE)

    message = str(exc_info.value)
    assert "protocol-runtime.md" in message, message
    assert "SHADOW" in message and "ACTIVE" in message, message
    assert "fallback" in message.lower(), message


# ---------------------------------------------------------------------------
# Final wiring respects the new state
# ---------------------------------------------------------------------------


def test_final_wiring_default_protocol_runtime_is_active() -> None:
    """Packet 4.0: ``FinalRolloutConfig`` default is now ACTIVE.

    Now that the entire SHADOW → ACTIVE checklist is closed
    (1.5a' flipped ``_ACTIVATION_CONTROLLER_FALLBACK_MODE``),
    the default rollout flips so every production lifeform sees
    protocol-driven behaviour automatically. Reverting requires
    flipping the default in
    ``packages/vz-runtime/.../integration/final_wiring.py`` AND
    re-enabling fallback mode (this test is the canary).
    """

    from volvence_zero.integration.final_wiring import FinalRolloutConfig

    config = FinalRolloutConfig()
    assert config.protocol_runtime is WiringLevel.ACTIVE


def test_final_wiring_with_active_protocol_runtime_now_succeeds() -> None:
    """``FinalRolloutConfig(protocol_runtime=ACTIVE)`` builds cleanly.

    Post packet 1.5a': flipping the rollout config to ACTIVE
    produces a normal modules list with the protocol_runtime
    owner at ACTIVE. Previously this raised
    ``FallbackActivationActiveError``.
    """

    from volvence_zero.integration.final_wiring import (
        FinalRolloutConfig,
        build_final_runtime_modules,
    )
    from volvence_zero.substrate.adapter import PlaceholderSubstrateAdapter

    config = FinalRolloutConfig(protocol_runtime=WiringLevel.ACTIVE)
    adapter = PlaceholderSubstrateAdapter(model_id="active-gate-guard-test")
    modules = build_final_runtime_modules(
        config=config,
        substrate_adapter=adapter,
    )
    owner = next(m for m in modules if m.slot_name == "active_mixture")
    assert owner.wiring_level is WiringLevel.ACTIVE
