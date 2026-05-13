"""CapabilityWiring 行为契约测试 (architecture-uplift A3).

验证 ``RuntimeModule.capabilities`` ClassVar + ``capability_overrides``
__init__ 参数 + ``ResolvedProfile.apply_to_config()`` 三件套:

- 现有所有 module 行为完全不变 (默认 capability map 为空 → byte-equivalent)
- ``capability_wiring(name)`` 在未知 capability 上 fallback 到 module-level
  wiring_level
- ``capability_overrides`` 对未声明的 capability fail-loudly
- ``ResolvedProfile.apply_to_config(FinalRolloutConfig())`` 是 pure function
- ``FinalRolloutConfig.capability_overrides_for(owner)`` 默认返回空 mapping
"""

from __future__ import annotations

import dataclasses
from types import MappingProxyType

import pytest

from volvence_zero.agent.profile_registry import (
    ProfileCapability,
    ProfileRegistry,
    ProfileSpec,
    resolve_profile,
)
from volvence_zero.integration.final_wiring import FinalRolloutConfig
from volvence_zero.runtime.kernel import (
    CapabilityWiring,
    ContractViolationError,
    RuntimeModule,
    WiringLevel,
)


class _DummyModule(RuntimeModule):
    """Test fixture: bare module that declares two sub-capabilities."""

    slot_name = "dummy"
    owner = "_DummyModule"
    value_type = dict
    capabilities = MappingProxyType(
        {
            "cap_a": WiringLevel.SHADOW,
            "cap_b": WiringLevel.ACTIVE,
        }
    )

    async def process(self, upstream):  # pragma: no cover — not exercised
        raise NotImplementedError


class _ModuleWithoutCapabilities(RuntimeModule):
    """Test fixture: module that never opted into A3 (capabilities map empty)."""

    slot_name = "dummy_legacy"
    owner = "_ModuleWithoutCapabilities"
    value_type = dict

    async def process(self, upstream):  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# RuntimeModule.capabilities + capability_overrides
# ---------------------------------------------------------------------------


def test_module_without_capabilities_preserves_existing_behaviour() -> None:
    """A module that hasn't opted into A3 must behave exactly as before:
    capability_wiring(any) returns module-level wiring_level."""
    m = _ModuleWithoutCapabilities(wiring_level=WiringLevel.ACTIVE)
    assert m.capability_wiring("anything") is WiringLevel.ACTIVE
    assert m.capability_wiring("else") is WiringLevel.ACTIVE
    assert m.capability_active("anything") is True
    assert m.capability_shadow("anything") is False


def test_module_with_capabilities_uses_declared_defaults() -> None:
    m = _DummyModule(wiring_level=WiringLevel.ACTIVE)
    # cap_a default SHADOW
    assert m.capability_wiring("cap_a") is WiringLevel.SHADOW
    assert m.capability_shadow("cap_a") is True
    assert m.capability_active("cap_a") is False
    # cap_b default ACTIVE
    assert m.capability_wiring("cap_b") is WiringLevel.ACTIVE
    # unknown capability falls back to module-level
    assert m.capability_wiring("cap_c") is WiringLevel.ACTIVE


def test_capability_overrides_apply() -> None:
    m = _DummyModule(
        wiring_level=WiringLevel.ACTIVE,
        capability_overrides={"cap_a": WiringLevel.ACTIVE},
    )
    assert m.capability_wiring("cap_a") is WiringLevel.ACTIVE
    # other declared capability unaffected
    assert m.capability_wiring("cap_b") is WiringLevel.ACTIVE


def test_capability_overrides_for_unknown_capability_fails_loudly() -> None:
    """spec §A3.2 fail-loudly: overriding an undeclared capability must raise."""
    with pytest.raises(ContractViolationError, match="unknown capability"):
        _DummyModule(
            wiring_level=WiringLevel.ACTIVE,
            capability_overrides={"cap_not_declared": WiringLevel.ACTIVE},
        )


def test_capability_overrides_immutability() -> None:
    """The merged _capability_wiring map must be read-only; mutating it
    externally must raise."""
    m = _DummyModule(wiring_level=WiringLevel.ACTIVE)
    with pytest.raises(TypeError):
        m._capability_wiring["cap_a"] = WiringLevel.ACTIVE  # type: ignore[index]  # noqa: SLF001


# ---------------------------------------------------------------------------
# CapabilityWiring dataclass
# ---------------------------------------------------------------------------


def test_capability_wiring_dataclass_is_frozen() -> None:
    """spec §A3.1: CapabilityWiring must be frozen (R8 SSOT)."""
    cw = CapabilityWiring(
        capability_name="cap_a",
        owner="_DummyModule",
        wiring_level=WiringLevel.SHADOW,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        cw.wiring_level = WiringLevel.ACTIVE  # type: ignore[misc]


# ---------------------------------------------------------------------------
# FinalRolloutConfig.capability_wirings + capability_overrides_for
# ---------------------------------------------------------------------------


def test_final_rollout_config_capability_wirings_empty_by_default() -> None:
    """Default config has no capability overrides (byte-equivalent)."""
    cfg = FinalRolloutConfig()
    assert cfg.capability_wirings == {}
    assert cfg.capability_overrides_for("evaluation") == {}
    assert cfg.capability_overrides_for("anything") == {}


def test_capability_overrides_for_returns_owner_specific_map() -> None:
    cfg = FinalRolloutConfig(
        capability_wirings=MappingProxyType(
            {
                "evaluation": MappingProxyType({"cheap_layer": WiringLevel.ACTIVE}),
                "credit": MappingProxyType({"least_control": WiringLevel.SHADOW}),
            }
        )
    )
    assert cfg.capability_overrides_for("evaluation") == {
        "cheap_layer": WiringLevel.ACTIVE
    }
    assert cfg.capability_overrides_for("credit") == {
        "least_control": WiringLevel.SHADOW
    }
    # unknown owner → empty mapping
    assert cfg.capability_overrides_for("unknown_owner") == {}


# ---------------------------------------------------------------------------
# ResolvedProfile.apply_to_config
# ---------------------------------------------------------------------------


def test_apply_to_config_is_pure() -> None:
    """spec §A1.4: apply_to_config returns new FinalRolloutConfig, never mutates."""
    reg = ProfileRegistry()
    reg.register_capability(
        ProfileCapability(
            name="my-cap",
            applies_to_owner="evaluation",
            wiring_overrides={"some_sub_cap": WiringLevel.ACTIVE},
        )
    )
    reg.register_profile(ProfileSpec(label="p1", capabilities=("my-cap",)))
    reg.validate()
    resolved = reg.resolve_profile("p1")

    base = FinalRolloutConfig()
    new_cfg = resolved.apply_to_config(base)

    # base is unchanged
    assert base.capability_wirings == {}
    # new_cfg has the override
    assert new_cfg.capability_overrides_for("evaluation") == {
        "some_sub_cap": WiringLevel.ACTIVE
    }


def test_apply_to_config_layers_over_existing_capability_wirings() -> None:
    """When base already has capability_wirings, apply_to_config unions on top."""
    reg = ProfileRegistry()
    reg.register_capability(
        ProfileCapability(
            name="new-cap",
            applies_to_owner="evaluation",
            wiring_overrides={"new_sub": WiringLevel.ACTIVE},
        )
    )
    reg.register_profile(ProfileSpec(label="p2", capabilities=("new-cap",)))
    reg.validate()
    resolved = reg.resolve_profile("p2")

    base = FinalRolloutConfig(
        capability_wirings=MappingProxyType(
            {
                "evaluation": MappingProxyType({"existing_sub": WiringLevel.SHADOW}),
            }
        )
    )
    new_cfg = resolved.apply_to_config(base)

    # both old and new entries present
    eval_overrides = new_cfg.capability_overrides_for("evaluation")
    assert eval_overrides["existing_sub"] is WiringLevel.SHADOW
    assert eval_overrides["new_sub"] is WiringLevel.ACTIVE


def test_apply_to_config_rejects_non_final_rollout_config() -> None:
    """fail-loudly when passed wrong type."""
    reg = ProfileRegistry()
    reg.register_capability(
        ProfileCapability(name="c", applies_to_owner="evaluation")
    )
    reg.register_profile(ProfileSpec(label="p", capabilities=("c",)))
    reg.validate()
    resolved = reg.resolve_profile("p")
    with pytest.raises(TypeError, match="FinalRolloutConfig"):
        resolved.apply_to_config("not a config")


def test_pe_eta_baseline_apply_is_byte_equivalent() -> None:
    """The baseline profile must not introduce any capability override."""
    resolved = resolve_profile("pe-eta")
    base = FinalRolloutConfig()
    new_cfg = resolved.apply_to_config(base)
    assert new_cfg.capability_wirings == base.capability_wirings == {}


# ---------------------------------------------------------------------------
# Existing module 行为不回归 sanity check
# ---------------------------------------------------------------------------


def test_existing_real_module_construction_unchanged() -> None:
    """Importing and constructing a real production module without
    ``capability_overrides`` must still work; A3 changes are additive."""
    from volvence_zero.evaluation.backbone import EvaluationModule

    # No capability_overrides ⇒ pre-A3 behaviour
    m = EvaluationModule()
    assert m.wiring_level is EvaluationModule.default_wiring_level
    # capability lookup falls back to module-level wiring
    assert m.capability_wiring("anything") is m.wiring_level
