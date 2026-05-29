"""Tests for the substrate profile registry + running-mode detection."""

from __future__ import annotations

import pytest

from dlaas_platform_api.substrate_profiles import (
    ADAPTER_POLICY_NONE,
    ADAPTER_POLICY_PERSONA_LORA,
    MODE_SHARED_FROZEN,
    MODE_SYNTHETIC,
    SubstrateProfile,
    SubstrateProfileRegistry,
    UnknownSubstrateProfile,
    default_substrate_profile_registry,
    running_substrate_mode,
)


def test_default_registry_lists_known_profiles() -> None:
    registry = default_substrate_profile_registry()
    ids = {p.substrate_profile_id for p in registry.list()}
    assert {"shared-frozen", "shared-frozen-persona-lora", "synthetic-dev"} <= ids
    assert registry.default_profile_id == "shared-frozen"


def test_get_empty_returns_default() -> None:
    registry = default_substrate_profile_registry()
    profile = registry.get("")
    assert profile.substrate_profile_id == "shared-frozen"


def test_get_unknown_raises() -> None:
    registry = default_substrate_profile_registry()
    with pytest.raises(UnknownSubstrateProfile):
        registry.get("does-not-exist")


def test_persona_lora_profile_permits_lora() -> None:
    registry = default_substrate_profile_registry()
    assert registry.get("shared-frozen-persona-lora").permits_persona_lora()
    assert not registry.get("shared-frozen").permits_persona_lora()


def test_profile_validates_mode_and_policy() -> None:
    with pytest.raises(ValueError, match="mode"):
        SubstrateProfile(substrate_profile_id="x", mode="bogus")
    with pytest.raises(ValueError, match="adapter_policy"):
        SubstrateProfile(
            substrate_profile_id="x",
            mode=MODE_SHARED_FROZEN,
            adapter_policy="bogus",
        )


def test_registry_rejects_duplicate_and_bad_default() -> None:
    p = SubstrateProfile(substrate_profile_id="a", mode=MODE_SYNTHETIC)
    with pytest.raises(ValueError, match="duplicate"):
        SubstrateProfileRegistry((p, p), default_profile_id="a")
    with pytest.raises(ValueError, match="default_profile_id"):
        SubstrateProfileRegistry((p,), default_profile_id="missing")


def test_running_substrate_mode_none_is_synthetic() -> None:
    assert running_substrate_mode(None) == MODE_SYNTHETIC


def test_running_substrate_mode_synthetic_runtime() -> None:
    from volvence_zero.substrate import SyntheticOpenWeightResidualRuntime

    runtime = SyntheticOpenWeightResidualRuntime()
    assert running_substrate_mode(runtime) == MODE_SYNTHETIC


def test_to_json_round_trips_fields() -> None:
    profile = SubstrateProfile(
        substrate_profile_id="p",
        mode=MODE_SHARED_FROZEN,
        adapter_policy=ADAPTER_POLICY_PERSONA_LORA,
        allow_rare_heavy_refresh=True,
        model_id_hint="Qwen/Qwen2.5-1.5B-Instruct",
    )
    payload = profile.to_json()
    assert payload["adapter_policy"] == ADAPTER_POLICY_PERSONA_LORA
    assert payload["allow_rare_heavy_refresh"] is True
    assert payload["mode"] == MODE_SHARED_FROZEN
    assert ADAPTER_POLICY_NONE != payload["adapter_policy"]
