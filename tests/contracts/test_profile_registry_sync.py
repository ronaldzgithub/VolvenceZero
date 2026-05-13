"""ProfileRegistry ↔ legacy build_standard_dialogue_runner SSOT 同步契约测试。

实现 architecture-uplift A1 spec §迁移协议 阶段 1 的"双跑 contract test":

- registry 已注册 11 个内置 ProfileSpec (含 ``eta-no-pe`` 别名与
  ``heuristic-baseline``)
- 每个 legacy ``build_standard_dialogue_runner`` if-elif 分支对应的
  profile_label 都能从 registry resolve 出 ResolvedProfile
- 每个非 baseline profile 的 capability bundle 非空
- 每个 capability 的 ``applies_to_owner`` 都对应 FinalRolloutConfig 的合法
  字段（避免拼写错误）

阶段 1 Done 标志: 双跑 contract test 全 PASS, registry 不进 dispatch.
"""

from __future__ import annotations

import dataclasses

import pytest

from volvence_zero.agent.profile_registry import (
    ProfileRegistryViolationError,
    builtin_profile_registry,
    list_builtin_capabilities,
    list_builtin_profiles,
    resolve_profile,
)
from volvence_zero.integration.final_wiring import FinalRolloutConfig
from volvence_zero.runtime.kernel import WiringLevel


# legacy build_standard_dialogue_runner if-elif 分支覆盖的 profile_label
# (含别名 eta-no-pe + heuristic-baseline). 来源:
# packages/vz-runtime/src/volvence_zero/agent/dialogue/_legacy.py:7862-8001
_LEGACY_PROFILE_LABELS: tuple[str, ...] = (
    "pe-eta",
    "atlas-titans-cms-uplift",
    "pe-eta-online-only",
    "pe-eta-no-writeback",
    "pe-eta-no-rare-heavy",
    "pe-eta-no-semantic-label",
    "pe-eta-no-reflection-cache",
    "pe-eta-pe-readout-only",
    "pe-drive-off",
    "eta-no-pe",  # alias of pe-drive-off
    "timescale-off",
    "eta-off",
    "heuristic-baseline",
)


def test_registry_validates_at_import_time() -> None:
    """The module-level singleton registry must be valid (validate() PASS).

    Implicitly exercised by builtin_profile_registry(); this test makes the
    expectation explicit.
    """
    reg = builtin_profile_registry()
    # If invariants are broken, validate() raises during construction.
    assert reg is not None
    # A second call returns the same singleton.
    assert builtin_profile_registry() is reg


def test_registry_lists_all_known_profiles() -> None:
    profiles = list_builtin_profiles()
    # 12 canonical profile labels (pe-drive-off has one alias eta-no-pe that
    # is NOT in list_profiles since it lists canonical labels only).
    assert len(profiles) >= 12, (
        f"expected ≥12 canonical profiles, got {len(profiles)}: {profiles}"
    )


def test_every_legacy_label_resolves() -> None:
    """Every legacy if-elif branch label (含 alias) must resolve."""
    unresolved: list[str] = []
    for label in _LEGACY_PROFILE_LABELS:
        try:
            resolve_profile(label)
        except ProfileRegistryViolationError as exc:
            unresolved.append(f"{label!r}: {exc}")
    assert not unresolved, (
        "registry cannot resolve every legacy profile_label:\n"
        + "\n".join(f"  - {u}" for u in unresolved)
    )


def test_pe_eta_baseline_has_empty_capabilities() -> None:
    """baseline profile uses default behavior; no capability override."""
    resolved = resolve_profile("pe-eta")
    assert resolved.capabilities == (), (
        f"pe-eta baseline should have empty capability bundle, "
        f"got {[c.name for c in resolved.capabilities]}"
    )


def test_non_baseline_profiles_have_non_empty_capabilities() -> None:
    """Every non-baseline legacy profile must declare at least one capability;
    otherwise it would be byte-equivalent to pe-eta, defeating the purpose."""
    for label in _LEGACY_PROFILE_LABELS:
        if label == "pe-eta":
            continue
        resolved = resolve_profile(label)
        assert resolved.capabilities, (
            f"profile {label!r} has empty capability bundle; "
            f"would be byte-equivalent to baseline"
        )


def test_eta_no_pe_alias_resolves_to_pe_drive_off() -> None:
    """Alias must resolve to the canonical profile."""
    aliased = resolve_profile("eta-no-pe")
    canonical = resolve_profile("pe-drive-off")
    assert aliased.label == canonical.label == "pe-drive-off"
    assert tuple(c.name for c in aliased.capabilities) == tuple(
        c.name for c in canonical.capabilities
    )


def test_unknown_profile_label_fails_loudly() -> None:
    """fail-loudly contract per spec §错误处理."""
    with pytest.raises(ProfileRegistryViolationError, match="unknown profile_label"):
        resolve_profile("definitely-not-a-real-profile")


def test_capability_applies_to_owner_matches_final_rollout_config_fields() -> None:
    """Every capability's applies_to_owner must be a FinalRolloutConfig field.

    This catches typos like 'temporal_abstraction' (wrong) vs 'temporal' (right)
    early — otherwise A3 capability_wiring lookup would silently fall through
    to the default wiring_level, hiding the bug.
    """
    field_names = {f.name for f in dataclasses.fields(FinalRolloutConfig)}
    reg = builtin_profile_registry()
    mismatched: list[str] = []
    for cap_name in list_builtin_capabilities():
        cap = reg._capabilities[cap_name]  # noqa: SLF001 — registry inspection
        if cap.applies_to_owner not in field_names:
            mismatched.append(
                f"capability {cap_name!r} targets owner {cap.applies_to_owner!r} "
                f"which is not a FinalRolloutConfig field"
            )
    assert not mismatched, (
        "capability applies_to_owner does not match FinalRolloutConfig fields:\n"
        + "\n".join(f"  - {m}" for m in mismatched)
    )


def test_conflicting_capabilities_blocked_in_topo_order() -> None:
    """Capabilities marked conflicts_with cannot coexist in the same profile.

    Verified indirectly: validate() at module load must have caught any
    accidental coexistence among _BUILTIN_PROFILES. Here we also assert
    that 'no-semantic-label-temporal-policy' and 'no-reflection-cache-
    temporal-policy' are correctly marked conflicting.
    """
    reg = builtin_profile_registry()
    cap_a = reg._capabilities["no-semantic-label-temporal-policy"]  # noqa: SLF001
    cap_b = reg._capabilities["no-reflection-cache-temporal-policy"]  # noqa: SLF001
    assert (
        "no-semantic-label-temporal-policy" in cap_b.conflicts_with
        or "no-reflection-cache-temporal-policy" in cap_a.conflicts_with
    ), "the two temporal-policy capabilities must be marked conflicts_with"


def test_capability_flag_overrides_are_immutable() -> None:
    """frozen dataclass + MappingProxyType prevents post-construction mutation
    (no-swallow-errors-no-hasattr-abuse: SSOT must be tamper-proof)."""
    resolved = resolve_profile("atlas-titans-cms-uplift")
    cap = resolved.capabilities[0]
    with pytest.raises(TypeError):
        # type: ignore[index]
        cap.flag_overrides["new_key"] = "value"


def test_merged_flag_overrides_resolution_order() -> None:
    """When a profile has multiple capabilities, merged_flag_overrides must
    follow topological order (requires-before; siblings preserve registration
    order)."""
    resolved = resolve_profile("pe-eta-online-only")
    # online-only has two capabilities; both override different keys, so the
    # merge is simply a union
    keys = set(resolved.merged_flag_overrides.keys())
    assert "reflection_mode" in keys
    assert "rare_heavy_enabled" in keys


def test_resolve_returns_topologically_sorted_capabilities() -> None:
    """If a capability declares requires, the dependency must appear before
    it in the resolved tuple."""
    # Our built-in capabilities don't currently declare requires, so this test
    # exercises the trivial case (input order = topo order). Future capabilities
    # introduced by COG-1 / COG-3 etc. will exercise the non-trivial path.
    resolved = resolve_profile("pe-eta-online-only")
    names = [c.name for c in resolved.capabilities]
    # With no requires edges, input order is preserved.
    assert names == ["reflection-proposal-only", "rare-heavy-off"]


def test_validate_with_known_owner_slots_passes() -> None:
    """When called with the FinalRolloutConfig field set, every capability's
    applies_to_owner must be present — a stricter form of
    test_capability_applies_to_owner_matches_final_rollout_config_fields."""
    field_names = frozenset(f.name for f in dataclasses.fields(FinalRolloutConfig))
    # Reuse the singleton (already validated without owner check at import).
    reg = builtin_profile_registry()
    # Should not raise.
    reg.validate(known_owner_slots=field_names)


def test_registry_rejects_duplicate_profile_registration() -> None:
    """register_profile is fail-loudly on duplicates."""
    from volvence_zero.agent.profile_registry import ProfileRegistry, ProfileSpec

    reg = ProfileRegistry()
    reg.register_profile(ProfileSpec(label="x"))
    with pytest.raises(ProfileRegistryViolationError, match="already registered"):
        reg.register_profile(ProfileSpec(label="x"))


def test_registry_rejects_alias_collision() -> None:
    """An alias that collides with a canonical label must fail-loudly."""
    from volvence_zero.agent.profile_registry import ProfileRegistry, ProfileSpec

    reg = ProfileRegistry()
    reg.register_profile(ProfileSpec(label="canon-a"))
    # Different canonical, aliasing to canon-a — must fail
    with pytest.raises(ProfileRegistryViolationError, match="alias"):
        reg.register_profile(ProfileSpec(label="canon-b", aliases=("canon-a",)))


def test_registry_rejects_cyclic_requires() -> None:
    """fail-loudly on requires cycle per spec §错误处理."""
    from volvence_zero.agent.profile_registry import (
        ProfileCapability,
        ProfileRegistry,
        ProfileSpec,
    )

    reg = ProfileRegistry()
    reg.register_capability(
        ProfileCapability(name="a", applies_to_owner="evaluation", requires=("b",))
    )
    reg.register_capability(
        ProfileCapability(name="b", applies_to_owner="evaluation", requires=("a",))
    )
    reg.register_profile(ProfileSpec(label="cycle", capabilities=("a", "b")))
    with pytest.raises(ProfileRegistryViolationError, match="cycle"):
        reg.validate()


def test_wiring_level_enum_values_are_canonical() -> None:
    """Sanity: spec assumes the 3 values match exactly."""
    assert {WiringLevel.DISABLED, WiringLevel.SHADOW, WiringLevel.ACTIVE} == set(
        WiringLevel
    )
