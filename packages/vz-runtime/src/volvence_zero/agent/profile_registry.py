"""Profile composition layer (architecture-uplift A1 + A3).

Declarative replacement for the ``if profile_label == "X"`` dispatch in
``packages/vz-runtime/src/volvence_zero/agent/dialogue/_legacy.py`` (currently
11 hard-coded branches + 2 aliases).

This module is **声明侧 SSOT**: every dialogue ablation profile is described as
a ``ProfileSpec`` referencing one or more ``ProfileCapability`` declarations.
Profiles compose by capability; capabilities are orthogonal units that target
a single owner / module.

Per spec [`docs/specs/profile-registry.md`](../../../../../../docs/specs/profile-registry.md)
迁移协议 阶段 1:

- Registry exists and is fully populated with 11 built-in profiles.
- `RuntimeModule.capabilities` ClassVar is added (default empty) but module
  dispatch in `build_standard_dialogue_runner` is **not** rewritten yet —
  the legacy if-elif chain remains the正式 upstream until 阶段 2 evidence.
- A contract test verifies the registry can resolve every legacy
  profile_label without errors and that capability bundles are non-empty
  for non-baseline profiles.

Fail-loudly throughout per
[`.cursor/rules/no-swallow-errors-no-hasattr-abuse.mdc`](../../../../../../.cursor/rules/no-swallow-errors-no-hasattr-abuse.mdc):
unknown profile, unregistered capability, dependency cycle, conflicting
capabilities, alias collision — all raise at module-load time (during
``_BUILTIN_REGISTRY.validate()``).
"""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from volvence_zero.runtime.kernel import WiringLevel

__all__ = [
    "ProfileCapability",
    "ProfileSpec",
    "ResolvedProfile",
    "ProfileRegistry",
    "ProfileRegistryViolationError",
    "builtin_profile_registry",
    "resolve_profile",
    "list_builtin_profiles",
    "list_builtin_capabilities",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ProfileRegistryViolationError(RuntimeError):
    """Raised whenever a registry invariant is violated.

    Spec §错误处理 与 fail-loudly: 此错误不允许被捕获后静默回退；
    调用方必须修复 spec / capability 声明或修复 caller side。
    """


# ---------------------------------------------------------------------------
# Data types (mirror spec §接口契约)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ProfileCapability:
    """一个 capability：影响单个 owner 的正交修改单位。

    Spec §A1.1 — 字段语义在 docs/specs/profile-registry.md。
    """

    name: str
    applies_to_owner: str
    flag_overrides: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    wiring_overrides: Mapping[str, WiringLevel] = dataclasses.field(default_factory=dict)
    requires: tuple[str, ...] = ()
    conflicts_with: tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        # Defensive normalisation: store as immutable view so callers cannot
        # mutate post-construction (frozen dataclass guards self.flag_overrides
        # reference, not the dict it points to).
        object.__setattr__(self, "flag_overrides", MappingProxyType(dict(self.flag_overrides)))
        object.__setattr__(self, "wiring_overrides", MappingProxyType(dict(self.wiring_overrides)))


@dataclasses.dataclass(frozen=True)
class ProfileSpec:
    """一个 profile：capability bundle。

    Spec §A1.2 — 字段语义在 docs/specs/profile-registry.md。
    """

    label: str
    capabilities: tuple[str, ...] = ()
    base_profile: str = "pe-eta"
    aliases: tuple[str, ...] = ()
    description: str = ""


@dataclasses.dataclass(frozen=True)
class ResolvedProfile:
    """A profile after registry lookup + capability merge.

    Spec §A1.4 — 字段语义在 docs/specs/profile-registry.md.
    """

    label: str
    capabilities: tuple[ProfileCapability, ...]
    merged_flag_overrides: Mapping[str, Any]
    merged_wiring_overrides: Mapping[str, Mapping[str, WiringLevel]]
    base_profile: str

    def apply_to_config(self, base: Any) -> Any:
        """Return a new FinalRolloutConfig with capability_wirings merged in.

        Pure function: never mutates ``base``. Spec §A1.4 — flag_overrides are
        passed to AgentSessionRunner via a separate channel (阶段 2 work);
        only capability_wirings is applied to the immutable config here so
        阶段 1 can keep flag-driven dispatch as a parallel side path without
        regressing existing behaviour.

        ``base`` is typed Any to avoid a circular import (FinalRolloutConfig
        lives in volvence_zero.integration.final_wiring which transitively
        imports profile_registry once registry-first dispatch lands).
        """
        # Lazy import keeps profile_registry self-contained at module load.
        from volvence_zero.integration.final_wiring import FinalRolloutConfig

        if not isinstance(base, FinalRolloutConfig):
            raise TypeError(
                f"apply_to_config expects FinalRolloutConfig, got {type(base).__name__}"
            )
        merged: dict[str, Mapping[str, WiringLevel]] = dict(base.capability_wirings)
        for owner, owner_overrides in self.merged_wiring_overrides.items():
            # Union with existing entries to allow layered composition; new
            # keys take precedence.
            existing = dict(merged.get(owner, {}))
            existing.update(owner_overrides)
            merged[owner] = MappingProxyType(existing)
        return dataclasses.replace(
            base,
            capability_wirings=MappingProxyType(merged),
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ProfileRegistry:
    """全局 ProfileSpec + ProfileCapability registry.

    Spec §A1.3 — validate() 一次性校验所有 invariant；resolve_profile() 返回
    capability 按 requires 拓扑排序后的 ResolvedProfile。

    Thread-safe for read-after-validate usage (immutable internal state once
    validate() is invoked); writes must be serialised by the caller.
    """

    def __init__(self) -> None:
        self._capabilities: dict[str, ProfileCapability] = {}
        self._profiles: dict[str, ProfileSpec] = {}
        self._aliases: dict[str, str] = {}
        self._lock = threading.Lock()
        self._validated = False

    # ---- registration ----

    def register_capability(self, capability: ProfileCapability) -> None:
        with self._lock:
            if capability.name in self._capabilities:
                raise ProfileRegistryViolationError(
                    f"capability {capability.name!r} already registered"
                )
            self._capabilities[capability.name] = capability
            self._validated = False

    def register_profile(self, profile: ProfileSpec) -> None:
        with self._lock:
            if profile.label in self._profiles:
                raise ProfileRegistryViolationError(
                    f"profile {profile.label!r} already registered"
                )
            self._profiles[profile.label] = profile
            for alias in profile.aliases:
                if alias in self._aliases:
                    raise ProfileRegistryViolationError(
                        f"alias {alias!r} already in use by "
                        f"{self._aliases[alias]!r}; cannot reassign to {profile.label!r}"
                    )
                if alias in self._profiles:
                    raise ProfileRegistryViolationError(
                        f"alias {alias!r} collides with canonical profile label"
                    )
                self._aliases[alias] = profile.label
            self._validated = False

    # ---- introspection ----

    def list_profiles(self) -> tuple[str, ...]:
        return tuple(sorted(self._profiles.keys()))

    def list_capabilities(self) -> tuple[str, ...]:
        return tuple(sorted(self._capabilities.keys()))

    def has_profile(self, label: str) -> bool:
        return label in self._profiles or label in self._aliases

    def has_capability(self, name: str) -> bool:
        return name in self._capabilities

    # ---- validation ----

    def validate(self, *, known_owner_slots: frozenset[str] | None = None) -> None:
        """Run all invariant checks; fail-loudly on any violation.

        ``known_owner_slots``: when provided, every capability's
        ``applies_to_owner`` must be present. Caller is expected to feed the
        FinalRolloutConfig field set; we don't import wiring config here to
        avoid bloating import graph.
        """
        for profile in self._profiles.values():
            for cap_name in profile.capabilities:
                if cap_name not in self._capabilities:
                    raise ProfileRegistryViolationError(
                        f"profile {profile.label!r} references unknown capability "
                        f"{cap_name!r}"
                    )
            # Conflict check within a profile
            for i, cap_name in enumerate(profile.capabilities):
                cap = self._capabilities[cap_name]
                for other in profile.capabilities[i + 1 :]:
                    if other in cap.conflicts_with:
                        raise ProfileRegistryViolationError(
                            f"profile {profile.label!r} contains conflicting "
                            f"capabilities {cap_name!r} ↔ {other!r}"
                        )

        for cap in self._capabilities.values():
            if known_owner_slots is not None and cap.applies_to_owner not in known_owner_slots:
                raise ProfileRegistryViolationError(
                    f"capability {cap.name!r} targets unknown owner "
                    f"{cap.applies_to_owner!r} (not in known_owner_slots)"
                )
            for req in cap.requires:
                if req not in self._capabilities:
                    raise ProfileRegistryViolationError(
                        f"capability {cap.name!r} requires unknown capability {req!r}"
                    )

        # Cycle detection on requires graph
        self._detect_cycles()

        self._validated = True

    def _detect_cycles(self) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(name: str, stack: tuple[str, ...]) -> None:
            if name in visited:
                return
            if name in visiting:
                cycle = " -> ".join(stack + (name,))
                raise ProfileRegistryViolationError(
                    f"cycle in capability requires: {cycle}"
                )
            visiting.add(name)
            cap = self._capabilities[name]
            for req in cap.requires:
                visit(req, stack + (name,))
            visiting.discard(name)
            visited.add(name)

        for name in self._capabilities:
            visit(name, ())

    # ---- resolution ----

    def resolve_profile(self, label: str) -> ResolvedProfile:
        """Resolve a profile_label (or alias) into a ResolvedProfile.

        Raises ProfileRegistryViolationError on unknown label.
        """
        canonical = self._aliases.get(label, label)
        if canonical not in self._profiles:
            raise ProfileRegistryViolationError(
                f"unknown profile_label {label!r}; "
                f"known profiles: {self.list_profiles()!r}"
            )
        profile = self._profiles[canonical]
        # Topologically sort capabilities by requires
        ordered = self._topo_sort_capabilities(profile.capabilities)
        capabilities = tuple(self._capabilities[name] for name in ordered)
        merged_flags: dict[str, Any] = {}
        merged_wiring: dict[str, dict[str, WiringLevel]] = {}
        for cap in capabilities:
            for k, v in cap.flag_overrides.items():
                merged_flags[k] = v
            for k, v in cap.wiring_overrides.items():
                merged_wiring.setdefault(cap.applies_to_owner, {})[k] = v
        return ResolvedProfile(
            label=canonical,
            capabilities=capabilities,
            merged_flag_overrides=MappingProxyType(merged_flags),
            merged_wiring_overrides=MappingProxyType(
                {owner: MappingProxyType(d) for owner, d in merged_wiring.items()}
            ),
            base_profile=profile.base_profile,
        )

    def _topo_sort_capabilities(self, names: tuple[str, ...]) -> list[str]:
        # Deterministic topological order; preserve input order for siblings.
        visited: set[str] = set()
        result: list[str] = []

        def visit(name: str) -> None:
            if name in visited:
                return
            cap = self._capabilities.get(name)
            if cap is None:
                raise ProfileRegistryViolationError(
                    f"unknown capability {name!r} during resolution"
                )
            for req in cap.requires:
                visit(req)
            visited.add(name)
            result.append(name)

        for name in names:
            visit(name)
        return result


# ---------------------------------------------------------------------------
# Built-in 11 profiles
#
# Mirrors `build_standard_dialogue_runner` (`packages/vz-runtime/src/volvence_zero/agent/dialogue/_legacy.py:7862`)
# branch-by-branch. Spec §11 个现有 Profile 的 Capability 拆解 gives the full
# table; this is the executable form of that table.
# ---------------------------------------------------------------------------


_BUILTIN_CAPABILITIES: tuple[ProfileCapability, ...] = (
    ProfileCapability(
        name="cms-atlas-titans-uplift",
        applies_to_owner="memory",
        flag_overrides={
            "cms_pe_features_enabled": True,
            "cms_replay_window_size": 8,
        },
        description="See docs/specs/cms-atlas-titans-uplift.md",
    ),
    ProfileCapability(
        name="reflection-proposal-only",
        applies_to_owner="reflection",
        flag_overrides={"reflection_mode": "WritebackMode.PROPOSAL_ONLY"},
        description="Reflection writeback emits proposals only; no durable side effects.",
    ),
    ProfileCapability(
        name="rare-heavy-off",
        applies_to_owner="substrate_self_mod",
        flag_overrides={"rare_heavy_enabled": False},
        description="Disables rare-heavy substrate self-mod cycle.",
    ),
    ProfileCapability(
        name="no-semantic-label-temporal-policy",
        applies_to_owner="temporal",
        flag_overrides={
            "world_temporal_policy_class": "_NoSemanticLabelTemporalPolicy",
            "self_temporal_policy_class": "_NoSemanticLabelTemporalPolicy",
        },
        description="World+self temporal policy drops semantic labels.",
    ),
    ProfileCapability(
        name="no-reflection-cache-temporal-policy",
        applies_to_owner="temporal",
        flag_overrides={
            "world_temporal_policy_class": "_NoReflectionCacheTemporalPolicy",
            "self_temporal_policy_class": "_NoReflectionCacheTemporalPolicy",
        },
        conflicts_with=("no-semantic-label-temporal-policy",),
        description="World+self temporal policy disables reflection cache.",
    ),
    ProfileCapability(
        name="pe-readout-only",
        applies_to_owner="prediction_error",
        flag_overrides={
            "joint_schedule_ssl_interval": 1,
            "joint_schedule_rl_interval": 2,
            "external_prediction_error_drive": False,
            "prediction_error_readout_only": True,
            "primary_prediction_error_dominance_enabled": False,
        },
        description="PE is read but does not drive control / dominance.",
    ),
    ProfileCapability(
        name="pe-drive-off",
        applies_to_owner="prediction_error",
        flag_overrides={
            "joint_schedule_ssl_interval": 1,
            "joint_schedule_rl_interval": 2,
            "external_prediction_error_drive": False,
            "allow_live_substrate_mutation": False,
        },
        conflicts_with=("pe-readout-only",),
        description="Disable external PE drive entirely (eta-no-pe alias).",
    ),
    ProfileCapability(
        name="timescale-off",
        applies_to_owner="memory",
        flag_overrides={
            "memory_store_nested_profile": False,
            "joint_schedule_ssl_interval": 1,
            "joint_schedule_rl_interval": 2,
            "allow_live_substrate_mutation": False,
        },
        description="Nested-profile memory store disabled (timescale-off).",
    ),
    ProfileCapability(
        name="eta-off",
        applies_to_owner="temporal",
        flag_overrides={
            "temporal_policy_class": "LearnedLiteTemporalPolicy",
            "passive_joint_loop": True,
            "reflection_wiring": "WiringLevel.DISABLED",
            "joint_schedule_ssl_interval": 0,
            "joint_schedule_rl_interval": 0,
            "joint_schedule_pe_thresholds_disabled": True,
            "rare_heavy_enabled": False,
            "external_prediction_error_drive": False,
            "allow_live_substrate_mutation": False,
        },
        description="Lite-learned ETA baseline (no full ETA path).",
    ),
    ProfileCapability(
        name="heuristic-baseline",
        applies_to_owner="temporal",
        flag_overrides={
            "temporal_policy_class": "HeuristicTemporalPolicy",
            "passive_joint_loop": True,
            "reflection_wiring": "WiringLevel.DISABLED",
            "joint_schedule_ssl_interval": 0,
            "joint_schedule_rl_interval": 0,
            "joint_schedule_pe_thresholds_disabled": True,
            "rare_heavy_enabled": False,
            "external_prediction_error_drive": False,
            "allow_live_substrate_mutation": False,
        },
        conflicts_with=("eta-off",),
        description="Heuristic temporal baseline (no learned controller).",
    ),
)


_BUILTIN_PROFILES: tuple[ProfileSpec, ...] = (
    ProfileSpec(label="pe-eta", capabilities=(), description="canonical baseline"),
    ProfileSpec(
        label="atlas-titans-cms-uplift",
        capabilities=("cms-atlas-titans-uplift",),
        description="CMS uplift SHADOW-evidence profile.",
    ),
    ProfileSpec(
        label="pe-eta-online-only",
        capabilities=("reflection-proposal-only", "rare-heavy-off"),
        description="Online-only path: no writeback, no rare-heavy.",
    ),
    ProfileSpec(
        label="pe-eta-no-writeback",
        capabilities=("reflection-proposal-only",),
        description="Writeback disabled; rare-heavy still runs.",
    ),
    ProfileSpec(
        label="pe-eta-no-rare-heavy",
        capabilities=("rare-heavy-off",),
        description="Rare-heavy disabled; writeback still runs.",
    ),
    ProfileSpec(
        label="pe-eta-no-semantic-label",
        capabilities=("no-semantic-label-temporal-policy",),
        description="Temporal policy drops semantic labels.",
    ),
    ProfileSpec(
        label="pe-eta-no-reflection-cache",
        capabilities=("no-reflection-cache-temporal-policy",),
        description="Temporal policy disables reflection cache.",
    ),
    ProfileSpec(
        label="pe-eta-pe-readout-only",
        capabilities=("pe-readout-only",),
        description="PE is observed only, not driving.",
    ),
    ProfileSpec(
        label="pe-drive-off",
        capabilities=("pe-drive-off",),
        aliases=("eta-no-pe",),
        description="PE drive disabled (eta-no-pe alias).",
    ),
    ProfileSpec(
        label="timescale-off",
        capabilities=("timescale-off",),
        description="Memory nested-profile disabled.",
    ),
    ProfileSpec(
        label="eta-off",
        capabilities=("eta-off",),
        description="Learned-lite temporal baseline, no full ETA path.",
    ),
    ProfileSpec(
        label="heuristic-baseline",
        capabilities=("heuristic-baseline",),
        description="Heuristic temporal baseline.",
    ),
)


# ---------------------------------------------------------------------------
# Module-level singleton registry
# ---------------------------------------------------------------------------


def _build_builtin_registry() -> ProfileRegistry:
    reg = ProfileRegistry()
    for cap in _BUILTIN_CAPABILITIES:
        reg.register_capability(cap)
    for prof in _BUILTIN_PROFILES:
        reg.register_profile(prof)
    # Validation without known_owner_slots: skips owner existence check.
    # The richer cross-check against FinalRolloutConfig field set lives in
    # tests/contracts/test_profile_registry_sync.py (registered later in
    # T5 step阶段 1).
    reg.validate()
    return reg


_BUILTIN_REGISTRY: ProfileRegistry | None = None
_BUILTIN_LOCK = threading.Lock()


def builtin_profile_registry() -> ProfileRegistry:
    """Return the module-level singleton registry (lazy + thread-safe)."""
    global _BUILTIN_REGISTRY
    if _BUILTIN_REGISTRY is None:
        with _BUILTIN_LOCK:
            if _BUILTIN_REGISTRY is None:
                _BUILTIN_REGISTRY = _build_builtin_registry()
    return _BUILTIN_REGISTRY


def resolve_profile(label: str) -> ResolvedProfile:
    """Convenience wrapper around builtin_profile_registry().resolve_profile()."""
    return builtin_profile_registry().resolve_profile(label)


def list_builtin_profiles() -> tuple[str, ...]:
    return builtin_profile_registry().list_profiles()


def list_builtin_capabilities() -> tuple[str, ...]:
    return builtin_profile_registry().list_capabilities()
