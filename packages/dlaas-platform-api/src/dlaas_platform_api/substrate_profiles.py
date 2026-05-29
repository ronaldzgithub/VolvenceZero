"""Substrate profile registry — adoption-time substrate selection.

The adoption contract carries a :class:`SubstrateSelection` with a
``substrate_profile_id`` / ``mode`` / ``adapter_policy``. Historically
the catalog endpoint returned a hardcoded list and nothing validated
the requested profile against the substrate the process actually loaded,
so a tenant could adopt against ``synthetic-dev`` on a GPU-frozen
deployment (or vice versa) and only discover the mismatch at first turn.

This module is the single source of truth for the catalog and for the
adopt/wake validation:

* :class:`SubstrateProfile` — one composable profile descriptor.
* :class:`SubstrateProfileRegistry` — name -> profile, with a default.
* :func:`default_substrate_profile_registry` — the shipped profiles.
* :func:`running_substrate_mode` — derive the running substrate's mode
  from the injected runtime so adopt can fail loud on a mismatch.

R8 / no-swallow-errors: an unknown profile id raises
:class:`UnknownSubstrateProfile` rather than silently defaulting; the
adopt handler turns that into a 400, and a mode mismatch into a 409.
"""

from __future__ import annotations

from dataclasses import dataclass

# Adapter policy vocabulary. ``none`` forbids persona-LoRA registration
# and activation for instances on this profile; ``persona_lora`` permits
# the figure-vertical persona-LoRA overlay (still offline-gated to bake).
ADAPTER_POLICY_NONE = "none"
ADAPTER_POLICY_PERSONA_LORA = "persona_lora"
_VALID_ADAPTER_POLICIES = frozenset(
    {ADAPTER_POLICY_NONE, ADAPTER_POLICY_PERSONA_LORA}
)

# Substrate mode vocabulary. ``shared_frozen`` = a real frozen base
# model shared across tenants; ``synthetic`` = the dependency-free
# synthetic runtime used for dev / CI.
MODE_SHARED_FROZEN = "shared_frozen"
MODE_SYNTHETIC = "synthetic"
_VALID_MODES = frozenset({MODE_SHARED_FROZEN, MODE_SYNTHETIC})

# Runtime backend vocabulary. ``transformers`` = the HF residual runtime
# (single decode in flight, residual capture + steering bake); ``vllm`` =
# the high-throughput multi-LoRA serving engine (concurrent decode, no
# residual capture). Both run the frozen base; the backend is an
# operational choice surfaced on the profile so a tenant / operator can
# pick the serving engine.
BACKEND_TRANSFORMERS = "transformers"
BACKEND_VLLM = "vllm"
_VALID_BACKENDS = frozenset({BACKEND_TRANSFORMERS, BACKEND_VLLM})


class UnknownSubstrateProfile(LookupError):
    """Raised when a requested ``substrate_profile_id`` is not registered."""


@dataclass(frozen=True)
class SubstrateProfile:
    """One composable substrate adoption profile.

    Fields mirror :class:`dlaas_platform_contracts.SubstrateSelection`
    so the catalog can advertise exactly what a tenant may request.

    * ``substrate_profile_id`` — stable catalog id.
    * ``mode`` — ``shared_frozen`` / ``synthetic``.
    * ``adapter_policy`` — ``none`` / ``persona_lora``.
    * ``allow_rare_heavy_refresh`` — whether rare-heavy substrate
      refresh is permitted for instances on this profile (still
      offline-gated; never a live base-weight update).
    * ``model_id_hint`` — informational base model id for operators;
      not used to load weights (the process loads one runtime).
    """

    substrate_profile_id: str
    mode: str
    adapter_policy: str = ADAPTER_POLICY_NONE
    allow_rare_heavy_refresh: bool = False
    model_id_hint: str = ""
    runtime_backend: str = BACKEND_TRANSFORMERS
    # Runtime build params (multi-pod): a pod loads its substrate from
    # these. ``model_id`` empty + ``mode == synthetic`` -> synthetic
    # runtime (no weights). For shared_frozen, ``model_id`` is the HF id.
    model_id: str = ""
    device: str = "auto"
    dtype: str = "auto"
    max_loras: int = 4
    max_lora_rank: int = 16

    def __post_init__(self) -> None:
        if not self.substrate_profile_id.strip():
            raise ValueError(
                "SubstrateProfile.substrate_profile_id must be non-empty"
            )
        if self.mode not in _VALID_MODES:
            raise ValueError(
                f"SubstrateProfile.mode must be one of {sorted(_VALID_MODES)}, "
                f"got {self.mode!r}"
            )
        if self.adapter_policy not in _VALID_ADAPTER_POLICIES:
            raise ValueError(
                "SubstrateProfile.adapter_policy must be one of "
                f"{sorted(_VALID_ADAPTER_POLICIES)}, got {self.adapter_policy!r}"
            )
        if self.runtime_backend not in _VALID_BACKENDS:
            raise ValueError(
                "SubstrateProfile.runtime_backend must be one of "
                f"{sorted(_VALID_BACKENDS)}, got {self.runtime_backend!r}"
            )

    def permits_persona_lora(self) -> bool:
        """Whether persona LoRA may register / activate on this profile."""

        return self.adapter_policy == ADAPTER_POLICY_PERSONA_LORA

    def to_json(self) -> dict[str, object]:
        return {
            "substrate_profile_id": self.substrate_profile_id,
            "mode": self.mode,
            "adapter_policy": self.adapter_policy,
            "allow_rare_heavy_refresh": self.allow_rare_heavy_refresh,
            "model_id_hint": self.model_id_hint,
            "runtime_backend": self.runtime_backend,
            "model_id": self.model_id,
            "device": self.device,
            "dtype": self.dtype,
        }


class SubstrateProfileRegistry:
    """Name -> :class:`SubstrateProfile` with a designated default."""

    def __init__(
        self,
        profiles: tuple[SubstrateProfile, ...],
        *,
        default_profile_id: str,
    ) -> None:
        if not profiles:
            raise ValueError(
                "SubstrateProfileRegistry requires at least one profile"
            )
        index: dict[str, SubstrateProfile] = {}
        for profile in profiles:
            if profile.substrate_profile_id in index:
                raise ValueError(
                    "SubstrateProfileRegistry: duplicate profile id "
                    f"{profile.substrate_profile_id!r}"
                )
            index[profile.substrate_profile_id] = profile
        if default_profile_id not in index:
            raise ValueError(
                "SubstrateProfileRegistry: default_profile_id "
                f"{default_profile_id!r} is not among the registered profiles"
            )
        self._profiles = index
        self._default_profile_id = default_profile_id

    @property
    def default_profile_id(self) -> str:
        return self._default_profile_id

    def has(self, profile_id: str) -> bool:
        return bool(profile_id) and profile_id in self._profiles

    def get(self, profile_id: str) -> SubstrateProfile:
        """Return the profile for ``profile_id`` (default when empty).

        Raises :class:`UnknownSubstrateProfile` for a non-empty id that
        is not registered (fail loud — the adopt handler maps this to a
        400 so a typo cannot silently fall through to the default).
        """

        if not profile_id:
            return self._profiles[self._default_profile_id]
        profile = self._profiles.get(profile_id)
        if profile is None:
            raise UnknownSubstrateProfile(
                f"unknown substrate_profile_id {profile_id!r}; known: "
                f"{sorted(self._profiles)}"
            )
        return profile

    def list(self) -> tuple[SubstrateProfile, ...]:
        return tuple(
            self._profiles[name] for name in sorted(self._profiles)
        )


_DEFAULT_REGISTRY: SubstrateProfileRegistry | None = None


def default_substrate_profile_registry() -> SubstrateProfileRegistry:
    """Return the process-wide default substrate profile registry."""

    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is not None:
        return _DEFAULT_REGISTRY
    _DEFAULT_REGISTRY = SubstrateProfileRegistry(
        profiles=(
            SubstrateProfile(
                substrate_profile_id="shared-frozen",
                mode=MODE_SHARED_FROZEN,
                adapter_policy=ADAPTER_POLICY_NONE,
                allow_rare_heavy_refresh=False,
            ),
            SubstrateProfile(
                substrate_profile_id="shared-frozen-persona-lora",
                mode=MODE_SHARED_FROZEN,
                adapter_policy=ADAPTER_POLICY_PERSONA_LORA,
                allow_rare_heavy_refresh=False,
            ),
            SubstrateProfile(
                substrate_profile_id="synthetic-dev",
                mode=MODE_SYNTHETIC,
                adapter_policy=ADAPTER_POLICY_NONE,
                allow_rare_heavy_refresh=False,
            ),
            SubstrateProfile(
                substrate_profile_id="synthetic-dev-persona-lora",
                mode=MODE_SYNTHETIC,
                adapter_policy=ADAPTER_POLICY_PERSONA_LORA,
                allow_rare_heavy_refresh=False,
            ),
            SubstrateProfile(
                substrate_profile_id="vllm-shared-frozen-persona-lora",
                mode=MODE_SHARED_FROZEN,
                adapter_policy=ADAPTER_POLICY_PERSONA_LORA,
                allow_rare_heavy_refresh=False,
                runtime_backend=BACKEND_VLLM,
            ),
        ),
        default_profile_id="shared-frozen",
    )
    return _DEFAULT_REGISTRY


def build_runtime_for_profile(profile: SubstrateProfile) -> object | None:
    """Build the substrate runtime a pod should load for ``profile``.

    * ``mode == synthetic`` -> ``None`` (each session builds its own
      synthetic runtime; matches DLaaS synthetic mode).
    * ``runtime_backend == vllm`` -> ``VLLMOpenWeightResidualRuntime``
      (requires the optional ``vz-substrate[vllm]`` extra + GPU).
    * otherwise -> a frozen transformers runtime via
      ``build_transformers_runtime_with_fallback``
      (``allow_live_substrate_mutation=False`` for R2 sharing).

    Heavy substrate imports are deferred to call time so importing this
    module stays cheap.
    """

    if profile.mode == MODE_SYNTHETIC:
        return None
    if not profile.model_id.strip():
        raise ValueError(
            f"substrate profile {profile.substrate_profile_id!r} is "
            f"shared_frozen but has no model_id to load."
        )
    if profile.runtime_backend == BACKEND_VLLM:
        from volvence_zero.substrate import VLLMOpenWeightResidualRuntime

        return VLLMOpenWeightResidualRuntime(
            model_id=profile.model_id,
            max_loras=profile.max_loras,
            max_lora_rank=profile.max_lora_rank,
            dtype=profile.dtype,
        )
    from volvence_zero.substrate import (
        build_transformers_runtime_with_fallback,
    )

    return build_transformers_runtime_with_fallback(
        model_id=profile.model_id,
        device=profile.device,
        allow_live_substrate_mutation=False,
    )


def running_substrate_mode(runtime: object | None) -> str:
    """Derive the running substrate's mode from the injected runtime.

    DLaaS synthetic mode injects ``None`` (each session builds its own
    synthetic runtime); an explicitly injected synthetic runtime also
    counts as synthetic. Anything else is a real shared frozen base.
    """

    if runtime is None:
        return MODE_SYNTHETIC
    from volvence_zero.substrate import SyntheticOpenWeightResidualRuntime

    if isinstance(runtime, SyntheticOpenWeightResidualRuntime):
        return MODE_SYNTHETIC
    return MODE_SHARED_FROZEN


def running_substrate_backend(runtime: object | None) -> str:
    """Return the running substrate's backend id, or "" for synthetic/none."""

    if runtime is None:
        return ""
    try:
        from volvence_zero.substrate import (
            SyntheticOpenWeightResidualRuntime,
            VLLMOpenWeightResidualRuntime,
        )
    except ImportError:
        return ""
    if isinstance(runtime, SyntheticOpenWeightResidualRuntime):
        return ""
    if isinstance(runtime, VLLMOpenWeightResidualRuntime):
        return BACKEND_VLLM
    return BACKEND_TRANSFORMERS


__all__ = [
    "ADAPTER_POLICY_NONE",
    "ADAPTER_POLICY_PERSONA_LORA",
    "BACKEND_TRANSFORMERS",
    "BACKEND_VLLM",
    "MODE_SHARED_FROZEN",
    "MODE_SYNTHETIC",
    "SubstrateProfile",
    "SubstrateProfileRegistry",
    "UnknownSubstrateProfile",
    "build_runtime_for_profile",
    "default_substrate_profile_registry",
    "running_substrate_mode",
    "running_substrate_backend",
]
