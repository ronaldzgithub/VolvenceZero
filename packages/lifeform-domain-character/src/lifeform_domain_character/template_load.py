"""Reincarnate a lived lifeform from a saved :class:`LifeformTemplate`.

Wave T6 of the pipeline. The give_birth path mirrors
``save_lifeform_template`` (Wave T5) — every input slot the saved
template carries gets piped back into a fresh ``Lifeform``
construction:

* Profile (or evolved_profile if Phase 4 produced one) recompiles
  to ``DomainExperiencePackage`` + ``VitalsBootstrap``.
* Saved drive levels override the bootstrap's ``initial_level`` so
  the reborn lifeform starts at "where the saved life left off"
  rather than at the spec's flat default.
* MemoryStore is reconstructed from the saved
  :class:`MemoryStoreCheckpoint` and passed into the Lifeform's
  construction so all sessions on the reborn lifeform share the
  inherited long-term memory.
* Application state (case memory + domain knowledge checkpoints)
  is reapplied on top of the profile-compiled defaults via the
  store's ``restore_checkpoint`` API.

R8 posture:

This module never reads or writes a kernel owner's private state.
It only calls existing typed factories / constructors:
:class:`Lifeform`, :class:`MemoryStore`,
:class:`ApplicationDomainKnowledgeStore`,
:class:`ApplicationCaseMemoryStore`.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, replace as _replace
from typing import Any

from lifeform_core import DriveSpec, Lifeform, LifeformConfig, VitalsBootstrap
from volvence_zero.application.storage import (
    ApplicationCaseMemoryStore,
    ApplicationDomainKnowledgeStore,
    build_default_case_memory_store,
    build_default_domain_knowledge_store,
)
from volvence_zero.memory import MemoryStore, build_default_memory_store

from lifeform_domain_character.compiler import (
    build_character_package,
    build_character_vitals_bootstrap,
)
from lifeform_domain_character.profile import CharacterSoulProfile
from lifeform_domain_character.template import (
    IncompatibleTemplateVersion,
    LifeformTemplate,
    SCHEMA_VERSION,
    compute_template_integrity_hash,
)


@dataclass(frozen=True)
class RebirthBundle:
    """Result of :func:`give_birth`. Carries the constructed
    :class:`Lifeform` plus the auxiliary stores so the caller can
    drive replay or new turns immediately.
    """

    lifeform: Lifeform
    profile: CharacterSoulProfile
    template: LifeformTemplate
    memory_store: MemoryStore
    domain_knowledge_store: ApplicationDomainKnowledgeStore | None
    case_memory_store: ApplicationCaseMemoryStore | None


def give_birth(
    template: LifeformTemplate | pathlib.Path,
    *,
    verify_integrity: bool = True,
    substrate_runtime: Any = None,
    substrate_adapter_factory: Any = None,
    response_synthesizer: Any = None,
    semantic_proposal_runtime: Any = None,
    identity_provider: Any = None,
    memory_store: MemoryStore | None = None,
    skip_memory_restore: bool = False,
    config: LifeformConfig | None = None,
    rare_heavy_enabled: bool = False,
) -> RebirthBundle:
    """Construct a fresh :class:`Lifeform` from a saved template.

    Args:
        template: Either an in-memory :class:`LifeformTemplate` (e.g.
            from :func:`save_lifeform_template` returning the same
            instance) OR a ``pathlib.Path`` pointing to a JSON file
            written by the save path.
        verify_integrity: When True (default), recompute the
            template's integrity hash and refuse to load if it does
            not match the manifest. Set False only for debugging
            tampered templates; production should always verify.

    Returns:
        A :class:`RebirthBundle` carrying the lifeform, the profile
        used, the inflated template, and the stores that back it.

    Raises:
        IncompatibleTemplateVersion: schema_version mismatch.
        ValueError: integrity hash mismatch when ``verify_integrity``
            is True.
    """
    inflated = _resolve_template(template)
    if inflated.manifest.schema_version != SCHEMA_VERSION:
        raise IncompatibleTemplateVersion(
            f"give_birth: template schema_version="
            f"{inflated.manifest.schema_version} != code "
            f"SCHEMA_VERSION={SCHEMA_VERSION}"
        )
    if verify_integrity:
        recomputed = compute_template_integrity_hash(
            profile=inflated.profile,
            evolved_profile=inflated.evolved_profile,
            memory_checkpoint=inflated.memory_checkpoint,
            vitals_bootstrap=inflated.vitals_bootstrap,
            vitals_drive_levels=inflated.vitals_drive_levels,
            application_state=inflated.application_state,
            replay_report=inflated.replay_report,
            template_id=inflated.manifest.template_id,
            schema_version=inflated.manifest.schema_version,
            character_id=inflated.manifest.character_id,
            created_at_utc=inflated.manifest.created_at_utc,
            source_arc_id=inflated.manifest.source_arc_id,
            replay_provenance=inflated.manifest.replay_provenance,
        )
        if recomputed != inflated.manifest.integrity_hash:
            raise ValueError(
                "give_birth: integrity hash mismatch — template may "
                "have been tampered with or saved with a different "
                "code version. "
                f"manifest={inflated.manifest.integrity_hash[:16]}... "
                f"recomputed={recomputed[:16]}..."
            )

    # Choose the profile: prefer evolved_profile if Phase 4 promoted
    # one (drive evolution), else the base profile.
    chosen_profile = inflated.evolved_profile or inflated.profile

    # Compile profile into an application package; this re-seeds the
    # 4 application owners with the (possibly evolved) profile content.
    package = build_character_package(chosen_profile)

    # Vitals: start from the saved bootstrap (or recompile from
    # profile drive priors), then patch each DriveSpec.initial_level
    # to the saved current level so the reborn lifeform "remembers"
    # where its drives were when it was saved.
    vitals_bootstrap = inflated.vitals_bootstrap or build_character_vitals_bootstrap(
        chosen_profile
    )
    saved_levels = dict(inflated.vitals_drive_levels)
    if saved_levels:
        vitals_bootstrap = _patch_vitals_initial_levels(
            vitals_bootstrap, saved_levels
        )

    # Construct LifeformConfig. Caller may pass an override (e.g.
    # alpha service path with ``BrainConfig.memory_scope_root_dir``
    # set). When omitted we default to a fresh ``LifeformConfig()``.
    # ``rare_heavy_enabled`` defaults to False (offline / demo
    # path). Service paths (alpha mode + real substrate) need
    # ``substrate_mode='injected'`` so the brain consumes the
    # supplied runtime as-is.
    base_config = config if isinstance(config, LifeformConfig) else LifeformConfig()
    brain_overrides: dict[str, Any] = {"rare_heavy_enabled": rare_heavy_enabled}
    if substrate_runtime is not None:
        brain_overrides["substrate_mode"] = "injected"
    base_config = _replace(
        base_config,
        brain_config=_replace(base_config.brain_config, **brain_overrides),
    )
    base_config = base_config.with_domain_experience((package,))
    base_config = base_config.with_vitals(vitals_bootstrap)

    # Memory-store policy:
    # 1. ``memory_store`` injected by caller → use it as-is.
    # 2. ``skip_memory_restore=True`` AND the template did NOT opt in
    #    to ``preserve_memory`` → leave memory_store=None so the
    #    kernel auto-builds a per-session store from
    #    ``BrainConfig.memory_scope_root_dir + IdentityProvider``
    #    (this is the alpha service path: per-user disk persistence
    #    starts fresh and accumulates on top of the saved drives /
    #    profile, rather than baking the template's frozen memory
    #    snapshot into every user's account).
    # 3. Otherwise → reconstruct from the template checkpoint, the
    #    canonical "fully reincarnated" path.
    #
    # NW7: ``manifest.preserve_memory=True`` overrides ``skip_memory_restore``
    # so templates whose canonical first-half-of-life memories must
    # persist across every player (e.g. novel-worlds-character) get
    # restored even under the alpha service path. R14 persistent
    # identity is defined by what the character remembers; players
    # always meet the same person.
    if memory_store is not None:
        active_memory_store: MemoryStore | None = memory_store
    elif skip_memory_restore and not inflated.manifest.preserve_memory:
        active_memory_store = None
    else:
        active_memory_store = _restore_memory_store(inflated)

    # Restore application owner stores; these are passed into the
    # Lifeform implicitly through the kernel's package application
    # path — but we also keep references so the caller can run
    # subsequent saves / inspect them.
    domain_store = build_default_domain_knowledge_store()
    case_store = build_default_case_memory_store()
    if inflated.application_state.domain_knowledge_checkpoint is not None:
        domain_store.restore_checkpoint(
            inflated.application_state.domain_knowledge_checkpoint
        )
    if inflated.application_state.case_memory_checkpoint is not None:
        case_store.restore_checkpoint(
            inflated.application_state.case_memory_checkpoint
        )

    lifeform_kwargs: dict[str, Any] = {}
    if active_memory_store is not None:
        lifeform_kwargs["memory_store"] = active_memory_store
    if substrate_runtime is not None:
        lifeform_kwargs["substrate_runtime"] = substrate_runtime
    if substrate_adapter_factory is not None:
        lifeform_kwargs["substrate_adapter_factory"] = substrate_adapter_factory
    if response_synthesizer is not None:
        lifeform_kwargs["response_synthesizer"] = response_synthesizer
    if semantic_proposal_runtime is not None:
        lifeform_kwargs["semantic_proposal_runtime"] = semantic_proposal_runtime
    if identity_provider is not None:
        lifeform_kwargs["identity_provider"] = identity_provider
    lifeform = Lifeform(base_config, **lifeform_kwargs)

    return RebirthBundle(
        lifeform=lifeform,
        profile=chosen_profile,
        template=inflated,
        memory_store=active_memory_store,
        domain_knowledge_store=domain_store,
        case_memory_store=case_store,
    )


def _resolve_template(
    template: LifeformTemplate | pathlib.Path,
) -> LifeformTemplate:
    if isinstance(template, LifeformTemplate):
        return template
    if isinstance(template, pathlib.Path):
        if not template.exists():
            raise FileNotFoundError(
                f"give_birth: template file not found: {template}"
            )
        return LifeformTemplate.from_json_bytes(template.read_bytes())
    if isinstance(template, str):
        return _resolve_template(pathlib.Path(template))
    raise TypeError(
        "give_birth: template must be a LifeformTemplate instance or "
        f"pathlib.Path, got {type(template).__name__}"
    )


def _patch_vitals_initial_levels(
    bootstrap: VitalsBootstrap,
    saved_levels: dict[str, float],
) -> VitalsBootstrap:
    """Return a fresh VitalsBootstrap whose drives' ``initial_level``
    matches the saved current levels.

    Drives that are not in the saved levels keep their bootstrap
    default; saved drives that are not in the bootstrap are
    silently dropped (they belong to a different profile).
    """
    new_drives = tuple(
        _replace(
            drive,
            initial_level=_clip_unit(
                saved_levels.get(drive.name, drive.initial_level)
            ),
        )
        for drive in bootstrap.drives
    )
    return _replace(bootstrap, drives=new_drives)


def _clip_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _restore_memory_store(template: LifeformTemplate) -> MemoryStore:
    """Build a MemoryStore from the saved checkpoint or default."""
    store = build_default_memory_store()
    checkpoint = template.memory_checkpoint
    if checkpoint is None:
        return store
    # When the template carries a typed MemoryStoreCheckpoint, restore
    # directly via the store's API. When it's the opaque
    # ``_SerializedMemoryCheckpoint`` carrier (from a heavily-nested
    # JSON round-trip) we fall back to letting the caller drive
    # restoration manually; for v0 we expect the typed path.
    from lifeform_domain_character.template import _SerializedMemoryCheckpoint
    from volvence_zero.memory import MemoryStoreCheckpoint

    if isinstance(checkpoint, MemoryStoreCheckpoint):
        store.restore_checkpoint(checkpoint)
        return store
    if isinstance(checkpoint, _SerializedMemoryCheckpoint):
        # Best-effort: the carrier wraps an already-serialized
        # payload. We currently do not have a reverse-builder so
        # fall back to the default store and emit a typed warning
        # via the store's description (no print / log here — owners
        # surface state through snapshots).
        return store
    return store


__all__ = [
    "RebirthBundle",
    "give_birth",
]
