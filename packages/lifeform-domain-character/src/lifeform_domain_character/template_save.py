"""Save a lived character lifeform's state into a :class:`LifeformTemplate`.

Wave T5 of the Lifeform Template + Birth pipeline. The save path is
deliberately explicit about its inputs — the caller passes references
to the same stores it used to construct the lifeform — so we never
reach into private kernel state. This keeps R8 satisfied: every
piece of state in the saved template comes from an existing
owner-side export API (or from the caller's own reference to a
mutable store like ``MemoryStore``).

Use site:

    from lifeform_domain_character import (
        ExperientialReplayDriver,
        build_zhang_wuji_demo_arc,
        build_zhang_wuji_lifeform,
        save_lifeform_template,
    )
    from volvence_zero.memory import build_default_memory_store

    memory_store = build_default_memory_store()
    bundle = build_zhang_wuji_lifeform(memory_store=memory_store)
    arc = build_zhang_wuji_demo_arc()
    report = ExperientialReplayDriver().run_arc(arc=arc, lifeform=bundle.lifeform)
    saved_path = save_lifeform_template(
        profile=bundle.profile,
        template_id="zhang-wuji-after-demo-arc",
        output_dir=pathlib.Path("artifacts/lifeform-templates"),
        memory_store=memory_store,
        replay_report=report,
        source_arc_id=arc.arc_id,
        replay_provenance="zhang-wuji-demo-arc-v0 / wave T11",
    )
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any

from lifeform_core import VitalsBootstrap
from volvence_zero.application.storage import (
    ApplicationCaseMemoryStore,
    ApplicationDomainKnowledgeStore,
    CaseMemoryCheckpoint,
    DomainKnowledgeCheckpoint,
)
from volvence_zero.memory import MemoryStore, MemoryStoreCheckpoint
from volvence_zero.owner_hydration import OwnerPersistenceSnapshot

from lifeform_domain_character.profile import CharacterSoulProfile
from lifeform_domain_character.replay import ReplayReport
from lifeform_domain_character.template import (
    SCHEMA_VERSION,
    ApplicationOwnerState,
    LifeformTemplate,
    LifeformTemplateManifest,
    compute_template_integrity_hash,
    utc_iso_now,
)


@dataclass(frozen=True)
class SaveLifeformTemplateResult:
    """Returned by :func:`save_lifeform_template`. Carries both the
    in-memory template and the path it was written to so the caller
    can chain `give_birth` immediately after.
    """

    template: LifeformTemplate
    template_path: pathlib.Path


def save_lifeform_template(
    *,
    profile: CharacterSoulProfile,
    template_id: str,
    output_dir: pathlib.Path,
    memory_store: MemoryStore | None = None,
    domain_knowledge_store: ApplicationDomainKnowledgeStore | None = None,
    case_memory_store: ApplicationCaseMemoryStore | None = None,
    vitals_bootstrap: VitalsBootstrap | None = None,
    vitals_drive_levels: tuple[tuple[str, float], ...] = (),
    replay_report: ReplayReport | None = None,
    evolved_profile: CharacterSoulProfile | None = None,
    source_arc_id: str | None = None,
    replay_provenance: str = "",
    overwrite_existing: bool = False,
    preserve_memory: bool = False,
    owner_hydration_snapshots: tuple[OwnerPersistenceSnapshot, ...] = (),
    hydratable_owners: tuple[tuple[str, Any], ...] = (),
) -> SaveLifeformTemplateResult:
    """Extract lived character lifeform state into a saveable template.

    Args:
        profile: The reviewed :class:`CharacterSoulProfile` the lifeform
            was constructed from. Required (the manifest's
            ``character_id`` is its ``profile_id``).
        template_id: Caller-supplied unique id; used as the file name
            ``<output_dir>/<template_id>.json``.
        output_dir: Directory the template JSON is written to. Created
            if missing.
        memory_store: The :class:`MemoryStore` the lifeform was
            constructed with (or one shared across replay sessions).
            Saved via ``MemoryStore.create_checkpoint``. ``None`` is
            valid for ablation runs but means ``give_birth`` will
            spawn a fresh store.
        domain_knowledge_store: Optional store reference. When passed
            we save its current state via
            ``store.create_checkpoint(...)``.
        case_memory_store: Same shape as
            ``domain_knowledge_store`` but for case memory.
        vitals_bootstrap: The :class:`VitalsBootstrap` used at
            construction. Stored verbatim so ``give_birth`` knows
            the band shapes / thresholds for re-init.
        vitals_drive_levels: Captured drive levels at end of life
            (typically pulled from
            ``session.vitals_snapshot.drive_levels``). ``give_birth``
            uses these to seed initial levels.
        replay_report: Audit trail produced by
            :class:`ExperientialReplayDriver`. Optional; ``None``
            means the lifeform was never replayed.
        evolved_profile: When Phase 4 drive evolution promoted
            updated drive specs, the new profile goes here. Must
            share ``profile_id`` with ``profile``.
        source_arc_id: ``NarrativeArc.arc_id`` if the lifeform lived
            through a specific arc.
        replay_provenance: Free-form audit string from the reviewer
            describing where this template came from.
        overwrite_existing: When True, an existing template with the
            same path is overwritten. Default False fails loudly to
            prevent accidental clobber.
        preserve_memory: NW7 / R14. When True, ``give_birth`` ignores
            the alpha-mode ``skip_memory_restore`` flag and always
            restores ``memory_checkpoint`` into the live session. Use
            this for templates whose canonical first-half-of-life
            memories must persist across every player (e.g.
            ``novel-worlds-character``). Defaults to ``False`` so
            existing personal-companion templates retain their fresh
            per-user scope behaviour.
        owner_hydration_snapshots: Optional owner-published snapshots
            captured by the caller. In the subjective character bake
            path this carries ``semantic_state`` so the nine semantic
            owners survive ``give_birth``.
        hydratable_owners: Optional ``(owner_name, owner)`` pairs.
            Each owner must implement ``export_persistence_snapshot``;
            failures are intentionally loud.

    Returns:
        A :class:`SaveLifeformTemplateResult` carrying the typed
        :class:`LifeformTemplate` and the on-disk path.
    """
    if not profile.profile_id:
        raise ValueError("save_lifeform_template: profile.profile_id is empty")
    if not template_id.strip():
        raise ValueError("save_lifeform_template: template_id is empty")

    output_dir.mkdir(parents=True, exist_ok=True)
    template_path = output_dir / f"{template_id}.json"
    if template_path.exists() and not overwrite_existing:
        raise FileExistsError(
            f"save_lifeform_template: {template_path} already exists; "
            "pass overwrite_existing=True to replace it."
        )

    memory_checkpoint = _capture_memory_checkpoint(memory_store, template_id)
    application_state = _capture_application_owner_state(
        domain_knowledge_store=domain_knowledge_store,
        case_memory_store=case_memory_store,
        template_id=template_id,
    )
    hydration_snapshots = owner_hydration_snapshots + _capture_hydratable_owners(
        hydratable_owners
    )
    created_at = utc_iso_now()
    integrity = compute_template_integrity_hash(
        profile=profile,
        evolved_profile=evolved_profile,
        memory_checkpoint=memory_checkpoint,
        vitals_bootstrap=vitals_bootstrap,
        vitals_drive_levels=vitals_drive_levels,
        application_state=application_state,
        replay_report=replay_report,
        template_id=template_id,
        schema_version=SCHEMA_VERSION,
        character_id=profile.profile_id,
        created_at_utc=created_at,
        source_arc_id=source_arc_id,
        replay_provenance=replay_provenance or "(unspecified)",
        preserve_memory=preserve_memory,
        owner_hydration_snapshots=hydration_snapshots,
    )
    manifest = LifeformTemplateManifest(
        template_id=template_id,
        schema_version=SCHEMA_VERSION,
        character_id=profile.profile_id,
        created_at_utc=created_at,
        source_arc_id=source_arc_id,
        replay_provenance=replay_provenance or "(unspecified)",
        integrity_hash=integrity,
        preserve_memory=preserve_memory,
    )
    template = LifeformTemplate(
        manifest=manifest,
        profile=profile,
        evolved_profile=evolved_profile,
        memory_checkpoint=memory_checkpoint,
        vitals_bootstrap=vitals_bootstrap,
        vitals_drive_levels=vitals_drive_levels,
        application_state=application_state,
        replay_report=replay_report,
        owner_hydration_snapshots=hydration_snapshots,
    )
    template_path.write_bytes(template.to_json_bytes())
    return SaveLifeformTemplateResult(template=template, template_path=template_path)


def _capture_memory_checkpoint(
    store: MemoryStore | None,
    template_id: str,
) -> MemoryStoreCheckpoint | None:
    """Snapshot a ``MemoryStore`` if provided.

    We pull the typed :class:`MemoryStoreCheckpoint` directly so the
    template carries the structured form (not an opaque blob). The
    serializer in ``template.py`` flattens it to JSON via dataclass
    introspection on save.
    """
    if store is None:
        return None
    return store.create_checkpoint(checkpoint_id=f"template-save:{template_id}")


def _capture_application_owner_state(
    *,
    domain_knowledge_store: ApplicationDomainKnowledgeStore | None,
    case_memory_store: ApplicationCaseMemoryStore | None,
    template_id: str,
) -> ApplicationOwnerState:
    """Snapshot the application owner stores. Strategy playbook /
    boundary policy state are derived from the profile at runtime
    (not held as mutable owner state) so we leave their tuples empty
    here; ``give_birth`` reconstructs them from ``profile`` directly.
    """
    domain_checkpoint: DomainKnowledgeCheckpoint | None = None
    case_checkpoint: CaseMemoryCheckpoint | None = None
    if domain_knowledge_store is not None:
        domain_checkpoint = domain_knowledge_store.create_checkpoint(
            checkpoint_id=f"template-save:domain_knowledge:{template_id}"
        )
    if case_memory_store is not None:
        case_checkpoint = case_memory_store.create_checkpoint(
            checkpoint_id=f"template-save:case_memory:{template_id}"
        )
    return ApplicationOwnerState(
        domain_knowledge_checkpoint=domain_checkpoint,
        case_memory_checkpoint=case_checkpoint,
        strategy_playbook_rules=(),
        boundary_policy_hints=(),
    )


def _capture_hydratable_owners(
    owners: tuple[tuple[str, Any], ...],
) -> tuple[OwnerPersistenceSnapshot, ...]:
    snapshots: list[OwnerPersistenceSnapshot] = []
    for owner_name, owner in owners:
        if not owner_name.strip():
            raise ValueError("hydratable owner_name must be non-empty")
        snapshot = owner.export_persistence_snapshot()
        if snapshot.owner_name != owner_name:
            raise ValueError(
                "save_lifeform_template: hydratable owner published "
                f"owner_name={snapshot.owner_name!r}, expected {owner_name!r}"
            )
        snapshots.append(snapshot)
    owner_names = tuple(snapshot.owner_name for snapshot in snapshots)
    if len(set(owner_names)) != len(owner_names):
        raise ValueError(
            "save_lifeform_template: duplicate owner_hydration snapshot "
            f"owner names {owner_names!r}"
        )
    return tuple(snapshots)


def vitals_drive_levels_from_session(session: Any) -> tuple[tuple[str, float], ...]:
    """Pull the canonical ``(name, level)`` tuple from a live session.

    Convenience for callers that don't want to import VitalsSnapshot
    types just to extract a drive levels list. Returns empty tuple
    when vitals are not wired.
    """
    snapshot = getattr(session, "vitals_snapshot", None)
    if snapshot is None:
        return ()
    return tuple(
        (drive.name, float(drive.level)) for drive in snapshot.drive_levels
    )


__all__ = [
    "SaveLifeformTemplateResult",
    "save_lifeform_template",
    "vitals_drive_levels_from_session",
]
