"""LifeformTemplate — saveable / re-birthable character lifeform schema.

The full lived state of a character lifeform compressed into a typed
artifact that can be written to disk and loaded back into a fresh
``Lifeform`` instance later. The shape is deliberately a frozen
dataclass so a future schema migration can be detected via the
``schema_version`` field on :class:`LifeformTemplateManifest`.

Why this lives in lifeform-domain-character (not lifeform-core):

The template carries a typed ``CharacterSoulProfile``, which is a
``lifeform-domain-character`` concept. ``lifeform-core`` cannot
reverse-import that without forming a cycle. Verticals other than
character (companion / coding) can introduce their own
template types if save / give_birth becomes useful for them; the
infrastructure pieces (memory checkpoint, temporal snapshot,
regime bootstrap, vitals state) all live in their respective
kernel wheels and any vertical-side template can compose them.

R8 posture:

This module is **schema + serialization**, not a runtime owner. It
publishes no snapshot, has no ``process()`` method, and never
mutates a kernel store directly. Save / give_birth helpers (Wave T5
/ T6) are the only producers / consumers; both call existing
owner-side export / restore APIs.

Copyright posture:

The template's profile / replay_report / application_state contain
reviewer-paraphrased summaries — not verbatim novel text. A user
who wants to ship templates derived from a copyrighted novel is
responsible for license compliance; this module does not embed any
verbatim source content.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from lifeform_core import VitalsBootstrap
from volvence_zero.application import (
    BoundaryPriorHint,
    CaseMemoryRecord,
    DomainKnowledgeRecord,
    PlaybookRule,
)
from volvence_zero.application.storage import (
    CaseMemoryCheckpoint,
    DomainKnowledgeCheckpoint,
)
from volvence_zero.memory import MemoryStoreCheckpoint

from lifeform_domain_character.profile import CharacterSoulProfile
from lifeform_domain_character.replay import ReplayReport


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LifeformTemplateManifest:
    """Audit metadata for a saved lifeform template.

    Fields:

    * ``template_id`` — caller-supplied id; expected to be unique
      within the destination store.
    * ``schema_version`` — pinned to :data:`SCHEMA_VERSION` at save
      time. ``give_birth`` refuses to load incompatible versions.
    * ``character_id`` — should match
      :attr:`CharacterSoulProfile.profile_id`. Mismatch indicates a
      corrupted or hand-edited template.
    * ``created_at_utc`` — ISO-8601 UTC timestamp.
    * ``source_arc_id`` — the :class:`NarrativeArc` the lifeform
      lived through, if any. ``None`` means the template was saved
      from a base lifeform that never ran a replay.
    * ``replay_provenance`` — short audit string the human reviewer
      writes (e.g. "manually curated 10-scene demo arc").
    * ``integrity_hash`` — SHA-256 over the canonical JSON
      serialization of the LifeformTemplate **with** this field
      replaced by the empty string before hashing. Verified on load.
    """

    template_id: str
    schema_version: int
    character_id: str
    created_at_utc: str
    source_arc_id: str | None
    replay_provenance: str
    integrity_hash: str

    def __post_init__(self) -> None:
        for field_name in (
            "template_id",
            "character_id",
            "created_at_utc",
            "replay_provenance",
            "integrity_hash",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"LifeformTemplateManifest.{field_name} must be non-empty"
                )
        if not isinstance(self.schema_version, int) or self.schema_version < 1:
            raise ValueError(
                "LifeformTemplateManifest.schema_version must be a positive int"
            )


@dataclass(frozen=True)
class ApplicationOwnerState:
    """Bundled snapshots of the four character-vertical application
    owners. ``None`` for any field is acceptable when the owner did
    not produce a checkpoint at save time (e.g. ablation runs).
    """

    domain_knowledge_checkpoint: DomainKnowledgeCheckpoint | None
    case_memory_checkpoint: CaseMemoryCheckpoint | None
    strategy_playbook_rules: tuple[PlaybookRule, ...]
    boundary_policy_hints: tuple[BoundaryPriorHint, ...]


@dataclass(frozen=True)
class LifeformTemplate:
    """One complete saveable / re-birthable character lifeform.

    Compose from a lived ``Lifeform`` via ``save_lifeform_template``
    (Wave T5); load back via ``give_birth`` (Wave T6).
    """

    manifest: LifeformTemplateManifest
    profile: CharacterSoulProfile
    evolved_profile: CharacterSoulProfile | None
    memory_checkpoint: MemoryStoreCheckpoint | None
    vitals_bootstrap: VitalsBootstrap | None
    vitals_drive_levels: tuple[tuple[str, float], ...]
    application_state: ApplicationOwnerState
    replay_report: ReplayReport | None

    def __post_init__(self) -> None:
        if self.manifest.character_id != self.profile.profile_id:
            raise ValueError(
                "LifeformTemplate manifest.character_id="
                f"{self.manifest.character_id!r} does not match "
                f"profile.profile_id={self.profile.profile_id!r}; "
                "this is a corrupted or mis-saved template."
            )
        if (
            self.evolved_profile is not None
            and self.evolved_profile.profile_id != self.profile.profile_id
        ):
            raise ValueError(
                "LifeformTemplate.evolved_profile.profile_id must match "
                "profile.profile_id; the evolved profile is a successor of "
                "the base profile, not a different character."
            )
        for name, value in self.vitals_drive_levels:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("vitals_drive_levels names must be non-empty strings")
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"vitals_drive_levels[{name!r}] value must be a number, "
                    f"got {type(value).__name__}"
                )

    def to_json_bytes(self) -> bytes:
        """Serialize to canonical UTF-8 JSON bytes.

        The ``integrity_hash`` field on the manifest is preserved
        as-is; callers computing a hash should populate it via
        :func:`compute_template_integrity_hash` before constructing
        the template.
        """
        payload = _to_serializable(self)
        return json.dumps(
            payload, ensure_ascii=False, indent=None, sort_keys=True
        ).encode("utf-8")

    @classmethod
    def from_json_bytes(cls, data: bytes) -> "LifeformTemplate":
        parsed = json.loads(data.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError(
                "LifeformTemplate.from_json_bytes: top-level JSON must be a dict"
            )
        manifest_raw = parsed.get("manifest")
        if not isinstance(manifest_raw, dict):
            raise ValueError(
                "LifeformTemplate.from_json_bytes: missing 'manifest' object"
            )
        stored_version = manifest_raw.get("schema_version")
        if stored_version != SCHEMA_VERSION:
            raise IncompatibleTemplateVersion(
                f"Template schema_version={stored_version} is not "
                f"compatible with this code (expected {SCHEMA_VERSION}). "
                "Either upgrade the template via a migration tool or "
                "load with a matching version of lifeform-domain-character."
            )
        return _build_template_from_dict(parsed)


class IncompatibleTemplateVersion(ValueError):
    """Raised when a template's schema_version does not match the
    code's :data:`SCHEMA_VERSION`. The exception type is part of
    the public API so callers can ``except`` it specifically.
    """


def compute_template_integrity_hash(
    *,
    profile: CharacterSoulProfile,
    evolved_profile: CharacterSoulProfile | None,
    memory_checkpoint: MemoryStoreCheckpoint | None,
    vitals_bootstrap: VitalsBootstrap | None,
    vitals_drive_levels: tuple[tuple[str, float], ...],
    application_state: ApplicationOwnerState,
    replay_report: ReplayReport | None,
    template_id: str,
    schema_version: int,
    character_id: str,
    created_at_utc: str,
    source_arc_id: str | None,
    replay_provenance: str,
) -> str:
    """Return the SHA-256 hex digest used as ``integrity_hash``.

    Coverage: deterministic identity fields only — manifest (minus
    ``integrity_hash``), profile, evolved_profile, vitals_bootstrap,
    vitals_drive_levels, and application_state.

    Why ``memory_checkpoint`` and ``replay_report`` are NOT in the hash:

    * ``MemoryStoreCheckpoint`` carries dynamic ids (each save mints
      a new ``checkpoint_id``) so its serialized form is not stable
      across save+load. Its integrity is independently enforced by
      ``serialize_checkpoint`` / ``deserialize_checkpoint``'s own
      ``_schema_version`` check on load.
    * ``ReplayReport`` similarly contains floating-point drive
      levels that may suffer round-trip noise; its integrity is
      enforced by the typed schema.

    The hash therefore proves *the character's identity payload was
    not tampered with*; the runtime state (memory + replay history)
    has its own schema-versioned integrity. Both layers must
    validate before ``give_birth`` accepts a template.

    ``memory_checkpoint`` and ``replay_report`` parameters are kept
    on the signature for forward-compatibility; future schema
    revisions can fold them in if a stable canonicalization is
    introduced.
    """
    del memory_checkpoint, replay_report  # not in the v1 hash, see docstring
    placeholder_manifest = {
        "template_id": template_id,
        "schema_version": schema_version,
        "character_id": character_id,
        "created_at_utc": created_at_utc,
        "source_arc_id": source_arc_id,
        "replay_provenance": replay_provenance,
        "integrity_hash": "",
    }
    payload = {
        "manifest": placeholder_manifest,
        "profile": _to_serializable(profile),
        "evolved_profile": _to_serializable(evolved_profile),
        "vitals_bootstrap": _to_serializable(vitals_bootstrap),
        "vitals_drive_levels": _to_serializable(vitals_drive_levels),
        "application_state": _to_serializable(application_state),
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, indent=None, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def utc_iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _to_serializable(value: Any) -> Any:
    """Recursively convert a typed dataclass tree to JSON-compatible primitives.

    Mirrors ``vz-memory.persistence._to_serializable`` (deliberate
    duplication — each wheel has its own equivalent helper, and
    cross-wheel sharing is not worth the dependency overhead).
    """
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (tuple, list)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_serializable(val) for key, val in value.items()}
    if is_dataclass(value):
        return {
            field.name: _to_serializable(getattr(value, field.name))
            for field in fields(value)
        }
    return str(value)


# ---------------------------------------------------------------------------
# Reconstruction helpers (for from_json_bytes)
# ---------------------------------------------------------------------------


def _build_template_from_dict(payload: dict[str, Any]) -> LifeformTemplate:
    from lifeform_domain_character.profile import (
        CharacterBoundaryPrior,
        CharacterDrivePrior,
        CharacterKnowledgeSeed,
        CharacterSignatureCase,
        CharacterStrategyPrior,
    )

    manifest = LifeformTemplateManifest(**payload["manifest"])
    profile = _build_character_soul_profile(
        payload["profile"],
        ksc_cls=CharacterKnowledgeSeed,
        sc_cls=CharacterSignatureCase,
        sp_cls=CharacterStrategyPrior,
        bp_cls=CharacterBoundaryPrior,
        dp_cls=CharacterDrivePrior,
    )
    evolved_raw = payload.get("evolved_profile")
    evolved_profile = (
        _build_character_soul_profile(
            evolved_raw,
            ksc_cls=CharacterKnowledgeSeed,
            sc_cls=CharacterSignatureCase,
            sp_cls=CharacterStrategyPrior,
            bp_cls=CharacterBoundaryPrior,
            dp_cls=CharacterDrivePrior,
        )
        if evolved_raw is not None
        else None
    )
    memory_checkpoint = _build_memory_checkpoint(payload.get("memory_checkpoint"))
    vitals_bootstrap = _build_vitals_bootstrap(payload.get("vitals_bootstrap"))
    drive_levels_raw = payload.get("vitals_drive_levels") or []
    drive_levels = tuple(
        (str(item[0]), float(item[1])) for item in drive_levels_raw
    )
    application_state = _build_application_owner_state(
        payload.get("application_state") or {}
    )
    replay_report_raw = payload.get("replay_report")
    replay_report: ReplayReport | None = None
    if replay_report_raw is not None:
        replay_report = _build_replay_report(replay_report_raw)
    return LifeformTemplate(
        manifest=manifest,
        profile=profile,
        evolved_profile=evolved_profile,
        memory_checkpoint=memory_checkpoint,
        vitals_bootstrap=vitals_bootstrap,
        vitals_drive_levels=drive_levels,
        application_state=application_state,
        replay_report=replay_report,
    )


def _build_character_soul_profile(
    raw: dict[str, Any],
    *,
    ksc_cls: type,
    sc_cls: type,
    sp_cls: type,
    bp_cls: type,
    dp_cls: type,
) -> CharacterSoulProfile:
    knowledge_seeds = tuple(
        ksc_cls(
            seed_id=str(item["seed_id"]),
            domain=str(item["domain"]),
            title=str(item["title"]),
            summary=str(item["summary"]),
            snippet=str(item["snippet"]),
            evidence_locator=str(item["evidence_locator"]),
            confidence=float(item["confidence"]),
            evidence_strength=str(item.get("evidence_strength", "medium")),
            topic_tags=tuple(str(t) for t in item.get("topic_tags", ())),
        )
        for item in raw.get("knowledge_seeds", [])
    )
    signature_cases = tuple(
        sc_cls(
            case_id=str(item["case_id"]),
            domain=str(item["domain"]),
            problem_pattern=str(item["problem_pattern"]),
            user_state_pattern=str(item["user_state_pattern"]),
            risk_markers=tuple(str(t) for t in item.get("risk_markers", ())),
            track_tags=tuple(str(t) for t in item.get("track_tags", ())),
            regime_tags=tuple(str(t) for t in item.get("regime_tags", ())),
            intervention_ordering=tuple(
                str(t) for t in item.get("intervention_ordering", ())
            ),
            outcome_label=str(item["outcome_label"]),
            description=str(item["description"]),
            confidence=float(item["confidence"]),
            relevance_score=float(item.get("relevance_score", 0.75)),
            escalation_observed=bool(item.get("escalation_observed", False)),
            repair_observed=bool(item.get("repair_observed", False)),
        )
        for item in raw.get("signature_cases", [])
    )
    strategy_priors = tuple(
        sp_cls(
            rule_id=str(item["rule_id"]),
            problem_pattern=str(item["problem_pattern"]),
            recommended_regime=item.get("recommended_regime"),
            recommended_ordering=tuple(
                str(t) for t in item.get("recommended_ordering", ())
            ),
            recommended_pacing=str(item["recommended_pacing"]),
            avoid_patterns=tuple(str(t) for t in item.get("avoid_patterns", ())),
            applicability_scope=tuple(
                str(t) for t in item.get("applicability_scope", ())
            ),
            confidence=float(item["confidence"]),
            description=str(item["description"]),
            knowledge_weight_hint=float(
                item.get("knowledge_weight_hint", 0.45)
            ),
            experience_weight_hint=float(
                item.get("experience_weight_hint", 0.65)
            ),
        )
        for item in raw.get("strategy_priors", [])
    )
    boundary_priors = tuple(
        bp_cls(
            boundary_id=str(item["boundary_id"]),
            regime_id=item.get("regime_id"),
            trigger_reasons=tuple(
                str(t) for t in item.get("trigger_reasons", ())
            ),
            answer_depth_limit_hint=str(item["answer_depth_limit_hint"]),
            clarification_required=bool(item["clarification_required"]),
            refer_out_required=bool(item["refer_out_required"]),
            blocked_topics=tuple(str(t) for t in item.get("blocked_topics", ())),
            required_disclaimers=tuple(
                str(t) for t in item.get("required_disclaimers", ())
            ),
            confidence=float(item["confidence"]),
            description=str(item["description"]),
        )
        for item in raw.get("boundary_priors", [])
    )
    drive_priors = tuple(
        dp_cls(
            name=str(item["name"]),
            target=float(item["target"]),
            homeostatic_band=tuple(float(b) for b in item["homeostatic_band"]),
            decay_per_tick=float(item["decay_per_tick"]),
            pe_weight=float(item["pe_weight"]),
            initial_level=float(item.get("initial_level", 0.5)),
            recharge_per_turn=float(item.get("recharge_per_turn", 0.0)),
            recharge_per_regime=tuple(
                (str(name), float(value))
                for name, value in item.get("recharge_per_regime", ())
            ),
        )
        for item in raw.get("drive_priors", [])
    )
    return CharacterSoulProfile(
        profile_id=str(raw["profile_id"]),
        character_name=str(raw["character_name"]),
        source_title=str(raw["source_title"]),
        version=str(raw["version"]),
        reviewed_by=str(raw["reviewed_by"]),
        source_uri=str(raw["source_uri"]),
        description=str(raw["description"]),
        knowledge_seeds=knowledge_seeds,
        signature_cases=signature_cases,
        strategy_priors=strategy_priors,
        boundary_priors=boundary_priors,
        drive_priors=drive_priors,
        target_contexts=tuple(
            str(t)
            for t in raw.get(
                "target_contexts",
                ("character-companion", "fictional-roleplay"),
            )
        ),
    )


def _build_memory_checkpoint(raw: Any) -> MemoryStoreCheckpoint | None:
    """Return a fresh MemoryStoreCheckpoint from a serialized dict.

    Memory checkpoint is large and complex; we delegate reconstruction
    by piggybacking on the existing
    ``MemoryStore.load_from_backend`` path. To do that we'd need a
    backend-stub; for v0 we accept an opaque dict and lean on
    ``serialize_checkpoint`` / ``deserialize_checkpoint`` (called
    indirectly during save / load). Returns ``None`` when raw is
    None or empty.
    """
    if raw is None or raw == {} or raw == []:
        return None
    # In the LifeformTemplate's JSON form the memory_checkpoint is the
    # already-serialized dict from ``serialize_checkpoint``. For the
    # v0 schema we wrap it in a thin sentinel so the save path can
    # accept either a typed checkpoint or an already-serialized dict.
    # Reconstruction happens via the caller's MemoryStore on
    # give_birth (Wave T6).
    if isinstance(raw, dict) and "_serialized_payload" in raw:
        return _SerializedMemoryCheckpoint(payload=raw["_serialized_payload"])
    # Fallback: treat as already-serialized payload.
    return _SerializedMemoryCheckpoint(payload=raw)


@dataclass(frozen=True)
class _SerializedMemoryCheckpoint:
    """Opaque carrier for serialized MemoryStoreCheckpoint payloads.

    The save path captures a checkpoint via the memory persistence
    helper; the load path hands the payload back to the new
    :class:`MemoryStore` for reconstruction. Treating this as opaque
    avoids re-implementing the memory checkpoint reconstruction here.
    """

    payload: Any

    def is_serialized(self) -> bool:  # noqa: D401 - simple flag
        return True


def _build_vitals_bootstrap(raw: Any) -> VitalsBootstrap | None:
    if raw is None:
        return None
    from lifeform_core import DriveSpec

    drives_raw = raw.get("drives", [])
    drives = tuple(
        DriveSpec(
            name=str(item["name"]),
            target=float(item["target"]),
            homeostatic_band=tuple(float(b) for b in item["homeostatic_band"]),
            decay_per_tick=float(item["decay_per_tick"]),
            pe_weight=float(item["pe_weight"]),
            initial_level=float(item.get("initial_level", 0.5)),
            recharge_per_turn=float(item.get("recharge_per_turn", 0.0)),
            recharge_per_regime=dict(
                (str(name), float(value))
                for name, value in item.get("recharge_per_regime", {}).items()
            ),
        )
        for item in drives_raw
    )
    return VitalsBootstrap(
        schema_version=int(raw.get("schema_version", 1)),
        drives=drives,
        proactive_pe_threshold=float(raw.get("proactive_pe_threshold", 1.0)),
        proactive_followup_priority=float(
            raw.get("proactive_followup_priority", 0.55)
        ),
        proactive_cooldown_ticks=int(raw.get("proactive_cooldown_ticks", 60)),
    )


def _build_application_owner_state(raw: dict[str, Any]) -> ApplicationOwnerState:
    from volvence_zero.application.storage import (
        _reconstruct_case_memory_checkpoint as _rcmc,
        _reconstruct_domain_knowledge_checkpoint as _rdkc,
    )

    domain_raw = raw.get("domain_knowledge_checkpoint")
    case_raw = raw.get("case_memory_checkpoint")
    domain_checkpoint = _rdkc(domain_raw) if isinstance(domain_raw, dict) else None
    case_checkpoint = _rcmc(case_raw) if isinstance(case_raw, dict) else None
    playbook_rules = tuple(
        PlaybookRule(
            rule_id=str(item["rule_id"]),
            problem_pattern=str(item["problem_pattern"]),
            recommended_regime=item.get("recommended_regime"),
            recommended_ordering=tuple(
                str(t) for t in item.get("recommended_ordering", ())
            ),
            recommended_pacing=str(item["recommended_pacing"]),
            avoid_patterns=tuple(str(t) for t in item.get("avoid_patterns", ())),
            knowledge_weight_hint=float(item["knowledge_weight_hint"]),
            experience_weight_hint=float(item["experience_weight_hint"]),
            applicability_scope=tuple(
                str(t) for t in item.get("applicability_scope", ())
            ),
            confidence=float(item["confidence"]),
            description=str(item["description"]),
            continuum_band_id=item.get("continuum_band_id"),
            mean_continuum_position=float(
                item.get("mean_continuum_position", 0.0)
            ),
        )
        for item in raw.get("strategy_playbook_rules", [])
    )
    boundary_hints = tuple(
        BoundaryPriorHint(
            hint_id=str(item["hint_id"]),
            regime_id=item.get("regime_id"),
            trigger_reasons=tuple(
                str(t) for t in item.get("trigger_reasons", ())
            ),
            answer_depth_limit_hint=str(item["answer_depth_limit_hint"]),
            clarification_required=bool(item["clarification_required"]),
            refer_out_required=bool(item["refer_out_required"]),
            blocked_topics=tuple(str(t) for t in item.get("blocked_topics", ())),
            required_disclaimers=tuple(
                str(t) for t in item.get("required_disclaimers", ())
            ),
            confidence=float(item["confidence"]),
            description=str(item["description"]),
        )
        for item in raw.get("boundary_policy_hints", [])
    )
    return ApplicationOwnerState(
        domain_knowledge_checkpoint=domain_checkpoint,
        case_memory_checkpoint=case_checkpoint,
        strategy_playbook_rules=playbook_rules,
        boundary_policy_hints=boundary_hints,
    )


def _build_replay_report(raw: dict[str, Any]) -> ReplayReport:
    from lifeform_domain_character.replay import (
        ReplayReport as _ReplayReport,
        SceneReplayRecord,
    )

    per_scene = tuple(
        SceneReplayRecord(
            scene_id=str(item["scene_id"]),
            phase_label=str(item["phase_label"]),
            predicted_action_snippet=str(item.get("predicted_action_snippet", "")),
            canonical_action=str(item["canonical_action"]),
            outcome_kind=str(item["outcome_kind"]),
            pe_magnitude=float(item.get("pe_magnitude", 0.0)),
            active_regime=item.get("active_regime"),
            drive_level_after=tuple(
                (str(name), float(value))
                for name, value in item.get("drive_level_after", ())
            ),
        )
        for item in raw.get("per_scene", [])
    )
    return _ReplayReport(
        arc_id=str(raw["arc_id"]),
        character_id=str(raw["character_id"]),
        scenes_processed=int(raw.get("scenes_processed", len(per_scene))),
        per_scene=per_scene,
        total_pe_signal=float(raw.get("total_pe_signal", 0.0)),
        drive_drift=tuple(
            (str(name), float(delta))
            for name, delta in raw.get("drive_drift", ())
        ),
        regime_sequence_payoff_growth=int(
            raw.get("regime_sequence_payoff_growth", 0)
        ),
        final_vitals=tuple(
            (str(name), float(value))
            for name, value in raw.get("final_vitals", ())
        ),
        notes=tuple(str(n) for n in raw.get("notes", [])),
    )


__all__ = [
    "ApplicationOwnerState",
    "IncompatibleTemplateVersion",
    "LifeformTemplate",
    "LifeformTemplateManifest",
    "SCHEMA_VERSION",
    "compute_template_integrity_hash",
    "utc_iso_now",
]
