"""Wave T4 contract tests — LifeformTemplate schema + JSON round-trip.

Pin:

* Manifest required-field invariants (template_id / character_id /
  integrity_hash all non-empty; schema_version positive int).
* LifeformTemplate consistency (manifest.character_id ==
  profile.profile_id; evolved_profile shares profile_id when set).
* JSON round-trip preserves every typed field byte-for-byte through
  ``to_json_bytes`` / ``from_json_bytes`` for a representative
  template (built around the shipped 张无忌 profile).
* ``IncompatibleTemplateVersion`` raised on schema_version mismatch.
* Integrity hash stable across re-serialization but changes when any
  field changes.
"""

from __future__ import annotations

import json

import pytest

from lifeform_domain_character import (
    ApplicationOwnerState,
    IncompatibleTemplateVersion,
    LifeformTemplate,
    LifeformTemplateManifest,
    TEMPLATE_SCHEMA_VERSION,
    build_zhang_wuji_profile,
    compute_template_integrity_hash,
    utc_iso_now,
)


def _empty_application_state() -> ApplicationOwnerState:
    return ApplicationOwnerState(
        domain_knowledge_checkpoint=None,
        case_memory_checkpoint=None,
        strategy_playbook_rules=(),
        boundary_policy_hints=(),
    )


def _build_template(*, template_id: str = "t-zhang-wuji-test") -> LifeformTemplate:
    profile = build_zhang_wuji_profile()
    created = utc_iso_now()
    application_state = _empty_application_state()
    integrity = compute_template_integrity_hash(
        profile=profile,
        evolved_profile=None,
        memory_checkpoint=None,
        vitals_bootstrap=None,
        vitals_drive_levels=(),
        application_state=application_state,
        replay_report=None,
        template_id=template_id,
        schema_version=TEMPLATE_SCHEMA_VERSION,
        character_id=profile.profile_id,
        created_at_utc=created,
        source_arc_id=None,
        replay_provenance="unit-test-fixture",
    )
    manifest = LifeformTemplateManifest(
        template_id=template_id,
        schema_version=TEMPLATE_SCHEMA_VERSION,
        character_id=profile.profile_id,
        created_at_utc=created,
        source_arc_id=None,
        replay_provenance="unit-test-fixture",
        integrity_hash=integrity,
    )
    return LifeformTemplate(
        manifest=manifest,
        profile=profile,
        evolved_profile=None,
        memory_checkpoint=None,
        vitals_bootstrap=None,
        vitals_drive_levels=(),
        application_state=application_state,
        replay_report=None,
    )


# ---------------------------------------------------------------------------
# Manifest invariants
# ---------------------------------------------------------------------------


def test_manifest_rejects_empty_template_id() -> None:
    with pytest.raises(ValueError, match="template_id"):
        LifeformTemplateManifest(
            template_id="   ",
            schema_version=TEMPLATE_SCHEMA_VERSION,
            character_id="zhang-wuji",
            created_at_utc=utc_iso_now(),
            source_arc_id=None,
            replay_provenance="p",
            integrity_hash="0" * 64,
        )


def test_manifest_rejects_zero_schema_version() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        LifeformTemplateManifest(
            template_id="t",
            schema_version=0,
            character_id="zhang-wuji",
            created_at_utc=utc_iso_now(),
            source_arc_id=None,
            replay_provenance="p",
            integrity_hash="0" * 64,
        )


# ---------------------------------------------------------------------------
# Template consistency
# ---------------------------------------------------------------------------


def test_template_rejects_character_id_profile_mismatch() -> None:
    profile = build_zhang_wuji_profile()
    integrity = compute_template_integrity_hash(
        profile=profile,
        evolved_profile=None,
        memory_checkpoint=None,
        vitals_bootstrap=None,
        vitals_drive_levels=(),
        application_state=_empty_application_state(),
        replay_report=None,
        template_id="t",
        schema_version=TEMPLATE_SCHEMA_VERSION,
        character_id="WRONG-ID",
        created_at_utc=utc_iso_now(),
        source_arc_id=None,
        replay_provenance="p",
    )
    manifest = LifeformTemplateManifest(
        template_id="t",
        schema_version=TEMPLATE_SCHEMA_VERSION,
        character_id="WRONG-ID",
        created_at_utc=utc_iso_now(),
        source_arc_id=None,
        replay_provenance="p",
        integrity_hash=integrity,
    )
    with pytest.raises(ValueError, match="character_id"):
        LifeformTemplate(
            manifest=manifest,
            profile=profile,
            evolved_profile=None,
            memory_checkpoint=None,
            vitals_bootstrap=None,
            vitals_drive_levels=(),
            application_state=_empty_application_state(),
            replay_report=None,
        )


def test_template_rejects_evolved_profile_id_mismatch() -> None:
    profile = build_zhang_wuji_profile()
    # Build a "different character" profile that breaks the evolved
    # vs base profile_id consistency rule.
    from lifeform_domain_character import CharacterSoulProfile

    other_profile = CharacterSoulProfile(
        profile_id="other-character",
        character_name="Other",
        source_title="Other",
        version="0.0.1",
        reviewed_by="r",
        source_uri="profile:other:test",
        description="other",
        knowledge_seeds=profile.knowledge_seeds,
        signature_cases=profile.signature_cases,
        strategy_priors=profile.strategy_priors,
        boundary_priors=profile.boundary_priors,
        drive_priors=profile.drive_priors,
    )
    integrity = compute_template_integrity_hash(
        profile=profile,
        evolved_profile=other_profile,
        memory_checkpoint=None,
        vitals_bootstrap=None,
        vitals_drive_levels=(),
        application_state=_empty_application_state(),
        replay_report=None,
        template_id="t",
        schema_version=TEMPLATE_SCHEMA_VERSION,
        character_id=profile.profile_id,
        created_at_utc=utc_iso_now(),
        source_arc_id=None,
        replay_provenance="p",
    )
    manifest = LifeformTemplateManifest(
        template_id="t",
        schema_version=TEMPLATE_SCHEMA_VERSION,
        character_id=profile.profile_id,
        created_at_utc=utc_iso_now(),
        source_arc_id=None,
        replay_provenance="p",
        integrity_hash=integrity,
    )
    with pytest.raises(ValueError, match="evolved_profile"):
        LifeformTemplate(
            manifest=manifest,
            profile=profile,
            evolved_profile=other_profile,
            memory_checkpoint=None,
            vitals_bootstrap=None,
            vitals_drive_levels=(),
            application_state=_empty_application_state(),
            replay_report=None,
        )


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


def test_template_json_round_trip_preserves_profile_identity() -> None:
    template = _build_template()
    blob = template.to_json_bytes()
    restored = LifeformTemplate.from_json_bytes(blob)
    assert restored.profile.profile_id == template.profile.profile_id
    assert restored.profile.character_name == template.profile.character_name
    assert restored.manifest.template_id == template.manifest.template_id
    assert restored.manifest.integrity_hash == template.manifest.integrity_hash


def test_template_json_round_trip_preserves_profile_drive_count() -> None:
    template = _build_template()
    restored = LifeformTemplate.from_json_bytes(template.to_json_bytes())
    assert len(restored.profile.drive_priors) == len(template.profile.drive_priors)
    base_names = {d.name for d in template.profile.drive_priors}
    rest_names = {d.name for d in restored.profile.drive_priors}
    assert base_names == rest_names


def test_template_json_round_trip_is_idempotent() -> None:
    """save → load → save should produce byte-identical output. This
    is the property the integrity_hash ultimately leans on.
    """
    template = _build_template()
    blob_a = template.to_json_bytes()
    restored = LifeformTemplate.from_json_bytes(blob_a)
    blob_b = restored.to_json_bytes()
    assert blob_a == blob_b


def test_from_json_bytes_rejects_non_object_top_level() -> None:
    with pytest.raises(ValueError, match="top-level JSON must be a dict"):
        LifeformTemplate.from_json_bytes(b'["not", "an", "object"]')


def test_from_json_bytes_rejects_missing_manifest() -> None:
    with pytest.raises(ValueError, match="missing 'manifest'"):
        LifeformTemplate.from_json_bytes(b'{"profile": {}}')


def test_from_json_bytes_raises_on_schema_version_mismatch() -> None:
    template = _build_template()
    blob = template.to_json_bytes()
    parsed = json.loads(blob.decode("utf-8"))
    parsed["manifest"]["schema_version"] = TEMPLATE_SCHEMA_VERSION + 99
    bad_blob = json.dumps(parsed).encode("utf-8")
    with pytest.raises(IncompatibleTemplateVersion):
        LifeformTemplate.from_json_bytes(bad_blob)


# ---------------------------------------------------------------------------
# Integrity hash
# ---------------------------------------------------------------------------


def test_integrity_hash_is_64_char_hex() -> None:
    template = _build_template()
    assert len(template.manifest.integrity_hash) == 64
    int(template.manifest.integrity_hash, 16)  # raises if non-hex


def test_integrity_hash_is_stable_across_recompute() -> None:
    template = _build_template(template_id="hash-stable-id")
    profile = template.profile
    application_state = template.application_state
    again = compute_template_integrity_hash(
        profile=profile,
        evolved_profile=None,
        memory_checkpoint=None,
        vitals_bootstrap=None,
        vitals_drive_levels=(),
        application_state=application_state,
        replay_report=None,
        template_id=template.manifest.template_id,
        schema_version=TEMPLATE_SCHEMA_VERSION,
        character_id=profile.profile_id,
        created_at_utc=template.manifest.created_at_utc,
        source_arc_id=None,
        replay_provenance=template.manifest.replay_provenance,
    )
    assert again == template.manifest.integrity_hash


def test_integrity_hash_changes_when_template_id_changes() -> None:
    a = _build_template(template_id="version-a")
    b = _build_template(template_id="version-b")
    assert a.manifest.integrity_hash != b.manifest.integrity_hash
