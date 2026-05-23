"""Smoke tests for the L0 visual presence artifact + bundle wiring."""

from __future__ import annotations

import dataclasses

import pytest

from lifeform_domain_figure import (
    FigurePresenceArtifact,
    attach_presence_to_bundle,
    build_figure_presence_artifact,
    hash_consent_token,
    presence_fingerprint,
)
from lifeform_domain_figure.compiler import build_figure_artifact_bundle, FigureBundleInputs
from lifeform_domain_figure.figure_artifact import (
    FigureArtifactBundle,
    compute_bundle_integrity_hash,
)
from lifeform_domain_figure.profiles import build_einstein_profile
from lifeform_domain_figure.sample_corpus import synthetic_einstein_corpus


def _einstein_bundle() -> FigureArtifactBundle:
    profile = build_einstein_profile()
    inputs = FigureBundleInputs(
        profile=profile,
        figure_sources=synthetic_einstein_corpus(),
    )
    return build_figure_artifact_bundle(inputs)


def test_build_presence_artifact_round_trips_hashes() -> None:
    artifact = build_figure_presence_artifact(
        figure_id="einstein",
        persona_id_at_presence_service="persona_einstein_local",
        reference_image_uri="https://example.com/einstein/portrait.jpg",
        voice_clone_id="voice_einstein_v1",
        motion_model_id="hallo3-2026-04",
        supported_engines=("server-photoreal", "client-3d"),
        likeness_consent_token="estate-grant-2026-04-12-abcdef",
        license_label="estate-grant-2026",
    )
    assert isinstance(artifact, FigurePresenceArtifact)
    assert artifact.schema_version == 1
    # consent token is hashed, not stored
    assert artifact.likeness_consent_token_hash == hash_consent_token(
        "estate-grant-2026-04-12-abcdef"
    )
    assert artifact.figure_id == "einstein"
    assert artifact.integrity_hash, "integrity_hash must be non-empty"
    assert artifact.presence_artifact_id.startswith("figure-presence:einstein:")


def test_presence_artifact_rejects_short_consent_token() -> None:
    with pytest.raises(ValueError):
        build_figure_presence_artifact(
            figure_id="einstein",
            persona_id_at_presence_service="x",
            reference_image_uri="https://example.com/x.jpg",
            voice_clone_id=None,
            motion_model_id=None,
            supported_engines=("client-3d",),
            likeness_consent_token="short",
            license_label="dev",
        )


def test_presence_artifact_rejects_unknown_engine() -> None:
    with pytest.raises(ValueError):
        build_figure_presence_artifact(
            figure_id="einstein",
            persona_id_at_presence_service="x",
            reference_image_uri="https://example.com/x.jpg",
            voice_clone_id=None,
            motion_model_id=None,
            supported_engines=("not-an-engine",),  # type: ignore[arg-type]
            likeness_consent_token="abcdefghij",
            license_label="dev",
        )


def test_presence_fingerprint_empty_when_none() -> None:
    assert presence_fingerprint(None) == ""


def test_legacy_bundle_remains_byte_stable_without_presence() -> None:
    bundle = _einstein_bundle()
    # Without presence attached, integrity_hash equals what the legacy
    # compute_bundle_integrity_hash without presence_integrity would
    # have produced. Empty presence_integrity is the default and must
    # not change the hash.
    legacy = compute_bundle_integrity_hash(
        figure_id=bundle.figure_id,
        profile_version=bundle.profile_version,
        version_window=bundle.version_window,
        retrieval_integrity=bundle.retrieval_index.integrity_hash,
        coverage_integrity=bundle.coverage_map.integrity_hash,
        style_integrity=bundle.style_prior.integrity_hash,
        steering_integrity="absent",
        lora_integrity="absent",
        metadata_digest_fingerprint=bundle.metadata_digest_fingerprint,
        provenance_fingerprint=bundle.provenance_fingerprint,
    )
    assert bundle.integrity_hash == legacy
    assert bundle.presence is None


def test_attaching_presence_changes_bundle_id() -> None:
    bundle = _einstein_bundle()
    artifact = build_figure_presence_artifact(
        figure_id=bundle.figure_id,
        persona_id_at_presence_service="persona_einstein_local",
        reference_image_uri="https://example.com/einstein/portrait.jpg",
        voice_clone_id="voice_einstein_v1",
        motion_model_id="hallo3-2026-04",
        supported_engines=("server-photoreal", "client-3d"),
        likeness_consent_token="estate-grant-2026-04-12-abcdef",
        license_label="estate-grant-2026",
    )
    attached = attach_presence_to_bundle(bundle, presence=artifact)
    assert attached.presence is artifact
    assert attached.bundle_id != bundle.bundle_id
    assert attached.integrity_hash != bundle.integrity_hash


def test_re_baking_consent_yields_new_bundle_id() -> None:
    bundle = _einstein_bundle()
    a = build_figure_presence_artifact(
        figure_id=bundle.figure_id,
        persona_id_at_presence_service="persona_einstein_local",
        reference_image_uri="https://example.com/einstein/portrait.jpg",
        voice_clone_id="voice_einstein_v1",
        motion_model_id="hallo3-2026-04",
        supported_engines=("server-photoreal",),
        likeness_consent_token="estate-grant-2026-04-12-aaaa",
        license_label="estate-grant-2026",
    )
    b = build_figure_presence_artifact(
        figure_id=bundle.figure_id,
        persona_id_at_presence_service="persona_einstein_local",
        reference_image_uri="https://example.com/einstein/portrait.jpg",
        voice_clone_id="voice_einstein_v1",
        motion_model_id="hallo3-2026-04",
        supported_engines=("server-photoreal",),
        # different consent token = different hash = different bundle id
        likeness_consent_token="estate-grant-2026-04-12-bbbb",
        license_label="estate-grant-2026",
    )
    ba = attach_presence_to_bundle(bundle, presence=a)
    bb = attach_presence_to_bundle(bundle, presence=b)
    assert ba.bundle_id != bb.bundle_id
    assert ba.integrity_hash != bb.integrity_hash


def test_attach_presence_rejects_figure_id_mismatch() -> None:
    bundle = _einstein_bundle()
    artifact = build_figure_presence_artifact(
        figure_id="some-other-figure",
        persona_id_at_presence_service="x",
        reference_image_uri="https://example.com/x.jpg",
        voice_clone_id=None,
        motion_model_id=None,
        supported_engines=("client-3d",),
        likeness_consent_token="abcdefghij",
        license_label="dev",
    )
    with pytest.raises(ValueError):
        attach_presence_to_bundle(bundle, presence=artifact)


def test_presence_artifact_is_frozen() -> None:
    artifact = build_figure_presence_artifact(
        figure_id="einstein",
        persona_id_at_presence_service="x",
        reference_image_uri="https://example.com/x.jpg",
        voice_clone_id=None,
        motion_model_id=None,
        supported_engines=("client-3d",),
        likeness_consent_token="abcdefghij",
        license_label="dev",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        artifact.license_label = "tampered"  # type: ignore[misc]
