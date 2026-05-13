"""Contract tests for SubstrateFingerprint propagation (debt #47).

Validates:

1. ``SubstrateFingerprint`` rejects empty fields
2. ``LEGACY_FINGERPRINT`` is the migration shim and exposes ``is_legacy()``
3. ``fingerprint_set_sha256`` is deterministic and stable
4. Empty ``compatible_substrates`` does NOT change ``FigureArtifactBundle``
   integrity hash (byte-stable for legacy bundles)
5. Non-empty ``compatible_substrates`` DOES change the hash (R15)
6. ``GrowthAdvisorProfile.validated_substrates`` defaults to empty
7. ``RunRecord.sut_substrate_fingerprint`` defaults to None

See:

* ``docs/specs/substrate-upgrade-protocol.md``
* ``docs/moving forward/cross-cutting-foundation-packet.md`` §2.3
* ``docs/known-debts.md`` #47
"""

from __future__ import annotations

import pytest

from volvence_zero.substrate import (
    LEGACY_FINGERPRINT,
    SubstrateFingerprint,
    fingerprint_set_sha256,
)


# ---------------------------------------------------------------------------
# SubstrateFingerprint basic contract
# ---------------------------------------------------------------------------


def test_substrate_fingerprint_rejects_empty_model_id() -> None:
    with pytest.raises(ValueError, match="model_id must be non-empty"):
        SubstrateFingerprint(model_id="", version="v1", weights_sha256="abc")


def test_substrate_fingerprint_rejects_empty_version() -> None:
    with pytest.raises(ValueError, match="version must be non-empty"):
        SubstrateFingerprint(model_id="m", version="", weights_sha256="abc")


def test_substrate_fingerprint_rejects_empty_weights_sha256() -> None:
    with pytest.raises(ValueError, match="weights_sha256 must be non-empty"):
        SubstrateFingerprint(model_id="m", version="v1", weights_sha256="")


def test_legacy_fingerprint_is_migration_shim() -> None:
    assert LEGACY_FINGERPRINT.is_legacy()
    assert LEGACY_FINGERPRINT.weights_sha256 == "legacy"
    assert "legacy" in LEGACY_FINGERPRINT.to_short_id()


def test_short_id_includes_weights_prefix() -> None:
    fp = SubstrateFingerprint(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        version="v2.5",
        weights_sha256="abcdef0123456789" * 4,
    )
    short = fp.to_short_id()
    assert "Qwen/Qwen2.5-1.5B-Instruct" in short
    assert "v2.5" in short
    assert "abcdef01" in short


# ---------------------------------------------------------------------------
# fingerprint_set_sha256 determinism
# ---------------------------------------------------------------------------


def test_empty_fingerprint_set_returns_empty_string() -> None:
    assert fingerprint_set_sha256(()) == ""


def test_fingerprint_set_sha256_is_deterministic() -> None:
    fps = (
        SubstrateFingerprint(model_id="m1", version="v1", weights_sha256="aa"),
        SubstrateFingerprint(model_id="m2", version="v2", weights_sha256="bb"),
    )
    assert fingerprint_set_sha256(fps) == fingerprint_set_sha256(fps)


def test_fingerprint_set_sha256_changes_with_content() -> None:
    a = (SubstrateFingerprint(model_id="m1", version="v1", weights_sha256="aa"),)
    b = (SubstrateFingerprint(model_id="m1", version="v1", weights_sha256="bb"),)
    assert fingerprint_set_sha256(a) != fingerprint_set_sha256(b)


# ---------------------------------------------------------------------------
# FigureArtifactBundle integrity-hash backward compat
# ---------------------------------------------------------------------------


def test_compute_bundle_integrity_hash_empty_substrates_byte_stable() -> None:
    """Bundle hash unchanged when ``compatible_substrates=()``.

    Critical: existing 1063+ figure tests bake bundles with no
    ``compatible_substrates`` field; this default behaviour must not
    change their integrity_hash.
    """
    from lifeform_domain_figure.figure_artifact import compute_bundle_integrity_hash

    args = dict(
        figure_id="einstein",
        profile_version="v1",
        version_window=(1900, 1950),
        retrieval_integrity="rh",
        coverage_integrity="ch",
        style_integrity="sh",
        steering_integrity="",
        lora_integrity="",
    )
    hash_no_substrate = compute_bundle_integrity_hash(**args)
    hash_empty_substrate = compute_bundle_integrity_hash(
        **args, compatible_substrates=()
    )
    assert hash_no_substrate == hash_empty_substrate


def test_compute_bundle_integrity_hash_non_empty_substrates_changes_hash() -> None:
    """Non-empty ``compatible_substrates`` MUST change the hash (R15)."""
    from lifeform_domain_figure.figure_artifact import compute_bundle_integrity_hash

    args = dict(
        figure_id="einstein",
        profile_version="v1",
        version_window=(1900, 1950),
        retrieval_integrity="rh",
        coverage_integrity="ch",
        style_integrity="sh",
        steering_integrity="",
        lora_integrity="",
    )
    hash_empty = compute_bundle_integrity_hash(**args)
    hash_qwen = compute_bundle_integrity_hash(
        **args,
        compatible_substrates=(
            SubstrateFingerprint(
                model_id="Qwen/Qwen2.5-1.5B-Instruct",
                version="v2.5",
                weights_sha256="real_sha",
            ),
        ),
    )
    hash_llama = compute_bundle_integrity_hash(
        **args,
        compatible_substrates=(
            SubstrateFingerprint(
                model_id="meta-llama/Llama-3.1-8B",
                version="v3.1",
                weights_sha256="real_sha2",
            ),
        ),
    )
    assert hash_empty != hash_qwen
    assert hash_qwen != hash_llama


# ---------------------------------------------------------------------------
# GrowthAdvisorProfile + RunRecord backward compat
# ---------------------------------------------------------------------------


def test_growth_advisor_profile_validated_substrates_default_empty() -> None:
    from lifeform_domain_growth_advisor.profile import (
        GrowthAdvisorBoundaryPrior,
        GrowthAdvisorProfile,
    )

    profile = GrowthAdvisorProfile(
        profile_id="cheng-laoshi",
        advisor_name="Cheng Laoshi",
        source_title="reviewed",
        version="v1",
        reviewed_by="reviewer-1",
        source_uri="internal://reviewed",
        description="test",
        knowledge_seeds=(),
        signature_cases=(),
        strategy_priors=(),
        boundary_priors=(
            GrowthAdvisorBoundaryPrior(
                boundary_id="bp-no-hard-sell",
                regime_id=None,
                trigger_reasons=("test",),
                answer_depth_limit_hint="short",
                clarification_required=False,
                refer_out_required=False,
                blocked_topics=(),
                required_disclaimers=(),
                confidence=0.9,
                description="test",
            ),
        ),
    )
    assert profile.validated_substrates == ()


def test_arc_record_sut_substrate_fingerprint_defaults_none() -> None:
    from companion_bench.arc_runner import ArcRecord

    record = ArcRecord(
        arc_id="arc-1",
        scenario_id="s1",
        scenario_hash="abc",
        family="rapport_continuity",
        paraphrase_seed=0,
        submission_id="sub-1",
        sut_model_id="model-x",
        started_at="2026-05-13T12:00:00",
        finished_at="2026-05-13T12:30:00",
        sessions=(),
        user_simulator_model="sim-1",
        summary_extra={},
    )
    assert record.sut_substrate_fingerprint is None
    assert record.simulator_family is None
    payload = record.to_json()
    assert payload["sut_substrate_fingerprint"] is None
    assert payload["simulator_family"] is None
