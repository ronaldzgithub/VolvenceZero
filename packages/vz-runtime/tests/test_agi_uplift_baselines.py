from dataclasses import replace

import pytest

from volvence_zero.agent.baseline_manifest import (
    build_default_behavior_baseline_manifest,
)
from volvence_zero.agent.session import AgentSessionRunner
from volvence_zero.agent.paper_suite import (
    ArtifactDigest,
    ClaimVerdict,
    EvidenceBundle,
    PaperSuiteManifest,
    PaperSuiteProvenance,
    RetainProvenanceError,
    RetainProvenanceRequirements,
    manifest_hash,
    validate_evidence_bundle_for_external_use,
    validate_retain_provenance,
)
from volvence_zero.integration.final_wiring import (
    FinalRolloutConfig,
    resolve_final_rollout_config,
)
from volvence_zero.runtime import WiringLevel


def _manifest() -> PaperSuiteManifest:
    return PaperSuiteManifest(
        suite_id="retain-provenance-test",
        suite_kind="contract",
        suite_tier="paper-suite-full",
        version=1,
        baseline_label="baseline",
        repeat_count=3,
        seed_schedule=(0, 1, 2),
        profiles=(),
        primary_metrics=(),
        secondary_metrics=(),
        case_groups=(),
        artifact_expectations=("evidence_bundle.json",),
        description="Contract fixture for retain provenance validation.",
    )


def _provenance(manifest: PaperSuiteManifest) -> PaperSuiteProvenance:
    return PaperSuiteProvenance(
        git_sha="a" * 40,
        git_branch="evidence/test",
        working_tree_dirty=False,
        python_version="3.11",
        platform="test",
        dependency_versions=("vz-runtime==0.1.0",),
        dependency_digest="b" * 64,
        manifest_hash=manifest_hash(manifest),
        runtime_descriptor=(("model_id", "test-model"),),
        description="Complete retain provenance fixture.",
    )


def _requirements() -> RetainProvenanceRequirements:
    return RetainProvenanceRequirements(
        min_seed_count=3,
        require_substrate_fingerprint=True,
        require_artifact_digests=True,
    )


def _artifact() -> ArtifactDigest:
    return ArtifactDigest(
        artifact_path="paper_suite_aggregate.json",
        sha256="c" * 64,
        size_bytes=128,
    )


def test_retain_provenance_accepts_complete_evidence() -> None:
    manifest = _manifest()
    validate_retain_provenance(
        provenance=_provenance(manifest),
        manifest=manifest,
        requirements=_requirements(),
        substrate_fingerprint_verified=True,
        artifact_digests=(_artifact(),),
    )


@pytest.mark.parametrize(
    ("field_name", "field_value", "expected_message"),
    (
        ("git_sha", "unavailable", "git_sha"),
        ("git_branch", "unavailable", "git_branch"),
        ("working_tree_dirty", True, "working tree is dirty"),
        ("dependency_versions", (), "dependency_versions"),
        ("dependency_digest", "", "dependency_digest"),
        ("manifest_hash", "d" * 64, "manifest_hash"),
    ),
)
def test_retain_provenance_rejects_incomplete_provenance(
    field_name: str,
    field_value: object,
    expected_message: str,
) -> None:
    manifest = _manifest()
    provenance = replace(_provenance(manifest), **{field_name: field_value})
    with pytest.raises(RetainProvenanceError, match=expected_message):
        validate_retain_provenance(
            provenance=provenance,
            manifest=manifest,
            requirements=_requirements(),
            substrate_fingerprint_verified=True,
            artifact_digests=(_artifact(),),
        )


def test_external_bundle_validation_only_gates_retain_claims() -> None:
    manifest = _manifest()
    provenance = replace(_provenance(manifest), working_tree_dirty=True)
    weak_claim = ClaimVerdict(
        claim_id="claim",
        status="weak",
        required_gate_ids=(),
        supporting_artifacts=(),
        evidence=(),
        summary="Weak evidence.",
        description="Weak evidence remains exportable for review.",
    )
    weak_bundle = EvidenceBundle(
        bundle_id="weak",
        suite_kind="contract",
        manifest=manifest,
        provenance=provenance,
        run_summaries=(),
        aggregate_metrics=(),
        claim_verdicts=(weak_claim,),
    )
    validate_evidence_bundle_for_external_use(bundle=weak_bundle)
    with pytest.raises(RetainProvenanceError, match="working tree is dirty"):
        validate_evidence_bundle_for_external_use(
            bundle=replace(
                weak_bundle,
                claim_verdicts=(replace(weak_claim, status="retain"),),
            )
        )


def test_default_behavior_baseline_freezes_distinct_product_and_paper_surfaces() -> None:
    manifest = build_default_behavior_baseline_manifest()
    product = dict(manifest.product_brain_defaults)
    dialogue = dict(manifest.dialogue_runner_defaults)
    wiring = dict(manifest.rollout_wiring)

    assert product["substrate_mode"] == "synthetic"
    assert product["memory_scope_root_dir"] == "none"
    assert product["temporal_latent_dim"] == "3"
    assert product["allow_live_substrate_mutation"] == "false"
    assert dialogue["profile_label"] == "pe-eta"
    assert dialogue["allow_live_substrate_mutation"] == "true"
    assert wiring["temporal_ssl_backend"] == "disabled"
    assert wiring["internal_rl_backend"] == "disabled"


def test_temporal_latent_dimension_is_explicit_and_rollback_safe() -> None:
    expanded = AgentSessionRunner(temporal_latent_dim=16)
    rollback = AgentSessionRunner(temporal_latent_dim=3)
    assert expanded.temporal_latent_dim == 16
    assert rollback.temporal_latent_dim == 3


def test_temporal_latent_dimension_rejects_invalid_capacity() -> None:
    with pytest.raises(ValueError, match="temporal_latent_dim"):
        AgentSessionRunner(temporal_latent_dim=2)


_BACKEND_ENV_VARS = (
    "VZ_TORCH_BACKENDS",
    "VZ_TORCH_BACKENDS_FORCE",
    "VZ_SUBSTRATE_DEVICE",
    "VZ_TEMPORAL_SSL_BACKEND",
    "VZ_TEMPORAL_RUNTIME_BACKEND",
    "VZ_INTERNAL_RL_BACKEND",
    "VZ_CMS_TORCH_BACKEND",
)


def _clear_backend_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _BACKEND_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_per_owner_backend_override_supports_shadow(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("VZ_TEMPORAL_RUNTIME_BACKEND", "shadow")
    config = resolve_final_rollout_config(FinalRolloutConfig())
    assert config.temporal_runtime_backend is WiringLevel.SHADOW
    assert config.temporal_ssl_backend is WiringLevel.DISABLED
    assert config.internal_rl_backend is WiringLevel.DISABLED
    assert config.cms_torch_backend is WiringLevel.DISABLED


def test_invalid_per_owner_backend_override_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("VZ_CMS_TORCH_BACKEND", "observe")
    with pytest.raises(ValueError, match="VZ_CMS_TORCH_BACKEND"):
        resolve_final_rollout_config(FinalRolloutConfig())


async def test_ndim_session_runs_with_all_autograd_owners_in_shadow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("VZ_TEMPORAL_SSL_BACKEND", "shadow")
    monkeypatch.setenv("VZ_TEMPORAL_RUNTIME_BACKEND", "shadow")
    monkeypatch.setenv("VZ_INTERNAL_RL_BACKEND", "shadow")
    monkeypatch.setenv("VZ_CMS_TORCH_BACKEND", "shadow")
    config = resolve_final_rollout_config(FinalRolloutConfig())
    runner = AgentSessionRunner(
        config=config,
        temporal_latent_dim=16,
        rare_heavy_enabled=False,
    )

    result = await runner.run_turn("Exercise the ndim SHADOW evidence lane.")

    temporal = result.active_snapshots["temporal_abstraction"].value
    assert temporal.controller_state.code_dim == 16
    assert config.temporal_ssl_backend is WiringLevel.SHADOW
    assert config.temporal_runtime_backend is WiringLevel.SHADOW
    assert config.internal_rl_backend is WiringLevel.SHADOW
    assert config.cms_torch_backend is WiringLevel.SHADOW
