from __future__ import annotations

from dataclasses import replace

import pytest

from lifeform_domain_character import (
    CharacterArtifactRef,
    CharacterFidelityEvidence,
    CharacterLoRARef,
    CharacterPackageGateRecord,
    CharacterPackageManifest,
    rebind_fidelity_only,
    sha256_path,
    verify_manifest_artifacts,
)


def _ref(tmp_path, name: str, content: str, artifact_id: str):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return CharacterArtifactRef(
        locator=name,
        sha256=sha256_path(path),
        artifact_id=artifact_id,
        media_type="application/json",
    )


def _manifest(tmp_path, *, with_lora: bool = False, mode: str = "fidelity-only"):
    template = _ref(tmp_path, "template.json", "{}", "template-integrity")
    prefix = _ref(tmp_path, "prefix.json", "{}", "prefix-package")
    report = _ref(tmp_path, "fidelity.json", "{}", "fidelity-report")
    lora = None
    if with_lora:
        lora_dir = tmp_path / "lora"
        lora_dir.mkdir()
        (lora_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        lora = CharacterLoRARef(
            locator="lora",
            sha256=sha256_path(lora_dir),
            training_plan_hash="plan-v1",
            parameter_count=16,
        )
    evidence = CharacterFidelityEvidence(
        report_ref=report,
        evidence_source="llm_judge",
        verdict="diagnostic-pass",
        passed=True,
        held_out=True,
        source_immutable=True,
        feedback_free=True,
        includes_character_lora=with_lora,
        common_adapter_version="common-v1",
        compatibility_fingerprint="compat-v1",
    )
    shadow = CharacterPackageManifest.create(
        character_id="zhang-wuji",
        character_name="张无忌",
        base_model_id="Qwen/test",
        common_adapter_version="common-v1",
        compatibility_fingerprint="compat-v1",
        template_ref=template,
        prefix_kv_ref=prefix,
        lora_ref=lora,
        revalidation_mode=mode,
        description="test character package",
    )
    gate = CharacterPackageGateRecord(
        proposal_id=f"character-package:{shadow.package_id}:evaluation-v1",
        decision="allow",
        desired_gate="offline",
        fidelity_report_sha256=report.sha256,
        rollback_evidence="restore manifest v0",
        is_reversible=True,
        common_adapter_version="common-v1",
        compatibility_fingerprint="compat-v1",
    )
    return CharacterPackageManifest.create(
        character_id="zhang-wuji",
        character_name="张无忌",
        base_model_id="Qwen/test",
        common_adapter_version="common-v1",
        compatibility_fingerprint="compat-v1",
        template_ref=template,
        prefix_kv_ref=prefix,
        lora_ref=lora,
        fidelity_evidence=evidence,
        gate_record=gate,
        revalidation_mode=mode,
        description="test character package",
    )


def test_manifest_round_trip_active_gate_and_artifact_verification(tmp_path) -> None:
    manifest = _manifest(tmp_path, with_lora=False)
    path = tmp_path / "manifest.json"
    path.write_text(manifest.to_json(), encoding="utf-8")

    restored = CharacterPackageManifest.from_json(path.read_text(encoding="utf-8"))
    restored.require_active()
    resolved = verify_manifest_artifacts(restored, manifest_path=path)

    assert restored == manifest
    assert resolved[0].name == "template.json"
    assert resolved[2] is None


def test_active_gate_keeps_lora_closed_until_multi_arm_ablation(tmp_path) -> None:
    manifest = _manifest(tmp_path, with_lora=True)

    assert manifest.fidelity_evidence.includes_character_lora
    assert not manifest.active_eligible
    with pytest.raises(ValueError, match="remain SHADOW-only"):
        manifest.require_active()


def test_active_gate_must_bind_the_exact_ungated_candidate(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    wrong_gate = replace(
        manifest.gate_record,
        proposal_id=f"character-package:{'0' * 64}:evaluation-v1",
    )
    denied = CharacterPackageManifest.create(
        character_id=manifest.character_id,
        character_name=manifest.character_name,
        base_model_id=manifest.base_model_id,
        common_adapter_version=manifest.common_adapter_version,
        compatibility_fingerprint=manifest.compatibility_fingerprint,
        template_ref=manifest.template_ref,
        prefix_kv_ref=manifest.prefix_kv_ref,
        lora_ref=manifest.lora_ref,
        fidelity_evidence=manifest.fidelity_evidence,
        gate_record=wrong_gate,
        revalidation_mode=manifest.revalidation_mode,
        description=manifest.description,
    )

    with pytest.raises(ValueError, match="not ACTIVE-eligible"):
        denied.require_active()


def test_relocated_manifest_requires_gate_rebound_to_relocated_candidate(
    tmp_path,
) -> None:
    manifest = _manifest(tmp_path)
    relocated_shadow = CharacterPackageManifest.create(
        character_id=manifest.character_id,
        character_name=manifest.character_name,
        base_model_id=manifest.base_model_id,
        common_adapter_version=manifest.common_adapter_version,
        compatibility_fingerprint=manifest.compatibility_fingerprint,
        template_ref=replace(
            manifest.template_ref,
            locator=str((tmp_path / "template.json").resolve()),
        ),
        prefix_kv_ref=replace(
            manifest.prefix_kv_ref,
            locator=str((tmp_path / "prefix.json").resolve()),
        ),
        lora_ref=manifest.lora_ref,
        revalidation_mode=manifest.revalidation_mode,
        description=manifest.description,
    )
    rebound_gate = replace(
        manifest.gate_record,
        proposal_id=(
            f"character-package:{relocated_shadow.package_id}:evaluation-v2"
        ),
    )
    relocated = CharacterPackageManifest.create(
        character_id=relocated_shadow.character_id,
        character_name=relocated_shadow.character_name,
        base_model_id=relocated_shadow.base_model_id,
        common_adapter_version=relocated_shadow.common_adapter_version,
        compatibility_fingerprint=relocated_shadow.compatibility_fingerprint,
        template_ref=relocated_shadow.template_ref,
        prefix_kv_ref=relocated_shadow.prefix_kv_ref,
        lora_ref=relocated_shadow.lora_ref,
        fidelity_evidence=manifest.fidelity_evidence,
        gate_record=rebound_gate,
        revalidation_mode=relocated_shadow.revalidation_mode,
        description=relocated_shadow.description,
    )

    assert relocated_shadow.package_id != manifest.ungated_candidate_id
    relocated.require_active()


def test_artifact_tampering_fails_loudly(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    path = tmp_path / "manifest.json"
    path.write_text(manifest.to_json(), encoding="utf-8")
    (tmp_path / "prefix.json").write_text('{"tampered": true}', encoding="utf-8")

    with pytest.raises(ValueError, match="digest mismatch"):
        verify_manifest_artifacts(manifest, manifest_path=path)


def test_adapter_upgrade_requires_matching_evidence_and_mode(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    report = _ref(tmp_path, "fidelity-v2.json", "{}", "fidelity-v2")
    evidence = replace(
        manifest.fidelity_evidence,
        report_ref=report,
        common_adapter_version="common-v2",
        compatibility_fingerprint="compat-v2",
    )
    gate = replace(
        manifest.gate_record,
        fidelity_report_sha256=report.sha256,
        common_adapter_version="common-v2",
        compatibility_fingerprint="compat-v2",
    )
    upgraded_shadow = CharacterPackageManifest.create(
        character_id=manifest.character_id,
        character_name=manifest.character_name,
        base_model_id="Qwen/test",
        common_adapter_version="common-v2",
        compatibility_fingerprint="compat-v2",
        template_ref=manifest.template_ref,
        prefix_kv_ref=manifest.prefix_kv_ref,
        lora_ref=manifest.lora_ref,
        revalidation_mode=manifest.revalidation_mode,
        description=manifest.description,
    )
    gate = replace(
        gate,
        proposal_id=(
            f"character-package:{upgraded_shadow.package_id}:evaluation-v2"
        ),
    )
    rebound = rebind_fidelity_only(
        manifest,
        base_model_id="Qwen/test",
        common_adapter_version="common-v2",
        compatibility_fingerprint="compat-v2",
        fidelity_evidence=evidence,
        gate_record=gate,
    )

    assert rebound.common_adapter_version == "common-v2"
    assert rebound.package_id != manifest.package_id

    full_rebake = CharacterPackageManifest.create(
        character_id=manifest.character_id,
        character_name=manifest.character_name,
        base_model_id=manifest.base_model_id,
        common_adapter_version=manifest.common_adapter_version,
        compatibility_fingerprint=manifest.compatibility_fingerprint,
        template_ref=manifest.template_ref,
        prefix_kv_ref=manifest.prefix_kv_ref,
        lora_ref=manifest.lora_ref,
        fidelity_evidence=manifest.fidelity_evidence,
        gate_record=manifest.gate_record,
        revalidation_mode="full-rebake",
        description=manifest.description,
    )
    with pytest.raises(ValueError, match="full-rebake"):
        rebind_fidelity_only(
            full_rebake,
            base_model_id="Qwen/test",
            common_adapter_version="common-v2",
            compatibility_fingerprint="compat-v2",
            fidelity_evidence=evidence,
            gate_record=gate,
        )
