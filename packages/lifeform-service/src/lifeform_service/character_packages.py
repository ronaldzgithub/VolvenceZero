"""Service-side assembly of immutable character package manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from volvence_zero.runtime import WiringLevel
from volvence_zero.substrate import (
    CharacterPrefixKVPackage,
    CharacterPrefixKVRegistry,
    CharacterPrefixKVRegistryEntry,
    CommonAdapterBundle,
    default_persona_lora_pool,
)


@dataclass(frozen=True)
class CharacterRuntimeAssets:
    common_adapter_bundle: CommonAdapterBundle
    prefix_registry: CharacterPrefixKVRegistry
    manifest_package_ids: tuple[str, ...]


def load_character_runtime_assets(
    *,
    common_adapter_bundle_path: Path,
    manifest_paths: tuple[Path, ...],
    wiring_by_character: Mapping[str, WiringLevel],
) -> CharacterRuntimeAssets:
    """Load, verify, and register L2 packages against one L1 bundle."""

    if not manifest_paths:
        raise ValueError("at least one character manifest path is required.")
    common_bundle = CommonAdapterBundle.from_json(
        common_adapter_bundle_path.read_text(encoding="utf-8")
    )
    common_bundle.require_active()

    from lifeform_domain_character import (
        CharacterPackageManifest,
        verify_manifest_artifacts,
    )

    entries: list[CharacterPrefixKVRegistryEntry] = []
    manifest_ids: list[str] = []
    pool = default_persona_lora_pool()
    seen_characters: set[str] = set()
    for manifest_path in manifest_paths:
        manifest = CharacterPackageManifest.from_json(
            manifest_path.read_text(encoding="utf-8")
        )
        if manifest.character_id in seen_characters:
            raise ValueError(
                f"duplicate character manifest for {manifest.character_id!r}."
            )
        seen_characters.add(manifest.character_id)
        manifest.assert_common_adapter(
            base_model_id=common_bundle.base_model_id,
            common_adapter_version=common_bundle.common_adapter_version,
            compatibility_fingerprint=common_bundle.compatibility_fingerprint,
        )
        _, prefix_path, lora_path, _ = verify_manifest_artifacts(
            manifest,
            manifest_path=manifest_path,
        )
        wiring = wiring_by_character.get(
            manifest.character_id,
            WiringLevel.SHADOW,
        )
        if wiring is WiringLevel.DISABLED:
            manifest_ids.append(manifest.package_id)
            continue
        if wiring is WiringLevel.ACTIVE:
            manifest.require_active()
        if prefix_path is not None:
            prefix = CharacterPrefixKVPackage.from_json(
                prefix_path.read_text(encoding="utf-8")
            )
            _validate_prefix_reference(manifest=manifest, prefix=prefix)
            entries.append(
                CharacterPrefixKVRegistryEntry(
                    manifest_package_id=manifest.package_id,
                    common_adapter_version=manifest.common_adapter_version,
                    compatibility_fingerprint=(
                        manifest.compatibility_fingerprint
                    ),
                    wiring_level=wiring,
                    prefix_package=prefix,
                )
            )
        if wiring is WiringLevel.ACTIVE and manifest.lora_ref is not None:
            if lora_path is None:
                raise RuntimeError(
                    "active character LoRA manifest resolved no checkpoint path."
                )
            pool.register(
                figure_id=manifest.character_id,
                source_bundle_id=manifest.package_id,
                backend_id=manifest.lora_ref.backend_id,
                training_plan_hash=manifest.lora_ref.training_plan_hash,
                adapter_layers=(),
                parameter_count=manifest.lora_ref.parameter_count,
                description=(
                    f"Character LoRA admitted by manifest {manifest.package_id}."
                ),
                peft_checkpoint_dir=str(lora_path),
            )
        manifest_ids.append(manifest.package_id)

    return CharacterRuntimeAssets(
        common_adapter_bundle=common_bundle,
        prefix_registry=CharacterPrefixKVRegistry(
            base_model_id=common_bundle.base_model_id,
            common_adapter_version=common_bundle.common_adapter_version,
            compatibility_fingerprint=common_bundle.compatibility_fingerprint,
            entries=tuple(entries),
        ),
        manifest_package_ids=tuple(manifest_ids),
    )


def _validate_prefix_reference(*, manifest, prefix: CharacterPrefixKVPackage) -> None:
    ref = manifest.prefix_kv_ref
    if ref is None:
        raise ValueError("manifest has no prefix_kv_ref.")
    if prefix.package_id != ref.artifact_id:
        raise ValueError(
            "character prefix package_id does not match manifest artifact_id."
        )
    if prefix.character_id != manifest.character_id:
        raise ValueError(
            "character prefix character_id does not match manifest."
        )
    if prefix.model_id != manifest.base_model_id:
        raise ValueError("character prefix model_id does not match manifest.")
    if prefix.source_template_integrity_hash != manifest.template_ref.artifact_id:
        raise ValueError(
            "character prefix template integrity hash does not match manifest."
        )


__all__ = ["CharacterRuntimeAssets", "load_character_runtime_assets"]
