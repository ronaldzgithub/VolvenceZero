"""Service-side assembly of immutable character package manifests."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

from volvence_zero.runtime import WiringLevel
from volvence_zero.substrate import (
    CharacterPrefixKVPackage,
    CharacterPrefixKVRegistry,
    CharacterPrefixKVRegistryEntry,
    CommonAdapterBundle,
    PersonaLoRAPool,
)


@dataclass(frozen=True)
class CharacterSessionBinding:
    """Immutable service binding resolved from one admitted manifest.

    The service consumes this value as one package-level exchange: the
    template path, runtime Prefix/KV routing key, and optional character LoRA
    key cannot drift independently after a session has been created.
    """

    character_id: str
    manifest_package_id: str
    template_path: Path
    wiring_level: WiringLevel
    prefix_registry_key: str
    character_lora_figure_id: str

    def __post_init__(self) -> None:
        if not self.character_id.strip():
            raise ValueError("character session binding character_id is empty.")
        if not self.manifest_package_id.strip():
            raise ValueError("character session binding manifest_package_id is empty.")
        if not self.template_path.is_file():
            raise FileNotFoundError(self.template_path)
        if self.wiring_level is WiringLevel.DISABLED:
            raise ValueError("disabled character manifests cannot publish session bindings.")
        for name, value in (
            ("prefix_registry_key", self.prefix_registry_key),
            ("character_lora_figure_id", self.character_lora_figure_id),
        ):
            if value and value != self.character_id:
                raise ValueError(f"character session binding {name} must match character_id.")
        if self.wiring_level is WiringLevel.SHADOW and self.character_lora_figure_id:
            raise ValueError("SHADOW character bindings cannot activate a LoRA carrier.")


@dataclass(frozen=True)
class CharacterRuntimeAssets:
    common_adapter_bundle: CommonAdapterBundle
    prefix_registry: CharacterPrefixKVRegistry
    manifest_package_ids: tuple[str, ...]
    session_bindings: tuple[CharacterSessionBinding, ...]
    character_lora_pool: PersonaLoRAPool

    def get_binding(self, character_id: str) -> CharacterSessionBinding | None:
        compact = character_id.strip()
        if not compact:
            return None
        return next(
            (binding for binding in self.session_bindings if binding.character_id == compact),
            None,
        )

    def require_binding(self, character_id: str) -> CharacterSessionBinding:
        binding = self.get_binding(character_id)
        if binding is None:
            raise LookupError(f"no enabled character manifest is loaded for {character_id!r}.")
        return binding

    @property
    def character_ids(self) -> tuple[str, ...]:
        return tuple(binding.character_id for binding in self.session_bindings)


def write_character_runtime_stack_attestation(
    *,
    output_dir: Path,
    common_adapter_bundle: CommonAdapterBundle,
    character_runtime_assets: CharacterRuntimeAssets | None,
    substrate_model_id: str,
    substrate_device: str,
) -> Path:
    """Write one immutable process-start attestation for the admitted L1/L2 stack."""

    common_adapter_bundle.require_active()
    if common_adapter_bundle.base_model_id != substrate_model_id:
        raise ValueError("runtime stack substrate_model_id does not match the common adapter.")
    bindings: list[dict[str, object]] = []
    manifest_package_ids: tuple[str, ...] = ()
    if character_runtime_assets is not None:
        assets_bundle = character_runtime_assets.common_adapter_bundle
        if assets_bundle.bundle_id != common_adapter_bundle.bundle_id:
            raise ValueError("runtime stack character assets use a different common adapter.")
        manifest_package_ids = character_runtime_assets.manifest_package_ids
        for binding in character_runtime_assets.session_bindings:
            entry = character_runtime_assets.prefix_registry.get(binding.prefix_registry_key)
            bindings.append(
                {
                    "character_id": binding.character_id,
                    "manifest_package_id": binding.manifest_package_id,
                    "wiring_level": binding.wiring_level.value,
                    "prefix_package_id": (entry.prefix_package.package_id if entry is not None else None),
                    "character_lora_figure_id": (binding.character_lora_figure_id or None),
                }
            )
    payload: dict[str, object] = {
        "schema_version": "character-runtime-stack-attestation.v1",
        "scope": "process-start",
        "substrate_model_id": substrate_model_id,
        "substrate_device": substrate_device,
        "common_adapter": {
            "bundle_id": common_adapter_bundle.bundle_id,
            "common_adapter_version": (common_adapter_bundle.common_adapter_version),
            "compatibility_fingerprint": (common_adapter_bundle.compatibility_fingerprint),
            "base_model_weights_sha256": (common_adapter_bundle.base_model_weights_sha256),
        },
        "manifest_package_ids": list(manifest_package_ids),
        "session_bindings": bindings,
        "rollback": {
            "l1": "restart-without---common-adapter-bundle",
            "l2": "restart-with-character-package-mode-shadow-or-disabled",
        },
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload["attestation_sha256"] = hashlib.sha256(canonical).hexdigest()
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "character_runtime_stack_attestation.json"
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError(f"character runtime stack attestation drift: {path}")
        return path
    path.write_bytes(encoded)
    return path


def load_character_runtime_assets(
    *,
    common_adapter_bundle_path: Path,
    manifest_paths: tuple[Path, ...],
    wiring_by_character: Mapping[str, WiringLevel],
    default_wiring: WiringLevel = WiringLevel.SHADOW,
) -> CharacterRuntimeAssets:
    """Load, verify, and register L2 packages against one L1 bundle."""

    if not manifest_paths:
        raise ValueError("at least one character manifest path is required.")
    common_bundle = CommonAdapterBundle.from_json(common_adapter_bundle_path.read_text(encoding="utf-8"))
    common_bundle.require_active()

    from lifeform_domain_character import (
        CharacterPackageManifest,
        verify_manifest_artifacts,
    )

    entries: list[CharacterPrefixKVRegistryEntry] = []
    bindings: list[CharacterSessionBinding] = []
    manifest_ids: list[str] = []
    pool = PersonaLoRAPool()
    seen_characters: set[str] = set()
    for manifest_path in manifest_paths:
        manifest = CharacterPackageManifest.from_json(manifest_path.read_text(encoding="utf-8"))
        if manifest.character_id in seen_characters:
            raise ValueError(f"duplicate character manifest for {manifest.character_id!r}.")
        seen_characters.add(manifest.character_id)
        manifest.assert_common_adapter(
            base_model_id=common_bundle.base_model_id,
            common_adapter_version=common_bundle.common_adapter_version,
            compatibility_fingerprint=common_bundle.compatibility_fingerprint,
        )
        template_path, prefix_path, lora_path, _ = verify_manifest_artifacts(
            manifest,
            manifest_path=manifest_path,
        )
        wiring = wiring_by_character.get(
            manifest.character_id,
            default_wiring,
        )
        if wiring is WiringLevel.DISABLED:
            manifest_ids.append(manifest.package_id)
            continue
        if wiring is WiringLevel.ACTIVE:
            manifest.require_active()
            if manifest.lora_ref is not None:
                raise RuntimeError(
                    "ACTIVE character lora_ref requires prefix-only, LoRA-only, "
                    "and prefix+LoRA ablation evidence; the current evidence "
                    "schema does not admit this carrier."
                )
        if prefix_path is not None:
            prefix = CharacterPrefixKVPackage.from_json(prefix_path.read_text(encoding="utf-8"))
            _validate_prefix_reference(manifest=manifest, prefix=prefix)
            entries.append(
                CharacterPrefixKVRegistryEntry(
                    manifest_package_id=manifest.package_id,
                    common_adapter_version=manifest.common_adapter_version,
                    compatibility_fingerprint=(manifest.compatibility_fingerprint),
                    wiring_level=wiring,
                    prefix_package=prefix,
                )
            )
        if wiring is WiringLevel.ACTIVE and manifest.lora_ref is not None:
            if lora_path is None:
                raise RuntimeError("active character LoRA manifest resolved no checkpoint path.")
            pool.register(
                figure_id=manifest.character_id,
                source_bundle_id=manifest.package_id,
                backend_id=manifest.lora_ref.backend_id,
                training_plan_hash=manifest.lora_ref.training_plan_hash,
                adapter_layers=(),
                parameter_count=manifest.lora_ref.parameter_count,
                description=(f"Character LoRA admitted by manifest {manifest.package_id}."),
                peft_checkpoint_dir=str(lora_path),
            )
        bindings.append(
            CharacterSessionBinding(
                character_id=manifest.character_id,
                manifest_package_id=manifest.package_id,
                template_path=template_path,
                wiring_level=wiring,
                prefix_registry_key=(manifest.character_id if prefix_path is not None else ""),
                character_lora_figure_id=(
                    manifest.character_id if wiring is WiringLevel.ACTIVE and manifest.lora_ref is not None else ""
                ),
            )
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
        session_bindings=tuple(bindings),
        character_lora_pool=pool,
    )


def _validate_prefix_reference(*, manifest, prefix: CharacterPrefixKVPackage) -> None:
    ref = manifest.prefix_kv_ref
    if ref is None:
        raise ValueError("manifest has no prefix_kv_ref.")
    if prefix.package_id != ref.artifact_id:
        raise ValueError("character prefix package_id does not match manifest artifact_id.")
    if prefix.character_id != manifest.character_id:
        raise ValueError("character prefix character_id does not match manifest.")
    if prefix.model_id != manifest.base_model_id:
        raise ValueError("character prefix model_id does not match manifest.")
    if prefix.source_template_integrity_hash != manifest.template_ref.artifact_id:
        raise ValueError("character prefix template integrity hash does not match manifest.")


__all__ = [
    "CharacterRuntimeAssets",
    "CharacterSessionBinding",
    "load_character_runtime_assets",
    "write_character_runtime_stack_attestation",
]
