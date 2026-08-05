from __future__ import annotations

from types import SimpleNamespace
import json

import pytest

import lifeform_domain_character
from lifeform_service import cli
from lifeform_service.character_packages import (
    load_character_runtime_assets,
    write_character_runtime_stack_attestation,
)
from volvence_zero.runtime import WiringLevel
from volvence_zero.substrate import (
    CharacterPrefixKVPackage,
    CommonAdapterBundle,
    default_persona_lora_pool,
)


class _CommonBundle:
    bundle_id = "bundle-v1"
    base_model_id = "Qwen/test"
    base_model_weights_sha256 = "1" * 64
    common_adapter_version = "common-v1"
    compatibility_fingerprint = "compat-v1"

    def require_active(self) -> None:
        return None


class _Manifest:
    def __init__(
        self,
        character_id: str,
        *,
        active_eligible: bool = True,
        with_lora: bool = False,
    ) -> None:
        self.character_id = character_id
        self.character_name = character_id
        self.package_id = f"manifest:{character_id}"
        self.base_model_id = _CommonBundle.base_model_id
        self.common_adapter_version = _CommonBundle.common_adapter_version
        self.compatibility_fingerprint = _CommonBundle.compatibility_fingerprint
        self.template_ref = SimpleNamespace(artifact_id=f"template:{character_id}")
        self.prefix_kv_ref = SimpleNamespace(artifact_id=f"prefix:{character_id}")
        self.lora_ref = (
            SimpleNamespace(
                backend_id="peft-character-lora-v1",
                training_plan_hash=f"plan:{character_id}",
                parameter_count=16,
            )
            if with_lora
            else None
        )
        self.active_eligible = active_eligible

    def assert_common_adapter(self, **actual) -> None:
        assert actual == {
            "base_model_id": self.base_model_id,
            "common_adapter_version": self.common_adapter_version,
            "compatibility_fingerprint": self.compatibility_fingerprint,
        }

    def require_active(self) -> None:
        if not self.active_eligible:
            raise ValueError("character package is not ACTIVE-eligible")


def _install_loader_fakes(monkeypatch, tmp_path, manifests) -> None:
    by_character = {manifest.character_id: manifest for manifest in manifests}

    monkeypatch.setattr(
        CommonAdapterBundle,
        "from_json",
        classmethod(lambda _cls, _payload: _CommonBundle()),
    )
    monkeypatch.setattr(
        lifeform_domain_character.CharacterPackageManifest,
        "from_json",
        classmethod(lambda _cls, payload: by_character[payload.strip()]),
    )

    def verify(manifest, *, manifest_path):
        template = tmp_path / f"{manifest.character_id}.template.json"
        prefix = tmp_path / f"{manifest.character_id}.prefix.json"
        template.write_text("{}", encoding="utf-8")
        prefix.write_text(manifest.character_id, encoding="utf-8")
        lora = None
        if manifest.lora_ref is not None:
            lora = tmp_path / f"{manifest.character_id}.lora"
            lora.mkdir(exist_ok=True)
            (lora / "adapter_config.json").write_text("{}", encoding="utf-8")
        return template, prefix, lora, None

    monkeypatch.setattr(lifeform_domain_character, "verify_manifest_artifacts", verify)

    def prefix_from_json(_cls, payload):
        character_id = payload.strip()
        return SimpleNamespace(
            package_id=f"prefix:{character_id}",
            character_id=character_id,
            character_name=character_id,
            model_id=_CommonBundle.base_model_id,
            source_template_integrity_hash=f"template:{character_id}",
        )

    monkeypatch.setattr(
        CharacterPrefixKVPackage,
        "from_json",
        classmethod(prefix_from_json),
    )


def _write_inputs(tmp_path, manifests):
    common = tmp_path / "common.json"
    common.write_text("{}", encoding="utf-8")
    paths = []
    for manifest in manifests:
        path = tmp_path / f"{manifest.character_id}.manifest.json"
        path.write_text(manifest.character_id, encoding="utf-8")
        paths.append(path)
    return common, tuple(paths)


def test_loader_publishes_two_bindings_and_keeps_shadow_lora_unregistered(
    tmp_path,
    monkeypatch,
) -> None:
    first = _Manifest("character-one")
    second = _Manifest("character-two", with_lora=True)
    manifests = (first, second)
    _install_loader_fakes(monkeypatch, tmp_path, manifests)
    common, paths = _write_inputs(tmp_path, manifests)

    assets = load_character_runtime_assets(
        common_adapter_bundle_path=common,
        manifest_paths=paths,
        wiring_by_character={"character-one": WiringLevel.ACTIVE},
        default_wiring=WiringLevel.SHADOW,
    )

    assert assets.character_ids == ("character-one", "character-two")
    assert assets.require_binding("character-one").wiring_level is WiringLevel.ACTIVE
    assert assets.require_binding("character-two").wiring_level is WiringLevel.SHADOW
    assert assets.prefix_registry.require("character-one").wiring_level is WiringLevel.ACTIVE
    assert assets.prefix_registry.require("character-two").wiring_level is WiringLevel.SHADOW
    assert not assets.character_lora_pool.has("character-one")
    assert not assets.character_lora_pool.has("character-two")
    assert not default_persona_lora_pool().has("character-one")


def test_loader_active_wiring_requires_manifest_promotion_gate(
    tmp_path,
    monkeypatch,
) -> None:
    denied = _Manifest("denied-character", active_eligible=False)
    _install_loader_fakes(monkeypatch, tmp_path, (denied,))
    common, paths = _write_inputs(tmp_path, (denied,))

    with pytest.raises(ValueError, match="not ACTIVE-eligible"):
        load_character_runtime_assets(
            common_adapter_bundle_path=common,
            manifest_paths=paths,
            wiring_by_character={"denied-character": WiringLevel.ACTIVE},
        )


def test_loader_keeps_character_lora_shadow_only_until_ablation_contract(
    tmp_path,
    monkeypatch,
) -> None:
    lora_manifest = _Manifest("lora-character", with_lora=True)
    _install_loader_fakes(monkeypatch, tmp_path, (lora_manifest,))
    common, paths = _write_inputs(tmp_path, (lora_manifest,))

    with pytest.raises(RuntimeError, match="ablation evidence"):
        load_character_runtime_assets(
            common_adapter_bundle_path=common,
            manifest_paths=paths,
            wiring_by_character={"lora-character": WiringLevel.ACTIVE},
        )


def test_loader_omits_disabled_manifest_from_session_selection(
    tmp_path,
    monkeypatch,
) -> None:
    disabled = _Manifest("disabled-character")
    _install_loader_fakes(monkeypatch, tmp_path, (disabled,))
    common, paths = _write_inputs(tmp_path, (disabled,))

    assets = load_character_runtime_assets(
        common_adapter_bundle_path=common,
        manifest_paths=paths,
        wiring_by_character={"disabled-character": WiringLevel.DISABLED},
    )

    assert assets.character_ids == ()
    assert assets.manifest_package_ids == ("manifest:disabled-character",)
    with pytest.raises(LookupError, match="no enabled character manifest"):
        assets.require_binding("disabled-character")


def test_cli_shared_runtime_receives_admitted_common_and_character_registry(
    tmp_path,
    monkeypatch,
) -> None:
    manifest = _Manifest("character-one")
    _install_loader_fakes(monkeypatch, tmp_path, (manifest,))
    common, paths = _write_inputs(tmp_path, (manifest,))
    assets = load_character_runtime_assets(
        common_adapter_bundle_path=common,
        manifest_paths=paths,
        wiring_by_character={"character-one": WiringLevel.ACTIVE},
    )
    captured = {}

    def build_runtime(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            model_id="Qwen/test",
            runtime_origin="hf-local",
        )

    monkeypatch.setattr(
        "volvence_zero.substrate.build_transformers_runtime_with_fallback",
        build_runtime,
    )
    args = SimpleNamespace(
        substrate_mode="hf-shared",
        substrate_model_id="Qwen/test",
        substrate_model_source=None,
        substrate_device="cpu",
        substrate_local_files_only=True,
        substrate_layer_indices=None,
        substrate_activation_width=8,
        substrate_max_length=None,
        substrate_expected_weights_sha256="",
        steering_activation_step=None,
        companion_evidence_profile=None,
    )

    runtime = cli._build_shared_substrate(
        args,
        common_adapter_bundle=assets.common_adapter_bundle,
        character_runtime_assets=assets,
    )

    assert runtime.model_id == "Qwen/test"
    assert captured["common_adapter_bundle"] is assets.common_adapter_bundle
    assert captured["character_prefix_registry"] is assets.prefix_registry


def test_cli_loads_admitted_stack_and_applies_active_default(
    tmp_path,
    monkeypatch,
) -> None:
    manifest = _Manifest("character-one")
    _install_loader_fakes(monkeypatch, tmp_path, (manifest,))
    common, paths = _write_inputs(tmp_path, (manifest,))
    args = SimpleNamespace(
        common_adapter_bundle=common,
        character_package_manifest=list(paths),
        character_package_wiring=[],
        character_package_mode="active",
        substrate_mode="hf-shared",
        substrate_model_id="Qwen/test",
    )

    bundle, assets = cli._load_admitted_character_stack(args)

    assert bundle is not None
    assert assets is not None
    assert bundle.bundle_id == "bundle-v1"
    assert assets.require_binding("character-one").wiring_level is WiringLevel.ACTIVE


def test_runtime_stack_attestation_is_immutable_and_lists_active_binding(
    tmp_path,
    monkeypatch,
) -> None:
    manifest = _Manifest("character-one")
    _install_loader_fakes(monkeypatch, tmp_path, (manifest,))
    common, paths = _write_inputs(tmp_path, (manifest,))
    assets = load_character_runtime_assets(
        common_adapter_bundle_path=common,
        manifest_paths=paths,
        wiring_by_character={"character-one": WiringLevel.ACTIVE},
    )

    first = write_character_runtime_stack_attestation(
        output_dir=tmp_path / "evidence",
        common_adapter_bundle=assets.common_adapter_bundle,
        character_runtime_assets=assets,
        substrate_model_id="Qwen/test",
        substrate_device="cpu",
    )
    second = write_character_runtime_stack_attestation(
        output_dir=tmp_path / "evidence",
        common_adapter_bundle=assets.common_adapter_bundle,
        character_runtime_assets=assets,
        substrate_model_id="Qwen/test",
        substrate_device="cpu",
    )

    assert first == second
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["common_adapter"]["bundle_id"] == "bundle-v1"
    assert payload["session_bindings"] == [
        {
            "character_id": "character-one",
            "character_lora_figure_id": None,
            "manifest_package_id": "manifest:character-one",
            "prefix_package_id": "prefix:character-one",
            "wiring_level": "active",
        }
    ]
    assert len(payload["attestation_sha256"]) == 64
