from __future__ import annotations

from types import SimpleNamespace

import pytest

import lifeform_domain_character
from lifeform_service.character_packages import load_character_runtime_assets
from volvence_zero.runtime import WiringLevel
from volvence_zero.substrate import (
    CharacterPrefixKVPackage,
    CommonAdapterBundle,
    default_persona_lora_pool,
)


class _CommonBundle:
    base_model_id = "Qwen/test"
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
