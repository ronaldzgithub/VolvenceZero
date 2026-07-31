from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from lifeform_service.character_packages import (
    CharacterRuntimeAssets,
    CharacterSessionBinding,
)
from lifeform_service.session_manager import CharacterSelectionError, SessionManager
from lifeform_service.templates import TemplateContext
from lifeform_service.vertical_registry import VerticalRegistry
from lifeform_service.verticals import VerticalSpec
from volvence_zero.runtime import WiringLevel
from volvence_zero.substrate import CharacterPrefixKVRegistry, PersonaLoRAPool


class _Session:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.turn_summaries = ()


class _Lifeform:
    def __init__(self) -> None:
        self.character_id = ""
        self.character_lora_pool = None

    async def start(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    def bind_character_package(self, *, character_id: str, lora_pool) -> None:
        self.character_id = character_id
        self.character_lora_pool = lora_pool

    def create_session(self, *, session_id: str) -> _Session:
        return _Session(session_id)


class _CharacterAdapter:
    def __init__(self) -> None:
        self.loaded_paths: list[Path] = []
        self.lifeforms: list[_Lifeform] = []

    def build_session_context_from_package_template(
        self,
        *,
        template_path: Path,
        runtime,
        identity_provider,
        memory_scope_root_dir,
        alpha_enabled: bool,
    ):
        self.loaded_paths.append(template_path)
        life = _Lifeform()
        self.lifeforms.append(life)
        return life, TemplateContext(payload={"template_path": str(template_path)})


def _spec(name: str, *, adapter=None, character_id: str = "") -> VerticalSpec:
    return VerticalSpec(
        name=name,
        factory=lambda _runtime: _Lifeform(),
        has_temporal_bootstrap=False,
        has_regime_bootstrap=False,
        template_adapter=adapter,
        character_id=character_id,
    )


def _assets(tmp_path: Path) -> CharacterRuntimeAssets:
    first = tmp_path / "zhang.json"
    second = tmp_path / "zhao.json"
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")
    pool = PersonaLoRAPool()
    return CharacterRuntimeAssets(
        common_adapter_bundle=object(),
        prefix_registry=CharacterPrefixKVRegistry(
            base_model_id="Qwen/test",
            common_adapter_version="common-v1",
            compatibility_fingerprint="compat-v1",
            entries=(),
        ),
        manifest_package_ids=("manifest-zhang", "manifest-zhao"),
        session_bindings=(
            CharacterSessionBinding(
                character_id="zhang-wuji",
                manifest_package_id="manifest-zhang",
                template_path=first,
                wiring_level=WiringLevel.ACTIVE,
                prefix_registry_key="zhang-wuji",
                character_lora_figure_id="zhang-wuji",
            ),
            CharacterSessionBinding(
                character_id="zhao-min",
                manifest_package_id="manifest-zhao",
                template_path=second,
                wiring_level=WiringLevel.SHADOW,
                prefix_registry_key="zhao-min",
                character_lora_figure_id="",
            ),
        ),
        character_lora_pool=pool,
    )


def _manager(tmp_path: Path) -> tuple[SessionManager, _CharacterAdapter]:
    adapter = _CharacterAdapter()
    registry = VerticalRegistry.from_mapping(
        (
            _spec(
                "character",
                adapter=adapter,
                character_id="zhang-wuji",
            ),
            _spec("companion"),
        ),
        default_name="character",
    )
    return (
        SessionManager(
            vertical_registry=registry,
            idle_eviction_seconds=None,
            character_runtime_assets=_assets(tmp_path),
        ),
        adapter,
    )


async def test_same_process_binds_two_manifest_characters_per_session(
    tmp_path,
) -> None:
    manager, adapter = _manager(tmp_path)

    first = await manager.create_session(session_id="first")
    second = await manager.create_session(
        session_id="second",
        character_id="zhao-min",
    )

    assert manager.character_id_for(first.session_id) == "zhang-wuji"
    assert manager.character_id_for(second.session_id) == "zhao-min"
    assert [path.name for path in adapter.loaded_paths] == ["zhang.json", "zhao.json"]
    assert [life.character_id for life in adapter.lifeforms] == [
        "zhang-wuji",
        "zhao-min",
    ]
    assert all(
        life.character_lora_pool is manager._character_runtime_assets.character_lora_pool
        for life in adapter.lifeforms
    )


async def test_session_rejects_unloaded_or_incompatible_character_selection(
    tmp_path,
) -> None:
    manager, _adapter = _manager(tmp_path)

    with pytest.raises(ValueError, match="not an enabled loaded manifest"):
        await manager.create_session(
            session_id="wrong",
            character_id="another-character",
        )
    with pytest.raises(ValueError, match="cannot load character package"):
        await manager.create_session(
            session_id="generic",
            vertical_name="companion",
            character_id="zhang-wuji",
        )

    assert manager.session_count() == 0


async def test_session_rejects_template_override_of_manifest_binding(tmp_path) -> None:
    manager, _adapter = _manager(tmp_path)

    with pytest.raises(ValueError, match="cannot override"):
        await manager.create_session(
            session_id="override",
            character_id="zhao-min",
            template_id="other-template",
        )


async def test_session_rejects_character_lora_when_contract_disables_lora(
    tmp_path,
) -> None:
    manager, _adapter = _manager(tmp_path)
    manager._persona_lora_enabled = False

    with pytest.raises(CharacterSelectionError, match="contract disables LoRA"):
        await manager.create_session(
            session_id="lora-disabled",
            character_id="zhang-wuji",
        )


async def test_disabled_or_missing_default_manifest_cannot_be_bypassed_by_omission(
    tmp_path,
) -> None:
    manager, _adapter = _manager(tmp_path)
    assets = manager._character_runtime_assets
    assert assets is not None
    manager._character_runtime_assets = replace(
        assets,
        session_bindings=(assets.require_binding("zhao-min"),),
    )

    with pytest.raises(CharacterSelectionError, match="default character_id"):
        await manager.create_session(session_id="missing-default")
