from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from lifeform_domain_character import (
    CharacterTemplateAdapter,
    build_zhang_wuji_profile,
    save_lifeform_template,
)
from lifeform_service.templates import ContentAddressedTemplateBinding
from lifeform_service import SessionManager
from volvence_zero.memory import build_default_memory_store


SOURCE_SHA256 = "1" * 64
CHECKPOINT_SHA256 = "2" * 64
ARTIFACT_SHA256 = "3" * 64
TEMPLATE_ID = (
    f"nwtpl_scene_v1_{CHECKPOINT_SHA256}_{ARTIFACT_SHA256}"
)


def _publish_scene_bundle(root: Path) -> tuple[Path, ContentAddressedTemplateBinding]:
    staging = root / "staging"
    saved = save_lifeform_template(
        profile=build_zhang_wuji_profile(),
        template_id=TEMPLATE_ID,
        output_dir=staging,
        memory_store=build_default_memory_store(),
        source_arc_id=None,
        replay_provenance=" + ".join(
            (
                f"scene-source:{SOURCE_SHA256}",
                f"checkpoint:{CHECKPOINT_SHA256}",
                f"artifact:{ARTIFACT_SHA256}",
                "compiler:scene-bake-v2",
            )
        ),
        preserve_memory=True,
    )
    data = saved.template_path.read_bytes()
    bundle_sha256 = hashlib.sha256(data).hexdigest()
    path = root / "novel-worlds" / "blobs" / f"{bundle_sha256}.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    binding = ContentAddressedTemplateBinding(
        template_id=TEMPLATE_ID,
        template_uri=f"novel-worlds/blobs/{bundle_sha256}.json",
        template_bundle_sha256=bundle_sha256,
        template_source_sha256=SOURCE_SHA256,
    )
    return path, binding


def _consume(
    path: Path,
    binding: ContentAddressedTemplateBinding,
):
    return CharacterTemplateAdapter().build_session_context_from_content_addressed_template(
        template_path=path,
        binding=binding,
        runtime=None,
        identity_provider=None,
        memory_scope_root_dir=None,
        alpha_enabled=False,
    )


def test_real_scene_bundle_is_consumed_from_exact_content_addressed_path(
    tmp_path: Path,
) -> None:
    path, binding = _publish_scene_bundle(tmp_path)

    lifeform, context = _consume(binding.resolve_under(tmp_path), binding)

    assert lifeform is not None
    payload = context.payload["_character_payload"]
    assert payload.source_template_id == TEMPLATE_ID
    assert payload.source_arc_id is None
    assert binding.resolve_under(tmp_path) == path.resolve()


async def test_launcher_style_session_manager_consumes_real_scene_bundle(
    tmp_path: Path,
) -> None:
    path, binding = _publish_scene_bundle(tmp_path)
    adapter = CharacterTemplateAdapter()
    manager = SessionManager(
        lifeform_factory=lambda _runtime: pytest.fail(
            "content-addressed request fell back to the default factory"
        ),
        vertical_name="novel-worlds-character",
        template_adapter=adapter,
        templates_root_dir=tmp_path / "novel-worlds",
        idle_eviction_seconds=None,
    )

    session = await manager.create_session(
        session_id="real-content-addressed-scene",
        template_binding=binding,
    )
    context = manager.template_context_for(session.session_id)

    assert context is not None
    assert context.payload["_character_payload"].source_template_id == TEMPLATE_ID
    assert path.name == f"{binding.template_bundle_sha256}.json"
    assert await manager.close_session(session.session_id)


def test_scene_bundle_rejects_whole_file_hash_tampering(tmp_path: Path) -> None:
    path, binding = _publish_scene_bundle(tmp_path)
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="whole-file SHA-256 mismatch"):
        _consume(path, binding)


def test_scene_bundle_rejects_manifest_id_tampering(tmp_path: Path) -> None:
    path, binding = _publish_scene_bundle(tmp_path)
    wrong = ContentAddressedTemplateBinding(
        template_id=f"{TEMPLATE_ID}-other",
        template_uri=binding.template_uri,
        template_bundle_sha256=binding.template_bundle_sha256,
        template_source_sha256=binding.template_source_sha256,
    )

    with pytest.raises(ValueError, match="manifest template_id mismatch"):
        _consume(path, wrong)


def test_scene_bundle_rejects_source_attestation_tampering(tmp_path: Path) -> None:
    path, binding = _publish_scene_bundle(tmp_path)
    wrong = ContentAddressedTemplateBinding(
        template_id=binding.template_id,
        template_uri=binding.template_uri,
        template_bundle_sha256=binding.template_bundle_sha256,
        template_source_sha256="4" * 64,
    )

    with pytest.raises(ValueError, match="source attestation mismatch"):
        _consume(path, wrong)


def test_scene_bundle_rejects_internal_manifest_integrity_tampering(
    tmp_path: Path,
) -> None:
    path, binding = _publish_scene_bundle(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["profile"]["description"] = "tampered after manifest signing"
    data = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    tampered_path = path.with_name(f"{digest}.json")
    tampered_path.write_bytes(data)
    tampered_binding = ContentAddressedTemplateBinding(
        template_id=binding.template_id,
        template_uri=f"novel-worlds/blobs/{digest}.json",
        template_bundle_sha256=digest,
        template_source_sha256=binding.template_source_sha256,
    )

    with pytest.raises(ValueError, match="integrity hash mismatch"):
        _consume(tampered_path, tampered_binding)


def test_binding_rejects_path_and_filename_tampering(tmp_path: Path) -> None:
    _path, binding = _publish_scene_bundle(tmp_path)

    with pytest.raises(ValueError, match="must exactly match"):
        ContentAddressedTemplateBinding(
            template_id=binding.template_id,
            template_uri=f"novel-worlds/blobs/../{binding.template_bundle_sha256}.json",
            template_bundle_sha256=binding.template_bundle_sha256,
            template_source_sha256=binding.template_source_sha256,
        )
    with pytest.raises(ValueError, match="filename digest"):
        ContentAddressedTemplateBinding(
            template_id=binding.template_id,
            template_uri=f"novel-worlds/blobs/{'5' * 64}.json",
            template_bundle_sha256=binding.template_bundle_sha256,
            template_source_sha256=binding.template_source_sha256,
        )


def test_binding_rejects_symlink_escape(tmp_path: Path) -> None:
    path, binding = _publish_scene_bundle(tmp_path)

    outside = tmp_path.parent / (
        f"{tmp_path.name}-outside-{binding.template_bundle_sha256}.json"
    )
    outside.write_bytes(path.read_bytes())
    path.unlink()
    try:
        os.symlink(outside, path)
    except OSError as exc:
        pytest.skip(f"host does not permit test symlinks: {exc}")
    with pytest.raises(ValueError, match="outside the configured templates root"):
        binding.resolve_under(tmp_path)
