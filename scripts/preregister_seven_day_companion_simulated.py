#!/usr/bin/env python3
"""Write the immutable seven-day simulated companion preregistration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from lifeform_service.character_packages import load_character_runtime_assets
from lifeform_domain_character import (
    CharacterPackageManifest,
    verify_manifest_artifacts,
)
from volvence_zero.agent.seven_day_companion_preregistration import (
    build_seven_day_companion_preregistration,
    write_seven_day_companion_preregistration,
)
from volvence_zero.runtime import WiringLevel


def _relative_artifact(path: Path, *, repo_root: Path) -> tuple[str, str]:
    target = path.resolve()
    try:
        relative = target.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"seven-day frozen artifacts must be under repo_root: {target}") from exc
    if not target.is_file():
        raise FileNotFoundError(target)
    return relative.as_posix(), hashlib.sha256(target.read_bytes()).hexdigest()


def _character_runtime_stack(args: argparse.Namespace) -> dict[str, object] | None:
    bundle_path = args.common_adapter_bundle
    manifest_paths = tuple(args.character_package_manifest)
    requested_any = bool(bundle_path or manifest_paths or args.character_id)
    if not requested_any:
        return None
    if bundle_path is None or not manifest_paths or not args.character_id:
        raise ValueError(
            "character-stack preregistration requires --common-adapter-bundle, "
            "at least one --character-package-manifest, and --character-id"
        )
    root = args.repo_root.resolve()
    assets = load_character_runtime_assets(
        common_adapter_bundle_path=bundle_path.resolve(),
        manifest_paths=tuple(path.resolve() for path in manifest_paths),
        wiring_by_character={},
        default_wiring=WiringLevel.ACTIVE,
    )
    assets.require_binding(args.character_id)
    bundle = assets.common_adapter_bundle
    bundle_locator, bundle_sha256 = _relative_artifact(bundle_path, repo_root=root)
    manifests = []
    for path in manifest_paths:
        manifest = CharacterPackageManifest.from_json(path.read_text(encoding="utf-8"))
        manifest.require_active()
        if manifest.prefix_kv_ref is None:
            raise ValueError(f"ACTIVE character manifest lacks Prefix/KV: {manifest.character_id}")
        locator, digest = _relative_artifact(path, repo_root=root)
        template_path, prefix_path, lora_path, report_path = verify_manifest_artifacts(
            manifest,
            manifest_path=path.resolve(),
        )
        if lora_path is not None:
            raise ValueError("seven-day ACTIVE character stack does not admit Character LoRA")
        nested_paths = tuple(nested for nested in (template_path, prefix_path, report_path) if nested is not None)
        artifact_files = []
        for nested_path in nested_paths:
            nested_locator, nested_digest = _relative_artifact(
                nested_path,
                repo_root=root,
            )
            artifact_files.append(
                {
                    "locator": nested_locator,
                    "sha256": nested_digest,
                }
            )
        manifests.append(
            {
                "locator": locator,
                "sha256": digest,
                "package_id": manifest.package_id,
                "character_id": manifest.character_id,
                "prefix_package_id": manifest.prefix_kv_ref.artifact_id,
                "artifact_files": artifact_files,
            }
        )
    return {
        "mode": "base+common-adapter+character-package",
        "vertical": args.character_vertical,
        "selected_character_id": args.character_id,
        "wiring_level": "active",
        "sut_model_family": args.sut_model_family,
        "sut_max_new_tokens": args.sut_max_new_tokens,
        "common_adapter": {
            "locator": bundle_locator,
            "sha256": bundle_sha256,
            "bundle_id": bundle.bundle_id,
            "common_adapter_version": bundle.common_adapter_version,
            "compatibility_fingerprint": bundle.compatibility_fingerprint,
            "base_model_id": bundle.base_model_id,
            "base_model_weights_sha256": bundle.base_model_weights_sha256,
        },
        "character_manifests": manifests,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--created-at-unix-ms", type=int)
    parser.add_argument("--common-adapter-bundle", type=Path)
    parser.add_argument(
        "--character-package-manifest",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--character-id")
    parser.add_argument("--character-vertical", default="zhang_wuji")
    parser.add_argument("--sut-model-family", default="qwen")
    parser.add_argument("--sut-max-new-tokens", type=int, default=96)
    args = parser.parse_args()
    created_at = args.created_at_unix_ms or int(time.time() * 1000)
    runtime_stack = _character_runtime_stack(args)
    payload = build_seven_day_companion_preregistration(
        repo_root=args.repo_root,
        created_at_unix_ms=created_at,
        runtime_stack=runtime_stack,
    )
    digest = write_seven_day_companion_preregistration(
        payload=payload,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": digest,
                "schema_version": payload["schema_version"],
                "runtime_mode": (runtime_stack["mode"] if runtime_stack is not None else "base-only"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
