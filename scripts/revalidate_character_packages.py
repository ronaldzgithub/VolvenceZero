#!/usr/bin/env python3
"""Batch revalidate character manifests after a common-adapter upgrade.

``fidelity-only`` manifests are rebound only when new held-out evidence and a
matching OFFLINE gate record exist. ``full-rebake`` manifests are reported as
pending and never silently re-signed against stale Prefix/KV tensors.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _src in sorted((REPO_ROOT / "packages").glob("*/src")):
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from lifeform_domain_character import (  # noqa: E402
    CharacterArtifactRef,
    CharacterLoRARef,
    CharacterPackageManifest,
    character_fidelity_evidence_from_json,
    character_package_gate_record_from_json,
    resolve_artifact_path,
    rebind_fidelity_only,
    verify_manifest_artifacts,
)
from volvence_zero.substrate import CommonAdapterBundle  # noqa: E402


def _locator(path: Path, *, manifest_path: Path) -> str:
    try:
        return path.resolve().relative_to(manifest_path.parent.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _relocate_manifest_refs(
    manifest: CharacterPackageManifest,
    *,
    old_manifest_path: Path,
    new_manifest_path: Path,
) -> CharacterPackageManifest:
    def artifact(ref: CharacterArtifactRef | None):
        if ref is None:
            return None
        path = resolve_artifact_path(ref, manifest_path=old_manifest_path)
        return replace(ref, locator=_locator(path, manifest_path=new_manifest_path))

    def lora(ref: CharacterLoRARef | None):
        if ref is None:
            return None
        path = resolve_artifact_path(ref, manifest_path=old_manifest_path)
        return replace(ref, locator=_locator(path, manifest_path=new_manifest_path))

    return CharacterPackageManifest.create(
        character_id=manifest.character_id,
        character_name=manifest.character_name,
        base_model_id=manifest.base_model_id,
        common_adapter_version=manifest.common_adapter_version,
        compatibility_fingerprint=manifest.compatibility_fingerprint,
        template_ref=artifact(manifest.template_ref),
        prefix_kv_ref=artifact(manifest.prefix_kv_ref),
        lora_ref=lora(manifest.lora_ref),
        fidelity_evidence=manifest.fidelity_evidence,
        gate_record=manifest.gate_record,
        revalidation_mode=manifest.revalidation_mode,
        description=manifest.description,
    )


def revalidate_batch(
    *,
    common_bundle_path: Path,
    manifest_paths: tuple[Path, ...],
    evidence_dir: Path,
    output_dir: Path,
) -> tuple[dict[str, object], bool]:
    bundle = CommonAdapterBundle.from_json(
        common_bundle_path.read_text(encoding="utf-8")
    )
    bundle.require_active()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    pending = False
    for source_path in manifest_paths:
        source_path = source_path.expanduser().resolve()
        manifest = CharacterPackageManifest.from_json(
            source_path.read_text(encoding="utf-8")
        )
        verify_manifest_artifacts(manifest, manifest_path=source_path)
        if (
            manifest.common_adapter_version == bundle.common_adapter_version
            and manifest.compatibility_fingerprint == bundle.compatibility_fingerprint
        ):
            rows.append(
                {
                    "character_id": manifest.character_id,
                    "status": "already-compatible",
                    "source_package_id": manifest.package_id,
                }
            )
            continue
        if manifest.revalidation_mode == "full-rebake":
            pending = True
            rows.append(
                {
                    "character_id": manifest.character_id,
                    "status": "requires-full-rebake",
                    "source_package_id": manifest.package_id,
                    "reason": (
                        "manifest forbids fidelity-only rebinding; bake Prefix/KV "
                        "on base + the new common adapter"
                    ),
                }
            )
            continue
        output_path = output_dir / f"{manifest.character_id}.manifest.json"
        relocated = _relocate_manifest_refs(
            manifest,
            old_manifest_path=source_path,
            new_manifest_path=output_path,
        )
        evidence_path = evidence_dir / f"{manifest.character_id}.fidelity-evidence.json"
        gate_path = evidence_dir / f"{manifest.character_id}.gate-record.json"
        if not evidence_path.is_file() or not gate_path.is_file():
            pending = True
            rows.append(
                {
                    "character_id": manifest.character_id,
                    "status": "missing-fidelity-or-gate",
                    "source_package_id": manifest.package_id,
                    "evidence_path": str(evidence_path),
                    "gate_path": str(gate_path),
                }
            )
            continue
        evidence = character_fidelity_evidence_from_json(
            evidence_path.read_text(encoding="utf-8")
        )
        gate = character_package_gate_record_from_json(
            gate_path.read_text(encoding="utf-8")
        )
        rebound = rebind_fidelity_only(
            relocated,
            base_model_id=bundle.base_model_id,
            common_adapter_version=bundle.common_adapter_version,
            compatibility_fingerprint=bundle.compatibility_fingerprint,
            fidelity_evidence=evidence,
            gate_record=gate,
        )
        output_path.write_text(rebound.to_json() + "\n", encoding="utf-8")
        verify_manifest_artifacts(rebound, manifest_path=output_path)
        rows.append(
            {
                "character_id": manifest.character_id,
                "status": "revalidated",
                "source_package_id": manifest.package_id,
                "package_id": rebound.package_id,
                "output": str(output_path),
            }
        )
    report: dict[str, object] = {
        "schema_version": "character-package-batch-revalidation.v1",
        "common_adapter_bundle_id": bundle.bundle_id,
        "common_adapter_version": bundle.common_adapter_version,
        "compatibility_fingerprint": bundle.compatibility_fingerprint,
        "complete": not pending,
        "results": rows,
    }
    return report, pending


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-adapter-bundle", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    report, pending = revalidate_batch(
        common_bundle_path=args.common_adapter_bundle.expanduser().resolve(),
        manifest_paths=tuple(args.manifests),
        evidence_dir=args.evidence_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"report: {args.report.expanduser().resolve()}")
    return 2 if pending else 0


if __name__ == "__main__":
    raise SystemExit(main())
