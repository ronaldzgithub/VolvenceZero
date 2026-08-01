#!/usr/bin/env python3
"""Download and safely extract the official MSC v0.1 research corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

from companion_bench.msc_corpus import load_msc_manifest, load_msc_split


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise ValueError(f"MSC archive contains unsafe path {member.name!r}")
            if member.issym() or member.islnk():
                raise ValueError(f"MSC archive contains unsupported link {member.name!r}")
        bundle.extractall(destination, members=members, filter="data")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/external/msc/v0.1")
    )
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--accept-noncommercial-license", action="store_true")
    args = parser.parse_args()
    if not args.accept_noncommercial_license:
        parser.error(
            "MSC is admitted only for noncommercial research pending commercial "
            "clearance; pass --accept-noncommercial-license to acknowledge this"
        )
    manifest = load_msc_manifest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.archive or args.output_dir / "msc_v0.1.tar.gz"
    if args.archive is None and not archive.is_file():
        urllib.request.urlretrieve(str(manifest["source_url"]), archive)
    actual_sha = _sha256(archive)
    if actual_sha != manifest["archive_sha256"]:
        raise ValueError(
            f"MSC archive SHA-256 mismatch: expected {manifest['archive_sha256']}, "
            f"got {actual_sha}"
        )
    extraction = args.output_dir / "extracted"
    extraction.mkdir(parents=True, exist_ok=True)
    _safe_extract(archive, extraction)
    audits = {}
    for split in ("train", "validation", "heldout"):
        _, audit = load_msc_split(extraction, split=split, strict=True)
        audits[split] = audit.__dict__
    provenance = {
        "schema_version": "msc-download-provenance.v1",
        "source_url": manifest["source_url"],
        "archive_sha256": actual_sha,
        "license_policy": manifest["license_policy"],
        "splits": audits,
    }
    (args.output_dir / "DOWNLOAD_PROVENANCE.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
