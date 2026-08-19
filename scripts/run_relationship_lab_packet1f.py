#!/usr/bin/env python3
"""Run the frozen Relationship Lab P1f public-evidence audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time


_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
for _relative in (
    "packages/companion-ref-harness/src",
    "packages/lifeform-domain-emogpt/src",
    "packages/lifeform-evolution/src",
):
    sys.path.insert(0, str(_REPO_ROOT / _relative))

from companion_ref_harness.embed import (  # noqa: E402
    Embedder,
    SentenceTransformerEmbedder,
)
from huggingface_hub import snapshot_download  # noqa: E402
from huggingface_hub.errors import LocalEntryNotFoundError  # noqa: E402
from lifeform_domain_emogpt.lab import (  # noqa: E402
    RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME,
    load_relationship_transfer_dataset,
)
from lifeform_evolution.relationship_lab_packet1e import (  # noqa: E402
    load_relationship_packet1e_report,
)
from lifeform_evolution.relationship_lab_packet1f import (  # noqa: E402
    RelationshipP1fVerdict,
    assess_relationship_packet1f,
    canonical_relationship_p1f_embedder_name,
    load_relationship_packet1f_report,
    write_relationship_packet1f_report,
)


_DEFAULT_P1E_REPORT = (
    _REPO_ROOT
    / "artifacts"
    / "relationship_lab"
    / "qwen25_3b_packet1e_v2_conditioned_top4_20260820"
    / "packet1e_report.json"
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-p1e-report",
        default=str(_DEFAULT_P1E_REPORT),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            _REPO_ROOT / "artifacts" / "relationship_lab" / f"bge_m3_packet1f_v3_public_evidence_{int(time.time())}"
        ),
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow materializing the frozen BGE-M3 snapshot when absent.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate source lineage and local snapshot without embedding data.",
    )
    return parser.parse_args(argv)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_digest(snapshot: pathlib.Path) -> str:
    manifest = []
    for path in sorted(
        (item for item in snapshot.rglob("*") if item.is_file()),
        key=lambda item: str(item.relative_to(snapshot)),
    ):
        manifest.append(
            (
                str(path.relative_to(snapshot)),
                path.stat().st_size,
                _sha256_file(path),
            )
        )
    if not manifest:
        raise FileNotFoundError(f"P1f embedding snapshot is empty: {snapshot}")
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class _FrozenCachedEmbedder:
    def __init__(self, *, delegate: Embedder, canonical_name: str) -> None:
        self._delegate = delegate
        self._canonical_name = canonical_name
        self._cache: dict[str, tuple[float, ...]] = {}

    @property
    def dim(self) -> int:
        return self._delegate.dim

    @property
    def name(self) -> str:
        return self._canonical_name

    def embed(self, text: str) -> tuple[float, ...]:
        cached = self._cache.get(text)
        if cached is None:
            cached = self._delegate.embed(text)
            self._cache[text] = cached
        return cached


def _materialize_snapshot(
    *,
    model_source: str,
    allow_download: bool,
) -> tuple[pathlib.Path | None, bool]:
    try:
        cached = snapshot_download(
            repo_id=model_source,
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        if not allow_download:
            return None, False
        downloaded = snapshot_download(
            repo_id=model_source,
            local_files_only=False,
        )
        return pathlib.Path(downloaded), True
    return pathlib.Path(cached), False


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    dataset = load_relationship_transfer_dataset(package_name=RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME)
    contract = dataset.public_evidence_contract
    if contract is None:
        raise RuntimeError("validated P1f dataset lost its public evidence contract")
    source_report = load_relationship_packet1e_report(pathlib.Path(args.source_p1e_report))
    if (
        source_report.artifact_id != contract.source_p1e_report_artifact_id
        or source_report.verdict.value != contract.source_required_verdict
    ):
        raise ValueError("P1f source report does not satisfy the frozen trigger")
    snapshot, downloaded = _materialize_snapshot(
        model_source=contract.semantic_audit_model_source,
        allow_download=args.allow_download,
    )
    if snapshot is None:
        print(
            json.dumps(
                {
                    "package_name": dataset.package_name,
                    "dataset_fingerprint": dataset.dataset_fingerprint,
                    "source_p1e_report_artifact_id": source_report.artifact_id,
                    "model_source": contract.semantic_audit_model_source,
                    "available": False,
                    "ready": False,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 3
    weights_sha256 = _snapshot_digest(snapshot)
    if weights_sha256 != contract.semantic_audit_weights_sha256:
        raise ValueError("local BGE-M3 snapshot digest diverges from P1f contract")
    preflight = {
        "package_name": dataset.package_name,
        "dataset_fingerprint": dataset.dataset_fingerprint,
        "public_evidence_contract_sha256": contract.contract_sha256,
        "source_p1e_report_artifact_id": source_report.artifact_id,
        "model_source": contract.semantic_audit_model_source,
        "snapshot_path": str(snapshot),
        "weights_sha256": weights_sha256,
        "downloaded": downloaded,
        "available": True,
        "ready": True,
    }
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    delegate = SentenceTransformerEmbedder(model_id=str(snapshot), device="cpu")
    embedder = _FrozenCachedEmbedder(
        delegate=delegate,
        canonical_name=canonical_relationship_p1f_embedder_name(contract),
    )
    report = assess_relationship_packet1f(
        dataset=dataset,
        source_p1e_report=source_report,
        embedder=embedder,
        weights_sha256=weights_sha256,
    )
    json_path, markdown_path = write_relationship_packet1f_report(
        report,
        output_dir=pathlib.Path(args.output_dir),
    )
    loaded = load_relationship_packet1f_report(json_path)
    if loaded.artifact_id != report.artifact_id:
        raise RuntimeError("P1f strict report round-trip changed artifact identity")
    print(
        json.dumps(
            {
                **preflight,
                "artifact_id": report.artifact_id,
                "json_report": str(json_path),
                "markdown_report": str(markdown_path),
                "correct_count": report.correct_count,
                "evidence_unit_count": len(report.evidence_units),
                "top1_accuracy": report.top1_accuracy,
                "minimum_correct_anchor_margin": (report.minimum_correct_anchor_margin),
                "mean_correct_anchor_margin": report.mean_correct_anchor_margin,
                "verdict": report.verdict.value,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report.verdict is RelationshipP1fVerdict.CONSUMER_PROTOCOL_FREEZE_CANDIDATE else 2


if __name__ == "__main__":
    raise SystemExit(main())
