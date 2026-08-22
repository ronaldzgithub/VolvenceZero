#!/usr/bin/env python3
"""Run the post-P1m named-reader transmission canary on local frozen BGE."""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys


_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
for _relative in (
    "packages/companion-ref-harness/src",
    "packages/lifeform-domain-emogpt/src",
    "packages/lifeform-evolution/src",
    "packages/vz-cognition/src",
    "packages/vz-contracts/src",
    "packages/vz-memory/src",
    "packages/vz-substrate/src",
    "packages/vz-temporal/src",
):
    sys.path.insert(0, str(_REPO_ROOT / _relative))

from companion_ref_harness.embed import SentenceTransformerEmbedder  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from huggingface_hub.utils import LocalEntryNotFoundError  # noqa: E402
from lifeform_evolution.relationship_lab_p4_named_reader import (  # noqa: E402
    run_relationship_p4_named_reader_transmission,
    validate_relationship_p4_named_reader_report_files,
    write_relationship_p4_named_reader_markdown,
    write_relationship_p4_named_reader_report,
)
from lifeform_evolution.relationship_lab_packet1m_qualification import (  # noqa: E402
    frozen_snapshot_manifest_sha256,
    load_relationship_p1m_qualification_plan,
    load_relationship_p1m_qualification_protocol,
    load_relationship_p1m_qualification_report,
    validate_relationship_p1m_qualification_report_files,
)


_DEFAULT_P1M_DIR = (
    _REPO_ROOT
    / "artifacts"
    / "relationship_lab"
    / "qwen25_3b_packet1m_v1_qualification_20260822"
)
_DEFAULT_OUTPUT_DIR = (
    _REPO_ROOT
    / "artifacts"
    / "relationship_lab"
    / "p4_named_reader_transmission_v1_20260822"
)


class CachedSentenceEmbedder:
    def __init__(self, snapshot: pathlib.Path) -> None:
        self._delegate = SentenceTransformerEmbedder(
            model_id=str(snapshot),
            device="cpu",
        )
        self._cache: dict[str, tuple[float, ...]] = {}

    def embed(self, text: str) -> tuple[float, ...]:
        cached = self._cache.get(text)
        if cached is None:
            cached = tuple(float(item) for item in self._delegate.embed(text))
            self._cache[text] = cached
        return cached


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1m-dir", default=str(_DEFAULT_P1M_DIR))
    parser.add_argument("--output-dir", default=str(_DEFAULT_OUTPUT_DIR))
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help="rerun the frozen mechanism and validate existing create-only artifacts",
    )
    return parser.parse_args(argv)


def snapshot_manifest_digest(snapshot: pathlib.Path) -> str:
    return frozen_snapshot_manifest_sha256(snapshot)


def materialize_snapshot(
    *,
    repo_id: str,
    revision: str,
    allow_download: bool,
) -> pathlib.Path:
    try:
        resolved = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        if not allow_download:
            raise
        resolved = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_files_only=False,
        )
    return pathlib.Path(resolved)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    p1m_dir = pathlib.Path(args.p1m_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_path = output_dir / "named_reader_transmission_report.json"
    markdown_output_path = output_dir / "named_reader_transmission_report.md"
    if args.validate_existing and not (
        output_path.is_file() and markdown_output_path.is_file()
    ):
        raise FileNotFoundError(
            "P4 named-reader validation requires both existing artifacts"
        )
    if not args.validate_existing and (
        output_path.exists() or markdown_output_path.exists()
    ):
        raise FileExistsError(
            "P4 named-reader artifacts are create-only; choose a new output directory"
        )

    plan = load_relationship_p1m_qualification_plan(
        p1m_dir / "qualification_plan.json"
    )
    protocol = load_relationship_p1m_qualification_protocol(
        p1m_dir / "qualification_protocol.json"
    )
    report = load_relationship_p1m_qualification_report(
        p1m_dir / "qualification_report.json"
    )
    validate_relationship_p1m_qualification_report_files(
        report,
        output_dir=p1m_dir,
        protocol=protocol,
        plan=plan,
    )
    snapshot = materialize_snapshot(
        repo_id=protocol.bge_model_source,
        revision=protocol.bge_model_revision,
        allow_download=args.allow_download,
    )
    weights_sha256 = snapshot_manifest_digest(snapshot)
    if weights_sha256 != protocol.bge_weights_sha256:
        raise ValueError("P4 named-reader BGE snapshot lineage drift")
    embedder = CachedSentenceEmbedder(snapshot)
    transmission = asyncio.run(
        run_relationship_p4_named_reader_transmission(
            p1m_protocol=protocol,
            p1m_report=report,
            embedder=embedder,
            embedding_model_id=protocol.reader_artifact.embedding_model_id,
            embedding_weights_sha256=weights_sha256,
        )
    )
    if args.validate_existing:
        path = output_path
        markdown_path = markdown_output_path
    else:
        path = write_relationship_p4_named_reader_report(
            transmission,
            output_dir=output_dir,
        )
        markdown_path = write_relationship_p4_named_reader_markdown(
            transmission,
            output_dir=output_dir,
        )
    validate_relationship_p4_named_reader_report_files(
        transmission,
        report_path=path,
        markdown_path=markdown_path,
    )
    print(
        json.dumps(
            {
                "artifact_id": transmission.artifact_id,
                "verdict": transmission.verdict,
                "matched_action_change_count": (
                    transmission.matched_action_change_count
                ),
                "preferred_action_match_gain": (
                    transmission.preferred_action_match_gain
                ),
                "positive_outcome_gain": transmission.positive_outcome_gain,
                "report_path": str(path),
                "terminal_integrity_validated": True,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
