#!/usr/bin/env python3
"""Run the P4.4 exact PE-credit learned-gate isolation on local BGE."""

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

from lifeform_evolution.relationship_lab_p4_named_reader import (  # noqa: E402
    run_relationship_p4_named_reader_transmission,
    validate_relationship_p4_named_reader_report_files,
)
from lifeform_evolution.relationship_lab_p4_pe_learning import (  # noqa: E402
    run_relationship_p4_pe_credit_learning,
    validate_relationship_p4_pe_learning_report_files,
    write_relationship_p4_pe_learning_report,
)
from lifeform_evolution.relationship_lab_packet1m_qualification import (  # noqa: E402
    load_relationship_p1m_qualification_plan,
    load_relationship_p1m_qualification_protocol,
    load_relationship_p1m_qualification_report,
    validate_relationship_p1m_qualification_report_files,
)
from run_relationship_lab_p4_named_reader import (  # noqa: E402
    CachedSentenceEmbedder,
    materialize_snapshot,
    snapshot_manifest_digest,
)


_DEFAULT_P1M_DIR = (
    _REPO_ROOT
    / "artifacts"
    / "relationship_lab"
    / "qwen25_3b_packet1m_v1_qualification_20260822"
)
_DEFAULT_P4_READER_DIR = (
    _REPO_ROOT
    / "artifacts"
    / "relationship_lab"
    / "p4_named_reader_transmission_v1_20260822"
)
_DEFAULT_OUTPUT_DIR = (
    _REPO_ROOT
    / "artifacts"
    / "relationship_lab"
    / "p4_pe_credit_learning_v1_20260822"
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1m-dir", default=str(_DEFAULT_P1M_DIR))
    parser.add_argument(
        "--p4-reader-dir",
        default=str(_DEFAULT_P4_READER_DIR),
    )
    parser.add_argument("--output-dir", default=str(_DEFAULT_OUTPUT_DIR))
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--validate-existing", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    p1m_dir = pathlib.Path(args.p1m_dir)
    p4_reader_dir = pathlib.Path(args.p4_reader_dir)
    output_dir = pathlib.Path(args.output_dir)
    json_path = output_dir / "pe_credit_learning_report.json"
    markdown_path = output_dir / "pe_credit_learning_report.md"
    if args.validate_existing and not (
        json_path.is_file() and markdown_path.is_file()
    ):
        raise FileNotFoundError(
            "P4.4 validation requires both existing artifacts"
        )
    if not args.validate_existing and (
        json_path.exists() or markdown_path.exists()
    ):
        raise FileExistsError(
            "P4.4 artifacts are create-only; choose a new output directory"
        )

    plan = load_relationship_p1m_qualification_plan(
        p1m_dir / "qualification_plan.json"
    )
    protocol = load_relationship_p1m_qualification_protocol(
        p1m_dir / "qualification_protocol.json"
    )
    p1m_report = load_relationship_p1m_qualification_report(
        p1m_dir / "qualification_report.json"
    )
    validate_relationship_p1m_qualification_report_files(
        p1m_report,
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
        raise ValueError("P4.4 BGE snapshot lineage drift")
    embedder = CachedSentenceEmbedder(snapshot)

    source_report = asyncio.run(
        run_relationship_p4_named_reader_transmission(
            p1m_protocol=protocol,
            p1m_report=p1m_report,
            embedder=embedder,
            embedding_model_id=protocol.reader_artifact.embedding_model_id,
            embedding_weights_sha256=weights_sha256,
        )
    )
    validate_relationship_p4_named_reader_report_files(
        source_report,
        report_path=p4_reader_dir / "named_reader_transmission_report.json",
        markdown_path=p4_reader_dir / "named_reader_transmission_report.md",
    )
    learning_report = asyncio.run(
        run_relationship_p4_pe_credit_learning(
            p1m_protocol=protocol,
            source_report=source_report,
            embedder=embedder,
        )
    )
    if args.validate_existing:
        written_json = json_path
        written_markdown = markdown_path
    else:
        written_json, written_markdown = (
            write_relationship_p4_pe_learning_report(
                learning_report,
                output_dir=output_dir,
            )
        )
    validate_relationship_p4_pe_learning_report_files(
        learning_report,
        json_path=written_json,
        markdown_path=written_markdown,
    )
    print(
        json.dumps(
            {
                "artifact_id": learning_report.artifact_id,
                "verdict": learning_report.verdict,
                "matched_action_change_count": (
                    learning_report.matched_action_change_count
                ),
                "causal_next_pulse_action_change_count": (
                    learning_report.causal_next_pulse_action_change_count
                ),
                "preferred_action_match_gain": (
                    learning_report.preferred_action_match_gain
                ),
                "positive_outcome_gain": (
                    learning_report.positive_outcome_gain
                ),
                "report_path": str(written_json),
                "terminal_integrity_validated": True,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
