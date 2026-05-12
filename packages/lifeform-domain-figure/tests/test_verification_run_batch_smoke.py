"""Wave I smoke: figure_verify run-batch covers all 7 verification axes.

Validates the closure contract for debt #28 / Wave I: every
:class:`CheckKind` in ``IMPLEMENTED_CHECK_KINDS`` lands at least one
ledger entry per provenance, even when figure-context / per-source
metadata extras are missing (those land ``NEEDS_REVIEW`` rather
than silently disappearing).

The test exercises the script as if invoked from CLI by importing
its main + manipulating ``sys.argv`` indirectly via ``argv``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


# Add the scripts dir to sys.path so ``import figure_verify`` works.
_SCRIPTS_DIR = (
    Path(__file__).resolve().parent.parent / "scripts"
)
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import figure_verify  # noqa: E402

from lifeform_domain_figure.verification import (  # noqa: E402
    CheckKind,
    IMPLEMENTED_CHECK_KINDS,
    VerificationLedger,
)


def _provenance_line(*, source_id: str, byte_sha256: str, **extras) -> str:
    payload = {
        "source_id": source_id,
        "figure_id": "einstein",
        "source_url": f"https://example.invalid/{source_id}",
        "license_label": "Public Domain (PD-old-100)",
        "legal_clearance": "public_domain_global",
        "capture_method": "scan_reviewed_ocr",
        "captured_by": "curator-test",
        "captured_at_iso": "2026-05-12T00:00:00Z",
        "byte_sha256": byte_sha256,
        "provenance_note": "Reviewed for run-batch smoke.",
        "jurisdiction_hint": "US/EU",
        **extras,
    }
    return json.dumps(payload, ensure_ascii=False)


def _write_provenance_jsonl(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_run_batch_writes_seven_axes_per_anchor_offline_no_context(
    tmp_path: Path,
) -> None:
    """Without --figure-context-file, the 4 metadata axes still land
    a NEEDS_REVIEW ledger entry per anchor — never 'missing-check'."""

    sha_a = "a" * 64
    provenance_path = tmp_path / "prov.jsonl"
    _write_provenance_jsonl(
        provenance_path,
        [
            _provenance_line(
                source_id="src-a",
                byte_sha256=sha_a,
                document_year=1905,
                figure_lifespan=[1879, 1955],
            ),
        ],
    )
    rc = figure_verify.main(
        [
            "run-batch",
            "--root",
            str(tmp_path),
            "--provenance-file",
            str(provenance_path),
            "--metadata-mode",
            "offline",
        ]
    )
    assert rc == 0
    ledger = VerificationLedger(tmp_path)
    latest = ledger.latest_per_kind(sha_a)
    # Every implemented check kind must have an entry; none missing.
    for kind in IMPLEMENTED_CHECK_KINDS:
        assert kind in latest, (
            f"Wave I closure: run-batch must emit an entry for every kind, "
            f"missing {kind.value!r}"
        )


def test_run_batch_metadata_axes_default_to_needs_review_offline(
    tmp_path: Path,
) -> None:
    """Offline stub clients raise NotImplementedError on .fetch_*; the
    run-batch wrapper must catch that and emit NEEDS_REVIEW (not crash)."""

    sha = "b" * 64
    provenance_path = tmp_path / "prov.jsonl"
    _write_provenance_jsonl(
        provenance_path,
        [
            _provenance_line(
                source_id="src-b",
                byte_sha256=sha,
                document_year=1916,
                figure_lifespan=[1879, 1955],
                candidate_work_id="W4205692301",
                source_doi="10.1002/andp.19163540702",
                source_language="de",
            ),
        ],
    )
    figure_context_path = tmp_path / "einstein.context.json"
    figure_context_path.write_text(
        json.dumps(
            {
                "expected_qid": "Q937",
                "expected_birth_year": 1879,
                "expected_occupations": ["physicist"],
                "expected_openalex_author_id": "A5023888391",
                "coauthor_anchor_works": [],
                "figure_native_languages": ["de"],
            }
        ),
        encoding="utf-8",
    )
    rc = figure_verify.main(
        [
            "run-batch",
            "--root",
            str(tmp_path),
            "--provenance-file",
            str(provenance_path),
            "--figure-context-file",
            str(figure_context_path),
            "--metadata-mode",
            "offline",
        ]
    )
    assert rc == 0
    ledger = VerificationLedger(tmp_path)
    latest = ledger.latest_per_kind(sha)
    metadata_kinds = {
        CheckKind.IDENTITY_DISAMBIGUATION,
        CheckKind.AUTHORSHIP_ATTRIBUTION,
        CheckKind.VERSION_RECONCILIATION,
        CheckKind.TRANSLATION_LINEAGE,
    }
    for kind in metadata_kinds:
        check = latest[kind]
        assert check is not None
        assert check.verdict.value == "needs_review", (
            f"offline mode should yield NEEDS_REVIEW on metadata axis "
            f"{kind.value!r}; got {check.verdict!r}"
        )


def test_run_batch_invalid_metadata_mode_returns_usage_error(
    tmp_path: Path,
) -> None:
    sha = "c" * 64
    provenance_path = tmp_path / "prov.jsonl"
    _write_provenance_jsonl(
        provenance_path,
        [
            _provenance_line(
                source_id="src-c",
                byte_sha256=sha,
                document_year=1905,
                figure_lifespan=[1879, 1955],
            ),
        ],
    )
    # argparse will reject choices outside {"offline", "live"}; we
    # capture SystemExit since argparse calls sys.exit on parse error.
    with pytest.raises(SystemExit):
        figure_verify.main(
            [
                "run-batch",
                "--root",
                str(tmp_path),
                "--provenance-file",
                str(provenance_path),
                "--metadata-mode",
                "fictitious",
            ]
        )


def test_run_batch_singleton_anchor_gets_cross_source_byte_pass(
    tmp_path: Path,
) -> None:
    """Singletons (no document_group_key) get a trivially-PASS
    cross_source_byte entry so the gate doesn't see 'missing-check'
    on the dedup axis."""

    sha = "d" * 64
    provenance_path = tmp_path / "prov.jsonl"
    _write_provenance_jsonl(
        provenance_path,
        [
            _provenance_line(
                source_id="src-d",
                byte_sha256=sha,
                document_year=1905,
                figure_lifespan=[1879, 1955],
            ),
        ],
    )
    figure_verify.main(
        [
            "run-batch",
            "--root",
            str(tmp_path),
            "--provenance-file",
            str(provenance_path),
        ]
    )
    ledger = VerificationLedger(tmp_path)
    latest = ledger.latest_per_kind(sha)
    cross_check = latest[CheckKind.CROSS_SOURCE_BYTE]
    assert cross_check.verdict.value == "pass"
    assert "singleton" in " ".join(cross_check.evidence)
