"""Wave J contract: curated CLI round-trip is byte-stable (R15).

Same cleaning store + same metadata file -> running ``figure-bake
bake-bundle --corpus-mode curated`` twice produces two bundles
whose ``integrity_hash`` is byte-identical. This is the R15
rollback guarantee on the curated path: a curator who edits
metadata gets a new bundle id; one who doesn't, gets the same id.

We don't shell out to the CLI here — we go through
``cmd_bake_bundle`` directly with an argparse Namespace, which is
the same code path the CLI binary takes.
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

# Reuse the cleaning_fixtures helper module from the figure tests.
_FIGURE_TESTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "packages"
    / "lifeform-domain-figure"
    / "tests"
)
if str(_FIGURE_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_FIGURE_TESTS_DIR))

from cleaning_fixtures import (  # noqa: E402
    build_minimal_cpae_pdf_bytes,
    build_wikisource_html_bytes,
)

from lifeform_domain_figure.cleaning import CleaningStore  # noqa: E402
from lifeform_domain_figure.cleaning.cleaners import (  # noqa: E402
    clean_raw_document,
)
from lifeform_domain_figure.cleaning.parsers import (  # noqa: E402
    CPAE_PDF_CONTENT_TYPE,
    WIKISOURCE_HTML_CONTENT_TYPE,
    parse_by_content_type,
)
from lifeform_domain_figure.cli._commands import cmd_bake_bundle  # noqa: E402


def _stage_minimal_cleaning_store(root: Path) -> dict[str, str]:
    """Stage one CPAE + one Wikisource raw + cleaned entry."""

    store = CleaningStore(root)
    out: dict[str, str] = {}
    for archive, data, content_type, source_url in (
        (
            "cpae",
            build_minimal_cpae_pdf_bytes(),
            CPAE_PDF_CONTENT_TYPE,
            "https://einsteinpapers.press.princeton.edu/vol2-doc/24",
        ),
        (
            "wikisource",
            build_wikisource_html_bytes(),
            WIKISOURCE_HTML_CONTENT_TYPE,
            "https://en.wikisource.org/wiki/Sample",
        ),
    ):
        sha = store.put_raw(data, source_url=source_url, content_type=content_type)
        raw = parse_by_content_type(
            data, source_url=source_url, content_type=content_type
        )
        cleaned = clean_raw_document(raw)
        store.put_cleaned(cleaned)
        out[archive] = sha
    return out


def _write_metadata(path: Path, shas: dict[str, str]) -> None:
    rows = [
        {
            "raw_sha256": shas["cpae"],
            "figure_id": "einstein",
            "archive": "cpae",
            "source_kind": "paper",
            "source_id": "cpae-rt-paper",
            "legal_clearance": "public_domain_global",
            "capture_method": "scan_reviewed_ocr",
            "captured_by": "curator-roundtrip",
            "captured_at_iso": "2026-05-12T00:00:00Z",
            "provenance_note": "Round-trip contract.",
            "license_label_override": "Public Domain (PD-old)",
            "archive_payload": {
                "document_id": "cpae-rt-1",
                "document_kind": "article",
                "volume": 2,
                "document_number": 24,
                "title": "Round-trip paper",
                "year": 1905,
                "language": "en",
            },
        },
        {
            "raw_sha256": shas["wikisource"],
            "figure_id": "einstein",
            "archive": "wikisource",
            "source_kind": "paper",
            "source_id": "ws-rt-paper",
            "legal_clearance": "public_domain_global",
            "capture_method": "transcribed",
            "captured_by": "curator-roundtrip",
            "captured_at_iso": "2026-05-12T00:00:00Z",
            "provenance_note": "Round-trip contract.",
            "license_label_override": "Public Domain (CC0)",
            "archive_payload": {
                "page_title": "Round-trip Wikisource Page",
                "language": "en",
                "year": 1905,
            },
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


def _read_first_bundle_dir(bundle_root: Path) -> Path:
    figure_dir = bundle_root / "einstein"
    candidates = sorted(p for p in figure_dir.iterdir() if p.is_dir())
    assert candidates, f"no bundle written to {figure_dir}"
    return candidates[0]


def _bake_curated(*, tmp_path: Path, run_id: str) -> dict:
    """Drive ``cmd_bake_bundle`` directly and return the JSON payload it printed."""

    cleaning_root = tmp_path / "store"
    cleaning_root.mkdir(parents=True, exist_ok=True)
    metadata_file = cleaning_root / "metadata.jsonl"
    if not (cleaning_root / "raw").exists():
        shas = _stage_minimal_cleaning_store(cleaning_root)
        _write_metadata(metadata_file, shas)
    bundle_root = tmp_path / "bundles"
    audit_root = tmp_path / "audit"
    bundle_root.mkdir(parents=True, exist_ok=True)
    audit_root.mkdir(parents=True, exist_ok=True)
    args = Namespace(
        figure="einstein",
        corpus_mode="curated",
        cleaning_root=str(cleaning_root),
        curated_metadata_file=str(metadata_file),
        verification_root=None,
        require_verification_pass=False,
        time_window_id=None,
        bundle_root=str(bundle_root),
        audit_root=str(audit_root),
    )
    rc = cmd_bake_bundle(args)
    assert rc == 0, f"cmd_bake_bundle returned non-zero {rc} for run {run_id}"
    bundle_dir = _read_first_bundle_dir(bundle_root)
    return {
        "bundle_dir": str(bundle_dir),
        "bundle_root": str(bundle_root),
    }


def test_curated_bundle_round_trip_same_metadata_yields_same_bundle_id(
    tmp_path: Path,
) -> None:
    """Two consecutive bakes from the same cleaning store + same
    metadata file produce the same bundle id (R15)."""

    from lifeform_domain_figure import load_figure_bundle

    out_a = _bake_curated(tmp_path=tmp_path / "run_a", run_id="a")
    out_b = _bake_curated(tmp_path=tmp_path / "run_b", run_id="b")
    bundle_a_dirs = sorted(
        (Path(out_a["bundle_root"]) / "einstein").iterdir()
    )
    bundle_b_dirs = sorted(
        (Path(out_b["bundle_root"]) / "einstein").iterdir()
    )
    assert len(bundle_a_dirs) == 1 and len(bundle_b_dirs) == 1
    a_id = bundle_a_dirs[0].name
    b_id = bundle_b_dirs[0].name
    assert a_id == b_id, (
        f"R15 violated: same inputs produced different bundle ids: "
        f"a={a_id!r} b={b_id!r}"
    )
    bundle_a = load_figure_bundle(
        root_dir=out_a["bundle_root"], bundle_id=a_id, figure_id="einstein"
    )
    bundle_b = load_figure_bundle(
        root_dir=out_b["bundle_root"], bundle_id=b_id, figure_id="einstein"
    )
    assert bundle_a.integrity_hash == bundle_b.integrity_hash
    assert bundle_a.provenance_fingerprint == bundle_b.provenance_fingerprint
    assert bundle_a.provenance_fingerprint != ""


def test_curated_bundle_metadata_change_yields_different_bundle_id(
    tmp_path: Path,
) -> None:
    """Editing the curator metadata (license_label_override) MUST produce a
    different bundle id — the provenance_fingerprint folds into the hash."""

    cleaning_root = tmp_path / "store"
    cleaning_root.mkdir()
    shas = _stage_minimal_cleaning_store(cleaning_root)
    _write_metadata(cleaning_root / "metadata.jsonl", shas)
    bundle_root_a = tmp_path / "bundles_a"
    audit_root_a = tmp_path / "audit_a"
    bundle_root_a.mkdir()
    audit_root_a.mkdir()
    args_a = Namespace(
        figure="einstein",
        corpus_mode="curated",
        cleaning_root=str(cleaning_root),
        curated_metadata_file=str(cleaning_root / "metadata.jsonl"),
        verification_root=None,
        require_verification_pass=False,
        time_window_id=None,
        bundle_root=str(bundle_root_a),
        audit_root=str(audit_root_a),
    )
    assert cmd_bake_bundle(args_a) == 0
    a_id = sorted((bundle_root_a / "einstein").iterdir())[0].name
    # Edit metadata: change license_label_override on one record.
    metadata_file_b = cleaning_root / "metadata_b.jsonl"
    rows = [
        json.loads(line)
        for line in (cleaning_root / "metadata.jsonl").read_text().splitlines()
        if line.strip()
    ]
    rows[0]["license_label_override"] = "Public Domain (revised by reviewer)"
    metadata_file_b.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    bundle_root_b = tmp_path / "bundles_b"
    audit_root_b = tmp_path / "audit_b"
    bundle_root_b.mkdir()
    audit_root_b.mkdir()
    args_b = Namespace(
        figure="einstein",
        corpus_mode="curated",
        cleaning_root=str(cleaning_root),
        curated_metadata_file=str(metadata_file_b),
        verification_root=None,
        require_verification_pass=False,
        time_window_id=None,
        bundle_root=str(bundle_root_b),
        audit_root=str(audit_root_b),
    )
    assert cmd_bake_bundle(args_b) == 0
    b_id = sorted((bundle_root_b / "einstein").iterdir())[0].name
    assert a_id != b_id, (
        f"changing license_label_override must yield a new bundle id; "
        f"both runs produced {a_id!r}"
    )
