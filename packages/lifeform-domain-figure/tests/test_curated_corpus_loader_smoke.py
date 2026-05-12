"""Smoke tests for ``load_curated_corpus_from_cleaning_store`` (Wave J).

The loader walks an L1 cleaning store + a curator-staged metadata
JSONL and emits a :class:`CuratedCorpusBundle` ready to feed
:class:`FigureBundleInputs`. Tests use minimal in-memory fixtures
that round-trip through the real bridges (``cleaned_to_*_payload``
+ ``*_to_*_source``) so a regression in any bridge surfaces here.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

# cleaning_fixtures is a sibling test helper module, not a wheel module.
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from cleaning_fixtures import (  # noqa: E402
    build_gutenberg_text_bytes,
    build_minimal_cpae_pdf_bytes,
    build_wikisource_html_bytes,
)

from lifeform_domain_figure import (  # noqa: E402
    CuratedCorpusBundle,
    CuratedSourceMetadata,
    FigureBundleInputs,
    FigureCorpusSourceBundle,
    build_einstein_profile,
    build_figure_artifact_bundle,
    build_figure_ingestion_envelope,
    load_curated_corpus_from_cleaning_store,
    load_curated_metadata_jsonl,
)
from lifeform_domain_figure.cleaning import CleaningStore  # noqa: E402
from lifeform_domain_figure.cleaning.cleaners import (  # noqa: E402
    CURRENT_CLEANER_PIPELINE_VERSION,
    clean_raw_document,
)
from lifeform_domain_figure.cleaning.parsers import (  # noqa: E402
    CPAE_PDF_CONTENT_TYPE,
    GUTENBERG_TEXT_CONTENT_TYPE,
    WIKISOURCE_HTML_CONTENT_TYPE,
    parse_by_content_type,
)


def _seed_cleaning_store(
    root: Path,
) -> dict[str, str]:
    """Populate a CleaningStore with one CPAE + one Wikisource + one Gutenberg
    fixture; return ``{archive: raw_sha256}``.
    """

    store = CleaningStore(root)
    out: dict[str, str] = {}
    cases = [
        ("cpae", build_minimal_cpae_pdf_bytes(), CPAE_PDF_CONTENT_TYPE,
         "https://einsteinpapers.press.princeton.edu/vol2-doc/24"),
        ("wikisource", build_wikisource_html_bytes(), WIKISOURCE_HTML_CONTENT_TYPE,
         "https://en.wikisource.org/wiki/Sample"),
        ("gutenberg", build_gutenberg_text_bytes(), GUTENBERG_TEXT_CONTENT_TYPE,
         "https://www.gutenberg.org/files/0/0-0.txt"),
    ]
    for archive, data, content_type, source_url in cases:
        sha = store.put_raw(data, source_url=source_url, content_type=content_type)
        raw = parse_by_content_type(
            data, source_url=source_url, content_type=content_type
        )
        cleaned = clean_raw_document(raw)
        store.put_cleaned(cleaned)
        out[archive] = sha
    return out


def _write_metadata_file(root: Path, shas: dict[str, str]) -> Path:
    """Build a curator metadata JSONL covering the 3 anchors."""

    rows = [
        {
            "raw_sha256": shas["cpae"],
            "figure_id": "einstein",
            "archive": "cpae",
            "source_kind": "paper",
            "source_id": "cpae-test-paper",
            "legal_clearance": "public_domain_global",
            "capture_method": "scan_reviewed_ocr",
            "captured_by": "curator-test",
            "captured_at_iso": "2026-05-12T00:00:00Z",
            "provenance_note": "Curator review of synthetic CPAE fixture.",
            "license_label_override": "Public Domain (PD-old)",
            "archive_payload": {
                "document_id": "cpae-test-1",
                "document_kind": "article",
                "volume": 2,
                "document_number": 24,
                "title": "Test paper",
                "year": 1905,
                "language": "en",
            },
        },
        {
            "raw_sha256": shas["wikisource"],
            "figure_id": "einstein",
            "archive": "wikisource",
            "source_kind": "paper",
            "source_id": "wikisource-test",
            "legal_clearance": "public_domain_global",
            "capture_method": "transcribed",
            "captured_by": "curator-test",
            "captured_at_iso": "2026-05-12T00:00:00Z",
            "provenance_note": "Curator review of synthetic wikisource fixture.",
            "license_label_override": "Public Domain (CC0)",
            "archive_payload": {
                "page_title": "Sample Wikisource Page",
                "language": "en",
                "year": 1905,
            },
        },
        {
            "raw_sha256": shas["gutenberg"],
            "figure_id": "einstein",
            "archive": "gutenberg",
            "source_kind": "paper",
            "source_id": "gutenberg-test",
            "legal_clearance": "public_domain_global",
            "capture_method": "ocr",
            "captured_by": "curator-test",
            "captured_at_iso": "2026-05-12T00:00:00Z",
            "provenance_note": "Curator review of synthetic Gutenberg fixture.",
            "license_label_override": "Project Gutenberg License",
            "archive_payload": {
                "ebook_id": 12345,
                "title": "Test ebook",
                "language": "en",
                "year": 1916,
            },
        },
    ]
    path = root / "curated.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return path


def test_loader_emits_typed_sources_per_archive(tmp_path: Path) -> None:
    cleaning_root = tmp_path / "store"
    cleaning_root.mkdir()
    shas = _seed_cleaning_store(cleaning_root)
    metadata_file = _write_metadata_file(cleaning_root, shas)
    bundle = load_curated_corpus_from_cleaning_store(
        cleaning_root=cleaning_root,
        figure_id="einstein",
        metadata_file=metadata_file,
    )
    assert isinstance(bundle, CuratedCorpusBundle)
    assert bundle.figure_id == "einstein"
    # 3 papers (CPAE + Wikisource + Gutenberg all chosen as paper kind).
    assert len(bundle.papers) == 3
    assert bundle.letters == ()
    assert bundle.lectures == ()
    assert bundle.notebooks == ()
    assert len(bundle.provenance_records) == 3
    # Provenance fingerprints differ across archives (license_label_override
    # propagates to provenance.license_label).
    license_labels = {prov.license_label for prov in bundle.provenance_records}
    assert {
        "Public Domain (PD-old)",
        "Public Domain (CC0)",
        "Project Gutenberg License",
    }.issubset(license_labels)


def test_loader_dedups_metadata_and_drops_unmatched(tmp_path: Path) -> None:
    """Anchors that are in the cleaning store but NOT in the metadata
    file are dropped; anchors named in the metadata file but missing
    from the store raise FileNotFoundError."""

    cleaning_root = tmp_path / "store"
    cleaning_root.mkdir()
    shas = _seed_cleaning_store(cleaning_root)
    # Metadata file omits the gutenberg anchor — it must not appear in
    # the resulting bundle.
    rows = [
        {
            "raw_sha256": shas["cpae"],
            "figure_id": "einstein",
            "archive": "cpae",
            "source_kind": "paper",
            "source_id": "cpae-only",
            "legal_clearance": "public_domain_global",
            "capture_method": "scan_reviewed_ocr",
            "captured_by": "curator-test",
            "captured_at_iso": "2026-05-12T00:00:00Z",
            "provenance_note": "Cherry-picked.",
            "archive_payload": {
                "document_id": "cpae-test-1",
                "document_kind": "article",
                "volume": 2,
                "document_number": 24,
                "title": "Test paper",
                "year": 1905,
                "language": "en",
            },
        },
    ]
    metadata_file = cleaning_root / "subset.jsonl"
    metadata_file.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    bundle = load_curated_corpus_from_cleaning_store(
        cleaning_root=cleaning_root,
        figure_id="einstein",
        metadata_file=metadata_file,
    )
    assert len(bundle.papers) == 1
    assert bundle.source_count_by_kind["paper"] == 1


def test_loader_raises_on_metadata_anchor_missing_from_store(
    tmp_path: Path,
) -> None:
    cleaning_root = tmp_path / "store"
    cleaning_root.mkdir()
    # Empty store — metadata refers to a sha that doesn't exist.
    fake_sha = "f" * 64
    rows = [
        {
            "raw_sha256": fake_sha,
            "figure_id": "einstein",
            "archive": "cpae",
            "source_kind": "paper",
            "source_id": "cpae-test",
            "legal_clearance": "public_domain_global",
            "capture_method": "scan_reviewed_ocr",
            "captured_by": "curator-test",
            "captured_at_iso": "2026-05-12T00:00:00Z",
            "provenance_note": "test",
            "archive_payload": {
                "document_id": "cpae-test-1",
                "document_kind": "article",
                "volume": 2,
                "document_number": 24,
                "title": "Test paper",
                "year": 1905,
                "language": "en",
            },
        },
    ]
    metadata_file = cleaning_root / "metadata.jsonl"
    metadata_file.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError, match="has metadata but no cleaned"):
        load_curated_corpus_from_cleaning_store(
            cleaning_root=cleaning_root,
            figure_id="einstein",
            metadata_file=metadata_file,
        )


def test_loader_validates_metadata_schema(tmp_path: Path) -> None:
    cleaning_root = tmp_path / "store"
    cleaning_root.mkdir()
    metadata_file = cleaning_root / "bad.jsonl"
    metadata_file.write_text(
        json.dumps({"raw_sha256": "abc", "archive": "cpae"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing required key"):
        load_curated_metadata_jsonl(metadata_file)


def test_loader_rejects_duplicate_raw_sha(tmp_path: Path) -> None:
    cleaning_root = tmp_path / "store"
    cleaning_root.mkdir()
    sha = "a" * 64
    row = {
        "raw_sha256": sha,
        "figure_id": "einstein",
        "archive": "cpae",
        "source_kind": "paper",
        "source_id": "cpae-test",
        "legal_clearance": "public_domain_global",
        "capture_method": "scan_reviewed_ocr",
        "captured_by": "curator-test",
        "captured_at_iso": "2026-05-12T00:00:00Z",
        "provenance_note": "dup test",
        "archive_payload": {
            "document_id": "cpae-test-1",
            "document_kind": "article",
            "volume": 2,
            "document_number": 24,
            "title": "Test paper",
            "year": 1905,
            "language": "en",
        },
    }
    metadata_file = cleaning_root / "dup.jsonl"
    metadata_file.write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate raw_sha256"):
        load_curated_metadata_jsonl(metadata_file)


def test_curated_bundle_feeds_build_figure_artifact_bundle(tmp_path: Path) -> None:
    """End-to-end: loader output feeds build_figure_artifact_bundle and
    yields a non-trivial bundle whose ``provenance_fingerprint`` is
    non-empty (because curator supplied real provenance records)."""

    cleaning_root = tmp_path / "store"
    cleaning_root.mkdir()
    shas = _seed_cleaning_store(cleaning_root)
    metadata_file = _write_metadata_file(cleaning_root, shas)
    curated = load_curated_corpus_from_cleaning_store(
        cleaning_root=cleaning_root,
        figure_id="einstein",
        metadata_file=metadata_file,
    )
    profile = build_einstein_profile()
    corpus_bundle = FigureCorpusSourceBundle(
        figure_id="einstein",
        papers=curated.papers,
        letters=curated.letters,
        lectures=curated.lectures,
        notebooks=curated.notebooks,
    )
    envelope_set = build_figure_ingestion_envelope(
        corpus_bundle, uploader="curator-test"
    )
    bundle = build_figure_artifact_bundle(
        FigureBundleInputs(
            profile=profile,
            envelopes=envelope_set.envelopes,
            provenance_records=curated.provenance_records,
        )
    )
    assert bundle.provenance_fingerprint != ""
    assert "einstein" in bundle.bundle_id


def test_curated_bundle_round_trip_byte_identical(tmp_path: Path) -> None:
    """R15 contract: same cleaning store + same metadata file -> same
    bundle.integrity_hash byte-for-byte."""

    cleaning_root = tmp_path / "store"
    cleaning_root.mkdir()
    shas = _seed_cleaning_store(cleaning_root)
    metadata_file = _write_metadata_file(cleaning_root, shas)
    bundles = []
    for _attempt in range(2):
        curated = load_curated_corpus_from_cleaning_store(
            cleaning_root=cleaning_root,
            figure_id="einstein",
            metadata_file=metadata_file,
        )
        corpus_bundle = FigureCorpusSourceBundle(
            figure_id="einstein",
            papers=curated.papers,
            letters=curated.letters,
            lectures=curated.lectures,
            notebooks=curated.notebooks,
        )
        envelope_set = build_figure_ingestion_envelope(
            corpus_bundle, uploader="curator-test-rt"
        )
        bundle = build_figure_artifact_bundle(
            FigureBundleInputs(
                profile=build_einstein_profile(),
                envelopes=envelope_set.envelopes,
                provenance_records=curated.provenance_records,
            )
        )
        bundles.append(bundle)
    assert bundles[0].integrity_hash == bundles[1].integrity_hash
    assert bundles[0].bundle_id == bundles[1].bundle_id
    assert bundles[0].provenance_fingerprint == bundles[1].provenance_fingerprint


def test_curated_metadata_record_validates_archive_and_kind() -> None:
    with pytest.raises(ValueError, match="archive"):
        CuratedSourceMetadata(
            raw_sha256="a" * 64,
            figure_id="einstein",
            archive="weird",
            source_kind="paper",
            source_id="x",
            legal_clearance="public_domain_global",
            capture_method="scan_reviewed_ocr",
            captured_by="x",
            captured_at_iso="x",
            provenance_note="x",
        )
    with pytest.raises(ValueError, match="source_kind"):
        CuratedSourceMetadata(
            raw_sha256="a" * 64,
            figure_id="einstein",
            archive="cpae",
            source_kind="manuscript",
            source_id="x",
            legal_clearance="public_domain_global",
            capture_method="scan_reviewed_ocr",
            captured_by="x",
            captured_at_iso="x",
            provenance_note="x",
        )
