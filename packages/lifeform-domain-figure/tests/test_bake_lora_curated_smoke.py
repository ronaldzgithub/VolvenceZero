"""Wave N smoke: ``bake-lora --corpus-mode curated`` runs the LoRA
training plan against curated cleaning-store envelopes (not synthetic
corpus) so the LoRA delta is consistent with the curated bundle's
``domain_package``.

We use ``--backend synthetic`` to keep CI fast; the PEFT path is
covered by ``test_lora_bake_smoke.py::test_peft_backend_bake_real_loop_smoke``
which still uses synthetic envelopes (the curated path is orthogonal
to the bake backend choice).
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from volvence_zero.substrate import default_persona_lora_pool

# Reuse cleaning_fixtures sibling module from the figure tests.
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from cleaning_fixtures import (  # noqa: E402
    build_minimal_cpae_pdf_bytes,
    build_wikisource_html_bytes,
)

from lifeform_domain_figure import (  # noqa: E402
    FigureBundleInputs,
    FigureCorpusSourceBundle,
    build_einstein_profile,
    build_figure_artifact_bundle,
    build_figure_ingestion_envelope,
    load_curated_corpus_from_cleaning_store,
    save_figure_bundle,
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
from lifeform_domain_figure.cli._commands import cmd_bake_lora  # noqa: E402


def _stage_cleaning_store(root: Path) -> dict[str, str]:
    store = CleaningStore(root)
    out: dict[str, str] = {}
    for archive, data, content_type, source_url in (
        ("cpae", build_minimal_cpae_pdf_bytes(), CPAE_PDF_CONTENT_TYPE,
         "https://einsteinpapers.press.princeton.edu/vol2-doc/24"),
        ("wikisource", build_wikisource_html_bytes(),
         WIKISOURCE_HTML_CONTENT_TYPE,
         "https://en.wikisource.org/wiki/Wave-N-Curated-LoRA"),
    ):
        sha = store.put_raw(data, source_url=source_url, content_type=content_type)
        raw = parse_by_content_type(
            data, source_url=source_url, content_type=content_type
        )
        store.put_cleaned(clean_raw_document(raw))
        out[archive] = sha
    return out


def _write_metadata_file(path: Path, shas: dict[str, str]) -> None:
    rows = [
        {
            "raw_sha256": shas["cpae"],
            "figure_id": "einstein",
            "archive": "cpae",
            "source_kind": "paper",
            "source_id": "wave-n-cpae",
            "legal_clearance": "public_domain_global",
            "capture_method": "scan_reviewed_ocr",
            "captured_by": "wave-n-test",
            "captured_at_iso": "2026-05-12T00:00:00Z",
            "provenance_note": "Wave N curated bake test.",
            "license_label_override": "Public Domain",
            "archive_payload": {
                "document_id": "wave-n-cpae-1",
                "document_kind": "article",
                "volume": 2,
                "document_number": 24,
                "title": "Wave N Test Paper",
                "year": 1905,
                "language": "en",
            },
        },
        {
            "raw_sha256": shas["wikisource"],
            "figure_id": "einstein",
            "archive": "wikisource",
            "source_kind": "paper",
            "source_id": "wave-n-ws",
            "legal_clearance": "public_domain_global",
            "capture_method": "transcribed",
            "captured_by": "wave-n-test",
            "captured_at_iso": "2026-05-12T00:00:00Z",
            "provenance_note": "Wave N curated bake test.",
            "license_label_override": "Public Domain",
            "archive_payload": {
                "page_title": "Wave N Sample Wikisource",
                "language": "en",
                "year": 1905,
            },
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


def _bake_curated_bundle(*, cleaning_root: Path, metadata_file: Path,
                          bundle_root: Path) -> str:
    """Helper: bake a curated bundle so cmd_bake_lora has a base to attach to."""

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
        corpus_bundle, uploader="wave-n-test"
    )
    bundle = build_figure_artifact_bundle(
        FigureBundleInputs(
            profile=profile,
            envelopes=envelope_set.envelopes,
            provenance_records=curated.provenance_records,
        )
    )
    save_figure_bundle(bundle, root_dir=bundle_root)
    return bundle.bundle_id


def test_bake_lora_curated_synthetic_backend_uses_curated_envelopes(
    tmp_path: Path,
) -> None:
    """``bake-lora --corpus-mode curated --backend synthetic`` walks the
    curated cleaning store + metadata file; the resulting LoRA's
    ``training_plan_hash`` is therefore distinct from a synthetic-corpus
    LoRA on the same figure."""

    cleaning_root = tmp_path / "store"
    cleaning_root.mkdir()
    shas = _stage_cleaning_store(cleaning_root)
    metadata_file = cleaning_root / "metadata.jsonl"
    _write_metadata_file(metadata_file, shas)

    bundle_root = tmp_path / "bundles"
    bundle_root.mkdir()
    audit_root = tmp_path / "audit"
    audit_root.mkdir()
    bundle_id = _bake_curated_bundle(
        cleaning_root=cleaning_root,
        metadata_file=metadata_file,
        bundle_root=bundle_root,
    )

    pool = default_persona_lora_pool()
    if pool.has("einstein"):
        pool.deregister("einstein")

    args = Namespace(
        figure="einstein",
        bundle=bundle_id,
        corpus_mode="curated",
        cleaning_root=str(cleaning_root),
        curated_metadata_file=str(metadata_file),
        backend="synthetic",
        rank=4,
        target_layer_index=0,
        peft_model_id="sshleifer/tiny-gpt2",
        peft_target_modules="c_attn",
        peft_alpha=16,
        peft_dropout=0.0,
        peft_max_steps=2,
        peft_device="cpu",
        evaluation_snapshot="default-clean",
        rollback_evidence=f"prev_lora=absent;base={bundle_id}",
        validation_delta=0.05,
        capacity_cost=0.30,
        bundle_root=str(bundle_root),
        audit_root=str(audit_root),
    )
    rc = cmd_bake_lora(args)
    assert rc == 0, "cmd_bake_lora curated should succeed"
    # Pool now carries a record for einstein with backend_id synthetic-v1
    assert pool.has("einstein")
    record = pool.lookup("einstein")
    assert record.backend_id == "synthetic-v1"
    pool.deregister("einstein")


def test_bake_lora_curated_requires_cleaning_root_and_metadata(
    tmp_path: Path,
) -> None:
    """Curated mode without --cleaning-root or --curated-metadata-file
    must exit usage error (EXIT_USAGE=1) — never silently fall back."""

    cleaning_root = tmp_path / "store"
    cleaning_root.mkdir()
    shas = _stage_cleaning_store(cleaning_root)
    metadata_file = cleaning_root / "metadata.jsonl"
    _write_metadata_file(metadata_file, shas)
    bundle_root = tmp_path / "bundles"
    bundle_root.mkdir()
    audit_root = tmp_path / "audit"
    audit_root.mkdir()
    bundle_id = _bake_curated_bundle(
        cleaning_root=cleaning_root,
        metadata_file=metadata_file,
        bundle_root=bundle_root,
    )

    args_missing_both = Namespace(
        figure="einstein",
        bundle=bundle_id,
        corpus_mode="curated",
        cleaning_root=None,
        curated_metadata_file=None,
        backend="synthetic",
        rank=4,
        target_layer_index=0,
        peft_model_id="sshleifer/tiny-gpt2",
        peft_target_modules="c_attn",
        peft_alpha=16,
        peft_dropout=0.0,
        peft_max_steps=2,
        peft_device="cpu",
        evaluation_snapshot="default-clean",
        rollback_evidence="prev=absent",
        validation_delta=0.05,
        capacity_cost=0.30,
        bundle_root=str(bundle_root),
        audit_root=str(audit_root),
    )
    assert cmd_bake_lora(args_missing_both) == 1


def test_bake_lora_synthetic_mode_unchanged(tmp_path: Path) -> None:
    """Backward-compat: --corpus-mode synthetic (or omitted) keeps the
    legacy envelope source — uses synthetic_einstein_corpus()."""

    from lifeform_domain_figure import build_einstein_lifeform

    bundle_root = tmp_path / "bundles"
    bundle_root.mkdir()
    audit_root = tmp_path / "audit"
    audit_root.mkdir()
    # Use a synthetic Einstein bundle as base.
    synth = build_einstein_lifeform()
    save_figure_bundle(synth.artifact_bundle, root_dir=bundle_root)

    pool = default_persona_lora_pool()
    if pool.has("einstein"):
        pool.deregister("einstein")

    args = Namespace(
        figure="einstein",
        bundle=synth.artifact_bundle.bundle_id,
        corpus_mode="synthetic",
        cleaning_root=None,
        curated_metadata_file=None,
        backend="synthetic",
        rank=4,
        target_layer_index=0,
        peft_model_id="sshleifer/tiny-gpt2",
        peft_target_modules="c_attn",
        peft_alpha=16,
        peft_dropout=0.0,
        peft_max_steps=2,
        peft_device="cpu",
        evaluation_snapshot="default-clean",
        rollback_evidence="prev_lora=absent",
        validation_delta=0.05,
        capacity_cost=0.30,
        bundle_root=str(bundle_root),
        audit_root=str(audit_root),
    )
    assert cmd_bake_lora(args) == 0
    assert pool.has("einstein")
    pool.deregister("einstein")
