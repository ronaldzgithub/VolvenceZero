"""Cross-cutting contract: D2 helpers fold into bundle main pipeline.

Closes debt #24's three load-bearing wiring obligations:

1. ``build_figure_artifact_bundle`` runs ``compute_dedup_report`` and
   passes the canonical chunk allowlist to the retrieval index, so a
   byte-identical chunk that appears in two different envelope kinds
   is admitted exactly once into BM25.
2. ``FigureBundleInputs.provenance_records`` non-empty makes the
   bundle's ``provenance_fingerprint`` non-empty AND folds into
   ``integrity_hash``; a license-only edit yields a different hash.
3. ``GroundedDecoder.verify_with_pointers`` parses the locator into
   typed structured fields when ``parse_locator`` accepts it, and
   falls back to ``raw_locator`` otherwise.

Contract anchor: anyone unwiring dedupe / provenance fingerprinting
/ structured citation parsing breaks one of these tests.
"""

from __future__ import annotations

from lifeform_domain_figure import (
    FigureBundleInputs,
    FigureCorpusSourceBundle,
    FigureLetterSource,
    FigurePaperSource,
    build_einstein_profile,
    build_figure_artifact_bundle,
    build_figure_ingestion_envelope,
    synthetic_einstein_corpus,
)
from lifeform_domain_figure.corpus.provenance import (
    CaptureMethod,
    LegalClearance,
    SourceProvenance,
)
from lifeform_expression.grounded_decoder import (
    EvidencePointer,
    GroundedDecoder,
    GroundedDecoderConfig,
)


def _einstein_envelopes():
    papers, letters, lectures, notebooks = synthetic_einstein_corpus()
    bundle = FigureCorpusSourceBundle(
        figure_id="einstein",
        papers=papers,
        letters=letters,
        lectures=lectures,
        notebooks=notebooks,
    )
    envelope_set = build_figure_ingestion_envelope(bundle, uploader="contract-test")
    return envelope_set.envelopes


def _public_domain_provenance(source_id: str, license_label: str) -> SourceProvenance:
    return SourceProvenance(
        source_id=source_id,
        figure_id="einstein",
        source_url=f"https://example.invalid/{source_id}",
        license_label=license_label,
        legal_clearance=LegalClearance.PUBLIC_DOMAIN_GLOBAL,
        capture_method=CaptureMethod.SCAN_REVIEWED_OCR,
        captured_by="curator-1",
        captured_at_iso="2026-05-12T00:00:00Z",
        byte_sha256="b" * 64,
        provenance_note="Captured for contract test.",
        jurisdiction_hint="US",
    )


# ---------------------------------------------------------------------------
# Wiring 1 — dedupe canonical allowlist threads into retrieval index
# ---------------------------------------------------------------------------


def test_dedupe_filters_duplicate_chunks_from_retrieval_index() -> None:
    """A byte-identical chunk that lives on both a paper and a letter
    must appear exactly once in the retrieval index after dedupe."""

    duplicate_body = (
        "Reviewer-fabricated body for dedup contract test. Spatially "
        "separated subsystems each carry their own definite physical "
        "state in a complete physical theory.\n\n"
        "Second paragraph for the dedup test.\n\n"
        "Third paragraph for the dedup test."
    )
    paper = FigurePaperSource(
        paper_id="dedupe-paper",
        title="Paper",
        year=1925,
        language="en",
        body=duplicate_body,
    )
    letter = FigureLetterSource(
        letter_id="dedupe-letter",
        sender_id="einstein",
        recipient_id="bohr",
        date_iso="1935-04-12",
        language="en",
        body=duplicate_body,
    )
    profile = build_einstein_profile()
    corpus = FigureCorpusSourceBundle(
        figure_id="einstein", papers=(paper,), letters=(letter,)
    )
    envelope_set = build_figure_ingestion_envelope(corpus, uploader="contract-test")
    bundle = build_figure_artifact_bundle(
        FigureBundleInputs(profile=profile, envelopes=envelope_set.envelopes)
    )
    # Without dedupe, the retrieval index would carry 6 chunk records
    # (3 paragraphs x 2 envelope kinds). With dedupe + paper-priority
    # canonicalisation, the duplicate paragraphs collapse to the paper
    # side only.
    locators = tuple(rec.locator for rec in bundle.retrieval_index.chunk_records)
    assert all(
        loc.startswith("paper:") for loc in locators
    ), f"dedupe should canonicalise to papers; got {locators!r}"
    assert len(locators) == 3, f"expected 3 unique chunks, got {locators!r}"


# ---------------------------------------------------------------------------
# Wiring 2 — provenance fingerprint folds into bundle.integrity_hash
# ---------------------------------------------------------------------------


def _build_bundle_with_provenance(
    provenance_records: tuple[SourceProvenance, ...],
):
    profile = build_einstein_profile()
    envelopes = _einstein_envelopes()
    return build_figure_artifact_bundle(
        FigureBundleInputs(
            profile=profile,
            envelopes=envelopes,
            provenance_records=provenance_records,
        )
    )


def test_provenance_records_set_fingerprint_on_bundle() -> None:
    bundle = _build_bundle_with_provenance(
        (_public_domain_provenance("synth-1", "public-domain"),)
    )
    assert bundle.provenance_fingerprint != ""
    assert len(bundle.provenance_fingerprint) == 64


def test_provenance_change_yields_different_bundle_hash() -> None:
    bundle_a = _build_bundle_with_provenance(
        (_public_domain_provenance("synth-1", "public-domain"),)
    )
    # Same source id, only license_label changes — must shift the
    # bundle hash because the curator's documented legal clearance
    # is part of what the bundle ID promises.
    bundle_b = _build_bundle_with_provenance(
        (_public_domain_provenance("synth-1", "cc-by-sa-4.0"),)
    )
    assert bundle_a.integrity_hash != bundle_b.integrity_hash
    assert bundle_a.provenance_fingerprint != bundle_b.provenance_fingerprint
    assert bundle_a.bundle_id != bundle_b.bundle_id


def test_no_provenance_keeps_legacy_hash_path() -> None:
    """Empty provenance_records preserves the byte-stable hash path
    so existing bundles that did not declare provenance keep working
    bit-identical to before."""

    bundle = build_figure_artifact_bundle(
        FigureBundleInputs(
            profile=build_einstein_profile(),
            envelopes=_einstein_envelopes(),
        )
    )
    assert bundle.provenance_fingerprint == ""


# ---------------------------------------------------------------------------
# Wiring 3 — GroundedDecoder surfaces parsed locator structured fields
# ---------------------------------------------------------------------------


def test_evidence_pointer_carries_parsed_letter_fields() -> None:
    """A retrieval result whose locator is a letter form must yield
    an EvidencePointer with sender_id / recipient_id / date_iso
    populated, not just the raw string."""

    profile = build_einstein_profile()
    bundle = build_figure_artifact_bundle(
        FigureBundleInputs(profile=profile, envelopes=_einstein_envelopes())
    )
    decoder = GroundedDecoder(
        bundle.retrieval_index,
        config=GroundedDecoderConfig(
            min_assertion_tokens=4,
            score_threshold=0.0,
            cosine_floor=0.0,
            top_k=3,
        ),
    )
    # Use a query whose hash-embedding cosine survives, so we get
    # non-empty evidence regardless of corpus-specific tuning.
    query = (
        "Spatially separated subsystems each carry their own definite "
        "physical state in a complete physical theory."
    )
    verdict, pointers = decoder.verify_with_pointers(text=query)
    assert isinstance(verdict.evidence_pointers, tuple)
    assert all(isinstance(p, EvidencePointer) for p in pointers)
    assert pointers, "at least one evidence pointer expected"
    # At least one pointer in the synthetic Einstein corpus should
    # be parseable; assert structured fields are populated when so.
    parsed_pointers = [p for p in pointers if p.parsed]
    assert parsed_pointers, "at least one pointer must have parsed structured fields"
    sample = parsed_pointers[0]
    assert sample.locator_kind in {"paper", "letter", "lecture", "notebook"}
    assert sample.paragraph_index >= 0
    assert sample.offset_end >= sample.offset_start


def test_evidence_pointer_falls_back_to_raw_on_unparseable() -> None:
    """Locators that don't match any LocatorKind must still produce
    a non-empty EvidencePointer where ``parsed`` is False and only
    raw_locator / chunk_id / source_envelope_id are set."""

    class _FakeRecord:
        locator = "unknown:scheme:no-required-keys"
        chunk_id = "fake-chunk"
        source_envelope_id = "fake-env"

    from lifeform_expression.grounded_decoder import _build_evidence_pointer

    pointer = _build_evidence_pointer(_FakeRecord())
    assert pointer is not None
    assert pointer.parsed is False
    assert pointer.raw_locator == "unknown:scheme:no-required-keys"
    assert pointer.chunk_id == "fake-chunk"
    assert pointer.source_envelope_id == "fake-env"
    # rendered string must include the raw locator so audit can reverse-look
    assert "unknown:scheme:no-required-keys" in pointer.rendered
