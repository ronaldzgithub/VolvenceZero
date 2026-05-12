"""Curated corpus loaders — bridge L1 cleaning store to bundle inputs.

Wave J closure (debt #19 partial). The L0 crawler + L1 cleaner +
L2 verifier together produce three filesystem artifacts:

* ``<root>/raw/<sha>/{bytes,sidecar.json}`` — content-addressable
  raw bytes anchored on ``raw_sha256``.
* ``<root>/cleaned/<sha>/v<N>/{text.txt,cleaning_log.json}`` — the
  cleaned text + cleaning operation log per pipeline version.
* ``<root>/verification/<sha>/checks.jsonl`` — append-only
  verifier verdicts per axis (filled by ``figure_verify run-batch``).

To compile a real curated bundle from those artifacts the bundle
builder needs **typed source records** + **provenance records**.
The translation requires three pieces of metadata the L1 store
cannot derive from raw bytes alone:

1. **Archive identity** (which adapter to use): cpae / wikisource /
   gutenberg / internet_archive.
2. **Source kind** (paper / letter / lecture / notebook).
3. **Archive-specific structural metadata** (CPAE volume +
   document_number; Wikisource page_title; Gutenberg ebook_id;
   Internet Archive identifier; plus title / year / language /
   sender / recipient / date / venue / audience as appropriate).

These come from a curator-staged JSONL **metadata file** —
typically committed to git next to the seeds list. Each line is a
:class:`CuratedSourceMetadata` record keyed by ``raw_sha256``.

The loader walks the cleaning store, picks the latest cleaned
version per anchor, looks up the metadata row, and uses the
existing ``cleaned_to_*_payload`` + ``*_to_*_source`` bridges to
produce ``Figure*Source`` records + matching
:class:`SourceProvenance` records — both ready to feed
:class:`FigureBundleInputs`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lifeform_domain_figure.cleaning.bridging import (
    cleaned_to_cpae_payload,
    cleaned_to_gutenberg_payload,
    cleaned_to_internet_archive_payload,
    cleaned_to_source_provenance,
    cleaned_to_wikisource_payload,
)
from lifeform_domain_figure.cleaning.parsers import parse_by_content_type
from lifeform_domain_figure.cleaning.raw_document import RawDocument
from lifeform_domain_figure.cleaning.store import CleaningStore
from lifeform_domain_figure.cleaning.cleaners import (
    CURRENT_CLEANER_PIPELINE_VERSION,
)
from lifeform_domain_figure.corpus.archives.cpae import (
    CPAEDocumentKind,
    cpae_to_letter_source,
    cpae_to_paper_source,
)
from lifeform_domain_figure.corpus.archives.gutenberg import (
    gutenberg_to_paper_source,
)
from lifeform_domain_figure.corpus.archives.internet_archive import (
    internet_archive_to_lecture_source,
    internet_archive_to_paper_source,
)
from lifeform_domain_figure.corpus.archives.wikisource import (
    wikisource_to_lecture_source,
    wikisource_to_paper_source,
)
from lifeform_domain_figure.corpus.ingest_lectures import FigureLectureSource
from lifeform_domain_figure.corpus.ingest_letters import FigureLetterSource
from lifeform_domain_figure.corpus.ingest_notebooks import FigureNotebookSource
from lifeform_domain_figure.corpus.ingest_papers import FigurePaperSource
from lifeform_domain_figure.corpus.provenance import (
    CaptureMethod,
    LegalClearance,
    SourceProvenance,
)


VALID_ARCHIVES = frozenset({"cpae", "wikisource", "gutenberg", "internet_archive"})
VALID_SOURCE_KINDS = frozenset({"paper", "letter", "lecture", "notebook"})


@dataclass(frozen=True)
class CuratedSourceMetadata:
    """Per-source metadata supplied by the curator's curation pass.

    One record per ``raw_sha256``. The loader joins each record
    with the matching cleaned text from the store and emits a
    typed ``Figure*Source`` + :class:`SourceProvenance`.

    ``archive_payload`` is a free-form dict holding the
    archive-specific fields (volume / document_number / page_title
    / etc.). The loader validates required keys per archive at
    dispatch time so a typo surfaces fail-loud at compile time.
    """

    raw_sha256: str
    figure_id: str
    archive: str
    source_kind: str
    source_id: str
    legal_clearance: str
    capture_method: str
    captured_by: str
    captured_at_iso: str
    provenance_note: str
    archive_payload: dict[str, Any] = field(default_factory=dict)
    license_label_override: str = ""
    jurisdiction_hint: str = ""

    def __post_init__(self) -> None:
        if self.archive not in VALID_ARCHIVES:
            raise ValueError(
                f"CuratedSourceMetadata.archive must be one of "
                f"{sorted(VALID_ARCHIVES)!r}; got {self.archive!r}"
            )
        if self.source_kind not in VALID_SOURCE_KINDS:
            raise ValueError(
                f"CuratedSourceMetadata.source_kind must be one of "
                f"{sorted(VALID_SOURCE_KINDS)!r}; got {self.source_kind!r}"
            )
        if len(self.raw_sha256) != 64:
            raise ValueError(
                f"CuratedSourceMetadata.raw_sha256 must be 64-char hex; "
                f"got {self.raw_sha256!r}"
            )
        for name in (
            "figure_id",
            "source_id",
            "legal_clearance",
            "capture_method",
            "captured_by",
            "captured_at_iso",
            "provenance_note",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"CuratedSourceMetadata.{name} must be a non-empty string "
                    f"(source_id={self.source_id!r})"
                )


@dataclass(frozen=True)
class CuratedCorpusBundle:
    """Output of :func:`load_curated_corpus_from_cleaning_store`.

    Carries enough material to feed
    :class:`FigureBundleInputs(provenance_records=..., envelopes=...)`:
    the typed source tuples (one per kind), a tuple of provenance
    records ordered by ``raw_sha256`` for deterministic
    fingerprinting, and a per-kind count for diagnostics.
    """

    figure_id: str
    papers: tuple[FigurePaperSource, ...]
    letters: tuple[FigureLetterSource, ...]
    lectures: tuple[FigureLectureSource, ...]
    notebooks: tuple[FigureNotebookSource, ...]
    provenance_records: tuple[SourceProvenance, ...]
    source_count_by_kind: dict[str, int]


def load_curated_metadata_jsonl(
    path: Path,
) -> dict[str, CuratedSourceMetadata]:
    """Read a JSONL of curator metadata records, keyed by ``raw_sha256``.

    Each line is one :class:`CuratedSourceMetadata`. Duplicate
    ``raw_sha256`` keys raise ``ValueError`` so a curator typo
    cannot silently overwrite a prior row.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"load_curated_metadata_jsonl: metadata file not found: {path}"
        )
    out: dict[str, CuratedSourceMetadata] = {}
    with path.open("r", encoding="utf-8-sig") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"{path}:{line_no} metadata line must decode to JSON object"
                )
            try:
                meta = CuratedSourceMetadata(
                    raw_sha256=str(payload["raw_sha256"]),
                    figure_id=str(payload["figure_id"]),
                    archive=str(payload["archive"]),
                    source_kind=str(payload["source_kind"]),
                    source_id=str(payload["source_id"]),
                    legal_clearance=str(payload["legal_clearance"]),
                    capture_method=str(payload["capture_method"]),
                    captured_by=str(payload["captured_by"]),
                    captured_at_iso=str(payload["captured_at_iso"]),
                    provenance_note=str(payload["provenance_note"]),
                    archive_payload=dict(payload.get("archive_payload", {})),
                    license_label_override=str(
                        payload.get("license_label_override", "")
                    ),
                    jurisdiction_hint=str(
                        payload.get("jurisdiction_hint", "")
                    ),
                )
            except KeyError as exc:
                raise ValueError(
                    f"{path}:{line_no} missing required key {exc!s}"
                ) from exc
            if meta.raw_sha256 in out:
                raise ValueError(
                    f"{path}:{line_no} duplicate raw_sha256={meta.raw_sha256[:12]}..."
                )
            out[meta.raw_sha256] = meta
    return out


def load_curated_corpus_from_cleaning_store(
    *,
    cleaning_root: Path,
    figure_id: str,
    metadata_file: Path,
    pipeline_version: int = CURRENT_CLEANER_PIPELINE_VERSION,
) -> CuratedCorpusBundle:
    """Walk the L1 cleaning store and assemble a :class:`CuratedCorpusBundle`.

    For every ``raw_sha256`` listed in the metadata file the loader
    pulls the cleaned text from ``<cleaning_root>/cleaned/<sha>/v<N>/``
    + the raw bytes from ``<cleaning_root>/raw/<sha>/`` (for the
    license_notice that powers :class:`SourceProvenance`), then
    routes to the matching ``cleaned_to_*_payload`` +
    ``*_to_*_source`` bridge.

    Anchors that exist in the cleaning store but are NOT named in
    the metadata file are silently dropped (the curator may have
    fetched URLs they later decided not to ship). Anchors named in
    the metadata file but missing from the cleaning store raise
    ``FileNotFoundError`` (no silent shipment of un-cleaned bytes).
    """

    if cleaning_root is None or not cleaning_root.exists():
        raise FileNotFoundError(
            f"load_curated_corpus_from_cleaning_store: cleaning_root "
            f"{cleaning_root!r} does not exist"
        )
    metadata = load_curated_metadata_jsonl(metadata_file)
    store = CleaningStore(cleaning_root)
    papers: list[FigurePaperSource] = []
    letters: list[FigureLetterSource] = []
    lectures: list[FigureLectureSource] = []
    notebooks: list[FigureNotebookSource] = []
    provenance_records: list[SourceProvenance] = []
    counts: dict[str, int] = {"paper": 0, "letter": 0, "lecture": 0, "notebook": 0}
    # Iterate metadata in sorted sha order so repeated runs against
    # the same cleaning store produce byte-identical bundle outputs
    # (R15 deterministic provenance fingerprint).
    for raw_sha in sorted(metadata.keys()):
        meta = metadata[raw_sha]
        if meta.figure_id != figure_id:
            # Skip rows for a different figure id but flag in counts
            # so the curator notices a stale metadata file.
            continue
        cleaned = store.get_cleaned(raw_sha, pipeline_version)
        if cleaned is None:
            raise FileNotFoundError(
                f"load_curated_corpus_from_cleaning_store: anchor "
                f"{raw_sha[:12]}... has metadata but no cleaned text "
                f"(pipeline_version={pipeline_version}); run figure_clean "
                f"re-clean-all first."
            )
        raw_bytes, raw_sidecar = store.get_raw(raw_sha)
        raw_doc = parse_by_content_type(
            raw_bytes,
            source_url=raw_sidecar.source_url,
            content_type=raw_sidecar.content_type,
        )
        provenance = cleaned_to_source_provenance(
            cleaned,
            raw_doc,
            source_id=meta.source_id,
            figure_id=meta.figure_id,
            source_url=raw_sidecar.source_url,
            legal_clearance=LegalClearance(meta.legal_clearance),
            capture_method=CaptureMethod(meta.capture_method),
            captured_by=meta.captured_by,
            captured_at_iso=meta.captured_at_iso,
            provenance_note=meta.provenance_note,
            license_label_override=meta.license_label_override,
            jurisdiction_hint=meta.jurisdiction_hint,
        )
        provenance_records.append(provenance)
        kind, source = _dispatch_source(
            cleaned=cleaned, raw_sidecar=raw_sidecar, meta=meta, figure_id=figure_id
        )
        if kind == "paper":
            assert isinstance(source, FigurePaperSource)
            papers.append(source)
        elif kind == "letter":
            assert isinstance(source, FigureLetterSource)
            letters.append(source)
        elif kind == "lecture":
            assert isinstance(source, FigureLectureSource)
            lectures.append(source)
        elif kind == "notebook":
            assert isinstance(source, FigureNotebookSource)
            notebooks.append(source)
        else:  # pragma: no cover — guarded by VALID_SOURCE_KINDS
            raise ValueError(f"unsupported source_kind={kind!r}")
        counts[kind] = counts.get(kind, 0) + 1
    if not (papers or letters or lectures or notebooks):
        raise ValueError(
            f"load_curated_corpus_from_cleaning_store: produced an empty "
            f"corpus for figure_id={figure_id!r}; metadata_file may be "
            f"empty or none of its rows match the figure"
        )
    return CuratedCorpusBundle(
        figure_id=figure_id,
        papers=tuple(papers),
        letters=tuple(letters),
        lectures=tuple(lectures),
        notebooks=tuple(notebooks),
        provenance_records=tuple(provenance_records),
        source_count_by_kind=counts,
    )


def _dispatch_source(
    *,
    cleaned,
    raw_sidecar,
    meta: CuratedSourceMetadata,
    figure_id: str,
) -> tuple[str, Any]:
    """Bridge cleaned text + curator metadata into a typed source record."""

    archive = meta.archive
    kind = meta.source_kind
    payload_dict = meta.archive_payload
    if archive == "cpae":
        cpae_payload = cleaned_to_cpae_payload(
            cleaned,
            document_id=str(payload_dict["document_id"]),
            document_kind=CPAEDocumentKind(str(payload_dict["document_kind"])),
            volume=int(payload_dict["volume"]),
            document_number=int(payload_dict["document_number"]),
            title=str(payload_dict["title"]),
            year=int(payload_dict["year"]),
            language=str(payload_dict["language"]),
            source_url=raw_sidecar.source_url,
            sender_id=str(payload_dict.get("sender_id", "")),
            recipient_id=str(payload_dict.get("recipient_id", "")),
            date_iso=str(payload_dict.get("date_iso", "")),
        )
        if kind == "paper":
            return ("paper", cpae_to_paper_source(cpae_payload, figure_id=figure_id))
        if kind == "letter":
            return ("letter", cpae_to_letter_source(cpae_payload, figure_id=figure_id))
        raise ValueError(
            f"cpae archive only supports source_kind paper / letter; "
            f"got {kind!r} for source_id={meta.source_id!r}"
        )
    if archive == "wikisource":
        ws_payload = cleaned_to_wikisource_payload(
            cleaned,
            page_title=str(payload_dict["page_title"]),
            language=str(payload_dict["language"]),
            source_url=raw_sidecar.source_url,
            year=payload_dict.get("year"),
            author_id=str(payload_dict.get("author_id", "")),
            venue_id=str(payload_dict.get("venue_id", "")),
            date_iso=str(payload_dict.get("date_iso", "")),
            audience=str(payload_dict.get("audience", "")),
        )
        if kind == "paper":
            return (
                "paper",
                wikisource_to_paper_source(ws_payload, figure_id=figure_id),
            )
        if kind == "lecture":
            return (
                "lecture",
                wikisource_to_lecture_source(ws_payload, figure_id=figure_id),
            )
        raise ValueError(
            f"wikisource archive only supports source_kind paper / lecture; "
            f"got {kind!r} for source_id={meta.source_id!r}"
        )
    if archive == "gutenberg":
        gp_payload = cleaned_to_gutenberg_payload(
            cleaned,
            ebook_id=int(payload_dict["ebook_id"]),
            title=str(payload_dict["title"]),
            language=str(payload_dict["language"]),
            source_url=raw_sidecar.source_url,
            section_label=str(payload_dict.get("section_label", "")),
            year=payload_dict.get("year"),
            author_id=str(payload_dict.get("author_id", "")),
        )
        if kind == "paper":
            return (
                "paper",
                gutenberg_to_paper_source(gp_payload, figure_id=figure_id),
            )
        raise ValueError(
            f"gutenberg archive only supports source_kind paper; "
            f"got {kind!r} for source_id={meta.source_id!r}"
        )
    if archive == "internet_archive":
        ia_payload = cleaned_to_internet_archive_payload(
            cleaned,
            identifier=str(payload_dict["identifier"]),
            title=str(payload_dict["title"]),
            language=str(payload_dict["language"]),
            source_url=raw_sidecar.source_url,
            creator_id=str(payload_dict.get("creator_id", "")),
            year=payload_dict.get("year"),
            venue_id=str(payload_dict.get("venue_id", "")),
            date_iso=str(payload_dict.get("date_iso", "")),
            audience=str(payload_dict.get("audience", "")),
        )
        if kind == "paper":
            return (
                "paper",
                internet_archive_to_paper_source(ia_payload, figure_id=figure_id),
            )
        if kind == "lecture":
            return (
                "lecture",
                internet_archive_to_lecture_source(ia_payload, figure_id=figure_id),
            )
        raise ValueError(
            f"internet_archive archive only supports source_kind paper / lecture; "
            f"got {kind!r} for source_id={meta.source_id!r}"
        )
    raise ValueError(f"unsupported archive={archive!r}")


__all__ = [
    "CuratedCorpusBundle",
    "CuratedSourceMetadata",
    "VALID_ARCHIVES",
    "VALID_SOURCE_KINDS",
    "load_curated_corpus_from_cleaning_store",
    "load_curated_metadata_jsonl",
]
