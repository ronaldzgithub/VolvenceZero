"""Neutral raw / cleaned document schema for the L1 cleaning pipeline.

The cleaning pipeline is figure-vertical-internal data preparation. It
sits **before** the existing archive ``*Payload`` schemas
(``CPAEPayload`` / ``WikisourcePayload`` / ``GutenbergPayload`` /
``InternetArchivePayload``) and turns ``(bytes, content_type,
source_url)`` into a typed cleaned text record. The bridge back to the
archive payloads lives in
:mod:`lifeform_domain_figure.cleaning.bridging`; this module only owns
the neutral schema.

Two records:

* :class:`RawDocument` — what a parser produces from raw bytes. Carries
  the extracted text plus parser self-reported quality / metadata.
* :class:`CleanedDocument` — what the cleaner pipeline produces from a
  ``RawDocument``. Carries a cleaning log of every applied
  :class:`CleaningOpRecord` so re-clean runs are reproducible and
  auditable.

Neither record references any figure-vertical typed source
(``FigurePaperSource`` etc.). Crossing into a typed source is an
explicit second step (bridging) so the cleaner stays decoupled from
the curator's archive choice.

Versioning
----------

Each parser carries a string ``parser_version`` of the form
``"<parser-id>:<int>"`` (e.g., ``"cpae-pdf:1"``). The integer monotone
increments when a parser changes output for a fixed input.

The cleaner pipeline carries an integer ``cleaner_pipeline_version``
that names a specific ordered sequence of cleaning ops. Bumping any
single cleaner op's behaviour bumps the pipeline version. See
:mod:`lifeform_domain_figure.cleaning.cleaners` for the registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CleaningOp(str, Enum):
    """Closed vocabulary of cleaning operations the pipeline can apply.

    Adding a new op kind requires (a) extending this enum, (b) adding
    a cleaner module under
    :mod:`lifeform_domain_figure.cleaning.cleaners`, and (c) bumping
    :data:`lifeform_domain_figure.cleaning.cleaners.CURRENT_CLEANER_PIPELINE_VERSION`.
    """

    BOILERPLATE_STRIP = "boilerplate_strip"
    WHITESPACE_NORMALIZE = "whitespace_normalize"
    TYPOGRAPHY_NORMALIZE = "typography_normalize"
    DEDUPE_INTRA_DOC = "dedupe_intra_doc"
    PII_REDACT = "pii_redact"
    PARAGRAPH_NORMALIZE = "paragraph_normalize"


@dataclass(frozen=True)
class CleaningOpRecord:
    """One applied cleaning op as recorded in the cleaning log.

    ``op_version`` is independent of the pipeline version so a single
    op's behaviour change is traceable at the op level even when the
    pipeline composition is unchanged.

    ``chars_before`` / ``chars_after`` are the text length before and
    after the op fired; the invariant is
    ``chars_after <= chars_before`` (cleaning is monotonically
    non-expanding).
    """

    op: CleaningOp
    op_version: str
    chars_before: int
    chars_after: int

    def __post_init__(self) -> None:
        if not isinstance(self.op_version, str) or not self.op_version.strip():
            raise ValueError(
                f"CleaningOpRecord.op_version must be non-empty for op={self.op.value!r}"
            )
        if self.chars_before < 0 or self.chars_after < 0:
            raise ValueError(
                f"CleaningOpRecord char counts must be >= 0 (op={self.op.value!r}, "
                f"before={self.chars_before}, after={self.chars_after})"
            )
        if self.chars_after > self.chars_before:
            raise ValueError(
                f"CleaningOpRecord.chars_after ({self.chars_after}) must be <= "
                f"chars_before ({self.chars_before}) for op={self.op.value!r} "
                f"(cleaning ops must be monotonically non-expanding)"
            )


@dataclass(frozen=True)
class RawDocument:
    """Output of a parser: extracted text + parser self-report.

    Parsers do not own the cleaning pipeline; they only normalise
    bytes to a typed record so the cleaner can run on plain text with
    quality hints attached.

    Fields:

    * ``text`` — extracted plain text. May contain boilerplate,
      whitespace noise, hyphenated line breaks, etc.; that is the
      cleaner's job.
    * ``parser_version`` — ``"<parser-id>:<int>"``.
    * ``layout_quality`` — parser's 0..1 self-rating of how confident
      it is in structural extraction (page boundaries, paragraph
      detection). Parsers are free to set 1.0 for trivially structured
      inputs (plain text).
    * ``ocr_confidence`` — parser's 0..1 self-rating of OCR quality.
      Non-OCR parsers MUST set 1.0.
    * ``encoding_detected`` — name of the byte encoding the parser
      used (e.g., ``"utf-8"``). For inputs that arrive already as
      Python ``str`` (mostly tests) the parser sets ``"utf-8"``.
    * ``language_detected`` — ISO-639 code; ``""`` if unknown.
    * ``license_notice`` — page-level license text the parser scraped
      out (e.g., Wikisource ``{{PD-old}}`` text, Gutenberg license
      block). Empty string if the parser did not find one.
    * ``raw_sha256`` — sha256 of the original input bytes. The
      content-addressable key.
    """

    text: str
    parser_version: str
    layout_quality: float
    ocr_confidence: float
    encoding_detected: str
    language_detected: str
    license_notice: str
    raw_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError(
                f"RawDocument.text must be non-empty (parser_version={self.parser_version!r}, "
                f"raw_sha256={self.raw_sha256!r})"
            )
        if not isinstance(self.parser_version, str) or ":" not in self.parser_version:
            raise ValueError(
                f"RawDocument.parser_version must be of the form "
                f"'<parser-id>:<int>'; got {self.parser_version!r}"
            )
        for field_name, value in (
            ("layout_quality", self.layout_quality),
            ("ocr_confidence", self.ocr_confidence),
        ):
            if not isinstance(value, (int, float)) or not (0.0 <= float(value) <= 1.0):
                raise ValueError(
                    f"RawDocument.{field_name} must be a float in [0, 1]; got {value!r}"
                )
        if not isinstance(self.encoding_detected, str) or not self.encoding_detected.strip():
            raise ValueError(
                f"RawDocument.encoding_detected must be non-empty; "
                f"got {self.encoding_detected!r}"
            )
        if not isinstance(self.raw_sha256, str) or len(self.raw_sha256) != 64:
            raise ValueError(
                f"RawDocument.raw_sha256 must be a 64-char hex sha256; "
                f"got {self.raw_sha256!r}"
            )


@dataclass(frozen=True)
class CleanedDocument:
    """Output of the cleaner pipeline.

    Bound to its source ``RawDocument`` by ``raw_sha256`` (the
    content-addressable key) and to a specific
    ``cleaner_pipeline_version``. Two ``CleanedDocument`` records with
    the same ``raw_sha256`` but different ``cleaner_pipeline_version``
    are intentionally allowed to coexist: the storage layer
    (:class:`lifeform_domain_figure.cleaning.store.CleaningStore`)
    keeps them in separate ``v{N}/`` directories so a cleaner upgrade
    does not destroy prior versions.

    ``cleaning_log`` is the ordered tuple of ops that fired, with
    char-count deltas. ``parser_version`` is transitively recorded so
    the pipeline path ``parser_version + cleaner_pipeline_version`` is
    a complete reproduction key.
    """

    text: str
    raw_sha256: str
    cleaner_pipeline_version: int
    cleaning_log: tuple[CleaningOpRecord, ...]
    parser_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError(
                f"CleanedDocument.text must be str (raw_sha256={self.raw_sha256!r})"
            )
        if not isinstance(self.raw_sha256, str) or len(self.raw_sha256) != 64:
            raise ValueError(
                f"CleanedDocument.raw_sha256 must be a 64-char hex sha256; "
                f"got {self.raw_sha256!r}"
            )
        if (
            not isinstance(self.cleaner_pipeline_version, int)
            or self.cleaner_pipeline_version <= 0
        ):
            raise ValueError(
                f"CleanedDocument.cleaner_pipeline_version must be a positive int; "
                f"got {self.cleaner_pipeline_version!r}"
            )
        if not isinstance(self.cleaning_log, tuple):
            raise ValueError(
                "CleanedDocument.cleaning_log must be a tuple of CleaningOpRecord"
            )
        for index, record in enumerate(self.cleaning_log):
            if not isinstance(record, CleaningOpRecord):
                raise ValueError(
                    f"CleanedDocument.cleaning_log[{index}] must be CleaningOpRecord; "
                    f"got {type(record).__name__}"
                )
        if not isinstance(self.parser_version, str) or ":" not in self.parser_version:
            raise ValueError(
                f"CleanedDocument.parser_version must be of the form "
                f"'<parser-id>:<int>'; got {self.parser_version!r}"
            )


__all__ = [
    "CleanedDocument",
    "CleaningOp",
    "CleaningOpRecord",
    "RawDocument",
]
