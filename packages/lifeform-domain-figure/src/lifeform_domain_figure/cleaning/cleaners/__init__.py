"""Cleaner pipeline orchestrator.

The cleaner pipeline is a versioned ordered tuple of cleaning ops:

* ``CLEANER_PIPELINE_V1`` — the launch composition (boilerplate →
  whitespace → typography → dedupe → pii → paragraph). Bump
  :data:`CURRENT_CLEANER_PIPELINE_VERSION` and add a new
  ``CLEANER_PIPELINE_VN`` constant when the composition or any
  individual op's behaviour changes; never mutate an existing pipeline
  composition.

The orchestrator function :func:`clean_raw_document` runs the pipeline
matching the requested ``pipeline_version`` and returns a
:class:`CleanedDocument` with the per-op
:class:`CleaningOpRecord` log filled in. ``cleaner_for(version)``
exposes the same lookup so the storage layer can record which op
versions ran.

Per-op character invariants:

* Each op MUST be monotonically non-expanding. The orchestrator
  raises if an op increases the text length (the schema's
  ``CleaningOpRecord.__post_init__`` enforces this on each record).
* The orchestrator records ``chars_before`` / ``chars_after`` per op,
  not just the pipeline-level delta, so single-op regressions are
  visible in the cleaning log.
"""

from __future__ import annotations

from collections.abc import Callable

from lifeform_domain_figure.cleaning.cleaners.boilerplate import (
    OP_VERSION as BOILERPLATE_OP_VERSION,
)
from lifeform_domain_figure.cleaning.cleaners.boilerplate import strip_boilerplate
from lifeform_domain_figure.cleaning.cleaners.dedupe import (
    OP_VERSION as DEDUPE_OP_VERSION,
)
from lifeform_domain_figure.cleaning.cleaners.dedupe import dedupe_paragraphs
from lifeform_domain_figure.cleaning.cleaners.paragraph import (
    OP_VERSION as PARAGRAPH_OP_VERSION,
)
from lifeform_domain_figure.cleaning.cleaners.paragraph import normalise_paragraphs
from lifeform_domain_figure.cleaning.cleaners.pii import (
    OP_VERSION as PII_OP_VERSION,
)
from lifeform_domain_figure.cleaning.cleaners.pii import redact_pii
from lifeform_domain_figure.cleaning.cleaners.typography import (
    OP_VERSION as TYPOGRAPHY_OP_VERSION,
)
from lifeform_domain_figure.cleaning.cleaners.typography import normalise_typography
from lifeform_domain_figure.cleaning.cleaners.whitespace import (
    OP_VERSION as WHITESPACE_OP_VERSION,
)
from lifeform_domain_figure.cleaning.cleaners.whitespace import normalise_whitespace
from lifeform_domain_figure.cleaning.raw_document import (
    CleanedDocument,
    CleaningOp,
    CleaningOpRecord,
    RawDocument,
)


CleanerStep = tuple[CleaningOp, str, Callable[[str], str]]

CLEANER_PIPELINE_V1: tuple[CleanerStep, ...] = (
    (CleaningOp.BOILERPLATE_STRIP, BOILERPLATE_OP_VERSION, strip_boilerplate),
    (CleaningOp.WHITESPACE_NORMALIZE, WHITESPACE_OP_VERSION, normalise_whitespace),
    (CleaningOp.TYPOGRAPHY_NORMALIZE, TYPOGRAPHY_OP_VERSION, normalise_typography),
    (CleaningOp.DEDUPE_INTRA_DOC, DEDUPE_OP_VERSION, dedupe_paragraphs),
    (CleaningOp.PII_REDACT, PII_OP_VERSION, redact_pii),
    (CleaningOp.PARAGRAPH_NORMALIZE, PARAGRAPH_OP_VERSION, normalise_paragraphs),
)


_PIPELINES: dict[int, tuple[CleanerStep, ...]] = {
    1: CLEANER_PIPELINE_V1,
}

CURRENT_CLEANER_PIPELINE_VERSION = 1


def cleaner_for(pipeline_version: int) -> tuple[CleanerStep, ...]:
    """Return the ordered cleaner steps for ``pipeline_version``.

    Raises ``ValueError`` for unregistered versions; callers are
    expected to pin against :data:`CURRENT_CLEANER_PIPELINE_VERSION`
    or an explicitly chosen historical version.
    """

    try:
        return _PIPELINES[pipeline_version]
    except KeyError as exc:
        raise ValueError(
            f"cleaner_for: unknown pipeline_version={pipeline_version!r}; "
            f"registered={sorted(_PIPELINES)!r}"
        ) from exc


def clean_raw_document(
    raw: RawDocument,
    *,
    pipeline_version: int = CURRENT_CLEANER_PIPELINE_VERSION,
) -> CleanedDocument:
    """Run the cleaner pipeline against ``raw`` and return a CleanedDocument.

    The op record list is captured in pipeline order; even no-op steps
    (``chars_before == chars_after``) are recorded so the audit trail
    is complete.
    """

    steps = cleaner_for(pipeline_version)
    text = raw.text
    log: list[CleaningOpRecord] = []
    for op, op_version, fn in steps:
        before = len(text)
        text = fn(text)
        after = len(text)
        log.append(
            CleaningOpRecord(
                op=op,
                op_version=op_version,
                chars_before=before,
                chars_after=after,
            )
        )
    return CleanedDocument(
        text=text,
        raw_sha256=raw.raw_sha256,
        cleaner_pipeline_version=pipeline_version,
        cleaning_log=tuple(log),
        parser_version=raw.parser_version,
    )


__all__ = [
    "CLEANER_PIPELINE_V1",
    "CURRENT_CLEANER_PIPELINE_VERSION",
    "CleanerStep",
    "clean_raw_document",
    "cleaner_for",
]
