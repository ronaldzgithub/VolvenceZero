"""Curation flow: assemble + freeze the immutable ``FigureCorpusBundle``.

Two responsibilities:

* :mod:`lifeform_domain_figure.curation.freeze` — pure builder that takes
  reviewed inputs and produces a sha256-addressed bundle. No I/O.
* :mod:`lifeform_domain_figure.curation.reviewer` (D6) — workflow object
  that enforces ``reviewed_by`` provenance and per-source license declared
  before bundle freeze.

Both stay schema-only / pure: this directory does **not** issue HTTP
requests, write files, or touch kernel state.
"""

from __future__ import annotations

from lifeform_domain_figure.curation.freeze import (
    BuildBundleInputs,
    bundle_id_from_hash,
    build_figure_corpus_bundle,
    canonical_serialize,
    compute_corpus_bundle_integrity_hash,
    primary_source_from_envelope_chunk,
    utc_iso_now,
)


__all__ = [
    "BuildBundleInputs",
    "bundle_id_from_hash",
    "build_figure_corpus_bundle",
    "canonical_serialize",
    "compute_corpus_bundle_integrity_hash",
    "primary_source_from_envelope_chunk",
    "utc_iso_now",
]
