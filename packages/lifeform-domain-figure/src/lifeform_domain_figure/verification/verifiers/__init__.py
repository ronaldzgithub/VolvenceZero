"""Verifier registry for the L2 verification pipeline.

The registry maps each :class:`CheckKind` to a verifier callable.

* L2 first batch (debt #28 L2 first batch, 2026-05-10):
  ``DATE_PLAUSIBILITY`` / ``LICENSE_PAGE_LEVEL`` / ``CROSS_SOURCE_BYTE``
  — pure-function, no external metadata dependency.
* L2 second batch (debt #28 L2 second batch + debt #26 closure,
  2026-05-10): ``IDENTITY_DISAMBIGUATION`` /
  ``AUTHORSHIP_ATTRIBUTION`` / ``VERSION_RECONCILIATION`` /
  ``TRANSLATION_LINEAGE`` — depend on the V2 metadata clients
  (Wikidata / OpenAlex / Crossref) landed in
  :mod:`lifeform_domain_figure.metadata`.

Because the second-batch verifiers carry extra dependencies
(metadata clients, figure context like expected QID / OpenAlex
author id), they are NOT directly invocable through the simple
``SINGLE_SOURCE_AUTO_VERIFIERS`` mapping — callers wire them up
through :class:`MetadataDependentVerifierContext` (a typed bundle
of clients + curator-supplied figure parameters) and then dispatch
manually.

The simple-mapping verifiers (first batch) remain in
``SINGLE_SOURCE_AUTO_VERIFIERS`` / ``MULTI_SOURCE_AUTO_VERIFIERS``
for batch CLI use.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from lifeform_domain_figure.metadata.crossref import CrossrefClient
from lifeform_domain_figure.metadata.openalex import OpenAlexClient
from lifeform_domain_figure.metadata.wikidata import WikidataClient
from lifeform_domain_figure.verification.records import (
    CheckKind,
    IMPLEMENTED_CHECK_KINDS,
    VerificationCheck,
)
from lifeform_domain_figure.verification.verifiers.authorship_attribution import (
    verify_authorship_attribution,
)
from lifeform_domain_figure.verification.verifiers.cross_source_byte import (
    verify_cross_source_byte,
)
from lifeform_domain_figure.verification.verifiers.date_plausibility import (
    verify_date_plausibility,
)
from lifeform_domain_figure.verification.verifiers.identity_disambiguation import (
    verify_identity_disambiguation,
)
from lifeform_domain_figure.verification.verifiers.license_page_level import (
    verify_license_page_level,
)
from lifeform_domain_figure.verification.verifiers.translation_lineage import (
    verify_translation_lineage,
)
from lifeform_domain_figure.verification.verifiers.version_reconciliation import (
    verify_version_reconciliation,
)


SingleSourceVerifier = Callable[..., VerificationCheck]
MultiSourceVerifier = Callable[..., tuple[VerificationCheck, ...]]


SINGLE_SOURCE_AUTO_VERIFIERS: dict[CheckKind, SingleSourceVerifier] = {
    CheckKind.DATE_PLAUSIBILITY: verify_date_plausibility,
    CheckKind.LICENSE_PAGE_LEVEL: verify_license_page_level,
    CheckKind.IDENTITY_DISAMBIGUATION: verify_identity_disambiguation,
    CheckKind.AUTHORSHIP_ATTRIBUTION: verify_authorship_attribution,
    CheckKind.TRANSLATION_LINEAGE: verify_translation_lineage,
    CheckKind.VERSION_RECONCILIATION: verify_version_reconciliation,
}


MULTI_SOURCE_AUTO_VERIFIERS: dict[CheckKind, MultiSourceVerifier] = {
    CheckKind.CROSS_SOURCE_BYTE: verify_cross_source_byte,
}


@dataclass(frozen=True)
class MetadataDependentVerifierContext:
    """Typed bundle of metadata clients + curator-supplied figure context.

    Constructed once per figure (per CLI run) and passed to the
    second-batch verifiers when the batch driver wires them up.
    Tests use partial contexts (e.g., only the Wikidata client +
    expected_qid) to exercise individual verifiers without standing
    up all four clients.

    Fields are deliberately optional so a partial context (e.g.,
    only the verifiers that need Wikidata) constructs without
    NotImplementedError stubs for unused clients.
    """

    wikidata_client: WikidataClient | None = None
    openalex_client: OpenAlexClient | None = None
    crossref_client: CrossrefClient | None = None
    expected_qid: str = ""
    expected_birth_year: int | None = None
    expected_occupations: tuple[str, ...] = ()
    expected_openalex_author_id: str = ""
    coauthor_anchor_works: tuple[str, ...] = ()
    figure_native_languages: tuple[str, ...] = ()


def all_registered_kinds() -> frozenset[CheckKind]:
    """Return every :class:`CheckKind` that has a registry entry.

    Should always equal the full :class:`CheckKind` enum.
    """

    return frozenset(SINGLE_SOURCE_AUTO_VERIFIERS) | frozenset(
        MULTI_SOURCE_AUTO_VERIFIERS
    )


__all__ = [
    "IMPLEMENTED_CHECK_KINDS",
    "MULTI_SOURCE_AUTO_VERIFIERS",
    "MetadataDependentVerifierContext",
    "MultiSourceVerifier",
    "SINGLE_SOURCE_AUTO_VERIFIERS",
    "SingleSourceVerifier",
    "all_registered_kinds",
    "verify_authorship_attribution",
    "verify_cross_source_byte",
    "verify_date_plausibility",
    "verify_identity_disambiguation",
    "verify_license_page_level",
    "verify_translation_lineage",
    "verify_version_reconciliation",
]
