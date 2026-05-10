"""Figure-vertical L2 corpus verification + audit pipeline.

This subpackage implements layer L2 of `docs/known-debts.md` debt #28
(see `docs/specs/figure-corpus-verification.md` for the spec):

    SourceProvenance (anchored on byte_sha256, == L1 raw_sha256)
        -> auto verifier(s)
        -> VerificationCheck records
        -> VerificationLedger.append (data/verification/{sha}/checks.jsonl)
        -> bundle gate (assert_all_provenances_pass) at build_figure_artifact_bundle

Layer L0 (crawler frontier) is a separate follow-up; the 4 deferred
verifier kinds (IDENTITY_DISAMBIGUATION / AUTHORSHIP_ATTRIBUTION /
VERSION_RECONCILIATION / TRANSLATION_LINEAGE) depend on the metadata
client (debt #26) and only have schema + NotImplementedError stubs in
this packet.

Public surface:

* Schema: :class:`CheckKind`, :data:`IMPLEMENTED_CHECK_KINDS`,
  :class:`Verdict`, :class:`VerificationCheck`.
* Storage: :class:`VerificationLedger`.
* Verifiers (first batch): :func:`verify_date_plausibility`,
  :func:`verify_license_page_level`, :func:`verify_cross_source_byte`.
* Verifier registry: :data:`SINGLE_SOURCE_AUTO_VERIFIERS`,
  :data:`MULTI_SOURCE_AUTO_VERIFIERS`, :func:`all_registered_kinds`.
* Bundle gate: :class:`VerificationGateError`,
  :func:`assert_all_provenances_pass`.
"""

from __future__ import annotations

from lifeform_domain_figure.verification.gate import (
    VerificationGateError,
    assert_all_provenances_pass,
)
from lifeform_domain_figure.verification.ledger import VerificationLedger
from lifeform_domain_figure.verification.records import (
    CheckKind,
    IMPLEMENTED_CHECK_KINDS,
    Verdict,
    VerificationCheck,
)
from lifeform_domain_figure.verification.verifiers import (
    MULTI_SOURCE_AUTO_VERIFIERS,
    SINGLE_SOURCE_AUTO_VERIFIERS,
    MetadataDependentVerifierContext,
    MultiSourceVerifier,
    SingleSourceVerifier,
    all_registered_kinds,
    verify_authorship_attribution,
    verify_cross_source_byte,
    verify_date_plausibility,
    verify_identity_disambiguation,
    verify_license_page_level,
    verify_translation_lineage,
    verify_version_reconciliation,
)


__all__ = [
    "CheckKind",
    "IMPLEMENTED_CHECK_KINDS",
    "MULTI_SOURCE_AUTO_VERIFIERS",
    "MetadataDependentVerifierContext",
    "MultiSourceVerifier",
    "SINGLE_SOURCE_AUTO_VERIFIERS",
    "SingleSourceVerifier",
    "Verdict",
    "VerificationCheck",
    "VerificationGateError",
    "VerificationLedger",
    "all_registered_kinds",
    "assert_all_provenances_pass",
    "verify_authorship_attribution",
    "verify_cross_source_byte",
    "verify_date_plausibility",
    "verify_identity_disambiguation",
    "verify_license_page_level",
    "verify_translation_lineage",
    "verify_version_reconciliation",
]
