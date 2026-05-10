"""Cross-cutting contract: bundle gate refuses non-verified sources.

This test exercises the **end-to-end gate behaviour** that debt #28
L2 promises:

1. Default (gate off) build is a pure pass-through of L1-era inputs;
   no behavioural change for any pre-#28 caller.
2. With ``require_verification_pass=True`` and a complete all-PASS
   ledger, the gate admits the bundle.
3. With ``require_verification_pass=True`` but missing or non-PASS
   ledger entries, the gate raises :class:`VerificationGateError`
   and the message is informative enough for an operator to act
   (carries source_id + check_kind + reason).

Coverage scope: this is the contract test that surfaces gaps when a
new :class:`CheckKind` migrates from "deferred stub" to "implemented"
without updating gate coverage. Adding such a kind to
:data:`IMPLEMENTED_CHECK_KINDS` immediately starts requiring a
ledger entry for it; existing PASS-only ledgers will then surface a
``missing-check`` failure here, forcing the follow-up packet to add
appropriate coverage.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lifeform_domain_figure.compiler import (
    FigureBundleInputs,
    build_figure_artifact_bundle,
)
from lifeform_domain_figure.corpus.ingest_papers import ingest_papers
from lifeform_domain_figure.corpus.provenance import (
    CaptureMethod,
    LegalClearance,
    SourceProvenance,
)
from lifeform_domain_figure.profiles.einstein import build_einstein_profile
from lifeform_domain_figure.sample_corpus import synthetic_einstein_corpus
from lifeform_domain_figure.verification import (
    CheckKind,
    IMPLEMENTED_CHECK_KINDS,
    Verdict,
    VerificationCheck,
    VerificationGateError,
    VerificationLedger,
)


_REVIEWED_AT = "2026-05-10T12:00:00+00:00"
_REVIEWER_BY_KIND = {
    CheckKind.DATE_PLAUSIBILITY: "auto:date_plausibility:1",
    CheckKind.LICENSE_PAGE_LEVEL: "auto:license_page_level:1",
    CheckKind.CROSS_SOURCE_BYTE: "auto:cross_source_byte:1",
    CheckKind.IDENTITY_DISAMBIGUATION: "auto:identity_disambiguation:1",
    CheckKind.AUTHORSHIP_ATTRIBUTION: "auto:authorship_attribution:1",
    CheckKind.VERSION_RECONCILIATION: "auto:version_reconciliation:1",
    CheckKind.TRANSLATION_LINEAGE: "auto:translation_lineage:1",
}


def _build_inputs(
    *,
    provenance_records: tuple[SourceProvenance, ...],
    ledger: VerificationLedger | None,
    require: bool,
) -> FigureBundleInputs:
    papers, _letters, _lectures, _notebooks = synthetic_einstein_corpus()
    envelope = ingest_papers(
        (papers[0],), uploader="contract-test", upload_ts_ms=1_700_000_000_000
    )
    return FigureBundleInputs(
        profile=build_einstein_profile(),
        envelopes=(envelope,),
        provenance_records=provenance_records,
        verification_ledger=ledger,
        require_verification_pass=require,
    )


def _make_provenance(*, sha: str = "d" * 64, source_id: str = "test-paper-1") -> SourceProvenance:
    return SourceProvenance(
        source_id=source_id,
        figure_id="einstein",
        source_url="https://example/einstein/x",
        license_label="public domain",
        legal_clearance=LegalClearance.PUBLIC_DOMAIN_GLOBAL,
        capture_method=CaptureMethod.TRANSCRIBED,
        captured_by="contract-test",
        captured_at_iso=_REVIEWED_AT,
        byte_sha256=sha,
        provenance_note="contract test fixture",
    )


def _populate_ledger_for_implemented_kinds(
    sha: str, ledger: VerificationLedger
) -> None:
    for kind in IMPLEMENTED_CHECK_KINDS:
        ledger.append(
            VerificationCheck(
                check_kind=kind,
                verdict=Verdict.PASS,
                evidence=("contract test PASS",),
                reviewer_id=_REVIEWER_BY_KIND[kind],
                reviewed_at_iso=_REVIEWED_AT,
                source_byte_sha256=sha,
            )
        )


def test_default_off_zero_regression() -> None:
    inputs = _build_inputs(provenance_records=(), ledger=None, require=False)
    bundle = build_figure_artifact_bundle(inputs)
    assert bundle.figure_id == inputs.profile.profile_id


def test_gate_on_full_pass_admits_bundle(tmp_path: Path) -> None:
    ledger = VerificationLedger(tmp_path)
    sha = "e" * 64
    provenance = _make_provenance(sha=sha)
    _populate_ledger_for_implemented_kinds(sha, ledger)
    inputs = _build_inputs(
        provenance_records=(provenance,), ledger=ledger, require=True
    )
    bundle = build_figure_artifact_bundle(inputs)
    assert bundle.figure_id == inputs.profile.profile_id


def test_gate_on_one_missing_kind_blocks(tmp_path: Path) -> None:
    ledger = VerificationLedger(tmp_path)
    sha = "f" * 64
    provenance = _make_provenance(sha=sha)
    ledger.append(
        VerificationCheck(
            check_kind=CheckKind.DATE_PLAUSIBILITY,
            verdict=Verdict.PASS,
            evidence=("only date check",),
            reviewer_id=_REVIEWER_BY_KIND[CheckKind.DATE_PLAUSIBILITY],
            reviewed_at_iso=_REVIEWED_AT,
            source_byte_sha256=sha,
        )
    )
    inputs = _build_inputs(
        provenance_records=(provenance,), ledger=ledger, require=True
    )
    with pytest.raises(VerificationGateError) as exc:
        build_figure_artifact_bundle(inputs)
    msg = str(exc.value)
    assert "missing-check" in msg
    assert "license_page_level" in msg
    assert "cross_source_byte" in msg


def test_gate_on_one_failure_blocks_with_source_id(tmp_path: Path) -> None:
    ledger = VerificationLedger(tmp_path)
    sha = "0a" + "0" * 62
    provenance = _make_provenance(sha=sha, source_id="failing-source-1")
    _populate_ledger_for_implemented_kinds(sha, ledger)
    ledger.append(
        VerificationCheck(
            check_kind=CheckKind.LICENSE_PAGE_LEVEL,
            verdict=Verdict.FAIL,
            evidence=("hard-conflict",),
            reviewer_id=_REVIEWER_BY_KIND[CheckKind.LICENSE_PAGE_LEVEL],
            reviewed_at_iso="2026-05-10T13:00:00+00:00",
            source_byte_sha256=sha,
        )
    )
    inputs = _build_inputs(
        provenance_records=(provenance,), ledger=ledger, require=True
    )
    with pytest.raises(VerificationGateError) as exc:
        build_figure_artifact_bundle(inputs)
    msg = str(exc.value)
    assert "failing-source-1" in msg
    assert "license_page_level" in msg


def test_implemented_check_kinds_subset_of_check_kind() -> None:
    """If a CheckKind is added to IMPLEMENTED, it must already exist in CheckKind.

    Reciprocally, every CheckKind has a verifier registry entry (no
    silent enum-without-implementation drift). The
    ``all_registered_kinds()`` helper enforces this.
    """

    from lifeform_domain_figure.verification import all_registered_kinds

    all_kinds = frozenset(CheckKind)
    assert IMPLEMENTED_CHECK_KINDS.issubset(all_kinds)
    assert all_registered_kinds() == all_kinds


def test_all_seven_kinds_have_real_verifiers() -> None:
    """As of debt #28 L2 second batch + debt #26 closure (2026-05-10),
    every CheckKind has a real verifier registered.

    Previously deferred kinds (IDENTITY_DISAMBIGUATION /
    AUTHORSHIP_ATTRIBUTION / VERSION_RECONCILIATION /
    TRANSLATION_LINEAGE) now resolve to live-metadata-client-backed
    callables (in :mod:`lifeform_domain_figure.verification.verifiers`).
    The contract here is structural: the registry must expose every
    enum member, and callable signatures must require typed arguments
    (so missing-arg invocations raise TypeError, never silently pass).
    """

    from lifeform_domain_figure.verification import (
        SINGLE_SOURCE_AUTO_VERIFIERS,
        MULTI_SOURCE_AUTO_VERIFIERS,
        all_registered_kinds,
    )

    assert all_registered_kinds() == frozenset(CheckKind)
    metadata_dependent_singles = (
        CheckKind.IDENTITY_DISAMBIGUATION,
        CheckKind.AUTHORSHIP_ATTRIBUTION,
        CheckKind.TRANSLATION_LINEAGE,
        CheckKind.VERSION_RECONCILIATION,
    )
    for kind in metadata_dependent_singles:
        verifier = SINGLE_SOURCE_AUTO_VERIFIERS[kind]
        with pytest.raises(TypeError):
            verifier()
    multi_kind_verifier = MULTI_SOURCE_AUTO_VERIFIERS[CheckKind.CROSS_SOURCE_BYTE]
    with pytest.raises(ValueError):
        multi_kind_verifier((), document_group_key="x")
