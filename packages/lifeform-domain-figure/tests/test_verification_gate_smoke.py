"""Smoke tests for the bundle verification gate (debt #28 L2)."""

from __future__ import annotations

from datetime import datetime, timezone
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
from lifeform_domain_figure.verification.gate import VerificationGateError
from lifeform_domain_figure.verification.ledger import VerificationLedger
from lifeform_domain_figure.verification.records import (
    CheckKind,
    Verdict,
    VerificationCheck,
)


_REVIEWED_AT = "2026-05-10T12:00:00+00:00"


def _build_envelope_and_provenance() -> tuple:
    papers, _letters, _lectures, _notebooks = synthetic_einstein_corpus()
    paper = papers[0]
    envelope = ingest_papers(
        (paper,), uploader="test-reviewer", upload_ts_ms=1_700_000_000_000
    )
    paper_byte_sha = "c" * 64
    provenance = SourceProvenance(
        source_id=paper.paper_id,
        figure_id="einstein",
        source_url="https://example/einstein/paper-1",
        license_label="public domain",
        legal_clearance=LegalClearance.PUBLIC_DOMAIN_GLOBAL,
        capture_method=CaptureMethod.TRANSCRIBED,
        captured_by="reviewer-x",
        captured_at_iso=_REVIEWED_AT,
        byte_sha256=paper_byte_sha,
        provenance_note="test fixture for L2 gate",
    )
    return envelope, provenance, paper_byte_sha


def _all_pass_for(sha: str, ledger: VerificationLedger) -> None:
    for kind in CheckKind:
        ledger.append(
            VerificationCheck(
                check_kind=kind,
                verdict=Verdict.PASS,
                evidence=("synthetic test PASS",),
                reviewer_id=f"auto:{kind.value}:1",
                reviewed_at_iso=_REVIEWED_AT,
                source_byte_sha256=sha,
            )
        )


def test_default_gate_off_zero_regression() -> None:
    envelope, _provenance, _sha = _build_envelope_and_provenance()
    profile = build_einstein_profile()
    inputs = FigureBundleInputs(profile=profile, envelopes=(envelope,))
    bundle = build_figure_artifact_bundle(inputs)
    assert bundle.figure_id == profile.profile_id


def test_gate_on_with_full_pass_succeeds(tmp_path: Path) -> None:
    envelope, provenance, sha = _build_envelope_and_provenance()
    ledger = VerificationLedger(tmp_path)
    _all_pass_for(sha, ledger)
    inputs = FigureBundleInputs(
        profile=build_einstein_profile(),
        envelopes=(envelope,),
        provenance_records=(provenance,),
        verification_ledger=ledger,
        require_verification_pass=True,
    )
    bundle = build_figure_artifact_bundle(inputs)
    assert bundle.figure_id == inputs.profile.profile_id


def test_gate_on_missing_ledger_raises() -> None:
    envelope, provenance, _sha = _build_envelope_and_provenance()
    inputs = FigureBundleInputs(
        profile=build_einstein_profile(),
        envelopes=(envelope,),
        provenance_records=(provenance,),
        verification_ledger=None,
        require_verification_pass=True,
    )
    with pytest.raises(VerificationGateError, match="verification_ledger is None"):
        build_figure_artifact_bundle(inputs)


def test_gate_on_empty_provenance_raises(tmp_path: Path) -> None:
    envelope, _provenance, _sha = _build_envelope_and_provenance()
    ledger = VerificationLedger(tmp_path)
    inputs = FigureBundleInputs(
        profile=build_einstein_profile(),
        envelopes=(envelope,),
        provenance_records=(),
        verification_ledger=ledger,
        require_verification_pass=True,
    )
    with pytest.raises(VerificationGateError, match="provenance_records is empty"):
        build_figure_artifact_bundle(inputs)


def test_gate_on_missing_check_raises(tmp_path: Path) -> None:
    envelope, provenance, sha = _build_envelope_and_provenance()
    ledger = VerificationLedger(tmp_path)
    ledger.append(
        VerificationCheck(
            check_kind=CheckKind.DATE_PLAUSIBILITY,
            verdict=Verdict.PASS,
            evidence=("only date check",),
            reviewer_id="auto:date_plausibility:1",
            reviewed_at_iso=_REVIEWED_AT,
            source_byte_sha256=sha,
        )
    )
    inputs = FigureBundleInputs(
        profile=build_einstein_profile(),
        envelopes=(envelope,),
        provenance_records=(provenance,),
        verification_ledger=ledger,
        require_verification_pass=True,
    )
    with pytest.raises(VerificationGateError) as exc:
        build_figure_artifact_bundle(inputs)
    assert "missing-check" in str(exc.value)
    assert "license_page_level" in str(exc.value)


def test_gate_on_one_fail_raises_with_source_id(tmp_path: Path) -> None:
    envelope, provenance, sha = _build_envelope_and_provenance()
    ledger = VerificationLedger(tmp_path)
    _all_pass_for(sha, ledger)
    ledger.append(
        VerificationCheck(
            check_kind=CheckKind.LICENSE_PAGE_LEVEL,
            verdict=Verdict.FAIL,
            evidence=("hard-conflict found",),
            reviewer_id="auto:license_page_level:1",
            reviewed_at_iso="2026-05-10T13:00:00+00:00",
            source_byte_sha256=sha,
        )
    )
    inputs = FigureBundleInputs(
        profile=build_einstein_profile(),
        envelopes=(envelope,),
        provenance_records=(provenance,),
        verification_ledger=ledger,
        require_verification_pass=True,
    )
    with pytest.raises(VerificationGateError) as exc:
        build_figure_artifact_bundle(inputs)
    msg = str(exc.value)
    assert provenance.source_id in msg
    assert "license_page_level" in msg
    assert "verdict=fail" in msg


def test_human_override_promotes_fail_to_pass(tmp_path: Path) -> None:
    envelope, provenance, sha = _build_envelope_and_provenance()
    ledger = VerificationLedger(tmp_path)
    _all_pass_for(sha, ledger)
    ledger.append(
        VerificationCheck(
            check_kind=CheckKind.LICENSE_PAGE_LEVEL,
            verdict=Verdict.FAIL,
            evidence=("auto false positive",),
            reviewer_id="auto:license_page_level:1",
            reviewed_at_iso="2026-05-10T13:00:00+00:00",
            source_byte_sha256=sha,
        )
    )
    later_iso = datetime.now(timezone.utc).isoformat()
    ledger.append(
        VerificationCheck(
            check_kind=CheckKind.LICENSE_PAGE_LEVEL,
            verdict=Verdict.PASS,
            evidence=("reviewer manually approved",),
            reviewer_id="human:reviewer-x",
            reviewed_at_iso=later_iso,
            source_byte_sha256=sha,
        )
    )
    inputs = FigureBundleInputs(
        profile=build_einstein_profile(),
        envelopes=(envelope,),
        provenance_records=(provenance,),
        verification_ledger=ledger,
        require_verification_pass=True,
    )
    bundle = build_figure_artifact_bundle(inputs)
    assert bundle.figure_id == inputs.profile.profile_id
