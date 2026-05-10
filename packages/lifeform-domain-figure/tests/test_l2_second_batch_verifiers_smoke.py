"""Smoke tests for the 4 L2 second-batch metadata-dependent verifiers."""

from __future__ import annotations

from lifeform_domain_figure.corpus.provenance import (
    CaptureMethod,
    LegalClearance,
    SourceProvenance,
)
from lifeform_domain_figure.metadata.openalex import OpenAlexWorkPayload
from lifeform_domain_figure.metadata.wikidata import WikidataPersonPayload
from lifeform_domain_figure.verification.records import CheckKind, Verdict
from lifeform_domain_figure.verification.verifiers.authorship_attribution import (
    verify_authorship_attribution,
)
from lifeform_domain_figure.verification.verifiers.identity_disambiguation import (
    verify_identity_disambiguation,
)
from lifeform_domain_figure.verification.verifiers.translation_lineage import (
    verify_translation_lineage,
)
from lifeform_domain_figure.verification.verifiers.version_reconciliation import (
    verify_version_reconciliation,
)


_TS = "2026-05-10T12:00:00+00:00"
_SHA = "a" * 64


def _provenance(*, source_id: str = "src-1") -> SourceProvenance:
    return SourceProvenance(
        source_id=source_id,
        figure_id="einstein",
        source_url="https://example/x",
        license_label="public domain",
        legal_clearance=LegalClearance.PUBLIC_DOMAIN_GLOBAL,
        capture_method=CaptureMethod.TRANSCRIBED,
        captured_by="reviewer-x",
        captured_at_iso=_TS,
        byte_sha256=_SHA,
        provenance_note="test fixture",
    )


# ---- IDENTITY_DISAMBIGUATION -----------------------------------------------


class _StubWikidataClient:
    def __init__(self, person: WikidataPersonPayload | Exception) -> None:
        self._person = person

    def fetch_person(self, *, qid: str) -> WikidataPersonPayload:
        if isinstance(self._person, Exception):
            raise self._person
        return self._person


def test_identity_pass_when_birth_year_matches_and_occupations_overlap() -> None:
    client = _StubWikidataClient(
        WikidataPersonPayload(
            qid="Q937",
            label="Albert Einstein",
            birth_year=1879,
            death_year=1955,
            occupation_labels=("physicist", "professor"),
        )
    )
    check = verify_identity_disambiguation(
        _provenance(),
        wikidata_client=client,
        expected_qid="Q937",
        expected_birth_year=1879,
        expected_occupations=("physicist",),
        now_iso=_TS,
    )
    assert check.check_kind is CheckKind.IDENTITY_DISAMBIGUATION
    assert check.verdict is Verdict.PASS


def test_identity_fail_on_birth_year_mismatch() -> None:
    client = _StubWikidataClient(
        WikidataPersonPayload(qid="Q937", label="Other", birth_year=1900, death_year=1970)
    )
    check = verify_identity_disambiguation(
        _provenance(),
        wikidata_client=client,
        expected_qid="Q937",
        expected_birth_year=1879,
        now_iso=_TS,
    )
    assert check.verdict is Verdict.FAIL
    assert any("diff=21" in e for e in check.evidence)


def test_identity_needs_review_on_zero_occupation_overlap() -> None:
    client = _StubWikidataClient(
        WikidataPersonPayload(
            qid="Q937", label="X", birth_year=1879, death_year=1955,
            occupation_labels=("politician",),
        )
    )
    check = verify_identity_disambiguation(
        _provenance(),
        wikidata_client=client,
        expected_qid="Q937",
        expected_birth_year=1879,
        expected_occupations=("physicist",),
        now_iso=_TS,
    )
    assert check.verdict is Verdict.NEEDS_REVIEW


def test_identity_needs_review_when_fetch_fails() -> None:
    client = _StubWikidataClient(RuntimeError("network down"))
    check = verify_identity_disambiguation(
        _provenance(),
        wikidata_client=client,
        expected_qid="Q937",
        expected_birth_year=1879,
        now_iso=_TS,
    )
    assert check.verdict is Verdict.NEEDS_REVIEW
    assert any("network down" in e for e in check.evidence)


# ---- AUTHORSHIP_ATTRIBUTION ------------------------------------------------


class _StubOpenAlexClient:
    def __init__(self, works: tuple[OpenAlexWorkPayload, ...]) -> None:
        self._works = works

    def fetch_author_works(
        self, *, openalex_author_id: str
    ) -> tuple[OpenAlexWorkPayload, ...]:
        return self._works


def _work(work_id: str = "W1") -> OpenAlexWorkPayload:
    return OpenAlexWorkPayload(
        openalex_id=work_id,
        title="t",
        publication_year=1905,
        venue="v",
        language="de",
        concept_labels=(),
    )


def test_authorship_pass_on_direct_match() -> None:
    client = _StubOpenAlexClient((_work("W1"), _work("W2")))
    check = verify_authorship_attribution(
        _provenance(),
        openalex_client=client,
        expected_openalex_author_id="A1",
        candidate_work_id="W1",
        now_iso=_TS,
    )
    assert check.check_kind is CheckKind.AUTHORSHIP_ATTRIBUTION
    assert check.verdict is Verdict.PASS


def test_authorship_fail_when_no_match_no_overlap() -> None:
    client = _StubOpenAlexClient((_work("W1"),))
    check = verify_authorship_attribution(
        _provenance(),
        openalex_client=client,
        expected_openalex_author_id="A1",
        candidate_work_id="W99",
        now_iso=_TS,
    )
    assert check.verdict is Verdict.FAIL


def test_authorship_needs_review_with_coauthor_overlap_data() -> None:
    client = _StubOpenAlexClient((_work("W1"), _work("W2")))
    check = verify_authorship_attribution(
        _provenance(),
        openalex_client=client,
        expected_openalex_author_id="A1",
        candidate_work_id="W99",
        coauthor_anchor_works=("W1",),
        candidate_coauthor_openalex_ids=("A2",),
        now_iso=_TS,
    )
    assert check.verdict is Verdict.NEEDS_REVIEW


# ---- VERSION_RECONCILIATION ------------------------------------------------


class _StubCrossrefClient:
    def __init__(self, message: dict) -> None:
        self._message = message

    def fetch_work(self, *, doi: str):
        raise NotImplementedError("test stub uses fetch_raw_message only")

    def fetch_raw_message(self, *, doi: str) -> dict:
        return self._message


def test_version_pass_when_no_relations() -> None:
    client = _StubCrossrefClient({"DOI": "10.1/x", "title": ["t"]})
    check = verify_version_reconciliation(
        _provenance(),
        crossref_client=client,
        source_doi="10.1/x",
        now_iso=_TS,
    )
    assert check.check_kind is CheckKind.VERSION_RECONCILIATION
    assert check.verdict is Verdict.PASS


def test_version_needs_review_on_multiple_versions() -> None:
    client = _StubCrossrefClient(
        {
            "DOI": "10.1/x",
            "relation": {
                "is-version-of": [{"id": "10.5/preprint"}],
                "replaces": [{"DOI": "10.6/old"}],
            },
        }
    )
    check = verify_version_reconciliation(
        _provenance(),
        crossref_client=client,
        source_doi="10.1/x",
        now_iso=_TS,
    )
    assert check.verdict is Verdict.NEEDS_REVIEW


def test_version_pass_when_canonical_hint_matches() -> None:
    client = _StubCrossrefClient(
        {
            "DOI": "10.1/canonical",
            "relation": {"is-version-of": [{"id": "10.5/preprint"}]},
        }
    )
    check = verify_version_reconciliation(
        _provenance(),
        crossref_client=client,
        source_doi="10.1/canonical",
        canonical_doi_hint="10.1/canonical",
        now_iso=_TS,
    )
    assert check.verdict is Verdict.PASS


# ---- TRANSLATION_LINEAGE ---------------------------------------------------


def test_translation_pass_when_lang_differs_and_translator_present() -> None:
    client = _StubCrossrefClient(
        {
            "DOI": "10.1/translated",
            "translator": [{"given": "Anna", "family": "Smith"}],
        }
    )
    check = verify_translation_lineage(
        _provenance(),
        crossref_client=client,
        source_doi="10.1/translated",
        source_language="en",
        figure_native_languages=("de",),
        now_iso=_TS,
    )
    assert check.check_kind is CheckKind.TRANSLATION_LINEAGE
    assert check.verdict is Verdict.PASS


def test_translation_needs_review_when_lang_differs_no_translator() -> None:
    client = _StubCrossrefClient({"DOI": "10.1/x"})
    check = verify_translation_lineage(
        _provenance(),
        crossref_client=client,
        source_doi="10.1/x",
        source_language="en",
        figure_native_languages=("de",),
        now_iso=_TS,
    )
    assert check.verdict is Verdict.NEEDS_REVIEW


def test_translation_pass_when_lang_matches_no_translator() -> None:
    client = _StubCrossrefClient({"DOI": "10.1/x"})
    check = verify_translation_lineage(
        _provenance(),
        crossref_client=client,
        source_doi="10.1/x",
        source_language="de",
        figure_native_languages=("de",),
        now_iso=_TS,
    )
    assert check.verdict is Verdict.PASS


def test_translation_needs_review_on_lang_match_with_translator() -> None:
    client = _StubCrossrefClient(
        {
            "DOI": "10.1/x",
            "translator": [{"given": "Bob", "family": "Jones"}],
        }
    )
    check = verify_translation_lineage(
        _provenance(),
        crossref_client=client,
        source_doi="10.1/x",
        source_language="de",
        figure_native_languages=("de",),
        now_iso=_TS,
    )
    assert check.verdict is Verdict.NEEDS_REVIEW
