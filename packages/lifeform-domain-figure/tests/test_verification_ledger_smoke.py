"""Smoke tests for L2 VerificationLedger persistence (debt #28)."""

from __future__ import annotations

import json
from pathlib import Path

from lifeform_domain_figure.verification.ledger import VerificationLedger
from lifeform_domain_figure.verification.records import (
    CheckKind,
    Verdict,
    VerificationCheck,
)


_SHA_A = "a" * 64
_SHA_B = "b" * 64


def _make_check(
    *,
    kind: CheckKind = CheckKind.DATE_PLAUSIBILITY,
    verdict: Verdict = Verdict.PASS,
    reviewer: str = "auto:date_plausibility:1",
    reviewed_at: str = "2026-05-10T12:00:00+00:00",
    sha: str = _SHA_A,
    evidence: tuple[str, ...] = ("default evidence",),
) -> VerificationCheck:
    return VerificationCheck(
        check_kind=kind,
        verdict=verdict,
        evidence=evidence,
        reviewer_id=reviewer,
        reviewed_at_iso=reviewed_at,
        source_byte_sha256=sha,
    )


def test_append_creates_anchor_dir_and_jsonl(tmp_path: Path) -> None:
    ledger = VerificationLedger(tmp_path)
    check = _make_check()
    path = ledger.append(check)
    assert path == tmp_path / "verification" / _SHA_A / "checks.jsonl"
    assert path.exists()
    line = path.read_text(encoding="utf-8").strip()
    payload = json.loads(line)
    assert payload["check_kind"] == "date_plausibility"
    assert payload["verdict"] == "pass"
    assert payload["source_byte_sha256"] == _SHA_A


def test_get_checks_round_trip(tmp_path: Path) -> None:
    ledger = VerificationLedger(tmp_path)
    first = _make_check(verdict=Verdict.PASS, reviewed_at="2026-05-10T01:00:00+00:00")
    second = _make_check(
        verdict=Verdict.FAIL,
        reviewed_at="2026-05-10T02:00:00+00:00",
        evidence=("override",),
    )
    ledger.append(first)
    ledger.append(second)
    fetched = ledger.get_checks(_SHA_A)
    assert len(fetched) == 2
    assert fetched[0] == first
    assert fetched[1] == second


def test_get_checks_missing_anchor_returns_empty(tmp_path: Path) -> None:
    ledger = VerificationLedger(tmp_path)
    assert ledger.get_checks(_SHA_A) == ()


def test_latest_per_kind_keeps_last_appended(tmp_path: Path) -> None:
    ledger = VerificationLedger(tmp_path)
    auto = _make_check(
        kind=CheckKind.DATE_PLAUSIBILITY,
        verdict=Verdict.PASS,
        reviewer="auto:date_plausibility:1",
        reviewed_at="2026-05-10T01:00:00+00:00",
    )
    human_override = _make_check(
        kind=CheckKind.DATE_PLAUSIBILITY,
        verdict=Verdict.FAIL,
        reviewer="human:reviewer-x",
        reviewed_at="2026-05-10T02:00:00+00:00",
        evidence=("manual override",),
    )
    ledger.append(auto)
    ledger.append(human_override)
    latest = ledger.latest_per_kind(_SHA_A)
    assert latest[CheckKind.DATE_PLAUSIBILITY] == human_override


def test_latest_per_kind_collects_distinct_kinds(tmp_path: Path) -> None:
    ledger = VerificationLedger(tmp_path)
    date_check = _make_check(kind=CheckKind.DATE_PLAUSIBILITY)
    license_check = _make_check(
        kind=CheckKind.LICENSE_PAGE_LEVEL,
        reviewer="auto:license_page_level:1",
        evidence=("license ok",),
    )
    ledger.append(date_check)
    ledger.append(license_check)
    latest = ledger.latest_per_kind(_SHA_A)
    assert set(latest.keys()) == {
        CheckKind.DATE_PLAUSIBILITY,
        CheckKind.LICENSE_PAGE_LEVEL,
    }


def test_list_anchors_sorted_and_only_with_jsonl(tmp_path: Path) -> None:
    ledger = VerificationLedger(tmp_path)
    ledger.append(_make_check(sha=_SHA_B))
    ledger.append(_make_check(sha=_SHA_A))
    anchors = list(ledger.list_anchors())
    assert anchors == sorted([_SHA_A, _SHA_B])


def test_list_anchors_empty_when_no_root(tmp_path: Path) -> None:
    ledger = VerificationLedger(tmp_path)
    assert list(ledger.list_anchors()) == []
