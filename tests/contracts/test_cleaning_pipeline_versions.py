"""Contract guards for the L1 cleaning pipeline (debt #28).

Two classes of invariants covered here:

1. **Versioning + storage isolation** — the cleaner pipeline is
   versioned; the store MUST keep different ``cleaner_pipeline_version``
   outputs in separate directories under the same ``raw_sha256``;
   re-cleaning a known input on a new version MUST NOT change the
   raw sha256.

2. **Module-boundary AST guards** — every Python source under
   ``packages/lifeform-domain-figure/src/lifeform_domain_figure/cleaning/``
   MUST NOT directly import any typed figure source record
   (``FigurePaperSource`` / ``FigureLetterSource`` /
   ``FigureLectureSource`` / ``FigureNotebookSource``) — bridging
   into typed sources is the explicit two-step ``bridging.py``
   responsibility (R8 / `ssot-module-boundaries.mdc`).

   They MUST also NOT import any HTTP client library — the cleaning
   pipeline is decoupled from V2 HTTPArchiveFetcher (debt #19);
   wiring the two is a separate packet.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from lifeform_domain_figure.cleaning.cleaners import (
    CURRENT_CLEANER_PIPELINE_VERSION,
    clean_raw_document,
)
import lifeform_domain_figure.cleaning.cleaners as cleaners_pkg
from lifeform_domain_figure.cleaning.raw_document import (
    CleanedDocument,
    CleaningOp,
    CleaningOpRecord,
    RawDocument,
)
from lifeform_domain_figure.cleaning.store import CleaningStore


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLEANING_ROOT = (
    _REPO_ROOT
    / "packages"
    / "lifeform-domain-figure"
    / "src"
    / "lifeform_domain_figure"
    / "cleaning"
)


_FORBIDDEN_TYPED_SOURCE_NAMES = {
    "FigurePaperSource",
    "FigureLetterSource",
    "FigureLectureSource",
    "FigureNotebookSource",
}


_FORBIDDEN_HTTP_MODULES = {
    "requests",
    "httpx",
    "aiohttp",
    "urllib",
    "urllib2",
    "urllib3",
    "http",
    "http.client",
    "tornado.httpclient",
}


def _iter_cleaning_python_files() -> list[Path]:
    return sorted(p for p in _CLEANING_ROOT.rglob("*.py"))


def test_cleaning_root_directory_exists() -> None:
    assert _CLEANING_ROOT.exists(), (
        f"contract anchor directory missing: {_CLEANING_ROOT}"
    )
    files = _iter_cleaning_python_files()
    assert files, f"no Python files found under {_CLEANING_ROOT}"


def test_no_cleaning_module_imports_typed_figure_source() -> None:
    """``cleaning/*`` MUST NOT import any ``Figure*Source`` typed record.

    Two-step bridging via :mod:`lifeform_domain_figure.cleaning.bridging`
    is the only sanctioned path into a typed source.
    """

    violations: list[tuple[Path, int, str]] = []
    for path in _iter_cleaning_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in _FORBIDDEN_TYPED_SOURCE_NAMES:
                        violations.append((path, node.lineno, alias.name))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name.split(".")[-1]
                    if name in _FORBIDDEN_TYPED_SOURCE_NAMES:
                        violations.append((path, node.lineno, alias.name))
    assert not violations, (
        "cleaning/ files must not import Figure*Source typed records "
        "(use cleaning/bridging.py + existing *_to_*_source instead). "
        f"Violations: {violations}"
    )


def test_no_cleaning_module_imports_http_client() -> None:
    """``cleaning/*`` MUST NOT depend on any HTTP client.

    The L1 cleaner is fed by the curator (or the future V2
    HTTPArchiveFetcher) — not by inline HTTP. Mixing the two
    responsibilities re-creates the SSRF / fetch-safety surface that
    debt #19 / #26 explicitly carves out as a separate packet.
    """

    violations: list[tuple[Path, int, str]] = []
    for path in _iter_cleaning_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in _FORBIDDEN_HTTP_MODULES:
                        violations.append((path, node.lineno, alias.name))
            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    continue
                root = node.module.split(".")[0]
                if root in _FORBIDDEN_HTTP_MODULES or node.module in _FORBIDDEN_HTTP_MODULES:
                    violations.append((path, node.lineno, node.module))
    assert not violations, (
        "cleaning/ files must not import HTTP clients "
        "(L1 is decoupled from V2 fetcher per debt #28 / #19). "
        f"Violations: {violations}"
    )


def test_raw_sha_is_stable_across_cleaner_versions(tmp_path: Path) -> None:
    """Same input bytes MUST produce the same raw_sha256 regardless of cleaner version."""

    store = CleaningStore(tmp_path)
    data = b"the same exact bytes for the contract test"
    sha_first = store.put_raw(data, source_url="test://sha", content_type="text/plain")
    sha_second = store.put_raw(data, source_url="test://sha", content_type="text/plain")
    assert sha_first == sha_second == hashlib.sha256(data).hexdigest()


def test_store_isolates_v1_and_v2_directories(tmp_path: Path) -> None:
    """``data/cleaned/{sha}/v1/`` and ``v2/`` MUST coexist independently."""

    store = CleaningStore(tmp_path)
    raw_sha = "f" * 64
    v1 = CleanedDocument(
        text="cleaned text under v1",
        raw_sha256=raw_sha,
        cleaner_pipeline_version=1,
        cleaning_log=(
            CleaningOpRecord(
                op=CleaningOp.WHITESPACE_NORMALIZE,
                op_version="1",
                chars_before=30,
                chars_after=21,
            ),
        ),
        parser_version="test:1",
    )
    v2 = CleanedDocument(
        text="cleaned text under v2 has more aggressive cleaning",
        raw_sha256=raw_sha,
        cleaner_pipeline_version=2,
        cleaning_log=(
            CleaningOpRecord(
                op=CleaningOp.BOILERPLATE_STRIP,
                op_version="2",
                chars_before=80,
                chars_after=51,
            ),
        ),
        parser_version="test:1",
    )
    store.put_cleaned(v1)
    store.put_cleaned(v2)
    assert (tmp_path / "cleaned" / raw_sha / "v1" / "text.txt").exists()
    assert (tmp_path / "cleaned" / raw_sha / "v2" / "text.txt").exists()
    assert store.list_cleaned_versions(raw_sha) == (1, 2)
    assert store.get_cleaned(raw_sha, 1) == v1
    assert store.get_cleaned(raw_sha, 2) == v2


def test_pipeline_v1_and_temporary_v2_produce_different_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A registered v2 pipeline composition MUST produce different output from v1.

    We register a synthetic v2 by injecting an extra step into
    ``_PIPELINES`` for the duration of the test. This is the
    contract surface the future production v2 will use; failing it
    means the orchestrator silently ignored the version selector.
    """

    def _drop_first_paragraph(text: str) -> str:
        if "\n\n" in text:
            return text.split("\n\n", 1)[1]
        return text

    fake_v2 = cleaners_pkg.CLEANER_PIPELINE_V1 + (
        (CleaningOp.PARAGRAPH_NORMALIZE, "v2-extra", _drop_first_paragraph),
    )
    pipelines_attr = cleaners_pkg._PIPELINES
    monkeypatch.setitem(pipelines_attr, 2, fake_v2)
    raw = RawDocument(
        text=(
            "First paragraph that gets dropped under the v2 composition.\n\n"
            "Second paragraph that survives both v1 and v2.\n\n"
            "Third paragraph that also survives."
        ),
        parser_version="test:1",
        layout_quality=1.0,
        ocr_confidence=1.0,
        encoding_detected="utf-8",
        language_detected="en",
        license_notice="",
        raw_sha256="a" * 64,
    )
    cleaned_v1 = clean_raw_document(raw, pipeline_version=1)
    cleaned_v2 = clean_raw_document(raw, pipeline_version=2)
    assert cleaned_v1.text != cleaned_v2.text
    assert cleaned_v1.raw_sha256 == cleaned_v2.raw_sha256
    assert cleaned_v1.cleaner_pipeline_version == 1
    assert cleaned_v2.cleaner_pipeline_version == 2
    assert "First paragraph" in cleaned_v1.text
    assert "First paragraph" not in cleaned_v2.text
    store = CleaningStore(tmp_path)
    store.put_cleaned(cleaned_v1)
    store.put_cleaned(cleaned_v2)
    assert store.list_cleaned_versions("a" * 64) == (1, 2)


def test_cleaning_log_records_are_non_empty_and_monotonic() -> None:
    raw = RawDocument(
        text="\n\n\n  multiple   spaces  here   \n\n\n",
        parser_version="test:1",
        layout_quality=1.0,
        ocr_confidence=1.0,
        encoding_detected="utf-8",
        language_detected="en",
        license_notice="",
        raw_sha256="b" * 64,
    )
    cleaned = clean_raw_document(raw, pipeline_version=CURRENT_CLEANER_PIPELINE_VERSION)
    assert cleaned.cleaning_log, "cleaning_log must record at least one op"
    for record in cleaned.cleaning_log:
        assert record.chars_after <= record.chars_before
        assert record.op_version, "every CleaningOpRecord must carry an op_version"
