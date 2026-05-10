"""Smoke tests for the content-addressable cleaning store (debt #28)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lifeform_domain_figure.cleaning.cleaners import (
    CURRENT_CLEANER_PIPELINE_VERSION,
    clean_raw_document,
)
from lifeform_domain_figure.cleaning.parsers import parse_archive_org_ocr_json
from lifeform_domain_figure.cleaning.parsers.archive_org_ocr import (
    ARCHIVE_ORG_OCR_CONTENT_TYPE,
)
from lifeform_domain_figure.cleaning.raw_document import (
    CleanedDocument,
    CleaningOp,
    CleaningOpRecord,
)
from lifeform_domain_figure.cleaning.store import CleaningStore, RawSidecar
from cleaning_fixtures import build_archive_org_ocr_json_bytes


_DUMMY_SHA = "0" * 64


def _make_cleaned(text: str, *, version: int) -> CleanedDocument:
    return CleanedDocument(
        text=text,
        raw_sha256=_DUMMY_SHA,
        cleaner_pipeline_version=version,
        cleaning_log=(
            CleaningOpRecord(
                op=CleaningOp.WHITESPACE_NORMALIZE,
                op_version="1",
                chars_before=len(text) + 1,
                chars_after=len(text),
            ),
        ),
        parser_version="test:1",
    )


def test_put_raw_is_idempotent_and_returns_sha(tmp_path: Path) -> None:
    store = CleaningStore(tmp_path)
    data = b"hello cleaning world"
    sha_a = store.put_raw(data, source_url="test://a", content_type="text/plain")
    sha_b = store.put_raw(data, source_url="test://a", content_type="text/plain")
    assert sha_a == sha_b == hashlib.sha256(data).hexdigest()
    bytes_path = tmp_path / "raw" / sha_a / "bytes"
    sidecar_path = tmp_path / "raw" / sha_a / "sidecar.json"
    assert bytes_path.exists()
    assert sidecar_path.exists()
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["source_url"] == "test://a"
    assert sidecar["content_type"] == "text/plain"
    assert sidecar["byte_len"] == len(data)


def test_get_raw_round_trip(tmp_path: Path) -> None:
    store = CleaningStore(tmp_path)
    data = build_archive_org_ocr_json_bytes()
    sha = store.put_raw(
        data,
        source_url="test://archive.org/details/sample",
        content_type=ARCHIVE_ORG_OCR_CONTENT_TYPE,
    )
    fetched_bytes, fetched_sidecar = store.get_raw(sha)
    assert fetched_bytes == data
    assert isinstance(fetched_sidecar, RawSidecar)
    assert fetched_sidecar.source_url == "test://archive.org/details/sample"
    assert fetched_sidecar.content_type == ARCHIVE_ORG_OCR_CONTENT_TYPE


def test_get_raw_missing_raises_file_not_found(tmp_path: Path) -> None:
    store = CleaningStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.get_raw("a" * 64)


def test_put_cleaned_isolates_versions(tmp_path: Path) -> None:
    store = CleaningStore(tmp_path)
    v1 = _make_cleaned("first version text", version=1)
    v2 = _make_cleaned("second version text - longer body", version=2)
    dir_v1 = store.put_cleaned(v1)
    dir_v2 = store.put_cleaned(v2)
    assert dir_v1 == tmp_path / "cleaned" / _DUMMY_SHA / "v1"
    assert dir_v2 == tmp_path / "cleaned" / _DUMMY_SHA / "v2"
    assert (dir_v1 / "text.txt").read_text(encoding="utf-8") == "first version text"
    assert (dir_v2 / "text.txt").read_text(encoding="utf-8") == "second version text - longer body"


def test_get_cleaned_round_trip(tmp_path: Path) -> None:
    store = CleaningStore(tmp_path)
    cleaned = _make_cleaned("body text payload", version=1)
    store.put_cleaned(cleaned)
    fetched = store.get_cleaned(_DUMMY_SHA, 1)
    assert fetched is not None
    assert fetched.text == cleaned.text
    assert fetched.cleaner_pipeline_version == cleaned.cleaner_pipeline_version
    assert fetched.cleaning_log == cleaned.cleaning_log
    assert fetched.parser_version == cleaned.parser_version


def test_get_cleaned_missing_returns_none(tmp_path: Path) -> None:
    store = CleaningStore(tmp_path)
    assert store.get_cleaned(_DUMMY_SHA, 1) is None


def test_list_raw_and_list_cleaned_versions(tmp_path: Path) -> None:
    store = CleaningStore(tmp_path)
    sha_a = store.put_raw(b"alpha", source_url="test://a", content_type="text/plain")
    sha_b = store.put_raw(b"beta", source_url="test://b", content_type="text/plain")
    cleaned_a_v1 = CleanedDocument(
        text="alpha",
        raw_sha256=sha_a,
        cleaner_pipeline_version=1,
        cleaning_log=(),
        parser_version="test:1",
    )
    cleaned_a_v2 = CleanedDocument(
        text="alpha",
        raw_sha256=sha_a,
        cleaner_pipeline_version=2,
        cleaning_log=(),
        parser_version="test:1",
    )
    cleaned_b_v1 = CleanedDocument(
        text="beta",
        raw_sha256=sha_b,
        cleaner_pipeline_version=1,
        cleaning_log=(),
        parser_version="test:1",
    )
    store.put_cleaned(cleaned_a_v1)
    store.put_cleaned(cleaned_a_v2)
    store.put_cleaned(cleaned_b_v1)
    raws = sorted(store.list_raw())
    assert raws == sorted([sha_a, sha_b])
    assert store.list_cleaned_versions(sha_a) == (1, 2)
    assert store.list_cleaned_versions(sha_b) == (1,)
    assert store.list_cleaned_versions("0" * 64) == ()


def test_full_pipeline_through_store(tmp_path: Path) -> None:
    store = CleaningStore(tmp_path)
    data = build_archive_org_ocr_json_bytes()
    sha = store.put_raw(
        data,
        source_url="test://archive.org/details/sample",
        content_type=ARCHIVE_ORG_OCR_CONTENT_TYPE,
    )
    raw = parse_archive_org_ocr_json(
        data,
        source_url="test://archive.org/details/sample",
        content_type=ARCHIVE_ORG_OCR_CONTENT_TYPE,
    )
    assert raw.raw_sha256 == sha
    cleaned = clean_raw_document(raw)
    cleaned_dir = store.put_cleaned(cleaned)
    log_payload = json.loads((cleaned_dir / "cleaning_log.json").read_text(encoding="utf-8"))
    assert log_payload["cleaner_pipeline_version"] == CURRENT_CLEANER_PIPELINE_VERSION
    assert log_payload["raw_sha256"] == sha
    assert isinstance(log_payload["ops"], list) and log_payload["ops"]
