"""Packet 7.2: DirectoryScanUptake adapter tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from lifeform_protocol_runtime import (
    MockLlmJsonClient,
    scan_directory_for_protocols,
)


def _mock_client():
    return MockLlmJsonClient(
        identity={
            "advisor_name": "scan-advisor",
            "description": "test scan advisor",
            "self_traits": ["x"],
            "forbidden_traits": [],
            "regimes": [],
        },
        boundary={
            "boundaries": [
                {
                    "boundary_id": "bd:scan:test",
                    "description": "test boundary",
                    "trigger_reasons": ["test trigger"],
                    "blocked_topics": [],
                    "severity": "soft_remind",
                }
            ]
        },
        strategy={
            "strategies": [
                {
                    "rule_id": "rule:scan:test",
                    "problem_pattern": "user asks question",
                    "recommended_ordering": ["acknowledge"],
                    "recommended_pacing": "moderate",
                }
            ],
        },
    )


def test_scan_empty_directory(tmp_path: Path) -> None:
    results = scan_directory_for_protocols(
        tmp_path, llm_client=_mock_client()
    )
    assert results == ()


def test_scan_skips_unsupported_extensions(tmp_path: Path) -> None:
    (tmp_path / "ignored.json").write_text("{}", encoding="utf-8")
    results = scan_directory_for_protocols(
        tmp_path, llm_client=_mock_client()
    )
    assert len(results) == 1
    assert results[0].status == "skipped"
    assert results[0].candidate is None


def test_scan_processes_markdown_file(tmp_path: Path) -> None:
    md_file = tmp_path / "guide.md"
    md_file.write_text(
        "# Title\n\nA short description for the role.\n", encoding="utf-8"
    )
    results = scan_directory_for_protocols(
        tmp_path, llm_client=_mock_client()
    )
    assert len(results) == 1
    assert results[0].status == "ok"
    assert results[0].candidate is not None


def test_scan_recursive(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "guide.txt").write_text(
        "A short description for the assistant role.\n",
        encoding="utf-8",
    )
    results = scan_directory_for_protocols(
        tmp_path, llm_client=_mock_client(), recursive=True
    )
    ok = [r for r in results if r.status == "ok"]
    assert len(ok) == 1


def test_scan_non_recursive(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.md").write_text("# X\nbody\n", encoding="utf-8")
    (tmp_path / "top.md").write_text("# Y\nbody\n", encoding="utf-8")
    results = scan_directory_for_protocols(
        tmp_path, llm_client=_mock_client(), recursive=False
    )
    ok = [r for r in results if r.status == "ok"]
    assert len(ok) == 1
    assert "top.md" in ok[0].source_path


def test_scan_directory_path_validation() -> None:
    with pytest.raises(ValueError, match="not a directory"):
        scan_directory_for_protocols(
            "non-existent-dir-xyz",
            llm_client=_mock_client(),
        )
