"""Smoke tests for ``scripts/external_bench/run_eqbench3.py``.

We verify the launcher's argparse + track resolution + attestation
plumbing without actually spawning ``lifeform-serve`` (that needs a
real Qwen download and a judge API key, both out of scope for unit
tests). The runner has its own integration test gate at Packet 7
when the harness checkout exists.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
RUNNER_PATH = REPO_ROOT / "scripts" / "external_bench" / "run_eqbench3.py"


@pytest.fixture(scope="module")
def runner_module():
    """Load the runner script as a Python module without invoking main()."""
    if not RUNNER_PATH.exists():
        pytest.skip(f"runner not found at {RUNNER_PATH}")
    spec = importlib.util.spec_from_file_location("run_eqbench3_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_eqbench3_runner"] = module
    spec.loader.exec_module(module)
    return module


def test_runner_module_imports_cleanly(runner_module) -> None:
    assert hasattr(runner_module, "main")
    assert hasattr(runner_module, "_TRACK_CATALOG")
    assert hasattr(runner_module, "_attestation_block")


def test_track_catalog_has_three_tracks(runner_module) -> None:
    catalog = runner_module._TRACK_CATALOG  # noqa: SLF001
    assert set(catalog.keys()) == {"companion", "companion-cold", "raw"}
    assert catalog["raw"].request_mode == "raw"
    assert catalog["companion"].request_mode == "lifeform"
    assert catalog["companion-cold"].request_mode == "lifeform"


def test_track_model_id_includes_substrate_suffix(runner_module) -> None:
    track = runner_module._TRACK_CATALOG["companion"]  # noqa: SLF001
    model_id = track.model_id("Qwen/Qwen2.5-1.5B-Instruct")
    assert model_id == "lifeform-companion@Qwen-Qwen2.5-1.5B-Instruct"


def test_resolve_tracks_accepts_csv(runner_module) -> None:
    tracks = runner_module._resolve_tracks("companion,raw")  # noqa: SLF001
    assert [t.name for t in tracks] == ["companion", "raw"]


def test_resolve_tracks_rejects_unknown(runner_module) -> None:
    with pytest.raises(SystemExit):
        runner_module._resolve_tracks("companion,wizard")  # noqa: SLF001


def test_argparse_defaults_match_documented(runner_module) -> None:
    parser = runner_module._build_parser()  # noqa: SLF001
    args = parser.parse_args([])
    assert args.tracks == "companion,companion-cold,raw"
    assert args.substrate_model_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert args.iterations == 1
    assert args.threads == 1
    assert args.with_elo is False  # rubric-only by default per debt #29


def test_attestation_block_includes_red_line_declarations(runner_module) -> None:
    """Each track's attestation must affirm the four debt #29 red lines."""

    parser = runner_module._build_parser()  # noqa: SLF001
    args = parser.parse_args([])
    track = runner_module._TRACK_CATALOG["companion"]  # noqa: SLF001
    attestation = runner_module._attestation_block(args=args, track=track)  # noqa: SLF001

    assert attestation["frozen_substrate"] is True
    assert attestation["no_kernel_modification"] is True
    assert attestation["no_benchmark_text_in_system_prompt"] is True
    assert attestation["no_internal_architecture_terms_in_model_card"] is True
    assert attestation["track"] == "companion"
    assert attestation["request_mode"] == "lifeform"
    assert "timestamp_iso" in attestation


def test_main_aborts_when_harness_dir_is_missing(runner_module, tmp_path) -> None:
    """Without an eqbench3 checkout, ``main`` must SystemExit fast with guidance."""

    fake_dir = tmp_path / "missing_eqbench3"
    with pytest.raises(SystemExit) as excinfo:
        runner_module.main(
            [
                "--tracks",
                "companion",
                "--harness-dir",
                str(fake_dir),
            ]
        )
    msg = str(excinfo.value)
    assert "harness not found" in msg or "EQ-bench" in msg


def test_extract_rubric_average_handles_missing_field(runner_module) -> None:
    assert runner_module._try_extract_rubric_average({}) is None  # noqa: SLF001
    assert runner_module._try_extract_rubric_average("not a dict") is None  # noqa: SLF001


def test_extract_rubric_average_finds_top_level_field(runner_module) -> None:
    assert (
        runner_module._try_extract_rubric_average({"rubric_average": 73.4})  # noqa: SLF001
        == 73.4
    )


def test_extract_rubric_average_finds_nested_field(runner_module) -> None:
    assert (
        runner_module._try_extract_rubric_average(  # noqa: SLF001
            {"run-key-001": {"results": {"rubric_average": 64.5}}}
        )
        == 64.5
    )
