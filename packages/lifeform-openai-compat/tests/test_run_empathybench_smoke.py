"""Smoke tests for ``scripts/external_bench/run_empathybench.py``.

Mirrors ``test_run_eqbench3_smoke.py`` for the generic
empathybench-style runner. Verifies argparse + track resolution +
attestation plumbing without spawning anything.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
RUNNER_PATH = REPO_ROOT / "scripts" / "external_bench" / "run_empathybench.py"


@pytest.fixture(scope="module")
def runner_module():
    if not RUNNER_PATH.exists():
        pytest.skip(f"runner not found at {RUNNER_PATH}")
    spec = importlib.util.spec_from_file_location("run_empathybench_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_empathybench_runner"] = module
    spec.loader.exec_module(module)
    return module


def test_runner_module_imports_cleanly(runner_module) -> None:
    assert hasattr(runner_module, "main")
    assert hasattr(runner_module, "_TRACK_CATALOG")


def test_track_catalog_matches_eqbench3_runner(runner_module) -> None:
    catalog = runner_module._TRACK_CATALOG  # noqa: SLF001
    assert set(catalog.keys()) == {"companion", "companion-cold", "raw"}


def test_argparse_requires_harness_command(runner_module) -> None:
    parser = runner_module._build_parser()  # noqa: SLF001
    with pytest.raises(SystemExit):
        parser.parse_args([])  # missing --harness-command should exit


def test_argparse_accepts_shell_quoted_harness_command(runner_module) -> None:
    parser = runner_module._build_parser()  # noqa: SLF001
    args = parser.parse_args(
        [
            "--harness-command",
            "python external/emotionbench/run.py --cycles 1",
        ]
    )
    assert args.harness_command == "python external/emotionbench/run.py --cycles 1"
    assert args.harness_label == "generic"


def test_attestation_block_records_harness_command(runner_module) -> None:
    parser = runner_module._build_parser()  # noqa: SLF001
    args = parser.parse_args(
        [
            "--harness-command",
            "echo stub",
            "--harness-label",
            "manual-transcripts",
        ]
    )
    track = runner_module._TRACK_CATALOG["companion"]  # noqa: SLF001
    attestation = runner_module._attestation_block(args=args, track=track)  # noqa: SLF001
    assert attestation["frozen_substrate"] is True
    assert attestation["no_kernel_modification"] is True
    assert attestation["no_benchmark_text_in_system_prompt"] is True
    assert attestation["no_internal_architecture_terms_in_model_card"] is True
    assert attestation["harness_label"] == "manual-transcripts"
    assert attestation["harness_command"] == "echo stub"


def test_resolve_tracks_rejects_unknown(runner_module) -> None:
    with pytest.raises(SystemExit):
        runner_module._resolve_tracks("companion,wizard")  # noqa: SLF001
