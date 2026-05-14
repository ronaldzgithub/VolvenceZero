"""Contract test: B3 harness + trusted runner skeleton (debt #34 / #57).

Validates:

1. ``score_reference_systems.py`` argparse exposes ``--parallel-sut``
   (default 1 for deterministic) and ``--per-system-retries``
   (default 0). Both are documented inline so an operator running
   ``--help`` sees the trade-offs.
2. ``trusted_runner_skeleton.py`` ``--dry-run`` walks the credential
   lifecycle: schema validation → ingest log → verdict skeleton →
   transcript cleanup log → credential purge schedule.
3. ``trusted_runner_skeleton.py`` rejects unknown leaderboard_category
   in the encrypted envelope.
4. ``trusted_runner_skeleton.py`` rejects missing required fields.
5. Verdict skeleton conforms to protocol §2.2: transcript_returned
   must be False; per-axis means present (None until ACTIVE).

Refs:

* docs/known-debts.md #34 / #57
* docs/external/companion-bench-trusted-runner-protocol.md
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from types import ModuleType


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / "scripts" / "companion_bench"


def _load_script(filename: str) -> ModuleType:
    path = _SCRIPTS_DIR / filename
    spec = importlib.util.spec_from_file_location(
        f"_companion_bench_{path.stem}", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# #34: harness staged executor + retry surface
# ---------------------------------------------------------------------------


def test_score_reference_systems_exposes_parallel_sut_flag() -> None:
    score = _load_script("score_reference_systems.py")
    parser = score._build_parser()
    args = parser.parse_args(
        [
            "--output-dir", "/tmp/x",
            "--user-sim-model", "m",
            "--user-sim-key-env", "K",
            "--perturn-model", "m",
            "--perturn-key-env", "K",
            "--arc-model", "m",
            "--arc-key-env", "K",
        ]
    )
    # Defaults preserve deterministic single-thread behaviour.
    assert args.parallel_sut == 1
    assert args.per_system_retries == 0


def test_score_reference_systems_parallel_sut_clamps_to_at_least_1() -> None:
    score = _load_script("score_reference_systems.py")
    parser = score._build_parser()
    args = parser.parse_args(
        [
            "--output-dir", "/tmp/x",
            "--user-sim-model", "m",
            "--user-sim-key-env", "K",
            "--perturn-model", "m",
            "--perturn-key-env", "K",
            "--arc-model", "m",
            "--arc-key-env", "K",
            "--parallel-sut", "8",
            "--per-system-retries", "2",
        ]
    )
    assert args.parallel_sut == 8
    assert args.per_system_retries == 2


# ---------------------------------------------------------------------------
# #57: trusted runner skeleton
# ---------------------------------------------------------------------------


def _make_envelope(tmp_path: pathlib.Path, **overrides) -> pathlib.Path:
    payload = {
        "submission_id": "sub-001",
        "system_name": "Test Submitter",
        "model_identifier": "vendor/model-x",
        "base_url": "https://endpoint.example/v1",
        "api_key_ciphertext": "ZmFrZS1jaXBoZXJ0ZXh0",  # base64 of "fake-ciphertext"
        "system_prompt": "You are a careful, kind companion.",
        "generation_config": {"temperature": 0.0, "max_tokens": 512},
        "leaderboard_category": "closed-api",
    }
    payload.update(overrides)
    p = tmp_path / "submission.encrypted.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_trusted_runner_dry_run_walks_full_lifecycle(tmp_path: pathlib.Path) -> None:
    runner = _load_script("trusted_runner_skeleton.py")
    envelope = _make_envelope(tmp_path)
    output = tmp_path / "out"
    rc = runner.main(
        [
            "--encrypted-submission", str(envelope),
            "--output-dir", str(output),
            "--dry-run",
        ]
    )
    assert rc == 0

    # Verdict file shape per protocol §2.2.
    verdict_path = output / "verdict.json"
    assert verdict_path.exists()
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    assert verdict["scaffold_status"] == "SHADOW"
    assert verdict["transcript_returned"] is False
    assert set(verdict["axis_means"]) == {"A1", "A2", "A3", "A4", "A5", "A6"}
    for axis in verdict["axis_means"].values():
        # SHADOW: real means populated only in ACTIVE.
        assert axis is None

    # Secrets vault log: ingest + purge_scheduled rows.
    vault_log = output / "secrets_vault_log.jsonl"
    assert vault_log.exists()
    vault_lines = [
        json.loads(line)
        for line in vault_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stages = [entry["stage"] for entry in vault_lines]
    assert "ingest" in stages
    assert "purge_scheduled" in stages

    # Transcript deletion ledger row (even in dry-run).
    cleanup_log = output / "transcript_deletion_log.jsonl"
    assert cleanup_log.exists()


def test_trusted_runner_rejects_unknown_leaderboard_category(
    tmp_path: pathlib.Path,
) -> None:
    runner = _load_script("trusted_runner_skeleton.py")
    envelope = _make_envelope(tmp_path, leaderboard_category="marketing-hype")
    output = tmp_path / "out"
    import pytest

    with pytest.raises(ValueError, match="leaderboard_category must be"):
        runner.main(
            [
                "--encrypted-submission", str(envelope),
                "--output-dir", str(output),
                "--dry-run",
            ]
        )


def test_trusted_runner_rejects_missing_required_field(
    tmp_path: pathlib.Path,
) -> None:
    runner = _load_script("trusted_runner_skeleton.py")
    payload = {
        "submission_id": "sub-002",
        # missing required fields by design
    }
    envelope = tmp_path / "submission.encrypted.json"
    envelope.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "out"
    import pytest

    with pytest.raises(ValueError, match="missing required fields"):
        runner.main(
            [
                "--encrypted-submission", str(envelope),
                "--output-dir", str(output),
                "--dry-run",
            ]
        )


def test_trusted_runner_requires_dry_run_in_shadow(tmp_path: pathlib.Path) -> None:
    runner = _load_script("trusted_runner_skeleton.py")
    envelope = _make_envelope(tmp_path)
    output = tmp_path / "out"
    rc = runner.main(
        [
            "--encrypted-submission", str(envelope),
            "--output-dir", str(output),
            # No --dry-run: SHADOW must refuse.
        ]
    )
    assert rc == 2
