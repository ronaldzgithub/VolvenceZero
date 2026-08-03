from __future__ import annotations

import json
from pathlib import Path

import pytest

from volvence_forge.cli import main
from volvence_forge.config import ForgeConfig, ForgePaths
from volvence_forge.foundation import (
    ForgeError,
    ReplayStructuredBackend,
    SchemaContractError,
    SchemaStore,
    read_json,
)
from volvence_forge.task_benchmark import run_task_benchmark


REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET = ".cursor/rules/cursor-convergence-workflow.mdc"
SUITE_PATH = REPO_ROOT / "forge" / "benchmarks" / "task_level_held_out.v1.json"


class _CapturingReplayBackend(ReplayStructuredBackend):
    def __init__(self, responses: list[dict[str, object]]) -> None:
        super().__init__(responses=responses)
        self.user_prompts: list[str] = []

    def complete_json(self, *, system, user, schema):
        self.user_prompts.append(user)
        return super().complete_json(system=system, user=user, schema=schema)


def _config() -> ForgeConfig:
    return ForgeConfig.load(ForgePaths.discover(repo_root=REPO_ROOT))


def _expected_responses() -> list[dict[str, object]]:
    suite = read_json(SUITE_PATH)
    return [
        {
            "case_id": case["expected"]["case_id"],
            "classification": case["expected"]["classification"],
            "next_action": case["expected"]["next_action"],
            "target_lane": case["expected"]["target_lane"],
            "preserve_evidence": case["expected"]["preserve_evidence"],
            "confidence": max(case["minimum_confidence"], 0.95),
        }
        for case in suite["cases"]
    ]


def test_baseline_benchmark_passes_and_does_not_leak_labels(tmp_path: Path) -> None:
    backend = _CapturingReplayBackend(_expected_responses())
    report_path = tmp_path / "report.json"

    result = run_task_benchmark(
        config=_config(),
        target=TARGET,
        backend=backend,
        report_path=report_path,
    )

    assert result.status == "PASS"
    assert result.baseline_pass_rate == 1.0
    assert result.candidate_pass_rate is None
    report = read_json(report_path)
    assert report["diagnostic_only"] is True
    assert report["causal_claim_authorized"] is False
    assert report["baseline"]["failures"] == []
    report["baseline"]["label"] = "candidate"
    with pytest.raises(SchemaContractError):
        SchemaStore(REPO_ROOT / "forge" / "schemas").validate(
            report, "task_benchmark_report.schema.json"
        )
    assert len(backend.user_prompts) == 8
    for prompt in backend.user_prompts:
        assert '"expected":' not in prompt
        assert '"critical":' not in prompt
        assert '"minimum_confidence":' not in prompt


def test_candidate_regression_blocks_on_critical_case(tmp_path: Path) -> None:
    baseline = _expected_responses()
    candidate = _expected_responses()
    candidate[6] = {
        **candidate[6],
        "classification": "execution_failure",
        "next_action": "propose_bounded_fix",
        "target_lane": "development",
    }
    candidate_asset = tmp_path / "candidate.mdc"
    candidate_asset.write_text("# Candidate harness rule\n", encoding="utf-8")

    result = run_task_benchmark(
        config=_config(),
        target=TARGET,
        backend=ReplayStructuredBackend(baseline + candidate),
        candidate_asset_path=candidate_asset,
        report_path=tmp_path / "report.json",
    )

    assert result.status == "BLOCK"
    assert result.baseline_pass_rate == 1.0
    assert result.candidate_pass_rate == 0.875
    report = read_json(result.report_path)
    assert report["candidate_delta"] == -0.125
    assert report["candidate"]["critical_failure_count"] == 1
    assert report["candidate"]["failures"][0]["case_id"] == "tb_positive_live_outcome"


@pytest.mark.parametrize(
    "target",
    (
        "docs/specs/rsi-forge.md",
        "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/"
        "runtime_assets/companion_playbook_overlay.json",
    ),
)
def test_benchmark_rejects_protected_or_inapplicable_target(
    tmp_path: Path, target: str
) -> None:
    with pytest.raises(ForgeError):
        run_task_benchmark(
            config=_config(),
            target=target,
            backend=ReplayStructuredBackend(_expected_responses()),
            report_path=tmp_path / "report.json",
        )


def test_cli_runs_task_benchmark_with_replay_backend(tmp_path: Path) -> None:
    replay_path = tmp_path / "replay.json"
    replay_path.write_text(
        json.dumps({"model": "held-out-fixture", "responses": _expected_responses()}),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"

    exit_code = main(
        [
            "--repo-root",
            str(REPO_ROOT),
            "benchmark",
            TARGET,
            "--backend",
            "replay",
            "--replay-responses",
            str(replay_path),
            "--output",
            str(report_path),
        ]
    )

    assert exit_code == 0
    assert read_json(report_path)["status"] == "PASS"
