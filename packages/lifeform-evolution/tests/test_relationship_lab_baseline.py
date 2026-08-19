from __future__ import annotations

import hashlib
import json

from lifeform_domain_emogpt.lab import RelationshipAction, sha256_json
from lifeform_evolution.relationship_lab_baseline import (
    StatelessActionCompletion,
    action_choice_schema_path,
    freeze_stateless_baseline_attestation,
    run_stateless_baseline,
    stateless_prompt_path,
    write_stateless_baseline_run,
)
from lifeform_evolution.relationship_lab_gate0 import (
    Gate0CalibrationConfig,
    GateCheckStatus,
    run_relationship_gate0_calibration,
)


class _AlwaysStayPolicy:
    model_id = "frozen-fake-policy"
    weights_sha256 = sha256_json("fake-weights")
    prompt_sha256 = sha256_json("fake-prompt")
    generation_config_sha256 = sha256_json("fake-generation")

    def choose(self, *, current_input: str, seed: int) -> StatelessActionCompletion:
        assert current_input
        assert seed >= 0
        return StatelessActionCompletion(
            raw_output='{"action_id":"stay_present_without_probe"}',
            chosen_action_id=RelationshipAction.STAY_PRESENT_WITHOUT_PROBE,
            prompt_tokens=40,
            completion_tokens=8,
        )


class _InvalidOutputPolicy(_AlwaysStayPolicy):
    def choose(self, *, current_input: str, seed: int) -> StatelessActionCompletion:
        del current_input, seed
        return StatelessActionCompletion(
            raw_output="I would stay.",
            chosen_action_id=None,
            prompt_tokens=40,
            completion_tokens=4,
        )


def test_stateless_runner_is_matched_and_excludes_heldout() -> None:
    run = run_stateless_baseline(_AlwaysStayPolicy())
    assert len(run.decisions) == 24
    assert run.valid_decisions == 24
    assert run.correct_decisions == 12
    assert run.context_tokens_total == 960
    assert {item.split.value for item in run.decisions} == {"train", "validation"}
    grouped: dict[tuple[str, int], list] = {}
    for decision in run.decisions:
        grouped.setdefault((decision.pair_id, decision.seed), []).append(decision)
    assert grouped
    for rows in grouped.values():
        assert len(rows) == 2
        assert rows[0].current_input_sha256 == rows[1].current_input_sha256
        assert rows[0].raw_output == rows[1].raw_output
        assert rows[0].chosen_action_id == rows[1].chosen_action_id
        assert rows[0].correct != rows[1].correct


def test_stateless_run_freezes_ledger_backed_attestation_and_closes_gate0(
    tmp_path,
) -> None:
    run = run_stateless_baseline(_AlwaysStayPolicy())
    attestation = freeze_stateless_baseline_attestation(
        run,
        frozen_at_iso="2026-08-19T02:00:00+00:00",
    )
    assert attestation.decision_ledger_sha256 == run.decision_ledger_sha256
    assert attestation.valid_decisions == len(run.decisions)
    report = run_relationship_gate0_calibration(
        config=Gate0CalibrationConfig(samples_per_action=64),
        baseline=attestation,
        created_at_iso="2026-08-19T02:01:00+00:00",
    )
    assert report.gate0_passed

    ledger, summary, frozen = write_stateless_baseline_run(
        run,
        output_dir=tmp_path,
        frozen_at_iso="2026-08-19T02:00:00+00:00",
    )
    assert hashlib.sha256(ledger.read_bytes()).hexdigest() == (run.decision_ledger_sha256)
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 24
    summary_payload = json.loads(summary.read_text(encoding="utf-8"))
    assert summary_payload["decision_ledger_sha256"] == run.decision_ledger_sha256
    frozen_payload = json.loads(frozen.read_text(encoding="utf-8"))
    assert frozen_payload["artifact_id"] == attestation.artifact_id


def test_invalid_structured_outputs_cannot_close_baseline_tooth() -> None:
    run = run_stateless_baseline(_InvalidOutputPolicy())
    assert run.valid_decisions == 0
    assert run.correct_decisions == 0
    attestation = freeze_stateless_baseline_attestation(
        run,
        frozen_at_iso="2026-08-19T02:00:00+00:00",
    )
    report = run_relationship_gate0_calibration(
        config=Gate0CalibrationConfig(samples_per_action=64),
        baseline=attestation,
        created_at_iso="2026-08-19T02:01:00+00:00",
    )
    assert report.machinery_ready
    assert not report.gate0_passed
    statuses = {check.check_id: check.status for check in report.checks}
    assert statuses["frozen_baseline_non_saturation"] is GateCheckStatus.FAIL


def test_stateless_prompt_and_schema_are_dedicated_assets() -> None:
    prompt = stateless_prompt_path().read_text(encoding="utf-8")
    schema = json.loads(action_choice_schema_path().read_text(encoding="utf-8"))
    assert "user history" in prompt
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]["action_id"]["enum"]) == {
        "stay_present_without_probe",
        "respect_space_with_return_option",
        "neutral_noop",
    }
