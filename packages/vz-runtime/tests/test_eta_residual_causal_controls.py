from __future__ import annotations

from volvence_zero.agent.eta_proof_benchmark import (
    default_eta_proof_cases,
    run_eta_internal_rl_proof_benchmark,
)


def test_eta_residual_controls_record_actual_matched_interventions() -> None:
    cases = default_eta_proof_cases()
    train_case = next(case for case in cases if case.split == "train")
    heldout_case = next(case for case in cases if case.split != "train")
    report = run_eta_internal_rl_proof_benchmark(
        cases=(train_case, heldout_case),
        profile_labels=(
            "full-internal-rl",
            "full-zero-control",
            "full-shuffled-control",
            "full-reversed-control",
        ),
        backend_label="trace",
        train_epochs=1,
    )

    records_by_profile = {
        profile.profile_label: tuple(
            record
            for episode in profile.episode_reports
            for record in episode.intervention_records
        )
        for profile in report.profile_reports
    }
    assert all(records_by_profile.values())

    for record in records_by_profile["full-internal-rl"]:
        assert record.residual_control_mode == "identity"
        assert record.applied_control == record.control_before_ablation

    for record in records_by_profile["full-zero-control"]:
        assert record.residual_control_mode == "zero"
        assert record.applied_control == tuple(
            0.0 for _ in record.control_before_ablation
        )

    for record in records_by_profile["full-shuffled-control"]:
        assert record.residual_control_mode == "shuffled"
        assert sorted(record.applied_control) == sorted(
            record.control_before_ablation
        )
        assert record.applied_control != record.control_before_ablation

    for record in records_by_profile["full-reversed-control"]:
        assert record.residual_control_mode == "reversed"
        assert record.applied_control == tuple(
            reversed(record.control_before_ablation)
        )
