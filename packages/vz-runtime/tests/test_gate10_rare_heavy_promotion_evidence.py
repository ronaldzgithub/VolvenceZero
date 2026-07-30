from __future__ import annotations

from volvence_zero.agent.gate10_rare_heavy_promotion_evidence import (
    GATE10_ARMS,
    GATE10_REQUIRED_FILES,
    GATE10_SEEDS,
    export_gate10_evidence_bundle,
    run_gate10_evidence,
    verify_gate10_evidence_bundle,
)


def test_gate10_runs_four_arms_and_full_chain_rollback() -> None:
    report = run_gate10_evidence(seed_schedule=(GATE10_SEEDS[0],))

    assert {row.arm for row in report.results} == set(GATE10_ARMS)
    review = next(
        row
        for row in report.results
        if row.arm == "candidate-review-only"
    )
    rejected = next(
        row
        for row in report.results
        if row.arm == "rejected-candidate"
    )
    rollback = next(
        row
        for row in report.results
        if row.arm == "rollback-to-previous"
    )
    assert review.import_applied is False
    assert review.review_only_side_effect_count == 0
    assert rejected.automatic_rejection is True
    assert rollback.rollback_triggered is True
    assert rollback.rollback_exact is True
    assert rollback.checkpoint_fingerprint_match is True
    assert rollback.substrate_fingerprint_match is True


def test_gate10_formal_bundle_has_required_files(tmp_path) -> None:
    report = run_gate10_evidence()
    export_gate10_evidence_bundle(report, output_dir=tmp_path)
    verification = verify_gate10_evidence_bundle(tmp_path)

    assert verification["passed"] is True
    assert {path.name for path in tmp_path.iterdir()} == set(
        GATE10_REQUIRED_FILES
    )
