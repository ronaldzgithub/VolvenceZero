from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from lifeform_synthetic_data.audit import (
    audit_projection_records,
    audit_trajectories,
    write_audit_bundle,
)
from lifeform_synthetic_data.contracts import TrainingUse
from lifeform_synthetic_data.projections import (
    PROJECTION_SCHEMA_VERSION,
    ProjectionRecord,
    ProjectionView,
)
from lifeform_synthetic_data.scenario import load_unified_v1_blueprints
from lifeform_synthetic_data.world import compile_structural_trajectory


def _golden96():
    return tuple(
        compile_structural_trajectory(
            blueprint,
            replicate_index=0,
            seed=index,
            run_id="audit-golden96",
            created_at="2026-07-20T00:00:00Z",
            git_sha="test",
        )
        for index, blueprint in enumerate(load_unified_v1_blueprints())
    )


def test_golden96_passes_hard_audit_and_writes_delivery_bundle(
    tmp_path: Path,
) -> None:
    trajectories = _golden96()

    report = audit_trajectories(trajectories, expected_count=96)
    paths = write_audit_bundle(
        report,
        trajectories,
        output_dir=tmp_path,
    )

    assert report.passed
    assert report.hard_failure_count == 0
    assert len(paths) == 6
    assert all(path.is_file() for path in paths)


def test_provenance_violation_fails_loud_hard_gate() -> None:
    trajectories = list(_golden96())
    first = trajectories[0]
    trajectories[0] = replace(
        first,
        provenance=replace(
            first.provenance,
            consent_basis="unknown_real_world_source",
        ),
    )

    report = audit_trajectories(tuple(trajectories), expected_count=96)

    assert not report.passed
    failed = {check.check_id for check in report.checks if not check.passed}
    assert "heldout_copyright_source_isolation" in failed


def test_projection_audit_blocks_eval_record_as_training_target() -> None:
    trajectory = _golden96()[0]
    record = ProjectionRecord(
        schema_version=PROJECTION_SCHEMA_VERSION,
        view=ProjectionView.EVALUATION_ONLY,
        record_id="eval:bad",
        master_trajectory_id=trajectory.trajectory_id,
        master_trajectory_hash="a" * 64,
        split=trajectory.split,
        training_use=TrainingUse.TARGET,
        payload_json='{"judge_output":{"score":5}}',
        source_refs=(trajectory.trajectory_id,),
    )

    checks = audit_projection_records((record,))

    assert not checks[0].passed
    assert checks[0].hard_gate
