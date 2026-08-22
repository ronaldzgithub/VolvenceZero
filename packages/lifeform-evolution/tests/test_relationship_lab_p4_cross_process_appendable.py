from __future__ import annotations

import json
import hashlib
import os
import pathlib
import shutil

import pytest

from lifeform_evolution.relationship_lab_p4_cross_process_appendable import (
    P4CrossProcessArm,
    load_relationship_p4_cross_process_protocol,
    run_relationship_p4_cross_process_appendable_preflight,
    run_relationship_p4_cross_process_worker,
    validate_relationship_p4_cross_process_report_files,
)
from lifeform_domain_emogpt.lab import sha256_json


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_WORKER_SCRIPT = (
    _REPO_ROOT
    / "scripts"
    / "run_relationship_lab_p4_cross_process_appendable.py"
)


def test_protocol_freezes_seen_only_cross_process_firewall() -> None:
    protocol = load_relationship_p4_cross_process_protocol()

    assert protocol.subject_ids == (
        "p4-canary-fixture-subject-01",
        "p4-canary-fixture-subject-02",
    )
    assert protocol.onboarding_pulses_per_subject == 4
    assert protocol.decision_probes_per_subject == 8
    assert protocol.max_versions == 16
    assert protocol.donor_by_subject == {
        "p4-canary-fixture-subject-01": "p4-canary-fixture-subject-02",
        "p4-canary-fixture-subject-02": "p4-canary-fixture-subject-01",
    }
    assert "no independent subjects" in protocol.claim_boundary
    assert "no model generation" in protocol.claim_boundary


@pytest.fixture(scope="module")
def cross_process_artifact(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[pathlib.Path, object]:
    output = tmp_path_factory.mktemp("p4 cross process") / "preflight"
    report = run_relationship_p4_cross_process_appendable_preflight(
        output_dir=output,
        worker_script=_WORKER_SCRIPT,
        python_executable=sys_executable(),
    )
    return output, report


def test_cross_process_preflight_uses_72_real_children_and_disk_state(
    cross_process_artifact: tuple[pathlib.Path, object],
    tmp_path: pathlib.Path,
) -> None:
    output, report = cross_process_artifact

    assert len(report.pulses) == 72
    assert report.correct_empty_forecast_presence_change_count == 16
    assert report.correct_swapped_recommended_action_change_count == 14
    assert report.mechanical_cross_process_chain_observed is True
    assert report.independent_subject_count == 0
    assert report.formal_evidence_authorized is False
    assert report.model_output_count == 0
    assert report.qwen_output_count == 0
    assert report.residual_intervention_count == 0
    assert len({item.invocation_nonce for item in report.pulses}) == 72
    assert all(item.receipt_child_pid != os.getpid() for item in report.pulses)
    assert all(item.host_child_pid == item.receipt_child_pid for item in report.pulses)
    assert all(item.receipt_parent_pid == os.getpid() for item in report.pulses)

    for subject_id in (
        "p4-canary-fixture-subject-01",
        "p4-canary-fixture-subject-02",
    ):
        correct = tuple(
            item
            for item in report.pulses
            if item.arm is P4CrossProcessArm.CORRECT_PRIOR_STATE
            and item.subject_id == subject_id
        )
        empty = tuple(
            item
            for item in report.pulses
            if item.arm is P4CrossProcessArm.EMPTY_PRIOR_STATE
            and item.subject_id == subject_id
        )
        swapped = tuple(
            item
            for item in report.pulses
            if item.arm is P4CrossProcessArm.SWAPPED_SUBJECT_PRIOR_STATE
            and item.subject_id == subject_id
        )
        assert tuple(item.post_backend_version for item in correct) == tuple(
            range(1, 13)
        )
        assert tuple(item.owner_loaded for item in correct) == (
            False,
            *(True for _ in range(11)),
        )
        assert tuple(item.post_backend_version for item in empty) == (1,) * 12
        assert not any(item.owner_loaded for item in empty)
        assert tuple(item.post_backend_version for item in swapped) == tuple(
            range(1, 13)
        )
        assert tuple(item.owner_loaded for item in swapped) == (
            False,
            *(True for _ in range(11)),
        )
        assert all(
            item.source_boundary < item.output_boundary for item in swapped
        )

    validate_relationship_p4_cross_process_report_files(output_dir=output)
    report_path = output / "cross_process_owner_hydration_report.json"
    markdown_path = output / "cross_process_owner_hydration_report.md"
    assert b"\r\n" not in report_path.read_bytes()
    assert b"\r\n" not in markdown_path.read_bytes()
    with pytest.raises(FileExistsError):
        run_relationship_p4_cross_process_appendable_preflight(
            output_dir=output,
            worker_script=_WORKER_SCRIPT,
            python_executable=sys_executable(),
        )

    forbidden_request = tmp_path / "forbidden-request.json"
    receipt = tmp_path / "forbidden-receipt.json"
    original_request_path = output.joinpath(
        *pathlib.PurePosixPath(report.pulses[0].request_path).parts
    )
    forbidden_payload = json.loads(original_request_path.read_text(encoding="utf-8"))
    forbidden_payload["history"] = ["parent-side bypass"]
    forbidden_payload["parent_pid"] = os.getppid()
    forbidden_request.write_text(
        json.dumps(forbidden_payload),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="keys drifted"):
        run_relationship_p4_cross_process_worker(
            request_path=forbidden_request,
            receipt_path=receipt,
            run_root=output,
        )
    assert not receipt.exists()



def test_validate_existing_accepts_relocated_copy(
    cross_process_artifact: tuple[pathlib.Path, object],
    tmp_path: pathlib.Path,
) -> None:
    output, _ = cross_process_artifact
    relocated = tmp_path / "relocated directory with spaces" / "artifact"
    shutil.copytree(output, relocated)

    validate_relationship_p4_cross_process_report_files(output_dir=relocated)


def test_validate_existing_rejects_deleted_receipt_firewall_key(
    cross_process_artifact: tuple[pathlib.Path, object],
    tmp_path: pathlib.Path,
) -> None:
    output, _ = cross_process_artifact
    tampered = _artifact_copy(output, tmp_path, "missing-firewall")
    report, pulse, receipt_path, receipt = _first_pulse_files(tampered)
    del receipt["gate_invoked"]
    _reseal_receipt_and_report(
        root=tampered,
        report=report,
        pulse=pulse,
        receipt_path=receipt_path,
        receipt=receipt,
    )

    with pytest.raises(ValueError, match="worker receipt keys drifted"):
        validate_relationship_p4_cross_process_report_files(output_dir=tampered)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("forecast_count", 1, "onboarding receipt forecast drift"),
        ("post_backend_version", 2, "backend version drift"),
    ),
)
def test_validate_existing_rejects_receipt_value_tampering(
    cross_process_artifact: tuple[pathlib.Path, object],
    tmp_path: pathlib.Path,
    field: str,
    value: object,
    message: str,
) -> None:
    output, _ = cross_process_artifact
    tampered = _artifact_copy(output, tmp_path, f"tampered-{field}")
    report, pulse, receipt_path, receipt = _first_pulse_files(tampered)
    receipt[field] = value
    _reseal_receipt_and_report(
        root=tampered,
        report=report,
        pulse=pulse,
        receipt_path=receipt_path,
        receipt=receipt,
    )

    with pytest.raises(ValueError, match=message):
        validate_relationship_p4_cross_process_report_files(output_dir=tampered)


def test_validate_existing_rejects_resealed_noncanonical_session(
    cross_process_artifact: tuple[pathlib.Path, object],
    tmp_path: pathlib.Path,
) -> None:
    output, _ = cross_process_artifact
    tampered = _artifact_copy(output, tmp_path, "tampered-session")
    report, pulse, receipt_path, receipt = _first_pulse_files(tampered)
    session_path = tampered.joinpath(
        *pathlib.PurePosixPath(pulse["session_path"]).parts
    )
    session = _read_json(session_path)
    session["observation_summary"] = "resealed but no longer canonical"
    session_sha256 = _write_json(session_path, session)
    receipt["session_sha256"] = session_sha256
    pulse["session_sha256"] = session_sha256
    _reseal_receipt_and_report(
        root=tampered,
        report=report,
        pulse=pulse,
        receipt_path=receipt_path,
        receipt=receipt,
    )

    with pytest.raises(ValueError, match="public session lineage drift"):
        validate_relationship_p4_cross_process_report_files(output_dir=tampered)


def test_validate_existing_rejects_resealed_recommended_action(
    cross_process_artifact: tuple[pathlib.Path, object],
    tmp_path: pathlib.Path,
) -> None:
    output, _ = cross_process_artifact
    tampered = _artifact_copy(output, tmp_path, "tampered-recommendation")
    report = _read_json(tampered / "cross_process_owner_hydration_report.json")
    pulses = report["pulses"]
    by_key = {
        (item["arm"], item["subject_id"], item["pulse_index"]): item
        for item in pulses
    }
    pulse = next(
        item
        for item in pulses
        if item["arm"] == P4CrossProcessArm.CORRECT_PRIOR_STATE.value
        and item["session_kind"] == "decision_probe"
        and item["recommended_action_id"]
        != by_key[
            (
                P4CrossProcessArm.SWAPPED_SUBJECT_PRIOR_STATE.value,
                item["subject_id"],
                item["pulse_index"],
            )
        ]["recommended_action_id"]
    )
    swapped_action = by_key[
        (
            P4CrossProcessArm.SWAPPED_SUBJECT_PRIOR_STATE.value,
            pulse["subject_id"],
            pulse["pulse_index"],
        )
    ]["recommended_action_id"]
    session_path = tampered.joinpath(
        *pathlib.PurePosixPath(pulse["session_path"]).parts
    )
    session = _read_json(session_path)
    replacement = next(
        action
        for action in session["candidate_action_ids"]
        if action not in {pulse["recommended_action_id"], swapped_action}
    )
    receipt_path = tampered.joinpath(
        *pathlib.PurePosixPath(pulse["receipt_path"]).parts
    )
    receipt = _read_json(receipt_path)
    receipt["recommended_action_id"] = replacement
    pulse["recommended_action_id"] = replacement
    _reseal_receipt_and_report(
        root=tampered,
        report=report,
        pulse=pulse,
        receipt_path=receipt_path,
        receipt=receipt,
    )

    with pytest.raises(ValueError, match="receipt/checkpoint drift"):
        validate_relationship_p4_cross_process_report_files(output_dir=tampered)


def test_validate_existing_rejects_report_metric_tampering(
    cross_process_artifact: tuple[pathlib.Path, object],
    tmp_path: pathlib.Path,
) -> None:
    output, _ = cross_process_artifact
    tampered_root = _artifact_copy(output, tmp_path, "tampered-report")
    report_path = tampered_root / "cross_process_owner_hydration_report.json"
    tampered = _read_json(report_path)
    tampered["metrics"]["correct_empty_forecast_presence_change_count"] -= 1
    _write_json(report_path, tampered)
    with pytest.raises(ValueError, match="artifact drift"):
        validate_relationship_p4_cross_process_report_files(output_dir=tampered_root)


def _artifact_copy(
    source: pathlib.Path,
    tmp_path: pathlib.Path,
    name: str,
) -> pathlib.Path:
    target = tmp_path / name
    shutil.copytree(source, target)
    return target


def _first_pulse_files(
    root: pathlib.Path,
) -> tuple[dict[str, object], dict[str, object], pathlib.Path, dict[str, object]]:
    report = _read_json(root / "cross_process_owner_hydration_report.json")
    pulse = report["pulses"][0]
    receipt_path = root.joinpath(
        *pathlib.PurePosixPath(pulse["receipt_path"]).parts
    )
    return report, pulse, receipt_path, _read_json(receipt_path)


def _reseal_receipt_and_report(
    *,
    root: pathlib.Path,
    report: dict[str, object],
    pulse: dict[str, object],
    receipt_path: pathlib.Path,
    receipt: dict[str, object],
) -> None:
    pulse["receipt_sha256"] = _write_json(receipt_path, receipt)
    unsigned_report = dict(report)
    unsigned_report.pop("artifact_id")
    report["artifact_id"] = sha256_json(unsigned_report)
    _write_json(root / "cross_process_owner_hydration_report.json", report)


def _read_json(path: pathlib.Path) -> dict[str, object]:
    payload = json.loads(path.read_bytes().decode("utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_json(path: pathlib.Path, payload: object) -> str:
    data = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def sys_executable() -> str:
    import sys

    return sys.executable
