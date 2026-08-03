from __future__ import annotations

import json
from pathlib import Path

import pytest

from lifeform_evolution.seven_day_state_control import (
    SEVEN_DAY_SHUFFLED_SOURCE_DAYS,
    SevenDayFilesystemStateController,
)
from lifeform_evolution.seven_day_process_host import (
    ServiceProcessStart,
    ServiceProcessStop,
    StateControlledSubprocessLifecycle,
    SubprocessSevenDayServiceHost,
)


def _controller(
    tmp_path: Path,
    *,
    arm: str,
    reference: Path | None = None,
    donor: Path | None = None,
) -> SevenDayFilesystemStateController:
    return SevenDayFilesystemStateController(
        evidence_root=tmp_path,
        active_scope_root=tmp_path / f"active-{arm}",
        archive_root=tmp_path / f"archive-{arm}",
        user_id="synthetic-user",
        experiment_arm_label=arm,
        correct_reference_archive_root=reference,
        donor_archive_root=donor,
    )


def _write_active(
    controller: SevenDayFilesystemStateController,
    value: str,
) -> None:
    controller.active_scope_dir.mkdir(parents=True, exist_ok=True)
    (controller.active_scope_dir / "owner_v1.json").write_text(
        value,
        encoding="utf-8",
    )
    (
        controller.active_scope_dir
        / "evaluation__relationship_continuity_v1.json"
    ).write_text(f"measurement:{value}", encoding="utf-8")


def test_correct_state_archives_then_stages_exact_copy(tmp_path: Path) -> None:
    controller = _controller(tmp_path, arm="correct-user-state")
    controller.prepare_initial_day()
    _write_active(controller, "day-one")
    evidence = controller.archive_and_stage_after_day(day_index=1)
    assert evidence.next_day_source_arm == "correct-user-state"
    assert evidence.next_day_source_day_index == 1
    assert evidence.archived_state_sha256 == (
        evidence.next_day_loaded_state_sha256
    )
    assert len(evidence.measurement_checkpoint_sha256) == 64
    assert (controller.active_scope_dir / "owner_v1.json").read_text() == (
        "day-one"
    )


def test_stateless_archives_without_staging_prior_state(tmp_path: Path) -> None:
    controller = _controller(tmp_path, arm="stateless")
    controller.prepare_initial_day()
    _write_active(controller, "day-one")
    evidence = controller.archive_and_stage_after_day(day_index=1)
    assert evidence.next_day_source_arm is None
    assert evidence.next_day_loaded_state_sha256 is None
    assert controller.active_scope_dir.is_dir()
    assert tuple(path.name for path in controller.active_scope_dir.iterdir()) == (
        "evaluation__relationship_continuity_v1.json",
    )
    assert (tmp_path / "archive-stateless/day-1/owner_v1.json").is_file()


def test_swapped_state_loads_matched_donor_archive(tmp_path: Path) -> None:
    donor = tmp_path / "donor"
    (donor / "day-1").mkdir(parents=True)
    (donor / "day-1/owner_v1.json").write_text(
        "donor-state",
        encoding="utf-8",
    )
    (donor / "day-1/evaluation__relationship_continuity_v1.json").write_text(
        "donor-measurement",
        encoding="utf-8",
    )
    controller = _controller(
        tmp_path,
        arm="swapped-user-state",
        donor=donor,
    )
    controller.prepare_initial_day()
    _write_active(controller, "target-state")
    evidence = controller.archive_and_stage_after_day(day_index=1)
    assert evidence.next_day_source_arm == (
        "matched-donor-correct-user-state"
    )
    assert (controller.active_scope_dir / "owner_v1.json").read_text() == (
        "donor-state"
    )
    assert (
        controller.active_scope_dir
        / "evaluation__relationship_continuity_v1.json"
    ).read_text() == "measurement:target-state"


def test_shuffled_history_uses_frozen_non_monotonic_day_schedule(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    for day_index in range(1, 7):
        (reference / f"day-{day_index}").mkdir(parents=True)
        (reference / f"day-{day_index}/owner_v1.json").write_text(
            f"reference-{day_index}",
            encoding="utf-8",
        )
        (
            reference
            / f"day-{day_index}/evaluation__relationship_continuity_v1.json"
        ).write_text(
            f"reference-measurement-{day_index}",
            encoding="utf-8",
        )
    controller = _controller(
        tmp_path,
        arm="shuffled-history",
        reference=reference,
    )
    controller.prepare_initial_day()
    _write_active(controller, "day-one")
    observed = []
    for day_index in range(1, 7):
        evidence = controller.archive_and_stage_after_day(
            day_index=day_index
        )
        observed.append(evidence.next_day_source_day_index)
        if day_index < 6:
            (controller.active_scope_dir / "new-owner.json").write_text(
                f"arm-day-{day_index + 1}",
                encoding="utf-8",
            )
    assert tuple(observed) == SEVEN_DAY_SHUFFLED_SOURCE_DAYS
    assert tuple(observed) != tuple(sorted(observed))


def test_controller_rejects_paths_outside_evidence_root(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="inside the evidence root"):
        SevenDayFilesystemStateController(
            evidence_root=tmp_path / "evidence",
            active_scope_root=tmp_path / "outside",
            archive_root=tmp_path / "evidence/archive",
            user_id="synthetic-user",
            experiment_arm_label="correct-user-state",
        )


def test_state_control_is_bound_to_process_restart_evidence(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path, arm="correct-user-state")
    calls: list[str] = []

    class Host:
        def start_initial(self) -> str:
            calls.append("start-initial")
            return "instance-1"

        def stop_for_restart(self) -> ServiceProcessStop:
            calls.append("stop")
            assert controller.active_scope_dir.is_dir()
            assert not (tmp_path / "archive-correct-user-state/day-1").exists()
            return ServiceProcessStop(
                previous_instance_id="instance-1",
                persistence_scope_sha256="a" * 64,
            )

        def start_after_restart(self) -> ServiceProcessStart:
            calls.append("start-after-restart")
            assert (tmp_path / "archive-correct-user-state/day-1").is_dir()
            return ServiceProcessStart(
                next_instance_id="instance-2",
                healthcheck_passed=True,
                persistence_scope_sha256="a" * 64,
            )

        def close(self) -> None:
            return None

    lifecycle = StateControlledSubprocessLifecycle(
        host=Host(),
        state_controller=controller,
    )
    assert lifecycle.start_initial() == "instance-1"
    _write_active(controller, "day-one")
    evidence = lifecycle.restart_after_day(day_index=1)
    assert evidence.previous_instance_id == "instance-1"
    assert evidence.next_instance_id == "instance-2"
    assert evidence.state_intervention.next_day_source_day_index == 1
    assert evidence.previous_persistence_scope_sha256 == "a" * 64
    assert evidence.next_persistence_scope_sha256 == "a" * 64
    assert calls == ["start-initial", "stop", "start-after-restart"]


def test_restart_rejects_server_reported_scope_drift(tmp_path: Path) -> None:
    controller = _controller(tmp_path, arm="correct-user-state")

    class Host:
        def start_initial(self) -> str:
            return "instance-1"

        def stop_for_restart(self) -> ServiceProcessStop:
            return ServiceProcessStop(
                previous_instance_id="instance-1",
                persistence_scope_sha256="a" * 64,
            )

        def start_after_restart(self) -> ServiceProcessStart:
            return ServiceProcessStart(
                next_instance_id="instance-2",
                healthcheck_passed=True,
                persistence_scope_sha256="b" * 64,
            )

        def close(self) -> None:
            return None

    lifecycle = StateControlledSubprocessLifecycle(
        host=Host(),
        state_controller=controller,
    )
    lifecycle.start_initial()
    _write_active(controller, "day-one")
    with pytest.raises(RuntimeError, match="persistence scope"):
        lifecycle.restart_after_day(day_index=1)


def test_subprocess_host_rejects_health_endpoint_scope_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Service:
        instance_id = "not-started"

        def replace_instance_id(self, instance_id: str) -> None:
            self.instance_id = instance_id

    class Process:
        def poll(self) -> None:
            return None

    class Response:
        status = 200

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "status": "ok",
                    "persistence_scope_sha256": "b" * 64,
                }
            ).encode("utf-8")

    host = SubprocessSevenDayServiceHost(
        command=("fixture",),
        service=Service(),
        health_url="http://127.0.0.1:1/v1/health",
        expected_persistence_scope_sha256="a" * 64,
        log_dir=tmp_path / "logs",
        cwd=tmp_path,
        startup_timeout_s=1.0,
    )
    host._process = Process()
    monkeypatch.setattr(
        "lifeform_evolution.seven_day_process_host.urllib.request.urlopen",
        lambda *_args, **_kwargs: Response(),
    )

    with pytest.raises(RuntimeError, match="persistence scope drift"):
        host._wait_until_healthy()


def test_custom_mechanism_arm_requires_explicit_state_policy(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="unsupported seven-day experiment arm"):
        SevenDayFilesystemStateController(
            evidence_root=tmp_path,
            active_scope_root=tmp_path / "active-missing-policy",
            archive_root=tmp_path / "archive-missing-policy",
            user_id="synthetic-user",
            experiment_arm_label="gate1-pe-temporal-on-v1",
        )

    controller = SevenDayFilesystemStateController(
        evidence_root=tmp_path,
        active_scope_root=tmp_path / "active-gate1",
        archive_root=tmp_path / "archive-gate1",
        user_id="synthetic-user",
        experiment_arm_label="gate1-pe-temporal-on-v1",
        state_loading_policy="correct-user-state",
    )
    controller.prepare_initial_day()
    _write_active(controller, "gate1-day-one")
    evidence = controller.archive_and_stage_after_day(day_index=1)

    assert evidence.experiment_arm_label == "gate1-pe-temporal-on-v1"
    assert evidence.state_loading_policy == "correct-user-state"
