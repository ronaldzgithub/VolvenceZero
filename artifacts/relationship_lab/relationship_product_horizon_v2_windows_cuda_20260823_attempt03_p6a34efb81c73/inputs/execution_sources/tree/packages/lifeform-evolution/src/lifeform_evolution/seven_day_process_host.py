"""Managed process host for seven-day product-lifecycle evidence."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import time
from typing import BinaryIO, Mapping, Protocol
import urllib.error
import urllib.request

from lifeform_evolution.seven_day_companion import (
    ProcessRestartEvidence,
)
from lifeform_evolution.seven_day_state_control import (
    SevenDayFilesystemStateController,
)


class RebindableSevenDayService(Protocol):
    @property
    def instance_id(self) -> str: ...

    def replace_instance_id(self, instance_id: str) -> None: ...


@dataclass(frozen=True)
class ServiceProcessStop:
    previous_instance_id: str
    persistence_scope_sha256: str


@dataclass(frozen=True)
class ServiceProcessStart:
    next_instance_id: str
    healthcheck_passed: bool
    persistence_scope_sha256: str


class SevenDayServiceHost(Protocol):
    def start_initial(self) -> str: ...

    def stop_for_restart(self) -> ServiceProcessStop: ...

    def start_after_restart(self) -> ServiceProcessStart: ...

    def close(self) -> None: ...


class SubprocessSevenDayServiceHost:
    """Start and restart one argv-only service command without a shell."""

    def __init__(
        self,
        *,
        command: tuple[str, ...],
        service: RebindableSevenDayService,
        health_url: str,
        expected_persistence_scope_sha256: str,
        log_dir: str | Path,
        cwd: str | Path,
        environment: Mapping[str, str] | None = None,
        startup_timeout_s: float = 120.0,
        stop_timeout_s: float = 20.0,
    ) -> None:
        if not command or any(not part for part in command):
            raise ValueError("service command must be a non-empty argv tuple")
        if not health_url.strip():
            raise ValueError("health_url must be non-empty")
        if (
            len(expected_persistence_scope_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_persistence_scope_sha256
            )
        ):
            raise ValueError(
                "expected_persistence_scope_sha256 must be a SHA-256"
            )
        if startup_timeout_s <= 0 or stop_timeout_s <= 0:
            raise ValueError("process timeouts must be positive")
        self._command = command
        self._service = service
        self._health_url = health_url
        self._expected_persistence_scope_sha256 = (
            expected_persistence_scope_sha256
        )
        self._log_dir = Path(log_dir)
        self._cwd = Path(cwd)
        self._environment = (
            dict(environment) if environment is not None else os.environ.copy()
        )
        self._startup_timeout_s = startup_timeout_s
        self._stop_timeout_s = stop_timeout_s
        self._process: subprocess.Popen[bytes] | None = None
        self._log_handle: BinaryIO | None = None
        self._generation = 0
        self._last_health_scope_sha256: str | None = None

    def start_initial(self) -> str:
        if self._process is not None:
            raise RuntimeError("seven-day service host is already running")
        return self._start()

    def stop_for_restart(self) -> ServiceProcessStop:
        if self._process is None:
            raise RuntimeError("seven-day service host has not started")
        previous = self._service.instance_id
        scope_before = self._last_health_scope_sha256
        if scope_before is None:
            raise RuntimeError("seven-day service lacks health scope evidence")
        self._stop()
        return ServiceProcessStop(
            previous_instance_id=previous,
            persistence_scope_sha256=scope_before,
        )

    def start_after_restart(self) -> ServiceProcessStart:
        if self._process is not None:
            raise RuntimeError("seven-day service host is already running")
        next_instance = self._start()
        scope_after = self._last_health_scope_sha256
        if scope_after is None:
            raise RuntimeError("restarted service lacks health scope evidence")
        return ServiceProcessStart(
            next_instance_id=next_instance,
            healthcheck_passed=(
                scope_after == self._expected_persistence_scope_sha256
            ),
            persistence_scope_sha256=scope_after,
        )

    def close(self) -> None:
        if self._process is not None:
            self._stop()

    def _start(self) -> str:
        self._generation += 1
        self._log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._log_dir / f"service-{self._generation}.log"
        self._log_handle = log_path.open("wb")
        try:
            self._process = subprocess.Popen(
                self._command,
                cwd=self._cwd,
                env=self._environment,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                shell=False,
            )
            instance_id = (
                f"service-generation-{self._generation}-pid-"
                f"{self._process.pid}"
            )
            health_scope_sha256 = self._wait_until_healthy()
        except (OSError, RuntimeError, TimeoutError):
            if self._process is not None:
                self._terminate_failed_start()
            raise
        self._service.replace_instance_id(instance_id)
        self._last_health_scope_sha256 = health_scope_sha256
        return instance_id

    def _wait_until_healthy(self) -> str:
        assert self._process is not None
        deadline = time.monotonic() + self._startup_timeout_s
        last_error = "health endpoint did not respond"
        while time.monotonic() < deadline:
            exit_code = self._process.poll()
            if exit_code is not None:
                raise RuntimeError(
                    f"seven-day service exited during startup: {exit_code}"
                )
            try:
                with urllib.request.urlopen(
                    self._health_url,
                    timeout=min(2.0, self._startup_timeout_s),
                ) as response:
                    if 200 <= response.status < 300:
                        raw = response.read()
                        try:
                            payload = json.loads(raw.decode("utf-8"))
                        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                            raise RuntimeError(
                                "seven-day health response is not valid JSON"
                            ) from exc
                        if not isinstance(payload, Mapping):
                            raise RuntimeError(
                                "seven-day health response must be an object"
                            )
                        scope = payload.get("persistence_scope_sha256")
                        if payload.get("status") != "ok":
                            raise RuntimeError(
                                "seven-day health response status is not ok"
                            )
                        if scope != self._expected_persistence_scope_sha256:
                            raise RuntimeError(
                                "seven-day service health persistence scope drift"
                            )
                        assert isinstance(scope, str)
                        return scope
                    last_error = f"health endpoint returned {response.status}"
            except urllib.error.URLError as exc:
                last_error = str(exc.reason)
            time.sleep(0.1)
        raise TimeoutError(
            "seven-day service health check timed out: " + last_error
        )

    def _stop(self) -> None:
        assert self._process is not None
        process = self._process
        process.terminate()
        try:
            process.wait(timeout=self._stop_timeout_s)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=self._stop_timeout_s)
        self._process = None
        self._last_health_scope_sha256 = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def _terminate_failed_start(self) -> None:
        assert self._process is not None
        process = self._process
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=self._stop_timeout_s)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=self._stop_timeout_s)
        self._process = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None


class StateControlledSubprocessLifecycle:
    """Compose exact filesystem interventions with real process restart."""

    def __init__(
        self,
        *,
        host: SevenDayServiceHost,
        state_controller: SevenDayFilesystemStateController,
    ) -> None:
        self._host = host
        self._state_controller = state_controller

    def start_initial(self) -> str:
        self._state_controller.prepare_initial_day()
        return self._host.start_initial()

    def restart_after_day(self, *, day_index: int) -> ProcessRestartEvidence:
        stopped = self._host.stop_for_restart()
        intervention = self._state_controller.archive_and_stage_after_day(
            day_index=day_index
        )
        started = self._host.start_after_restart()
        scope_unchanged = (
            stopped.persistence_scope_sha256
            == started.persistence_scope_sha256
        )
        return ProcessRestartEvidence(
            after_day_index=day_index,
            previous_instance_id=stopped.previous_instance_id,
            next_instance_id=started.next_instance_id,
            healthcheck_passed=started.healthcheck_passed,
            persistence_scope_unchanged=scope_unchanged,
            previous_persistence_scope_sha256=(
                stopped.persistence_scope_sha256
            ),
            next_persistence_scope_sha256=started.persistence_scope_sha256,
            state_intervention=intervention,
        )

    def close(self) -> None:
        self._host.close()


__all__ = [
    "RebindableSevenDayService",
    "ServiceProcessStart",
    "ServiceProcessStop",
    "SevenDayServiceHost",
    "StateControlledSubprocessLifecycle",
    "SubprocessSevenDayServiceHost",
]
