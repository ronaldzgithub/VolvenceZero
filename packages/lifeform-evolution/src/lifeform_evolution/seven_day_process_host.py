"""Managed process host for seven-day product-lifecycle evidence."""

from __future__ import annotations

from dataclasses import dataclass
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
class ServiceProcessRestart:
    previous_instance_id: str
    next_instance_id: str
    healthcheck_passed: bool
    persistence_scope_unchanged: bool


class SevenDayServiceHost(Protocol):
    def start_initial(self) -> str: ...

    def restart(self) -> ServiceProcessRestart: ...

    def close(self) -> None: ...


class SubprocessSevenDayServiceHost:
    """Start and restart one argv-only service command without a shell."""

    def __init__(
        self,
        *,
        command: tuple[str, ...],
        service: RebindableSevenDayService,
        health_url: str,
        persistence_scope_id: str,
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
        if not persistence_scope_id.strip():
            raise ValueError("persistence_scope_id must be non-empty")
        if startup_timeout_s <= 0 or stop_timeout_s <= 0:
            raise ValueError("process timeouts must be positive")
        self._command = command
        self._service = service
        self._health_url = health_url
        self._persistence_scope_id = persistence_scope_id
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

    def start_initial(self) -> str:
        if self._process is not None:
            raise RuntimeError("seven-day service host is already running")
        return self._start()

    def restart(self) -> ServiceProcessRestart:
        if self._process is None:
            raise RuntimeError("seven-day service host has not started")
        previous = self._service.instance_id
        scope_before = self._persistence_scope_id
        self._stop()
        next_instance = self._start()
        return ServiceProcessRestart(
            previous_instance_id=previous,
            next_instance_id=next_instance,
            healthcheck_passed=True,
            persistence_scope_unchanged=(
                scope_before == self._persistence_scope_id
            ),
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
            self._wait_until_healthy()
        except (OSError, RuntimeError, TimeoutError):
            if self._process is not None:
                self._terminate_failed_start()
            raise
        self._service.replace_instance_id(instance_id)
        return instance_id

    def _wait_until_healthy(self) -> None:
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
                        return
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
        intervention = self._state_controller.archive_and_stage_after_day(
            day_index=day_index
        )
        process = self._host.restart()
        return ProcessRestartEvidence(
            after_day_index=day_index,
            previous_instance_id=process.previous_instance_id,
            next_instance_id=process.next_instance_id,
            healthcheck_passed=process.healthcheck_passed,
            persistence_scope_unchanged=(
                process.persistence_scope_unchanged
            ),
            state_intervention=intervention,
        )

    def close(self) -> None:
        self._host.close()


__all__ = [
    "RebindableSevenDayService",
    "ServiceProcessRestart",
    "SevenDayServiceHost",
    "StateControlledSubprocessLifecycle",
    "SubprocessSevenDayServiceHost",
]
