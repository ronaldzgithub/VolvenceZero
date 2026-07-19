"""Abstract base for cloud GPU runners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


class CloudRunnerNotConfigured(RuntimeError):
    """Raised when a cloud runner is used without proper setup."""
    pass


@dataclass
class CloudJob:
    """Represents a submitted cloud job."""
    job_id: str
    probe_id: str
    cell: str
    seed: int
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[dict] = None
    error: Optional[str] = None


class CloudRunner(ABC):
    """Abstract base class for cloud GPU runners.

    Subclasses implement the actual cloud API calls.
    The runner interface mirrors SequentialRunner/ParallelRunner:
      runner.run(probe_id, profile) -> ExperimentReport
    """

    def __init__(self, config_path: Optional[str] = None):
        self._config_path = config_path
        self._config: dict = {}

    @abstractmethod
    def setup(self) -> None:
        """Validate credentials and prepare the cloud environment.

        Should raise CloudRunnerNotConfigured with setup instructions
        if the environment is not ready.
        """
        ...

    @abstractmethod
    def submit_unit(
        self,
        probe_id: str,
        cell: str,
        seed: int,
        level: str,
        knob_overrides: Optional[dict] = None,
    ) -> CloudJob:
        """Submit a single unit for execution on cloud GPU.

        Returns a CloudJob handle for polling.
        """
        ...

    @abstractmethod
    def poll_job(self, job: CloudJob) -> CloudJob:
        """Poll job status. Updates job.status and job.result in place."""
        ...

    @abstractmethod
    def cancel_job(self, job: CloudJob) -> None:
        """Cancel a running job."""
        ...

    def run(self, probe_id: str, profile: Any) -> Any:
        """Run full experiment on cloud. Default implementation submits all units."""
        self.setup()
        raise CloudRunnerNotConfigured(
            "Cloud runner execution not yet implemented. "
            "Use local SequentialRunner or ParallelRunner instead.\n\n"
            "To set up cloud GPU:\n"
            "  Modal: pip install modal && modal token new\n"
            "  RunPod: pip install runpod && export RUNPOD_API_KEY=xxx\n"
            "See configs/cloud/ for configuration templates."
        )
