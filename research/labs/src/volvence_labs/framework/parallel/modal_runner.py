"""Modal cloud GPU runner.

Submits individual probe units to Modal's serverless GPU platform.
Requires: pip install modal && modal token new
Config: configs/cloud/modal.yaml

Usage:
    runner = ModalRunner()
    runner.setup()
    job = runner.submit_unit("refusal-direction-v1", "baseline", seed=0, level="shadow")
    result = runner.poll_job(job)  # blocks until done; updates job.status/result

If MODAL is not installed or token missing, ``setup()`` raises
``CloudRunnerNotConfigured`` with installation instructions.
"""

from __future__ import annotations

import time
import uuid
from typing import Optional

from .base import CloudRunner, CloudRunnerNotConfigured, CloudJob


class ModalRunner(CloudRunner):
    """Modal.com serverless GPU runner."""

    def setup(self) -> None:
        try:
            import modal  # noqa: F401
        except ImportError:
            raise CloudRunnerNotConfigured(
                "Modal not installed.\n"
                "  pip install modal\n"
                "  modal token new\n"
                "  See: https://modal.com/docs/guide"
            )
        try:
            from modal.config import Config
            cfg = Config()
            if not getattr(cfg, "_profile", None):
                raise CloudRunnerNotConfigured(
                    "Modal token not configured.\n  Run: modal token new"
                )
        except CloudRunnerNotConfigured:
            raise
        except Exception:
            pass
        try:
            from . import modal_app
            if modal_app.app is None:
                raise CloudRunnerNotConfigured(
                    "modal_app.app failed to initialize. Check that 'modal' "
                    "is correctly installed and the token is valid."
                )
            self._app = modal_app.app
            self._run_unit_fn = modal_app.run_unit
        except Exception as e:
            raise CloudRunnerNotConfigured(f"Failed to load modal_app: {e}")

    def submit_unit(
        self,
        probe_id: str,
        cell: str,
        seed: int,
        level: str,
        knob_overrides: Optional[dict] = None,
    ) -> CloudJob:
        if not hasattr(self, "_run_unit_fn"):
            self.setup()

        job_id = f"modal_{uuid.uuid4().hex[:12]}"
        job = CloudJob(
            job_id=job_id,
            probe_id=probe_id,
            cell=cell,
            seed=seed,
            status="pending",
        )
        try:
            with self._app.run():
                call = self._run_unit_fn.spawn(
                    probe_id=probe_id,
                    cell=cell,
                    seed=seed,
                    level=level,
                    knob_overrides=knob_overrides or {},
                )
                job.status = "running"
                job._modal_call_id = call.object_id  # type: ignore[attr-defined]
                job._modal_call = call  # type: ignore[attr-defined]
        except Exception as e:
            job.status = "failed"
            job.error = f"submit failed: {e}"
        return job

    def poll_job(self, job: CloudJob, timeout: float = 600.0) -> CloudJob:
        if job.status in ("completed", "failed"):
            return job

        call = getattr(job, "_modal_call", None)
        if call is None:
            job.status = "failed"
            job.error = "no Modal call handle attached to job"
            return job

        try:
            t0 = time.time()
            while time.time() - t0 < timeout:
                try:
                    result = call.get(timeout=5)
                    job.status = "completed"
                    job.result = result
                    return job
                except Exception:
                    time.sleep(2)
            job.status = "failed"
            job.error = f"poll timeout after {timeout}s"
        except Exception as e:
            job.status = "failed"
            job.error = f"poll error: {e}"
        return job

    def cancel_job(self, job: CloudJob) -> None:
        call = getattr(job, "_modal_call", None)
        if call is not None:
            try:
                call.cancel()
            except Exception:
                pass
        job.status = "failed"
        job.error = "cancelled"
