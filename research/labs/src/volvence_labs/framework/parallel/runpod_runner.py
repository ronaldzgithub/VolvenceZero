"""RunPod serverless GPU runner (stub).

Requires: pip install runpod && export RUNPOD_API_KEY=xxx
Config: configs/cloud/runpod.yaml
"""

from __future__ import annotations

from typing import Optional

from .base import CloudRunner, CloudRunnerNotConfigured, CloudJob


class RunPodRunner(CloudRunner):
    """RunPod serverless GPU runner.

    NOT YET FUNCTIONAL — requires RunPod account and API key.
    This stub provides the interface and setup instructions.
    """

    def setup(self) -> None:
        try:
            import runpod  # noqa: F401
        except ImportError:
            raise CloudRunnerNotConfigured(
                "runpod not installed.\n"
                "  pip install runpod\n"
                "  export RUNPOD_API_KEY=xxx\n"
                "  See: https://docs.runpod.io/serverless"
            )

        import os
        if not os.environ.get("RUNPOD_API_KEY"):
            raise CloudRunnerNotConfigured(
                "RUNPOD_API_KEY not set.\n"
                "  export RUNPOD_API_KEY=your_api_key\n"
                "  Get key from: https://www.runpod.io/console/user/settings"
            )

    def submit_unit(
        self,
        probe_id: str,
        cell: str,
        seed: int,
        level: str,
        knob_overrides: Optional[dict] = None,
    ) -> CloudJob:
        raise CloudRunnerNotConfigured(
            "RunPodRunner.submit_unit() not yet implemented.\n"
            "Planned: runpod.serverless.run() with probe_id/cell/seed payload."
        )

    def poll_job(self, job: CloudJob) -> CloudJob:
        raise CloudRunnerNotConfigured("RunPodRunner.poll_job() not yet implemented.")

    def cancel_job(self, job: CloudJob) -> None:
        raise CloudRunnerNotConfigured("RunPodRunner.cancel_job() not yet implemented.")
