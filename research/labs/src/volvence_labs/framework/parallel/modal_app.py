"""Modal cloud GPU app for volvence-labs probes.

Defines remote functions that run a single probe unit on a GPU. Used by
``ModalRunner.submit_unit()`` to dispatch work to the cloud.

Usage (local CLI test):

    pip install modal && modal token new
    modal run -m volvence_labs.framework.parallel.modal_app::run_unit \\
        --probe-id refusal-direction-v1 --cell baseline --seed 0

Usage (programmatic):

    from volvence_labs.framework.parallel import ModalRunner
    runner = ModalRunner()
    runner.setup()
    job = runner.submit_unit("refusal-direction-v1", "baseline", 0, "shadow")
    while job.status not in ("completed", "failed"):
        job = runner.poll_job(job)
        time.sleep(2)
    print(job.result)

Environment variables:
    HF_TOKEN: huggingface token (set as Modal secret ``volvence-hf``)
    VOLVENCE_LABS_GPU: GPU type (default "T4"; override "A10G", "H100", etc.)

Cost note: T4 is ~$0.59/hr on Modal; A10G is ~$1.10/hr. Per-unit run for
a 7B model on shadow profile is ~10s, so 288 units ~ $0.05-$0.10.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Optional

# Modal is an optional dependency; gate all imports.
try:
    import modal
    _MODAL_AVAILABLE = True
except ImportError:
    _MODAL_AVAILABLE = False
    modal = None  # type: ignore


def _build_image() -> Any:
    """Construct the Modal image with our dependencies + repo source."""
    if not _MODAL_AVAILABLE:
        return None
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch>=2.2",
            "transformers>=4.40",
            "numpy>=1.26",
            "safetensors",
            "huggingface_hub[hf_transfer]",
            "scipy",
            "scikit-learn",
        )
        .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
        .add_local_python_source("volvence_labs")
    )
    return image


_GPU_TYPE = os.environ.get("VOLVENCE_LABS_GPU", "T4")


if _MODAL_AVAILABLE:
    app = modal.App("volvence-labs-runner")
    image = _build_image()
    model_cache = modal.Volume.from_name("volvence-hf-cache", create_if_missing=True)

    @app.function(
        image=image,
        gpu=_GPU_TYPE,
        volumes={"/root/.cache/huggingface": model_cache},
        secrets=[modal.Secret.from_name("volvence-hf", required_keys=["HF_TOKEN"])],
        timeout=3600,
        retries=2,
    )
    def run_unit(
        probe_id: str,
        cell: str,
        seed: int,
        level: str = "shadow",
        knob_overrides: Optional[dict] = None,
    ) -> dict:
        """Execute one probe unit on a Modal GPU and return JSON-serializable result."""
        import volvence_labs.probes  # noqa: F401  registers all probes
        from volvence_labs.framework.probe import get_registry, ProbeContext
        from volvence_labs.framework.scheduler.runner import _run_unit
        from volvence_labs.framework.wiring import AblationCell, WiringLevel

        unit_report = _run_unit(
            probe_id=probe_id,
            cell=AblationCell(cell),
            seed=seed,
            level=WiringLevel(level),
            knob_overrides=knob_overrides or {},
        )
        return {
            "ok": unit_report.ok,
            "cell": unit_report.cell,
            "seed": unit_report.seed,
            "level": unit_report.level,
            "metrics": dict(unit_report.metrics),
            "duration_s": unit_report.duration_s,
            "run_id": unit_report.run_id,
            "error": unit_report.error,
        }

else:
    app = None
    run_unit = None
