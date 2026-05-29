"""Spawn + supervise per-GPU DLaaS pod processes (debt #17 phase 2).

Each :class:`PodSpec` becomes one child process running
:mod:`dlaas_platform_api.pod_server` bound to a GPU (via
``CUDA_VISIBLE_DEVICES``) and a loopback port. The supervisor registers
a :class:`RemoteInstanceManager` per pod into a
:class:`MultiPodLauncher`, so the parent api routes ai_ids to pods.

Process spawning is real (``subprocess.Popen``) but cannot be validated
in a CPU-only / single-GPU CI; the routing it wires
(:class:`MultiPodLauncher` + :class:`RemoteInstanceManager`) is unit
tested with fakes. Operators run this on real multi-GPU hosts.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass

from dlaas_platform_launcher import (
    MultiPodLauncher,
    RemoteInstanceManager,
    RuntimePod,
)


@dataclass(frozen=True)
class PodSpec:
    """One pod = one substrate on one GPU at a loopback port."""

    pod_id: str
    port: int
    substrate_profile: str
    model_id: str = ""
    runtime_backend: str = "transformers"
    mode: str = "shared_frozen"
    device: str = "cuda"
    gpu_id: str = ""
    host: str = "127.0.0.1"
    capacity: int = 64
    runtime_pool_id: str = "default"

    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class PodProcessSupervisor:
    """Spawns pod processes and registers their proxies on a launcher."""

    def __init__(self, specs: tuple[PodSpec, ...]) -> None:
        if not specs:
            raise ValueError("PodProcessSupervisor requires at least one PodSpec")
        self._specs = specs
        self._procs: list[subprocess.Popen] = []

    def build_launcher(self) -> MultiPodLauncher:
        """Spawn pods and return a MultiPodLauncher wired to their proxies."""

        launcher = MultiPodLauncher()
        for spec in self._specs:
            self._spawn(spec)
            launcher.register_pod(
                RuntimePod(
                    runtime_pod_id=spec.pod_id,
                    runtime_pool_id=spec.runtime_pool_id,
                    substrate_profile=spec.substrate_profile,
                    gpu_id=spec.gpu_id,
                    capacity=spec.capacity,
                ),
                RemoteInstanceManager(base_url=spec.base_url()),
            )
        return launcher

    def _spawn(self, spec: PodSpec) -> None:  # pragma: no cover - spawns a process
        env = dict(os.environ)
        if spec.gpu_id:
            env["CUDA_VISIBLE_DEVICES"] = spec.gpu_id
        cmd = [
            sys.executable,
            "-m",
            "dlaas_platform_api.pod_server",
            "--host",
            spec.host,
            "--port",
            str(spec.port),
            "--substrate-profile",
            spec.substrate_profile,
            "--mode",
            spec.mode,
            "--runtime-backend",
            spec.runtime_backend,
            "--model-id",
            spec.model_id,
            "--device",
            spec.device,
        ]
        self._procs.append(subprocess.Popen(cmd, env=env))

    def stop(self) -> None:  # pragma: no cover - process teardown
        for proc in self._procs:
            proc.terminate()
        for proc in self._procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._procs.clear()


def pod_specs_from_env() -> tuple[PodSpec, ...]:
    """Parse pod specs from ``VZ_MULTI_POD_SPECS`` (``;``-separated).

    Each spec: ``pod_id,port,substrate_profile,model_id,gpu_id`` —
    e.g. ``pod0,9101,shared-frozen-persona-lora,Qwen/Qwen2.5-1.5B,0;``
    ``pod1,9102,shared-frozen-persona-lora,Qwen/Qwen2.5-1.5B,1``.
    """

    raw = os.environ.get("VZ_MULTI_POD_SPECS", "").strip()
    if not raw:
        return ()
    specs: list[PodSpec] = []
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        pod_id, port, profile = parts[0], int(parts[1]), parts[2]
        model_id = parts[3] if len(parts) > 3 else ""
        gpu_id = parts[4] if len(parts) > 4 else ""
        specs.append(
            PodSpec(
                pod_id=pod_id,
                port=port,
                substrate_profile=profile,
                model_id=model_id,
                gpu_id=gpu_id,
            )
        )
    return tuple(specs)


__all__ = ["PodProcessSupervisor", "PodSpec", "pod_specs_from_env"]
