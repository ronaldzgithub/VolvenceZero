"""Environment configuration loader.

Reads configs/env.yaml and provides typed access to model paths, GPU policy, etc.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class ModelsConfig:
    cache_dir: Path
    hf_cache_dir: Path
    hf_token: Optional[str]
    default_llm: str
    default_vision: str


@dataclass(frozen=True)
class GpuConfig:
    policy: str  # "cpu" | "cuda" | "mps" | "auto"
    prefer: tuple[str, ...]


@dataclass(frozen=True)
class SchedulerConfig:
    max_workers: int
    unit_timeout: int


@dataclass(frozen=True)
class EnvConfig:
    models: ModelsConfig
    gpu: GpuConfig
    scheduler: SchedulerConfig

    def resolve_device(self) -> str:
        """Return the torch device string based on gpu policy."""
        if self.gpu.policy != "auto":
            return self.gpu.policy
        try:
            import torch
            for dev in self.gpu.prefer:
                if dev == "cuda" and torch.cuda.is_available():
                    return "cuda"
                if dev == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
                if dev == "cpu":
                    return "cpu"
        except ImportError:
            pass
        return "cpu"


def load_env(root: Optional[os.PathLike | str] = None) -> EnvConfig:
    """Load env config from configs/env.yaml relative to root."""
    import yaml

    if root is None:
        root = os.environ.get("VOLVENCE_LABS_ROOT", os.getcwd())
    root = Path(root)
    env_path = root / "configs" / "env.yaml"

    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
    else:
        data = {}

    models_data = data.get("models", {})
    gpu_data = data.get("gpu", {})
    sched_data = data.get("scheduler", {})

    hf_token = models_data.get("hf_token") or os.environ.get("HF_TOKEN")
    defaults = models_data.get("defaults", {})

    return EnvConfig(
        models=ModelsConfig(
            cache_dir=root / models_data.get("cache_dir", ".labs/models"),
            hf_cache_dir=root / models_data.get("hf_cache_dir", ".labs/hf_cache"),
            hf_token=hf_token,
            default_llm=defaults.get("llm", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
            default_vision=defaults.get("vision", "facebook/ijepa_vits16"),
        ),
        gpu=GpuConfig(
            policy=gpu_data.get("policy", "cpu"),
            prefer=tuple(gpu_data.get("prefer", ["cuda", "mps", "cpu"])),
        ),
        scheduler=SchedulerConfig(
            max_workers=int(sched_data.get("max_workers", 4)),
            unit_timeout=int(sched_data.get("unit_timeout", 3600)),
        ),
    )
