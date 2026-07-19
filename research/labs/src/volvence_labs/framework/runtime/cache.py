"""Feature cache: persist extracted features to disk.

Avoids re-running expensive model forward passes. Features are stored as
numpy arrays in .labs/features/<model_sha>/<dataset_sha>/<split>/<idx>.npy.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

from ..env import EnvConfig, load_env


class FeatureCache:
    """Disk-backed feature cache keyed by (model_sha, dataset_sha, split, idx).

    Usage:
        cache = FeatureCache(model_sha="abc123", dataset_sha="def456")
        features = cache.get("train", 42)
        if features is None:
            features = model.extract(...)
            cache.put("train", 42, features)
    """

    def __init__(
        self,
        model_sha: str,
        dataset_sha: str,
        *,
        root: Optional[str] = None,
        env: Optional[EnvConfig] = None,
    ):
        if env is None:
            env = load_env(root)
        base = Path(root or os.environ.get("VOLVENCE_LABS_ROOT", os.getcwd()))
        self._cache_dir = base / ".labs" / "features" / model_sha[:16] / dataset_sha[:16]
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._model_sha = model_sha
        self._dataset_sha = dataset_sha

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    def _path_for(self, split: str, idx: int) -> Path:
        split_dir = self._cache_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        return split_dir / f"{idx:06d}.npy"

    def get(self, split: str, idx: int) -> Optional[np.ndarray]:
        """Get cached features. Returns None if not cached."""
        path = self._path_for(split, idx)
        if path.exists():
            return np.load(path)
        return None

    def put(self, split: str, idx: int, features: np.ndarray) -> None:
        """Cache features to disk."""
        path = self._path_for(split, idx)
        np.save(path, features)

    def get_batch(self, split: str, indices: list[int]) -> dict[int, Optional[np.ndarray]]:
        """Get multiple cached features. Returns {idx: array_or_None}."""
        return {idx: self.get(split, idx) for idx in indices}

    def put_batch(self, split: str, features_dict: dict[int, np.ndarray]) -> None:
        """Cache multiple features."""
        for idx, feat in features_dict.items():
            self.put(split, idx, feat)

    def has(self, split: str, idx: int) -> bool:
        """Check if features are cached."""
        return self._path_for(split, idx).exists()

    def count(self, split: str) -> int:
        """Count cached features for a split."""
        split_dir = self._cache_dir / split
        if not split_dir.exists():
            return 0
        return len(list(split_dir.glob("*.npy")))

    def clear(self, split: Optional[str] = None) -> int:
        """Clear cached features. Returns number of files deleted."""
        import shutil
        if split:
            target = self._cache_dir / split
        else:
            target = self._cache_dir
        if not target.exists():
            return 0
        count = len(list(target.rglob("*.npy")))
        shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        return count
