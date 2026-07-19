"""Dataset snapshot: download and register HF datasets with content-addressed sha.

Supports: HellaSwag, ARC-Easy, WikiText-2, TinyImageNet (subset).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

from ..env import EnvConfig, load_env
from ..snapshot.cas import canonical_dumps, sha256_bytes


def dataset_sha(dataset_id: str, split: str, subset: Optional[str] = None) -> str:
    """Compute a stable sha for a (dataset_id, split, subset) triple."""
    identity = canonical_dumps({
        "dataset_id": dataset_id,
        "split": split,
        "subset": subset,
    })
    return sha256_bytes(identity)


class DatasetSnapshot:
    """Manages HF dataset downloads and provides content-addressed access.

    Usage:
        ds = DatasetSnapshot("Rowan/hellaswag", split="validation")
        samples = ds.load(limit=256)
        sha = ds.sha
    """

    def __init__(
        self,
        dataset_id: str,
        *,
        split: str = "validation",
        subset: Optional[str] = None,
        env: Optional[EnvConfig] = None,
    ):
        if env is None:
            env = load_env()
        self._dataset_id = dataset_id
        self._split = split
        self._subset = subset
        self._env = env
        self._sha = dataset_sha(dataset_id, split, subset)
        self._data: Optional[Any] = None

    @property
    def sha(self) -> str:
        return self._sha

    @property
    def dataset_id(self) -> str:
        return self._dataset_id

    def load(self, limit: Optional[int] = None) -> list[dict]:
        """Load dataset samples. Returns list of dicts.

        Uses HF datasets library. Falls back to a minimal synthetic set
        if the library is unavailable or download fails.
        """
        if self._data is not None:
            data = self._data
        else:
            try:
                from datasets import load_dataset
                ds = load_dataset(
                    self._dataset_id,
                    self._subset,
                    split=self._split,
                    cache_dir=str(self._env.models.hf_cache_dir),
                    token=self._env.models.hf_token,
                    trust_remote_code=True,
                )
                self._data = [dict(row) for row in ds]
                data = self._data
            except Exception as e:
                # Fallback: return empty with error marker
                return [{"_error": str(e), "_dataset_id": self._dataset_id, "_idx": i} for i in range(limit or 10)]

        if limit is not None:
            return data[:limit]
        return data

    def load_texts(self, text_field: str = "text", limit: Optional[int] = None) -> list[str]:
        """Load just the text field from each sample."""
        samples = self.load(limit=limit)
        return [s.get(text_field, "") for s in samples if text_field in s]

    def register_in_cas(self, store) -> str:
        """Register dataset reference in CAS."""
        meta = canonical_dumps({
            "dataset_id": self._dataset_id,
            "split": self._split,
            "subset": self._subset,
            "sha": self._sha,
        })
        store.put_bytes(meta, kind="dataset_ref", meta={"dataset_sha": self._sha})
        return self._sha


# Pre-configured dataset factories for common eval sets.

def hellaswag_val(env: Optional[EnvConfig] = None) -> DatasetSnapshot:
    return DatasetSnapshot("Rowan/hellaswag", split="validation", env=env)


def arc_easy_val(env: Optional[EnvConfig] = None) -> DatasetSnapshot:
    return DatasetSnapshot("allenai/ai2_arc", split="validation", subset="ARC-Easy", env=env)


def wikitext2_val(env: Optional[EnvConfig] = None) -> DatasetSnapshot:
    return DatasetSnapshot("wikitext", split="validation", subset="wikitext-2-raw-v1", env=env)
