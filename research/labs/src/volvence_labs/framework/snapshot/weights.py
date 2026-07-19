"""Model weight storage strategy.

Design (DESIGN.md + model_storage.md):
- Model weights live in .labs/models/<model_sha>/
- CAS stores only the sha + meta reference, NOT the weights themselves.
- HF snapshot_download is the canonical fetch mechanism.
- model_sha = sha256 of the HF revision commit hash (deterministic).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

from ..env import EnvConfig, load_env
from .cas import CASStore, canonical_dumps, sha256_bytes
from .paths import LabsPaths


def model_sha_for_revision(model_id: str, revision: str = "main") -> str:
    """Compute a stable sha for a (model_id, revision) pair.

    This is NOT the sha of the weights themselves (too expensive to compute
    on multi-GB files). It's a content-addressed key derived from the HF
    repo identity + commit hash.
    """
    identity = canonical_dumps({"model_id": model_id, "revision": revision})
    return sha256_bytes(identity)


def ensure_model_downloaded(
    model_id: str,
    *,
    revision: str = "main",
    env: Optional[EnvConfig] = None,
) -> Path:
    """Download model if not present; return local path.

    Uses huggingface_hub.snapshot_download under the hood.
    Falls back to global HF cache (~/.cache/huggingface) if local cache unavailable.
    """
    if env is None:
        env = load_env()

    cache_dir = env.models.hf_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    token = env.models.hf_token
    if token:
        os.environ.setdefault("HF_TOKEN", token)

    from huggingface_hub import snapshot_download

    # Try global HF cache first (no cache_dir override → uses HF_HOME / ~/.cache/huggingface)
    try:
        local_dir = snapshot_download(
            model_id,
            revision=revision,
            token=token,
            local_files_only=True,
        )
        return Path(local_dir)
    except Exception:
        pass

    # Try local project cache
    try:
        local_dir = snapshot_download(
            model_id,
            revision=revision,
            cache_dir=str(cache_dir),
            token=token,
            local_files_only=True,
        )
        return Path(local_dir)
    except Exception:
        pass

    # Not cached anywhere — download to local cache
    local_dir = snapshot_download(
        model_id,
        revision=revision,
        cache_dir=str(cache_dir),
        token=token,
    )
    return Path(local_dir)


def register_model_in_cas(
    store: CASStore,
    model_id: str,
    revision: str = "main",
    local_path: Optional[Path] = None,
) -> str:
    """Register a model reference in CAS (sha + meta only, not weights).

    Returns the model_sha stored in CAS.
    """
    m_sha = model_sha_for_revision(model_id, revision)
    meta = {
        "model_id": model_id,
        "revision": revision,
        "local_path": str(local_path) if local_path else None,
    }
    # Store a small JSON blob as the CAS entry.
    store.put_bytes(
        canonical_dumps(meta),
        kind="model_ref",
        meta={"model_sha": m_sha, "model_id": model_id},
    )
    return m_sha
