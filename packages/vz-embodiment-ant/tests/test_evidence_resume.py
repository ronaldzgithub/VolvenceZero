"""Reliable per-seed partial state for resumable ant evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from volvence_ant.evidence.resume import (
    AntResumeStateError,
    SeedPartialStore,
    ant_stage_fingerprint,
)


def _store(tmp_path: Path, *, config: dict | None = None) -> SeedPartialStore:
    fingerprint = ant_stage_fingerprint(
        stage="matched_control",
        config=config or {"ticks": 10, "seeds": (0, 1, 2, 3, 4)},
    )
    return SeedPartialStore(
        results_root=tmp_path,
        stage="matched_control",
        fingerprint=fingerprint,
        requested_seeds=(0, 1, 2, 3, 4),
    )


def test_partial_resume_loads_only_atomically_completed_seeds(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.commit(seed=1, report={"seed": 1, "arms": []})
    store.commit(seed=3, report={"seed": 3, "arms": []})

    completed = store.load()
    remaining = tuple(seed for seed in store.requested_seeds if seed not in completed)
    assert tuple(sorted(completed)) == (1, 3)
    assert remaining == (0, 2, 4)


def test_partial_metadata_mismatch_fails_loudly(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.initialize()
    metadata = store.root / "run.json"
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["requested_seeds"] = [0]
    metadata.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AntResumeStateError, match="metadata mismatch"):
        store.load()


def test_conflicting_seed_partial_is_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.commit(seed=0, report={"seed": 0, "value": 1})
    with pytest.raises(AntResumeStateError, match="conflicting partial"):
        store.commit(seed=0, report={"seed": 0, "value": 2})
