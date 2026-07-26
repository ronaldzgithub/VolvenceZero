"""Reliable per-seed partial state for resumable ant evidence.

This module also owns the resume-compatibility contract shared by the P1
progress journal, the P2 shard journal and the promotion bundle: all three are
archives that an interrupted formal run rehydrates from, so all three must bind
the same four keys (sense schema, input dim, latent dim, ant count) required by
``docs/specs/digital-ant-embodiment.md``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from volvence_ant.evidence.ecology_checkpoint import (
    ecology_checkpoint_compatibility,
)
from volvence_ant.evidence import resume as resume_module
from volvence_ant.evidence.provenance import AntArtifactExistsError
from volvence_ant.evidence.resume import (
    AntResumeStateError,
    SeedPartialStore,
    ant_stage_fingerprint,
)
from volvence_ant.experiments.ecology_curriculum import EcologyCurriculumConfig
from volvence_ant.experiments.ecology_p1 import (
    EcologyP1Config,
    _progress_compatibility as _p1_progress_compatibility,
)
from volvence_ant.experiments.ecology_p2 import (
    EcologyP2Config,
    _progress_compatibility as _p2_progress_compatibility,
)
from volvence_ant.substrate import AntSenseSchema, sense_channels


_REQUIRED_COMPATIBILITY_KEYS = ("sense_schema", "input_dim", "latent_dim", "n_ants")


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


def _racing_writer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    target_name: str,
    intruder: dict,
) -> None:
    """Make a concurrent process win the store's check-then-write race.

    ``SeedPartialStore`` samples ``path.is_file()`` and only then writes, so
    both ``overwrite=False`` writes are unreachable in a single-threaded test
    unless the file appears in that window. This wrapper creates the file
    exactly there, which is the situation the ``overwrite=False`` argument
    exists for -- a second shard process committing the same seed.
    """

    real_write = resume_module.atomic_write_json

    def _write(path: Path, value: object, *, overwrite: bool = True) -> None:
        if path.name == target_name and not path.exists():
            real_write(path, intruder)
        real_write(path, value, overwrite=overwrite)

    monkeypatch.setattr(resume_module, "atomic_write_json", _write)


def test_a_concurrent_commit_of_the_same_seed_is_refused_by_the_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    store.initialize()
    _racing_writer(
        monkeypatch,
        target_name="seed-000002.json",
        intruder={"seed": 2, "writer": "concurrent"},
    )

    with pytest.raises(AntArtifactExistsError, match="refusing to overwrite"):
        store.commit(seed=2, report={"seed": 2, "value": 1})

    # The committed partial that won the race survived byte-for-byte; a
    # committed per-seed partial is immutable evidence.
    committed = store.root / "seeds" / "seed-000002.json"
    assert json.loads(committed.read_text(encoding="utf-8")) == {
        "seed": 2,
        "writer": "concurrent",
    }


def test_a_concurrent_metadata_marker_is_refused_by_the_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    _racing_writer(
        monkeypatch,
        target_name="run.json",
        intruder={"writer": "concurrent"},
    )

    with pytest.raises(AntArtifactExistsError, match="refusing to overwrite"):
        store.initialize()

    assert json.loads((store.root / "run.json").read_text(encoding="utf-8")) == {
        "writer": "concurrent"
    }


def test_stage_fingerprint_refuses_non_finite_configuration(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Out of range float"):
        ant_stage_fingerprint(stage="p1", config={"threshold": math.nan})


def test_p1_resume_compatibility_binds_sense_schema_and_input_dim() -> None:
    compatibility = dict(_p1_progress_compatibility(EcologyP1Config(n_ants=4)))
    for key in _REQUIRED_COMPATIBILITY_KEYS:
        assert key in compatibility, key
    assert compatibility["sense_schema"] == AntSenseSchema.ECOLOGY_V2.value
    assert compatibility["input_dim"] == str(
        len(sense_channels(AntSenseSchema.ECOLOGY_V2))
    )
    assert compatibility["n_ants"] == "4"


def test_p2_shard_compatibility_binds_sense_schema_and_input_dim() -> None:
    compatibility = dict(_p2_progress_compatibility(EcologyP2Config()))
    for key in _REQUIRED_COMPATIBILITY_KEYS:
        assert key in compatibility, key
    assert compatibility["sense_schema"] == AntSenseSchema.ECOLOGY_V2.value
    assert compatibility["input_dim"] == str(
        len(sense_channels(AntSenseSchema.ECOLOGY_V2))
    )


def test_resume_journals_bind_no_less_than_the_promotion_bundle() -> None:
    promotion = dict(
        ecology_checkpoint_compatibility(
            EcologyCurriculumConfig(n_ants=4, temporal_latent_dim=16)
        )
    )
    p1 = dict(_p1_progress_compatibility(EcologyP1Config(n_ants=4)))
    p2 = dict(_p2_progress_compatibility(EcologyP2Config(n_ants=4)))
    for key in _REQUIRED_COMPATIBILITY_KEYS:
        # artifact_kind differs by design; the body-shape binding must not.
        assert p1[key] == promotion[key], key
        assert p2[key] == promotion[key], key
