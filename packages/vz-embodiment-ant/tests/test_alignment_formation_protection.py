"""L1-B formation-protection contract and formal precheck locks."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from volvence_zero.integration import FinalRolloutConfig

from volvence_ant.experiments.ecology_probe import (
    EcologyProbeBackendLane,
    run_ecology_action_probes,
)


_ROOT = Path(__file__).resolve().parents[3]


def test_formal_precheck_binds_source_and_keeps_station_blocked() -> None:
    artifact_path = (
        _ROOT
        / "research/ant/results/ecology_recovery/same_physics_baseline/"
        "alignment_formation_protection_precheck.v1.json"
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == (
        "alignment-formation-protection-precheck.v1"
    )
    assert payload["status"] == "PRECHECK_PASS"
    assert payload["read_only"] is True
    assert payload["training_or_journal_write_performed"] is False
    source = payload["source"]["review_report"]
    source_path = _ROOT / source["path"]
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source[
        "sha256"
    ]
    mechanism = payload["mechanism"]
    assert mechanism["domain_semantics_consumed"] is False
    assert mechanism["wiring"] == "active"
    assert mechanism["max_update_steps"] == 160
    assert mechanism["conflict_scale"] == 0.25
    assert mechanism["rollback"] == {
        "wiring": "disabled",
        "max_update_steps": 0,
        "conflict_scale": 1.0,
    }
    probe = payload["probe"]
    assert probe["learning_writes_enabled"] is False
    assert probe["active_digest"] == probe["disabled_digest"]
    assert probe["byte_equivalent_forward"] is True
    assert probe["max_frozen_action_head_update_step"] < 160
    assert [
        row["body_id"]
        for row in probe["food_rows"]
        if row["target_aligned"] is False
    ] == [2]
    assert payload["decision"]["l1c_preregistration_may_be_created"] is True
    assert payload["decision"]["station_run_authorized"] is False
    assert payload["decision"]["station2_remains_unauthorized"] is True


def test_probe_refuses_ambiguous_backend_and_rollout_configuration() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        asyncio.run(
            run_ecology_action_probes(
                temporal_latent_dim=16,
                seed=700_003,
                backend_lane=EcologyProbeBackendLane.PURE,
                rollout_config=FinalRolloutConfig(),
            )
        )
