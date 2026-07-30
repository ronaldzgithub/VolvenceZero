from __future__ import annotations

import json

from volvence_zero.agent.gate11_longitudinal_source import (
    GATE11_LONGITUDINAL_SOURCE_SEEDS,
)
from volvence_zero.agent.gate11_per_user_continuity_evidence import (
    _confidence_interval_95 as gate11_confidence_interval,
)
from volvence_zero.agent.gate5_cms_pareto_evidence import (
    GATE5_ARM_NAMES,
    GATE5_FULL_ARM,
    GATE5_SINGLE_TIMESCALE_ARM,
    Gate5ArmMetrics,
)
from volvence_zero.agent.gate5_longitudinal_evidence import (
    compare_gate5_longitudinal_arms,
)
from volvence_zero.memory import MemoryWriteRequest, Track, MemoryStratum
from volvence_zero.memory.persistence import FileSystemPersistenceBackend
from volvence_zero.semantic_state import (
    SemanticProposal,
    SemanticProposalOperation,
)
from volvence_zero.agent.gate11_per_user_continuity_evidence import (
    _build_state_owners,
    _persist_state_owners,
    reconcile_gate11_preregistered_verdict,
)


def _metric(
    *,
    seed: int,
    arm: str,
    absorption: float,
    retention: float,
) -> Gate5ArmMetrics:
    cadence = (
        (1, 2, 4)
        if arm == GATE5_FULL_ARM
        else (1, 1, 1)
        if arm == GATE5_SINGLE_TIMESCALE_ARM
        else ()
    )
    return Gate5ArmMetrics(
        seed=seed,
        arm=arm,
        settled_transition_count=510,
        locked_transition_count=60,
        new_knowledge_absorption=absorption,
        old_knowledge_retention=retention,
        memory_churn=0.1,
        erroneous_promotion_rate=0.0,
        retrieval_hit_rate=1.0,
        retrieval_weighted_payoff=0.2,
        cms_parameter_count=204 if arm != "memory-only" else 0,
        cadence_intervals=cadence,
        frozen_substrate_mutation_count=0,
        lineage_complete=True,
    )


def test_gate11_confidence_interval_enforces_cross_seed_direction() -> None:
    lower, upper = gate11_confidence_interval((0.5, 0.5, 0.5))
    assert lower == upper == 0.5
    noisy_lower, _ = gate11_confidence_interval((0.5, 0.5, -0.1))
    assert noisy_lower < 0.0


def test_gate11_owner_state_roundtrips_through_filesystem(tmp_path) -> None:
    store, semantic, hydration, memory_loaded, semantic_loaded = (
        _build_state_owners(root=tmp_path, load=False)
    )
    assert memory_loaded is False
    assert semantic_loaded is False
    store.write(
        MemoryWriteRequest(
            content="opaque callback",
            track=Track.WORLD,
            stratum=MemoryStratum.EPISODIC,
            tags=("opaque",),
            subject_ids=("user-a",),
            audience_ids=("self",),
        ),
        timestamp_ms=1,
    )
    semantic.apply(
        slot="commitment",
        proposals=(
            SemanticProposal(
                proposal_id="opaque-commitment",
                target_slot="commitment",
                operation=SemanticProposalOperation.CREATE,
                summary="opaque summary",
                detail="opaque detail",
                evidence="typed-test-event",
                confidence=1.0,
                control_signal=0.5,
            ),
        ),
        turn_index=1,
    )
    _persist_state_owners(
        store=store,
        semantic_store=semantic,
        hydration=hydration,
    )
    restored_store, restored_semantic, _, memory_loaded, semantic_loaded = (
        _build_state_owners(root=tmp_path, load=True)
    )
    assert memory_loaded is True
    assert semantic_loaded is True
    assert restored_store.entry_count() == 1
    assert restored_semantic.records_for("commitment")[-1].summary == (
        "opaque summary"
    )
    backend = FileSystemPersistenceBackend(base_dir=str(tmp_path))
    assert set(backend.list_checkpoints(prefix="")) >= {
        "memory/store",
        "owner_hydration/semantic_state",
    }


def test_gate5_longitudinal_comparison_requires_effect_and_ci() -> None:
    metrics = []
    for seed in GATE11_LONGITUDINAL_SOURCE_SEEDS:
        for arm in GATE5_ARM_NAMES:
            metrics.append(
                _metric(
                    seed=seed,
                    arm=arm,
                    absorption=(
                        0.55 if arm == GATE5_FULL_ARM else 0.50
                    ),
                    retention=0.90,
                )
            )
    _, confidence, gates = compare_gate5_longitudinal_arms(metrics)
    assert gates["full_pareto_non_worse_all_controls"] is True
    assert gates["full_significant_vs_single_timescale"] is True
    assert confidence[GATE5_SINGLE_TIMESCALE_ARM][
        "absorption_confidence_interval_95"
    ][0] > 0.0


def test_gate11_reconciliation_removes_only_unregistered_gate(
    tmp_path,
) -> None:
    source = tmp_path / "v1"
    source.mkdir()
    for name in (
        "predictions.jsonl",
        "outcomes.jsonl",
        "prediction_errors.jsonl",
        "segments.jsonl",
        "credit.jsonl",
        "state_diff.jsonl",
    ):
        (source / name).write_text("{}\n", encoding="utf-8")
    (source / "manifest.yaml").write_text(
        '{"schema_version":"gate11-per-user-continuity.v1"}',
        encoding="utf-8",
    )
    (source / "ablation_results.json").write_text(
        """
        {
          "gates": {
            "correct_state_consistency_perfect": false,
            "correct_vs_stateless_effect": true
          },
          "seed_metrics": [
            {
              "arm": "correct-user-state",
              "continuity_composite": 0.75,
              "callback_consistency": 0.25
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    (source / "rollback_evidence.json").write_text(
        '{"passed":true}',
        encoding="utf-8",
    )
    (source / "promotion_verdict.json").write_text(
        '{"status":"not-supported","failed_gates":["correct_state_consistency_perfect"]}',
        encoding="utf-8",
    )
    target = tmp_path / "v2"
    reconcile_gate11_preregistered_verdict(
        source_bundle=source,
        output_dir=target,
    )
    verdict = json.loads(
        (target / "promotion_verdict.json").read_text(encoding="utf-8")
    )
    assert verdict["status"] == "longitudinal-supported"
    assert verdict["locked_arm_rerun_count"] == 0
