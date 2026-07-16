"""#89 residual: bounded-learned PE write gate replaces the fixed 0.15 threshold.

Contract under test:

- Initial behaviour is byte-identical to the historical fixed threshold
  (magnitude >= 0.15 writes, below does not).
- PE-driven writes are parked and settled two PE turns later against
  realized usefulness inside the store the gate's owner controls:
  retrieved-again entries count as useful and pull the threshold down;
  unused / decayed / deleted entries push it up.
- Drift is clamped to the envelope around the initial threshold
  (rollback = ``reset()`` -> exact historical behaviour).
- The gate readout is published via ``MemorySnapshot.lifecycle_metrics``
  and the threshold survives checkpoint round-trips.
"""

from __future__ import annotations

from volvence_zero.memory import MemoryStore
from volvence_zero.memory.contracts import RetrievalQuery
from volvence_zero.memory.pe_write_gate import (
    PE_WRITE_GATE_INITIAL_THRESHOLD,
    PeWriteGate,
)
from volvence_zero.prediction.error import (
    ActualOutcome,
    PredictedOutcome,
    PredictionError,
    PredictionErrorSnapshot,
)


def _pe_snapshot(*, magnitude: float, turn_index: int = 1) -> PredictionErrorSnapshot:
    return PredictionErrorSnapshot(
        evaluated_prediction=PredictedOutcome(0, 1, 0.6, 0.6, 0.6, 0.6, 0.7, "pred"),
        actual_outcome=ActualOutcome(1, 0.2, 0.1, 0.4, 0.3, "actual"),
        next_prediction=PredictedOutcome(1, 2, 0.5, 0.5, 0.5, 0.5, 0.6, "next"),
        error=PredictionError(
            task_error=-0.1,
            relationship_error=-0.05,
            regime_error=-0.05,
            action_error=-0.05,
            magnitude=magnitude,
            signed_reward=-0.05,
            description="gate probe",
        ),
        turn_index=turn_index,
        bootstrap=False,
        description="pe snapshot",
    )


def _apply(store: MemoryStore, *, magnitude: float, timestamp_ms: int) -> tuple[str, ...]:
    return store.apply_prediction_error_signal(
        prediction_error_snapshot=_pe_snapshot(magnitude=magnitude),
        timestamp_ms=timestamp_ms,
    )


def test_initial_gate_matches_historical_fixed_threshold():
    store = MemoryStore()
    assert store.pe_write_gate.threshold == PE_WRITE_GATE_INITIAL_THRESHOLD

    below = _apply(store, magnitude=0.14, timestamp_ms=1_000)
    assert not any(op.startswith("prediction-error-write:") for op in below)

    at = _apply(store, magnitude=0.15, timestamp_ms=2_000)
    assert any(op.startswith("prediction-error-write:") for op in at)


def test_unused_pe_writes_raise_threshold():
    store = MemoryStore()
    _apply(store, magnitude=0.6, timestamp_ms=1_000)
    # Two more PE turns without ever retrieving the written entry.
    _apply(store, magnitude=0.01, timestamp_ms=2_000)
    ops = _apply(store, magnitude=0.01, timestamp_ms=3_000)
    assert any("write-gate-settle:0+/1-" in op for op in ops)
    assert store.pe_write_gate.threshold > PE_WRITE_GATE_INITIAL_THRESHOLD
    assert store.pe_write_gate.settled_unused_count == 1


def test_retrieved_pe_write_settles_useful_and_lowers_threshold():
    store = MemoryStore()
    write_ops = _apply(store, magnitude=0.6, timestamp_ms=1_000)
    assert any(op.startswith("prediction-error-write:") for op in write_ops)

    # Retrieval touches the written entry -> realized usefulness.
    result = store.retrieve(
        RetrievalQuery(text="prediction error gate probe"),
        timestamp_ms=1_500,
    )
    assert any("prediction_error:" in entry.content for entry in result.entries)

    _apply(store, magnitude=0.01, timestamp_ms=2_000)
    ops = _apply(store, magnitude=0.01, timestamp_ms=3_000)
    assert any("write-gate-settle:1+/0-" in op for op in ops)
    assert store.pe_write_gate.threshold < PE_WRITE_GATE_INITIAL_THRESHOLD
    assert store.pe_write_gate.settled_useful_count == 1


def test_threshold_drift_is_clamped_to_envelope():
    gate = PeWriteGate()
    entries: dict[str, object] = {}
    for turn in range(200):
        gate.begin_turn()
        gate.record_write(entry_id=f"missing-{turn}", strength=0.5)
        gate.settle(entries)  # type: ignore[arg-type]
    assert gate.threshold == PE_WRITE_GATE_INITIAL_THRESHOLD + 0.10

    gate.reset()
    assert gate.threshold == PE_WRITE_GATE_INITIAL_THRESHOLD
    assert gate.pending_count == 0
    assert gate.settled_useful_count == 0
    assert gate.settled_unused_count == 0


def test_deleted_entry_counts_as_unused():
    store = MemoryStore()
    _apply(store, magnitude=0.6, timestamp_ms=1_000)
    written_ids = [
        entry_id
        for entry_id, entry in store._entries.items()
        if entry.content.startswith("prediction_error:")
    ]
    assert len(written_ids) == 1
    store._artifact_store.delete_entry(written_ids[0])

    _apply(store, magnitude=0.01, timestamp_ms=2_000)
    ops = _apply(store, magnitude=0.01, timestamp_ms=3_000)
    assert any("write-gate-settle:0+/1-" in op for op in ops)


def test_snapshot_publishes_gate_readout():
    store = MemoryStore()
    snapshot = store.snapshot(retrieved_entries=())
    metrics = dict(snapshot.lifecycle_metrics)
    assert metrics["pe_write_gate_threshold"] == PE_WRITE_GATE_INITIAL_THRESHOLD
    assert metrics["pe_write_gate_settled_useful"] == 0.0
    assert metrics["pe_write_gate_settled_unused"] == 0.0
    assert metrics["pe_write_gate_pending"] == 0.0


def test_checkpoint_round_trips_learned_threshold():
    store = MemoryStore()
    # Push the threshold off its initial value via an unused settlement.
    _apply(store, magnitude=0.6, timestamp_ms=1_000)
    _apply(store, magnitude=0.01, timestamp_ms=2_000)
    _apply(store, magnitude=0.01, timestamp_ms=3_000)
    learned = store.pe_write_gate.threshold
    assert learned != PE_WRITE_GATE_INITIAL_THRESHOLD

    checkpoint = store.create_checkpoint()
    assert checkpoint.pe_write_gate_threshold == learned

    restored = MemoryStore()
    restored.restore_checkpoint(checkpoint)
    assert restored.pe_write_gate.threshold == learned
    assert restored.pe_write_gate.pending_count == 0


def test_legacy_checkpoint_dict_restores_initial_threshold():
    from volvence_zero.memory.contracts import _reconstruct_checkpoint

    checkpoint = _reconstruct_checkpoint(
        {
            "checkpoint_id": "legacy",
            "entries": [],
            "pending_promotions": [],
            "pending_decays": [],
            "cms_state": None,
            "promotion_threshold": 0.7,
            "semantic_index": [],
        }
    )
    assert checkpoint is not None
    assert checkpoint.pe_write_gate_threshold == PE_WRITE_GATE_INITIAL_THRESHOLD
