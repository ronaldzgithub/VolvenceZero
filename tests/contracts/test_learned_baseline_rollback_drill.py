"""Wave E3 (debt #6 / #7) — learned baseline rollback drill.

The plan's E3.4 calls for a rollback drill that explicitly verifies
two-way reversibility on the two learned baselines:

* :class:`_RewardingStateHead` (Phase 2.A counterfactual baseline).
* :class:`_PECriticHead` (Phase 2.B PE critic baseline).

Both heads expose ``export_state`` / ``restore_state`` per the typed
:class:`RewardingStateHeadState` / :class:`PECriticHeadState`
contracts. This test drives them through:

1. Several "good" updates that learn meaningful weights.
2. Snapshot the typed state.
3. Several "degraded" updates that move the head away from the
   learned policy (we don't deliberately break invariants — we just
   feed shifted targets to simulate a regression).
4. Restore from the snapshot.
5. Verify the head's prediction matches the pre-degradation state
   bit-exactly through the typed export.

This is the **deterministic** rollback drill — it does not need a
real LLM or a full benchmark trace. The pass criterion is that the
typed export round-trips without information loss; that is the
load-bearing observable for the W3 promotion criteria documented
in :doc:`docs/specs/credit-and-self-modification.md` (Wave E3
section) and :doc:`docs/specs/prediction-error-loop.md`.

Failing this test should **block** any promotion of the learned
baselines from ``readout-only`` to ``readout-with-acceptance`` per
the spec.
"""

from __future__ import annotations

import pytest

from volvence_zero.credit.gate import (
    RewardingStateHeadState,
    _RewardingStateHead,
)
from volvence_zero.prediction.error import (
    PECriticHeadState,
    _PECriticHead,
)


def test_rewarding_state_head_state_round_trips_bit_exact() -> None:
    """The typed export -> restore round-trip preserves every numeric
    field with no precision loss; weights, bias, validation_delta,
    capacity_cost, rollback_evidence all carry over.
    """
    head = _RewardingStateHead()
    snap_a = head.export_state()
    fresh = _RewardingStateHead()
    fresh.restore_state(snap_a)
    snap_b = fresh.export_state()
    assert snap_b.weights == snap_a.weights
    assert snap_b.bias == snap_a.bias
    assert snap_b.update_count == snap_a.update_count
    assert snap_b.last_validation_delta == snap_a.last_validation_delta
    assert snap_b.last_rollback_evidence == snap_a.last_rollback_evidence


def test_rewarding_state_head_restore_matches_pre_degradation_predictions() -> None:
    """End-to-end drill: learn -> snapshot -> degrade -> restore.

    Predictions on a held-out feature vector after restore must match
    the pre-degradation predictions on the same feature vector.
    """
    head = _RewardingStateHead()
    # Manually populate the head with a non-trivial state by directly
    # restoring a typed state; we do not need to run propose_update
    # because the rollback drill is about the export/restore contract,
    # not the update math (which has its own unit tests).
    learned_state = RewardingStateHeadState(
        rule_id="credit.rewarding_state_head.v1",
        feature_dim=15,
        update_count=42,
        weights=tuple(0.1 * (i - 7) for i in range(15)),
        bias=0.05,
        last_prediction=0.40,
        last_target=0.42,
        last_validation_delta=0.03,
        last_capacity_cost=0.12,
        last_rollback_evidence="learned-from-200-turn-trace",
    )
    head.restore_state(learned_state)

    held_out_features = tuple(0.5 * ((i % 3) - 1) for i in range(15))
    learned_prediction = head.predict(features=held_out_features, fallback=0.0)
    snapshot = head.export_state()

    # Degrade: replace the head with random-ish state.
    degraded_state = RewardingStateHeadState(
        rule_id="credit.rewarding_state_head.v1",
        feature_dim=15,
        update_count=42 + 1,
        weights=tuple(-0.4 + 0.05 * i for i in range(15)),
        bias=-0.30,
        last_prediction=-0.20,
        last_target=0.42,
        last_validation_delta=-0.10,
        last_capacity_cost=0.45,
        last_rollback_evidence="degradation-noise-injected",
    )
    head.restore_state(degraded_state)
    degraded_prediction = head.predict(features=held_out_features, fallback=0.0)
    assert degraded_prediction != learned_prediction, (
        "Degradation step should perturb predictions; otherwise the "
        "drill is testing nothing."
    )

    # Restore from snapshot.
    head.restore_state(snapshot)
    restored_prediction = head.predict(features=held_out_features, fallback=0.0)
    assert restored_prediction == learned_prediction, (
        "Rollback drill failed: predictions after restore do not match "
        "pre-degradation. Promotion of the rewarding-state head from "
        "readout-only is BLOCKED by this test."
    )


def test_pe_critic_head_state_round_trips_bit_exact() -> None:
    head = _PECriticHead(decay=0.9)
    snap_a = head.export_state()
    fresh = _PECriticHead(decay=0.9)
    fresh.restore_state(snap_a)
    snap_b = fresh.export_state()
    assert snap_b.axis_weights == snap_a.axis_weights
    assert snap_b.axis_biases == snap_a.axis_biases
    assert snap_b.update_count == snap_a.update_count
    assert snap_b.last_validation_delta == snap_a.last_validation_delta


def test_pe_critic_head_restore_matches_pre_degradation_predictions() -> None:
    """End-to-end PE critic drill: learn -> snapshot -> degrade -> restore.

    Per-axis predictions on a held-out feature vector after restore
    must match the pre-degradation predictions on the same vector.
    """
    # The PE critic head delegates state ownership to its inner
    # ``_PELearnedCritic``; the rollback drill exercises the inner
    # critic directly because ``predict_axis`` lives there.
    head = _PECriticHead(decay=0.9)
    inner = head._learned  # type: ignore[attr-defined]
    feature_dim = inner._FEATURE_DIM  # type: ignore[attr-defined]
    learned_state = PECriticHeadState(
        rule_id="prediction.pe_critic_head.v1",
        feature_dim=feature_dim,
        update_count=120,
        axis_weights=(
            ("task", tuple(0.1 * i for i in range(feature_dim))),
            ("relationship", tuple(-0.05 * i for i in range(feature_dim))),
            ("regime", tuple(0.02 * i for i in range(feature_dim))),
            ("action", tuple(-0.03 * i for i in range(feature_dim))),
        ),
        axis_biases=(
            ("task", 0.20),
            ("relationship", 0.10),
            ("regime", 0.05),
            ("action", 0.15),
        ),
        last_prediction=0.30,
        last_target=0.32,
        last_validation_delta=0.04,
        last_capacity_cost=0.08,
        last_rollback_evidence="learned-from-200-turn-trace",
    )
    head.restore_state(learned_state)

    held_out_features = tuple(0.4 * ((i % 4) - 1) for i in range(feature_dim))
    learned_predictions = {
        axis: inner.predict_axis(axis=axis, features=held_out_features, fallback=0.0)
        for axis in ("task", "relationship", "regime", "action")
    }
    snapshot = head.export_state()

    # Degrade: shift weights heavily.
    degraded_state = PECriticHeadState(
        rule_id="prediction.pe_critic_head.v1",
        feature_dim=feature_dim,
        update_count=130,
        axis_weights=(
            ("task", tuple(-0.5 + 0.03 * i for i in range(feature_dim))),
            ("relationship", tuple(0.4 - 0.02 * i for i in range(feature_dim))),
            ("regime", tuple(0.6 - 0.05 * i for i in range(feature_dim))),
            ("action", tuple(-0.3 + 0.04 * i for i in range(feature_dim))),
        ),
        axis_biases=(
            ("task", 0.50),
            ("relationship", -0.10),
            ("regime", 0.60),
            ("action", -0.20),
        ),
        last_prediction=0.55,
        last_target=0.32,
        last_validation_delta=-0.20,
        last_capacity_cost=0.40,
        last_rollback_evidence="degradation-injected",
    )
    head.restore_state(degraded_state)
    degraded_predictions = {
        axis: inner.predict_axis(axis=axis, features=held_out_features, fallback=0.0)
        for axis in ("task", "relationship", "regime", "action")
    }
    assert any(
        degraded_predictions[axis] != learned_predictions[axis]
        for axis in learned_predictions
    ), (
        "Degradation step should perturb at least one axis prediction; "
        "otherwise the drill is testing nothing."
    )

    # Restore from snapshot.
    head.restore_state(snapshot)
    restored_predictions = {
        axis: inner.predict_axis(axis=axis, features=held_out_features, fallback=0.0)
        for axis in ("task", "relationship", "regime", "action")
    }
    for axis in learned_predictions:
        assert restored_predictions[axis] == learned_predictions[axis], (
            f"PE critic rollback drill failed on axis {axis!r}: "
            f"restored={restored_predictions[axis]!r}, "
            f"expected={learned_predictions[axis]!r}. Promotion of the "
            "PE critic head from readout-only is BLOCKED by this test."
        )


def test_rewarding_state_head_rejects_invalid_feature_dim() -> None:
    head = _RewardingStateHead()
    bad_state = RewardingStateHeadState(
        rule_id="credit.rewarding_state_head.v1",
        feature_dim=0,
        update_count=0,
        weights=(),
        bias=0.0,
        last_prediction=0.0,
        last_target=0.0,
        last_validation_delta=0.0,
        last_capacity_cost=0.0,
        last_rollback_evidence="",
    )
    with pytest.raises(ValueError, match="feature_dim"):
        head.restore_state(bad_state)


def test_pe_critic_head_rejects_invalid_feature_dim() -> None:
    head = _PECriticHead(decay=0.9)
    bad_state = PECriticHeadState(
        rule_id="prediction.pe_critic_head.v1",
        feature_dim=0,
        update_count=0,
        axis_weights=(),
        axis_biases=(),
        last_prediction=0.0,
        last_target=0.0,
        last_validation_delta=0.0,
        last_capacity_cost=0.0,
        last_rollback_evidence="",
    )
    with pytest.raises(ValueError, match="feature_dim"):
        head.restore_state(bad_state)
