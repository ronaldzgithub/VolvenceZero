"""Packet D (long-horizon-closure) — hydration failures must fail loud.

Per the ``no-swallow-errors-no-hasattr-abuse`` workspace rule and
the spec invariant 3 in ``docs/specs/owner-hydration.md``, hydration
implementations must raise typed ``HydrationError`` subclasses on
structural / version / owner mismatches — never silent fallbacks.

This test enumerates the documented failure modes for each owner
and asserts the expected exception type fires.
"""

from __future__ import annotations

import pytest

from volvence_zero.owner_hydration import (
    HydrationOwnerMismatchError,
    HydrationPayloadInvalidError,
    HydrationVersionMismatchError,
    OwnerPersistenceSnapshot,
)


def _bad_snap(owner: str, version: int = 1, payload=None):
    if payload is None:
        payload = {"records": {}, "completed_refs": {}, "revision_counts": {}, "record_lifecycle": {}, "record_followup_policy": {}, "record_outcome": {}}
    return OwnerPersistenceSnapshot(
        owner_name=owner,
        schema_version=version,
        payload=payload,
    )


def test_semantic_state_owner_name_mismatch_fails_loud() -> None:
    from volvence_zero.semantic_state.store import SemanticStateStore

    store = SemanticStateStore()
    with pytest.raises(HydrationOwnerMismatchError, match="owner_name"):
        store.hydrate_from_persistence(_bad_snap("not_semantic_state"))


def test_semantic_state_version_mismatch_fails_loud() -> None:
    from volvence_zero.semantic_state.store import SemanticStateStore

    store = SemanticStateStore()
    with pytest.raises(HydrationVersionMismatchError, match="schema_version"):
        store.hydrate_from_persistence(_bad_snap("semantic_state", version=999))


def test_semantic_state_payload_missing_required_key_fails_loud() -> None:
    from volvence_zero.semantic_state.store import SemanticStateStore

    store = SemanticStateStore()
    with pytest.raises(HydrationPayloadInvalidError, match="missing required key"):
        store.hydrate_from_persistence(
            OwnerPersistenceSnapshot(
                owner_name="semantic_state",
                schema_version=1,
                payload={"records": {}},  # missing the other 5 required keys
            )
        )


def test_semantic_state_unknown_slot_in_payload_fails_loud() -> None:
    from volvence_zero.semantic_state.store import SemanticStateStore

    store = SemanticStateStore()
    bad = {
        "records": {"not_a_real_slot": []},
        "completed_refs": {},
        "revision_counts": {},
        "record_lifecycle": {},
        "record_followup_policy": {},
        "record_outcome": {},
    }
    with pytest.raises(HydrationPayloadInvalidError, match="unknown slot"):
        store.hydrate_from_persistence(
            OwnerPersistenceSnapshot(
                owner_name="semantic_state",
                schema_version=1,
                payload=bad,
            )
        )


def test_social_record_store_owner_name_mismatch_fails_loud() -> None:
    from volvence_zero.social import SocialRecordStore

    store = SocialRecordStore()
    with pytest.raises(HydrationOwnerMismatchError, match="owner_name"):
        store.hydrate_from_persistence(
            OwnerPersistenceSnapshot(
                owner_name="not_social_record_store",
                schema_version=1,
                payload={},
            )
        )


def test_social_record_store_version_mismatch_fails_loud() -> None:
    from volvence_zero.social import SocialRecordStore

    store = SocialRecordStore()
    with pytest.raises(HydrationVersionMismatchError, match="schema_version"):
        store.hydrate_from_persistence(
            OwnerPersistenceSnapshot(
                owner_name="social_record_store",
                schema_version=99,
                payload={},
            )
        )


def test_social_record_store_unknown_tom_slot_fails_loud() -> None:
    from volvence_zero.social import SocialRecordStore

    store = SocialRecordStore()
    with pytest.raises(HydrationPayloadInvalidError, match="unknown ToM slot"):
        store.hydrate_from_persistence(
            OwnerPersistenceSnapshot(
                owner_name="social_record_store",
                schema_version=1,
                payload={"tom_records": {"not_a_slot": []}},
            )
        )


def test_social_record_store_invalid_atom_blob_fails_loud() -> None:
    from volvence_zero.social import SocialRecordStore

    store = SocialRecordStore()
    with pytest.raises(HydrationPayloadInvalidError, match="CommonGroundAtom"):
        store.hydrate_from_persistence(
            OwnerPersistenceSnapshot(
                owner_name="social_record_store",
                schema_version=1,
                payload={
                    "common_ground": {
                        "dyad_atoms": [
                            {
                                "atom_id": "cg-1",
                                "scope_id": "self+bob",
                                "scope_kind": "dyad",
                                # missing summary / recursion_depth / etc.
                            }
                        ]
                    }
                },
            )
        )


def test_regime_owner_name_mismatch_fails_loud() -> None:
    from volvence_zero.regime import RegimeModule

    module = RegimeModule()
    with pytest.raises(HydrationOwnerMismatchError, match="owner_name"):
        module.hydrate_from_persistence(
            OwnerPersistenceSnapshot(
                owner_name="not_regime",
                schema_version=1,
                payload={},
            )
        )


def test_regime_version_mismatch_fails_loud() -> None:
    from volvence_zero.regime import RegimeModule

    module = RegimeModule()
    with pytest.raises(HydrationVersionMismatchError, match="schema_version"):
        module.hydrate_from_persistence(
            OwnerPersistenceSnapshot(
                owner_name="regime",
                schema_version=99,
                payload={},
            )
        )


def test_regime_invalid_payload_fails_loud() -> None:
    from volvence_zero.regime import RegimeModule

    module = RegimeModule()
    with pytest.raises(HydrationPayloadInvalidError, match="payload"):
        module.hydrate_from_persistence(
            OwnerPersistenceSnapshot(
                owner_name="regime",
                schema_version=1,
                payload={"historical_effectiveness": {"bad": "shape"}},
            )
        )


def test_prediction_error_heads_owner_name_mismatch_fails_loud() -> None:
    from volvence_zero.prediction.error import PredictionErrorModule

    module = PredictionErrorModule()
    with pytest.raises(HydrationOwnerMismatchError, match="owner_name"):
        module.hydrate_from_persistence(
            OwnerPersistenceSnapshot(
                owner_name="not_prediction_error_heads",
                schema_version=1,
                payload={},
            )
        )


def test_prediction_error_heads_missing_key_fails_loud() -> None:
    from volvence_zero.prediction.error import PredictionErrorModule

    module = PredictionErrorModule()
    with pytest.raises(HydrationPayloadInvalidError, match="missing key"):
        module.hydrate_from_persistence(
            OwnerPersistenceSnapshot(
                owner_name="prediction_error_heads",
                schema_version=1,
                payload={"critic": {}},
            )
        )


def test_dual_track_gate_learner_invalid_weights_fails_loud() -> None:
    from volvence_zero.dual_track.gate_learner import DualTrackGateLearner

    learner = DualTrackGateLearner()
    with pytest.raises(HydrationPayloadInvalidError, match="weights"):
        learner.hydrate_from_persistence(
            OwnerPersistenceSnapshot(
                owner_name="dual_track_gate_learner",
                schema_version=1,
                payload={
                    "weights": "not-a-list",
                    "update_count": 0,
                    "abs_error_sum": 0.0,
                    "heuristic_abs_error_sum": 0.0,
                    "settled_comparison_count": 0,
                },
            )
        )


def test_credit_module_owner_name_mismatch_fails_loud() -> None:
    from volvence_zero.credit.gate import CreditModule

    module = CreditModule()
    with pytest.raises(HydrationOwnerMismatchError, match="owner_name"):
        module.hydrate_from_persistence(
            OwnerPersistenceSnapshot(
                owner_name="not_credit_heads",
                schema_version=1,
                payload={},
            )
        )


def test_credit_module_version_mismatch_fails_loud() -> None:
    from volvence_zero.credit.gate import CreditModule

    module = CreditModule()
    with pytest.raises(HydrationVersionMismatchError, match="schema_version"):
        module.hydrate_from_persistence(
            OwnerPersistenceSnapshot(
                owner_name="credit_heads",
                schema_version=99,
                payload={},
            )
        )


def test_credit_module_missing_key_fails_loud() -> None:
    from volvence_zero.credit.gate import CreditModule

    module = CreditModule()
    with pytest.raises(HydrationPayloadInvalidError, match="missing key"):
        module.hydrate_from_persistence(
            OwnerPersistenceSnapshot(
                owner_name="credit_heads",
                schema_version=1,
                payload={"rewarding_state_head": {}},
            )
        )


def test_followup_manager_owner_name_mismatch_fails_loud() -> None:
    from lifeform_core.followup_manager import FollowupManager

    fm = FollowupManager()
    with pytest.raises(HydrationOwnerMismatchError):
        fm.hydrate_from_persistence(
            OwnerPersistenceSnapshot(
                owner_name="something_else",
                schema_version=1,
                payload={"counter": 0, "seen_keys": [], "pending": []},
            )
        )


def test_followup_manager_version_mismatch_fails_loud() -> None:
    from lifeform_core.followup_manager import FollowupManager

    fm = FollowupManager()
    with pytest.raises(HydrationVersionMismatchError):
        fm.hydrate_from_persistence(
            OwnerPersistenceSnapshot(
                owner_name="followup_manager",
                schema_version=99,
                payload={"counter": 0, "seen_keys": [], "pending": []},
            )
        )


def test_followup_manager_invalid_pending_item_fails_loud() -> None:
    from lifeform_core.followup_manager import FollowupManager

    fm = FollowupManager()
    with pytest.raises(HydrationPayloadInvalidError, match="missing required key"):
        fm.hydrate_from_persistence(
            OwnerPersistenceSnapshot(
                owner_name="followup_manager",
                schema_version=1,
                payload={
                    "counter": 1,
                    "seen_keys": [],
                    "pending": [
                        {"followup_id": "x"}  # missing source/description/due_at_tick
                    ],
                },
            )
        )


def test_vitals_owner_name_mismatch_fails_loud() -> None:
    from lifeform_core.types import TickEvent  # noqa: F401 (sanity)
    from lifeform_core.vitals import (
        DriveSpec,
        VitalsBootstrap,
        VitalsModule,
    )

    bootstrap = VitalsBootstrap(
        schema_version=1,
        drives=(
            DriveSpec(
                name="d1",
                target=0.5,
                homeostatic_band=(0.4, 0.6),
                decay_per_tick=0.01,
                pe_weight=1.0,
            ),
        ),
        proactive_pe_threshold=1.0,
        proactive_followup_priority=0.5,
        proactive_cooldown_ticks=10,
    )
    vitals = VitalsModule(bootstrap)
    with pytest.raises(HydrationOwnerMismatchError):
        vitals.hydrate_from_persistence(
            OwnerPersistenceSnapshot(
                owner_name="not_vitals",
                schema_version=1,
                payload={
                    "levels": {},
                    "tick_index": 0,
                    "last_proactive_at": None,
                    "turn_count": 0,
                    "iqr_baseline": {},
                    "iqr_baseline_accum": {},
                    "baseline_observation_count": 0,
                },
            )
        )


def test_vitals_payload_missing_required_key_fails_loud() -> None:
    from lifeform_core.vitals import (
        DriveSpec,
        VitalsBootstrap,
        VitalsModule,
    )

    bootstrap = VitalsBootstrap(
        schema_version=1,
        drives=(
            DriveSpec(
                name="d1",
                target=0.5,
                homeostatic_band=(0.4, 0.6),
                decay_per_tick=0.01,
                pe_weight=1.0,
            ),
        ),
        proactive_pe_threshold=1.0,
        proactive_followup_priority=0.5,
        proactive_cooldown_ticks=10,
    )
    vitals = VitalsModule(bootstrap)
    with pytest.raises(HydrationPayloadInvalidError, match="missing required key"):
        vitals.hydrate_from_persistence(
            OwnerPersistenceSnapshot(
                owner_name="vitals",
                schema_version=1,
                payload={"levels": {}},  # missing rest
            )
        )
