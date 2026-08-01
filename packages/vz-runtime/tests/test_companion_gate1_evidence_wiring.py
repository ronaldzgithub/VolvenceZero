from __future__ import annotations

from volvence_zero.brain import Brain, BrainConfig
from volvence_zero.integration import FinalRolloutConfig
from volvence_zero.runtime import WiringLevel


def _config(*, pe_drive: bool) -> BrainConfig:
    return BrainConfig(
        external_prediction_error_drive=pe_drive,
        prediction_error_readout_only=not pe_drive,
        primary_prediction_error_dominance_enabled=pe_drive,
        final_rollout_config=FinalRolloutConfig(
            prediction_error_temporal_switch=WiringLevel.ACTIVE,
            prediction_error_runtime_modulation=WiringLevel.ACTIVE,
        ),
    )


def test_gate1_profile_threads_runtime_modulation_into_both_tracks() -> None:
    session = Brain(_config(pe_drive=True)).create_session(
        session_id="gate1-profile-threading"
    )

    assert (
        session.runner._joint_loop.world_temporal_policy
        .prediction_error_runtime_modulation_enabled
        is True
    )
    assert (
        session.runner._joint_loop.self_temporal_policy
        .prediction_error_runtime_modulation_enabled
        is True
    )


def test_gate1_pe_off_keeps_publication_but_blocks_temporal_learning() -> None:
    pe_on = Brain(_config(pe_drive=True)).create_session(
        session_id="gate1-pe-on"
    )
    pe_off = Brain(_config(pe_drive=False)).create_session(
        session_id="gate1-pe-off"
    )

    for session in (pe_on, pe_off):
        session.run_turn("I expected the plan to work, but the outcome changed.")
        session.run_turn("Now help me adjust while keeping our earlier context.")

    on_pe = pe_on.runner._upstream_snapshots["prediction_error"].value
    off_pe = pe_off.runner._upstream_snapshots["prediction_error"].value
    on_temporal = pe_on.runner._upstream_snapshots[
        "world_temporal_consolidation"
    ].value
    off_temporal = pe_off.runner._upstream_snapshots[
        "world_temporal_consolidation"
    ].value

    assert on_pe.bootstrap is False
    assert off_pe.bootstrap is False
    assert on_temporal.prediction_error_applied is True
    assert off_temporal.prediction_error_applied is False
