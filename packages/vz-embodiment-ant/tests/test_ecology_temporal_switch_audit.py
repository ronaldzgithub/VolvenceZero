"""P0-B temporal switch / segment closure audit tests (plan 05 s3.3).

The v1/v2 audit implemented P0-B as three existence checks on training
telemetry.  These tests exercise the real thing: a deterministic,
action-label-free transition protocol, a steady-state negative control with a
pre-declared switch-rate ceiling, positive-control switch localization, the
timeout-dominance check and the segment-credit on/off parity check.
"""

from __future__ import annotations

import pytest

from volvence_ant.experiments.ecology_mechanism_audit import (
    ECOLOGY_AUDIT_NEGATIVE_CONTROL_SWITCH_RATE_CEILING,
    ECOLOGY_AUDIT_PROTOCOL_PHASE_TICKS,
    ECOLOGY_AUDIT_SWITCH_LOCALIZATION_WINDOW,
    ECOLOGY_AUDIT_TIMEOUT_CLOSURE_RATIO_CEILING,
    EcologyMechanismAuditConfig,
    EcologyTransitionPhase,
    _boundary_ticks,
    _run_transition_trace,
    _segment_credit_parity,
    _steady_state_protocol,
    _temporal_switch_audit,
    _transition_protocol,
)
from volvence_ant.env import AntWorld, AntWorldConfig, ButterSource
from volvence_ant.runtime import (
    AntObjectiveKind,
    AntSenseSchema,
    AntSession,
    AntSessionConfig,
)


def _config() -> EcologyMechanismAuditConfig:
    return EcologyMechanismAuditConfig(
        n_ants=1,
        temporal_latent_dim=4,
        episode_rounds=1,
        episodes_per_stage=1,
        evaluation_rounds=3,
        seed=5,
    )


def _cold_checkpoint(*, temporal_latent_dim: int, seed: int):
    world = AntWorld(
        config=AntWorldConfig(seed=seed, step_size=0.4),
        world_objects=(ButterSource(object_id="probe", x=0.6, y=0.35),),
    )
    session = AntSession(
        world,
        config=AntSessionConfig(
            temporal_latent_dim=temporal_latent_dim,
            session_id=f"test:p0b:cold:{seed}",
            seed=seed,
            objective=AntObjectiveKind.ECOLOGY,
            sense_schema=AntSenseSchema.ECOLOGY_V2,
        ),
    )
    return session.export_learning_checkpoint(
        checkpoint_id=f"test:p0b:cold:{seed}",
        include_runtime_replay=False,
    )


def test_protocol_covers_every_declared_state_transition() -> None:
    protocol = _transition_protocol()
    phases = tuple(dict.fromkeys(pose.phase for pose in protocol))

    assert phases == tuple(EcologyTransitionPhase)
    assert len(protocol) == sum(
        ticks for _phase, ticks in ECOLOGY_AUDIT_PROTOCOL_PHASE_TICKS
    )
    # The protocol scripts STATE only. carrying_food is forced False solely
    # while the body is scripted outbound; pickup and delivery must be real
    # world events, so the carrying phases leave it to the world.
    carrying_phases = {
        pose.phase
        for pose in protocol
        if pose.carrying_food is not None
    }
    assert carrying_phases == {
        EcologyTransitionPhase.CRUISE,
        EcologyTransitionPhase.FOOD_APPROACH,
    }
    assert all(
        pose.carrying_food is False
        for pose in protocol
        if pose.carrying_food is not None
    )


def test_boundary_ticks_mark_each_state_change() -> None:
    boundaries = _boundary_ticks()

    assert tuple(phase for phase, _tick in boundaries) == tuple(
        EcologyTransitionPhase
    )[1:]
    assert tuple(tick for _phase, tick in boundaries) == (5, 10, 14, 19, 23)


def test_steady_state_control_is_a_single_constant_pose() -> None:
    steady = _steady_state_protocol()

    assert len({(pose.x, pose.y, pose.heading) for pose in steady}) == 1
    assert all(
        pose.phase is EcologyTransitionPhase.CRUISE for pose in steady
    )


def test_declared_temporal_thresholds_are_frozen() -> None:
    """plan 05 s2.1 -- pre-declared, and only tightenable afterwards."""

    assert ECOLOGY_AUDIT_NEGATIVE_CONTROL_SWITCH_RATE_CEILING == 0.2
    assert ECOLOGY_AUDIT_TIMEOUT_CLOSURE_RATIO_CEILING == 1.0
    assert ECOLOGY_AUDIT_SWITCH_LOCALIZATION_WINDOW == 4
    with pytest.raises(ValueError):
        EcologyMechanismAuditConfig(
            negative_control_switch_rate_ceiling=0.9,
        )
    with pytest.raises(ValueError):
        EcologyMechanismAuditConfig(switch_localization_window=99)


async def test_negative_control_does_not_chatter_and_positive_control_switches() -> None:
    config = _config()
    checkpoint = _cold_checkpoint(
        temporal_latent_dim=config.temporal_latent_dim,
        seed=11,
    )

    negative, _ = await _run_transition_trace(
        config=config,
        checkpoint=checkpoint,
        poses=_steady_state_protocol(),
        label="test-negative",
        segment_credit_enabled=True,
        include_boundaries=False,
    )
    positive, _ = await _run_transition_trace(
        config=config,
        checkpoint=checkpoint,
        poses=_transition_protocol(),
        label="test-positive",
        segment_credit_enabled=True,
        include_boundaries=True,
    )

    assert negative.switch_rate <= (
        config.negative_control_switch_rate_ceiling
    )
    assert negative.boundary_localizations == ()
    assert positive.switch_ticks
    assert any(item.localized for item in positive.boundary_localizations)
    assert len(positive.boundary_localizations) == len(_boundary_ticks())


async def test_positive_control_does_not_close_every_segment_on_timeout() -> None:
    config = _config()
    checkpoint = _cold_checkpoint(
        temporal_latent_dim=config.temporal_latent_dim,
        seed=11,
    )

    positive, _ = await _run_transition_trace(
        config=config,
        checkpoint=checkpoint,
        poses=_transition_protocol(),
        label="test-timeout",
        segment_credit_enabled=True,
        include_boundaries=True,
    )
    reasons = dict(positive.close_reason_counts)

    assert positive.closed_segment_count > 0
    assert reasons.get("beta-switch", 0) > 0
    assert reasons.get("environment-milestone", 0) > 0
    assert positive.timeout_closure_ratio < (
        config.timeout_closure_ratio_ceiling
    )


async def test_two_track_tick_log_carries_every_declared_field() -> None:
    config = _config()
    checkpoint = _cold_checkpoint(
        temporal_latent_dim=config.temporal_latent_dim,
        seed=11,
    )

    trace, _ = await _run_transition_trace(
        config=config,
        checkpoint=checkpoint,
        poses=_transition_protocol(),
        label="test-log",
        segment_credit_enabled=True,
        include_boundaries=True,
    )
    first = trace.ticks[0]

    assert len(trace.ticks) == len(_transition_protocol())
    assert first.world_beta_threshold > 0.0
    assert first.self_beta_threshold > 0.0
    assert {name for name, _ in first.track_switch_gates}
    assert first.external_switch_pressure == pytest.approx(
        first.fast_prior_switch_pressure
        + first.prediction_error_switch_pressure
    )
    assert any(item.segment_closed_this_tick for item in trace.ticks)
    assert any(item.carrying_food for item in trace.ticks)
    assert any(item.heat_harmful for item in trace.ticks)
    # plan 05:150 -- "SSL 前后 switch 参数和 histogram". The owner-published
    # SwitchGateStats histogram is unavailable (the ant's SSL trainer never
    # trains, see ECOLOGY_AUDIT_DECLARED_GAPS), so the audit derives its own
    # per-track histogram from this same per-tick log and publishes the switch
    # parameters at both ends of the trace.
    assert len(trace.world_beta_histogram) == 10
    assert len(trace.self_beta_histogram) == 10
    assert sum(trace.world_beta_histogram) == len(trace.ticks)
    assert sum(trace.self_beta_histogram) == len(trace.ticks)
    before = trace.switch_parameters_before
    after = trace.switch_parameters_after
    assert before is not None and after is not None
    assert before.label == "before"
    assert after.label == "after"
    assert before.world_beta_threshold == first.world_beta_threshold
    assert after.world_beta_threshold > 0.0
    assert after.self_beta_threshold > 0.0


async def test_segment_credit_on_off_keeps_sense_action_and_lineage_aligned() -> None:
    config = _config()
    checkpoint = _cold_checkpoint(
        temporal_latent_dim=config.temporal_latent_dim,
        seed=11,
    )
    protocol = _transition_protocol()

    _on_trace, credit_on = await _run_transition_trace(
        config=config,
        checkpoint=checkpoint,
        poses=protocol,
        label="test-credit-on",
        segment_credit_enabled=True,
        include_boundaries=True,
    )
    off_trace, credit_off = await _run_transition_trace(
        config=config,
        checkpoint=checkpoint,
        poses=protocol,
        label="test-credit-off",
        segment_credit_enabled=False,
        include_boundaries=True,
    )
    parity = _segment_credit_parity(
        config=config,
        credit_on=credit_on,
        credit_off=credit_off,
    )

    assert parity.passed
    assert parity.lineage_differences == ()
    assert parity.first_misaligned_tick == -1
    assert parity.max_turn_delta <= config.segment_credit_parity_tolerance
    # Only the credit aggregation may differ, and it must actually differ:
    # otherwise the parity check would be comparing a lane against itself.
    assert not off_trace.segment_credit_enabled
    assert off_trace.close_reason_counts != _on_trace.close_reason_counts


async def test_temporal_switch_audit_binds_all_three_controls() -> None:
    config = _config()
    checkpoint = _cold_checkpoint(
        temporal_latent_dim=config.temporal_latent_dim,
        seed=11,
    )

    audit = await _temporal_switch_audit(config=config, checkpoint=checkpoint)

    assert audit.positive_control.label == "positive-control"
    assert audit.negative_control.label == "negative-control"
    assert audit.segment_credit_off_control.label == "segment-credit-off"
    assert audit.positive_control.segment_credit_enabled
    assert not audit.segment_credit_off_control.segment_credit_enabled
    assert audit.positive_control.checkpoint_id == checkpoint.checkpoint_id
    assert audit.parity.passed
