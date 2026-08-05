from __future__ import annotations

import hashlib
import struct
from dataclasses import replace
from typing import Mapping

import pytest

from companion_bench.msc_corpus import MSCDyad, MSCSession, MSCUtterance
from companion_bench.msc_runtime_collection import (
    MSCSteeringResidualContext,
    MSCSteeringShadowContext,
    _parse_residual_context,
    _steering_from_response,
    collect_msc_full_runtime_contexts,
    parse_msc_collected_samples,
    serialize_msc_collected_samples,
)


def _dyad() -> MSCDyad:
    return MSCDyad(
        dyad_id="dyad-1",
        split="heldout",
        sessions=(
            MSCSession(
                session_index=1,
                utterances=(
                    MSCUtterance("speaker_1", "first target turn", 1),
                    MSCUtterance("speaker_2", "first partner turn", 2),
                    MSCUtterance("speaker_1", "second target turn", 3),
                ),
            ),
            MSCSession(
                session_index=2,
                utterances=(
                    MSCUtterance("speaker_2", "second partner turn", 1),
                    MSCUtterance("speaker_1", "third target turn", 2),
                ),
            ),
        ),
        initial_personas=(("target persona",), ("partner persona",)),
    )


class _Service:
    def __init__(self, *, user_id: str) -> None:
        self.user_id = user_id
        self.turns: list[tuple[str, str, str]] = []
        self.session_id = ""

    def create_session(
        self, *, session_id: str, user_id: str
    ) -> Mapping[str, object]:
        assert user_id == self.user_id
        self.session_id = session_id
        return {"session_id": session_id}

    def submit_observed_turn(
        self,
        *,
        session_id: str,
        user_input: str,
        active_speaker_id: str,
        observation_kind: str,
    ) -> Mapping[str, object]:
        assert session_id == self.session_id
        self.turns.append((user_input, active_speaker_id, observation_kind))
        values = (0.6, 0.8)
        values_sha256 = hashlib.sha256(
            struct.pack("!2d", *values)
        ).hexdigest()
        payload = {
            "schema_version": "msc-full-runtime-context.v1",
            "volvence_full_stack": True,
            "acceptance_passed": True,
            "propagate_event_count": 12,
            "active_speaker_id": active_speaker_id,
            "temporal_n_z": 3,
            "substrate_fallback_active": False,
            "runtime_slot_surface_sha256": "c" * 64,
            "context_lineage": {
                "model_fingerprint": {
                    "model_id": "frozen-model",
                    "version": "snapshot-v1",
                    "weights_sha256": "a" * 64,
                },
                "readout_kind": "latest-token-selected-layer-residual-l2.v1",
                "runtime_origin": "hf-local",
                "layer_indices": [11, 12],
                "activation_widths": [1, 1],
            },
            "context_representation": {
                "values": values,
                "values_sha256": values_sha256,
                "source_sha256": hashlib.sha256(
                    user_input.encode("utf-8")
                ).hexdigest(),
            },
            "input_token_count": 2,
            "output_token_count": 1,
            "total_token_count": 3,
            "generation_latency_ms": 1.0,
            "end_to_end_latency_ms": 2.0,
            "raw_text_retained": False,
            "evaluation_writeback_allowed": False,
        }
        return {"evidence_telemetry": {"msc_runtime_context": payload}}

    def end_observed_scene(
        self, *, session_id: str, drain_slow_loop: bool
    ) -> Mapping[str, object]:
        assert session_id == self.session_id
        assert drain_slow_loop
        return {
            "slow_loop_drained": True,
            "evidence_telemetry": {
                "msc_runtime_slow_loop_latency_ms": 3.0,
            },
        }

    def close_session(self, *, session_id: str) -> Mapping[str, object]:
        assert session_id == self.session_id
        return {"closed": True}


def test_collector_aligns_pre_target_contexts_and_incremental_costs() -> None:
    services: list[_Service] = []

    def factory(user_id: str) -> _Service:
        service = _Service(user_id=user_id)
        services.append(service)
        return service

    samples = collect_msc_full_runtime_contexts(
        (_dyad(),), service_factory=factory
    )

    assert tuple(sample.sample_id for sample in samples) == (
        "dyad-1:s1:u3:p2",
        "dyad-1:s2:u2:p4",
    )
    first, second = samples
    assert (
        first.interval_total_token_count,
        first.interval_latency_ms,
        first.observation_turn_count,
        first.scene_boundary_count,
    ) == (9, 6.0, 3, 0)
    assert (
        second.interval_total_token_count,
        second.interval_latency_ms,
        second.observation_turn_count,
        second.scene_boundary_count,
    ) == (6, 7.0, 2, 1)
    assert first.context.active_speaker_id == "speaker_2"
    assert second.context.active_speaker_id == "speaker_2"
    assert all(
        raw not in repr(samples)
        for raw in (
            "target persona",
            "first partner turn",
            "second partner turn",
        )
    )
    assert services[0].turns[0] == (
        "target persona",
        "speaker_1",
        "persona",
    )


def _steering_residual(*, conditioned: bool) -> MSCSteeringResidualContext:
    values = (0.6, 0.8)
    return MSCSteeringResidualContext(
        source_sha256="d" * 64,
        layer_indices=(0,),
        activation_widths=(2,),
        values=values,
        values_sha256=hashlib.sha256(struct.pack("!2d", *values)).hexdigest(),
        conditioned=conditioned,
        readout_kind="latest-token-hooked-layer-residual-l2.v1",
    )


def test_text_free_checkpoint_round_trips_sensor_off_steering_context() -> None:
    samples = collect_msc_full_runtime_contexts(
        (_dyad(),), service_factory=lambda user_id: _Service(user_id=user_id)
    )
    steering = MSCSteeringShadowContext(
        decision_id="gate:decision:1",
        observations=(
            ("belief_margin", 0.5),
            ("prediction_error_magnitude", 0.1),
        ),
        prediction_error_magnitude=0.4,
        noop_context=_steering_residual(conditioned=False),
        action_context=_steering_residual(conditioned=True),
        reader_artifact_id="reader-v1",
        executor_artifact_id="executor-v1",
        gate_policy_artifact_id="gate-v1",
        gate_policy_version=1,
        source_model_id="frozen-model",
        source_model_weights_sha256="a" * 64,
        layer_index=0,
        residual_norm=1.0,
        control_norm=0.1,
        control_norm_cap=0.25,
        shadow_hook_latency_ms=0.05,
        sensor_off_action_context=_steering_residual(conditioned=True),
        sensor_off_executor_artifact_id="executor-unconditional-v1",
        sensor_off_control_norm=0.1,
        sensor_off_shadow_hook_latency_ms=0.05,
    )
    changed = (
        replace(
            samples[0],
            steering_shadow=steering,
            interval_steering_hook_latency_ms=0.1,
        ),
        *samples[1:],
    )

    payload = serialize_msc_collected_samples(changed)

    assert "first partner turn" not in payload
    assert parse_msc_collected_samples(payload) == changed


def test_steering_payload_rejects_malformed_observations_without_filtering() -> None:
    response = {
        "evidence_telemetry": {
            "msc_runtime_context": {
                "steering_shadow": {
                    "raw_text_retained": False,
                    "evaluation_writeback_allowed": False,
                    "shadow_hook_executed": True,
                    "free_bias_present": False,
                    "zero_code_strict_noop": True,
                    "gate_observations": [["valid", 0.5], ["truncated"]],
                    "noop_context": {},
                    "action_context": {},
                }
            }
        }
    }

    with pytest.raises(ValueError, match="malformed item"):
        _steering_from_response(response)


def test_residual_parser_requires_a_real_boolean_conditioning_flag() -> None:
    values = (0.6, 0.8)
    payload = {
        "source_sha256": "d" * 64,
        "layer_indices": [0],
        "activation_widths": [2],
        "values": values,
        "values_sha256": hashlib.sha256(struct.pack("!2d", *values)).hexdigest(),
        "conditioned": "false",
        "readout_kind": "latest-token-hooked-layer-residual-l2.v1",
    }

    with pytest.raises(ValueError, match="must be boolean"):
        _parse_residual_context(payload)
