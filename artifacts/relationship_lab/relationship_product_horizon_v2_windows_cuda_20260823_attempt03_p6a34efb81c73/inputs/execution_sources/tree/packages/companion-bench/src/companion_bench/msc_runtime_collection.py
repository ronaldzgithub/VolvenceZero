# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Protocol-only collector for the MSC complete-runtime prediction arm.

The benchmark package knows the official corpus and the evidence DTO, but it
does not import Volvence internals.  A caller supplies a service client that
implements the small HTTP-shaped protocol below.  Raw corpus text exists only
for the duration of each request; returned values contain residual vectors,
hash lineage, and measured costs only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import struct
from typing import Protocol

from companion_bench.msc_corpus import MSCDyad
from companion_bench.prediction_research import (
    MSCFullRuntimeContext,
    MSCNextTurnExample,
    build_msc_next_turn_examples,
    parse_msc_full_runtime_context,
)


@dataclass(frozen=True)
class MSCSteeringResidualContext:
    source_sha256: str
    layer_indices: tuple[int, ...]
    activation_widths: tuple[int, ...]
    values: tuple[float, ...]
    values_sha256: str
    conditioned: bool
    readout_kind: str

    def __post_init__(self) -> None:
        if self.readout_kind != "latest-token-hooked-layer-residual-l2.v1":
            raise ValueError("MSC steering residual readout kind is unsupported")
        if (
            len(self.source_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.source_sha256)
        ):
            raise ValueError("MSC steering residual source SHA-256 is invalid")
        if (
            not self.layer_indices
            or tuple(sorted(self.layer_indices)) != self.layer_indices
            or len(set(self.layer_indices)) != len(self.layer_indices)
            or len(self.activation_widths) != len(self.layer_indices)
            or sum(self.activation_widths) != len(self.values)
        ):
            raise ValueError("MSC steering residual geometry is invalid")
        if not self.values or not all(math.isfinite(value) for value in self.values):
            raise ValueError("MSC steering residual values must be finite")
        norm = math.sqrt(sum(value * value for value in self.values))
        if not math.isclose(norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("MSC steering residual values must be L2 normalized")
        expected = hashlib.sha256(
            struct.pack(f"!{len(self.values)}d", *self.values)
        ).hexdigest()
        if self.values_sha256 != expected:
            raise ValueError("MSC steering residual values SHA-256 drift")


@dataclass(frozen=True)
class MSCSteeringShadowContext:
    decision_id: str
    observations: tuple[tuple[str, float], ...]
    prediction_error_magnitude: float
    noop_context: MSCSteeringResidualContext
    action_context: MSCSteeringResidualContext
    reader_artifact_id: str
    executor_artifact_id: str
    gate_policy_artifact_id: str
    gate_policy_version: int
    source_model_id: str
    source_model_weights_sha256: str
    layer_index: int
    residual_norm: float
    control_norm: float
    control_norm_cap: float
    shadow_hook_latency_ms: float
    sensor_off_action_context: MSCSteeringResidualContext | None = None
    sensor_off_executor_artifact_id: str = ""
    sensor_off_control_norm: float = 0.0
    sensor_off_shadow_hook_latency_ms: float = 0.0
    schema_version: str = "msc-steering-shadow-context.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "msc-steering-shadow-context.v1":
            raise ValueError("MSC steering SHADOW schema is unsupported")
        for value in (
            self.decision_id,
            self.reader_artifact_id,
            self.executor_artifact_id,
            self.gate_policy_artifact_id,
            self.source_model_id,
        ):
            if not value.strip():
                raise ValueError("MSC steering lineage strings must be non-empty")
        names = tuple(name for name, _ in self.observations)
        if not names or len(set(names)) != len(names):
            raise ValueError("MSC steering observations must be uniquely named")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for _, value in self.observations
        ):
            raise ValueError("MSC steering observations must be within [0, 1]")
        observed = dict(self.observations)
        expected_proxy = min(1.0, max(0.0, self.prediction_error_magnitude / 4.0))
        if not math.isclose(
            observed.get("prediction_error_magnitude", -1.0),
            expected_proxy,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("MSC steering PE proxy drift")
        if self.noop_context.conditioned or not self.action_context.conditioned:
            raise ValueError("MSC steering matched context conditioning flags drift")
        if (
            self.noop_context.layer_indices != self.action_context.layer_indices
            or self.noop_context.activation_widths
            != self.action_context.activation_widths
        ):
            raise ValueError("MSC steering matched context geometry drift")
        if self.layer_index not in self.noop_context.layer_indices:
            raise ValueError("MSC steering intervention layer is absent from context")
        if self.gate_policy_version < 1:
            raise ValueError("MSC steering gate policy version must be positive")
        if (
            len(self.source_model_weights_sha256) != 64
            or any(
                char not in "0123456789abcdef"
                for char in self.source_model_weights_sha256
            )
        ):
            raise ValueError("MSC steering model weights SHA-256 is invalid")
        for value in (
            self.residual_norm,
            self.control_norm,
            self.control_norm_cap,
            self.shadow_hook_latency_ms,
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("MSC steering norm/latency values are invalid")
        if self.control_norm > self.control_norm_cap + 1e-8:
            raise ValueError("MSC steering control norm exceeds cap")
        if self.sensor_off_action_context is None:
            if (
                self.sensor_off_executor_artifact_id
                or self.sensor_off_control_norm != 0.0
                or self.sensor_off_shadow_hook_latency_ms != 0.0
            ):
                raise ValueError("MSC steering sensor-off evidence is partial")
        else:
            if (
                not self.sensor_off_executor_artifact_id.strip()
                or not self.sensor_off_action_context.conditioned
                or self.sensor_off_action_context.layer_indices
                != self.noop_context.layer_indices
                or self.sensor_off_action_context.activation_widths
                != self.noop_context.activation_widths
                or not math.isfinite(self.sensor_off_control_norm)
                or self.sensor_off_control_norm < 0.0
                or self.sensor_off_control_norm > self.control_norm_cap + 1e-8
                or not math.isfinite(self.sensor_off_shadow_hook_latency_ms)
                or self.sensor_off_shadow_hook_latency_ms < 0.0
            ):
                raise ValueError("MSC steering sensor-off evidence is invalid")


@dataclass(frozen=True)
class MSCFullRuntimeCollectedSample:
    """One context plus its incremental full-runtime maintenance cost."""

    context: MSCFullRuntimeContext
    interval_input_token_count: int
    interval_output_token_count: int
    interval_total_token_count: int
    interval_latency_ms: float
    observation_turn_count: int
    scene_boundary_count: int
    steering_shadow: MSCSteeringShadowContext | None = None
    interval_steering_hook_latency_ms: float = 0.0
    schema_version: str = "msc-full-runtime-collected-sample.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "msc-full-runtime-collected-sample.v1":
            raise ValueError("MSC collected sample schema is unsupported")
        if (
            self.interval_input_token_count < 1
            or self.interval_output_token_count < 0
            or self.interval_total_token_count
            != self.interval_input_token_count + self.interval_output_token_count
        ):
            raise ValueError("MSC collected sample token accounting is invalid")
        if (
            not math.isfinite(self.interval_latency_ms)
            or self.interval_latency_ms < self.context.latency_ms
        ):
            raise ValueError("MSC collected sample latency accounting is invalid")
        if self.observation_turn_count < 1 or self.scene_boundary_count < 0:
            raise ValueError("MSC collected sample interval counts are invalid")
        if (
            not math.isfinite(self.interval_steering_hook_latency_ms)
            or self.interval_steering_hook_latency_ms < 0.0
            or self.interval_steering_hook_latency_ms
            > self.interval_latency_ms + 1e-8
        ):
            raise ValueError("MSC collected steering interval latency is invalid")

    @property
    def sample_id(self) -> str:
        return self.context.sample_id


class MSCFullRuntimeObservationService(Protocol):
    """Transport boundary implemented by the shared seven-day HTTP client."""

    def create_session(
        self, *, session_id: str, user_id: str
    ) -> Mapping[str, object]: ...

    def submit_observed_turn(
        self,
        *,
        session_id: str,
        user_input: str,
        active_speaker_id: str,
        observation_kind: str,
    ) -> Mapping[str, object]: ...

    def end_observed_scene(
        self, *, session_id: str, drain_slow_loop: bool
    ) -> Mapping[str, object]: ...

    def close_session(self, *, session_id: str) -> Mapping[str, object]: ...


MSCFullRuntimeServiceFactory = Callable[
    [str], MSCFullRuntimeObservationService
]


def msc_runtime_scope_ids(dyad: MSCDyad) -> tuple[str, str, str]:
    """Return hash-only scope, user, and session ids for one dyad."""

    digest = hashlib.sha256(
        f"{dyad.split}\0{dyad.dyad_id}".encode("utf-8")
    ).hexdigest()
    return digest, f"msc-user-{digest[:24]}", f"msc-session-{digest[:24]}"


def _context_from_response(
    response: Mapping[str, object], *, sample_id: str
) -> MSCFullRuntimeContext:
    telemetry = response.get("evidence_telemetry")
    if not isinstance(telemetry, Mapping):
        raise ValueError("MSC turn response lacks evidence_telemetry")
    payload = telemetry.get("msc_runtime_context")
    if not isinstance(payload, Mapping):
        raise ValueError("MSC turn response lacks msc_runtime_context")
    return parse_msc_full_runtime_context(payload, sample_id=sample_id)


def _parse_residual_context(payload: Mapping[str, object]) -> MSCSteeringResidualContext:
    values = payload.get("values")
    layers = payload.get("layer_indices")
    widths = payload.get("activation_widths")
    if (
        not isinstance(values, (tuple, list))
        or not isinstance(layers, (tuple, list))
        or not isinstance(widths, (tuple, list))
    ):
        raise ValueError("MSC steering residual arrays are missing")
    conditioned = payload.get("conditioned")
    if not isinstance(conditioned, bool):
        raise ValueError("MSC steering residual conditioned flag must be boolean")
    try:
        return MSCSteeringResidualContext(
            source_sha256=str(payload["source_sha256"]),
            layer_indices=tuple(int(value) for value in layers),
            activation_widths=tuple(int(value) for value in widths),
            values=tuple(float(value) for value in values),
            values_sha256=str(payload["values_sha256"]),
            conditioned=conditioned,
            readout_kind=str(payload["readout_kind"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("MSC steering residual context fields are invalid") from exc


def _steering_from_response(
    response: Mapping[str, object],
) -> MSCSteeringShadowContext | None:
    telemetry = response.get("evidence_telemetry")
    if not isinstance(telemetry, Mapping):
        raise ValueError("MSC turn response lacks evidence_telemetry")
    runtime_payload = telemetry.get("msc_runtime_context")
    if not isinstance(runtime_payload, Mapping):
        raise ValueError("MSC turn response lacks msc_runtime_context")
    payload = runtime_payload.get("steering_shadow")
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError("MSC steering SHADOW payload must be an object")
    if payload.get("raw_text_retained") is not False:
        raise ValueError("MSC steering payload retained raw text")
    if payload.get("evaluation_writeback_allowed") is not False:
        raise ValueError("MSC steering payload allows evaluation writeback")
    if payload.get("shadow_hook_executed") is not True:
        raise ValueError("MSC steering payload did not execute the SHADOW hook")
    if payload.get("free_bias_present") is not False:
        raise ValueError("MSC steering payload contains free bias")
    if payload.get("zero_code_strict_noop") is not True:
        raise ValueError("MSC steering payload lacks strict zero-code no-op")
    observations = payload.get("gate_observations")
    noop = payload.get("noop_context")
    action = payload.get("action_context")
    sensor_off = payload.get("sensor_off_action_context")
    if (
        not isinstance(observations, (tuple, list))
        or not isinstance(noop, Mapping)
        or not isinstance(action, Mapping)
        or (sensor_off is not None and not isinstance(sensor_off, Mapping))
    ):
        raise ValueError("MSC steering payload lacks observations/contexts")
    if any(
        not isinstance(item, (tuple, list)) or len(item) != 2
        for item in observations
    ):
        raise ValueError("MSC steering observations contain a malformed item")
    try:
        return MSCSteeringShadowContext(
            schema_version=str(payload["schema_version"]),
            decision_id=str(payload["decision_id"]),
            observations=tuple(
                (str(item[0]), float(item[1]))
                for item in observations
            ),
            prediction_error_magnitude=float(payload["prediction_error_magnitude"]),
            noop_context=_parse_residual_context(noop),
            action_context=_parse_residual_context(action),
            reader_artifact_id=str(payload["reader_artifact_id"]),
            executor_artifact_id=str(payload["executor_artifact_id"]),
            gate_policy_artifact_id=str(payload["gate_policy_artifact_id"]),
            gate_policy_version=int(payload["gate_policy_version"]),
            source_model_id=str(payload["source_model_id"]),
            source_model_weights_sha256=str(
                payload["source_model_weights_sha256"]
            ),
            layer_index=int(payload["layer_index"]),
            residual_norm=float(payload["residual_norm"]),
            control_norm=float(payload["control_norm"]),
            control_norm_cap=float(payload["control_norm_cap"]),
            shadow_hook_latency_ms=float(payload["shadow_hook_latency_ms"]),
            sensor_off_action_context=(
                _parse_residual_context(sensor_off)
                if isinstance(sensor_off, Mapping)
                else None
            ),
            sensor_off_executor_artifact_id=str(
                payload.get("sensor_off_executor_artifact_id", "")
            ),
            sensor_off_control_norm=float(
                payload.get("sensor_off_control_norm", 0.0)
            ),
            sensor_off_shadow_hook_latency_ms=float(
                payload.get("sensor_off_shadow_hook_latency_ms", 0.0)
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("MSC steering SHADOW fields are invalid") from exc


def _stored_steering(
    payload: Mapping[str, object] | None,
) -> MSCSteeringShadowContext | None:
    if payload is None:
        return None
    observations = payload.get("observations")
    noop = payload.get("noop_context")
    action = payload.get("action_context")
    sensor_off = payload.get("sensor_off_action_context")
    if (
        not isinstance(observations, list)
        or not isinstance(noop, Mapping)
        or not isinstance(action, Mapping)
        or (sensor_off is not None and not isinstance(sensor_off, Mapping))
    ):
        raise ValueError("stored MSC steering payload is incomplete")
    return MSCSteeringShadowContext(
        schema_version=str(payload["schema_version"]),
        decision_id=str(payload["decision_id"]),
        observations=tuple((str(item[0]), float(item[1])) for item in observations),
        prediction_error_magnitude=float(payload["prediction_error_magnitude"]),
        noop_context=_parse_residual_context(noop),
        action_context=_parse_residual_context(action),
        reader_artifact_id=str(payload["reader_artifact_id"]),
        executor_artifact_id=str(payload["executor_artifact_id"]),
        gate_policy_artifact_id=str(payload["gate_policy_artifact_id"]),
        gate_policy_version=int(payload["gate_policy_version"]),
        source_model_id=str(payload["source_model_id"]),
        source_model_weights_sha256=str(payload["source_model_weights_sha256"]),
        layer_index=int(payload["layer_index"]),
        residual_norm=float(payload["residual_norm"]),
        control_norm=float(payload["control_norm"]),
        control_norm_cap=float(payload["control_norm_cap"]),
        shadow_hook_latency_ms=float(payload["shadow_hook_latency_ms"]),
        sensor_off_action_context=(
            _parse_residual_context(sensor_off)
            if isinstance(sensor_off, Mapping)
            else None
        ),
        sensor_off_executor_artifact_id=str(
            payload.get("sensor_off_executor_artifact_id", "")
        ),
        sensor_off_control_norm=float(
            payload.get("sensor_off_control_norm", 0.0)
        ),
        sensor_off_shadow_hook_latency_ms=float(
            payload.get("sensor_off_shadow_hook_latency_ms", 0.0)
        ),
    )


def serialize_msc_collected_samples(
    samples: tuple[MSCFullRuntimeCollectedSample, ...],
) -> str:
    if not samples:
        raise ValueError("MSC collected-sample checkpoint cannot be empty")
    return json.dumps(
        {
            "schema_version": "msc-collected-sample-checkpoint.v1",
            "raw_text_retained": False,
            "samples": [asdict(sample) for sample in samples],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def parse_msc_collected_samples(
    payload: str,
) -> tuple[MSCFullRuntimeCollectedSample, ...]:
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("MSC collected-sample checkpoint is invalid JSON") from exc
    if (
        not isinstance(decoded, Mapping)
        or decoded.get("schema_version") != "msc-collected-sample-checkpoint.v1"
        or decoded.get("raw_text_retained") is not False
        or not isinstance(decoded.get("samples"), list)
        or not decoded["samples"]
    ):
        raise ValueError("MSC collected-sample checkpoint envelope is invalid")
    rows: list[MSCFullRuntimeCollectedSample] = []
    for raw in decoded["samples"]:
        if not isinstance(raw, Mapping):
            raise ValueError("MSC stored sample must be an object")
        context = raw.get("context")
        steering = raw.get("steering_shadow")
        if not isinstance(context, Mapping) or (
            steering is not None and not isinstance(steering, Mapping)
        ):
            raise ValueError("MSC stored sample context is invalid")
        rows.append(
            MSCFullRuntimeCollectedSample(
                schema_version=str(raw["schema_version"]),
                context=MSCFullRuntimeContext(
                    schema_version=str(context["schema_version"]),
                    sample_id=str(context["sample_id"]),
                    active_speaker_id=str(context["active_speaker_id"]),
                    values=tuple(float(value) for value in context["values"]),
                    values_sha256=str(context["values_sha256"]),
                    source_sha256=str(context["source_sha256"]),
                    model_id=str(context["model_id"]),
                    model_version=str(context["model_version"]),
                    weights_sha256=str(context["weights_sha256"]),
                    runtime_origin=str(context["runtime_origin"]),
                    readout_kind=str(context["readout_kind"]),
                    layer_indices=tuple(int(value) for value in context["layer_indices"]),
                    activation_widths=tuple(
                        int(value) for value in context["activation_widths"]
                    ),
                    temporal_n_z=int(context["temporal_n_z"]),
                    input_token_count=int(context["input_token_count"]),
                    output_token_count=int(context["output_token_count"]),
                    total_token_count=int(context["total_token_count"]),
                    generation_latency_ms=float(context["generation_latency_ms"]),
                    latency_ms=float(context["latency_ms"]),
                    propagate_event_count=int(context["propagate_event_count"]),
                    runtime_slot_surface_sha256=str(
                        context["runtime_slot_surface_sha256"]
                    ),
                ),
                interval_input_token_count=int(raw["interval_input_token_count"]),
                interval_output_token_count=int(raw["interval_output_token_count"]),
                interval_total_token_count=int(raw["interval_total_token_count"]),
                interval_latency_ms=float(raw["interval_latency_ms"]),
                observation_turn_count=int(raw["observation_turn_count"]),
                scene_boundary_count=int(raw["scene_boundary_count"]),
                steering_shadow=_stored_steering(steering),
                interval_steering_hook_latency_ms=float(
                    raw["interval_steering_hook_latency_ms"]
                ),
            )
        )
    return tuple(rows)


def _slow_loop_latency(response: Mapping[str, object]) -> float:
    if response.get("slow_loop_drained") is not True:
        raise ValueError("MSC session boundary did not drain the slow loop")
    telemetry = response.get("evidence_telemetry")
    if not isinstance(telemetry, Mapping):
        raise ValueError("MSC session boundary lacks evidence_telemetry")
    value = telemetry.get("msc_runtime_slow_loop_latency_ms")
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError("MSC session boundary latency is invalid")
    return float(value)


def _surface_signature(context: MSCFullRuntimeContext) -> tuple[object, ...]:
    return (
        context.model_id,
        context.model_version,
        context.weights_sha256,
        context.runtime_origin,
        context.readout_kind,
        context.layer_indices,
        context.activation_widths,
        context.temporal_n_z,
        context.runtime_slot_surface_sha256,
    )


def _require_expected_target(
    example: MSCNextTurnExample,
    *,
    dyad: MSCDyad,
    session_index: int,
    utterance_index: int,
    target_text: str,
    history_turns: int,
) -> None:
    if (
        example.dyad_id != dyad.dyad_id
        or example.split != dyad.split
        or example.target_speaker != "speaker_1"
        or example.session_index != session_index
        or example.target_text != target_text
        or example.history_turns != history_turns
        or not example.sample_id.endswith(
            f":s{session_index}:u{utterance_index}:p{history_turns}"
        )
    ):
        raise ValueError("MSC runtime collector target/example identity drift")


def collect_msc_full_runtime_contexts(
    dyads: tuple[MSCDyad, ...],
    *,
    service_factory: MSCFullRuntimeServiceFactory,
) -> tuple[MSCFullRuntimeCollectedSample, ...]:
    """Observe complete dyads and freeze pre-target runtime contexts.

    Cost intervals are incremental: the first prediction includes target-persona
    ingestion and all preceding turns; each later prediction includes only
    observations and slow-loop boundaries since the preceding prediction.
    One-time service/session startup and final post-target teardown are excluded.
    """

    if not dyads:
        raise ValueError("MSC runtime collection requires dyads")
    expected = build_msc_next_turn_examples(dyads)
    expected_iter = iter(expected)
    next_expected = next(expected_iter, None)
    collected: list[MSCFullRuntimeCollectedSample] = []
    frozen_surface: tuple[object, ...] | None = None

    for dyad in dyads:
        scope_digest, user_id, session_id = msc_runtime_scope_ids(dyad)
        service = service_factory(user_id)
        created = service.create_session(session_id=session_id, user_id=user_id)
        if created.get("session_id") != session_id:
            raise ValueError("MSC runtime service created the wrong session")

        interval_input_tokens = 0
        interval_output_tokens = 0
        interval_latency_ms = 0.0
        interval_turns = 0
        interval_boundaries = 0
        interval_steering_hook_latency_ms = 0.0
        latest_context: MSCFullRuntimeContext | None = None
        latest_steering: MSCSteeringShadowContext | None = None
        observation_index = 0
        history_turns = 0

        def observe(
            *,
            text: str,
            speaker: str,
            observation_kind: str,
            bound_service: MSCFullRuntimeObservationService = service,
            bound_session_id: str = session_id,
            bound_scope_digest: str = scope_digest,
        ) -> tuple[MSCFullRuntimeContext, MSCSteeringShadowContext | None]:
            nonlocal interval_input_tokens
            nonlocal interval_output_tokens
            nonlocal interval_latency_ms
            nonlocal interval_turns
            nonlocal interval_steering_hook_latency_ms
            nonlocal observation_index
            nonlocal frozen_surface

            response = bound_service.submit_observed_turn(
                session_id=bound_session_id,
                user_input=text,
                active_speaker_id=speaker,
                observation_kind=observation_kind,
            )
            observation_index += 1
            context = _context_from_response(
                response,
                sample_id=(
                    f"observation:{bound_scope_digest[:24]}:{observation_index}"
                ),
            )
            surface = _surface_signature(context)
            if frozen_surface is None:
                frozen_surface = surface
            elif surface != frozen_surface:
                raise ValueError("MSC full-runtime surface drifted during collection")
            interval_input_tokens += context.input_token_count
            interval_output_tokens += context.output_token_count
            interval_latency_ms += context.latency_ms
            interval_turns += 1
            steering = _steering_from_response(response)
            if steering is not None:
                interval_steering_hook_latency_ms += (
                    steering.shadow_hook_latency_ms
                    + steering.sensor_off_shadow_hook_latency_ms
                )
            return context, steering

        for persona in dyad.initial_personas[0]:
            latest_context, latest_steering = observe(
                text=persona,
                speaker="speaker_1",
                observation_kind="persona",
            )

        for session in dyad.sessions:
            for utterance in session.utterances:
                if utterance.speaker == "speaker_1" and history_turns > 0:
                    if next_expected is None:
                        raise ValueError(
                            "MSC runtime collector produced an unexpected target"
                        )
                    _require_expected_target(
                        next_expected,
                        dyad=dyad,
                        session_index=session.session_index,
                        utterance_index=utterance.utterance_index,
                        target_text=utterance.text,
                        history_turns=history_turns,
                    )
                    if latest_context is None:
                        raise ValueError("MSC runtime target lacks prior context")
                    if (
                        not next_expected.history
                        or latest_context.active_speaker_id
                        != next_expected.history[-1].speaker
                    ):
                        raise ValueError(
                            "MSC runtime context is not aligned to the latest speaker"
                        )
                    collected.append(
                        MSCFullRuntimeCollectedSample(
                            context=replace(
                                latest_context,
                                sample_id=next_expected.sample_id,
                            ),
                            interval_input_token_count=interval_input_tokens,
                            interval_output_token_count=interval_output_tokens,
                            interval_total_token_count=(
                                interval_input_tokens + interval_output_tokens
                            ),
                            interval_latency_ms=interval_latency_ms,
                            observation_turn_count=interval_turns,
                            scene_boundary_count=interval_boundaries,
                            steering_shadow=latest_steering,
                            interval_steering_hook_latency_ms=(
                                interval_steering_hook_latency_ms
                            ),
                        )
                    )
                    next_expected = next(expected_iter, None)
                    interval_input_tokens = 0
                    interval_output_tokens = 0
                    interval_latency_ms = 0.0
                    interval_turns = 0
                    interval_boundaries = 0
                    interval_steering_hook_latency_ms = 0.0

                latest_context, latest_steering = observe(
                    text=utterance.text,
                    speaker=utterance.speaker,
                    observation_kind="dialogue",
                )
                history_turns += 1

            ended = service.end_observed_scene(
                session_id=session_id,
                drain_slow_loop=True,
            )
            interval_latency_ms += _slow_loop_latency(ended)
            interval_boundaries += 1

        closed = service.close_session(session_id=session_id)
        if closed.get("closed") is not True:
            raise ValueError("MSC runtime service did not close the session")

    if next_expected is not None:
        raise ValueError("MSC runtime collector did not produce every target")
    if tuple(item.sample_id for item in collected) != tuple(
        item.sample_id for item in expected
    ):
        raise ValueError("MSC runtime collector sample order drift")
    return tuple(collected)


__all__ = (
    "MSCFullRuntimeCollectedSample",
    "MSCFullRuntimeObservationService",
    "MSCFullRuntimeServiceFactory",
    "MSCSteeringResidualContext",
    "MSCSteeringShadowContext",
    "collect_msc_full_runtime_contexts",
    "msc_runtime_scope_ids",
    "parse_msc_collected_samples",
    "serialize_msc_collected_samples",
)
