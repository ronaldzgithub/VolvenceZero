"""Seven-day simulated companion lifecycle orchestrator.

The orchestrator deliberately talks to the lifeform through its HTTP surface.
Owner persistence happens when the session is closed, process restart is an
explicit lifecycle port, and the next session hydrates from the same user
scope.  No kernel module or owner implementation is imported here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Protocol, Sequence, runtime_checkable
import urllib.error
import urllib.parse
import urllib.request

from lifeform_evolution.relationship_assistant_pilot import (
    PilotTranscriptTurn,
    RelationshipAssistantPilotHarness,
)
from lifeform_evolution.seven_day_state_control import (
    StateInterventionEvidence,
)


SEVEN_DAY_COMPANION_RUN_SCHEMA_VERSION = "seven-day-companion-run.v1"
SEVEN_DAY_COMPANION_DAY_SCHEMA_VERSION = "seven-day-companion-day.v1"
_DAY_MILLISECONDS = 86_400_000
_REQUIRED_EVENT_TAGS = frozenset({"callback", "emotion", "boundary"})


def _require_non_empty(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty")
    return value


def _require_sha256(value: str, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a SHA-256 digest") from exc
    return value


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


@dataclass(frozen=True)
class SimulatedSourceAttestation:
    simulator_model_id: str
    simulator_model_family: str
    sut_model_id: str
    sut_model_family: str
    model_and_adapter_fingerprint: str
    consent_scope: str = "synthetic-no-human-subject"
    pii_scan_artifact_sha256: str = ""
    judge_model_family: str | None = None
    common_adapter_bundle_id: str | None = None
    common_adapter_version: str | None = None
    common_adapter_compatibility_fingerprint: str | None = None
    character_manifest_package_id: str | None = None
    character_id: str | None = None
    character_prefix_package_id: str | None = None
    character_wiring_level: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.simulator_model_id, field="simulator_model_id")
        _require_non_empty(self.simulator_model_family, field="simulator_model_family")
        _require_non_empty(self.sut_model_id, field="sut_model_id")
        _require_non_empty(self.sut_model_family, field="sut_model_family")
        _require_sha256(
            self.model_and_adapter_fingerprint,
            field="model_and_adapter_fingerprint",
        )
        _require_non_empty(self.consent_scope, field="consent_scope")
        _require_sha256(
            self.pii_scan_artifact_sha256,
            field="pii_scan_artifact_sha256",
        )
        families = {
            self.simulator_model_family.strip().lower(),
            self.sut_model_family.strip().lower(),
        }
        if len(families) != 2:
            raise ValueError("simulator and SUT model families must differ")
        if self.judge_model_family is not None:
            judge = _require_non_empty(self.judge_model_family, field="judge_model_family").strip().lower()
            if judge in families:
                raise ValueError("judge model family must differ from simulator and SUT")
        stack_fields = {
            "common_adapter_bundle_id": self.common_adapter_bundle_id,
            "common_adapter_version": self.common_adapter_version,
            "common_adapter_compatibility_fingerprint": (self.common_adapter_compatibility_fingerprint),
            "character_manifest_package_id": (self.character_manifest_package_id),
            "character_id": self.character_id,
            "character_prefix_package_id": self.character_prefix_package_id,
            "character_wiring_level": self.character_wiring_level,
        }
        configured = tuple(value is not None for value in stack_fields.values())
        if any(configured) and not all(configured):
            raise ValueError("base+adapter+character source attestation must provide every runtime stack field.")
        if all(configured):
            for name, value in stack_fields.items():
                assert value is not None
                _require_non_empty(value, field=name)
            if self.character_wiring_level != "active":
                raise ValueError("seven-day character runtime attestation requires ACTIVE wiring.")


@dataclass(frozen=True)
class SevenDayScheduleDay:
    day_index: int
    exchange_count: int
    required_event_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.day_index < 1 or self.day_index > 7:
            raise ValueError("day_index must be in [1, 7]")
        if self.exchange_count <= 0:
            raise ValueError("exchange_count must be positive")
        unknown = set(self.required_event_tags) - _REQUIRED_EVENT_TAGS
        if unknown:
            raise ValueError(f"unknown event tags: {sorted(unknown)!r}")
        if len(set(self.required_event_tags)) != len(self.required_event_tags):
            raise ValueError("required_event_tags must be unique")


@dataclass(frozen=True)
class SevenDayScenarioSchedule:
    scenario_id: str
    persona_ref: str
    arc_type: str
    virtual_start_ms: int
    days: tuple[SevenDayScheduleDay, ...]

    def __post_init__(self) -> None:
        _require_non_empty(self.scenario_id, field="scenario_id")
        _require_non_empty(self.persona_ref, field="persona_ref")
        _require_non_empty(self.arc_type, field="arc_type")
        if self.virtual_start_ms < 0:
            raise ValueError("virtual_start_ms must be non-negative")
        if tuple(day.day_index for day in self.days) != tuple(range(1, 8)):
            raise ValueError("seven-day schedule must contain consecutive days 1..7")
        covered = {tag for day in self.days for tag in day.required_event_tags}
        if not _REQUIRED_EVENT_TAGS.issubset(covered):
            raise ValueError("seven-day schedule must declare callback, emotion, and boundary")


@dataclass(frozen=True)
class SimulatedUserTurn:
    text: str
    fsm_action: str | None = None
    fsm_payload: str | None = None
    event_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.text, field="simulated user turn text")
        unknown = set(self.event_tags) - _REQUIRED_EVENT_TAGS
        if unknown:
            raise ValueError(f"unknown simulated event tags: {sorted(unknown)!r}")


@dataclass(frozen=True)
class ProcessRestartEvidence:
    after_day_index: int
    previous_instance_id: str
    next_instance_id: str
    healthcheck_passed: bool
    persistence_scope_unchanged: bool
    previous_persistence_scope_sha256: str
    next_persistence_scope_sha256: str
    state_intervention: StateInterventionEvidence

    def __post_init__(self) -> None:
        if self.after_day_index < 1 or self.after_day_index > 6:
            raise ValueError("restart day must be in [1, 6]")
        _require_non_empty(self.previous_instance_id, field="previous_instance_id")
        _require_non_empty(self.next_instance_id, field="next_instance_id")
        if self.previous_instance_id == self.next_instance_id:
            raise ValueError("process restart must change instance identity")
        if not self.healthcheck_passed:
            raise RuntimeError("restarted service failed health check")
        if not self.persistence_scope_unchanged:
            raise RuntimeError("restart changed the owner persistence scope")
        _require_sha256(
            self.previous_persistence_scope_sha256,
            field="previous_persistence_scope_sha256",
        )
        _require_sha256(
            self.next_persistence_scope_sha256,
            field="next_persistence_scope_sha256",
        )
        if (
            self.previous_persistence_scope_sha256
            != self.next_persistence_scope_sha256
        ):
            raise RuntimeError("restart health reports different persistence scopes")


@dataclass(frozen=True)
class SevenDayTurnEvidence:
    exchange_index: int
    user_text: str
    assistant_text: str
    fsm_action: str | None
    fsm_payload: str | None
    event_tags: tuple[str, ...]
    fsm_probe_passed: bool | None
    pe_magnitude: float | None = None
    pe_bootstrap: bool | None = None
    world_temporal_prediction_error_applied: bool | None = None
    self_temporal_prediction_error_applied: bool | None = None
    gate_telemetry: tuple[tuple[str, object], ...] = ()
    response_rationale_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class SevenDayConsoleProbeActionEvidence:
    item_id: str
    action_id: str
    action: str
    correction_kind: str | None
    replacement_sha256: str | None
    created_at_ms: int
    status: str

    def __post_init__(self) -> None:
        _require_non_empty(self.item_id, field="console probe item_id")
        _require_non_empty(self.action_id, field="console probe action_id")
        if self.action not in {"keep", "delete"}:
            raise ValueError("console probe action must be keep or delete")
        if self.action == "keep":
            if self.correction_kind is not None or self.replacement_sha256 is not None:
                raise ValueError("keep probe may not carry correction evidence")
        elif self.correction_kind != "content_inaccurate":
            raise ValueError("delete probe requires typed correction evidence")
        elif self.replacement_sha256 is not None:
            raise ValueError("delete probe may not carry replacement evidence")
        if self.replacement_sha256 is not None:
            _require_sha256(
                self.replacement_sha256,
                field="console probe replacement_sha256",
            )
        if self.created_at_ms < 0:
            raise ValueError("console probe created_at_ms must be non-negative")
        _require_non_empty(self.status, field="console probe status")


@dataclass(frozen=True)
class SevenDayDayEvidence:
    schema_version: str
    run_id: str
    arm_label: str
    scenario_id: str
    day_index: int
    virtual_observed_at_ms: int
    session_id: str
    service_instance_id: str
    cold_start_continuity_metrics: Mapping[str, object]
    turns: tuple[SevenDayTurnEvidence, ...]
    console_probe_actions: tuple[SevenDayConsoleProbeActionEvidence, ...]
    continuity_metrics: Mapping[str, object]
    pilot_day_evidence_ref: str
    pilot_day_transcript_sha256: str
    end_scene_slow_loop_drained: bool
    owner_persisted_before_restart: bool
    restart_after_day: ProcessRestartEvidence | None
    end_scene_gate_telemetry: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True)
class SevenDayCompanionRun:
    schema_version: str
    run_id: str
    arm_label: str
    scenario_id: str
    paraphrase_seed: int
    persona_ref: str
    arc_type: str
    user_scope_hash: str
    source_attestation: SimulatedSourceAttestation
    days: tuple[SevenDayDayEvidence, ...]
    event_coverage: tuple[str, ...]
    process_restart_count: int
    all_restarts_exact: bool
    simulated_longitudinal_only: bool
    external_human_value_claim_allowed: bool
    production_promotion_authorized: bool

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@runtime_checkable
class SevenDayCompanionService(Protocol):
    @property
    def instance_id(self) -> str: ...

    def create_session(self, *, session_id: str, user_id: str) -> Mapping[str, object]: ...

    def submit_turn(self, *, session_id: str, user_input: str) -> Mapping[str, object]: ...

    def end_scene(self, *, session_id: str, drain_slow_loop: bool) -> Mapping[str, object]: ...

    def continuity_metrics(self, *, session_id: str, observed_at_ms: int) -> Mapping[str, object]: ...

    def relationship_memory(self, *, session_id: str) -> Mapping[str, object]: ...

    def relationship_memory_action(
        self,
        *,
        session_id: str,
        item_id: str,
        action: str,
        observed_at_ms: int,
        replacement: str | None = None,
        correction_kind: str | None = None,
    ) -> Mapping[str, object]: ...

    def close_session(self, *, session_id: str) -> Mapping[str, object]: ...


@runtime_checkable
class SevenDayProcessLifecycle(Protocol):
    def restart_after_day(self, *, day_index: int) -> ProcessRestartEvidence: ...


@runtime_checkable
class SevenDayUserDriver(Protocol):
    def next_turn(
        self,
        *,
        day_index: int,
        exchange_index: int,
        recent_assistant_turns: Sequence[str],
    ) -> SimulatedUserTurn: ...


class HTTPSevenDayCompanionService:
    """Small synchronous HTTP client for the evidence orchestrator."""

    def __init__(
        self,
        *,
        base_url: str,
        user_id: str,
        instance_id: str,
        vertical: str | None = None,
        character_id: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self._base_url = _require_non_empty(base_url, field="base_url").rstrip("/")
        self._user_id = _require_non_empty(user_id, field="user_id")
        self._instance_id = _require_non_empty(instance_id, field="instance_id")
        self._vertical = vertical.strip() if vertical else None
        self._character_id = character_id.strip() if character_id else None
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._timeout_s = timeout_s

    @property
    def instance_id(self) -> str:
        return self._instance_id

    def replace_instance_id(self, instance_id: str) -> None:
        self._instance_id = _require_non_empty(instance_id, field="instance_id")

    def create_session(self, *, session_id: str, user_id: str) -> Mapping[str, object]:
        if user_id != self._user_id:
            raise ValueError("HTTP client user scope mismatch")
        payload: dict[str, object] = {
            "session_id": session_id,
            "user_id": user_id,
        }
        if self._vertical:
            payload["vertical"] = self._vertical
        if self._character_id:
            payload["character_id"] = self._character_id
        return self._request("POST", "/v1/sessions", payload=payload)

    def submit_turn(self, *, session_id: str, user_input: str) -> Mapping[str, object]:
        return self._request(
            "POST",
            f"/v1/sessions/{urllib.parse.quote(session_id, safe='')}/turns",
            payload={"user_input": user_input},
        )

    def submit_observed_turn(
        self,
        *,
        session_id: str,
        user_input: str,
        active_speaker_id: str,
        observation_kind: str,
    ) -> Mapping[str, object]:
        """Submit one typed corpus observation through the same service path.

        This method is intentionally outside ``SevenDayCompanionService``:
        ordinary seven-day product evidence remains a user/assistant dialogue,
        while the MSC evidence profile observes both members of a dyad.
        """

        if active_speaker_id not in {"speaker_1", "speaker_2"}:
            raise ValueError(
                "active_speaker_id must be speaker_1 or speaker_2"
            )
        if observation_kind not in {"persona", "dialogue"}:
            raise ValueError("observation_kind must be persona or dialogue")
        return self._request(
            "POST",
            f"/v1/sessions/{urllib.parse.quote(session_id, safe='')}/turns",
            payload={
                "user_input": user_input,
                "active_speaker_id": active_speaker_id,
                "observation_kind": observation_kind,
            },
        )

    def end_scene(self, *, session_id: str, drain_slow_loop: bool) -> Mapping[str, object]:
        return self._request(
            "POST",
            f"/v1/sessions/{urllib.parse.quote(session_id, safe='')}/end-scene",
            payload={
                "drain_slow_loop": drain_slow_loop,
                "reason": "seven-day-simulated-day-boundary",
            },
        )

    def end_observed_scene(
        self, *, session_id: str, drain_slow_loop: bool
    ) -> Mapping[str, object]:
        """Close an evidence-only observed corpus session."""

        return self._request(
            "POST",
            f"/v1/sessions/{urllib.parse.quote(session_id, safe='')}/end-scene",
            payload={
                "drain_slow_loop": drain_slow_loop,
                "reason": "msc-runtime-session-boundary",
            },
        )

    def continuity_metrics(self, *, session_id: str, observed_at_ms: int) -> Mapping[str, object]:
        query = urllib.parse.urlencode({"session_id": session_id, "observed_at_ms": observed_at_ms})
        return self._request("GET", f"/v1/users/me/continuity-metrics?{query}")

    def relationship_memory(self, *, session_id: str) -> Mapping[str, object]:
        query = urllib.parse.urlencode({"session_id": session_id})
        return self._request("GET", f"/v1/users/me/relationship-memory?{query}")

    def relationship_memory_action(
        self,
        *,
        session_id: str,
        item_id: str,
        action: str,
        observed_at_ms: int,
        replacement: str | None = None,
        correction_kind: str | None = None,
    ) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "session_id": session_id,
            "action": action,
            "observed_at_ms": observed_at_ms,
        }
        if replacement is not None:
            payload["replacement"] = replacement
        if correction_kind is not None:
            payload["correction_kind"] = correction_kind
        encoded_item = urllib.parse.quote(item_id, safe="")
        return self._request(
            "POST",
            f"/v1/users/me/relationship-memory/{encoded_item}/action",
            payload=payload,
        )

    def close_session(self, *, session_id: str) -> Mapping[str, object]:
        return self._request(
            "DELETE",
            f"/v1/sessions/{urllib.parse.quote(session_id, safe='')}",
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        body = None if payload is None else _canonical_bytes(dict(payload))
        request = urllib.request.Request(
            f"{self._base_url}{path}",
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Alpha-User": self._user_id,
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"seven-day service {method} {path} failed: HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"seven-day service {method} {path} unavailable: {exc.reason}") from exc
        if not isinstance(decoded, Mapping):
            raise TypeError("seven-day service response must be an object")
        return decoded


class SevenDayCompanionOrchestrator:
    """Drive seven real service lifecycles on a deterministic virtual calendar."""

    def __init__(
        self,
        *,
        service: SevenDayCompanionService,
        lifecycle: SevenDayProcessLifecycle,
        pilot_harness: RelationshipAssistantPilotHarness,
    ) -> None:
        self._service = service
        self._lifecycle = lifecycle
        self._pilot_harness = pilot_harness

    def run(
        self,
        *,
        run_id: str,
        arm_label: str,
        paraphrase_seed: int,
        user_id: str,
        schedule: SevenDayScenarioSchedule,
        user_driver: SevenDayUserDriver,
        source_attestation: SimulatedSourceAttestation,
        drain_slow_loop: bool,
        output_path: str | Path | None = None,
    ) -> SevenDayCompanionRun:
        _require_non_empty(run_id, field="run_id")
        _require_non_empty(arm_label, field="arm_label")
        if paraphrase_seed < 0:
            raise ValueError("paraphrase_seed must be non-negative")
        _require_non_empty(user_id, field="user_id")
        days = []
        observed_event_tags = set()
        stable_user_scope_hash = None
        for day in schedule.days:
            virtual_observed_at_ms = schedule.virtual_start_ms + (day.day_index - 1) * _DAY_MILLISECONDS
            session_id = _session_id(
                run_id=run_id,
                arm_label=arm_label,
                scenario_id=schedule.scenario_id,
                day_index=day.day_index,
            )
            instance_id = self._service.instance_id
            created = self._service.create_session(
                session_id=session_id,
                user_id=user_id,
            )
            if created.get("session_id") != session_id:
                raise RuntimeError("service created the wrong session")
            expected_character_id = source_attestation.character_id
            if expected_character_id is not None and created.get("character_id") != expected_character_id:
                raise RuntimeError("service did not bind the preregistered character package")
            cold_start_metrics = dict(
                self._service.continuity_metrics(
                    session_id=session_id,
                    observed_at_ms=virtual_observed_at_ms,
                )
            )
            cold_start_scope = cold_start_metrics.get("user_scope_hash")
            if not isinstance(cold_start_scope, str) or not cold_start_scope:
                raise RuntimeError("cold-start continuity metrics lack user_scope_hash")
            transcript = []
            turn_evidence = []
            recent_assistant_turns = []
            day_event_tags = set()
            for exchange_index in range(1, day.exchange_count + 1):
                generated = user_driver.next_turn(
                    day_index=day.day_index,
                    exchange_index=exchange_index,
                    recent_assistant_turns=tuple(recent_assistant_turns[-6:]),
                )
                day_event_tags.update(generated.event_tags)
                observed_event_tags.update(generated.event_tags)
                response = self._service.submit_turn(
                    session_id=session_id,
                    user_input=generated.text,
                )
                assistant_text = response.get("response_text")
                if not isinstance(assistant_text, str) or not assistant_text.strip():
                    raise RuntimeError("service turn lacks response_text")
                recent_assistant_turns.append(assistant_text)
                fsm_probe_passed = response.get("fsm_probe_passed")
                if fsm_probe_passed is not None and not isinstance(fsm_probe_passed, bool):
                    raise RuntimeError("service fsm_probe_passed must be bool or null")
                pe_magnitude = response.get("pe_magnitude")
                if (
                    isinstance(pe_magnitude, bool)
                    or not isinstance(pe_magnitude, (int, float))
                    or float(pe_magnitude) < 0.0
                ):
                    raise RuntimeError("service pe_magnitude must be a non-negative number")
                pe_bootstrap = response.get("pe_bootstrap")
                world_pe_applied = response.get("world_temporal_prediction_error_applied")
                self_pe_applied = response.get("self_temporal_prediction_error_applied")
                for field_name, field_value in (
                    ("pe_bootstrap", pe_bootstrap),
                    (
                        "world_temporal_prediction_error_applied",
                        world_pe_applied,
                    ),
                    (
                        "self_temporal_prediction_error_applied",
                        self_pe_applied,
                    ),
                ):
                    if not isinstance(field_value, bool):
                        raise RuntimeError(f"service {field_name} must be bool")
                # Legacy/non-evidence services predate the optional typed
                # gate telemetry envelope.  Dedicated gate evaluators require
                # their registered fields, while the generic seven-day runner
                # remains backward compatible with an absent envelope.
                raw_gate_telemetry = response.get("evidence_telemetry", {})
                if not isinstance(raw_gate_telemetry, Mapping):
                    raise RuntimeError("service evidence_telemetry must be an object")
                gate_telemetry = tuple(sorted((str(key), value) for key, value in raw_gate_telemetry.items()))
                raw_rationale_tags = response.get("response_rationale_tags", ())
                if not isinstance(raw_rationale_tags, (list, tuple)) or not all(
                    isinstance(item, str) and item for item in raw_rationale_tags
                ):
                    raise RuntimeError("service response_rationale_tags must be a string list")
                rationale_tags = tuple(raw_rationale_tags)
                if expected_character_id is not None:
                    required_tags = {
                        f"character_id={expected_character_id}",
                        "character_prefix=active",
                        (f"character_prefix_kv={source_attestation.character_prefix_package_id}"),
                    }
                    missing_tags = required_tags.difference(rationale_tags)
                    if missing_tags:
                        raise RuntimeError(f"turn lacks ACTIVE character carrier attestation: {sorted(missing_tags)!r}")
                transcript.extend(
                    (
                        PilotTranscriptTurn(role="user", text=generated.text),
                        PilotTranscriptTurn(role="assistant", text=assistant_text),
                    )
                )
                turn_evidence.append(
                    SevenDayTurnEvidence(
                        exchange_index=exchange_index,
                        user_text=generated.text,
                        assistant_text=assistant_text,
                        fsm_action=generated.fsm_action,
                        fsm_payload=generated.fsm_payload,
                        event_tags=generated.event_tags,
                        fsm_probe_passed=fsm_probe_passed,
                        pe_magnitude=float(pe_magnitude),
                        pe_bootstrap=pe_bootstrap,
                        world_temporal_prediction_error_applied=(world_pe_applied),
                        self_temporal_prediction_error_applied=(self_pe_applied),
                        gate_telemetry=gate_telemetry,
                        response_rationale_tags=rationale_tags,
                    )
                )
            if not set(day.required_event_tags).issubset(day_event_tags):
                raise RuntimeError(f"day {day.day_index} missed declared event tags")
            console_probe_actions = self._run_console_probe(
                session_id=session_id,
                observed_at_ms=virtual_observed_at_ms,
            )
            metrics = dict(
                self._service.continuity_metrics(
                    session_id=session_id,
                    observed_at_ms=virtual_observed_at_ms,
                )
            )
            user_scope_hash = metrics.get("user_scope_hash")
            if not isinstance(user_scope_hash, str) or not user_scope_hash:
                raise RuntimeError("continuity metrics lack user_scope_hash")
            if stable_user_scope_hash is None:
                stable_user_scope_hash = user_scope_hash
            elif user_scope_hash != stable_user_scope_hash:
                raise RuntimeError("user scope changed across seven-day run")
            if cold_start_scope != stable_user_scope_hash:
                raise RuntimeError("cold-start user scope changed across seven-day run")
            ended = self._service.end_scene(
                session_id=session_id,
                drain_slow_loop=drain_slow_loop,
            )
            expected_drained = drain_slow_loop and bool(ended.get("closed_scene_id"))
            if bool(ended.get("slow_loop_drained")) != expected_drained:
                raise RuntimeError("end-scene slow-loop attestation drift")
            raw_end_scene_telemetry = ended.get("evidence_telemetry", {})
            if not isinstance(raw_end_scene_telemetry, Mapping):
                raise RuntimeError("end-scene evidence_telemetry must be an object")
            end_scene_gate_telemetry = tuple(
                sorted((str(key), value) for key, value in raw_end_scene_telemetry.items())
            )
            pilot = self._pilot_harness.capture_day(
                user_id=user_id,
                day_index=day.day_index,
                captured_at_ms=virtual_observed_at_ms,
                continuity_metrics=metrics,
                transcript=tuple(transcript),
            )
            closed = self._service.close_session(session_id=session_id)
            owner_persisted = closed.get("closed") is True
            if not owner_persisted:
                raise RuntimeError("session close did not attest persistence boundary")
            restart = None
            if day.day_index < 7:
                restart = self._lifecycle.restart_after_day(day_index=day.day_index)
                if restart.previous_instance_id != instance_id:
                    raise RuntimeError("restart previous instance drift")
                if self._service.instance_id != restart.next_instance_id:
                    raise RuntimeError("service instance was not rebound after restart")
                if restart.state_intervention.experiment_arm_label != arm_label:
                    raise RuntimeError("restart state intervention arm drift")
            days.append(
                SevenDayDayEvidence(
                    schema_version=SEVEN_DAY_COMPANION_DAY_SCHEMA_VERSION,
                    run_id=run_id,
                    arm_label=arm_label,
                    scenario_id=schedule.scenario_id,
                    day_index=day.day_index,
                    virtual_observed_at_ms=virtual_observed_at_ms,
                    session_id=session_id,
                    service_instance_id=instance_id,
                    cold_start_continuity_metrics=cold_start_metrics,
                    turns=tuple(turn_evidence),
                    console_probe_actions=console_probe_actions,
                    continuity_metrics=metrics,
                    pilot_day_evidence_ref=pilot.transcript_ref,
                    pilot_day_transcript_sha256=pilot.transcript_sha256,
                    end_scene_slow_loop_drained=expected_drained,
                    owner_persisted_before_restart=owner_persisted,
                    restart_after_day=restart,
                    end_scene_gate_telemetry=(end_scene_gate_telemetry),
                )
            )
        if not _REQUIRED_EVENT_TAGS.issubset(observed_event_tags):
            raise RuntimeError("seven-day run lacks callback/emotion/boundary coverage")
        assert stable_user_scope_hash is not None
        run = SevenDayCompanionRun(
            schema_version=SEVEN_DAY_COMPANION_RUN_SCHEMA_VERSION,
            run_id=run_id,
            arm_label=arm_label,
            scenario_id=schedule.scenario_id,
            paraphrase_seed=paraphrase_seed,
            persona_ref=schedule.persona_ref,
            arc_type=schedule.arc_type,
            user_scope_hash=stable_user_scope_hash,
            source_attestation=source_attestation,
            days=tuple(days),
            event_coverage=tuple(sorted(observed_event_tags)),
            process_restart_count=6,
            all_restarts_exact=all(
                day.restart_after_day is None
                or (day.restart_after_day.healthcheck_passed and day.restart_after_day.persistence_scope_unchanged)
                for day in days
            ),
            simulated_longitudinal_only=True,
            external_human_value_claim_allowed=False,
            production_promotion_authorized=False,
        )
        if output_path is not None:
            _write_run(run=run, output_path=Path(output_path))
        return run

    def _run_console_probe(
        self,
        *,
        session_id: str,
        observed_at_ms: int,
    ) -> tuple[SevenDayConsoleProbeActionEvidence, ...]:
        memory = self._service.relationship_memory(session_id=session_id)
        raw_proposals = memory.get("pending_proposals")
        if not isinstance(raw_proposals, list):
            raise RuntimeError("relationship memory response lacks pending_proposals")
        proposal_ids = []
        for item in raw_proposals:
            if not isinstance(item, Mapping):
                continue
            proposal_id = item.get("proposal_id")
            if item.get("target_owner_slot") == "memory" and isinstance(proposal_id, str) and proposal_id:
                proposal_ids.append(proposal_id)
        proposal_ids.sort()
        if len(proposal_ids) < 2:
            raise RuntimeError("daily console probe requires two memory proposals")
        evidence = []
        keep_id = proposal_ids[0]
        keep_response = self._service.relationship_memory_action(
            session_id=session_id,
            item_id=keep_id,
            action="keep",
            observed_at_ms=observed_at_ms,
        )
        evidence.append(
            self._console_probe_evidence(
                response=keep_response,
                item_id=keep_id,
                action="keep",
                correction_kind=None,
                replacement_sha256=None,
                observed_at_ms=observed_at_ms,
            )
        )
        delete_id = proposal_ids[1]
        delete_response = self._service.relationship_memory_action(
            session_id=session_id,
            item_id=delete_id,
            action="delete",
            observed_at_ms=observed_at_ms,
            correction_kind="content_inaccurate",
        )
        evidence.append(
            self._console_probe_evidence(
                response=delete_response,
                item_id=delete_id,
                action="delete",
                correction_kind="content_inaccurate",
                replacement_sha256=None,
                observed_at_ms=observed_at_ms,
            )
        )
        return tuple(evidence)

    @staticmethod
    def _console_probe_evidence(
        *,
        response: Mapping[str, object],
        item_id: str,
        action: str,
        correction_kind: str | None,
        replacement_sha256: str | None,
        observed_at_ms: int,
    ) -> SevenDayConsoleProbeActionEvidence:
        if response.get("item_id") != item_id or response.get("action") != action:
            raise RuntimeError("console probe response identity drift")
        action_id = response.get("action_id")
        status = response.get("status")
        created_at_ms = response.get("created_at_ms")
        if not isinstance(action_id, str) or not action_id:
            raise RuntimeError("console probe response lacks action_id")
        if not isinstance(status, str) or not status:
            raise RuntimeError("console probe response lacks status")
        if created_at_ms != observed_at_ms:
            raise RuntimeError("console probe virtual time drift")
        return SevenDayConsoleProbeActionEvidence(
            item_id=item_id,
            action_id=action_id,
            action=action,
            correction_kind=correction_kind,
            replacement_sha256=replacement_sha256,
            created_at_ms=observed_at_ms,
            status=status,
        )


def _session_id(*, run_id: str, arm_label: str, scenario_id: str, day_index: int) -> str:
    digest = hashlib.sha256(f"{run_id}\0{arm_label}\0{scenario_id}\0{day_index}".encode("utf-8")).hexdigest()
    return f"seven-day-{digest[:24]}"


def _write_run(*, run: SevenDayCompanionRun, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(run.to_json())
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(output_path)


__all__ = [
    "HTTPSevenDayCompanionService",
    "ProcessRestartEvidence",
    "SEVEN_DAY_COMPANION_DAY_SCHEMA_VERSION",
    "SEVEN_DAY_COMPANION_RUN_SCHEMA_VERSION",
    "SevenDayCompanionOrchestrator",
    "SevenDayCompanionRun",
    "SevenDayConsoleProbeActionEvidence",
    "SevenDayCompanionService",
    "SevenDayDayEvidence",
    "SevenDayProcessLifecycle",
    "SevenDayScenarioSchedule",
    "SevenDayScheduleDay",
    "SevenDayTurnEvidence",
    "SevenDayUserDriver",
    "SimulatedSourceAttestation",
    "SimulatedUserTurn",
    "StateInterventionEvidence",
]
