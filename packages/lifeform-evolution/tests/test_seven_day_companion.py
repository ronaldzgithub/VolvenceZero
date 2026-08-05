from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

import pytest

from lifeform_evolution.relationship_assistant_pilot import (
    RelationshipAssistantPilotHarness,
)
from lifeform_evolution.seven_day_companion import (
    HTTPSevenDayCompanionService,
    ProcessRestartEvidence,
    SevenDayCompanionOrchestrator,
    SevenDayScenarioSchedule,
    SevenDayScheduleDay,
    SimulatedSourceAttestation,
    SimulatedUserTurn,
    StateInterventionEvidence,
)


_METRICS = {
    "callback_hit_rate": 0.75,
    "boundary_violation_rate": 0.0,
    "wrong_user_attribution_rate": 0.0,
    "open_loop_closure_rate": 0.5,
    "user_correction_rate": 0.0,
    "remembered_item_usefulness": 1.0,
    "seven_day_trust_delta": 0.1,
    "sample_sizes": {"callback": 4},
    "user_scope_hash": "scope-hash",
}


class _Service:
    def __init__(self) -> None:
        self._instance_index = 1
        self.events: list[tuple[str, object]] = []

    @property
    def instance_id(self) -> str:
        return f"service-{self._instance_index}"

    def create_session(self, *, session_id: str, user_id: str) -> Mapping[str, object]:
        self.events.append(("create", (session_id, user_id)))
        return {"session_id": session_id}

    def submit_turn(self, *, session_id: str, user_input: str) -> Mapping[str, object]:
        self.events.append(("turn", (session_id, user_input)))
        return {
            "response_text": f"assistant response to {user_input}",
            "pe_magnitude": 0.2,
            "pe_bootstrap": False,
            "world_temporal_prediction_error_applied": True,
            "self_temporal_prediction_error_applied": True,
        }

    def end_scene(self, *, session_id: str, drain_slow_loop: bool) -> Mapping[str, object]:
        self.events.append(("end", (session_id, drain_slow_loop)))
        return {
            "closed_scene_id": f"{session_id}:scene",
            "slow_loop_drained": drain_slow_loop,
        }

    def continuity_metrics(self, *, session_id: str, observed_at_ms: int) -> Mapping[str, object]:
        self.events.append(("metrics", (session_id, observed_at_ms)))
        return dict(_METRICS)

    def relationship_memory(self, *, session_id: str) -> Mapping[str, object]:
        self.events.append(("memory", session_id))
        return {
            "pending_proposals": [
                {
                    "proposal_id": f"{session_id}:proposal-{index}",
                    "target_owner_slot": "memory",
                }
                for index in (1, 2)
            ],
            "durable_entries": [{"entry_id": "memory-1"}],
        }

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
        self.events.append(("memory-action", (item_id, action)))
        return {
            "item_id": item_id,
            "action_id": f"action:{item_id}",
            "action": action,
            "status": "applied",
            "created_at_ms": observed_at_ms,
            "replacement": replacement,
            "correction_kind": correction_kind,
        }

    def close_session(self, *, session_id: str) -> Mapping[str, object]:
        self.events.append(("close", session_id))
        return {"closed": True}


class _Lifecycle:
    def __init__(self, service: _Service) -> None:
        self._service = service

    def restart_after_day(self, *, day_index: int) -> ProcessRestartEvidence:
        previous = self._service.instance_id
        self._service._instance_index += 1
        return ProcessRestartEvidence(
            after_day_index=day_index,
            previous_instance_id=previous,
            next_instance_id=self._service.instance_id,
            healthcheck_passed=True,
            persistence_scope_unchanged=True,
            previous_persistence_scope_sha256="5" * 64,
            next_persistence_scope_sha256="5" * 64,
            state_intervention=StateInterventionEvidence(
                experiment_arm_label="correct-user-state",
                state_loading_policy="correct-user-state",
                after_day_index=day_index,
                archived_state_ref=f"archive/day-{day_index}",
                archived_state_sha256="3" * 64,
                measurement_checkpoint_sha256="4" * 64,
                next_day_source_arm="correct-user-state",
                next_day_source_day_index=day_index,
                next_day_loaded_state_sha256="3" * 64,
            ),
        )


class _CharacterService(_Service):
    def create_session(self, *, session_id: str, user_id: str) -> Mapping[str, object]:
        payload = dict(super().create_session(session_id=session_id, user_id=user_id))
        payload["character_id"] = "zhang-wuji"
        return payload

    def submit_turn(self, *, session_id: str, user_input: str) -> Mapping[str, object]:
        payload = dict(super().submit_turn(session_id=session_id, user_input=user_input))
        payload["response_rationale_tags"] = [
            "character_id=zhang-wuji",
            "character_prefix=active",
            "character_prefix_kv=prefix-v1",
        ]
        return payload


class _Driver:
    def next_turn(
        self,
        *,
        day_index: int,
        exchange_index: int,
        recent_assistant_turns: Sequence[str],
    ) -> SimulatedUserTurn:
        tags = []
        action = None
        if day_index == 1 and exchange_index == 1:
            tags.append("emotion")
            action = "establish_pattern"
        if day_index == 4 and exchange_index == 1:
            tags.append("boundary")
            action = "boundary_test"
        if day_index == 7 and exchange_index == 1:
            tags.append("callback")
            action = "callback_probe"
        return SimulatedUserTurn(
            text=f"day {day_index} exchange {exchange_index}",
            fsm_action=action,
            fsm_payload="typed payload" if action else None,
            event_tags=tuple(tags),
        )


def _schedule() -> SevenDayScenarioSchedule:
    days = []
    for day_index in range(1, 8):
        tags = ()
        if day_index == 1:
            tags = ("emotion",)
        elif day_index == 4:
            tags = ("boundary",)
        elif day_index == 7:
            tags = ("callback",)
        days.append(
            SevenDayScheduleDay(
                day_index=day_index,
                exchange_count=5,
                required_event_tags=tags,
            )
        )
    return SevenDayScenarioSchedule(
        scenario_id="F1-seven-day-001",
        persona_ref="persona-professional",
        arc_type="progressive-warmth",
        virtual_start_ms=1_800_000_000_000,
        days=tuple(days),
    )


def _attestation() -> SimulatedSourceAttestation:
    return SimulatedSourceAttestation(
        simulator_model_id="tinyllama-local",
        simulator_model_family="llama",
        sut_model_id="qwen-local",
        sut_model_family="qwen",
        model_and_adapter_fingerprint="1" * 64,
        pii_scan_artifact_sha256="2" * 64,
    )


def _character_attestation() -> SimulatedSourceAttestation:
    return SimulatedSourceAttestation(
        simulator_model_id="smollm-local",
        simulator_model_family="smollm",
        sut_model_id="qwen-local",
        sut_model_family="qwen",
        model_and_adapter_fingerprint="1" * 64,
        pii_scan_artifact_sha256="2" * 64,
        common_adapter_bundle_id="common-bundle-v1",
        common_adapter_version="common-v1",
        common_adapter_compatibility_fingerprint="compat-v1",
        character_manifest_package_id="manifest-v1",
        character_id="zhang-wuji",
        character_prefix_package_id="prefix-v1",
        character_wiring_level="active",
    )


def test_orchestrator_runs_seven_days_with_six_real_restart_boundaries(
    tmp_path: Path,
) -> None:
    service = _Service()
    harness = RelationshipAssistantPilotHarness(
        root_dir=tmp_path / "pilot",
        pilot_id="simulated-seven-day",
        invited_user_ids=frozenset({"synthetic-user"}),
    )
    run = SevenDayCompanionOrchestrator(
        service=service,
        lifecycle=_Lifecycle(service),
        pilot_harness=harness,
    ).run(
        run_id="run-1",
        arm_label="correct-user-state",
        paraphrase_seed=1401,
        user_id="synthetic-user",
        schedule=_schedule(),
        user_driver=_Driver(),
        source_attestation=_attestation(),
        drain_slow_loop=True,
        output_path=tmp_path / "run.json",
    )
    assert len(run.days) == 7
    assert run.process_restart_count == 6
    assert run.all_restarts_exact is True
    assert run.event_coverage == ("boundary", "callback", "emotion")
    assert run.external_human_value_claim_allowed is False
    assert run.production_promotion_authorized is False
    assert all(len(day.turns) == 5 for day in run.days)
    assert all(tuple(item.action for item in day.console_probe_actions) == ("keep", "delete") for day in run.days)
    assert all(day.owner_persisted_before_restart for day in run.days)
    assert len({day.service_instance_id for day in run.days}) == 7
    assert (tmp_path / "run.json").is_file()
    assert len(list((tmp_path / "pilot").rglob("day-*-transcript.json"))) == 7
    end_indexes = [index for index, (event, _) in enumerate(service.events) if event == "end"]
    assert all(service.events[index - 1][0] == "metrics" for index in end_indexes)


def test_orchestrator_requires_active_character_carrier_on_every_turn(
    tmp_path: Path,
) -> None:
    service = _CharacterService()
    run = SevenDayCompanionOrchestrator(
        service=service,
        lifecycle=_Lifecycle(service),
        pilot_harness=RelationshipAssistantPilotHarness(
            root_dir=tmp_path / "pilot",
            pilot_id="character-stack-seven-day",
            invited_user_ids=frozenset({"synthetic-user"}),
        ),
    ).run(
        run_id="character-stack-run",
        arm_label="correct-user-state",
        paraphrase_seed=1401,
        user_id="synthetic-user",
        schedule=_schedule(),
        user_driver=_Driver(),
        source_attestation=_character_attestation(),
        drain_slow_loop=True,
        output_path=tmp_path / "character-stack-run.json",
    )

    assert all("character_prefix=active" in turn.response_rationale_tags for day in run.days for turn in day.turns)


def test_schedule_must_cover_all_three_l4_event_families() -> None:
    schedule = _schedule()
    days = tuple(replace(day, required_event_tags=()) if day.day_index == 7 else day for day in schedule.days)
    with pytest.raises(ValueError, match="callback, emotion, and boundary"):
        replace(schedule, days=days)


def test_same_simulator_and_sut_family_is_rejected() -> None:
    with pytest.raises(ValueError, match="model families must differ"):
        replace(_attestation(), simulator_model_family="qwen")


def test_http_client_reuses_turn_path_for_typed_msc_observation() -> None:
    class RecordingHTTPClient(HTTPSevenDayCompanionService):
        def __init__(self) -> None:
            super().__init__(
                base_url="http://127.0.0.1:8765",
                user_id="msc-user",
                instance_id="msc-service",
            )
            self.request: tuple[str, str, Mapping[str, object] | None] | None = None

        def _request(
            self,
            method: str,
            path: str,
            *,
            payload: Mapping[str, object] | None = None,
        ) -> Mapping[str, object]:
            self.request = (method, path, payload)
            return {"accepted": True}

    client = RecordingHTTPClient()
    result = client.submit_observed_turn(
        session_id="dyad/1",
        user_input="observed corpus turn",
        active_speaker_id="speaker_2",
        observation_kind="dialogue",
    )

    assert result == {"accepted": True}
    assert client.request == (
        "POST",
        "/v1/sessions/dyad%2F1/turns",
        {
            "user_input": "observed corpus turn",
            "active_speaker_id": "speaker_2",
            "observation_kind": "dialogue",
        },
    )
    with pytest.raises(ValueError, match="active_speaker_id"):
        client.submit_observed_turn(
            session_id="dyad/1",
            user_input="turn",
            active_speaker_id="speaker_3",
            observation_kind="dialogue",
        )


def test_declared_day_event_must_be_emitted(tmp_path: Path) -> None:
    class MissingEventDriver(_Driver):
        def next_turn(
            self,
            *,
            day_index: int,
            exchange_index: int,
            recent_assistant_turns: Sequence[str],
        ) -> SimulatedUserTurn:
            turn = super().next_turn(
                day_index=day_index,
                exchange_index=exchange_index,
                recent_assistant_turns=recent_assistant_turns,
            )
            if day_index == 4:
                return replace(turn, event_tags=())
            return turn

    service = _Service()
    harness = RelationshipAssistantPilotHarness(
        root_dir=tmp_path,
        pilot_id="simulated-seven-day",
        invited_user_ids=frozenset({"synthetic-user"}),
    )
    orchestrator = SevenDayCompanionOrchestrator(
        service=service,
        lifecycle=_Lifecycle(service),
        pilot_harness=harness,
    )
    with pytest.raises(RuntimeError, match="missed declared event tags"):
        orchestrator.run(
            run_id="run-1",
            arm_label="correct-user-state",
            paraphrase_seed=1401,
            user_id="synthetic-user",
            schedule=_schedule(),
            user_driver=MissingEventDriver(),
            source_attestation=_attestation(),
            drain_slow_loop=True,
        )
