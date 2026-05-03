"""Dialogue benchmark user simulators — Slice A.3.

This module owns the three user-side ``DialogueUserTurnSource``
implementations:

* :class:`DeterministicUserSimulator` — closed-loop simulator that
  follows the scripted scenario stages and adapts the user prompt
  pool based on the previous turn's high-PE / acceptance signal.
* :class:`TranscriptOnlyUserSimulator` — replays a fixed transcript
  for systematic-replay benchmarks; never reacts to runtime state.
* :class:`OpenDialogueREPLReader` — interactive stdin reader used by
  the CLI ``cli.py`` open-dialogue REPL.

Implementation notes:

* All three classes implement the ``DialogueUserTurnSource`` Protocol
  defined in :mod:`volvence_zero.agent.dialogue.types`.
* :class:`DeterministicUserSimulator` calls helpers and references
  scoring thresholds defined inside :mod:`_legacy` (e.g.
  ``_turn_is_high_pe`` / ``PROOF_HIGH_PE_THRESHOLD``). Those names
  are imported lazily inside the method bodies to avoid an import
  cycle at module load time. Resolution happens at call time, by
  which point ``_legacy`` is fully loaded via the dialogue facade.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import asdict, dataclass, is_dataclass, replace
from enum import Enum
import json
from pathlib import Path
import pickle
from random import Random
from typing import Any, Protocol

from volvence_zero.agent.paper_suite import (
    ClaimVerdict,
    EvidenceBundle,
    PaperMetricSpec,
    PaperProfileSpec,
    PaperSuiteManifest,
    PaperSuiteProvenance,
    collect_paper_suite_provenance,
    export_json_artifact,
)
from volvence_zero.agent.session import AgentSessionRunner, AgentTurnResult, default_active_runner
from volvence_zero.agent.dialogue.types import (
    DialogueBenchmarkTurn,
    DialogueUserTurnSource,
    OpenDialogueEpisodeState,
    OpenDialogueScenario,
)
from volvence_zero.evaluation.backbone import (
    CrossSessionBenchmarkSuite,
    CrossSessionGrowthReport,
    EvaluationReport,
    EvaluationSnapshot,
    EvolutionDecision,
    JudgementCategory,
    MetricIntervalSummary,
    PairwiseMetricEffect,
    build_pairwise_metric_effect,
    build_metric_interval_summaries,
)
from volvence_zero.integration import FinalRolloutConfig
from volvence_zero.joint_loop import (
    ETANLJointLoop,
    JointLoopSchedule,
    PipelineConfig,
    RareHeavyArtifact,
    RareHeavyImportCheckpoint,
    RareHeavyImportResult,
    SSLRLTrainingPipeline,
)
from volvence_zero.memory import build_default_memory_store
from volvence_zero.reflection import WritebackMode
from volvence_zero.runtime import WiringLevel
from volvence_zero.substrate import (
    build_transformers_runtime_with_fallback,
    LocalSubstrateRuntimeMode,
    OpenWeightResidualRuntime,
    SubstrateFallbackMode,
    TrainingTrace,
    build_training_trace,
)
from volvence_zero.temporal import TemporalAbstractionSnapshot
from volvence_zero.temporal import (
    FullLearnedTemporalPolicy,
    HeuristicTemporalPolicy,
    LearnedLiteTemporalPolicy,
    PlaceholderTemporalPolicy,
    TemporalStep,
)

class DeterministicUserSimulator:
    def __init__(
        self,
        *,
        scenario: OpenDialogueScenario,
        seed: int = 0,
    ) -> None:
        self._scenario = scenario
        self._seed = seed
        self._rng = Random(seed)
        self._episode_state = OpenDialogueEpisodeState(scenario_id=scenario.scenario_id)

    @property
    def scenario(self) -> OpenDialogueScenario:
        return self._scenario

    @property
    def episode_state(self) -> OpenDialogueEpisodeState:
        return self._episode_state

    def next_turn(
        self,
        *,
        last_result: AgentTurnResult | None = None,
        last_turn: "DialogueBenchmarkTurn" | None = None,
    ) -> str | None:
        del last_result
        updated_state = self._advance_state(last_turn=last_turn)
        self._episode_state = updated_state
        if updated_state.completed:
            return None
        prompt_pool = self._prompt_pool(updated_state.last_stage)
        prompt_index = self._prompt_index(updated_state=updated_state, prompt_count=len(prompt_pool))
        return prompt_pool[prompt_index]

    def _advance_state(
        self,
        *,
        last_turn: "DialogueBenchmarkTurn" | None,
    ) -> OpenDialogueEpisodeState:
        from volvence_zero.agent.dialogue._legacy import (
            PROOF_HIGH_PE_THRESHOLD,
            PROOF_REWARD_THRESHOLD,
            _turn_is_high_pe,
        )
        state = self._episode_state
        if state.completed:
            return state
        if state.turn_index >= self._scenario.max_turns:
            return replace(state, completed=True, stop_reason="max-turns")
        if last_turn is None:
            return OpenDialogueEpisodeState(
                scenario_id=self._scenario.scenario_id,
                turn_index=1,
                pressure_level=1,
                adaptive_response_count=0,
                calm_turn_count=0,
                last_stage="opening",
                completed=False,
                stop_reason="running",
                user_policy_kind="runtime-linked",
            )
        high_pe = _turn_is_high_pe(
            last_turn,
            high_pe_threshold=PROOF_HIGH_PE_THRESHOLD,
            reward_threshold=PROOF_REWARD_THRESHOLD,
        )
        adaptive = (
            last_turn.joint_schedule_action != "evidence-only"
            or last_turn.rare_heavy_recommended
            or last_turn.bounded_writeback_applied
            or last_turn.reflection_promotion_eligible
        )
        calm_turn_count = state.calm_turn_count + 1 if (last_turn.acceptance_passed and not high_pe) else 0
        adaptive_response_count = state.adaptive_response_count + int(adaptive)
        if adaptive and calm_turn_count >= 2:
            return OpenDialogueEpisodeState(
                scenario_id=self._scenario.scenario_id,
                turn_index=state.turn_index,
                pressure_level=max(state.pressure_level - 1, 0),
                adaptive_response_count=adaptive_response_count,
                calm_turn_count=calm_turn_count,
                last_stage="consolidation",
                completed=True,
                stop_reason="stable-consolidation",
                user_policy_kind="runtime-linked",
            )
        if adaptive and not high_pe:
            next_stage = "consolidation" if calm_turn_count > 0 else "stabilization"
            pressure_level = max(state.pressure_level - 1, 0)
        elif adaptive:
            next_stage = "stabilization"
            pressure_level = max(state.pressure_level, 1)
        else:
            next_stage = "escalation"
            pressure_level = min(state.pressure_level + 1, 3)
        next_turn_index = state.turn_index + 1
        if next_turn_index > self._scenario.max_turns:
            return replace(state, completed=True, stop_reason="max-turns")
        return OpenDialogueEpisodeState(
            scenario_id=self._scenario.scenario_id,
            turn_index=next_turn_index,
            pressure_level=pressure_level,
            adaptive_response_count=adaptive_response_count,
            calm_turn_count=calm_turn_count,
            last_stage=next_stage,
            completed=False,
            stop_reason="running",
            user_policy_kind="runtime-linked",
        )

    def _prompt_pool(self, stage: str) -> tuple[str, ...]:
        if stage == "opening":
            return self._scenario.opening_turns
        if stage == "escalation":
            return self._scenario.escalation_turns
        if stage == "stabilization":
            return self._scenario.stabilization_turns
        return self._scenario.consolidation_turns

    def _prompt_index(self, *, updated_state: OpenDialogueEpisodeState, prompt_count: int) -> int:
        if prompt_count <= 1:
            return 0
        return (
            self._seed
            + updated_state.turn_index
            + updated_state.pressure_level
            + updated_state.adaptive_response_count
            + self._rng.randrange(prompt_count)
        ) % prompt_count


class TranscriptOnlyUserSimulator:
    def __init__(
        self,
        *,
        scenario: OpenDialogueScenario,
        seed: int = 0,
    ) -> None:
        self._scenario = scenario
        self._seed = seed
        self._rng = Random(seed)
        self._episode_state = OpenDialogueEpisodeState(
            scenario_id=scenario.scenario_id,
            user_policy_kind="transcript-only",
        )

    @property
    def scenario(self) -> OpenDialogueScenario:
        return self._scenario

    @property
    def episode_state(self) -> OpenDialogueEpisodeState:
        return self._episode_state

    def next_turn(
        self,
        *,
        last_result: AgentTurnResult | None = None,
        last_turn: "DialogueBenchmarkTurn" | None = None,
    ) -> str | None:
        del last_result
        updated_state = self._advance_state(last_turn=last_turn)
        self._episode_state = updated_state
        if updated_state.completed:
            return None
        prompt_pool = self._prompt_pool(updated_state.last_stage)
        prompt_index = self._prompt_index(
            updated_state=updated_state,
            prompt_count=len(prompt_pool),
            last_turn=last_turn,
        )
        return prompt_pool[prompt_index]

    def _advance_state(
        self,
        *,
        last_turn: "DialogueBenchmarkTurn" | None,
    ) -> OpenDialogueEpisodeState:
        state = self._episode_state
        if state.completed:
            return state
        if state.turn_index >= self._scenario.max_turns:
            return replace(state, completed=True, stop_reason="max-turns")
        if last_turn is None:
            return OpenDialogueEpisodeState(
                scenario_id=self._scenario.scenario_id,
                turn_index=1,
                pressure_level=1,
                adaptive_response_count=0,
                calm_turn_count=0,
                last_stage="opening",
                completed=False,
                stop_reason="running",
                user_policy_kind="transcript-only",
            )
        next_turn_index = state.turn_index + 1
        if next_turn_index > self._scenario.max_turns:
            return replace(state, completed=True, stop_reason="max-turns")
        stage = self._stage_for_turn(next_turn_index)
        pressure_level = {
            "opening": 1,
            "escalation": 2,
            "stabilization": 1,
            "consolidation": 0,
        }[stage]
        completed = stage == "consolidation" and next_turn_index >= self._scenario.max_turns
        return OpenDialogueEpisodeState(
            scenario_id=self._scenario.scenario_id,
            turn_index=next_turn_index,
            pressure_level=pressure_level,
            adaptive_response_count=0,
            calm_turn_count=state.calm_turn_count + int(len(last_turn.assistant_response_text.strip()) > 0),
            last_stage=stage,
            completed=completed,
            stop_reason="transcript-policy-complete" if completed else "running",
            user_policy_kind="transcript-only",
        )

    def _stage_for_turn(self, turn_index: int) -> str:
        if turn_index <= 1:
            return "opening"
        if turn_index <= max(2, self._scenario.max_turns // 2):
            return "escalation"
        if turn_index < self._scenario.max_turns:
            return "stabilization"
        return "consolidation"

    def _prompt_pool(self, stage: str) -> tuple[str, ...]:
        if stage == "opening":
            return self._scenario.opening_turns
        if stage == "escalation":
            return self._scenario.escalation_turns
        if stage == "stabilization":
            return self._scenario.stabilization_turns
        return self._scenario.consolidation_turns

    def _prompt_index(
        self,
        *,
        updated_state: OpenDialogueEpisodeState,
        prompt_count: int,
        last_turn: "DialogueBenchmarkTurn" | None,
    ) -> int:
        if prompt_count <= 1:
            return 0
        transcript_length = 0 if last_turn is None else len(last_turn.user_input) + len(last_turn.assistant_response_text)
        return (
            self._seed
            + updated_state.turn_index
            + transcript_length
            + self._rng.randrange(prompt_count)
        ) % prompt_count


class OpenDialogueREPLReader:
    def __init__(self, *, turn_source: DialogueUserTurnSource) -> None:
        self._turn_source = turn_source
        self._last_result: AgentTurnResult | None = None
        self._last_turn: DialogueBenchmarkTurn | None = None
        self._turn_count = 0

    @property
    def scenario(self) -> OpenDialogueScenario:
        return self._turn_source.scenario

    @property
    def episode_state(self) -> OpenDialogueEpisodeState:
        return self._turn_source.episode_state

    def __call__(self) -> str:
        user_input = self._turn_source.next_turn(
            last_result=self._last_result,
            last_turn=self._last_turn,
        )
        if user_input is None:
            raise EOFError(self._turn_source.episode_state.stop_reason)
        return user_input

    def observe_result(self, result: AgentTurnResult) -> None:
        self._turn_count += 1
        self._last_result = result
        self._last_turn = dialogue_turn_from_result(
            turn_index=self._turn_count,
            user_input=result.user_input,
            result=result,
        )
