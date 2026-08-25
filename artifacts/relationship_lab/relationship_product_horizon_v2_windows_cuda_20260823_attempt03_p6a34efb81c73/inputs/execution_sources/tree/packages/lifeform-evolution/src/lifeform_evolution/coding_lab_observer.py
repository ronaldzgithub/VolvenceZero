"""Coding-lab SHADOW observer (Packet 1).

Replays coding-lab episode trajectories through an observer ``Brain``
using ONLY official facade channels:

* interaction stream  -> ``BrainSession.run_turn`` (text observations);
* tool outcomes       -> ``BrainSession.submit_tool_result`` (typed,
  feeds execution_result / open_loop / belief owners and next-turn PE
  settlement via ``EnvironmentOutcome``);
* episode terminal    -> ``BrainSession.submit_dialogue_outcome`` with
  the task-execution vocabulary (``TASK_VERIFIED`` / ``TASK_REGRESSED``,
  source=ENVIRONMENT) — the single legal external-outcome channel.

SHADOW invariants:

* the observer runs POST-HOC on trajectory logs — it cannot influence
  the hand by construction;
* forecasts (行动前下注) are read from the PE owner's published
  ``next_prediction`` and recorded BEFORE the oracle outcome is
  submitted (bet-then-settle ordering is asserted, not assumed);
* zero ACTIVE wiring changes: the observer Brain is a private instance
  whose lifecycle begins and ends inside the observation run.

Scripted-hand trajectories carry ground-truth metadata
(``scripted_mode``); the observer NEVER forwards event metadata to the
brain — only actions, parameters and tool results.
"""

from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass
from typing import Any

from volvence_zero.brain import Brain, BrainConfig, BrainSession
from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidenceSource,
    DialogueExternalOutcomeKind,
)
from volvence_zero.memory import (
    MemoryEntry,
    MemoryStratum,
    MemoryWriteRequest,
    RetrievalQuery,
    StaticIdentityProvider,
    Track,
    UserIdentity,
)

from lifeform_domain_coding.lab.trajectory import read_trajectory

_OBSERVER_USER_ID = "coding-lab-observer"


@dataclass(frozen=True)
class ObserverBet:
    """One owner-published pre-outcome forecast."""

    turn_index: int
    predicted_task_progress: float
    predicted_action_payoff: float
    confidence: float
    prediction_id: str
    monotonic_seconds: float


@dataclass(frozen=True)
class EpisodeObservation:
    """Everything the observer recorded for one episode."""

    chain_id: str
    episode_index: int
    task_id: str
    category: str
    passed: bool
    acceptance_passed: bool
    regression_passed: bool
    invariant_violations: tuple[str, ...]
    bet_at_task_presented: ObserverBet
    bet_pre_oracle: ObserverBet
    outcome_submitted_monotonic: float
    settled_task_progress: float
    settled_signed_reward: float
    settled_task_error: float
    settled_magnitude: float
    external_outcome_refs: tuple[str, ...]
    turns_used: int

    def __post_init__(self) -> None:
        if self.bet_pre_oracle.monotonic_seconds >= self.outcome_submitted_monotonic:
            raise ValueError(
                "bet-then-settle ordering violated: the pre-oracle bet must be "
                "recorded strictly before the outcome submission"
            )


def _pe_snapshot(session: BrainSession) -> Any:
    published = session.runner.upstream_snapshots.get("prediction_error")
    if published is None:
        raise RuntimeError("observer brain published no prediction_error snapshot")
    return published.value


def _bet_from_snapshot(session: BrainSession) -> ObserverBet:
    value = _pe_snapshot(session)
    next_prediction = value.next_prediction
    return ObserverBet(
        turn_index=value.turn_index,
        predicted_task_progress=float(next_prediction.predicted_task_progress),
        predicted_action_payoff=float(next_prediction.predicted_action_payoff),
        confidence=float(next_prediction.confidence),
        prediction_id=str(next_prediction.prediction_id),
        monotonic_seconds=time.monotonic(),
    )


def _tool_summary(event_payload: dict[str, Any]) -> tuple[str, str, bool]:
    tool_name = str(event_payload.get("tool_name", ""))
    succeeded = bool(event_payload.get("succeeded", False))
    result = event_payload.get("result", {})
    if succeeded:
        detail_bits = []
        for key in ("resolved_path", "exit_code", "bytes_written", "matches", "entries"):
            if key in result:
                value = result[key]
                rendered = f"{len(value)} items" if isinstance(value, list) else str(value)
                detail_bits.append(f"{key}={rendered}")
        detail = "; ".join(detail_bits) or "ok"
    else:
        detail = f"{result.get('error_class', 'Error')}: {result.get('error_detail', '')}"
    if tool_name == "run_test" and succeeded:
        exit_code = int(result.get("exit_code", -1))
        succeeded = exit_code == 0
        detail = f"pytest exit_code={exit_code}; " + str(result.get("stdout", ""))[:400]
    return tool_name, detail[:800], succeeded


class CodingLabChainObserver:
    """SHADOW observer for one chain: fresh Brain, scoped persistent memory."""

    def __init__(
        self,
        *,
        chain_id: str,
        brain_state_root: pathlib.Path,
        user_id: str = _OBSERVER_USER_ID,
    ) -> None:
        self._chain_id = chain_id
        identity = UserIdentity(user_id=user_id, scope_key=user_id)
        self._brain = Brain(
            BrainConfig(memory_scope_root_dir=str(brain_state_root)),
            identity_provider=StaticIdentityProvider(identity=identity),
        )
        self._session = self._brain.create_session(session_id=f"coding-lab-{chain_id}")
        self._turn_counter = 0

    @property
    def session(self) -> BrainSession:
        return self._session

    async def _turn(self, text: str) -> None:
        await self._session.run_turn_async(text)
        self._turn_counter += 1

    def memory_entry_count(self) -> int:
        return self._session.memory_entry_count()

    def recall_experience(
        self,
        *,
        hint: str,
        facets: tuple[str, ...] = (),
        limit: int = 8,
    ) -> tuple[MemoryEntry, ...]:
        """Retrieve persisted experience through the memory owner's API.

        Packet 2 injection-pack support. Uses ``MemoryStore.retrieve``
        directly (official owner contract) instead of a conversational
        turn: a turn writes the hint into the TRANSIENT stratum first,
        and the just-written hint then wins retrieval against its own
        query text, crowding every episodic experience entry out of the
        top-k (observed in the 2026-08-12 smoke: all 5 retrieved entries
        were copies of the recall hint itself). Restricting strata to
        EPISODIC + DURABLE asks the owner for *experience*, never the
        current input buffer.

        The hint and facets must contain only harness-known task
        metadata (category, target files); see
        ``coding_lab_arms.recall_for_task`` for the leak rationale.
        """

        result = self._session.runner.memory_store.retrieve(
            RetrievalQuery(
                text=hint,
                strata=(MemoryStratum.EPISODIC, MemoryStratum.DURABLE),
                facets=facets,
                limit=limit,
            ),
            timestamp_ms=int(time.time() * 1000),
        )
        return result.entries

    def persist(self) -> bool:
        """Persist memory to the scoped backend (cross-process recovery)."""

        self._session.persist_owners()
        return bool(self._session.runner.memory_store.save_to_backend())

    async def observe_episode(
        self,
        *,
        episode_index: int,
        trajectory_path: pathlib.Path,
    ) -> EpisodeObservation:
        """Replay one logged episode through the observer brain."""

        events = read_trajectory(trajectory_path)
        task_presented = next(
            event for event in events if event["event_type"] == "task_presented"
        )
        oracle_event = next(
            event for event in events if event["event_type"] == "oracle_outcome"
        )
        task_payload = task_presented["payload"]
        oracle_payload = oracle_event["payload"]
        task_id = str(task_payload["task_id"])
        category = str(task_payload["category"])

        await self._turn(
            f"[coding-lab episode {episode_index}] New task {task_id} ({category}): "
            f"{task_payload['description']}"
        )
        bet_at_present = _bet_from_snapshot(self._session)

        step_events = [
            event
            for event in events
            if event["event_type"] in ("hand_decision", "tool_result")
        ]
        tool_sequence = 0
        for event in step_events:
            payload = event["payload"]
            if event["event_type"] == "hand_decision":
                if payload.get("kind") == "submit":
                    await self._turn(
                        "[hand] submitted the change for evaluation. "
                        f"Note: {str(payload.get('note', ''))[:200]}"
                    )
                continue
            tool_name, detail, succeeded = _tool_summary(payload)
            status = "succeeded" if succeeded else "failed"
            event_id = f"{task_id}:tool:{tool_sequence:03d}"
            self._session.submit_tool_result(
                event_id=event_id,
                tool_name=tool_name,
                action_id=f"{tool_name}:{event_id}",
                status=status,
                summary=f"{tool_name} {status}",
                detail=detail,
                confidence=1.0,
            )
            await self._turn(f"[hand] {tool_name} -> {status}: {detail[:200]}")
            tool_sequence += 1

        bet_pre_oracle = _bet_from_snapshot(self._session)

        passed = bool(oracle_payload["passed"])
        kind = (
            DialogueExternalOutcomeKind.TASK_VERIFIED
            if passed
            else DialogueExternalOutcomeKind.TASK_REGRESSED
        )
        violations = tuple(str(item) for item in oracle_payload.get("invariant_violations", ()))
        description = (
            f"oracle settled {task_id}: acceptance="
            f"{bool(oracle_payload['acceptance_passed'])} regression="
            f"{bool(oracle_payload['regression_passed'])}"
            + (f" invariant_violations={','.join(violations)}" if violations else "")
        )
        self._session.submit_dialogue_outcome(
            kind=kind,
            source=DialogueExternalOutcomeEvidenceSource.ENVIRONMENT,
            confidence=1.0,
            evidence_ref=f"trajectory:{trajectory_path.name}",
            description=description,
            action_turn_index=max(self._turn_counter - 1, 0),
        )
        outcome_submitted = time.monotonic()
        await self._turn(f"[oracle] episode settled: {'PASS' if passed else 'FAIL'}. {description}")
        # Episode = scene boundary: settle the background session-post slow
        # loop so reflection consolidation (memory promotion out of the
        # transient buffer) lands before the next episode's recall.
        await self._session.drain_session_post_slow_loop()
        # Persist the lived episode as EPISODIC experience through the memory
        # owner's formal write API (the plan-designated adapter channel;
        # storage / retrieval / decay stay owner-decided). Failures write
        # stronger than passes — they are the experiences the next recall
        # must be able to surface.
        # Post-submit CI evidence: assertion heads make the failure
        # actionable (bare violation ids proved uninterpretable to the
        # hand in the 2026-08-13 formal run). Same granularity as the
        # steelman transcript's [oracle-failure] lines — arms symmetric.
        failure_details = tuple(
            str(item) for item in oracle_payload.get("failure_details", ())
        )
        detail_text = ""
        if not passed and failure_details:
            detail_text = " | ci evidence: " + " ; ".join(failure_details[:3])
        self._session.runner.memory_store.write(
            MemoryWriteRequest(
                content=(
                    f"[coding-lab experience] task={task_id} category={category} "
                    f"outcome={'PASS' if passed else 'FAIL'}"
                    + (f" invariant_violations={','.join(violations)}" if violations else "")
                    + detail_text
                    + f" | task was: {str(task_payload['description'])[:500]}"
                ),
                track=Track.WORLD,
                stratum=MemoryStratum.EPISODIC,
                tags=("coding-lab", f"category:{category}", f"outcome:{'pass' if passed else 'fail'}"),
                strength=0.6 if passed else 0.85,
            ),
            timestamp_ms=int(time.time() * 1000),
        )

        settled = _pe_snapshot(self._session)
        return EpisodeObservation(
            chain_id=self._chain_id,
            episode_index=episode_index,
            task_id=task_id,
            category=category,
            passed=passed,
            acceptance_passed=bool(oracle_payload["acceptance_passed"]),
            regression_passed=bool(oracle_payload["regression_passed"]),
            invariant_violations=violations,
            bet_at_task_presented=bet_at_present,
            bet_pre_oracle=bet_pre_oracle,
            outcome_submitted_monotonic=outcome_submitted,
            settled_task_progress=float(settled.actual_outcome.task_progress),
            settled_signed_reward=float(settled.error.signed_reward),
            settled_task_error=float(settled.error.task_error),
            settled_magnitude=float(settled.error.magnitude),
            external_outcome_refs=tuple(settled.actual_outcome.external_outcome_refs),
            turns_used=self._turn_counter,
        )


@dataclass(frozen=True)
class ChainObservationResult:
    chain_id: str
    observations: tuple[EpisodeObservation, ...]
    persisted: bool
    memory_entry_count_before_restart: int


async def observe_calibration_chain(
    *,
    chain_id: str,
    trajectories_dir: pathlib.Path,
    brain_state_root: pathlib.Path,
) -> ChainObservationResult:
    """Observe every episode trajectory of one calibration chain, in order."""

    trajectory_paths = sorted(pathlib.Path(trajectories_dir).glob("episode-*.jsonl"))
    if not trajectory_paths:
        raise FileNotFoundError(f"no episode trajectories under {trajectories_dir!s}")
    observer = CodingLabChainObserver(chain_id=chain_id, brain_state_root=brain_state_root)
    observations = []
    for episode_index, trajectory_path in enumerate(trajectory_paths):
        observations.append(
            await observer.observe_episode(
                episode_index=episode_index,
                trajectory_path=trajectory_path,
            )
        )
    persisted = observer.persist()
    return ChainObservationResult(
        chain_id=chain_id,
        observations=tuple(observations),
        persisted=persisted,
        memory_entry_count_before_restart=observer.memory_entry_count(),
    )


def recovered_memory_entry_count(
    *,
    chain_id: str,
    brain_state_root: pathlib.Path,
) -> int:
    """Cross-process recovery probe: fresh Brain over the same scoped root.

    A brand-new Brain + session over the same ``memory_scope_root_dir``
    must hydrate the persisted store; the caller compares this count with
    the pre-restart count.
    """

    observer = CodingLabChainObserver(
        chain_id=f"{chain_id}-recovered",
        brain_state_root=brain_state_root,
    )
    return observer.memory_entry_count()


__all__ = [
    "ChainObservationResult",
    "CodingLabChainObserver",
    "EpisodeObservation",
    "ObserverBet",
    "observe_calibration_chain",
    "recovered_memory_entry_count",
]
