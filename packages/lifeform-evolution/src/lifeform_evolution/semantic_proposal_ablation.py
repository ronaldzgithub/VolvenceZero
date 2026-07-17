"""LLM-proposal dependency ablation (experiment 2 of
``docs/specs/semantic-grounding-evidence.md``).

Answers: with the same substrate, seeds and scenario scripts, how much
does semantic-owner state quality degrade when the semantic proposal
channel is replaced by ``NoOpSemanticProposalRuntime``?

Design (spec-frozen):

* Two matched arms — ``semantic-proposal-on`` (a real semantic
  perception runtime; LLM-backed in the hf lane, scripted in the
  synthetic smoke lane) vs ``semantic-proposal-off`` (explicit NoOp).
  Both arms run the SAME scripted probe cases, typed external events
  and (optional) shared substrate runtime. The switch is channel-level:
  with a non-LLM runtime the session never derives the ToM /
  common-ground LLM proposal runtimes either.
* Ground truth is declared per case as typed ``SlotExpectation``
  checks against published owner snapshot fields. Checks target
  OBSERVE-immune fields (blocked / deferred / completed statuses,
  revision counters, the confidence-floored commitment slot) because
  the NoOp runtime emits a low-confidence OBSERVE record every turn,
  which would make plain "has any active record" checks trivially true
  in both arms.
* Expectations are stratified by evidence channel:
  ``proposal-channel`` (only a semantic perception runtime can satisfy
  them from the user turn) vs ``typed-event`` (satisfied by external
  typed events through the adapter runtime — these must hold in BOTH
  arms, they are the invariance control).
* Readout-only (R12): the report is a non-gating reference artifact.
  Nothing here writes owner state beyond running normal turns, and the
  result never enters a reward or learning path.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from volvence_zero.agent.semantic_grounding import (
    SemanticGroundingReport,
    SemanticGroundingTurnCapture,
    SemanticGroundingTurnEvidence,
    build_semantic_grounding_report,
)
from volvence_zero.semantic_state import (
    ExternalSemanticEventBatch,
    NoOpSemanticProposalRuntime,
    SemanticProposal,
    SemanticProposalBatch,
    SemanticProposalOperation,
    SemanticProposalRuntime,
    semantic_events_from_profile,
    semantic_events_from_task_event,
    semantic_events_from_tool_result,
)

SEMANTIC_PROPOSAL_ABLATION_SCHEMA_VERSION = "semantic-proposal-ablation.v1"

ON_ARM_ID = "semantic-proposal-on"
OFF_ARM_ID = "semantic-proposal-off"

PROPOSAL_CHANNEL = "proposal-channel"
TYPED_EVENT_CHANNEL = "typed-event"

#: Spec verdict thresholds on the overall lifecycle hit-rate drop.
DEPENDENT_DROP_THRESHOLD = 0.3
SUFFICIENT_DROP_THRESHOLD = 0.1


class SemanticProposalAblationError(RuntimeError):
    """Raised when the ablation harness input violates its contract."""


# ---------------------------------------------------------------------------
# Probe contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScriptedProposalSpec:
    """What an ideal semantic perception channel would extract from one
    scripted user turn, for one owner slot. The synthetic on-arm runtime
    replays exactly these; the hf on-arm replaces the whole script with
    the live ``LLMSemanticProposalRuntime``."""

    slot: str
    operation: SemanticProposalOperation
    summary: str
    confidence: float = 0.62
    control_signal: float = 0.30


@dataclass(frozen=True)
class AblationProbeTurn:
    user_input: str
    scripted_proposals: tuple[ScriptedProposalSpec, ...] = ()
    typed_events: tuple[ExternalSemanticEventBatch, ...] = ()


@dataclass(frozen=True)
class SlotExpectation:
    """One typed ground-truth check against a published owner snapshot.

    ``field_name`` is read directly off the slot's snapshot value
    (tuples are measured by length, numbers by value, bools as 0/1);
    a missing attribute is contract drift and raises. ``channel``
    declares which evidence path is supposed to satisfy the check.
    """

    expectation_id: str
    slot: str
    after_turn: int
    field_name: str
    min_value: float
    channel: str

    def __post_init__(self) -> None:
        if self.channel not in {PROPOSAL_CHANNEL, TYPED_EVENT_CHANNEL}:
            raise ValueError(
                f"SlotExpectation.channel must be {PROPOSAL_CHANNEL!r} or "
                f"{TYPED_EVENT_CHANNEL!r}, got {self.channel!r}."
            )


@dataclass(frozen=True)
class AblationProbeCase:
    case_id: str
    description: str
    turns: tuple[AblationProbeTurn, ...]
    expectations: tuple[SlotExpectation, ...]

    def __post_init__(self) -> None:
        for expectation in self.expectations:
            if not 0 <= expectation.after_turn < len(self.turns):
                raise ValueError(
                    f"Expectation {expectation.expectation_id!r} in case "
                    f"{self.case_id!r} points at turn "
                    f"{expectation.after_turn}, but the case has "
                    f"{len(self.turns)} turns."
                )


# ---------------------------------------------------------------------------
# Scripted on-arm runtime (synthetic smoke lane)
# ---------------------------------------------------------------------------


class ScriptedSemanticProposalRuntime(SemanticProposalRuntime):
    """Deterministic stand-in for a semantic perception channel.

    Keyed on the exact scripted ``user_input`` string (each probe turn
    uses a unique utterance, so this is protocol correspondence with the
    probe script, not keyword inference). Unscripted inputs yield an
    empty batch.
    """

    runtime_id = "semantic-scripted-probe"

    def __init__(self, cases: tuple[AblationProbeCase, ...]) -> None:
        script: dict[str, dict[str, tuple[ScriptedProposalSpec, ...]]] = {}
        for case in cases:
            for turn in case.turns:
                if turn.user_input in script:
                    raise SemanticProposalAblationError(
                        "Probe turns must use unique user_input strings; "
                        f"duplicate: {turn.user_input!r}."
                    )
                per_slot: dict[str, list[ScriptedProposalSpec]] = {}
                for spec in turn.scripted_proposals:
                    per_slot.setdefault(spec.slot, []).append(spec)
                script[turn.user_input] = {
                    slot: tuple(specs) for slot, specs in per_slot.items()
                }
        self._script = script
        self._emission_counter = 0

    def propose(
        self,
        *,
        target_slot: str,
        user_input: str | None,
        substrate_snapshot: Any,
        memory_snapshot: Any,
        previous_snapshot: Any,
        turn_index: int,
    ) -> SemanticProposalBatch:
        del substrate_snapshot, memory_snapshot, previous_snapshot
        specs = self._script.get(user_input or "", {}).get(target_slot, ())
        proposals = []
        for spec in specs:
            self._emission_counter += 1
            proposals.append(
                SemanticProposal(
                    proposal_id=(
                        f"scripted:{target_slot}:{spec.operation.value}:"
                        f"{self._emission_counter}"
                    ),
                    target_slot=target_slot,
                    operation=spec.operation,
                    summary=spec.summary[:160],
                    detail=spec.summary[:320],
                    confidence=spec.confidence,
                    evidence=(user_input or "")[:320],
                    control_signal=spec.control_signal,
                )
            )
        return SemanticProposalBatch(
            proposals=tuple(proposals),
            runtime_id=self.runtime_id,
            schema_version=1,
            description=(
                f"Scripted probe runtime emitted {len(proposals)} "
                f"proposal(s) for {target_slot}."
            ),
        )


# ---------------------------------------------------------------------------
# Default 9-slot probe cases
# ---------------------------------------------------------------------------


def _proposal_expectation(
    expectation_id: str, slot: str, after_turn: int, field_name: str, min_value: float = 1.0
) -> SlotExpectation:
    return SlotExpectation(
        expectation_id=expectation_id,
        slot=slot,
        after_turn=after_turn,
        field_name=field_name,
        min_value=min_value,
        channel=PROPOSAL_CHANNEL,
    )


def _typed_expectation(
    expectation_id: str, slot: str, after_turn: int, field_name: str, min_value: float = 1.0
) -> SlotExpectation:
    return SlotExpectation(
        expectation_id=expectation_id,
        slot=slot,
        after_turn=after_turn,
        field_name=field_name,
        min_value=min_value,
        channel=TYPED_EVENT_CHANNEL,
    )


_CASE_COMMITMENT_ARC = AblationProbeCase(
    case_id="commitment-arc",
    description=(
        "Commitment advocacy -> completion -> deferral plus plan revision "
        "and a self-reported execution completion; all checks are "
        "OBSERVE-immune lifecycle fields."
    ),
    turns=(
        AblationProbeTurn(
            user_input="Please commit to sending me the harbor report by Friday.",
            scripted_proposals=(
                ScriptedProposalSpec(
                    slot="commitment",
                    operation=SemanticProposalOperation.CREATE,
                    summary="send-harbor-report-by-friday",
                ),
            ),
        ),
        AblationProbeTurn(
            user_input="That report really matters; treat it as a firm promise.",
            scripted_proposals=(
                ScriptedProposalSpec(
                    slot="commitment",
                    operation=SemanticProposalOperation.ACTIVATE,
                    summary="harbor-report-promise-activated",
                ),
                ScriptedProposalSpec(
                    slot="plan_intent",
                    operation=SemanticProposalOperation.REVISE,
                    summary="reprioritize-week-around-report",
                ),
            ),
        ),
        AblationProbeTurn(
            user_input="I received the harbor report; that promise is fulfilled.",
            scripted_proposals=(
                ScriptedProposalSpec(
                    slot="commitment",
                    operation=SemanticProposalOperation.COMPLETE,
                    summary="harbor-report-delivered",
                    confidence=0.66,
                ),
                ScriptedProposalSpec(
                    slot="open_loop",
                    operation=SemanticProposalOperation.CLOSE,
                    summary="close-report-followup",
                ),
            ),
        ),
        AblationProbeTurn(
            user_input="Let's put the crew briefing on hold until next month.",
            scripted_proposals=(
                ScriptedProposalSpec(
                    slot="commitment",
                    operation=SemanticProposalOperation.DEFER,
                    summary="crew-briefing-deferred",
                    confidence=0.52,
                ),
            ),
        ),
        AblationProbeTurn(
            user_input="I ran the checklist myself this morning and it all worked.",
            scripted_proposals=(
                ScriptedProposalSpec(
                    slot="execution_result",
                    operation=SemanticProposalOperation.COMPLETE,
                    summary="user-ran-checklist-success",
                ),
            ),
        ),
    ),
    expectations=(
        _proposal_expectation("commitment-created", "commitment", 0, "active_commitments"),
        _proposal_expectation("commitment-advocated", "commitment", 1, "advocacy_proposed_count"),
        _proposal_expectation("plan-revised", "plan_intent", 1, "plan_revision_count"),
        _proposal_expectation("commitment-completed", "commitment", 2, "outcome_completed_count"),
        _proposal_expectation("loop-closed", "open_loop", 2, "closure_refs"),
        _proposal_expectation("commitment-deferred", "commitment", 3, "followup_defer_only_count"),
        _proposal_expectation("execution-completed", "execution_result", 4, "completed_actions"),
    ),
)

_CASE_EMOTIONAL_REPAIR = AblationProbeCase(
    case_id="emotional-repair",
    description=(
        "Relational rupture -> sensitive boundary -> contradiction -> "
        "repair -> goal deferral; checks target blocked / completed / "
        "deferred statuses only."
    ),
    turns=(
        AblationProbeTurn(
            user_input="That reply felt dismissive and it honestly upset me.",
            scripted_proposals=(
                ScriptedProposalSpec(
                    slot="relationship_state",
                    operation=SemanticProposalOperation.BLOCK,
                    summary="tension-dismissive-reply",
                ),
            ),
        ),
        AblationProbeTurn(
            user_input="Please stop pressing me about my health; that topic is sensitive.",
            scripted_proposals=(
                ScriptedProposalSpec(
                    slot="user_model",
                    operation=SemanticProposalOperation.BLOCK,
                    summary="sensitive-topic-health",
                ),
            ),
        ),
        AblationProbeTurn(
            user_input="That contradicts what you told me about the schedule yesterday.",
            scripted_proposals=(
                ScriptedProposalSpec(
                    slot="belief_assumption",
                    operation=SemanticProposalOperation.BLOCK,
                    summary="schedule-contradiction",
                ),
            ),
        ),
        AblationProbeTurn(
            user_input="Thank you for adjusting; the tension from earlier feels resolved.",
            scripted_proposals=(
                ScriptedProposalSpec(
                    slot="relationship_state",
                    operation=SemanticProposalOperation.COMPLETE,
                    summary="repair-acknowledged",
                    confidence=0.64,
                ),
            ),
        ),
        AblationProbeTurn(
            user_input="Let's shelve the fitness goal until my workload settles.",
            scripted_proposals=(
                ScriptedProposalSpec(
                    slot="goal_value",
                    operation=SemanticProposalOperation.DEFER,
                    summary="fitness-goal-shelved",
                    confidence=0.58,
                ),
            ),
        ),
    ),
    expectations=(
        _proposal_expectation("tension-registered", "relationship_state", 0, "unresolved_tension_count"),
        _proposal_expectation("sensitive-boundary", "user_model", 1, "sensitive_boundaries"),
        _proposal_expectation("contradiction-flagged", "belief_assumption", 2, "contradiction_refs"),
        _proposal_expectation("repair-recorded", "relationship_state", 3, "recent_repair_count"),
        _proposal_expectation("goal-deferred", "goal_value", 4, "deferred_goal_count"),
    ),
)

_CASE_BOUNDARY_GOAL = AblationProbeCase(
    case_id="boundary-goal",
    description=(
        "Hard boundary -> goal conflict -> consent revocation -> plan "
        "constraint -> execution failure; blocked / closed statuses only."
    ),
    turns=(
        AblationProbeTurn(
            user_input="Never contact my family on my behalf; that is a hard boundary.",
            scripted_proposals=(
                ScriptedProposalSpec(
                    slot="boundary_consent",
                    operation=SemanticProposalOperation.BLOCK,
                    summary="no-family-contact",
                ),
            ),
        ),
        AblationProbeTurn(
            user_input="My goals are pulling against each other: shipping fast versus doing it right.",
            scripted_proposals=(
                ScriptedProposalSpec(
                    slot="goal_value",
                    operation=SemanticProposalOperation.BLOCK,
                    summary="speed-vs-quality-conflict",
                ),
            ),
        ),
        AblationProbeTurn(
            user_input="I withdraw the permission I gave you to read my calendar.",
            scripted_proposals=(
                ScriptedProposalSpec(
                    slot="boundary_consent",
                    operation=SemanticProposalOperation.CLOSE,
                    summary="calendar-consent-revoked",
                ),
            ),
        ),
        AblationProbeTurn(
            user_input="The vendor freeze blocks the migration plan for now.",
            scripted_proposals=(
                ScriptedProposalSpec(
                    slot="plan_intent",
                    operation=SemanticProposalOperation.BLOCK,
                    summary="migration-blocked-vendor-freeze",
                ),
            ),
        ),
        AblationProbeTurn(
            user_input="The export failed when I tried it on my own machine.",
            scripted_proposals=(
                ScriptedProposalSpec(
                    slot="execution_result",
                    operation=SemanticProposalOperation.BLOCK,
                    summary="export-failed-user-machine",
                ),
            ),
        ),
    ),
    expectations=(
        _proposal_expectation("boundary-denied", "boundary_consent", 0, "denied_boundaries"),
        _proposal_expectation("goal-conflicted", "goal_value", 1, "conflicted_goal_count"),
        _proposal_expectation("consent-revoked", "boundary_consent", 2, "revocation_count"),
        _proposal_expectation("plan-constrained", "plan_intent", 3, "active_constraints"),
        _proposal_expectation("execution-failed", "execution_result", 4, "failed_actions"),
    ),
)

_CASE_TYPED_EVENT_INVARIANCE = AblationProbeCase(
    case_id="typed-event-invariance",
    description=(
        "Structured external events (task / tool / profile) drive the "
        "owners through the adapter runtime, which is active in BOTH "
        "arms. These checks must hold regardless of the LLM channel; a "
        "drop here indicates a harness defect, not LLM dependence."
    ),
    turns=(
        AblationProbeTurn(
            user_input="Here is the weekly task update from the tracker.",
            typed_events=(
                semantic_events_from_task_event(
                    event_id="ablation-task-pending",
                    task_id="task-harbor-report",
                    status="pending",
                    summary="Prepare the weekly harbor report",
                    detail="Draft and review the harbor operations report.",
                    commitment_ref="commit-harbor-report",
                ),
            ),
        ),
        AblationProbeTurn(
            user_input="The report generator tool just finished its run.",
            typed_events=(
                semantic_events_from_tool_result(
                    event_id="ablation-tool-ok",
                    tool_name="report-generator",
                    action_id="generate-harbor-report",
                    status="succeeded",
                    summary="Harbor report generated",
                    detail="Report rendered and stored to the shared drive.",
                ),
            ),
        ),
        AblationProbeTurn(
            user_input="The mail delivery job reported its status as well.",
            typed_events=(
                semantic_events_from_tool_result(
                    event_id="ablation-tool-fail",
                    tool_name="mail-sender",
                    action_id="send-harbor-report",
                    status="failed",
                    summary="Report email failed to send",
                    detail="SMTP relay rejected the message.",
                ),
            ),
        ),
        AblationProbeTurn(
            user_input="My assistant synced my profile preferences to you.",
            typed_events=(
                semantic_events_from_profile(
                    event_id="ablation-profile",
                    source="assistant-sync",
                    preferences=("prefers async summaries",),
                    goals=("keep the harbor operation on schedule",),
                    consent_denials=("no outreach to the family",),
                    relationship_note="long-standing working relationship",
                ),
            ),
        ),
        AblationProbeTurn(
            user_input="The tracker marked the harbor report task as done.",
            typed_events=(
                semantic_events_from_task_event(
                    event_id="ablation-task-done",
                    task_id="task-harbor-report",
                    status="completed",
                    summary="Harbor report task completed",
                    detail="Report delivered and acknowledged.",
                    commitment_ref="commit-harbor-report",
                ),
            ),
        ),
    ),
    expectations=(
        _typed_expectation("task-commitment-created", "commitment", 0, "active_commitments"),
        _typed_expectation("tool-success-recorded", "execution_result", 1, "completed_actions"),
        _typed_expectation("tool-failure-recorded", "execution_result", 2, "failed_actions"),
        _typed_expectation("profile-denial-recorded", "boundary_consent", 3, "denied_boundaries"),
        _typed_expectation("task-commitment-completed", "commitment", 4, "outcome_completed_count"),
    ),
)

DEFAULT_ABLATION_PROBE_CASES: tuple[AblationProbeCase, ...] = (
    _CASE_COMMITMENT_ARC,
    _CASE_EMOTIONAL_REPAIR,
    _CASE_BOUNDARY_GOAL,
    _CASE_TYPED_EVENT_INVARIANCE,
)


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpectationOutcome:
    expectation_id: str
    case_id: str
    slot: str
    channel: str
    after_turn: int
    field_name: str
    min_value: float
    observed_value: float
    hit: bool


@dataclass(frozen=True)
class SlotArmMetrics:
    slot: str
    expectation_total: int
    expectation_hits: int
    hit_rate: float
    nonempty_turn_ratio: float


@dataclass(frozen=True)
class AblationArmResult:
    arm_id: str
    runtime_id: str
    turn_count: int
    overall_hit_rate: float
    proposal_channel_hit_rate: float
    typed_event_hit_rate: float
    mean_pe_magnitude: float
    slot_metrics: tuple[SlotArmMetrics, ...]
    expectation_outcomes: tuple[ExpectationOutcome, ...]
    grounding_verdict: str


@dataclass(frozen=True)
class SlotDependencyDelta:
    slot: str
    on_hit_rate: float
    off_hit_rate: float
    hit_rate_drop: float


@dataclass(frozen=True)
class BootstrapDropCI:
    mean_drop: float
    ci_low: float
    ci_high: float
    resample_count: int
    seed: int


@dataclass(frozen=True)
class SemanticProposalAblationReport:
    schema_version: str
    artifact_kind: str
    non_gating: bool
    substrate_fingerprint: str
    case_ids: tuple[str, ...]
    on_arm: AblationArmResult
    off_arm: AblationArmResult
    slot_dependencies: tuple[SlotDependencyDelta, ...]
    overall_drop: float
    proposal_channel_drop: float
    typed_event_drop: float
    bootstrap: BootstrapDropCI
    verdict: str
    description: str

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SemanticProposalAblationRunResult:
    """Full run output: the comparison report plus each arm's grounding
    cross-read (experiment coupling) and raw captured grounding turns
    so the CLI can persist them as separate artifacts."""

    report: SemanticProposalAblationReport
    on_grounding_report: SemanticGroundingReport | None
    off_grounding_report: SemanticGroundingReport | None
    on_grounding_turns: tuple[SemanticGroundingTurnEvidence, ...] = ()
    off_grounding_turns: tuple[SemanticGroundingTurnEvidence, ...] = ()


# ---------------------------------------------------------------------------
# Arm execution
# ---------------------------------------------------------------------------


@dataclass
class _ArmObservations:
    runtime_id: str
    turn_count: int = 0
    pe_total: float = 0.0
    expectation_outcomes: list[ExpectationOutcome] = field(default_factory=list)
    slot_nonempty_turns: dict[str, int] = field(default_factory=dict)
    slot_seen_turns: dict[str, int] = field(default_factory=dict)
    grounding_turns: list[SemanticGroundingTurnEvidence] = field(default_factory=list)


def _observed_field_value(snapshot_value: Any, field_name: str) -> float:
    value = getattr(snapshot_value, field_name)
    if isinstance(value, tuple):
        return float(len(value))
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    raise SemanticProposalAblationError(
        f"Expectation field {field_name!r} has unsupported type "
        f"{type(value).__name__}; expected tuple / int / float / bool."
    )


async def _run_arm(
    *,
    arm_id: str,
    cases: tuple[AblationProbeCase, ...],
    lifeform_factory: Callable[[], Any],
    runtime_id: str,
    capture_grounding: bool,
) -> _ArmObservations:
    observations = _ArmObservations(runtime_id=runtime_id)
    global_turn_index = 0
    for case in cases:
        lifeform = lifeform_factory()
        session = lifeform.create_session(
            session_id=f"ablation-{arm_id}-{case.case_id}"
        )
        expectations_by_turn: dict[int, list[SlotExpectation]] = {}
        for expectation in case.expectations:
            expectations_by_turn.setdefault(expectation.after_turn, []).append(
                expectation
            )
        capture = SemanticGroundingTurnCapture() if capture_grounding else None
        for position, turn in enumerate(case.turns):
            for events in turn.typed_events:
                session.submit_semantic_events(events)
            result = await session.run_turn(turn.user_input)
            observations.turn_count += 1
            global_turn_index += 1
            if result.prediction_error is not None:
                observations.pe_total += float(result.prediction_error.magnitude)
            if capture is not None:
                observations.grounding_turns.append(
                    capture.observe_turn(
                        turn_index=global_turn_index,
                        active_snapshots=result.active_snapshots,
                        case_id=case.case_id,
                    )
                )
            _record_nonempty_coverage(observations, result.active_snapshots)
            for expectation in expectations_by_turn.get(position, ()):
                snapshot = result.active_snapshots.get(expectation.slot)
                if snapshot is None:
                    raise SemanticProposalAblationError(
                        f"Case {case.case_id!r} expects slot "
                        f"{expectation.slot!r} after turn {position}, but "
                        "the slot was not published this turn. Semantic "
                        "owners fail-closed publish every turn; a missing "
                        "slot is a wiring defect, not evidence."
                    )
                observed = _observed_field_value(
                    snapshot.value, expectation.field_name
                )
                observations.expectation_outcomes.append(
                    ExpectationOutcome(
                        expectation_id=expectation.expectation_id,
                        case_id=case.case_id,
                        slot=expectation.slot,
                        channel=expectation.channel,
                        after_turn=expectation.after_turn,
                        field_name=expectation.field_name,
                        min_value=expectation.min_value,
                        observed_value=observed,
                        hit=observed >= expectation.min_value,
                    )
                )
    return observations


def _record_nonempty_coverage(
    observations: _ArmObservations, active_snapshots: Any
) -> None:
    """Per-slot non-empty coverage from the response_assembly readout
    (``semantic_record_counts`` is the published aggregate, so we do not
    re-derive record counts from owner internals)."""

    assembly = active_snapshots.get("response_assembly")
    if assembly is None:
        return
    for slot, count in assembly.value.semantic_record_counts:
        observations.slot_seen_turns[slot] = (
            observations.slot_seen_turns.get(slot, 0) + 1
        )
        if count > 0:
            observations.slot_nonempty_turns[slot] = (
                observations.slot_nonempty_turns.get(slot, 0) + 1
            )


def _hit_rate(outcomes: list[ExpectationOutcome]) -> float:
    if not outcomes:
        return 0.0
    return sum(1 for outcome in outcomes if outcome.hit) / len(outcomes)


def _arm_result(
    arm_id: str,
    observations: _ArmObservations,
    grounding_report: SemanticGroundingReport | None,
) -> AblationArmResult:
    outcomes = observations.expectation_outcomes
    proposal_outcomes = [o for o in outcomes if o.channel == PROPOSAL_CHANNEL]
    typed_outcomes = [o for o in outcomes if o.channel == TYPED_EVENT_CHANNEL]

    slots = sorted({o.slot for o in outcomes} | set(observations.slot_seen_turns))
    slot_metrics = []
    for slot in slots:
        slot_outcomes = [o for o in outcomes if o.slot == slot]
        seen = observations.slot_seen_turns.get(slot, 0)
        nonempty = observations.slot_nonempty_turns.get(slot, 0)
        slot_metrics.append(
            SlotArmMetrics(
                slot=slot,
                expectation_total=len(slot_outcomes),
                expectation_hits=sum(1 for o in slot_outcomes if o.hit),
                hit_rate=round(_hit_rate(slot_outcomes), 6),
                nonempty_turn_ratio=round(nonempty / seen, 6) if seen else 0.0,
            )
        )

    return AblationArmResult(
        arm_id=arm_id,
        runtime_id=observations.runtime_id,
        turn_count=observations.turn_count,
        overall_hit_rate=round(_hit_rate(outcomes), 6),
        proposal_channel_hit_rate=round(_hit_rate(proposal_outcomes), 6),
        typed_event_hit_rate=round(_hit_rate(typed_outcomes), 6),
        mean_pe_magnitude=(
            round(observations.pe_total / observations.turn_count, 6)
            if observations.turn_count
            else 0.0
        ),
        slot_metrics=tuple(slot_metrics),
        expectation_outcomes=tuple(outcomes),
        grounding_verdict=(
            grounding_report.verdict if grounding_report is not None else ""
        ),
    )


def _bootstrap_drop_ci(
    on_outcomes: tuple[ExpectationOutcome, ...],
    off_outcomes: tuple[ExpectationOutcome, ...],
    case_ids: tuple[str, ...],
    *,
    resample_count: int = 500,
    seed: int = 11,
) -> BootstrapDropCI:
    """Case-level bootstrap of the overall hit-rate drop (on - off)."""

    on_by_case = {case_id: [o for o in on_outcomes if o.case_id == case_id] for case_id in case_ids}
    off_by_case = {case_id: [o for o in off_outcomes if o.case_id == case_id] for case_id in case_ids}
    rng = random.Random(seed)
    drops: list[float] = []
    for _ in range(resample_count):
        sample = [rng.choice(case_ids) for _ in case_ids]
        on_sample = [o for case_id in sample for o in on_by_case[case_id]]
        off_sample = [o for case_id in sample for o in off_by_case[case_id]]
        drops.append(_hit_rate(on_sample) - _hit_rate(off_sample))
    drops.sort()
    if not drops:
        return BootstrapDropCI(0.0, 0.0, 0.0, resample_count, seed)
    low_index = max(0, int(0.025 * len(drops)))
    high_index = min(len(drops) - 1, int(0.975 * len(drops)))
    return BootstrapDropCI(
        mean_drop=round(sum(drops) / len(drops), 6),
        ci_low=round(drops[low_index], 6),
        ci_high=round(drops[high_index], 6),
        resample_count=resample_count,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------


async def run_semantic_proposal_ablation_async(
    *,
    cases: tuple[AblationProbeCase, ...] = DEFAULT_ABLATION_PROBE_CASES,
    on_runtime_factory: Callable[[], SemanticProposalRuntime] | None = None,
    substrate_runtime: Any = None,
    substrate_fingerprint: str = "synthetic-default",
    capture_grounding: bool = True,
    lifeform_builder: Callable[..., Any] | None = None,
) -> SemanticProposalAblationRunResult:
    """Run both arms over the same probe cases and build the report.

    ``on_runtime_factory`` builds the on-arm semantic runtime; the
    default is the scripted probe runtime (synthetic smoke lane). Pass a
    factory returning ``LLMSemanticProposalRuntime`` for the hf lane.
    ``substrate_runtime`` (optional) is shared by BOTH arms so the
    residual-capture path is identical; only the proposal channel
    differs. A fresh lifeform is built per case per arm so no state
    leaks across cases or arms.
    """

    if not cases:
        raise SemanticProposalAblationError(
            "run_semantic_proposal_ablation_async requires at least one "
            "probe case."
        )
    if lifeform_builder is None:
        from lifeform_domain_emogpt import build_companion_lifeform

        lifeform_builder = build_companion_lifeform

    resolved_on_factory = on_runtime_factory or (
        lambda: ScriptedSemanticProposalRuntime(cases)
    )

    on_probe_runtime_id: str = ""

    def _on_lifeform() -> Any:
        nonlocal on_probe_runtime_id
        runtime = resolved_on_factory()
        on_probe_runtime_id = runtime.runtime_id
        return lifeform_builder(
            substrate_runtime=substrate_runtime,
            semantic_proposal_runtime=runtime,
        )

    def _off_lifeform() -> Any:
        return lifeform_builder(
            substrate_runtime=substrate_runtime,
            semantic_proposal_runtime=NoOpSemanticProposalRuntime(),
        )

    on_observations = await _run_arm(
        arm_id=ON_ARM_ID,
        cases=cases,
        lifeform_factory=_on_lifeform,
        runtime_id="pending",
        capture_grounding=capture_grounding,
    )
    on_observations.runtime_id = on_probe_runtime_id
    off_observations = await _run_arm(
        arm_id=OFF_ARM_ID,
        cases=cases,
        lifeform_factory=_off_lifeform,
        runtime_id=NoOpSemanticProposalRuntime.runtime_id,
        capture_grounding=capture_grounding,
    )

    on_grounding = (
        build_semantic_grounding_report(tuple(on_observations.grounding_turns))
        if capture_grounding and on_observations.grounding_turns
        else None
    )
    off_grounding = (
        build_semantic_grounding_report(tuple(off_observations.grounding_turns))
        if capture_grounding and off_observations.grounding_turns
        else None
    )

    on_arm = _arm_result(ON_ARM_ID, on_observations, on_grounding)
    off_arm = _arm_result(OFF_ARM_ID, off_observations, off_grounding)

    slot_dependencies = _slot_dependencies(on_arm, off_arm)
    overall_drop = round(on_arm.overall_hit_rate - off_arm.overall_hit_rate, 6)
    proposal_drop = round(
        on_arm.proposal_channel_hit_rate - off_arm.proposal_channel_hit_rate, 6
    )
    typed_drop = round(
        on_arm.typed_event_hit_rate - off_arm.typed_event_hit_rate, 6
    )
    case_ids = tuple(case.case_id for case in cases)
    bootstrap = _bootstrap_drop_ci(
        on_arm.expectation_outcomes, off_arm.expectation_outcomes, case_ids
    )

    if overall_drop > DEPENDENT_DROP_THRESHOLD:
        verdict = "llm-proposal-dependent"
    elif overall_drop < SUFFICIENT_DROP_THRESHOLD:
        verdict = "typed-structure-sufficient"
    else:
        verdict = "mixed-per-slot"

    report = SemanticProposalAblationReport(
        schema_version=SEMANTIC_PROPOSAL_ABLATION_SCHEMA_VERSION,
        artifact_kind="semantic_proposal_ablation_report",
        non_gating=True,
        substrate_fingerprint=substrate_fingerprint,
        case_ids=case_ids,
        on_arm=on_arm,
        off_arm=off_arm,
        slot_dependencies=slot_dependencies,
        overall_drop=overall_drop,
        proposal_channel_drop=proposal_drop,
        typed_event_drop=typed_drop,
        bootstrap=bootstrap,
        verdict=verdict,
        description=(
            "Semantic-proposal dependency ablation (non-gating). "
            f"Verdict {verdict!r}: overall lifecycle hit-rate drop "
            f"{overall_drop:+.3f} (proposal-channel {proposal_drop:+.3f}, "
            f"typed-event {typed_drop:+.3f}); bootstrap 95% CI "
            f"[{bootstrap.ci_low:+.3f}, {bootstrap.ci_high:+.3f}]. "
            "An 'llm-proposal-dependent' verdict downgrades the external "
            "claim to 'LLM-assisted typed semantic tracking' and must be "
            "reported as-is."
        ),
    )
    return SemanticProposalAblationRunResult(
        report=report,
        on_grounding_report=on_grounding,
        off_grounding_report=off_grounding,
        on_grounding_turns=tuple(on_observations.grounding_turns),
        off_grounding_turns=tuple(off_observations.grounding_turns),
    )


def _slot_dependencies(
    on_arm: AblationArmResult, off_arm: AblationArmResult
) -> tuple[SlotDependencyDelta, ...]:
    on_by_slot = {metric.slot: metric for metric in on_arm.slot_metrics}
    off_by_slot = {metric.slot: metric for metric in off_arm.slot_metrics}
    deltas = []
    for slot in sorted(set(on_by_slot) | set(off_by_slot)):
        on_metric = on_by_slot.get(slot)
        off_metric = off_by_slot.get(slot)
        on_rate = on_metric.hit_rate if on_metric is not None else 0.0
        off_rate = off_metric.hit_rate if off_metric is not None else 0.0
        deltas.append(
            SlotDependencyDelta(
                slot=slot,
                on_hit_rate=on_rate,
                off_hit_rate=off_rate,
                hit_rate_drop=round(on_rate - off_rate, 6),
            )
        )
    deltas.sort(key=lambda delta: delta.hit_rate_drop, reverse=True)
    return tuple(deltas)


__all__ = [
    "DEFAULT_ABLATION_PROBE_CASES",
    "DEPENDENT_DROP_THRESHOLD",
    "OFF_ARM_ID",
    "ON_ARM_ID",
    "PROPOSAL_CHANNEL",
    "SEMANTIC_PROPOSAL_ABLATION_SCHEMA_VERSION",
    "SUFFICIENT_DROP_THRESHOLD",
    "TYPED_EVENT_CHANNEL",
    "AblationArmResult",
    "AblationProbeCase",
    "AblationProbeTurn",
    "BootstrapDropCI",
    "ExpectationOutcome",
    "ScriptedProposalSpec",
    "ScriptedSemanticProposalRuntime",
    "SemanticProposalAblationError",
    "SemanticProposalAblationReport",
    "SemanticProposalAblationRunResult",
    "SlotArmMetrics",
    "SlotDependencyDelta",
    "SlotExpectation",
    "run_semantic_proposal_ablation_async",
]
