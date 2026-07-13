"""Theory-of-Mind owners + structured LLM proposal runtime (R17).

This module bundles the four ToM owners and their structured LLM
proposal runtime. The runtime is a collaborator of the owners — not an
independent owner — so it lives in the same file rather than in a
separate ``_runtime.py`` shard.

Owner contract: each ToM module is the single owner of its own slot
(`belief_about_other` / `intent_about_other` / `feeling_about_other` /
`preference_about_other`). The LLM runtime only emits typed
:class:`SemanticProposal` records targeted at those slots; it does not
own state and it does not route renderer behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from json import JSONDecodeError
from typing import Any, Mapping, Protocol

from volvence_zero.llm_proposal_diagnostics import LLMProposalAttemptCounters
from volvence_zero.memory import MemorySnapshot
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel
from volvence_zero.semantic_state import (
    NoOpSemanticProposalRuntime,
    SemanticProposal,
    SemanticProposalBatch,
    SemanticProposalOperation,
    SemanticProposalRuntime,
)
from volvence_zero.social_cognition import (
    BeliefAboutOtherSnapshot,
    FeelingAboutOtherSnapshot,
    IntentAboutOtherSnapshot,
    OtherMindRecord,
    OtherMindRecordKind,
    OtherMindRecordStatus,
    PreferenceAboutOtherSnapshot,
    SELF_INTERLOCUTOR_ID,
    SocialPrediction,
    SocialPredictionError,
    SocialPredictionKind,
    SocialScopeKind,
)
from volvence_zero.substrate import SubstrateSnapshot

from .record_store import (
    PendingSocialPrediction,
    SocialRecordStore,
    apply_outcome_to_record,
    settle_pending_predictions,
)

from volvence_zero.semantic_state._llm_proposal_counters import (
    LLMProposalAttemptAccumulator,
)

from ._llm_debug import log_proposal_attempt, make_attempt_logger
from ._llm_parsing import strip_code_fence


class _OtherMindOwnerModule(RuntimeModule[Any]):
    record_kind: OtherMindRecordKind
    snapshot_type: type[Any]
    empty_description: str
    dependencies = ("substrate", "memory", "multi_party_identity")
    default_wiring_level = WiringLevel.SHADOW
    min_proposal_confidence = 0.50
    prediction_kind: SocialPredictionKind

    def __init__(
        self,
        *,
        proposal_runtime: SemanticProposalRuntime | None = None,
        user_input: str | None = None,
        turn_index: int = 0,
        wiring_level: WiringLevel | None = None,
        record_store: SocialRecordStore | None = None,
    ) -> None:
        # W1.C (CP-16): the optional session-held ``record_store`` gives
        # this owner cross-turn records + pending-prediction settlement.
        # When None (unit tests / standalone probes) the owner keeps its
        # original per-turn stateless behavior.
        super().__init__(wiring_level=wiring_level)
        self._proposal_runtime = proposal_runtime
        self._user_input = user_input
        self._turn_index = turn_index
        self._record_store = record_store

    async def process(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> Snapshot[Any]:
        new_records: tuple[OtherMindRecord, ...] = ()
        control_signal = 0.0
        if self._proposal_runtime is not None:
            substrate_snapshot = upstream.get("substrate")
            memory_snapshot = upstream.get("memory")
            batch = self._proposal_runtime.propose(
                target_slot=self.slot_name,
                user_input=self._user_input,
                substrate_snapshot=(
                    substrate_snapshot.value
                    if substrate_snapshot is not None
                    and isinstance(substrate_snapshot.value, SubstrateSnapshot)
                    else None
                ),
                memory_snapshot=(
                    memory_snapshot.value
                    if memory_snapshot is not None
                    and isinstance(memory_snapshot.value, MemorySnapshot)
                    else None
                ),
                previous_snapshot=None,
                turn_index=self._turn_index,
            )
            proposals = tuple(
                proposal
                for proposal in batch.proposals
                if proposal.target_slot == self.slot_name
                and proposal.confidence >= self.min_proposal_confidence
            )
            new_records = tuple(
                _record_from_proposal(
                    proposal=proposal,
                    kind=self.record_kind,
                    turn_index=self._turn_index,
                )
                for proposal in proposals
            )
            control_signal = _mean_control_signal(proposals)
        records, settled_errors = self._settle_and_merge(new_records)
        proposal_diagnostics = self._extract_proposal_diagnostics()
        return self.publish(
            self._snapshot(
                records=records,
                control_signal=control_signal,
                proposal_diagnostics=proposal_diagnostics,
                settled_errors=settled_errors,
            )
        )

    def _settle_and_merge(
        self, new_records: tuple[OtherMindRecord, ...]
    ) -> tuple[tuple[OtherMindRecord, ...], tuple[SocialPredictionError, ...]]:
        """Cross-turn settlement + promote/retire (W1.C, CP-16 core).

        Without a store the owner stays stateless: this turn's records
        pass through unchanged and nothing settles.
        """

        store = self._record_store
        if store is None:
            return (new_records, ())
        prior_records = store.tom_records(self.slot_name)
        pending = store.pending_tom_predictions(self.slot_name)
        evidence_by_scope: dict[str, tuple[tuple[str, str], ...]] = {}
        for record in new_records:
            evidence_by_scope[record.interlocutor_id] = (
                *evidence_by_scope.get(record.interlocutor_id, ()),
                (record.record_id, record.summary),
            )
        result = settle_pending_predictions(
            pending=pending,
            new_evidence_by_scope=evidence_by_scope,
            turn_index=self._turn_index,
            owner=self.owner,
            similarity=store.similarity,
        )
        outcome_by_record = {
            record_id: (outcome, error_id)
            for record_id, outcome, error_id in result.outcomes_by_record
        }
        updated_prior = tuple(
            apply_outcome_to_record(
                record,
                outcome_by_record[record.record_id][0],
                error_id=outcome_by_record[record.record_id][1],
            )
            if record.record_id in outcome_by_record
            else record
            for record in prior_records
        )
        merged_by_id: dict[str, OtherMindRecord] = {}
        for record in (*updated_prior, *new_records):
            merged_by_id[record.record_id] = record
        merged = tuple(merged_by_id.values())
        store.set_tom_records(self.slot_name, merged)
        # Rebuild the pending window: still-ambiguous entries keep their
        # original issue turn; every ACTIVE / CONTESTED record without a
        # pending entry issues a fresh prediction (CONTESTED must remain
        # settleable so a second disconfirmation can retire it).
        pending_by_record = {
            entry.source_record_id: entry for entry in result.still_pending
        }
        for record in store.tom_records(self.slot_name):
            if record.status is OtherMindRecordStatus.RETIRED:
                continue
            if record.record_id in pending_by_record:
                continue
            pending_by_record[record.record_id] = PendingSocialPrediction(
                prediction=self._prediction_for_record(record),
                source_record_id=record.record_id,
                issued_turn=self._turn_index,
            )
        store.set_pending_tom_predictions(
            self.slot_name, tuple(pending_by_record.values())
        )
        return (store.tom_records(self.slot_name), result.settled_errors)

    def _extract_proposal_diagnostics(self) -> LLMProposalAttemptCounters | None:
        """Return the runtime's typed counters when available.

        Returns ``None`` when:
        * No proposal runtime is wired (NoOp / scaffold paths).
        * The wired runtime is not LLM-backed (no ``attempt_counters``).
          We duck-check on the attribute name rather than ``isinstance``
          to keep this owner agnostic to which LLM-backed runtime
          subclass is wired (e.g. test fakes or future variants that
          implement the same counters protocol).
        """
        runtime = self._proposal_runtime
        if runtime is None:
            return None
        counters = getattr(runtime, "attempt_counters", None)
        if isinstance(counters, LLMProposalAttemptCounters):
            return counters
        return None

    def _snapshot(
        self,
        *,
        records: tuple[OtherMindRecord, ...],
        control_signal: float,
        proposal_diagnostics: LLMProposalAttemptCounters | None,
        settled_errors: tuple[SocialPredictionError, ...] = (),
    ) -> Any:
        return self.snapshot_type(
            records=records,
            active_predictions=self._active_predictions(records),
            control_signal=control_signal,
            description=(
                self.empty_description
                if not records
                else (
                    f"{self.owner} published explicit records={len(records)} "
                    f"settled={len(settled_errors)}."
                )
            ),
            proposal_diagnostics=proposal_diagnostics,
            settled_errors=settled_errors,
        )

    def _prediction_for_record(self, record: OtherMindRecord) -> SocialPrediction:
        return SocialPrediction(
            prediction_id=f"{self.slot_name}:{record.record_id}:prediction",
            kind=self.prediction_kind,
            scope_kind=SocialScopeKind.INTERLOCUTOR,
            scope_id=record.interlocutor_id,
            subject_ids=(record.interlocutor_id,),
            audience_ids=(SELF_INTERLOCUTOR_ID,),
            predicted_outcome=record.summary,
            confidence=record.confidence,
            evidence=(
                f"tom_record:{record.record_id}",
                f"tom_kind:{record.kind.value}",
                record.evidence,
            ),
        )

    def _active_predictions(
        self, records: tuple[OtherMindRecord, ...]
    ) -> tuple[SocialPrediction, ...]:
        """Publish owner-authored ToM predictions from typed records.

        W1.C: only ACTIVE records predict publicly. CONTESTED records
        stay settleable in the pending store (so a second
        disconfirmation retires them) but do not assert predictions;
        RETIRED records neither predict nor pend.
        """

        return tuple(
            self._prediction_for_record(record)
            for record in records
            if record.status is OtherMindRecordStatus.ACTIVE
        )


class BeliefAboutOtherModule(_OtherMindOwnerModule):
    slot_name = "belief_about_other"
    owner = "BeliefAboutOtherModule"
    value_type = BeliefAboutOtherSnapshot
    record_kind = OtherMindRecordKind.BELIEF
    prediction_kind = SocialPredictionKind.BELIEF_ABOUT_OTHER
    snapshot_type = BeliefAboutOtherSnapshot
    empty_description = "R17 SHADOW scaffold: no belief-about-other records yet."


class IntentAboutOtherModule(_OtherMindOwnerModule):
    slot_name = "intent_about_other"
    owner = "IntentAboutOtherModule"
    value_type = IntentAboutOtherSnapshot
    record_kind = OtherMindRecordKind.INTENT
    prediction_kind = SocialPredictionKind.INTENT_ABOUT_OTHER
    snapshot_type = IntentAboutOtherSnapshot
    empty_description = "R17 SHADOW scaffold: no intent-about-other records yet."


class FeelingAboutOtherModule(_OtherMindOwnerModule):
    slot_name = "feeling_about_other"
    owner = "FeelingAboutOtherModule"
    value_type = FeelingAboutOtherSnapshot
    record_kind = OtherMindRecordKind.FEELING
    prediction_kind = SocialPredictionKind.FEELING_ABOUT_OTHER
    snapshot_type = FeelingAboutOtherSnapshot
    empty_description = "R17 SHADOW scaffold: no feeling-about-other records yet."


class PreferenceAboutOtherModule(_OtherMindOwnerModule):
    slot_name = "preference_about_other"
    owner = "PreferenceAboutOtherModule"
    value_type = PreferenceAboutOtherSnapshot
    record_kind = OtherMindRecordKind.PREFERENCE
    prediction_kind = SocialPredictionKind.PREFERENCE_ABOUT_OTHER
    snapshot_type = PreferenceAboutOtherSnapshot
    empty_description = "R17 SHADOW scaffold: no preference-about-other records yet."


def _record_from_proposal(
    *,
    proposal: SemanticProposal,
    kind: OtherMindRecordKind,
    turn_index: int,
) -> OtherMindRecord:
    return OtherMindRecord(
        record_id=proposal.proposal_id,
        interlocutor_id="primary",
        kind=kind,
        summary=proposal.summary,
        detail=proposal.detail,
        confidence=proposal.confidence,
        status=OtherMindRecordStatus.ACTIVE,
        source_turn=turn_index,
        prediction_error_refs=(),
        evidence=proposal.evidence,
    )


def _mean_control_signal(proposals: tuple[SemanticProposal, ...]) -> float:
    if not proposals:
        return 0.0
    return sum(proposal.control_signal for proposal in proposals) / len(proposals)


# ---------------------------------------------------------------------------
# Structured LLM proposal runtime (collaborator of the four ToM owners above)
# ---------------------------------------------------------------------------


_TOM_TARGET_SLOTS: frozenset[str] = frozenset(
    {
        "belief_about_other",
        "intent_about_other",
        "feeling_about_other",
        "preference_about_other",
    }
)
_MIN_TOM_CONFIDENCE = 0.50


class _GenerateProtocol(Protocol):
    def generate(
        self, *, prompt: str, max_new_tokens: int = ..., temperature: float = ...
    ) -> str: ...


@dataclass(frozen=True)
class _ToMDecision:
    target_slot: str
    summary: str
    detail: str
    evidence: str
    confidence: float
    control_signal: float


_TOM_PROMPT = (
    "You extract Theory-of-Mind observations from one dialogue turn.\n"
    "Return a JSON array. Each item must have exactly these fields:\n"
    "[\n"
    "  {{\n"
    '    \"target_slot\": \"belief_about_other|intent_about_other|feeling_about_other|preference_about_other\",\n'
    '    \"summary\": \"short stable claim\",\n'
    '    \"detail\": \"specific evidence-aware detail\",\n'
    '    \"evidence\": \"short quote or observation from the user message\",\n'
    '    \"confidence\": 0.0,\n'
    '    \"control_signal\": 0.0\n'
    "  }}\n"
    "]\n"
    "\n"
    "Do not infer demographics. Do not output markdown. If there is no "
    "clear Theory-of-Mind observation, return [].\n"
    "\n"
    "User message:\n"
    '\"\"\"\n'
    "{user_input}\n"
    '\"\"\"'
)


class LLMToMProposalRuntime(SemanticProposalRuntime):
    """Structured proposal source for R17 ToM owners."""

    runtime_id = "social-tom-llm-structured"

    def __init__(
        self,
        *,
        provider: _GenerateProtocol,
        base_runtime: SemanticProposalRuntime | None = None,
        max_new_tokens: int = 384,
    ) -> None:
        self._provider = provider
        self._base = base_runtime or NoOpSemanticProposalRuntime()
        self._max_new_tokens = max_new_tokens
        self._cache_key: tuple[str, int] | None = None
        self._cache_decisions: tuple[_ToMDecision, ...] | None = None
        # Opt-in diagnostic sink. ``None`` (the default) means the hot
        # path stays zero-overhead; setting ``VZ_LLM_PROPOSAL_DEBUG_LOG``
        # before host construction binds a JSONL append callable so a
        # diagnostic run can capture raw provider output + parse outcome
        # without changing constructor surface.
        self._debug_logger = make_attempt_logger()
        # Always-on typed counters (Wave E1). Owners read
        # ``attempt_counters`` and surface it on their snapshot so a
        # 0-records evidence run can be diagnosed without env vars.
        self._counters = LLMProposalAttemptAccumulator()

    @property
    def attempt_counters(self) -> LLMProposalAttemptCounters:
        """Return an immutable snapshot of cumulative LLM call counters.

        Owner modules read this each turn and republish on the typed
        snapshot's ``proposal_diagnostics`` field. The returned value
        is frozen; mutating callers must not assume identity.
        """
        return self._counters.snapshot()

    def propose(
        self,
        *,
        target_slot: str,
        user_input: str | None,
        substrate_snapshot: SubstrateSnapshot | None,
        memory_snapshot: MemorySnapshot | None,
        previous_snapshot: object | None,
        turn_index: int,
    ) -> SemanticProposalBatch:
        if target_slot not in _TOM_TARGET_SLOTS or not user_input:
            return self._base.propose(
                target_slot=target_slot,
                user_input=user_input,
                substrate_snapshot=substrate_snapshot,
                memory_snapshot=memory_snapshot,
                previous_snapshot=previous_snapshot,
                turn_index=turn_index,
            )

        decisions = self._decisions_for_turn(user_input=user_input, turn_index=turn_index)
        proposals = tuple(
            SemanticProposal(
                proposal_id=f"{decision.target_slot}:tom-llm:{turn_index}:{index}",
                target_slot=decision.target_slot,
                operation=SemanticProposalOperation.OBSERVE,
                summary=decision.summary,
                detail=decision.detail,
                confidence=decision.confidence,
                evidence=decision.evidence,
                control_signal=decision.control_signal,
            )
            for index, decision in enumerate(decisions)
            if decision.target_slot == target_slot
        )
        return SemanticProposalBatch(
            proposals=proposals,
            runtime_id=self.runtime_id,
            schema_version=1,
            description=(
                f"Structured ToM runtime emitted {len(proposals)} proposal(s) "
                f"for {target_slot} at turn {turn_index}."
            ),
        )

    def _decisions_for_turn(
        self,
        *,
        user_input: str,
        turn_index: int,
    ) -> tuple[_ToMDecision, ...]:
        cache_key = (user_input, turn_index)
        if self._cache_key == cache_key and self._cache_decisions is not None:
            return self._cache_decisions
        prompt = _TOM_PROMPT.format(user_input=user_input.strip()[:800])
        raw = self._provider.generate(
            prompt=prompt,
            max_new_tokens=self._max_new_tokens,
            temperature=0.0,
        )
        decisions, parse_status, parse_error = _parse_tom_decisions_with_diag(raw)
        log_proposal_attempt(
            self._debug_logger,
            runtime_id=self.runtime_id,
            target_slot=None,
            turn_index=turn_index,
            prompt=prompt,
            raw_output=raw,
            parsed_count=len(decisions),
            parse_status=parse_status,
            parse_error=parse_error,
        )
        # ``parsed_count`` here is decisions surviving the strict schema
        # parser. Owner-side ``min_proposal_confidence`` may further
        # shrink the set; the runtime tracks the parse outcome and the
        # owner reports its own emission count via a separate path
        # (the snapshot still surfaces parse counters here so a parse
        # failure is not hidden behind owner-side filtering).
        self._counters.record_attempt(
            parse_status=parse_status,
            parse_error=parse_error,
            parsed_count=len(decisions),
            emitted_count=len(decisions),
        )
        self._cache_key = cache_key
        self._cache_decisions = decisions
        return decisions


def _parse_tom_decisions(text: str) -> tuple[_ToMDecision, ...] | None:
    decisions, status, _ = _parse_tom_decisions_with_diag(text)
    if status == "parse_error":
        return None
    return decisions


def _parse_tom_decisions_with_diag(
    text: str,
) -> tuple[tuple[_ToMDecision, ...], str, str | None]:
    """Parse with diagnostic categories; never raises.

    Returns ``(decisions, status, parse_error)`` where ``status`` is one
    of ``"ok"`` / ``"parse_error"`` / ``"empty_or_rejected"`` and
    ``parse_error`` is the JSONDecodeError message when applicable. Used
    by both the production parser (``_parse_tom_decisions``) and the
    diagnostic sink in ``LLMToMProposalRuntime``.
    """
    cleaned = strip_code_fence(text)
    try:
        payload = json.loads(cleaned.strip())
    except JSONDecodeError as exc:
        return ((), "parse_error", str(exc))
    if not isinstance(payload, list):
        return ((), "parse_error", f"top-level not a list: {type(payload).__name__}")
    decisions: list[_ToMDecision] = []
    for item in payload:
        decision = _parse_tom_decision(item)
        if decision is not None:
            decisions.append(decision)
    if not decisions:
        return ((), "empty_or_rejected", None)
    return (tuple(decisions), "ok", None)


def _parse_tom_decision(item: object) -> _ToMDecision | None:
    if not isinstance(item, dict):
        return None
    target_slot = item.get("target_slot")
    summary = item.get("summary")
    detail = item.get("detail")
    evidence = item.get("evidence")
    confidence = item.get("confidence")
    control_signal = item.get("control_signal", 0.0)
    if target_slot not in _TOM_TARGET_SLOTS:
        return None
    if not isinstance(summary, str) or not summary.strip():
        return None
    if not isinstance(detail, str) or not detail.strip():
        return None
    if not isinstance(evidence, str) or not evidence.strip():
        return None
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return None
    if isinstance(control_signal, bool) or not isinstance(control_signal, (int, float)):
        return None
    confidence_value = float(confidence)
    control_value = float(control_signal)
    if confidence_value < _MIN_TOM_CONFIDENCE or confidence_value > 1.0:
        return None
    if control_value < 0.0 or control_value > 1.0:
        return None
    return _ToMDecision(
        target_slot=target_slot,
        summary=summary.strip()[:160],
        detail=detail.strip()[:500],
        evidence=evidence.strip()[:240],
        confidence=confidence_value,
        control_signal=control_value,
    )


__all__ = [
    "BeliefAboutOtherModule",
    "FeelingAboutOtherModule",
    "IntentAboutOtherModule",
    "LLMToMProposalRuntime",
    "PreferenceAboutOtherModule",
]
