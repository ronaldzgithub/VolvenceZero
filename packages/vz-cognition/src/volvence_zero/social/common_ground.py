"""Common-ground owner + structured LLM proposal runtime (R19).

Phase 1 W1.E of the EQ-owner uplift: ``CommonGroundModule.process``
now consumes its declared dependencies (``conversational_role`` +
``belief_about_other``) to derive dyad common-ground atoms from typed
Theory-of-Mind BELIEF records. Before W1.E the owner declared the
dependencies but discarded the upstream view (``del upstream``); the
only path to atoms was an explicit ``proposal_runtime`` injection.

The structured LLM proposal runtime is a collaborator of
``CommonGroundModule`` and lives in this same file rather than in a
separate ``_runtime.py`` shard, so the owner and its proposal source
can be reasoned about as one unit.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from json import JSONDecodeError
from typing import Any, Mapping, Protocol

from volvence_zero.llm_proposal_diagnostics import LLMProposalAttemptCounters
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel
from volvence_zero.social_cognition import (
    MAX_COMMON_GROUND_RECURSION_DEPTH,
    BeliefAboutOtherSnapshot,
    CommonGroundAtom,
    CommonGroundSnapshot,
    ConversationalRoleSnapshot,
    SocialPrediction,
    SocialPredictionError,
    SocialPredictionKind,
    SocialScopeKind,
)

from .record_store import (
    PendingSocialPrediction,
    SocialRecordStore,
    settle_pending_predictions,
)

from volvence_zero.semantic_state._llm_proposal_counters import (
    LLMProposalAttemptAccumulator,
)

from ._llm_debug import log_proposal_attempt, make_attempt_logger
from ._llm_parsing import strip_code_fence


_MIN_COMMON_GROUND_CONFIDENCE = 0.50
_VALID_SCOPE_KINDS = {SocialScopeKind.DYAD.value, SocialScopeKind.GROUP.value}
# Confidence floor on ``ConversationalRoleSnapshot.role_confidence``
# below which we suppress the typed-derived dyad atoms. The default
# kernel publishes a low-confidence cold-start role each turn; we
# only build common-ground atoms when the role is meaningfully
# confident (mirrors ``InterlocutorThresholds.min_confidence`` on
# the interlocutor-state owner).
_ROLE_CONFIDENCE_FLOOR = 0.30


def _typed_snapshot(
    snapshot: Snapshot[Any] | None, expected_type: type
) -> Any:
    """Return ``snapshot.value`` only when it matches ``expected_type``.

    Upstream slots that are SHADOW or DISABLED surface as placeholder
    snapshots whose ``value`` is NOT a typed ``*Snapshot`` instance.
    The owner reads typed fields, so we narrow the type before
    consuming. Missing / placeholder slots yield ``None`` and the
    caller treats that as "no upstream evidence".
    """
    if snapshot is None:
        return None
    if not isinstance(snapshot.value, expected_type):
        return None
    return snapshot.value


def _dyad_scope_id(*, speaker: str, addressee: str) -> str:
    """Return a stable dyad scope identifier ordered by participant id.

    Two participants always produce the same scope_id regardless of
    whose turn it is (the speaker / addressee swap on the next user
    turn must not split the dyad). Falls back to ``speaker+addressee``
    when both are equal (single-actor pseudo-dyad).
    """
    if speaker == addressee:
        return f"{speaker}+{addressee}"
    a, b = sorted((speaker, addressee))
    return f"{a}+{b}"


class _GenerateProtocol(Protocol):
    def generate(
        self, *, prompt: str, max_new_tokens: int = ..., temperature: float = ...
    ) -> str: ...


@dataclass(frozen=True)
class CommonGroundProposal:
    scope_kind: SocialScopeKind
    scope_id: str
    summary: str
    accepted_by_ids: tuple[str, ...]
    evidence: tuple[str, ...]
    confidence: float
    recursion_depth: int
    control_signal: float = 0.0


@dataclass(frozen=True)
class CommonGroundProposalBatch:
    proposals: tuple[CommonGroundProposal, ...]
    runtime_id: str
    description: str


class CommonGroundModule(RuntimeModule[CommonGroundSnapshot]):
    slot_name = "common_ground"
    owner = "CommonGroundModule"
    value_type = CommonGroundSnapshot
    dependencies = (
        "multi_party_identity",
        "conversational_role",
        "belief_about_other",
        "memory",
    )
    default_wiring_level = WiringLevel.SHADOW

    def __init__(
        self,
        *,
        dyad_atoms: tuple[CommonGroundAtom, ...] = (),
        group_atoms: tuple[CommonGroundAtom, ...] = (),
        proposal_runtime: "LLMCommonGroundProposalRuntime | None" = None,
        user_input: str | None = None,
        turn_index: int = 0,
        wiring_level: WiringLevel | None = None,
        record_store: SocialRecordStore | None = None,
    ) -> None:
        # W1.C (CP-17): the optional session-held ``record_store`` gives
        # this owner cross-turn atoms + pending-prediction settlement.
        super().__init__(wiring_level=wiring_level)
        self._dyad_atoms = dyad_atoms
        self._group_atoms = group_atoms
        self._proposal_runtime = proposal_runtime
        self._user_input = user_input
        self._turn_index = turn_index
        self._record_store = record_store

    async def process(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> Snapshot[CommonGroundSnapshot]:
        role_snapshot = _typed_snapshot(
            upstream.get("conversational_role"), ConversationalRoleSnapshot
        )
        belief_snapshot = _typed_snapshot(
            upstream.get("belief_about_other"), BeliefAboutOtherSnapshot
        )
        upstream_atoms = self._derive_upstream_dyad_atoms(
            role_snapshot=role_snapshot,
            belief_snapshot=belief_snapshot,
        )
        runtime_atoms = self._runtime_atoms()
        new_dyad_atoms = (
            *self._dyad_atoms,
            *upstream_atoms,
            *(atom for atom in runtime_atoms if atom.scope_kind.value == "dyad"),
        )
        new_group_atoms = (
            *self._group_atoms,
            *(atom for atom in runtime_atoms if atom.scope_kind.value == "group"),
        )
        dyad_atoms, group_atoms, settled_errors = self._settle_and_merge(
            new_dyad_atoms=new_dyad_atoms,
            new_group_atoms=new_group_atoms,
        )
        control_signal = _mean_confidence(upstream_atoms + runtime_atoms)
        proposal_diagnostics = self._extract_proposal_diagnostics()
        return self.publish(
            CommonGroundSnapshot(
                dyad_atoms=dyad_atoms,
                group_atoms=group_atoms,
                active_predictions=self._active_predictions(dyad_atoms + group_atoms),
                control_signal=control_signal,
                description=(
                    "R19 SHADOW scaffold: "
                    f"dyad_atoms={len(dyad_atoms)} group_atoms={len(group_atoms)} "
                    f"settled={len(settled_errors)}."
                ),
                proposal_diagnostics=proposal_diagnostics,
                settled_errors=settled_errors,
            )
        )

    def _settle_and_merge(
        self,
        *,
        new_dyad_atoms: tuple[CommonGroundAtom, ...],
        new_group_atoms: tuple[CommonGroundAtom, ...],
    ) -> tuple[
        tuple[CommonGroundAtom, ...],
        tuple[CommonGroundAtom, ...],
        tuple[SocialPredictionError, ...],
    ]:
        """W1.C (CP-17): cross-turn atoms + pending-prediction settlement.

        Prior-turn predictions settle against THIS turn's newly derived
        atoms for the same scope (repair / clarification evidence shows
        up as a new atom whose summary contradicts the prediction).
        Without a store the owner stays stateless (original behavior).
        """

        store = self._record_store
        if store is None:
            return (new_dyad_atoms, new_group_atoms, ())
        new_atoms = new_dyad_atoms + new_group_atoms
        evidence_by_scope: dict[str, tuple[tuple[str, str], ...]] = {}
        for atom in new_atoms:
            evidence_by_scope[atom.scope_id] = (
                *evidence_by_scope.get(atom.scope_id, ()),
                (atom.atom_id, atom.summary),
            )
        result = settle_pending_predictions(
            pending=store.pending_common_ground_predictions,
            new_evidence_by_scope=evidence_by_scope,
            turn_index=self._turn_index,
            owner=self.owner,
            similarity=store.similarity,
        )
        merged_dyad: dict[str, CommonGroundAtom] = {}
        for atom in (*store.common_ground_dyad_atoms, *new_dyad_atoms):
            merged_dyad[atom.atom_id] = atom
        merged_group: dict[str, CommonGroundAtom] = {}
        for atom in (*store.common_ground_group_atoms, *new_group_atoms):
            merged_group[atom.atom_id] = atom
        dyad_atoms = tuple(merged_dyad.values())
        group_atoms = tuple(merged_group.values())
        store.set_common_ground_atoms(
            dyad_atoms=dyad_atoms, group_atoms=group_atoms
        )
        dyad_atoms = store.common_ground_dyad_atoms
        group_atoms = store.common_ground_group_atoms
        pending_by_atom = {
            entry.source_record_id: entry for entry in result.still_pending
        }
        for prediction in self._active_predictions(dyad_atoms + group_atoms):
            atom_id = prediction.prediction_id.removeprefix(
                "common_ground:"
            ).removesuffix(":prediction")
            if atom_id not in pending_by_atom:
                pending_by_atom[atom_id] = PendingSocialPrediction(
                    prediction=prediction,
                    source_record_id=atom_id,
                    issued_turn=self._turn_index,
                )
        store.set_pending_common_ground_predictions(
            tuple(pending_by_atom.values())
        )
        return (dyad_atoms, group_atoms, result.settled_errors)

    def _active_predictions(
        self, atoms: tuple[CommonGroundAtom, ...]
    ) -> tuple[SocialPrediction, ...]:
        return tuple(
            SocialPrediction(
                prediction_id=f"common_ground:{atom.atom_id}:prediction",
                kind=SocialPredictionKind.COMMON_GROUND_RESOLUTION,
                scope_kind=atom.scope_kind,
                scope_id=atom.scope_id,
                subject_ids=_unique_ids(atom.accepted_by_ids),
                audience_ids=_unique_ids(atom.accepted_by_ids),
                predicted_outcome=atom.summary,
                confidence=atom.confidence,
                evidence=(
                    f"common_ground_atom:{atom.atom_id}",
                    *atom.evidence,
                ),
            )
            for atom in atoms
        )

    def _extract_proposal_diagnostics(self) -> LLMProposalAttemptCounters | None:
        """Return the wired runtime's typed counters when available.

        Returns ``None`` when no runtime is wired or when the wired
        runtime is not LLM-backed (no ``attempt_counters`` attribute).
        Duck-typed on the attribute name to remain agnostic to
        future LLM-backed runtime variants.
        """
        runtime = self._proposal_runtime
        if runtime is None:
            return None
        counters = getattr(runtime, "attempt_counters", None)
        if isinstance(counters, LLMProposalAttemptCounters):
            return counters
        return None

    def _derive_upstream_dyad_atoms(
        self,
        *,
        role_snapshot: ConversationalRoleSnapshot | None,
        belief_snapshot: BeliefAboutOtherSnapshot | None,
    ) -> tuple[CommonGroundAtom, ...]:
        """Derive typed dyad common-ground atoms from upstream snapshots.

        Phase 1 W1.E of the EQ-owner uplift. The owner consumes:

        * ``conversational_role`` for the canonical dyad framing
          (``active_speaker_id`` + single addressee). When the
          conversation is multi-addressee or the role view is
          missing / low-confidence, no upstream atoms are emitted
          (group-level atoms are still spec-future-work).
        * ``belief_about_other`` for typed BELIEF records the kernel
          has accumulated about the addressee. Each high-confidence
          record becomes one dyad atom whose ``accepted_by_ids`` is
          the (speaker, addressee) tuple and whose ``recursion_depth``
          is 1 ("we know that they hold this belief").

        Empty / missing snapshots produce no atoms. The proposal
        runtime path remains additive on top of this typed derivation,
        so explicit proposals continue to work as before.
        """
        if role_snapshot is None:
            return ()
        if role_snapshot.role_confidence < _ROLE_CONFIDENCE_FLOOR:
            return ()
        if len(role_snapshot.addressee_ids) != 1:
            return ()
        if belief_snapshot is None or not belief_snapshot.records:
            return ()
        speaker = role_snapshot.active_speaker_id
        addressee = role_snapshot.addressee_ids[0]
        scope_id = _dyad_scope_id(speaker=speaker, addressee=addressee)
        accepted = (speaker, addressee)
        atoms: list[CommonGroundAtom] = []
        for index, record in enumerate(belief_snapshot.records):
            if record.confidence < _MIN_COMMON_GROUND_CONFIDENCE:
                continue
            evidence = (record.evidence.strip(),) if record.evidence.strip() else ()
            atoms.append(
                CommonGroundAtom(
                    atom_id=(
                        f"cg:dyad:{scope_id}:belief:{record.record_id}:{self._turn_index}"
                    ),
                    scope_id=scope_id,
                    scope_kind=SocialScopeKind.DYAD,
                    summary=record.summary[:240],
                    recursion_depth=1,
                    confidence=record.confidence,
                    accepted_by_ids=accepted,
                    evidence=evidence,
                )
            )
            del index
        return tuple(atoms)

    def _runtime_atoms(self) -> tuple[CommonGroundAtom, ...]:
        if self._proposal_runtime is None:
            return ()
        batch = self._proposal_runtime.propose(
            user_input=self._user_input,
            turn_index=self._turn_index,
        )
        return tuple(
            _atom_from_proposal(proposal=proposal, turn_index=self._turn_index)
            for proposal in batch.proposals
        )


def _atom_from_proposal(
    *,
    proposal: CommonGroundProposal,
    turn_index: int,
) -> CommonGroundAtom:
    return CommonGroundAtom(
        atom_id=f"cg:{proposal.scope_kind.value}:{proposal.scope_id}:{turn_index}",
        scope_id=proposal.scope_id,
        scope_kind=proposal.scope_kind,
        summary=proposal.summary,
        recursion_depth=proposal.recursion_depth,
        confidence=proposal.confidence,
        accepted_by_ids=proposal.accepted_by_ids,
        evidence=proposal.evidence,
    )


def _mean_confidence(atoms: tuple[CommonGroundAtom, ...]) -> float:
    if not atoms:
        return 0.0
    return sum(atom.confidence for atom in atoms) / len(atoms)


def _unique_ids(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


# ---------------------------------------------------------------------------
# Structured LLM proposal runtime (collaborator of CommonGroundModule above)
# ---------------------------------------------------------------------------


_COMMON_GROUND_PROMPT = (
    "Extract common-ground observations from one dialogue turn.\n"
    "Return a JSON array. Each item must have exactly these fields:\n"
    "[\n"
    "  {{\n"
    '    \"scope_kind\": \"dyad|group\",\n'
    '    \"scope_id\": \"stable dyad or group id\",\n'
    '    \"summary\": \"shared fact or mutual acceptance\",\n'
    '    \"accepted_by_ids\": [\"id1\", \"id2\"],\n'
    '    \"evidence\": \"short quote or observation\",\n'
    '    \"confidence\": 0.0,\n'
    '    \"recursion_depth\": 2,\n'
    '    \"control_signal\": 0.0\n'
    "  }}\n"
    "]\n"
    "\n"
    "Only include facts mutually accepted or clearly shared. Do not infer "
    "common ground from keywords alone. If none is clear, return [].\n"
    "\n"
    "User message:\n"
    '\"\"\"\n'
    "{user_input}\n"
    '\"\"\"'
)


class LLMCommonGroundProposalRuntime:
    """Structured source for CommonGroundModule proposals."""

    runtime_id = "common-ground-llm-structured"

    def __init__(
        self,
        *,
        provider: _GenerateProtocol,
        max_new_tokens: int = 384,
    ) -> None:
        self._provider = provider
        self._max_new_tokens = max_new_tokens
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

        Owner modules read this each turn and republish on the
        ``CommonGroundSnapshot.proposal_diagnostics`` field.
        """
        return self._counters.snapshot()

    def propose(
        self,
        *,
        user_input: str | None,
        turn_index: int,
    ) -> CommonGroundProposalBatch:
        if not user_input:
            return CommonGroundProposalBatch(
                proposals=(),
                runtime_id=self.runtime_id,
                description="No user input; common-ground runtime emitted no proposals.",
            )
        prompt = _COMMON_GROUND_PROMPT.format(user_input=user_input.strip()[:800])
        raw = self._provider.generate(
            prompt=prompt,
            max_new_tokens=self._max_new_tokens,
            temperature=0.0,
        )
        proposals, parse_status, parse_error = _parse_common_ground_proposals_with_diag(raw)
        log_proposal_attempt(
            self._debug_logger,
            runtime_id=self.runtime_id,
            target_slot=None,
            turn_index=turn_index,
            prompt=prompt,
            raw_output=raw,
            parsed_count=len(proposals),
            parse_status=parse_status,
            parse_error=parse_error,
        )
        self._counters.record_attempt(
            parse_status=parse_status,
            parse_error=parse_error,
            parsed_count=len(proposals),
            emitted_count=len(proposals),
        )
        return CommonGroundProposalBatch(
            proposals=proposals,
            runtime_id=self.runtime_id,
            description=(
                f"Structured common-ground runtime emitted "
                f"{len(proposals)} proposal(s) at turn {turn_index}."
            ),
        )


def _parse_common_ground_proposals(text: str) -> tuple[CommonGroundProposal, ...] | None:
    proposals, status, _ = _parse_common_ground_proposals_with_diag(text)
    if status == "parse_error":
        return None
    return proposals


def _parse_common_ground_proposals_with_diag(
    text: str,
) -> tuple[tuple[CommonGroundProposal, ...], str, str | None]:
    """Parse with diagnostic categories; never raises.

    Returns ``(proposals, status, parse_error)`` where ``status`` is
    one of ``"ok"`` / ``"parse_error"`` / ``"empty_or_rejected"`` and
    ``parse_error`` is the JSONDecodeError message when applicable.
    Used by both the production parser
    (``_parse_common_ground_proposals``) and the diagnostic sink in
    ``LLMCommonGroundProposalRuntime``.
    """
    cleaned = strip_code_fence(text)
    try:
        payload = json.loads(cleaned.strip())
    except JSONDecodeError as exc:
        return ((), "parse_error", str(exc))
    if not isinstance(payload, list):
        return ((), "parse_error", f"top-level not a list: {type(payload).__name__}")
    proposals: list[CommonGroundProposal] = []
    for item in payload:
        proposal = _parse_common_ground_proposal(item)
        if proposal is not None:
            proposals.append(proposal)
    if not proposals:
        return ((), "empty_or_rejected", None)
    return (tuple(proposals), "ok", None)


def _parse_common_ground_proposal(item: object) -> CommonGroundProposal | None:
    if not isinstance(item, dict):
        return None
    scope_kind_raw = item.get("scope_kind")
    scope_id = item.get("scope_id")
    summary = item.get("summary")
    accepted_by_ids = item.get("accepted_by_ids")
    evidence = item.get("evidence")
    confidence = item.get("confidence")
    recursion_depth = item.get("recursion_depth")
    control_signal = item.get("control_signal", 0.0)
    if not isinstance(scope_kind_raw, str) or scope_kind_raw not in _VALID_SCOPE_KINDS:
        return None
    if not isinstance(scope_id, str) or not scope_id.strip():
        return None
    if not isinstance(summary, str) or not summary.strip():
        return None
    if not isinstance(accepted_by_ids, list):
        return None
    accepted = tuple(item for item in accepted_by_ids if isinstance(item, str))
    if len(accepted) != len(accepted_by_ids) or not accepted or len(set(accepted)) != len(accepted):
        return None
    if isinstance(evidence, str):
        evidence_tuple = (evidence.strip(),) if evidence.strip() else ()
    elif isinstance(evidence, list):
        evidence_tuple = tuple(item.strip() for item in evidence if isinstance(item, str) and item.strip())
    else:
        return None
    if not evidence_tuple:
        return None
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return None
    confidence_value = float(confidence)
    if confidence_value < _MIN_COMMON_GROUND_CONFIDENCE or confidence_value > 1.0:
        return None
    if isinstance(recursion_depth, bool) or not isinstance(recursion_depth, int):
        return None
    if recursion_depth < 0 or recursion_depth > MAX_COMMON_GROUND_RECURSION_DEPTH:
        return None
    if isinstance(control_signal, bool) or not isinstance(control_signal, (int, float)):
        return None
    control_value = float(control_signal)
    if control_value < 0.0 or control_value > 1.0:
        return None
    return CommonGroundProposal(
        scope_kind=SocialScopeKind(scope_kind_raw),
        scope_id=scope_id.strip(),
        summary=summary.strip()[:240],
        accepted_by_ids=accepted,
        evidence=evidence_tuple[:4],
        confidence=confidence_value,
        recursion_depth=recursion_depth,
        control_signal=control_value,
    )


__all__ = [
    "CommonGroundModule",
    "CommonGroundProposal",
    "CommonGroundProposalBatch",
    "LLMCommonGroundProposalRuntime",
]
