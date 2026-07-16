"""Reliable-apprenticeship alignment owner (``apprenticeship_alignment``).

Implements the kernel-side instantiation of *reliable active
apprenticeship learning* (Hanneke, Yang, Wang & Song, ALT 2025):

* the operator's teaching is an expert answer that constrains a
  version space ``V`` of operator-intended cognition;
* the AI may act reliably only inside the *agreement region* (its
  cognition is pinned by all guidance so far); otherwise it sits in the
  *disagreement region* and should defer / surface uncertainty;
* the *eluder*-style informativeness of guidance is reported as
  ``guidance_surprise`` and fused into the PE chain by the PE owner;
* the operator may be an *imperfect / noisy expert* (Massart / Tsybakov
  noise), so a contradiction is only *confirmed* once the conflicting
  constraints clear a reliability margin or recur — otherwise a single
  operator slip is not treated as a material defect;
* when no hypothesis satisfies all guidance the version space collapses
  (``INCONSISTENT``) and the minimal conflicting constraint set is
  located — the rigorous judge of "the material contradicts itself".

The owner consumes only public snapshots (R8) and publishes a frozen
``ApprenticeshipAlignmentSnapshot``. Belief revision is emitted as
``SemanticProposal`` records (drained by the session-post writeback
path, single-writer ``belief_assumption`` / ``goal_value``), never by
mutating another owner's state here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import json
from json import JSONDecodeError
from importlib.resources import files
from typing import Any, Mapping, Protocol

from volvence_zero.runtime import (
    RuntimeModule,
    RuntimePlaceholderValue,
    Snapshot,
    WiringLevel,
)
from volvence_zero.semantic_embedding import (
    semantic_cosine,
    semantic_embedding,
    semantic_topic_similarity,
    stub_semantic_tokens as _semantic_tokens,
)
from volvence_zero.semantic_state import (
    BeliefAssumptionSnapshot,
    BoundaryConsentSnapshot,
    GoalValueSnapshot,
    SemanticProposal,
    SemanticProposalOperation,
    SemanticRecord,
    UserModelSnapshot,
    apply_semantic_writeback_result,
)

from volvence_zero.apprenticeship.contracts import (
    ApprenticeshipAlignmentSnapshot,
    ConstraintLevel,
    ContradictionFinding,
    IntentConstraint,
    MismatchRef,
    MismatchType,
    ReliabilityState,
    VersionSpaceStatus,
    clamp_signed,
    clamp_unit,
)


# ---------------------------------------------------------------------------
# Centralized prompt loading (llm-prompt-centralization).
# ---------------------------------------------------------------------------
def load_apprenticeship_prompt_template(name: str = "extraction.md") -> str:
    return (
        files("volvence_zero.apprenticeship")
        .joinpath("prompts", name)
        .read_text(encoding="utf-8")
    )


# ---------------------------------------------------------------------------
# Constraint extraction. The default is a deterministic, LLM-free
# extractor that treats one teaching turn as one holistic constraint
# (no keyword stance inference). Stance-aware extraction — splitting a
# turn into typed factual / abstract sub-claims with +/- polarity — is
# the production path (an LLM structured-output extractor) and is also
# what tests inject via ``MappingConstraintExtractor`` to exercise the
# contradiction / mismatch logic.
# ---------------------------------------------------------------------------
class GuidanceConstraintExtractor(ABC):
    extractor_id: str

    @abstractmethod
    def extract(
        self, *, guidance_text: str, turn_index: int
    ) -> tuple[IntentConstraint, ...]:
        """Return typed operator-intent constraints for one teaching turn."""


class _GenerateProtocol(Protocol):
    def generate(
        self, *, prompt: str, max_new_tokens: int = ..., temperature: float = ...
    ) -> str: ...


def build_intent_constraint(
    *,
    constraint_id: str,
    statement: str,
    level: str = ConstraintLevel.ABSTRACT.value,
    polarity: int = 1,
    target_key: str | None = None,
    confidence: float = 0.5,
    source_turn: int = 0,
) -> IntentConstraint:
    """Construct an :class:`IntentConstraint`.

    Topic similarity (coverage / contradiction matching) is computed
    from ``target_key`` via character-bigram token overlap, which
    separates content for both Latin and CJK text (the stub embedding
    cannot). The ``embedding`` field is reserved for the production path
    (a real embedding head) and is left empty here.
    """
    topic = (target_key or statement).strip()
    return IntentConstraint(
        constraint_id=constraint_id,
        statement=statement[:240],
        level=level,
        polarity=int(polarity),
        target_key=topic[:160],
        confidence=clamp_unit(confidence),
        source_turn=source_turn,
        embedding=(),
    )


class HolisticGuidanceConstraintExtractor(GuidanceConstraintExtractor):
    """Default LLM-free extractor: one holistic constraint per turn.

    Produces a single neutral-assertion (``polarity=+1``) abstract
    constraint from the full guidance text. It deliberately does NOT
    infer stance / negation from keywords, so on its own it surfaces
    surprise / agreement / novelty but never fabricates a contradiction
    (which genuinely requires stance — supplied by the LLM extractor in
    production or by explicit typed constraints in tests).
    """

    extractor_id = "apprenticeship-holistic"

    def extract(
        self, *, guidance_text: str, turn_index: int
    ) -> tuple[IntentConstraint, ...]:
        text = (guidance_text or "").strip()
        if not text:
            return ()
        return (
            build_intent_constraint(
                constraint_id=f"guidance:{turn_index}:0",
                statement=text,
                level=ConstraintLevel.ABSTRACT.value,
                polarity=1,
                target_key=text,
                confidence=0.5,
                source_turn=turn_index,
            ),
        )


class MappingConstraintExtractor(GuidanceConstraintExtractor):
    """Replay extractor returning pre-typed constraints keyed by turn.

    Used both as a test seam (inject opposing-polarity constraints to
    drive version-space collapse) and as the integration shape an
    upstream ingestion / LLM adapter fills in.
    """

    extractor_id = "apprenticeship-mapping"

    def __init__(self, per_turn: Mapping[int, tuple[IntentConstraint, ...]]) -> None:
        self._per_turn = dict(per_turn)

    def extract(
        self, *, guidance_text: str, turn_index: int
    ) -> tuple[IntentConstraint, ...]:
        del guidance_text
        return tuple(self._per_turn.get(turn_index, ()))


class LLMGuidanceConstraintExtractor(GuidanceConstraintExtractor):
    """Structured LLM extractor for production apprenticeship turns.

    The prompt body is loaded from ``prompts/extraction.md`` so the LLM-facing
    instruction stays centralized. If the model returns malformed JSON, this
    falls back to the deterministic holistic extractor instead of fabricating
    constraints from a bad parse.
    """

    extractor_id = "apprenticeship-llm-structured"

    def __init__(self, provider: _GenerateProtocol) -> None:
        self._provider = provider
        self._fallback = HolisticGuidanceConstraintExtractor()

    def extract(
        self, *, guidance_text: str, turn_index: int
    ) -> tuple[IntentConstraint, ...]:
        text = (guidance_text or "").strip()
        if not text:
            return ()
        prompt = (
            f"{load_apprenticeship_prompt_template()}\n\n"
            "Current teaching turn:\n"
            '"""\n'
            f"{text}\n"
            '"""\n\n'
            "Return JSON only, as either a list of constraint objects or an object "
            'with a "constraints" list.'
        )
        raw = self._provider.generate(
            prompt=prompt,
            max_new_tokens=512,
            temperature=0.0,
        )
        try:
            payload = json.loads(_strip_json_fence(raw))
        except (JSONDecodeError, TypeError):
            return self._fallback.extract(
                guidance_text=text,
                turn_index=turn_index,
            )
        rows = payload.get("constraints") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            return self._fallback.extract(
                guidance_text=text,
                turn_index=turn_index,
            )
        constraints: list[IntentConstraint] = []
        for index, row in enumerate(rows[:8]):
            if not isinstance(row, dict):
                continue
            statement = str(row.get("statement", "")).strip()
            if not statement:
                continue
            level = str(row.get("level", ConstraintLevel.ABSTRACT.value)).strip()
            if level not in {ConstraintLevel.FACTUAL.value, ConstraintLevel.ABSTRACT.value}:
                level = ConstraintLevel.ABSTRACT.value
            try:
                polarity = int(row.get("polarity", 1))
            except (TypeError, ValueError):
                polarity = 1
            polarity = max(-1, min(1, polarity))
            try:
                confidence = float(row.get("confidence", 0.55))
            except (TypeError, ValueError):
                confidence = 0.55
            target_key = str(row.get("target_key", statement)).strip() or statement
            constraints.append(
                build_intent_constraint(
                    constraint_id=f"guidance:{turn_index}:{index}",
                    statement=statement,
                    level=level,
                    polarity=polarity,
                    target_key=target_key,
                    confidence=confidence,
                    source_turn=turn_index,
                )
            )
        if not constraints:
            return self._fallback.extract(
                guidance_text=text,
                turn_index=turn_index,
            )
        return tuple(constraints)


def _strip_json_fence(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


# ---------------------------------------------------------------------------
# Cognition projection: the records the guidance is compared against.
# ---------------------------------------------------------------------------
def _tokens(text: str) -> frozenset[str]:
    return frozenset(_semantic_tokens(text)) if text else frozenset()


def _constraint_tokens(constraint: IntentConstraint) -> frozenset[str]:
    return _tokens(constraint.target_key or constraint.statement)


def _constraint_embedding(constraint: IntentConstraint) -> tuple[float, ...]:
    if constraint.embedding:
        return tuple(constraint.embedding)
    return semantic_embedding(constraint.target_key or constraint.statement, dim=16)


class _CognitionRecord:
    __slots__ = ("embedding", "record_id", "tokens", "level")

    def __init__(self, *, record_id: str, text: str, level: str) -> None:
        self.record_id = record_id
        self.tokens = _tokens(text)
        self.embedding = semantic_embedding(text, dim=16)
        self.level = level


def _records_from_belief(value: BeliefAssumptionSnapshot) -> list[_CognitionRecord]:
    out: list[_CognitionRecord] = []
    seen: set[str] = set()
    for record in value.beliefs + value.assumptions:
        if record.record_id in seen:
            continue
        seen.add(record.record_id)
        out.append(
            _CognitionRecord(
                record_id=record.record_id,
                text=f"{record.summary} {record.detail}",
                level=ConstraintLevel.FACTUAL.value,
            )
        )
    return out


def _records_from_goal(value: GoalValueSnapshot) -> list[_CognitionRecord]:
    return [
        _CognitionRecord(
            record_id=record.record_id,
            text=f"{record.summary} {record.detail}",
            level=ConstraintLevel.ABSTRACT.value,
        )
        for record in value.explicit_goals
    ]


def _records_from_user_model(value: UserModelSnapshot) -> list[_CognitionRecord]:
    records: tuple[SemanticRecord, ...] = (
        value.stable_preferences + value.durable_goals + value.sensitive_boundaries
    )
    return [
        _CognitionRecord(
            record_id=record.record_id,
            text=f"{record.summary} {record.detail}",
            level=ConstraintLevel.ABSTRACT.value,
        )
        for record in records
    ]


def _records_from_boundary(value: BoundaryConsentSnapshot) -> list[_CognitionRecord]:
    records = value.granted_consents + value.denied_boundaries
    return [
        _CognitionRecord(
            record_id=record.record_id,
            text=f"{record.summary} {record.detail}",
            level=ConstraintLevel.ABSTRACT.value,
        )
        for record in records
    ]


def _nearest(
    constraint: IntentConstraint, cognition: list[_CognitionRecord]
) -> tuple[float, str]:
    embedding = _constraint_embedding(constraint)
    if not embedding:
        return (0.0, "")
    best_score = 0.0
    best_id = ""
    for record in cognition:
        score = max(0.0, semantic_cosine(embedding, record.embedding))
        if score > best_score:
            best_score = score
            best_id = record.record_id
    return (clamp_unit(best_score), best_id)


def _novelty_mismatch_type(level: str) -> str:
    return (
        MismatchType.FACTUAL_NOVELTY.value
        if level == ConstraintLevel.FACTUAL.value
        else MismatchType.ABSTRACT_NOVELTY.value
    )


def _contradiction_mismatch_type(level: str) -> str:
    return (
        MismatchType.DUALITY.value
        if level == ConstraintLevel.FACTUAL.value
        else MismatchType.VALUE_CONFLICT.value
    )


# ---------------------------------------------------------------------------
# Reconciler thresholds. Owner-internal SSOT; consumers read the
# published snapshot, never these constants.
# ---------------------------------------------------------------------------
class ApprenticeshipThresholds:
    # Surprise = 1 - coverage. Below ``agreement`` the cognition pins
    # the guidance (agreement region => reliable). At-or-above
    # ``mismatch`` the guidance is uncovered enough to surface a
    # mismatch; ``shrink`` is the version-space-shrinking floor.
    agreement: float = 0.40
    mismatch: float = 0.45
    shrink: float = 0.45
    # Topic-match cosine for two constraints to be contradiction
    # candidates (same topic, opposing polarity).
    contradiction_topic: float = 0.60
    # Tsybakov-style reliability margin: a contradiction is confirmed
    # only when both constraints clear this confidence, OR (Massart-style
    # recurrence) the opposing topic pair has been seen at least
    # ``recurrence_floor`` times.
    reliability: float = 0.55
    recurrence_floor: int = 2
    # Bounded version-space window so the owner stays memory-bounded.
    max_constraints: int = 64
    # #90: in the disagreement region, guidance this uncovered warrants
    # actively requesting feedback rather than committing silently. Set
    # to the mismatch floor so a request coincides with a surfaced
    # mismatch (an INCONSISTENT version space always requests regardless).
    feedback_request_surprise: float = 0.45


def reconcile_guidance(
    *,
    new_constraints: tuple[IntentConstraint, ...],
    prior_constraints: tuple[IntentConstraint, ...],
    cognition: list[_CognitionRecord],
    recurrence: dict[str, int],
    thresholds: ApprenticeshipThresholds,
) -> tuple[
    tuple[MismatchRef, ...],
    tuple[ContradictionFinding, ...],
    float,
    bool,
    str,
    float,
]:
    """Pure reconciliation step.

    Returns ``(mismatches, contradictions, guidance_surprise,
    in_agreement_region, version_space_status, consistency_margin)``.
    """
    mismatches: list[MismatchRef] = []
    surprises: list[float] = []
    agreement_flags: list[bool] = []

    for constraint in new_constraints:
        coverage, nearest_id = _nearest(constraint, cognition)
        surprise = clamp_unit(1.0 - coverage)
        surprises.append(surprise)
        agreement_flags.append(surprise < thresholds.agreement)
        if surprise >= thresholds.mismatch:
            mismatches.append(
                MismatchRef(
                    guidance_constraint_id=constraint.constraint_id,
                    level=constraint.level,
                    mismatch_type=_novelty_mismatch_type(constraint.level),
                    belief_ref=nearest_id,
                    severity=round(surprise, 4),
                    description=(
                        f"guidance '{constraint.statement[:60]}' uncovered by "
                        f"current cognition (coverage={coverage:.2f}, "
                        f"nearest={nearest_id or 'none'})"
                    ),
                )
            )

    contradictions = _detect_contradictions(
        new_constraints=new_constraints,
        prior_constraints=prior_constraints,
        recurrence=recurrence,
        thresholds=thresholds,
    )

    guidance_surprise = (
        round(sum(surprises) / len(surprises), 4) if surprises else 0.0
    )
    has_new = bool(new_constraints)
    in_agreement = (
        has_new and all(agreement_flags) and not contradictions
    )

    if contradictions:
        status = VersionSpaceStatus.INCONSISTENT.value
    elif not has_new:
        status = VersionSpaceStatus.IDLE.value
    elif guidance_surprise >= thresholds.shrink:
        status = VersionSpaceStatus.SHRINKING.value
    else:
        status = VersionSpaceStatus.CONSISTENT.value

    max_contradiction = max(
        (finding.severity for finding in contradictions), default=0.0
    )
    consistency_margin = round(clamp_unit(1.0 - max_contradiction), 4)

    return (
        tuple(mismatches),
        contradictions,
        guidance_surprise,
        in_agreement,
        status,
        consistency_margin,
    )


def _shared_topic_bucket(shared: frozenset[str], fallback: str) -> str:
    """Stable recurrence key from the topic two constraints share.

    Keyed on the *shared* tokens so the same opposing teaching maps to
    the same bucket across turns (Massart-style recurrence), regardless
    of which polarity is the current candidate.
    """
    if not shared:
        return fallback[:24]
    return "|".join(sorted(shared)[:8])


def _detect_contradictions(
    *,
    new_constraints: tuple[IntentConstraint, ...],
    prior_constraints: tuple[IntentConstraint, ...],
    recurrence: dict[str, int],
    thresholds: ApprenticeshipThresholds,
) -> tuple[ContradictionFinding, ...]:
    findings: list[ContradictionFinding] = []
    seen_pairs: set[frozenset[str]] = set()
    # Compare each new constraint against prior + earlier-new constraints.
    pool = list(prior_constraints) + list(new_constraints)
    pool_tokens = [_constraint_tokens(item) for item in pool]
    base = len(prior_constraints)
    for index, candidate in enumerate(new_constraints):
        cand_tokens = pool_tokens[base + index]
        if not cand_tokens or candidate.polarity == 0:
            continue
        # Only look at constraints that came before this candidate in the
        # combined pool to avoid double counting symmetric pairs.
        ceiling = base + index
        for position in range(ceiling):
            other = pool[position]
            other_tokens = pool_tokens[position]
            if not other_tokens or other.polarity == 0:
                continue
            if candidate.polarity * other.polarity >= 0:
                continue  # same stance, not a contradiction
            # M1 (#91 follow-up): backend-aware topic similarity — real
            # embedding cosine when a substrate backend is installed,
            # byte-identical stub-token Jaccard otherwise.
            topic_sim = semantic_topic_similarity(
                candidate.target_key or candidate.statement,
                other.target_key or other.statement,
            )
            if topic_sim < thresholds.contradiction_topic:
                continue
            pair_key = frozenset({candidate.constraint_id, other.constraint_id})
            if pair_key in seen_pairs:
                continue
            bucket = _shared_topic_bucket(
                cand_tokens & other_tokens, candidate.target_key
            )
            recurrence[bucket] = recurrence.get(bucket, 0) + 1
            margin_ok = (
                min(candidate.confidence, other.confidence) >= thresholds.reliability
            )
            recurred = recurrence[bucket] >= thresholds.recurrence_floor
            if not (margin_ok or recurred):
                continue  # unconfirmed: treat as operator noise, not a defect
            seen_pairs.add(pair_key)
            severity = round(
                clamp_unit(
                    topic_sim
                    * min(candidate.confidence, other.confidence)
                ),
                4,
            )
            findings.append(
                ContradictionFinding(
                    finding_id=f"contradiction:{other.constraint_id}->{candidate.constraint_id}",
                    constraint_ids=(other.constraint_id, candidate.constraint_id),
                    level=candidate.level,
                    severity=severity,
                    description=(
                        f"opposing guidance on the same topic "
                        f"(sim={topic_sim:.2f}, margin_ok={margin_ok}, "
                        f"recurred={recurred}): "
                        f"'{other.statement[:40]}' vs '{candidate.statement[:40]}'"
                    ),
                )
            )
    return tuple(findings)


def _random_feedback_sample(*, guidance_text: str, turn_index: int) -> bool:
    """Deterministic pseudo-random request used only for matched controls.

    The active-learning-off ablation must not read owner disagreement to decide
    whether to ask for labels. Hashing turn identity gives a reproducible
    random-sampling baseline without importing process-global RNG state.
    """

    seed = f"{turn_index}:{guidance_text}".encode("utf-8")
    digest = hashlib.blake2b(seed, digest_size=2).digest()
    value = int.from_bytes(digest, "big") / 65535.0
    return value < 0.33


def _idle_snapshot(reason: str) -> ApprenticeshipAlignmentSnapshot:
    return ApprenticeshipAlignmentSnapshot(
        version_space_status=VersionSpaceStatus.IDLE.value,
        consistency_margin=1.0,
        reliability=ReliabilityState.IDLE.value,
        in_agreement_region=False,
        guidance_surprise=0.0,
        active_constraint_count=0,
        mismatch_refs=(),
        contradiction_findings=(),
        revision_proposal_refs=(),
        description=f"Apprenticeship alignment idle: {reason}.",
    )


class ApprenticeshipAlignmentModule(RuntimeModule[ApprenticeshipAlignmentSnapshot]):
    """Owner of the ``apprenticeship_alignment`` slot.

    Consumes the four cognition owners it compares guidance against, plus
    ``regime`` for current-state context. SHADOW by default (migration:
    SHADOW -> validate -> ACTIVE; reversible per R15). When
    ``revision_enabled`` is True (Phase 3) it also generates AGM
    minimal-change ``SemanticProposal`` records, drained by the
    session-post writeback path — it never writes the belief/goal stores
    itself.
    """

    slot_name = "apprenticeship_alignment"
    owner = "ApprenticeshipAlignmentModule"
    value_type = ApprenticeshipAlignmentSnapshot
    dependencies = (
        "belief_assumption",
        "goal_value",
        "user_model",
        "boundary_consent",
        "regime",
    )
    default_wiring_level = WiringLevel.SHADOW

    def __init__(
        self,
        *,
        wiring_level: WiringLevel | None = None,
        user_input: str | None = None,
        turn_index: int = 0,
        apprenticeship: bool = False,
        extractor: GuidanceConstraintExtractor | None = None,
        feedback_policy: str = "owner",
        revision_enabled: bool = False,
        thresholds: ApprenticeshipThresholds | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        if feedback_policy not in {"owner", "random", "disabled"}:
            raise ValueError(
                "feedback_policy must be one of 'owner', 'random', 'disabled'"
            )
        self._user_input = user_input
        self._turn_index = turn_index
        self._apprenticeship = apprenticeship
        self._extractor = extractor or HolisticGuidanceConstraintExtractor()
        self._feedback_policy = feedback_policy
        self._revision_enabled = revision_enabled
        self._thresholds = thresholds or ApprenticeshipThresholds()
        self._constraints: list[IntentConstraint] = []
        self._recurrence: dict[str, int] = {}
        self._pending_revision_proposals: tuple[SemanticProposal, ...] = ()

    # -- public accessors -------------------------------------------------
    def drain_revision_proposals(self) -> tuple[SemanticProposal, ...]:
        """Return and clear this turn's belief-revision proposals.

        The session-post writeback path applies these via
        :func:`apply_apprenticeship_revisions` so ``belief_assumption`` /
        ``goal_value`` stay single-writer.
        """
        proposals = self._pending_revision_proposals
        self._pending_revision_proposals = ()
        return proposals

    @property
    def constraint_count(self) -> int:
        return len(self._constraints)

    # -- runtime ----------------------------------------------------------
    async def process(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> Snapshot[ApprenticeshipAlignmentSnapshot]:
        cognition = self._collect_cognition(upstream)
        snapshot = self._run(
            cognition=cognition,
            guidance_text=self._user_input,
            turn_index=self._turn_index,
        )
        return self.publish(snapshot)

    async def process_standalone(
        self, **kwargs: Any
    ) -> Snapshot[ApprenticeshipAlignmentSnapshot]:
        guidance_text = kwargs.get("guidance_text", self._user_input)
        turn_index = int(kwargs.get("turn_index", self._turn_index))
        if "apprenticeship" in kwargs:
            self._apprenticeship = bool(kwargs["apprenticeship"])
        cognition = self._collect_cognition_from_values(
            belief=kwargs.get("belief_assumption"),
            goal=kwargs.get("goal_value"),
            user_model=kwargs.get("user_model"),
            boundary=kwargs.get("boundary_consent"),
        )
        snapshot = self._run(
            cognition=cognition,
            guidance_text=guidance_text,
            turn_index=turn_index,
            explicit_constraints=kwargs.get("constraints"),
        )
        return self.publish(snapshot)

    # -- internals --------------------------------------------------------
    def _collect_cognition(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> list[_CognitionRecord]:
        return self._collect_cognition_from_values(
            belief=self._value(upstream.get("belief_assumption")),
            goal=self._value(upstream.get("goal_value")),
            user_model=self._value(upstream.get("user_model")),
            boundary=self._value(upstream.get("boundary_consent")),
        )

    @staticmethod
    def _value(snapshot: Snapshot[Any] | None) -> Any:
        if snapshot is None:
            return None
        value = snapshot.value
        if isinstance(value, RuntimePlaceholderValue):
            return None
        return value

    @staticmethod
    def _collect_cognition_from_values(
        *,
        belief: Any,
        goal: Any,
        user_model: Any,
        boundary: Any,
    ) -> list[_CognitionRecord]:
        records: list[_CognitionRecord] = []
        if isinstance(belief, BeliefAssumptionSnapshot):
            records.extend(_records_from_belief(belief))
        if isinstance(goal, GoalValueSnapshot):
            records.extend(_records_from_goal(goal))
        if isinstance(user_model, UserModelSnapshot):
            records.extend(_records_from_user_model(user_model))
        if isinstance(boundary, BoundaryConsentSnapshot):
            records.extend(_records_from_boundary(boundary))
        return records

    def _run(
        self,
        *,
        cognition: list[_CognitionRecord],
        guidance_text: str | None,
        turn_index: int,
        explicit_constraints: tuple[IntentConstraint, ...] | None = None,
    ) -> ApprenticeshipAlignmentSnapshot:
        self._pending_revision_proposals = ()
        if not self._apprenticeship:
            return _idle_snapshot("not an apprenticeship turn")

        if explicit_constraints is not None:
            new_constraints = tuple(explicit_constraints)
        else:
            new_constraints = self._extractor.extract(
                guidance_text=guidance_text or "", turn_index=turn_index
            )
        if not new_constraints:
            return _idle_snapshot("no guidance constraints extracted")

        prior = tuple(self._constraints)
        (
            mismatches,
            contradictions,
            guidance_surprise,
            in_agreement,
            status,
            consistency_margin,
        ) = reconcile_guidance(
            new_constraints=new_constraints,
            prior_constraints=prior,
            cognition=cognition,
            recurrence=self._recurrence,
            thresholds=self._thresholds,
        )

        # Append to the bounded version-space window.
        self._constraints.extend(new_constraints)
        if len(self._constraints) > self._thresholds.max_constraints:
            self._constraints = self._constraints[-self._thresholds.max_constraints :]

        reliability = (
            ReliabilityState.RELIABLE.value
            if in_agreement
            else ReliabilityState.DEFERRING.value
        )

        # #90 active-learning feedback request: in the disagreement region
        # with informative-enough guidance, or a self-contradictory version
        # space, the owner should actively seek guidance rather than commit
        # silently. Derived from this owner's own state (no keyword match).
        is_inconsistent = status == VersionSpaceStatus.INCONSISTENT.value
        owner_should_request_feedback = is_inconsistent or (
            reliability == ReliabilityState.DEFERRING.value
            and guidance_surprise >= self._thresholds.feedback_request_surprise
        )
        if self._feedback_policy == "owner":
            should_request_feedback = owner_should_request_feedback
        elif self._feedback_policy == "disabled":
            should_request_feedback = False
        else:
            should_request_feedback = _random_feedback_sample(
                guidance_text=guidance_text or "",
                turn_index=turn_index,
            )
        max_contradiction_severity = max(
            (finding.severity for finding in contradictions), default=0.0
        )
        feedback_request_urgency = (
            clamp_unit(max(guidance_surprise, max_contradiction_severity))
            if should_request_feedback
            else 0.0
        )
        if not should_request_feedback:
            feedback_request_reason = ""
        elif self._feedback_policy == "random":
            feedback_request_reason = (
                "random_sampling_baseline: feedback requested by matched-control "
                "sampling, not owner disagreement"
            )
        elif is_inconsistent:
            feedback_request_reason = (
                "version_space_inconsistent: teaching contradicts itself; "
                "seek disambiguating guidance"
            )
        else:
            feedback_request_reason = (
                "disagreement_region: guidance uncovered by current cognition "
                f"(surprise={guidance_surprise:.2f}); confirm before committing"
            )

        revision_refs: tuple[str, ...] = ()
        if self._revision_enabled:
            proposals = self._build_revision_proposals(
                new_constraints=new_constraints,
                mismatches=mismatches,
                contradictions=contradictions,
                turn_index=turn_index,
            )
            self._pending_revision_proposals = proposals
            revision_refs = tuple(p.proposal_id for p in proposals)

        description = (
            f"Apprenticeship alignment: status={status} "
            f"reliability={reliability} surprise={guidance_surprise:.2f} "
            f"agreement={in_agreement} margin={consistency_margin:.2f} "
            f"new_constraints={len(new_constraints)} "
            f"mismatches={len(mismatches)} contradictions={len(contradictions)} "
            f"revisions={len(revision_refs)} "
            f"window={len(self._constraints)}."
        )
        return ApprenticeshipAlignmentSnapshot(
            version_space_status=status,
            consistency_margin=consistency_margin,
            reliability=reliability,
            in_agreement_region=in_agreement,
            guidance_surprise=guidance_surprise,
            active_constraint_count=len(self._constraints),
            mismatch_refs=mismatches,
            contradiction_findings=contradictions,
            revision_proposal_refs=revision_refs,
            description=description,
            guidance_constraints=tuple(new_constraints),
            should_request_feedback=should_request_feedback,
            feedback_request_reason=feedback_request_reason,
            feedback_request_urgency=feedback_request_urgency,
        )

    def _build_revision_proposals(
        self,
        *,
        new_constraints: tuple[IntentConstraint, ...],
        mismatches: tuple[MismatchRef, ...],
        contradictions: tuple[ContradictionFinding, ...],
        turn_index: int,
    ) -> tuple[SemanticProposal, ...]:
        """AGM minimal-change revision proposals (Phase 3).

        * A confirmed (reliable) novel mismatch CREATEs / REVISEs only
          the single touched record — preserving unrelated beliefs
          (AGM Inclusion + Preservation).
        * A confirmed contradiction BLOCKs the conflicting belief and
          requires confirmation — competing hypotheses are held, not
          silently overwritten (noisy-expert safety).
        """
        constraint_by_id = {c.constraint_id: c for c in new_constraints}
        proposals: list[SemanticProposal] = []

        for mismatch in mismatches:
            constraint = constraint_by_id.get(mismatch.guidance_constraint_id)
            if constraint is None:
                continue
            if constraint.confidence < self._thresholds.reliability:
                continue  # unreliable guidance: do not revise on noise
            target_slot = (
                "belief_assumption"
                if constraint.level == ConstraintLevel.FACTUAL.value
                else "goal_value"
            )
            operation = (
                SemanticProposalOperation.REVISE
                if mismatch.belief_ref
                else SemanticProposalOperation.CREATE
            )
            proposals.append(
                SemanticProposal(
                    proposal_id=(
                        f"apprenticeship:{turn_index}:{constraint.constraint_id}:"
                        f"{operation.value}"
                    ),
                    target_slot=target_slot,
                    operation=operation,
                    summary=constraint.statement[:160],
                    detail=constraint.statement[:320],
                    confidence=constraint.confidence,
                    evidence=(
                        f"apprenticeship-guidance turn={turn_index} "
                        f"mismatch={mismatch.mismatch_type} "
                        f"belief_ref={mismatch.belief_ref or 'none'}"
                    ),
                    control_signal=clamp_unit(mismatch.severity),
                )
            )

        for finding in contradictions:
            for constraint_id in finding.constraint_ids:
                constraint = constraint_by_id.get(constraint_id)
                if constraint is None:
                    continue
                target_slot = (
                    "belief_assumption"
                    if constraint.level == ConstraintLevel.FACTUAL.value
                    else "goal_value"
                )
                proposals.append(
                    SemanticProposal(
                        proposal_id=(
                            f"apprenticeship:{turn_index}:{finding.finding_id}:"
                            f"{constraint_id}:block"
                        ),
                        target_slot=target_slot,
                        operation=SemanticProposalOperation.BLOCK,
                        summary=constraint.statement[:160],
                        detail=finding.description[:320],
                        confidence=constraint.confidence,
                        evidence=(
                            f"apprenticeship-contradiction turn={turn_index} "
                            f"finding={finding.finding_id}"
                        ),
                        control_signal=clamp_unit(finding.severity),
                        requires_confirmation=True,
                    )
                )
        return tuple(proposals)


def apply_apprenticeship_revisions(
    *,
    store: Any,
    proposals: tuple[SemanticProposal, ...],
    turn_index: int,
) -> tuple[str, ...]:
    """Apply drained revision proposals through the single-writer store.

    Thin wrapper over :func:`apply_semantic_writeback_result` so the
    apprenticeship owner's belief revision uses the exact same
    single-writer path as reflection writeback (R8 SSOT). Returns the
    operation audit strings.
    """
    if not proposals:
        return ()
    return apply_semantic_writeback_result(
        store=store,
        proposals=proposals,
        turn_index=turn_index,
    )
