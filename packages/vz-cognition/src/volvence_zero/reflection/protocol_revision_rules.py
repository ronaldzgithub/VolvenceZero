"""Reflection rules for protocol revision proposals (packets 3.1 + 3.2).

Each rule is a pure function from ``(pe_history,
active_mixture_history)`` to a tuple of
:class:`ProtocolRevisionProposal`. Rules are stateless and
deterministic (history → proposals is a pure mapping); the
``ProtocolReflectionEngine`` is the only stateful holder.

Packet 3.1 lands the rules-aggregator scaffold with no rules
populated. Packet 3.2 implements the three core rules:

* :func:`propose_strategy_decay` — strategy_prior with
  consistently negative attribution-weighted PE.
* :func:`propose_knowledge_archival` — knowledge_seed with no
  observed retrieval over the window.
* :func:`propose_case_archival` — signature_case with no
  observed match over the window.
"""

from __future__ import annotations

from collections import defaultdict

from volvence_zero.behavior_protocol import (
    ActiveMixtureSnapshot,
    ProposalEvidence,
    ProtocolRevisionChangeKind,
    ProtocolRevisionProposal,
    ProtocolRevisionTargetField,
    ReviewLevel,
)
from volvence_zero.prediction import PredictionErrorSnapshot


# ---------------------------------------------------------------------------
# Tunables — exposed as module-level constants so tests can read them
# ---------------------------------------------------------------------------


# Minimum number of turns over which a strategy must have been
# attributed for ``propose_strategy_decay`` to consider it.
STRATEGY_DECAY_MIN_TURNS: int = 5

# Mean signed_reward threshold below which a strategy is flagged.
# More negative = more strict (we only flag really bad strategies).
STRATEGY_DECAY_PE_THRESHOLD: float = -0.3


def _attribution_pairs(
    pe_history: tuple[PredictionErrorSnapshot, ...],
    active_mixture_history: tuple[ActiveMixtureSnapshot, ...],
) -> list[tuple[float, dict[str, float]]]:
    """Pair PE turn ``t`` with active_mixture from turn ``t-1``.

    PE at turn t reflects the outcome of last turn's mixture, so
    the per-protocol attribution-weighted reward is
    ``signed_reward * weight`` for each protocol active at turn
    t-1. Returns a list of ``(signed_reward, {protocol_id: weight})``
    pairs in chronological order, dropping turns where we don't
    yet have a paired prior mixture.
    """

    if not pe_history or not active_mixture_history:
        return []

    # Align by index: PE[i] pairs with active_mixture[i-1] when both
    # exist. This is the same alignment the runtime owner uses.
    pairs: list[tuple[float, dict[str, float]]] = []
    history_size = min(len(pe_history), len(active_mixture_history))
    for i in range(1, history_size):
        pe = pe_history[i]
        last_mixture = active_mixture_history[i - 1]
        weights = {
            entry.protocol_id: entry.activation_weight
            for entry in last_mixture.active_protocols
        }
        pairs.append((float(pe.error.signed_reward), weights))
    return pairs


def propose_strategy_decay(
    *,
    pe_history: tuple[PredictionErrorSnapshot, ...],
    active_mixture_history: tuple[ActiveMixtureSnapshot, ...],
) -> tuple[ProtocolRevisionProposal, ...]:
    """Flag protocols whose attribution-weighted PE has been
    consistently negative.

    For each protocol that appeared in at least
    ``STRATEGY_DECAY_MIN_TURNS`` of the last
    ``len(pe_history)`` turns, compute the mean of
    ``signed_reward * weight`` across those turns. If the mean
    is below ``STRATEGY_DECAY_PE_THRESHOLD`` the rule emits a
    ``WEIGHT_DECAY`` proposal at L3.

    Note: this rule operates at *protocol* granularity (rather
    than per-strategy_prior). The ``target_entry_id`` is the
    protocol id; downstream apply layer (packet 3.3) decides
    whether to decay all strategies or pick the worst one.
    """

    pairs = _attribution_pairs(pe_history, active_mixture_history)
    if not pairs:
        return ()

    # Aggregate per-protocol stats.
    sum_attribution: dict[str, float] = defaultdict(float)
    count_turns: dict[str, int] = defaultdict(int)
    for signed_reward, weights in pairs:
        for protocol_id, weight in weights.items():
            if weight <= 0.0:
                continue
            sum_attribution[protocol_id] += signed_reward * weight
            count_turns[protocol_id] += 1

    proposals: list[ProtocolRevisionProposal] = []
    for protocol_id, total in sum_attribution.items():
        turns = count_turns[protocol_id]
        if turns < STRATEGY_DECAY_MIN_TURNS:
            continue
        mean_attribution = total / turns
        if mean_attribution >= STRATEGY_DECAY_PE_THRESHOLD:
            continue
        evidence = ProposalEvidence(
            observation_window_turns=turns,
            pe_signature=(
                f"mean_attributed_signed_reward={mean_attribution:.3f} "
                f"over {turns} turns (threshold "
                f"{STRATEGY_DECAY_PE_THRESHOLD:.2f})"
            ),
            summary=(
                f"protocol {protocol_id!r} averaged "
                f"{mean_attribution:.3f} attribution-weighted PE "
                f"across {turns} turns; below threshold."
            ),
        )
        proposals.append(
            ProtocolRevisionProposal(
                proposal_id=(
                    f"reflect:strategy_decay:{protocol_id}:"
                    f"window={turns}"
                ),
                target_protocol_id=protocol_id,
                target_field=ProtocolRevisionTargetField.STRATEGY_PRIOR,
                target_entry_id=protocol_id,  # protocol-granularity proposal
                change_kind=ProtocolRevisionChangeKind.WEIGHT_DECAY,
                evidence=evidence,
                proposed_payload={
                    "weight_multiplier": 0.5,
                    "mean_attributed_signed_reward": mean_attribution,
                },
                required_review_level=ReviewLevel.L3,
            )
        )

    return tuple(proposals)


KNOWLEDGE_ARCHIVAL_MIN_TURNS: int = 10
CASE_ARCHIVAL_MIN_TURNS: int = 10


def propose_knowledge_archival(
    *,
    pe_history: tuple[PredictionErrorSnapshot, ...],
    active_mixture_history: tuple[ActiveMixtureSnapshot, ...],
    knowledge_hit_history: tuple[tuple[str, ...], ...] = (),
    case_hit_history: tuple[tuple[str, ...], ...] = (),
) -> tuple[ProtocolRevisionProposal, ...]:
    """Packet 6.2: scan ``knowledge_hit_history`` for protocol-prefixed
    record_ids that NEVER appeared in any turn's hit set across the
    last ``KNOWLEDGE_ARCHIVAL_MIN_TURNS`` turns where the protocol
    was active.

    Detection: per protocol_id, find ``protocol:{protocol_id}:knowledge:*``
    record_ids that NEVER show up in any ``knowledge_hit_history`` window.
    Without owner-side knowledge of the protocol's seed list, this
    rule's signal is "never observed" — we look at all observed
    record_ids tagged ``protocol:`` and find protocol_ids whose seeds
    are entirely missing from the last N turns.

    Heuristic v0: only emits when we have ``KNOWLEDGE_ARCHIVAL_MIN_TURNS``
    turns of history AND at least one knowledge hit was observed
    (otherwise we don't have signal — protocol just isn't being used
    for retrieval at all).
    """

    if len(knowledge_hit_history) < KNOWLEDGE_ARCHIVAL_MIN_TURNS:
        return ()
    return ()  # Reserved for richer per-seed evidence in a follow-up
    # pass. The current heuristic doesn't cleanly identify a single
    # "stale seed" target_entry_id without owner-side knowledge of
    # which seed_ids belong to which protocol. The rule scaffold is
    # in place for a follow-up packet to populate when the
    # ProtocolRegistry exposes seed_ids per protocol via a snapshot
    # — until then this returns no proposals (no false positives).


def propose_case_archival(
    *,
    pe_history: tuple[PredictionErrorSnapshot, ...],
    active_mixture_history: tuple[ActiveMixtureSnapshot, ...],
    knowledge_hit_history: tuple[tuple[str, ...], ...] = (),
    case_hit_history: tuple[tuple[str, ...], ...] = (),
) -> tuple[ProtocolRevisionProposal, ...]:
    """Mirror of :func:`propose_knowledge_archival` for case_memory.

    Same caveats: scaffolding in place; awaiting per-protocol
    seed_id introspection on the registry snapshot before this
    rule can identify which specific case_id to archive.
    """

    if len(case_hit_history) < CASE_ARCHIVAL_MIN_TURNS:
        return ()
    return ()


# ---------------------------------------------------------------------------
# Packet 6.1 — propose strategy reinforcement (symmetric of decay)
# ---------------------------------------------------------------------------


# Mean attribution above which a protocol is flagged for reinforcement.
STRATEGY_REINFORCE_PE_THRESHOLD: float = 0.3
STRATEGY_REINFORCE_MIN_TURNS: int = 5


def propose_strategy_reinforce(
    *,
    pe_history: tuple[PredictionErrorSnapshot, ...],
    active_mixture_history: tuple[ActiveMixtureSnapshot, ...],
) -> tuple[ProtocolRevisionProposal, ...]:
    """Symmetric of :func:`propose_strategy_decay` — reinforce protocols
    whose mean attribution-weighted PE has been consistently positive.
    """

    pairs = _attribution_pairs(pe_history, active_mixture_history)
    if not pairs:
        return ()

    sum_attribution: dict[str, float] = defaultdict(float)
    count_turns: dict[str, int] = defaultdict(int)
    for signed_reward, weights in pairs:
        for protocol_id, weight in weights.items():
            if weight <= 0.0:
                continue
            sum_attribution[protocol_id] += signed_reward * weight
            count_turns[protocol_id] += 1

    proposals: list[ProtocolRevisionProposal] = []
    for protocol_id, total in sum_attribution.items():
        turns = count_turns[protocol_id]
        if turns < STRATEGY_REINFORCE_MIN_TURNS:
            continue
        mean_attribution = total / turns
        if mean_attribution <= STRATEGY_REINFORCE_PE_THRESHOLD:
            continue
        evidence = ProposalEvidence(
            observation_window_turns=turns,
            pe_signature=(
                f"mean_attributed_signed_reward={mean_attribution:.3f} "
                f"over {turns} turns (threshold "
                f"+{STRATEGY_REINFORCE_PE_THRESHOLD:.2f})"
            ),
            summary=(
                f"protocol {protocol_id!r} averaged "
                f"+{mean_attribution:.3f} PE over {turns} turns; "
                "reinforce strategy weights."
            ),
        )
        proposals.append(
            ProtocolRevisionProposal(
                proposal_id=(
                    f"reflect:strategy_reinforce:{protocol_id}:"
                    f"window={turns}"
                ),
                target_protocol_id=protocol_id,
                target_field=ProtocolRevisionTargetField.STRATEGY_PRIOR,
                target_entry_id=protocol_id,
                change_kind=ProtocolRevisionChangeKind.WEIGHT_REINFORCE,
                evidence=evidence,
                proposed_payload={
                    "weight_multiplier": 1.5,
                    "mean_attributed_signed_reward": mean_attribution,
                },
                required_review_level=ReviewLevel.L1,
            )
        )

    return tuple(proposals)


# ---------------------------------------------------------------------------
# Packet 6.1 — propose protocol retirement
# ---------------------------------------------------------------------------


PROTOCOL_RETIREMENT_PE_THRESHOLD: float = -0.5
PROTOCOL_RETIREMENT_MIN_TURNS: int = 12


def propose_protocol_retirement(
    *,
    pe_history: tuple[PredictionErrorSnapshot, ...],
    active_mixture_history: tuple[ActiveMixtureSnapshot, ...],
) -> tuple[ProtocolRevisionProposal, ...]:
    """Flag protocols whose mean attributed PE has been catastrophically
    negative for an extended window — propose RETIREMENT.

    Stricter than ``propose_strategy_decay`` (which proposes weight
    decay first); retirement is the terminal escalation.
    """

    pairs = _attribution_pairs(pe_history, active_mixture_history)
    if not pairs:
        return ()

    sum_attribution: dict[str, float] = defaultdict(float)
    count_turns: dict[str, int] = defaultdict(int)
    for signed_reward, weights in pairs:
        for protocol_id, weight in weights.items():
            if weight <= 0.0:
                continue
            sum_attribution[protocol_id] += signed_reward * weight
            count_turns[protocol_id] += 1

    proposals: list[ProtocolRevisionProposal] = []
    for protocol_id, total in sum_attribution.items():
        turns = count_turns[protocol_id]
        if turns < PROTOCOL_RETIREMENT_MIN_TURNS:
            continue
        mean_attribution = total / turns
        if mean_attribution >= PROTOCOL_RETIREMENT_PE_THRESHOLD:
            continue
        evidence = ProposalEvidence(
            observation_window_turns=turns,
            pe_signature=(
                f"sustained_negative_mean_pe={mean_attribution:.3f} "
                f"over {turns} turns (retirement threshold "
                f"{PROTOCOL_RETIREMENT_PE_THRESHOLD:.2f})"
            ),
            summary=(
                f"protocol {protocol_id!r} averaged "
                f"{mean_attribution:.3f} attributed PE over "
                f"{turns} turns; below catastrophic retirement "
                "threshold — propose retirement."
            ),
        )
        proposals.append(
            ProtocolRevisionProposal(
                proposal_id=f"reflect:protocol_retirement:{protocol_id}",
                target_protocol_id=protocol_id,
                target_field=ProtocolRevisionTargetField.STRATEGY_PRIOR,  # field-agnostic
                target_entry_id=protocol_id,
                change_kind=ProtocolRevisionChangeKind.PROTOCOL_RETIREMENT,
                evidence=evidence,
                proposed_payload={
                    "mean_attributed_signed_reward": mean_attribution,
                },
                required_review_level=ReviewLevel.L4,  # fail-safe queued
            )
        )

    return tuple(proposals)


# ---------------------------------------------------------------------------
# Packet 5.2 — propose new strategy_prior (NewStrategyPrior path)
# ---------------------------------------------------------------------------


# Minimum number of consecutive positive-PE turns (signed_reward > 0)
# observed without any protocol being attributed enough weight to
# count as "covered" — these are "successful but no rule matched"
# moments worth proposing a new strategy for.
ADD_STRATEGY_MIN_TURNS: int = 4

# Threshold above which signed_reward counts as "successful".
ADD_STRATEGY_SUCCESS_THRESHOLD: float = 0.4


def propose_strategy_addition(
    *,
    pe_history: tuple[PredictionErrorSnapshot, ...],
    active_mixture_history: tuple[ActiveMixtureSnapshot, ...],
) -> tuple[ProtocolRevisionProposal, ...]:
    """Detect "successful turns where existing strategies didn't fire"
    and propose adding a new strategy entry.

    Heuristic v0 (packet 5.2): for each protocol that has been in
    the active mixture across the window, if there are
    ``ADD_STRATEGY_MIN_TURNS`` consecutive turns where:

    * ``signed_reward >= ADD_STRATEGY_SUCCESS_THRESHOLD`` (we're
      doing well), AND
    * the protocol's per-turn weight is low (≤ 0.3) — meaning
      "we succeeded *despite* no protocol actively shaping the
      response"

    then emit an ADD_STRATEGY proposal targeting that protocol
    with a synthesised payload reflecting the dominant
    interlocutor regime label observed during those turns.

    This is a deliberately conservative starting heuristic — the
    LLM-augmented richer payload generation lands later (packet
    7.x). The point of packet 5.2 is to ship the path so the
    feedback loop for "system actually grows new rules" exists.
    """

    pairs = _attribution_pairs(pe_history, active_mixture_history)
    if len(pairs) < ADD_STRATEGY_MIN_TURNS:
        return ()

    # For each protocol, scan windowed runs of (success + low-weight).
    proposals: list[ProtocolRevisionProposal] = []
    seen_protocols: set[str] = set()
    for _, weights in pairs:
        for protocol_id in weights:
            seen_protocols.add(protocol_id)

    for protocol_id in sorted(seen_protocols):
        consecutive_success_low_weight = 0
        for signed_reward, weights in pairs:
            weight = weights.get(protocol_id, 0.0)
            if (
                signed_reward >= ADD_STRATEGY_SUCCESS_THRESHOLD
                and weight <= 0.3
            ):
                consecutive_success_low_weight += 1
            else:
                consecutive_success_low_weight = 0
            if consecutive_success_low_weight >= ADD_STRATEGY_MIN_TURNS:
                evidence = ProposalEvidence(
                    observation_window_turns=ADD_STRATEGY_MIN_TURNS,
                    pe_signature=(
                        f"{ADD_STRATEGY_MIN_TURNS}_consecutive_success_with_"
                        f"protocol_weight_below_0.3"
                    ),
                    summary=(
                        f"protocol {protocol_id!r} observed "
                        f"{ADD_STRATEGY_MIN_TURNS} consecutive successful "
                        "turns with low activation weight; an unmatched "
                        "successful pattern suggests a new strategy_prior."
                    ),
                )
                proposed_rule_id = (
                    f"reflection-add:{protocol_id}:autosynth"
                )
                proposals.append(
                    ProtocolRevisionProposal(
                        proposal_id=(
                            f"reflect:strategy_addition:{protocol_id}:"
                            f"autosynth"
                        ),
                        target_protocol_id=protocol_id,
                        target_field=ProtocolRevisionTargetField.STRATEGY_PRIOR,
                        target_entry_id=proposed_rule_id,
                        change_kind=ProtocolRevisionChangeKind.ADD_STRATEGY,
                        evidence=evidence,
                        proposed_payload={
                            "rule_id": proposed_rule_id,
                            "problem_pattern": (
                                "auto-synth: low-weight successful pattern"
                            ),
                            "recommended_ordering": (
                                "preserve_current_response",
                                "log_for_reviewer_attention",
                            ),
                            "recommended_pacing": "moderate",
                            "applicability_phase": (),
                            "initial_weight": 0.5,
                            "confidence": 0.5,
                            "description": (
                                "auto-synthesised by reflection rule; "
                                "review and refine to specialise the "
                                "problem_pattern + ordering"
                            ),
                        },
                        required_review_level=ReviewLevel.L3,
                    )
                )
                # Only one proposal per protocol per window.
                break

    return tuple(proposals)


def run_all_protocol_revision_rules(
    *,
    pe_history: tuple[PredictionErrorSnapshot, ...],
    active_mixture_history: tuple[ActiveMixtureSnapshot, ...],
    knowledge_hit_history: tuple[tuple[str, ...], ...] = (),
    case_hit_history: tuple[tuple[str, ...], ...] = (),
) -> tuple[ProtocolRevisionProposal, ...]:
    """Aggregate all rules into a single proposals tuple.

    Deduplicates by ``proposal_id``. Rules that don't take the
    ``knowledge_hit_history`` / ``case_hit_history`` parameters
    are called with the legacy 2-arg signature (the dispatch
    table maps each rule to the args it expects).
    """

    pe_only_rules = (
        propose_strategy_decay,
        propose_strategy_reinforce,
        propose_strategy_addition,
        propose_protocol_retirement,
    )
    archival_rules = (
        propose_knowledge_archival,
        propose_case_archival,
    )
    bucket: dict[str, ProtocolRevisionProposal] = {}
    for rule in pe_only_rules:
        for proposal in rule(
            pe_history=pe_history,
            active_mixture_history=active_mixture_history,
        ):
            bucket[proposal.proposal_id] = proposal
    for rule in archival_rules:
        for proposal in rule(
            pe_history=pe_history,
            active_mixture_history=active_mixture_history,
            knowledge_hit_history=knowledge_hit_history,
            case_hit_history=case_hit_history,
        ):
            bucket[proposal.proposal_id] = proposal
    return tuple(bucket.values())


__all__ = [
    "ADD_STRATEGY_MIN_TURNS",
    "ADD_STRATEGY_SUCCESS_THRESHOLD",
    "CASE_ARCHIVAL_MIN_TURNS",
    "KNOWLEDGE_ARCHIVAL_MIN_TURNS",
    "PROTOCOL_RETIREMENT_MIN_TURNS",
    "PROTOCOL_RETIREMENT_PE_THRESHOLD",
    "STRATEGY_DECAY_MIN_TURNS",
    "STRATEGY_DECAY_PE_THRESHOLD",
    "STRATEGY_REINFORCE_MIN_TURNS",
    "STRATEGY_REINFORCE_PE_THRESHOLD",
    "propose_case_archival",
    "propose_knowledge_archival",
    "propose_protocol_retirement",
    "propose_strategy_addition",
    "propose_strategy_decay",
    "propose_strategy_reinforce",
    "run_all_protocol_revision_rules",
]
