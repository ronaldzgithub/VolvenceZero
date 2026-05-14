"""Application-tier module-internal helpers (debt #9 wave 2).

Pure helper functions consumed by the Module classes in
``application.modules``: continuum-location and continuum-mixing
geometry, regime / domain / experience weight derivation,
case-memory hit scoring + ordering + pacing, response-mode and
response-target-position derivation, response-ordering plan
construction, prompt-residue summarisation, semantic-emotional
decision readout, expression-intent derivation, and the four
snapshot-count helpers used by the response-assembly module.
Also hosts the problem-pattern prototype constants
(``TASK_PRESSURE_PROTOTYPE`` etc.) that the case-memory and
response-assembly modules read.

Wave 2 of debt #9 split: these were lines 678-2237 of the
original monolithic ``runtime.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import TYPE_CHECKING, Any, Mapping

from volvence_zero.dual_track import DualTrackSnapshot
from volvence_zero.memory import MemoryEntry, MemorySnapshot, Track
from volvence_zero.runtime import RuntimeModule, RuntimePlaceholderValue, Snapshot, WiringLevel
from volvence_zero.social_cognition import (
    BeliefAboutOtherSnapshot,
    CommonGroundSnapshot,
    ConversationalRoleSnapshot,
    FeelingAboutOtherSnapshot,
    GroupSnapshot,
    IntentAboutOtherSnapshot,
    PreferenceAboutOtherSnapshot,
)
from volvence_zero.application.storage import (
    ApplicationCaseMemoryStore,
    ApplicationDomainKnowledgeStore,
    CaseMemoryRecord,
    DomainKnowledgeRecord,
)
from volvence_zero.application.retrieval_readout import (
    RetrievalControlReadoutInputs,
    RetrievalControlReadoutParameters,
    RetrievalReadoutCheckpoint,
    RetrievalControlReadoutStrategy,
)

if TYPE_CHECKING:
    from volvence_zero.prediction.error import PredictionErrorSnapshot
    from volvence_zero.regime import RegimeSnapshot
    from volvence_zero.temporal_types import TemporalAbstractionSnapshot


from volvence_zero.application.scoring_helpers import clamp01 as _clamp

# Wave 2 of debt #9 split: original ``runtime.py`` had the
# ``dedupe`` import inline between the dataclass section and the
# helper section; the line-range slicer lifted it into
# ``types.py``. We re-import it here so the helper functions that
# call ``_dedupe`` (e.g. ``_entry_risk_markers``,
# ``_case_hit_ordering``, etc.) keep resolving the name through
# their own module globals.
from volvence_zero.application.scoring_helpers import dedupe as _dedupe

# Star-import the typed surface from sibling ``types`` module.
# Helpers reference ~25 different dataclass / enum names; an
# explicit list would be brittle (every helper added in the
# future risks a NameError if its dataclass slips off the list).
# The dependency direction is one-way (``runtime_helpers`` imports
# from ``types``; nothing reverses) so this star import is safe.
from volvence_zero.application.types import *  # noqa: F401,F403

def _memory_text(memory_snapshot: MemorySnapshot | None) -> str:
    if memory_snapshot is None:
        return ""
    parts = [
        memory_snapshot.transient_summary,
        memory_snapshot.episodic_summary,
        memory_snapshot.durable_summary,
        *(entry.content for entry in memory_snapshot.retrieved_entries[:8]),
    ]
    return " ".join(part for part in parts if part)


def _continuum_profile(memory_snapshot: MemorySnapshot | None) -> Any | None:
    if memory_snapshot is None or memory_snapshot.cms_state is None:
        return None
    return memory_snapshot.cms_state.continuum_profile


def _continuum_location_from_band(
    *,
    profile: Any | None,
    band_id: str | None,
    fallback_source: str,
) -> ContinuumLocation | None:
    if profile is None or band_id is None:
        return None
    band = next((candidate for candidate in profile.bands if candidate.band_id == band_id), None)
    if band is None:
        return None
    reconstruction_edge = next(
        (edge for edge in profile.reconstruction_edges if edge.target_band_id == band.band_id),
        None,
    )
    return ContinuumLocation(
        profile_id=profile.profile_id,
        band_id=band.band_id,
        band_role=band.role,
        position=band.persistence_bias,
        update_frequency=band.update_frequency,
        reconstruction_source=reconstruction_edge.transfer_kind if reconstruction_edge is not None else fallback_source,
        description=(
            f"Continuum location band={band.band_id} role={band.role} position={band.persistence_bias:.2f} "
            f"frequency={band.update_frequency:.2f}."
        ),
    )


def _fallback_continuum_location(
    *,
    band_id: str,
    position: float,
    update_frequency: float,
    source: str,
) -> ContinuumLocation:
    role = {
        "online-fast": "fast-band",
        "session-medium": "session-band",
        "background-slow": "slow-band",
        "tower-readout": "readout",
    }.get(band_id, "application-case-band")
    return ContinuumLocation(
        profile_id="application-continuum-fallback",
        band_id=band_id,
        band_role=role,
        position=position,
        update_frequency=update_frequency,
        reconstruction_source=source,
        description=(
            f"Fallback continuum location band={band_id} position={position:.2f} "
            f"frequency={update_frequency:.2f}."
        ),
    )


def _continuum_location_for_entry(
    *,
    entry: MemoryEntry,
    memory_snapshot: MemorySnapshot | None,
) -> ContinuumLocation | None:
    profile = _continuum_profile(memory_snapshot)
    band_by_stratum = {
        "transient": "online-fast",
        "episodic": "session-medium",
        "durable": "background-slow",
        "derived": "tower-readout",
    }
    location = _continuum_location_from_band(
        profile=profile,
        band_id=band_by_stratum.get(entry.stratum),
        fallback_source="artifact-anchor",
    )
    if location is not None:
        return location
    fallback_band_id = band_by_stratum.get(entry.stratum, "tower-readout")
    fallback_position = {
        "transient": 0.18,
        "episodic": 0.46,
        "durable": 0.82,
        "derived": 0.58,
    }.get(entry.stratum, 0.5)
    fallback_frequency = {
        "transient": 1.0,
        "episodic": 0.5,
        "durable": 0.25,
        "derived": 0.2,
    }.get(entry.stratum, 0.33)
    return _fallback_continuum_location(
        band_id=fallback_band_id,
        position=fallback_position,
        update_frequency=fallback_frequency,
        source="artifact-anchor",
    )


def _continuum_location_from_record(record: CaseMemoryRecord) -> ContinuumLocation | None:
    if record.continuum_profile_id is None or record.continuum_band_id is None:
        fallback_band_id = (
            "background-slow"
            if record.delayed_signal_count >= 4 or record.confidence >= 0.78
            else "session-medium"
        )
        fallback_position = 0.82 if fallback_band_id == "background-slow" else 0.48
        fallback_frequency = 0.25 if fallback_band_id == "background-slow" else 0.5
        return _fallback_continuum_location(
            band_id=fallback_band_id,
            position=fallback_position,
            update_frequency=fallback_frequency,
            source="persisted-case-fallback",
        )
    return ContinuumLocation(
        profile_id=record.continuum_profile_id,
        band_id=record.continuum_band_id,
        band_role="application-case-band",
        position=record.continuum_position,
        update_frequency=record.continuum_update_frequency,
        reconstruction_source=record.reconstruction_source,
        description=(
            f"Persisted case continuum band={record.continuum_band_id} position={record.continuum_position:.2f} "
            f"frequency={record.continuum_update_frequency:.2f}."
        ),
    )


# W5 of ssot-cleanup-p0-p4: pure helpers extracted to
# ``application/scoring_helpers.py``. Legacy private aliases kept so
# any existing imports continue to work.
from volvence_zero.application.scoring_helpers import (
    clamp_signed as _clamp_signed,
    cosine_similarity as _cosine_similarity,
    semantic_embedding as _semantic_embedding,
    semantic_similarity as _semantic_similarity,
    semantic_tokens as _semantic_tokens,
    signed_centered as _signed_centered,
)


TASK_PRESSURE_PROTOTYPE = (
    "task pressure decision compare options concrete plan execution direct next step "
    "problem solving structure urgency organize"
)
SUPPORT_PRESENCE_PROTOTYPE = (
    "emotional support overwhelmed reassure steady warmth care calm gentle regulate "
    "feelings support-first stabilize"
)
REPAIR_PRESSURE_PROTOTYPE = (
    "repair trust conflict de-escalation apology rupture safety relational repair "
    "deescalate restore connection"
)
FAMILY_TRANSITION_PROTOTYPE = (
    "family transition separation co-parenting custody relationship breakdown household change "
    "emotionally intense transition"
)
PROFESSIONAL_PROCESS_PROTOTYPE = (
    "professional process legal procedural official guidance compliance sourced bounded advice "
    "jurisdiction sensitive process"
)
CAREER_DECISION_PROTOTYPE = (
    "career decision offer role transition trade-off professional path work change job choice"
)
JURISDICTION_CONTEXT_PROTOTYPE = (
    "local jurisdiction region local law court official local process local rules geography country city"
)
CHILD_IMPACT_PROTOTYPE = (
    "child family parenting custody co-parenting dependent vulnerable family impact"
)
DOMAIN_SENSITIVE_PROTOTYPE = (
    "legal professional official local law jurisdiction compliance procedural sensitive domain"
)
PROBLEM_PATTERN_PROTOTYPES: tuple[tuple[str, str], ...] = (
    ("family-transition-high-emotion", "emotionally intense family transition separation co-parenting overwhelm"),
    ("structured-decision-overwhelm", "too many options structure plan timeline organize compare next step"),
    ("relational-repair", "trust rupture conflict de-escalation apology relationship repair"),
)
USER_STATE_PROTOTYPES: tuple[tuple[str, str], ...] = (
    ("high-emotional-load", "overwhelmed anxious afraid distressed emotionally flooded tense"),
    ("needs-structure", "clarify organize structure sequence plan compare next step"),
    ("mixed-signal", "mixed ambivalent uncertain blended state both emotion and planning"),
)
PROBLEM_PATTERN_PROTOTYPES = PROBLEM_PATTERN_PROTOTYPES + (
    ("general-guidance", "general guidance mixed support and problem solving"),
)
RISK_BAND_ANCHORS: tuple[tuple[RiskBand, float], ...] = (
    (RiskBand.LOW, 0.18),
    (RiskBand.MEDIUM, 0.45),
    (RiskBand.HIGH, 0.72),
    (RiskBand.CRITICAL, 0.92),
)
KNOWLEDGE_DEPTH_ANCHORS: tuple[tuple[KnowledgeDepth, float], ...] = (
    (KnowledgeDepth.LIGHT, 0.22),
    (KnowledgeDepth.MEDIUM, 0.54),
    (KnowledgeDepth.DEEP, 0.84),
)
ANSWER_DEPTH_ANCHORS: tuple[tuple[str, float], ...] = (
    ("support-first", 0.28),
    ("standard", 0.56),
    ("high-level-only", 0.86),
)
PACING_ANCHORS: tuple[tuple[str, float], ...] = (
    ("structured", 0.20),
    ("slow", 0.52),
    ("gradual", 0.82),
)
OUTCOME_LABEL_ANCHORS: tuple[tuple[ExperienceOutcomeLabel, float], ...] = (
    (ExperienceOutcomeLabel.WORSENED, 0.18),
    (ExperienceOutcomeLabel.STABLE, 0.50),
    (ExperienceOutcomeLabel.IMPROVED, 0.82),
)


def _regime_bonus(regime_id: str | None, bonuses: Mapping[str, float]) -> float:
    """Legacy regime-id-keyed bonus lookup.

    DEPRECATED in W4 of ssot-cleanup-p0-p4. New code MUST read the
    domain bonus from ``ApplicationBrief.domain_bonus(domain)``;
    this helper is preserved only for sites the cutover has not
    yet reached, and a static contract test counts remaining
    callers to track the cleanup.
    """

    if regime_id is None:
        return 0.0
    return bonuses.get(regime_id, 0.0)


def _application_brief(regime_id: str | None):
    """Return the ``ApplicationBrief`` for ``regime_id``.

    Lazy import keeps ``vz-application`` from a hard dependency on
    ``vz-cognition`` at module load. The brief is the SSOT for
    regime-keyed application semantics; new code should read the
    brief instead of branching on regime id strings.
    """

    from volvence_zero.regime import application_brief_for_regime

    return application_brief_for_regime(regime_id)


from volvence_zero.application.scoring_helpers import (
    nearest_anchor_value as _nearest_anchor_value,
    ranked_labels as _ranked_labels,
    truncate_text as _truncate_text,
)


def _infer_track_weights(dual_track_snapshot: DualTrackSnapshot | None) -> tuple[float, float]:
    if dual_track_snapshot is None:
        return (0.5, 0.5)
    world_tension = max(0.05, dual_track_snapshot.world_track.tension_level)
    self_tension = max(0.05, dual_track_snapshot.self_track.tension_level)
    total = world_tension + self_tension
    return (_clamp(world_tension / total), _clamp(self_tension / total))


def _risk_band_from_state(
    *,
    dual_track_snapshot: DualTrackSnapshot | None,
    temporal_snapshot: "TemporalAbstractionSnapshot | None",
    regime_snapshot: "RegimeSnapshot | None",
) -> RiskBand:
    cross_track_tension = dual_track_snapshot.cross_track_tension if dual_track_snapshot is not None else 0.0
    switch_gate = (
        temporal_snapshot.controller_state.switch_gate
        if temporal_snapshot is not None
        else 0.0
    )
    regime_id = regime_snapshot.active_regime.regime_id if regime_snapshot is not None else None
    # W4 SSOT: risk-band severity bonus is published as a typed
    # ``risk_severity_bonus`` domain entry so consumers do not embed
    # regime-id dicts. The default 0.0 means: regimes that do not
    # care about risk-band severity contribute nothing here.
    brief = _application_brief(regime_id)
    severity = _clamp(
        cross_track_tension * 0.56
        + switch_gate * 0.26
        + brief.domain_bonus("risk_severity")
    )
    return _nearest_anchor_value(severity, RISK_BAND_ANCHORS)


def _knowledge_domains(
    *,
    dual_track_snapshot: DualTrackSnapshot | None,
    regime_id: str | None,
    world_weight: float,
    self_weight: float,
    abstract_action: str | None,
) -> tuple[str, ...]:
    # W4 of ssot-cleanup-p0-p4: read regime-specific domain bonuses
    # from ApplicationBrief instead of inline regime-id dicts.
    brief = _application_brief(regime_id)
    world_goals = " ".join(dual_track_snapshot.world_track.active_goals) if dual_track_snapshot is not None else ""
    self_goals = " ".join(dual_track_snapshot.self_track.active_goals) if dual_track_snapshot is not None else ""
    abstract_action_text = abstract_action or ""
    combined_text = " ".join(part for part in (world_goals, self_goals, abstract_action_text) if part)
    task_pull = _semantic_similarity(world_goals or combined_text, TASK_PRESSURE_PROTOTYPE)
    support_pull = _semantic_similarity(self_goals or combined_text, SUPPORT_PRESENCE_PROTOTYPE)
    repair_pull = _semantic_similarity(combined_text, REPAIR_PRESSURE_PROTOTYPE)
    family_transition_pull = _semantic_similarity(combined_text, FAMILY_TRANSITION_PROTOTYPE)
    professional_process_pull = _semantic_similarity(combined_text, PROFESSIONAL_PROCESS_PROTOTYPE)
    career_pull = _semantic_similarity(combined_text, CAREER_DECISION_PROTOTYPE)
    cross_track_tension = dual_track_snapshot.cross_track_tension if dual_track_snapshot is not None else abs(world_weight - self_weight)
    score_map = {
        "family_transition": _clamp(
            family_transition_pull * 0.60
            + support_pull * 0.08
            + repair_pull * 0.08
            + self_weight * 0.10
            + cross_track_tension * 0.08
        ),
        "professional_process": _clamp(
            professional_process_pull * 0.46
            + family_transition_pull * 0.18
            + task_pull * 0.10
            + world_weight * 0.14
            + brief.domain_bonus("professional_process")
        ),
        "career_decision": _clamp(
            career_pull * 0.60
            + task_pull * 0.14
            + world_weight * 0.16
            + brief.domain_bonus("career_decision")
        ),
        "structured_decision_support": _clamp(
            task_pull * 0.38
            + world_weight * 0.32
            + professional_process_pull * 0.08
            + career_pull * 0.08
            + brief.domain_bonus("structured_decision_support")
        ),
        "emotional_support_basics": _clamp(
            support_pull * 0.42
            + self_weight * 0.26
            + repair_pull * 0.10
            + family_transition_pull * 0.06
            + brief.domain_bonus("emotional_support_basics")
        ),
        "relational_repair": _clamp(
            repair_pull * 0.46
            + cross_track_tension * 0.18
            + support_pull * 0.10
            + self_weight * 0.08
            + brief.domain_bonus("relational_repair")
        ),
    }
    score_map["general_support_guidance"] = _clamp(0.18 + (1.0 - max(score_map.values(), default=0.0)) * 0.55)
    return _dedupe(_ranked_labels(score_map, max_count=3))


def _experience_domains(*, regime_id: str | None, self_weight: float, world_weight: float) -> tuple[str, ...]:
    # W4 SSOT: read experience-domain bonuses from ApplicationBrief.
    brief = _application_brief(regime_id)
    task_bias = _clamp(world_weight * 0.78 + brief.domain_bonus("structured_decision_patterns"))
    support_bias = _clamp(self_weight * 0.76 + brief.domain_bonus("stabilization_patterns"))
    repair_bias = _clamp(self_weight * 0.32 + brief.domain_bonus("repair_patterns"))
    score_map = {
        "structured_decision_patterns": task_bias,
        "repair_patterns": repair_bias,
        "stabilization_patterns": support_bias,
    }
    score_map["general_guidance_patterns"] = _clamp(0.20 + (1.0 - max(score_map.values(), default=0.0)) * 0.55)
    return _dedupe(_ranked_labels(score_map, max_count=2))


def _knowledge_weight(*, regime_id: str | None, world_weight: float, self_weight: float) -> float:
    # W4 SSOT: knowledge-vs-experience weight nudge is published by
    # the regime as ApplicationBrief.knowledge_weight_nudge.
    brief = _application_brief(regime_id)
    base = 0.5 + (world_weight - self_weight) * 0.35
    base += brief.knowledge_weight_nudge
    return _clamp(base)


def _regime_delayed_payoff_signal(
    *,
    regime_snapshot: "RegimeSnapshot | None",
    regime_id: str | None,
    abstract_action: str | None,
) -> tuple[float, float]:
    if regime_snapshot is None or regime_id is None:
        return (0.0, 0.0)
    payoffs = tuple(payoff for payoff in regime_snapshot.delayed_payoffs if payoff.regime_id == regime_id)
    if not payoffs:
        return (0.0, 0.0)
    matching_action_payoffs = tuple(
        payoff for payoff in payoffs if abstract_action is not None and payoff.abstract_action == abstract_action
    )
    selected = matching_action_payoffs or payoffs
    mean_payoff = sum(payoff.rolling_payoff for payoff in selected) / len(selected)
    sequence_payoff = next(
        (
            payoff.rolling_payoff
            for payoff in regime_snapshot.sequence_payoffs
            if payoff.regime_sequence and payoff.regime_sequence[-1] == regime_id
        ),
        mean_payoff,
    )
    return (_clamp(mean_payoff), _clamp(sequence_payoff))


def _rare_heavy_playbook_prior(
    *,
    rare_heavy_state: ApplicationRareHeavyState | None,
    regime_id: str | None,
) -> tuple[float, float]:
    if rare_heavy_state is None or not rare_heavy_state.distilled_playbook_rules:
        return (0.0, 0.0)
    matching_rules = tuple(
        rule
        for rule in rare_heavy_state.distilled_playbook_rules
        if regime_id is None or rule.recommended_regime in {None, regime_id}
    )
    if not matching_rules:
        return (0.0, 0.0)
    return (
        _clamp(sum(rule.knowledge_weight_hint for rule in matching_rules) / len(matching_rules)),
        _clamp(sum(rule.experience_weight_hint for rule in matching_rules) / len(matching_rules)),
    )


def _continuum_mixing_hints(memory_snapshot: MemorySnapshot | None) -> tuple[float, float, float, float, tuple[str, ...]]:
    profile = _continuum_profile(memory_snapshot)
    if profile is None or not profile.bands:
        return (0.5, 0.5, 0.0, 0.0, ())
    weighted_rows: list[tuple[Any, float]] = []
    for band in profile.bands:
        vector_strength = sum(abs(value) for value in band.vector) / max(len(band.vector), 1)
        pending_strength = sum(abs(value) for value in band.pending_signal) / max(len(band.pending_signal), 1)
        signal_strength = _clamp(
            vector_strength * 0.45
            + pending_strength * 0.18
            + band.retrieval_weight * 0.12
            + band.persistence_bias * 0.15
            + (1.0 - band.update_frequency) * 0.10
        )
        weighted_rows.append((band, max(signal_strength, 0.05)))
    total_weight = sum(weight for _, weight in weighted_rows)
    continuum_position = sum(band.persistence_bias * weight for band, weight in weighted_rows) / max(total_weight, 1e-6)
    continuum_frequency = sum(band.update_frequency * weight for band, weight in weighted_rows) / max(total_weight, 1e-6)
    slow_share = sum(
        weight
        for band, weight in weighted_rows
        if band.role in {"slow-band", "meta-init"}
    ) / max(total_weight, 1e-6)
    readout_share = sum(
        weight
        for band, weight in weighted_rows
        if band.role == "readout"
    ) / max(total_weight, 1e-6)
    edge_weight = max(len(profile.reconstruction_edges), 1)
    reconstruction_pressure = sum(edge.strength for edge in profile.reconstruction_edges) / edge_weight
    active_band_ids = tuple(
        band.band_id
        for band, weight in sorted(weighted_rows, key=lambda item: -item[1])[:3]
        if weight > 0.0
    )
    return (
        _clamp(continuum_position),
        _clamp(continuum_frequency),
        _clamp(slow_share),
        _clamp(readout_share * 0.65 + reconstruction_pressure * 0.35),
        active_band_ids,
    )


def _continuum_target_position(
    *,
    regime_id: str | None,
    world_weight: float,
    self_weight: float,
) -> float:
    # W4 of ssot-cleanup-p0-p4: read the regime-published target
    # position from ``ApplicationBrief.continuum_target_position``
    # instead of an if/elif chain on regime_id strings. New regimes
    # only need to land an ``ApplicationBrief`` in
    # ``volvence_zero.regime.templates``.
    brief = _application_brief(regime_id)
    if regime_id is not None and regime_id in {
        template.regime_id for template in _known_regime_ids()
    }:
        return brief.continuum_target_position
    return _clamp(0.35 + self_weight * 0.35 + (1.0 - world_weight) * 0.12)


def _known_regime_ids():
    """Cached lookup of known regime template ids.

    Used by W4 cutover sites that need to fall back to the freeform
    formula when ``regime_id`` is unknown / None. The brief always
    has a default ``continuum_target_position=0.5`` for unknown
    regimes; the freeform fallback is preserved here only because
    the historical behaviour for None / unknown blends ``self_weight``
    + ``world_weight`` continuously, and W4 chooses to keep that for
    backward-compat rather than collapse to 0.5.
    """

    from volvence_zero.regime.templates import REGIME_TEMPLATES

    return REGIME_TEMPLATES


def _continuum_source_bonus(source: str) -> float:
    return {
        "context-reset-reconstruction": 0.10,
        "slow-to-fast-reuse": 0.08,
        "associative-readout": 0.06,
        "reset-prior": 0.05,
        "persisted-case-fallback": 0.04,
        "rare-heavy-cluster": 0.04,
        "artifact-anchor": 0.03,
        "direct": 0.02,
    }.get(source, 0.01)


def _case_hit_playbook_score(
    *,
    hit: CaseEpisodeHit,
    regime_id: str | None,
    world_weight: float,
    self_weight: float,
) -> float:
    target_position = _continuum_target_position(
        regime_id=regime_id,
        world_weight=world_weight,
        self_weight=self_weight,
    )
    if hit.continuum_location is None:
        position_alignment = 0.5
        source_bonus = 0.0
        role_bonus = 0.0
    else:
        position_alignment = _clamp(1.0 - abs(hit.continuum_location.position - target_position))
        source_bonus = _continuum_source_bonus(hit.continuum_location.reconstruction_source)
        role_bonus = 0.0
        # W4 SSOT: read self-track / world-track focus from the brief.
        # ``self-track-leaning`` regimes (high support_focus or
        # repair_focus) prefer slow-band / meta-init case hits;
        # ``world-track-leaning`` regimes prefer session-band /
        # readout case hits. Adding a regime only requires updating
        # its ApplicationBrief.
        case_brief = _application_brief(regime_id)
        if (
            (case_brief.support_focus >= 0.6 or case_brief.repair_focus >= 0.4)
            and hit.continuum_location.band_role in {"slow-band", "meta-init"}
        ):
            role_bonus += 0.08
        if (
            case_brief.task_focus >= 0.40
            and hit.continuum_location.band_role in {"session-band", "readout"}
        ):
            role_bonus += 0.08
    ordering_bonus = 0.08 if _case_hit_ordering(hit) else 0.0
    return _clamp(
        hit.relevance_score * 0.42
        + hit.outcome.confidence * 0.22
        + position_alignment * 0.24
        + source_bonus
        + role_bonus
        + ordering_bonus
    )


def _retrieval_depth(
    *,
    regime_id: str | None,
    knowledge_weight: float,
    temporal_snapshot: "TemporalAbstractionSnapshot | None",
    continuum_position: float = 0.5,
    continuum_reconstruction_pressure: float = 0.0,
) -> KnowledgeDepth:
    switch_gate = temporal_snapshot.controller_state.switch_gate if temporal_snapshot is not None else 0.0
    # W4 SSOT: retrieval-depth nudge is published as a typed
    # ``retrieval_depth`` domain in ApplicationBrief.
    brief = _application_brief(regime_id)
    depth_score = _clamp(
        knowledge_weight * 0.58
        + switch_gate * 0.28
        + continuum_position * 0.10
        + continuum_reconstruction_pressure * 0.10
        + brief.domain_bonus("retrieval_depth")
    )
    return _nearest_anchor_value(depth_score, KNOWLEDGE_DEPTH_ANCHORS)


# Knowledge-domain set constants (R8 + first-principles: closed lists
# named once, referenced everywhere). Mirrors the curated knowledge-domain
# vocabulary owned by ``ApplicationBrief.domain_affinity`` in
# ``vz-cognition.regime.contracts``; treat as a closed list and update
# both sides together when adding a new reviewed domain.

# Domains that require an external citation in their generated knowledge
# records (citation gate).
_CITATION_REQUIRED_DOMAINS: frozenset[str] = frozenset({
    "family_transition",
    "professional_process",
    "career_decision",
})

# Domains whose advice is jurisdiction-sensitive (legal / professional
# process). Used by both the citation gate and the source-type selection.
_JURISDICTION_SENSITIVE_DOMAINS: frozenset[str] = frozenset({
    "family_transition",
    "professional_process",
})

# Domains whose default ``KnowledgeSourceType`` is INTERNAL_GUIDE
# (versus the default REVIEWED_ARTICLE) — empathy / repair material
# curated in-house rather than externally cited.
_INTERNAL_GUIDE_DOMAINS: frozenset[str] = frozenset({
    "relational_repair",
    "emotional_support_basics",
})


def _requires_citation(knowledge_domains: tuple[str, ...]) -> bool:
    return any(domain in _CITATION_REQUIRED_DOMAINS for domain in knowledge_domains)


def _requires_jurisdiction(knowledge_domains: tuple[str, ...]) -> bool:
    return any(domain in _JURISDICTION_SENSITIVE_DOMAINS for domain in knowledge_domains)


def _has_jurisdiction_context(text: str) -> bool:
    return _semantic_similarity(text, JURISDICTION_CONTEXT_PROTOTYPE) >= 0.52


def _domain_summary(domain: str, *, regime_id: str | None) -> str:
    if domain == "family_transition":
        return (
            "Separate emotional stabilization from legal or procedural next steps, and keep any "
            "child-safety or jurisdiction-sensitive guidance explicitly bounded."
        )
    if domain == "professional_process":
        return (
            "Use sourced high-level process guidance first, and avoid definitive professional conclusions "
            "before local specifics are confirmed."
        )
    if domain == "career_decision":
        return (
            "Frame trade-offs explicitly, reduce ambiguity, and prefer the smallest next step over a full "
            "life-plan answer."
        )
    if domain == "structured_decision_support":
        return (
            "Prefer option framing, trade-off comparison, and one grounded next action instead of broad "
            "multi-branch advice."
        )
    if domain == "relational_repair":
        return (
            "Prioritize de-escalation, acknowledgement, and safety before moving into explanation or planning."
        )
    if domain == "emotional_support_basics":
        return (
            "Acknowledge the felt experience first, then add structure gradually so the response does not "
            "skip past distress."
        )
    return (
        "Keep the response grounded, bounded, and shaped by the current regime rather than defaulting to a "
        "generic information dump."
    )


def _domain_topic_tags(domain: str) -> tuple[str, ...]:
    tags = {
        "family_transition": ("family", "transition", "procedure"),
        "professional_process": ("professional", "process", "bounded-advice"),
        "career_decision": ("career", "tradeoff", "next-step"),
        "structured_decision_support": ("decision", "structure", "options"),
        "relational_repair": ("repair", "de-escalation", "safety"),
        "emotional_support_basics": ("support", "stabilization", "presence"),
        "general_support_guidance": ("support", "boundedness"),
    }
    return tags.get(domain, ("general",))


def _domain_source_type(domain: str) -> KnowledgeSourceType:
    if domain in _JURISDICTION_SENSITIVE_DOMAINS:
        return KnowledgeSourceType.OFFICIAL_GUIDE
    if domain in _INTERNAL_GUIDE_DOMAINS:
        return KnowledgeSourceType.INTERNAL_GUIDE
    return KnowledgeSourceType.REVIEWED_ARTICLE


def _domain_jurisdiction_tags(domain: str, *, jurisdiction_required: bool) -> tuple[str, ...]:
    if jurisdiction_required and domain in _JURISDICTION_SENSITIVE_DOMAINS:
        return ("local-law-sensitive",)
    return ("general",)


def _entry_problem_pattern(entry: MemoryEntry) -> str:
    best_label = "general-guidance"
    best_score = -1.0
    for label, prototype in PROBLEM_PATTERN_PROTOTYPES:
        score = _semantic_similarity(entry.content, prototype)
        if score > best_score:
            best_label = label
            best_score = score
    return best_label


def _entry_user_state_pattern(entry: MemoryEntry) -> str:
    best_label = "mixed-signal"
    best_score = -1.0
    for label, prototype in USER_STATE_PROTOTYPES:
        score = _semantic_similarity(entry.content, prototype)
        if score > best_score:
            best_label = label
            best_score = score
    return best_label


def _entry_risk_markers(
    *,
    entry: MemoryEntry,
    prediction_error: "PredictionErrorSnapshot | None",
    retrieval_policy: RetrievalPolicySnapshot,
) -> tuple[str, ...]:
    markers: list[str] = []
    if retrieval_policy.risk_band in {RiskBand.HIGH, RiskBand.CRITICAL}:
        markers.append(f"risk-{retrieval_policy.risk_band.value}")
    child_impact_score = _semantic_similarity(entry.content, CHILD_IMPACT_PROTOTYPE)
    domain_sensitive_score = _semantic_similarity(entry.content, DOMAIN_SENSITIVE_PROTOTYPE)
    if child_impact_score >= 0.52:
        markers.append("child-impact")
    if domain_sensitive_score >= 0.52:
        markers.append("domain-sensitive")
    if prediction_error is not None and prediction_error.error.relationship_error <= -0.4:
        markers.append("relationship-instability")
    return _dedupe(tuple(markers))


def _entry_outcome_summary(
    *,
    prediction_error: "PredictionErrorSnapshot | None",
    entry: MemoryEntry,
) -> CaseOutcomeSummary:
    if prediction_error is None:
        return CaseOutcomeSummary(
            outcome_label=ExperienceOutcomeLabel.UNKNOWN,
            delayed_signal_count=0,
            escalation_observed=False,
            repair_observed=False,
            confidence=0.45,
            description=f"No prediction-error evidence yet for entry {entry.entry_id}.",
        )
    reward = prediction_error.error.signed_reward
    reward_alignment = _clamp(0.5 + reward * 0.5)
    relationship_repair_score = _clamp(0.5 + prediction_error.error.relationship_error * 0.5)
    relationship_escalation_score = _clamp(0.5 + (-prediction_error.error.relationship_error) * 0.5)
    outcome_label = _nearest_anchor_value(reward_alignment, OUTCOME_LABEL_ANCHORS)
    escalation_observed = relationship_escalation_score >= 0.68
    repair_observed = relationship_repair_score >= 0.58
    return CaseOutcomeSummary(
        outcome_label=outcome_label,
        delayed_signal_count=4,
        escalation_observed=escalation_observed,
        repair_observed=repair_observed,
        confidence=_clamp(0.52 + abs(reward - prediction_error.error.relationship_error) * 0.18 + abs(reward) * 0.18),
        description=(
            f"Outcome derived from prediction_error reward={reward:.2f} "
            f"relationship_error={prediction_error.error.relationship_error:.2f}."
        ),
    )


def _case_entries(memory_snapshot: MemorySnapshot | None, *, retrieval_policy: RetrievalPolicySnapshot) -> tuple[MemoryEntry, ...]:
    if memory_snapshot is None:
        return ()
    preferred_tracks: tuple[str, ...]
    if retrieval_policy.self_weight > retrieval_policy.world_weight:
        preferred_tracks = ("self", "shared", "world")
    else:
        preferred_tracks = ("world", "shared", "self")

    def score(entry: MemoryEntry) -> tuple[float, float, int]:
        track_priority = preferred_tracks.index(entry.track.value) if entry.track.value in preferred_tracks else len(preferred_tracks)
        experience_domain_bonus = 0.1 if retrieval_policy.experience_domains else 0.0
        return (float(track_priority), -(entry.strength + experience_domain_bonus), -entry.last_accessed_ms)

    ranked = sorted(memory_snapshot.retrieved_entries, key=score)
    return tuple(ranked[:3])


def _playbook_template(
    *,
    problem_pattern: str,
    regime_id: str | None,
    world_weight: float,
    self_weight: float,
) -> tuple[tuple[str, ...], str, tuple[str, ...]]:
    if problem_pattern == "family-transition-high-emotion":
        return (
            ("stabilize", "split_axes", "smallest_next_step"),
            "gradual",
            ("procedure-dump-too-early", "definitive-legal-conclusion"),
        )
    if problem_pattern == "structured-decision-overwhelm":
        return (
            ("narrow_scope", "option_compare", "smallest_next_step"),
            "structured",
            ("multi-branch-overload",),
        )
    if problem_pattern == "relational-repair":
        return (
            ("acknowledge", "deescalate", "bounded_next_step"),
            "slow",
            ("defensive-argumentation",),
        )
    if (
        _application_brief(regime_id).decision_kind_hint == "structure-first"
        or world_weight >= self_weight
    ):
        return (
            ("clarify_goal", "structure_options", "commit_next_step"),
            "structured",
            ("premature-emotional-bypass",),
        )
    return (
        ("acknowledge", "stabilize", "smallest_next_step"),
        "support-first",
        ("over-directive-solutioning",),
    )


def _case_hit_ordering(hit: CaseEpisodeHit) -> tuple[str, ...]:
    generic_labels = {"maintain-current-regime", "cluster-guided-ordering"}
    ordering = tuple(
        step.action_label
        for step in hit.intervention_steps
        if step.action_label not in generic_labels
    )
    return _dedupe(ordering)


def _case_hit_pacing(
    *,
    hit: CaseEpisodeHit,
    world_weight: float,
    self_weight: float,
) -> str:
    pacing_score = _clamp(
        (1.0 if "risk-high" in hit.risk_markers or "risk-critical" in hit.risk_markers else 0.0) * 0.32
        + (1.0 if hit.outcome.escalation_observed else 0.0) * 0.26
        + self_weight * 0.24
        + (1.0 - world_weight) * 0.18
    )
    return _nearest_anchor_value(pacing_score, PACING_ANCHORS)


def _case_hit_avoid_patterns(hit: CaseEpisodeHit) -> tuple[str, ...]:
    avoid_patterns: list[str] = []
    if "domain-sensitive" in hit.risk_markers:
        avoid_patterns.append("definitive-legal-conclusion")
    if "child-impact" in hit.risk_markers:
        avoid_patterns.append("procedure-dump-too-early")
    if hit.outcome.escalation_observed:
        avoid_patterns.append("premature-emotional-bypass")
    if not avoid_patterns:
        avoid_patterns.append("over-directive-solutioning")
    return _dedupe(tuple(avoid_patterns))


def _response_mode(
    *,
    regime_id: str | None,
    boundary_policy_snapshot: BoundaryPolicySnapshot,
    retrieval_policy_snapshot: RetrievalPolicySnapshot | None = None,
) -> ResponseMode:
    decision = boundary_policy_snapshot.active_decision
    if decision.refer_out_required:
        return ResponseMode.REFER_OUT
    if decision.clarification_required:
        return ResponseMode.CLARIFY
    return _response_mode_without_boundary(
        regime_id=regime_id,
        retrieval_policy_snapshot=retrieval_policy_snapshot,
    )


def _response_mode_without_boundary(
    *,
    regime_id: str | None,
    retrieval_policy_snapshot: RetrievalPolicySnapshot | None = None,
) -> ResponseMode:
    if retrieval_policy_snapshot is not None:
        if retrieval_policy_snapshot.response_mode_hint == "refer-out":
            return ResponseMode.REFER_OUT
        if retrieval_policy_snapshot.response_mode_hint == "clarify":
            return ResponseMode.CLARIFY
        if retrieval_policy_snapshot.response_mode_hint == "structure":
            return ResponseMode.STRUCTURE
        if retrieval_policy_snapshot.response_mode_hint == "support":
            return ResponseMode.SUPPORT
    # W4 SSOT: task-focused regimes default to STRUCTURE response
    # mode; everyone else defaults to SUPPORT. The threshold matches
    # ``ApplicationBrief.task_focus`` for problem_solving (0.85) /
    # guided_exploration (0.40).
    if _application_brief(regime_id).task_focus >= 0.40:
        return ResponseMode.STRUCTURE
    return ResponseMode.SUPPORT


def _boundary_constraints_expression_relevant(
    *,
    boundary_policy_snapshot: BoundaryPolicySnapshot,
    retrieval_policy_snapshot: RetrievalPolicySnapshot | None,
) -> bool:
    """Gate whether boundary decisions should become user-visible expression constraints.

    BoundaryPolicy remains the owner of risk/citation decisions. ResponseAssembly only
    surfaces those decisions when retrieval is leaning strongly enough toward domain
    knowledge for a disclaimer or clarification phrase to be relevant to the current turn.
    """

    decision = boundary_policy_snapshot.active_decision
    if decision.refer_out_required:
        return True
    if not (decision.clarification_required or decision.citation_required or decision.required_disclaimers):
        return False
    if retrieval_policy_snapshot is None:
        return decision.citation_required
    if decision.citation_required:
        return retrieval_policy_snapshot.knowledge_weight >= 0.55
    return (
        decision.risk_band is not RiskBand.LOW
        and retrieval_policy_snapshot.knowledge_weight >= 0.62
    )


def _required_disclaimer_phrase(disclaimer: str) -> str | None:
    if disclaimer == "jurisdiction-variance":
        return "Local rules and procedures can vary by jurisdiction."
    if disclaimer == "clarify-before-concluding":
        return "I may need one missing local detail before going further."
    if disclaimer == "professional-handoff":
        return "If this has real-world consequences, appropriate professional follow-up would be the safest next step."
    return None


def _response_control_scale(
    *,
    temporal_snapshot: "TemporalAbstractionSnapshot | None",
    ordering_plan: tuple[str, ...],
    response_mode: ResponseMode,
    boundary_policy_snapshot: BoundaryPolicySnapshot,
    retrieval_policy_snapshot: RetrievalPolicySnapshot | None = None,
) -> float:
    if temporal_snapshot is None or not temporal_snapshot.controller_state.code:
        return 0.0
    gate = temporal_snapshot.controller_state.switch_gate
    boundary_bonus = 0.05 if boundary_policy_snapshot.active_decision.refer_out_required else 0.03
    ordering_bonus = 0.03 if ordering_plan else 0.0
    mode_bonus = 0.03 if response_mode in {ResponseMode.CLARIFY, ResponseMode.STRUCTURE} else 0.0
    retrieval_bonus = (
        retrieval_policy_snapshot.answer_depth_bias * 0.04
        + retrieval_policy_snapshot.clarification_bias * 0.03
        if retrieval_policy_snapshot is not None
        else 0.0
    )
    return max(0.08, min(0.28, 0.06 + gate * 0.16 + boundary_bonus + ordering_bonus + mode_bonus + retrieval_bonus))


def _response_target_position(
    *,
    regime_id: str | None,
    response_mode: ResponseMode,
    boundary_policy_snapshot: BoundaryPolicySnapshot,
    case_memory_snapshot: CaseMemorySnapshot | None,
    strategy_playbook_snapshot: StrategyPlaybookSnapshot | None,
) -> float:
    if response_mode is ResponseMode.REFER_OUT:
        return 0.88
    # W4 SSOT: read regime-published continuum target from
    # ApplicationBrief instead of inline regime-id chain.
    brief = _application_brief(regime_id)
    if response_mode is ResponseMode.CLARIFY:
        # Repair- and support-leaning regimes need higher continuum
        # targets even on clarify turns; thresholded on support_focus
        # / repair_focus rather than on regime-id strings.
        if brief.repair_focus >= 0.4 or brief.support_focus >= 0.6:
            return 0.74
        case_position = case_memory_snapshot.mean_continuum_position if case_memory_snapshot is not None else 0.0
        if case_position >= 0.68:
            return 0.74
        return 0.58
    # problem_solving has a special "playbook-anchored" calculation;
    # we keep that branch but key it on the typed decision_kind_hint
    # ("structure-first") rather than the regime_id string.
    if brief.decision_kind_hint == "structure-first":
        playbook_position = (
            max((rule.mean_continuum_position for rule in strategy_playbook_snapshot.matched_rules), default=0.0)
            if strategy_playbook_snapshot is not None
            else 0.0
        )
        return _clamp(max(0.36, min(0.52, playbook_position or 0.42)))
    if regime_id is not None and any(
        template.regime_id == regime_id for template in _known_regime_ids()
    ):
        return brief.continuum_target_position
    return 0.5


def _response_ordering_plan(
    *,
    regime_id: str | None,
    response_mode: ResponseMode,
    boundary_policy_snapshot: BoundaryPolicySnapshot,
    case_memory_snapshot: CaseMemorySnapshot | None,
    strategy_playbook_snapshot: StrategyPlaybookSnapshot | None,
    retrieval_policy_snapshot: RetrievalPolicySnapshot | None = None,
) -> tuple[tuple[str, ...], float, str]:
    target_position = (
        retrieval_policy_snapshot.continuum_target_position_hint
        if retrieval_policy_snapshot is not None
        else _response_target_position(
            regime_id=regime_id,
            response_mode=response_mode,
            boundary_policy_snapshot=boundary_policy_snapshot,
            case_memory_snapshot=case_memory_snapshot,
            strategy_playbook_snapshot=strategy_playbook_snapshot,
        )
    )
    playbook_ordering = (
        strategy_playbook_snapshot.matched_rules[0].recommended_ordering
        if strategy_playbook_snapshot is not None and strategy_playbook_snapshot.matched_rules
        else ()
    )
    retrieval_ordering = retrieval_policy_snapshot.ordering_bias if retrieval_policy_snapshot is not None else ()
    playbook_support_first = bool(playbook_ordering) and playbook_ordering[0] in {"stabilize", "acknowledge", "deescalate"}
    playbook_mean_position = (
        sum(rule.mean_continuum_position for rule in strategy_playbook_snapshot.matched_rules)
        / max(len(strategy_playbook_snapshot.matched_rules), 1)
        if strategy_playbook_snapshot is not None and strategy_playbook_snapshot.matched_rules
        else 0.0
    )
    if (
        response_mode is not ResponseMode.REFER_OUT
        and playbook_mean_position > 0.0
        and playbook_mean_position < 0.52
        and playbook_ordering
        and not playbook_support_first
    ):
        target_position = min(target_position, 0.58)
        prefix = ("clarify_goal",)
        driver = "continuum-clarify-first"
        fallback_suffix = ("smallest_next_step",)
        return _dedupe(prefix + retrieval_ordering + playbook_ordering + fallback_suffix), target_position, driver
    if response_mode is ResponseMode.REFER_OUT:
        prefix = ("stabilize", "bounded_handoff")
        driver = "continuum-refer-out"
    elif response_mode is ResponseMode.CLARIFY:
        if playbook_support_first or target_position >= 0.66:
            prefix = ("stabilize", "clarify_goal")
            driver = "continuum-support-clarify"
        else:
            prefix = ("clarify_goal",)
            driver = "continuum-clarify-first"
    elif target_position >= 0.66:
        prefix = ("stabilize", "acknowledge")
        driver = (
            retrieval_policy_snapshot.ordering_driver_hint
            if retrieval_policy_snapshot is not None and retrieval_policy_snapshot.ordering_driver_hint != "retrieval-fallback"
            else "continuum-support-first"
        )
    elif target_position >= 0.52:
        prefix = ("clarify_goal", "split_axes")
        driver = (
            retrieval_policy_snapshot.ordering_driver_hint
            if retrieval_policy_snapshot is not None and retrieval_policy_snapshot.ordering_driver_hint != "retrieval-fallback"
            else "continuum-guided-clarify"
        )
    else:
        prefix = ("structure_options", "smallest_next_step")
        driver = (
            retrieval_policy_snapshot.ordering_driver_hint
            if retrieval_policy_snapshot is not None and retrieval_policy_snapshot.ordering_driver_hint != "retrieval-fallback"
            else "continuum-structure-first"
        )
    fallback_suffix = {
        "continuum-refer-out": ("smallest_next_step",),
        "continuum-support-clarify": ("smallest_next_step",),
        "continuum-clarify-first": ("smallest_next_step",),
        "continuum-support-first": ("smallest_next_step",),
        "continuum-guided-clarify": ("smallest_next_step",),
        "continuum-structure-first": ("commit_next_step",),
    }.get(driver, ("smallest_next_step",))
    ordering_plan = _dedupe(prefix + retrieval_ordering + playbook_ordering + fallback_suffix)
    return ordering_plan, target_position, driver


def _prompt_residue_summary(
    *,
    regime_name: str,
    regime_switched: bool,
    memory_snapshot: MemorySnapshot | None,
    reflection_snapshot: object | None,
    temporal_snapshot: "TemporalAbstractionSnapshot | None",
) -> tuple[str, float]:
    from volvence_zero.reflection import ReflectionSnapshot

    residue_parts = [f"Current mode: {regime_name}."]
    if regime_switched:
        residue_parts.append("The interaction frame has shifted, so acknowledge that naturally.")
    if memory_snapshot is not None and memory_snapshot.retrieved_entries:
        residue_parts.append(
            "Carry forward continuity from prior context: "
            + _truncate_text(memory_snapshot.retrieved_entries[0].content, max_chars=88)
        )
    if isinstance(reflection_snapshot, ReflectionSnapshot) and reflection_snapshot.lessons_extracted:
        residue_parts.append(f"Recent lesson: {next(iter(reflection_snapshot.lessons_extracted))}.")
    if isinstance(reflection_snapshot, ReflectionSnapshot) and reflection_snapshot.tensions_identified:
        residue_parts.append(f"Watch this tension: {next(iter(reflection_snapshot.tensions_identified))}.")
    if temporal_snapshot is not None:
        residue_parts.append(
            "Internal control frame: "
            + _truncate_text(temporal_snapshot.description, max_chars=96)
        )
    summary = " ".join(residue_parts)
    prompt_signal_count = len(residue_parts)
    explicit_signal_count = 5
    return summary, _clamp(prompt_signal_count / max(prompt_signal_count + explicit_signal_count, 1))


def _semantic_emotional_decision_readout(
    *,
    semantic_snapshots: Mapping[str, Snapshot[Any]],
    retrieval_policy_snapshot: RetrievalPolicySnapshot | None,
    ordering_plan: tuple[str, ...],
    response_mode: ResponseMode,
) -> tuple[float, str]:
    from volvence_zero.semantic_state import (
        BoundaryConsentSnapshot,
        GoalValueSnapshot,
        RelationshipStateSnapshot,
        UserModelSnapshot,
    )

    relationship_value = semantic_snapshots.get("relationship_state")
    relationship_snapshot = relationship_value.value if relationship_value is not None else None
    relationship_pressure = 0.0
    if isinstance(relationship_snapshot, RelationshipStateSnapshot):
        relationship_pressure = _clamp(
            relationship_snapshot.stabilization_need * 0.38
            + relationship_snapshot.emotional_load * 0.26
            + relationship_snapshot.repair_need * 0.20
            + relationship_snapshot.attunement_gap * 0.16
        )

    goal_value = semantic_snapshots.get("goal_value")
    goal_snapshot = goal_value.value if goal_value is not None else None
    goal_pressure = 0.0
    if isinstance(goal_snapshot, GoalValueSnapshot):
        goal_pressure = _clamp(
            goal_snapshot.value_conflict * 0.32
            + goal_snapshot.reversibility_need * 0.26
            + goal_snapshot.goal_shift_pressure * 0.22
            + (1.0 - goal_snapshot.decision_readiness) * 0.20
        )

    boundary_value = semantic_snapshots.get("boundary_consent")
    boundary_snapshot = boundary_value.value if boundary_value is not None else None
    boundary_pressure = 0.0
    if isinstance(boundary_snapshot, BoundaryConsentSnapshot):
        boundary_pressure = _clamp(
            boundary_snapshot.autonomy_risk * 0.42
            + boundary_snapshot.overreach_risk * 0.34
            + boundary_snapshot.professional_scope_pressure * 0.14
            + (1.0 - boundary_snapshot.consent_clarity) * 0.10
        )

    user_value = semantic_snapshots.get("user_model")
    user_snapshot = user_value.value if user_value is not None else None
    user_pressure = 0.0
    if isinstance(user_snapshot, UserModelSnapshot):
        user_pressure = _clamp(
            user_snapshot.overwhelm_pattern_strength * 0.54
            + (1.0 - user_snapshot.stability_score) * 0.18
            + (0.18 if user_snapshot.preferred_support_pacing == "support-first" else 0.0)
            + (0.10 if user_snapshot.decision_style == "values-first" else 0.0)
        )

    active_domains = set(retrieval_policy_snapshot.knowledge_domains if retrieval_policy_snapshot is not None else ())
    support_domain_active = bool(active_domains & {"emotional_support_basics", "relational_repair"})
    decision_domain_active = bool(
        active_domains
        & {
            "structured_decision_support",
            "career_decision",
            "family_transition",
            "professional_process",
        }
    )
    domain_pressure = 0.0
    if support_domain_active and decision_domain_active:
        domain_pressure = 0.82
    elif support_domain_active or decision_domain_active:
        domain_pressure = 0.42

    ordering_pressure = 0.0
    if "stabilize" in ordering_plan and (
        "clarify_goal" in ordering_plan
        or "split_axes" in ordering_plan
        or "smallest_next_step" in ordering_plan
    ):
        ordering_pressure = 0.68
    if response_mode is ResponseMode.CLARIFY and support_domain_active:
        ordering_pressure = max(ordering_pressure, 0.52)

    pressure = _clamp(
        relationship_pressure * 0.18
        + goal_pressure * 0.18
        + boundary_pressure * 0.08
        + user_pressure * 0.06
        + domain_pressure * 0.34
        + ordering_pressure * 0.16
    )

    if pressure < 0.24:
        return pressure, ""
    if boundary_pressure >= 0.42:
        return pressure, "hold_boundary_while_supporting"
    if relationship_pressure >= 0.46:
        return pressure, "repair_then_reframe"
    if goal_pressure >= 0.36 and decision_domain_active:
        return pressure, "clarify_values_then_options"
    if "smallest_next_step" in ordering_plan:
        return pressure, "small_reversible_next_step"
    return pressure, "stabilize_before_deciding"


def _response_expression_intent(
    *,
    regime_id: str,
    response_mode: ResponseMode,
    temporal_snapshot: "TemporalAbstractionSnapshot | None",
    semantic_control_signal: float,
    support_before_decision_pressure: float = 0.0,
) -> tuple[str, tuple[str, ...]]:
    """Pick a structured ``expression_intent`` for the active turn.

    History note (lifeform-bench feedback): every turn used to collapse to
    ``judgment-process`` because the previous heuristic flipped to
    judgment-process whenever ANY focus cue was present. That made the
    six product regimes feel identical at the expression layer (R14
    violation: regimes were not actually distinct identities, just labels).

    The new mapping:

    * ``response_mode`` overrides — REFER-OUT and CLARIFY are surface-level
      verdicts that must reach the renderer regardless of regime.
    * ``guided_exploration`` keeps ``judgment-process`` because it is the
      regime whose product promise IS to make reasoning visible. We also
      retain the focus cues that downstream prompts depend on.
    * Other regimes map to their natural turn shape: support-first /
      repair-first / structure-first / warmth-first / direct-answer.
    * Strong temporal-control or semantic-control signals can lift any
      regime into ``judgment-process`` — that is the explicit signal that
      "we are doing something deliberately different from autopilot".
    """
    if response_mode is ResponseMode.REFER_OUT:
        return "refer-out", ()
    # Strong control signals override regime defaults — the system is
    # deliberately deviating from autopilot, the user should see why.
    strong_temporal = bool(
        temporal_snapshot is not None
        and temporal_snapshot.active_abstract_action
        and getattr(temporal_snapshot, "switch_active", False)
    )
    strong_semantic = semantic_control_signal >= 0.32

    # W4 of ssot-cleanup-p0-p4: read regime defaults from the
    # ApplicationBrief instead of branching on regime_id strings.
    brief = _application_brief(regime_id)
    decision_kind_hint = brief.decision_kind_hint
    support_decision_threshold = brief.support_decision_threshold

    # W4 SSOT: a regime is "support-decision-shaped" iff its
    # ``ApplicationBrief.support_decision_threshold`` was lowered
    # below the default 0.44 (guided_exploration: 0.36) OR it has
    # task_focus >= 0.40 (problem_solving) OR support_focus >= 0.6
    # (emotional_support). The threshold itself is read from the
    # brief above so the gate stays a single source of truth.
    support_decision_regime = (
        brief.support_decision_threshold < 0.44
        or brief.task_focus >= 0.40
        or brief.support_focus >= 0.6
    )
    support_decision_active = (
        support_decision_regime
        and support_before_decision_pressure >= support_decision_threshold
    )
    if response_mode is ResponseMode.CLARIFY:
        if support_decision_active:
            return "support-before-decision", ()
        return "clarify-first", ()

    if support_decision_active and brief.task_focus >= 0.40:
        # guided_exploration / problem_solving both have task_focus>=0.40
        return "support-before-decision", ()
    if (
        support_decision_active
        and brief.support_focus >= 0.6
        and strong_temporal
    ):
        # emotional_support's brief publishes support_focus=0.85
        return "support-before-decision", ()

    if decision_kind_hint == "judgment-process" or strong_temporal or strong_semantic:
        focus: list[str] = []
        if decision_kind_hint == "judgment-process":
            focus.extend(
                [
                    "what cue in the user's message is driving the reply",
                    "what need the system is inferring right now",
                    "how the next sentence should adjust",
                ]
            )
        if strong_temporal and temporal_snapshot is not None:
            focus.append("how the active control frame changes the response")
        if strong_semantic:
            focus.append("which public semantic state should shape the answer")
        # If we got here only because of overrides without any focus content,
        # still produce a minimal cue so the renderer has something to show.
        if not focus:
            focus.append("why this turn deviates from the regime's default shape")
        return "judgment-process", _dedupe(tuple(focus))

    # Map the brief's decision_kind_hint directly to the typed kind label.
    # New regimes only need to land an ApplicationBrief; this if/else is
    # intentionally short and only covers the known canonical kinds so a
    # regime publishing an unknown hint falls through to "direct-answer"
    # instead of accidentally hijacking another renderer.
    if decision_kind_hint == "support-first":
        return "support-first", ()
    if decision_kind_hint == "repair-first":
        return "repair-first", ()
    if decision_kind_hint == "structure-first":
        return "structure-first", ()
    if decision_kind_hint == "warmth-first":
        return "warmth-first", ()
    if decision_kind_hint == "direct-answer":
        return "direct-answer", ()

    # Unknown / missing brief falls through to a neutral shape rather than
    # secretly going judgment-process again.
    return "direct-answer", ()


def _tom_snapshot_counts(
    snapshots: Mapping[str, Snapshot[Any]],
) -> tuple[tuple[str, int], ...]:
    counts: list[tuple[str, int]] = []
    for slot_name, snapshot_type in (
        ("belief_about_other", BeliefAboutOtherSnapshot),
        ("intent_about_other", IntentAboutOtherSnapshot),
        ("feeling_about_other", FeelingAboutOtherSnapshot),
        ("preference_about_other", PreferenceAboutOtherSnapshot),
    ):
        snapshot = snapshots.get(slot_name)
        value = snapshot.value if snapshot is not None else None
        if isinstance(value, snapshot_type):
            counts.append((slot_name, len(value.records)))
    return tuple(counts)


def _role_snapshot_counts(
    snapshots: Mapping[str, Snapshot[Any]],
) -> tuple[tuple[str, int], ...]:
    snapshot = snapshots.get("conversational_role")
    value = snapshot.value if snapshot is not None else None
    if isinstance(value, ConversationalRoleSnapshot):
        return (("conversational_role", len(value.active_predictions)),)
    return ()


def _common_ground_snapshot_counts(
    snapshots: Mapping[str, Snapshot[Any]],
) -> tuple[tuple[str, int], ...]:
    snapshot = snapshots.get("common_ground")
    value = snapshot.value if snapshot is not None else None
    if isinstance(value, CommonGroundSnapshot):
        return (("common_ground", len(value.dyad_atoms) + len(value.group_atoms)),)
    return ()


def _group_snapshot_counts(
    snapshots: Mapping[str, Snapshot[Any]],
) -> tuple[tuple[str, int], ...]:
    snapshot = snapshots.get("groups")
    value = snapshot.value if snapshot is not None else None
    if isinstance(value, GroupSnapshot):
        return (
            ("groups", len(value.groups)),
            ("group_joint_commitments", len(value.joint_commitments)),
        )
    return ()


def _response_speech_plan(
    *,
    expression_intent: str,
    response_mode: ResponseMode,
    regime_name: str,
    judgment_focus: tuple[str, ...],
    clarification_required: bool,
) -> ResponseSpeechPlan:
    """Render a ``ResponseSpeechPlan`` per ``expression_intent``.

    Each intent gets a distinct ``cue`` / ``inferred_need`` /
    ``response_adjustment`` script so downstream renderers (kernel
    ``_render_judgment_process_response`` and lifeform-expression
    ``GroundedResponseSynthesizer``) produce visibly different outputs per
    regime. This is the surface-level evidence for R14 (regime persistence).
    """
    question_budget = 1 if clarification_required else 0

    if expression_intent == "judgment-process":
        return ResponseSpeechPlan(
            cue="You are asking to see the judgment behind the reply, not just receive comfort.",
            inferred_need=(
                "You need evidence that this answer is being shaped by the current conversation rather than by a stock reassurance."
            ),
            response_adjustment=(
                "I should name that cue, state the need I infer, and answer from that basis in a compact way."
            ),
            question_budget=question_budget,
            required_steps=(
                "state_visible_cue",
                "state_inferred_need",
                "state_response_adjustment",
            ),
            description=(
                f"Speech plan for {regime_name}: judgment-process with {len(judgment_focus)} focus cues."
            ),
        )

    if expression_intent == "support-first":
        return ResponseSpeechPlan(
            cue="There is emotional weight here that has not been acknowledged yet.",
            inferred_need=(
                "You need to feel heard before any narrowing or solving — presence first, plan later."
            ),
            response_adjustment=(
                "I should slow the pace, name the weight, stay with it, and only then offer a small next step."
            ),
            question_budget=0,
            required_steps=(
                "acknowledge_pressure",
                "stay_present",
                "offer_small_step",
            ),
            description=f"Speech plan for {regime_name}: support-first.",
        )

    if expression_intent == "support-before-decision":
        return ResponseSpeechPlan(
            cue="This is both emotionally loaded and decision-shaped, so solving too fast would miss the real need.",
            inferred_need=(
                "You need the pressure acknowledged, the values clarified, and then one reversible next step rather than a verdict."
            ),
            response_adjustment=(
                "I should stabilize first, separate the feeling from the choice, name the active tradeoff, and keep the next move small."
            ),
            question_budget=1 if clarification_required else 0,
            required_steps=(
                "acknowledge_pressure",
                "clarify_values",
                "name_tradeoff",
                "offer_reversible_step",
            ),
            description=f"Speech plan for {regime_name}: support-before-decision.",
        )

    if expression_intent == "repair-first":
        return ResponseSpeechPlan(
            cue="Something just shifted in the relationship and we cannot move past it cleanly.",
            inferred_need=(
                "You need the rupture named honestly and a calm, concrete way back together — not a reset."
            ),
            response_adjustment=(
                "I should name what just happened, hold a steady tone, and offer one small repair move."
            ),
            question_budget=0,
            required_steps=(
                "name_rupture",
                "steady_tone",
                "offer_repair_move",
            ),
            description=f"Speech plan for {regime_name}: repair-first.",
        )

    if expression_intent == "structure-first":
        return ResponseSpeechPlan(
            cue="There is a tractable problem in front of us with at least one constraint we can name.",
            inferred_need=(
                "You need a structured next step, not vague reassurance and not a blank menu."
            ),
            response_adjustment=(
                "I should name the binding constraint, propose one concrete next step, and keep tone level."
            ),
            question_budget=question_budget,
            required_steps=(
                "name_binding_constraint",
                "propose_concrete_step",
            ),
            description=f"Speech plan for {regime_name}: structure-first.",
        )

    if expression_intent == "warmth-first":
        return ResponseSpeechPlan(
            cue="The interaction is still building — relational stance matters more than information density.",
            inferred_need=(
                "You need the reply to feel relational and curious, not transactional."
            ),
            response_adjustment=(
                "I should keep the tone warm, ask a single curious question if it fits, and avoid over-formality."
            ),
            question_budget=max(question_budget, 1),
            required_steps=(
                "warm_acknowledgement",
                "one_curious_question",
            ),
            description=f"Speech plan for {regime_name}: warmth-first.",
        )

    if expression_intent == "clarify-first":
        return ResponseSpeechPlan(
            cue="The request is currently ambiguous in a way that would make a confident answer wrong.",
            inferred_need=(
                "You need me to pick exactly one missing detail to clarify rather than guessing or enumerating options."
            ),
            response_adjustment=(
                "I should name the ambiguity, ask one focused clarifying question, and hold off on commitment."
            ),
            question_budget=max(question_budget, 1),
            required_steps=(
                "name_ambiguity",
                "ask_one_clarifying_question",
            ),
            description=f"Speech plan for {regime_name}: clarify-first.",
        )

    if expression_intent == "refer-out":
        return ResponseSpeechPlan(
            cue="The risk profile of this turn exceeds what a companion AI should treat definitively.",
            inferred_need=(
                "You need acknowledgement, then a clear pointer to a qualified human resource — not pseudo-expertise."
            ),
            response_adjustment=(
                "I should stay supportive and high-level, and explicitly suggest professional follow-up."
            ),
            question_budget=0,
            required_steps=(
                "acknowledge",
                "boundary_disclaimer",
                "refer_out",
            ),
            description=f"Speech plan for {regime_name}: refer-out.",
        )

    # ``direct-answer`` and any unknown intent fall through to the original
    # neutral plan so callers stay forward-compatible.
    return ResponseSpeechPlan(
        cue=f"Current response mode is {response_mode.value}.",
        inferred_need="Answer the latest user message directly within the current regime.",
        response_adjustment="Keep the reply compact and aligned with the response ordering plan.",
        question_budget=question_budget,
        required_steps=("answer_directly",),
        description=f"Speech plan for {regime_name}: direct answer.",
    )


# Wave 2 of debt #9: explicit ``__all__`` so ``from runtime_helpers
# import *`` picks up the leading-underscore helpers (``_continuum_*``,
# ``_case_*``, ``_response_*``, etc.) that the application Module
# classes call. Without this, Python's default star-import elides any
# name starting with ``_`` and the modules would fail at first call
# with NameError. We rebuild the list dynamically from module globals
# so adding a new helper does not require maintaining a parallel list.
__all__ = [
    _name
    for _name in list(globals())
    if not _name.startswith("__")
    and _name
    not in {
        # Imported / re-imported names that should NOT leak through
        # star-import to the modules. The modules import their own
        # versions explicitly.
        "annotations",
        "dataclass",
        "Enum",
        "math",
        "TYPE_CHECKING",
        "Any",
        "Mapping",
        "DualTrackSnapshot",
        "MemoryEntry",
        "MemorySnapshot",
        "Track",
        "RuntimeModule",
        "RuntimePlaceholderValue",
        "Snapshot",
        "WiringLevel",
        "BeliefAboutOtherSnapshot",
        "CommonGroundSnapshot",
        "ConversationalRoleSnapshot",
        "FeelingAboutOtherSnapshot",
        "GroupSnapshot",
        "IntentAboutOtherSnapshot",
        "PreferenceAboutOtherSnapshot",
        "ApplicationCaseMemoryStore",
        "ApplicationDomainKnowledgeStore",
        "CaseMemoryRecord",
        "DomainKnowledgeRecord",
        "RetrievalControlReadoutInputs",
        "RetrievalControlReadoutParameters",
        "RetrievalReadoutCheckpoint",
        "RetrievalControlReadoutStrategy",
        "_clamp",
    }
]
