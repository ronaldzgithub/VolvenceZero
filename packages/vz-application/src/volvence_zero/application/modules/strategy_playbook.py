"""StrategyPlaybookModule (debt #9 wave 2).

R5 application-tier owner: matches the active regime and
case-memory hits against playbook rules, surfaces matched
rules in ``StrategyPlaybookSnapshot``.

Wave 2 of debt #9 split: this was lines 3236-3402 of the
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

from volvence_zero.application.types import *  # noqa: F401,F403 -- typed surface
from volvence_zero.application.runtime_helpers import *  # noqa: F401,F403
from volvence_zero.application.rare_heavy_state import ApplicationRareHeavyState  # noqa: F401


class StrategyPlaybookModule(RuntimeModule[StrategyPlaybookSnapshot]):
    slot_name = "strategy_playbook"
    owner = "StrategyPlaybookModule"
    value_type = StrategyPlaybookSnapshot
    # Packet 5.1: declare active_mixture / protocol_phase as deps
    # so a kernel module reads them and the consumer audit gate
    # can pin a real ACTIVE-channel readiness signal. The reads
    # themselves are SHADOW-tolerant; per-consumer behavior shift
    # (e.g. weight rare_heavy rules by activation_weight) is a
    # follow-up packet.
    dependencies = (
        "case_memory",
        "regime",
        "dual_track",
        "active_mixture",
        "protocol_phase",
    )
    default_wiring_level = WiringLevel.ACTIVE

    def __init__(
        self,
        *,
        rare_heavy_state: ApplicationRareHeavyState | None = None,
        wiring_level: WiringLevel | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._rare_heavy_state = rare_heavy_state

    async def process(self, upstream: Mapping[str, Snapshot[Any]]) -> Snapshot[StrategyPlaybookSnapshot]:
        from volvence_zero.behavior_protocol import (
            ActiveMixtureSnapshot,
            ProtocolPhaseSnapshot,
        )
        from volvence_zero.regime import RegimeSnapshot

        case_memory_snapshot = upstream["case_memory"].value
        regime_snapshot = upstream["regime"].value
        dual_track_snapshot = upstream["dual_track"].value

        # Packet 5.1: read active_mixture / protocol_phase
        # SHADOW-tolerantly. ``protocol_phase_value`` reserved for
        # future phase-aware ranking.
        am_snap = upstream.get("active_mixture")
        pp_snap = upstream.get("protocol_phase")
        active_mixture_value = (
            am_snap.value
            if am_snap is not None and isinstance(am_snap.value, ActiveMixtureSnapshot)
            else None
        )
        _protocol_phase_value = (
            pp_snap.value
            if pp_snap is not None and isinstance(pp_snap.value, ProtocolPhaseSnapshot)
            else None
        )

        # ``protocol-online-learning-followups`` packet, sub-packet F1
        # (B2 closeout): build a ``compiled_rule_id → weight`` map from
        # the per-rule strategy weight table that
        # ``ProtocolRegistryModule`` publishes on ``active_mixture``.
        # The map is keyed on ``compiled_rule_id`` (namespaced format
        # ``protocol:{protocol_id}:playbook:{raw_rule_id}``) which is
        # byte-identical to the ``rule_id`` carried by
        # ``PlaybookRule`` instances pushed by the protocol compile
        # path. Empty / missing snapshot ⇒ empty map ⇒ falls back to
        # default weight 1.0 ⇒ stable sort preserves the original
        # order ⇒ pre-F1 byte-equivalence preserved.
        weight_by_compiled_rule_id: dict[str, float] = {}
        if active_mixture_value is not None:
            for entry in active_mixture_value.strategy_weights:
                if entry.compiled_rule_id:
                    weight_by_compiled_rule_id[entry.compiled_rule_id] = entry.weight

        if not isinstance(case_memory_snapshot, CaseMemorySnapshot):
            raise TypeError("case_memory must publish CaseMemorySnapshot.")
        if not isinstance(regime_snapshot, RegimeSnapshot):
            raise TypeError("regime must publish RegimeSnapshot.")
        if not isinstance(dual_track_snapshot, DualTrackSnapshot):
            raise TypeError("dual_track must publish DualTrackSnapshot.")
        world_weight, self_weight = _infer_track_weights(dual_track_snapshot)
        rules: list[PlaybookRule] = []
        active_band_ids: list[str] = []
        rare_heavy_rules = {
            rule.problem_pattern: rule
            for rule in (self._rare_heavy_state.distilled_playbook_rules if self._rare_heavy_state is not None else ())
        }
        for index, pattern in enumerate(case_memory_snapshot.active_problem_patterns, start=1):
            if pattern in rare_heavy_rules:
                rare_heavy_rule = rare_heavy_rules[pattern]
                if rare_heavy_rule.continuum_band_id is not None:
                    active_band_ids.append(rare_heavy_rule.continuum_band_id)
                rules.append(rare_heavy_rule)
                continue
            matching_hits = tuple(hit for hit in case_memory_snapshot.hits if hit.problem_pattern == pattern)
            ranked_hits = tuple(
                sorted(
                    matching_hits,
                    key=lambda hit: -_case_hit_playbook_score(
                        hit=hit,
                        regime_id=regime_snapshot.active_regime.regime_id,
                        world_weight=world_weight,
                        self_weight=self_weight,
                    ),
                )
            )
            source_hit = ranked_hits[0] if ranked_hits else None
            source_continuum_location = (
                source_hit.continuum_location if source_hit is not None else None
            )
            direct_ordering = next(
                (ordering for ordering in (_case_hit_ordering(hit) for hit in ranked_hits) if ordering),
                (),
            )
            if direct_ordering:
                if source_continuum_location is not None:
                    active_band_ids.append(source_continuum_location.band_id)
                rules.append(
                    PlaybookRule(
                        rule_id=f"playbook:case-derived:{pattern}:{index}",
                        problem_pattern=pattern,
                        recommended_regime=regime_snapshot.active_regime.regime_id,
                        recommended_ordering=direct_ordering,
                        recommended_pacing=_case_hit_pacing(
                            hit=source_hit,
                            world_weight=world_weight,
                            self_weight=self_weight,
                        ),
                        avoid_patterns=_case_hit_avoid_patterns(source_hit),
                        knowledge_weight_hint=_clamp(0.45 + world_weight * 0.30),
                        experience_weight_hint=_clamp(0.45 + self_weight * 0.30),
                        applicability_scope=(regime_snapshot.active_regime.regime_id, *source_hit.risk_markers[:2]),
                        confidence=_clamp(
                            max(source_hit.outcome.confidence, source_hit.relevance_score) * 0.7
                            + _case_hit_playbook_score(
                                hit=source_hit,
                                regime_id=regime_snapshot.active_regime.regime_id,
                                world_weight=world_weight,
                                self_weight=self_weight,
                            )
                            * 0.3
                        ),
                        continuum_band_id=(
                            source_continuum_location.band_id if source_continuum_location is not None else None
                        ),
                        mean_continuum_position=(
                            source_continuum_location.position if source_continuum_location is not None else 0.0
                        ),
                        description=(
                            f"Case-derived playbook rule for pattern={pattern} from "
                            f"{len(source_hit.intervention_steps)} intervention steps."
                        ),
                    )
                )
                continue
            ordering, pacing, avoid_patterns = _playbook_template(
                problem_pattern=pattern,
                regime_id=regime_snapshot.active_regime.regime_id,
                world_weight=world_weight,
                self_weight=self_weight,
            )
            if source_continuum_location is not None:
                active_band_ids.append(source_continuum_location.band_id)
            rules.append(
                PlaybookRule(
                    rule_id=f"playbook:{pattern}:{index}",
                    problem_pattern=pattern,
                    recommended_regime=regime_snapshot.active_regime.regime_id,
                    recommended_ordering=ordering,
                    recommended_pacing=pacing,
                    avoid_patterns=avoid_patterns,
                    knowledge_weight_hint=_clamp(0.45 + world_weight * 0.30),
                    experience_weight_hint=_clamp(0.45 + self_weight * 0.30),
                    applicability_scope=(regime_snapshot.active_regime.regime_id, *case_memory_snapshot.active_risk_markers[:2]),
                    confidence=_clamp(
                        0.45
                        + len(case_memory_snapshot.hits) * 0.06
                        + (
                            _case_hit_playbook_score(
                                hit=source_hit,
                                regime_id=regime_snapshot.active_regime.regime_id,
                                world_weight=world_weight,
                                self_weight=self_weight,
                            )
                            * 0.18
                            if source_hit is not None
                            else 0.0
                        )
                    ),
                    continuum_band_id=(
                        source_continuum_location.band_id if source_continuum_location is not None else None
                    ),
                    mean_continuum_position=(
                        source_continuum_location.position if source_continuum_location is not None else 0.0
                    ),
                    description=(
                        f"Playbook rule for pattern={pattern} aligned to regime={regime_snapshot.active_regime.regime_id} "
                        f"and cross_track_tension={dual_track_snapshot.cross_track_tension:.2f}."
                    ),
                )
            )
        # F1 ranking: stable-sort matched_rules by descending learnt
        # weight (default 1.0 for any rule not in the weight map).
        # Stable sort means ties (including the all-default-weight
        # case) preserve insertion order, so single-protocol /
        # uniform-weight scenarios stay byte-equivalent with the
        # pre-F1 output. Downstream
        # ``_response_ordering_plan`` reads ``matched_rules[0]
        # .recommended_ordering`` so a high-weight rule's ordering
        # naturally drives the plan.
        ranked_rules = tuple(
            sorted(
                rules,
                key=lambda r: -weight_by_compiled_rule_id.get(r.rule_id, 1.0),
            )
        )
        return self.publish(
            StrategyPlaybookSnapshot(
                matched_problem_patterns=case_memory_snapshot.active_problem_patterns,
                matched_rules=ranked_rules,
                continuum_profile_id=case_memory_snapshot.continuum_profile_id,
                active_band_ids=_dedupe(tuple(active_band_ids)),
                support_prior=_clamp(self_weight),
                task_prior=_clamp(world_weight),
                description=(
                    f"Strategy playbook produced {len(rules)} matched rules from "
                    f"{len(case_memory_snapshot.hits)} case hits."
                ),
            )
        )

    async def process_standalone(self, **kwargs: Any) -> Snapshot[StrategyPlaybookSnapshot]:
        raise NotImplementedError("StrategyPlaybookModule should be driven by CaseMemorySnapshot.")

