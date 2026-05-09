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
    dependencies = ("case_memory", "regime", "dual_track")
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
        from volvence_zero.regime import RegimeSnapshot

        case_memory_snapshot = upstream["case_memory"].value
        regime_snapshot = upstream["regime"].value
        dual_track_snapshot = upstream["dual_track"].value
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
        return self.publish(
            StrategyPlaybookSnapshot(
                matched_problem_patterns=case_memory_snapshot.active_problem_patterns,
                matched_rules=tuple(rules),
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

