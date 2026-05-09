"""ExperienceFastPriorModule (debt #9 wave 2).

R5 application-tier owner: turns slow-loop experience
consolidation into a fast-loop prior surface that retrieval
policy and case memory consume on the next turn.

Wave 2 of debt #9 split: this was lines 2368-2577 of the
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


class ExperienceFastPriorModule(RuntimeModule[ExperienceFastPriorSnapshot]):
    slot_name = "experience_fast_prior"
    owner = "ExperienceFastPriorModule"
    value_type = ExperienceFastPriorSnapshot
    dependencies = ("experience_consolidation",)
    default_wiring_level = WiringLevel.SHADOW

    def publish_snapshot(
        self,
        *,
        experience_consolidation_snapshot: ExperienceConsolidationSnapshot | None = None,
    ) -> Snapshot[ExperienceFastPriorSnapshot]:
        consolidation = experience_consolidation_snapshot
        if consolidation is None:
            return self.publish(
                ExperienceFastPriorSnapshot(
                    regime_biases=(),
                    knowledge_weight_bias=0.0,
                    experience_weight_bias=0.0,
                    action_biases=(),
                    family_biases=(),
                    sequence_biases=(),
                    prior_strength=0.0,
                    source_attribution_ids=(),
                    source_sequence_ids=(),
                    description="Experience fast prior unavailable because experience_consolidation is absent.",
                )
            )

        attribution_ids: list[str] = []
        sequence_ids: list[str] = []
        regime_bias_map: dict[str, list[tuple[float, str]]] = {}
        action_bias_map: dict[str, list[tuple[float, str]]] = {}
        family_bias_map: dict[int, list[tuple[float, str]]] = {}
        sequence_bias_map: dict[tuple[tuple[str, ...], int], list[tuple[float, str]]] = {}
        mix_bias_values: list[float] = []

        for attribution in consolidation.delayed_outcome_ledger:
            attribution_ids.append(attribution.attribution_id)
            outcome_direction = _signed_centered(attribution.outcome_score)
            regime_bias = outcome_direction * _signed_centered(attribution.regime_alignment) * 0.20
            if attribution.regime_id is not None:
                regime_bias_map.setdefault(attribution.regime_id, []).append((regime_bias, attribution.attribution_id))
            action_bias = outcome_direction * _signed_centered(attribution.abstract_action_alignment) * 0.18
            if attribution.abstract_action is not None:
                action_bias_map.setdefault(attribution.abstract_action, []).append(
                    (action_bias, attribution.attribution_id)
                )
            family_bias = (
                outcome_direction * 0.10
                + _signed_centered(attribution.abstract_action_alignment) * 0.08
                + _signed_centered(attribution.regime_alignment) * 0.04
            )
            if attribution.action_family_version > 0:
                family_bias_map.setdefault(attribution.action_family_version, []).append(
                    (family_bias, attribution.attribution_id)
                )
            mix_alignment_direction = _signed_centered(attribution.retrieval_mix_alignment)
            retrieval_lean = attribution.experience_weight - attribution.knowledge_weight
            mix_bias_values.append(outcome_direction * mix_alignment_direction * retrieval_lean * 0.22)

        for sequence_payoff in consolidation.sequence_payoffs:
            sequence_ids.append(sequence_payoff.sequence_id)
            payoff_bias = (
                _signed_centered(sequence_payoff.rolling_payoff)
                * min(sequence_payoff.sample_count / 3.0, 1.0)
                * 0.18
            )
            sequence_key = (sequence_payoff.regime_sequence, sequence_payoff.action_family_version)
            sequence_bias_map.setdefault(sequence_key, []).append((payoff_bias, sequence_payoff.sequence_id))

        delayed_credit_summary = consolidation.delayed_credit_summary
        if delayed_credit_summary is not None:
            summary_id = delayed_credit_summary.summary_id
            attribution_ids.append(summary_id)
            summary_direction = _signed_centered(delayed_credit_summary.outcome_score)
            if delayed_credit_summary.regime_id is not None:
                regime_bias_map.setdefault(delayed_credit_summary.regime_id, []).append(
                    (
                        summary_direction * _signed_centered(delayed_credit_summary.regime_alignment) * 0.22,
                        summary_id,
                    )
                )
            if delayed_credit_summary.abstract_action is not None:
                action_bias_map.setdefault(delayed_credit_summary.abstract_action, []).append(
                    (
                        summary_direction * _signed_centered(delayed_credit_summary.abstract_action_alignment) * 0.20,
                        summary_id,
                    )
                )
            if delayed_credit_summary.action_family_version > 0:
                family_bias_map.setdefault(delayed_credit_summary.action_family_version, []).append(
                    (
                        summary_direction * 0.12
                        + _signed_centered(delayed_credit_summary.abstract_action_alignment) * 0.10
                        + _signed_centered(delayed_credit_summary.sequence_payoff) * 0.08,
                        summary_id,
                    )
                )
                sequence_key = (
                    ((delayed_credit_summary.regime_id,) if delayed_credit_summary.regime_id is not None else ()),
                    delayed_credit_summary.action_family_version,
                )
                sequence_bias_map.setdefault(sequence_key, []).append(
                    (_signed_centered(delayed_credit_summary.sequence_payoff) * 0.20, summary_id)
                )
            retrieval_lean = delayed_credit_summary.experience_weight - delayed_credit_summary.knowledge_weight
            mix_bias_values.append(
                summary_direction
                * _signed_centered(delayed_credit_summary.retrieval_mix_alignment)
                * retrieval_lean
                * 0.24
            )

        regime_biases = tuple(
            ExperienceFastPriorRegimeBias(
                regime_id=regime_id,
                bias=_clamp_signed(sum(value for value, _ in values) / len(values)),
                source_attribution_ids=tuple(source_id for _, source_id in values),
                description=(
                    f"Fast prior regime bias for {regime_id} derived from {len(values)} delayed attribution(s)."
                ),
            )
            for regime_id, values in sorted(regime_bias_map.items())
        )
        action_biases = tuple(
            ExperienceFastPriorActionBias(
                abstract_action=abstract_action,
                bias=_clamp_signed(sum(value for value, _ in values) / len(values)),
                source_attribution_ids=tuple(source_id for _, source_id in values),
                description=(
                    f"Fast prior action bias for {abstract_action} derived from {len(values)} delayed attribution(s)."
                ),
            )
            for abstract_action, values in sorted(action_bias_map.items())
        )
        family_biases = tuple(
            ExperienceFastPriorFamilyBias(
                action_family_version=family_version,
                continuation_bias=_clamp_signed(sum(value for value, _ in values) / len(values)),
                source_attribution_ids=tuple(source_id for _, source_id in values),
                description=(
                    f"Fast prior family bias for action_family_version={family_version} derived from "
                    f"{len(values)} delayed attribution(s)."
                ),
            )
            for family_version, values in sorted(family_bias_map.items())
        )
        sequence_biases = tuple(
            ExperienceFastPriorSequenceBias(
                regime_sequence=regime_sequence,
                action_family_version=family_version,
                payoff_bias=_clamp_signed(sum(value for value, _ in values) / len(values)),
                source_sequence_ids=tuple(source_id for _, source_id in values),
                description=(
                    f"Fast prior sequence bias for regime_sequence={regime_sequence} family_version={family_version} "
                    f"derived from {len(values)} sequence payoff(s)."
                ),
            )
            for (regime_sequence, family_version), values in sorted(sequence_bias_map.items())
        )
        experience_weight_bias = _clamp_signed(
            sum(mix_bias_values) / len(mix_bias_values) if mix_bias_values else 0.0
        )
        knowledge_weight_bias = _clamp_signed(-experience_weight_bias)
        prior_strength_terms = (
            tuple(abs(item.bias) for item in regime_biases)
            + tuple(abs(item.bias) for item in action_biases)
            + tuple(abs(item.continuation_bias) for item in family_biases)
            + tuple(abs(item.payoff_bias) for item in sequence_biases)
            + ((abs(experience_weight_bias) + abs(knowledge_weight_bias)) / 2.0,)
        )
        prior_strength = _clamp(
            sum(prior_strength_terms) / len(prior_strength_terms) if prior_strength_terms else 0.0
        )
        return self.publish(
            ExperienceFastPriorSnapshot(
                regime_biases=regime_biases,
                knowledge_weight_bias=knowledge_weight_bias,
                experience_weight_bias=experience_weight_bias,
                action_biases=action_biases,
                family_biases=family_biases,
                sequence_biases=sequence_biases,
                prior_strength=prior_strength,
                source_attribution_ids=tuple(attribution_ids),
                source_sequence_ids=tuple(sequence_ids),
                description=(
                    f"Experience fast prior published {len(regime_biases)} regime bias(es), "
                    f"{len(action_biases)} action bias(es), {len(family_biases)} family bias(es), and "
                    f"{len(sequence_biases)} sequence bias(es)."
                ),
            )
        )

    async def process(self, upstream: Mapping[str, Snapshot[Any]]) -> Snapshot[ExperienceFastPriorSnapshot]:
        consolidation_value = upstream["experience_consolidation"].value
        consolidation = (
            consolidation_value if isinstance(consolidation_value, ExperienceConsolidationSnapshot) else None
        )
        return self.publish_snapshot(experience_consolidation_snapshot=consolidation)

    async def process_standalone(
        self,
        *,
        experience_consolidation_snapshot: ExperienceConsolidationSnapshot | None = None,
        **kwargs: Any,
    ) -> Snapshot[ExperienceFastPriorSnapshot]:
        del kwargs
        return self.publish_snapshot(experience_consolidation_snapshot=experience_consolidation_snapshot)

