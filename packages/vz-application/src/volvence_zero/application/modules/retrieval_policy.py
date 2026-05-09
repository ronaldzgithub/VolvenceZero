"""RetrievalPolicyModule (debt #9 wave 2).

R5 application-tier owner: derives the retrieval policy
snapshot (knowledge / experience weights, intent description,
regime / abstract-action handles) from upstream prediction-
error / regime / temporal-abstraction / dual-track snapshots.

Wave 2 of debt #9 split: this was lines 2579-2816 of the
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


class RetrievalPolicyModule(RuntimeModule[RetrievalPolicySnapshot]):
    slot_name = "retrieval_policy"
    owner = "RetrievalPolicyModule"
    value_type = RetrievalPolicySnapshot
    dependencies = ("world_temporal", "self_temporal", "dual_track", "regime", "memory", "experience_fast_prior")
    default_wiring_level = WiringLevel.ACTIVE

    def __init__(
        self,
        *,
        rare_heavy_state: ApplicationRareHeavyState | None = None,
        wiring_level: WiringLevel | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._rare_heavy_state = rare_heavy_state

    async def process(self, upstream: Mapping[str, Snapshot[Any]]) -> Snapshot[RetrievalPolicySnapshot]:
        from volvence_zero.regime import RegimeSnapshot
        from volvence_zero.temporal_types import TemporalAbstractionSnapshot

        world_temporal_snapshot = upstream["world_temporal"].value
        self_temporal_snapshot = upstream["self_temporal"].value
        dual_track_snapshot = upstream["dual_track"].value
        regime_snapshot = upstream["regime"].value
        memory_snapshot = upstream["memory"].value
        experience_fast_prior_snapshot = upstream["experience_fast_prior"].value
        if not isinstance(world_temporal_snapshot, TemporalAbstractionSnapshot):
            if isinstance(world_temporal_snapshot, RuntimePlaceholderValue) and isinstance(
                self_temporal_snapshot, RuntimePlaceholderValue
            ):
                return self._publish_temporal_disabled_fallback()
            raise TypeError("world_temporal must publish TemporalAbstractionSnapshot.")
        if not isinstance(self_temporal_snapshot, TemporalAbstractionSnapshot):
            raise TypeError("self_temporal must publish TemporalAbstractionSnapshot.")
        if not isinstance(dual_track_snapshot, DualTrackSnapshot):
            raise TypeError("dual_track must publish DualTrackSnapshot.")
        if not isinstance(regime_snapshot, RegimeSnapshot):
            raise TypeError("regime must publish RegimeSnapshot.")
        experience_fast_prior = (
            experience_fast_prior_snapshot
            if isinstance(experience_fast_prior_snapshot, ExperienceFastPriorSnapshot)
            else None
        )
        memory_value = memory_snapshot if isinstance(memory_snapshot, MemorySnapshot) else None
        world_weight, self_weight = _infer_track_weights(dual_track_snapshot)
        regime_id = regime_snapshot.active_regime.regime_id
        active_temporal_snapshot = (
            world_temporal_snapshot if world_weight >= self_weight else self_temporal_snapshot
        )
        abstract_action = active_temporal_snapshot.active_abstract_action
        knowledge_domains = _knowledge_domains(
            dual_track_snapshot=dual_track_snapshot,
            regime_id=regime_id,
            world_weight=world_weight,
            self_weight=self_weight,
            abstract_action=abstract_action,
        )
        experience_domains = _experience_domains(
            regime_id=regime_id,
            self_weight=self_weight,
            world_weight=world_weight,
        )
        (
            continuum_position,
            continuum_frequency,
            continuum_slow_share,
            continuum_reconstruction_pressure,
            continuum_active_band_ids,
        ) = _continuum_mixing_hints(memory_value)
        if self._rare_heavy_state is not None and self._rare_heavy_state.domain_template_biases:
            boosted = sorted(
                self._rare_heavy_state.domain_template_biases,
                key=lambda item: -item[1],
            )
            for domain, weight in boosted[:2]:
                if weight >= 0.55 and domain not in knowledge_domains:
                    knowledge_domains = knowledge_domains + (domain,)
        delayed_payoff_signal, sequence_payoff_signal = _regime_delayed_payoff_signal(
            regime_snapshot=regime_snapshot,
            regime_id=regime_id,
            abstract_action=abstract_action,
        )
        playbook_knowledge_hint, playbook_experience_hint = _rare_heavy_playbook_prior(
            rare_heavy_state=self._rare_heavy_state,
            regime_id=regime_id,
        )
        regime_fast_prior_bias = 0.0
        action_fast_prior_bias = 0.0
        family_fast_prior_bias = 0.0
        if experience_fast_prior is not None:
            regime_fast_prior_bias = next(
                (item.bias for item in experience_fast_prior.regime_biases if item.regime_id == regime_id),
                0.0,
            )
            action_fast_prior_bias = next(
                (
                    item.bias
                    for item in experience_fast_prior.action_biases
                    if item.abstract_action == abstract_action
                ),
                0.0,
            )
            family_fast_prior_bias = next(
                (
                    item.continuation_bias
                    for item in experience_fast_prior.family_biases
                    if item.action_family_version == active_temporal_snapshot.action_family_version
                ),
                0.0,
            )
        retrieval_readout_parameters = (
            self._rare_heavy_state.retrieval_readout_checkpoint.parameters
            if self._rare_heavy_state is not None and self._rare_heavy_state.retrieval_readout_checkpoint is not None
            else RetrievalControlReadoutParameters.default()
        )
        control_readout = RetrievalControlReadoutStrategy(parameters=retrieval_readout_parameters).build(
            RetrievalControlReadoutInputs(
                regime_id=regime_id,
                abstract_action=abstract_action,
                action_family_version=active_temporal_snapshot.action_family_version,
                switch_gate=active_temporal_snapshot.controller_state.switch_gate,
                knowledge_domains=knowledge_domains,
                experience_domains=experience_domains,
                world_weight=world_weight,
                self_weight=self_weight,
                continuum_position=continuum_position,
                continuum_frequency=continuum_frequency,
                continuum_slow_share=continuum_slow_share,
                continuum_reconstruction_pressure=continuum_reconstruction_pressure,
                delayed_payoff_signal=delayed_payoff_signal,
                sequence_payoff_signal=sequence_payoff_signal,
                playbook_knowledge_hint=playbook_knowledge_hint,
                playbook_experience_hint=playbook_experience_hint,
                knowledge_weight_bias=(
                    experience_fast_prior.knowledge_weight_bias if experience_fast_prior is not None else 0.0
                ),
                experience_weight_bias=(
                    experience_fast_prior.experience_weight_bias if experience_fast_prior is not None else 0.0
                ),
                regime_fast_prior_bias=regime_fast_prior_bias,
                action_fast_prior_bias=action_fast_prior_bias,
                family_fast_prior_bias=family_fast_prior_bias,
                fast_prior_strength=experience_fast_prior.prior_strength if experience_fast_prior is not None else 0.0,
                fast_prior_attribution_count=(
                    len(experience_fast_prior.source_attribution_ids) if experience_fast_prior is not None else 0
                ),
                fast_prior_sequence_count=(
                    len(experience_fast_prior.source_sequence_ids) if experience_fast_prior is not None else 0
                ),
                continuum_active_band_ids=continuum_active_band_ids,
            )
        )
        knowledge_domains = control_readout.knowledge_domains
        experience_domains = control_readout.experience_domains
        knowledge_weight = control_readout.knowledge_weight
        retrieval_depth = _retrieval_depth(
            regime_id=regime_id,
            knowledge_weight=knowledge_weight,
            temporal_snapshot=active_temporal_snapshot,
            continuum_position=continuum_position,
            continuum_reconstruction_pressure=continuum_reconstruction_pressure,
        )
        citation_required = _requires_citation(knowledge_domains)
        jurisdiction_required = _requires_jurisdiction(knowledge_domains)
        risk_band = _risk_band_from_state(
            dual_track_snapshot=dual_track_snapshot,
            temporal_snapshot=active_temporal_snapshot,
            regime_snapshot=regime_snapshot,
        )
        experience_weight = control_readout.experience_weight
        intent_description = (
            f"retrieval policy regime={regime_id} abstract_action={abstract_action} "
            f"knowledge_weight={knowledge_weight:.2f} experience_weight={experience_weight:.2f} "
            f"world_weight={world_weight:.2f} self_weight={self_weight:.2f} "
            f"continuum_position={continuum_position:.2f} slow_share={continuum_slow_share:.2f} "
            f"reconstruction_pressure={continuum_reconstruction_pressure:.2f}."
        )
        return self.publish(
            RetrievalPolicySnapshot(
                knowledge_domains=knowledge_domains,
                experience_domains=experience_domains,
                knowledge_weight=knowledge_weight,
                experience_weight=experience_weight,
                world_weight=world_weight,
                self_weight=self_weight,
                retrieval_depth=retrieval_depth,
                citation_required=citation_required,
                jurisdiction_required=jurisdiction_required,
                risk_band=risk_band,
                regime_id=regime_id,
                abstract_action=abstract_action,
                intent_description=intent_description,
                knowledge_domain_ranking=control_readout.knowledge_domain_ranking,
                experience_domain_ranking=control_readout.experience_domain_ranking,
                response_mode_hint=control_readout.response_mode_hint,
                clarification_bias=control_readout.clarification_bias,
                refer_out_bias=control_readout.refer_out_bias,
                answer_depth_bias=control_readout.answer_depth_bias,
                continuum_target_position_hint=control_readout.continuum_target_position,
                ordering_bias=control_readout.ordering_bias,
                ordering_driver_hint=control_readout.ordering_driver,
                description=(
                    f"{control_readout.description} Retrieval policy remains a compact control surface; "
                    "knowledge and experience owners publish evidence and priors without entering ETA ownership. "
                    f"checkpoint={'present' if self._rare_heavy_state is not None and self._rare_heavy_state.retrieval_readout_checkpoint is not None else 'default'}."
                ),
            )
        )

    def _publish_temporal_disabled_fallback(self) -> Snapshot[RetrievalPolicySnapshot]:
        return self.publish(
            RetrievalPolicySnapshot(
                knowledge_domains=("general",),
                experience_domains=("general_guidance_patterns",),
                knowledge_weight=0.5,
                experience_weight=0.5,
                world_weight=0.5,
                self_weight=0.5,
                retrieval_depth=KnowledgeDepth.LIGHT,
                citation_required=False,
                jurisdiction_required=False,
                risk_band=RiskBand.LOW,
                regime_id=None,
                abstract_action=None,
                intent_description=(
                    "retrieval policy fallback: temporal owners are disabled, "
                    "so retrieval stays conservative and does not infer an abstract action."
                ),
                description=(
                    "Retrieval policy published conservative fallback because temporal owners "
                    "were explicitly disabled by wiring; no temporal state was reconstructed."
                ),
            )
        )

    async def process_standalone(self, **kwargs: Any) -> Snapshot[RetrievalPolicySnapshot]:
        raise NotImplementedError("RetrievalPolicyModule should be driven by upstream runtime state.")

