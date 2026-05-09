"""ExperienceConsolidationModule (debt #9 wave 2).

R5 application-tier owner: surfaces
``ExperienceConsolidationSnapshot`` from the slow-loop
completed-result tuple. Lightweight wrapper around the
experience-delta / delayed-outcome accumulator.

Wave 2 of debt #9 split: this was lines 3865-3941 of the
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


class ExperienceConsolidationModule(RuntimeModule[ExperienceConsolidationSnapshot]):
    slot_name = "experience_consolidation"
    owner = "ExperienceConsolidationModule"
    value_type = ExperienceConsolidationSnapshot
    dependencies = ()
    default_wiring_level = WiringLevel.ACTIVE

    def publish_snapshot(
        self,
        *,
        completed_results: tuple[Any, ...] = (),
    ) -> Snapshot[ExperienceConsolidationSnapshot]:
        deltas: list[ExperienceDelta] = []
        delayed_outcome_ledger: list[ApplicationOutcomeAttribution] = []
        sequence_payoffs: list[ApplicationSequencePayoff] = []
        source_job_id = "experience-consolidation:none"
        continuum_profile_id: str | None = None
        active_band_ids: list[str] = []
        latest_prior_update: ApplicationPriorUpdate | None = None
        latest_writeback_report: ApplicationPriorWritebackReport | None = None
        delayed_credit_summary: DelayedCreditSummary | None = None
        for result in completed_results[-4:]:
            result_deltas = getattr(result, "experience_deltas", ())
            if getattr(result, "job_id", None) is not None:
                source_job_id = result.job_id
            deltas.extend(result_deltas)
            delayed_outcome_ledger.extend(getattr(result, "delayed_outcome_ledger", ()))
            sequence_payoffs.extend(getattr(result, "sequence_payoffs", ()))
            continuum_profile_id = continuum_profile_id or getattr(result, "continuum_profile_id", None)
            active_band_ids.extend(getattr(result, "case_band_ids", ()))
            active_band_ids.extend(getattr(result, "playbook_band_ids", ()))
            result_prior_update = getattr(result, "application_prior_update", None)
            if isinstance(result_prior_update, ApplicationPriorUpdate):
                latest_prior_update = result_prior_update
            result_writeback_report = getattr(result, "application_prior_writeback_report", None)
            if isinstance(result_writeback_report, ApplicationPriorWritebackReport):
                latest_writeback_report = result_writeback_report
            result_delayed_credit_summary = getattr(result, "delayed_credit_summary", None)
            if isinstance(result_delayed_credit_summary, DelayedCreditSummary):
                delayed_credit_summary = result_delayed_credit_summary
        promoted_case_count = sum(1 for delta in deltas if delta.target_slot == "case_memory" and not delta.blocked)
        playbook_delta_count = sum(1 for delta in deltas if delta.target_slot == "strategy_playbook")
        boundary_delta_count = sum(1 for delta in deltas if delta.target_slot == "boundary_policy")
        return self.publish(
            ExperienceConsolidationSnapshot(
                source_session_post_job_id=source_job_id,
                promoted_case_count=promoted_case_count,
                playbook_delta_count=playbook_delta_count,
                boundary_delta_count=boundary_delta_count,
                deltas=tuple(deltas[-6:]),
                description=(
                    f"Experience consolidation published {len(deltas[-6:])} recent deltas, "
                    f"{len(delayed_outcome_ledger[-6:])} delayed outcomes, and "
                    f"{len(sequence_payoffs[-4:])} sequence payoff summaries from "
                    f"{len(completed_results)} completed slow-loop result(s)."
                ),
                delayed_outcome_ledger=tuple(delayed_outcome_ledger[-6:]),
                sequence_payoffs=tuple(sequence_payoffs[-4:]),
                latest_prior_update=latest_prior_update,
                latest_writeback_report=latest_writeback_report,
                delayed_credit_summary=delayed_credit_summary,
                continuum_profile_id=continuum_profile_id,
                active_band_ids=_dedupe(tuple(active_band_ids)),
            )
        )

    async def process(self, upstream: Mapping[str, Snapshot[Any]]) -> Snapshot[ExperienceConsolidationSnapshot]:
        raise NotImplementedError("ExperienceConsolidationModule is published via process_standalone().")

    async def process_standalone(
        self,
        *,
        completed_results: tuple[Any, ...] = (),
        **kwargs: Any,
    ) -> Snapshot[ExperienceConsolidationSnapshot]:
        del kwargs
        return self.publish_snapshot(completed_results=completed_results)
