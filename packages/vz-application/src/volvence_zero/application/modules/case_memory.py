"""CaseMemoryModule (debt #9 wave 2).

R5 application-tier owner: derives ``CaseMemorySnapshot`` from
memory + retrieval policy snapshots; matches case episodes
against problem patterns and risk markers.

Wave 2 of debt #9 split: this was lines 2960-3234 of the
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


class CaseMemoryModule(RuntimeModule[CaseMemorySnapshot]):
    slot_name = "case_memory"
    owner = "CaseMemoryModule"
    value_type = CaseMemorySnapshot
    dependencies = ("retrieval_policy", "memory", "dual_track", "prediction_error")
    default_wiring_level = WiringLevel.ACTIVE

    def __init__(
        self,
        *,
        rare_heavy_state: ApplicationRareHeavyState | None = None,
        store: ApplicationCaseMemoryStore | None = None,
        wiring_level: WiringLevel | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._rare_heavy_state = rare_heavy_state
        self._store = store

    async def process(self, upstream: Mapping[str, Snapshot[Any]]) -> Snapshot[CaseMemorySnapshot]:
        from volvence_zero.prediction.error import PredictionErrorSnapshot

        retrieval_policy = upstream["retrieval_policy"].value
        memory_snapshot = upstream["memory"].value
        dual_track_snapshot = upstream["dual_track"].value
        prediction_error_snapshot = upstream["prediction_error"].value
        if not isinstance(retrieval_policy, RetrievalPolicySnapshot):
            raise TypeError("retrieval_policy must publish RetrievalPolicySnapshot.")
        if not isinstance(dual_track_snapshot, DualTrackSnapshot):
            raise TypeError("dual_track must publish DualTrackSnapshot.")
        prediction_error = (
            prediction_error_snapshot if isinstance(prediction_error_snapshot, PredictionErrorSnapshot) else None
        )
        memory_value = memory_snapshot if isinstance(memory_snapshot, MemorySnapshot) else None
        memory_continuum_profile = _continuum_profile(memory_value)
        entries = _case_entries(memory_value, retrieval_policy=retrieval_policy)
        hits: list[CaseEpisodeHit] = []
        active_problem_patterns: list[str] = []
        active_risk_markers: list[str] = []
        active_band_ids: list[str] = []
        continuum_positions: list[float] = []
        retrieval_policy_id = f"policy:{hash(retrieval_policy.intent_description) & 0xFFFF:04x}"
        records = (
            self._store.query(
                experience_domains=retrieval_policy.experience_domains,
                regime_id=retrieval_policy.regime_id,
                risk_band=retrieval_policy.risk_band.value,
                limit=3,
            )
            if self._store is not None
            else ()
        )
        for record in records:
            active_problem_patterns.append(record.problem_pattern)
            active_risk_markers.extend(record.risk_markers)
            continuum_location = _continuum_location_from_record(record)
            if continuum_location is not None:
                active_band_ids.append(continuum_location.band_id)
                continuum_positions.append(continuum_location.position)
            try:
                outcome_label = ExperienceOutcomeLabel(record.outcome_label)
            except ValueError as exc:
                # Fail loudly with producer context (case_id) instead of
                # surfacing a bare enum ValueError from deep inside
                # propagate(). The contract owner is this module; the
                # producer (figure/character/etc. compiler) violated it.
                valid = tuple(member.value for member in ExperienceOutcomeLabel)
                raise ValueError(
                    f"CaseMemoryRecord(case_id={record.case_id!r}) carries "
                    f"outcome_label={record.outcome_label!r}, which is not a "
                    f"valid ExperienceOutcomeLabel. Expected one of {valid}. "
                    f"Fix the producing domain pack / compiler."
                ) from exc
            hits.append(
                CaseEpisodeHit(
                    case_id=record.case_id,
                    domain=record.domain,
                    problem_pattern=record.problem_pattern,
                    user_state_pattern=record.user_state_pattern,
                    risk_markers=record.risk_markers,
                    track_tags=record.track_tags,
                    regime_tags=record.regime_tags,
                    intervention_steps=tuple(
                        CaseInterventionStep(
                            step_id=f"{record.case_id}:{step_index}",
                            step_order=step_index,
                            regime_id=retrieval_policy.regime_id,
                            abstract_action=retrieval_policy.abstract_action,
                            action_label=step,
                            description=f"Persisted case ordering step {step}.",
                        )
                        for step_index, step in enumerate(record.intervention_ordering, start=1)
                    ),
                    outcome=CaseOutcomeSummary(
                        outcome_label=outcome_label,
                        delayed_signal_count=record.delayed_signal_count,
                        escalation_observed=record.escalation_observed,
                        repair_observed=record.repair_observed,
                        confidence=record.confidence,
                        description=record.description,
                    ),
                    relevance_score=record.relevance_score,
                    description=record.description,
                    continuum_location=continuum_location,
                )
            )
        for index, entry in enumerate(entries, start=1):
            problem_pattern = _entry_problem_pattern(entry)
            user_state_pattern = _entry_user_state_pattern(entry)
            risk_markers = _entry_risk_markers(
                entry=entry,
                prediction_error=prediction_error,
                retrieval_policy=retrieval_policy,
            )
            outcome = _entry_outcome_summary(
                prediction_error=prediction_error,
                entry=entry,
            )
            active_problem_patterns.append(problem_pattern)
            active_risk_markers.extend(risk_markers)
            continuum_location = _continuum_location_for_entry(
                entry=entry,
                memory_snapshot=memory_value,
            )
            if continuum_location is not None:
                active_band_ids.append(continuum_location.band_id)
                continuum_positions.append(continuum_location.position)
            hits.append(
                CaseEpisodeHit(
                    case_id=f"case:{entry.entry_id}:{index}",
                    domain=next(iter(retrieval_policy.experience_domains), "general_guidance_patterns"),
                    problem_pattern=problem_pattern,
                    user_state_pattern=user_state_pattern,
                    risk_markers=risk_markers,
                    track_tags=(entry.track.value,),
                    regime_tags=(retrieval_policy.regime_id,) if retrieval_policy.regime_id is not None else (),
                    intervention_steps=(
                        CaseInterventionStep(
                            step_id=f"{entry.entry_id}:1",
                            step_order=1,
                            regime_id=retrieval_policy.regime_id,
                            abstract_action=retrieval_policy.abstract_action,
                            action_label="maintain-current-regime",
                            description=(
                                f"Historical entry '{entry.content[:48]}' aligned with regime={retrieval_policy.regime_id} "
                                f"and abstract_action={retrieval_policy.abstract_action}."
                            ),
                        ),
                    ),
                    outcome=outcome,
                    relevance_score=_clamp(entry.strength * 0.7 + retrieval_policy.experience_weight * 0.3),
                    description=(
                        f"Case memory hit derived from {entry.track.value}-track memory entry "
                        f"with problem_pattern={problem_pattern}."
                    ),
                    continuum_location=continuum_location,
                )
            )
        if self._rare_heavy_state is not None and self._rare_heavy_state.case_clusters:
            existing_patterns = set(active_problem_patterns)
            for index, cluster in enumerate(self._rare_heavy_state.case_clusters[:2], start=1):
                if cluster.problem_pattern in existing_patterns:
                    active_risk_markers.extend(cluster.risk_markers)
                    continue
                continuum_location = _continuum_location_from_band(
                    profile=memory_continuum_profile,
                    band_id=cluster.continuum_band_id or "background-slow",
                    fallback_source="rare-heavy-cluster",
                )
                if continuum_location is not None:
                    continuum_location = ContinuumLocation(
                        profile_id=continuum_location.profile_id,
                        band_id=continuum_location.band_id,
                        band_role=continuum_location.band_role,
                        position=cluster.continuum_position or continuum_location.position,
                        update_frequency=continuum_location.update_frequency,
                        reconstruction_source=continuum_location.reconstruction_source,
                        description=continuum_location.description,
                    )
                    active_band_ids.append(continuum_location.band_id)
                    continuum_positions.append(continuum_location.position)
                hits.append(
                    CaseEpisodeHit(
                        case_id=f"cluster:{cluster.cluster_id}:{index}",
                        domain=next(iter(retrieval_policy.experience_domains), "general_guidance_patterns"),
                        problem_pattern=cluster.problem_pattern,
                        user_state_pattern="cluster-derived",
                        risk_markers=cluster.risk_markers,
                        track_tags=("shared",),
                        regime_tags=(retrieval_policy.regime_id,) if retrieval_policy.regime_id is not None else (),
                        intervention_steps=(
                            CaseInterventionStep(
                                step_id=f"{cluster.cluster_id}:1",
                                step_order=1,
                                regime_id=retrieval_policy.regime_id,
                                abstract_action=retrieval_policy.abstract_action,
                                action_label="cluster-guided-ordering",
                                description=cluster.description,
                            ),
                        ),
                        outcome=CaseOutcomeSummary(
                            outcome_label=ExperienceOutcomeLabel.STABLE,
                            delayed_signal_count=max(cluster.exemplar_count, 1),
                            escalation_observed=False,
                            repair_observed=False,
                            confidence=_clamp(0.5 + cluster.mean_relevance * 0.3),
                            description=f"Derived from case cluster {cluster.cluster_id}.",
                        ),
                        relevance_score=cluster.mean_relevance,
                        description=f"Cluster-derived case memory hit for pattern={cluster.problem_pattern}.",
                        continuum_location=continuum_location,
                    )
                )
                active_problem_patterns.append(cluster.problem_pattern)
                active_risk_markers.extend(cluster.risk_markers)
                existing_patterns.add(cluster.problem_pattern)
        total_hits = max(len(hits), 1)
        support_prior = _clamp(
            sum(1.0 for hit in hits if Track.SELF.value in hit.track_tags) / total_hits
        )
        task_prior = _clamp(
            sum(1.0 for hit in hits if Track.WORLD.value in hit.track_tags) / total_hits
        )
        return self.publish(
            CaseMemorySnapshot(
                retrieval_policy_id=retrieval_policy_id,
                hits=tuple(hits),
                active_problem_patterns=_dedupe(tuple(active_problem_patterns)),
                active_risk_markers=_dedupe(tuple(active_risk_markers)),
                continuum_profile_id=(
                    memory_continuum_profile.profile_id
                    if memory_continuum_profile is not None
                    else hits[0].continuum_location.profile_id
                    if hits and hits[0].continuum_location is not None
                    else None
                ),
                active_band_ids=_dedupe(tuple(active_band_ids)),
                mean_continuum_position=(
                    sum(continuum_positions) / len(continuum_positions) if continuum_positions else 0.0
                ),
                support_prior=support_prior,
                task_prior=task_prior,
                description=(
                    f"Case memory produced {len(hits)} compact case hits for "
                    f"{len(retrieval_policy.experience_domains)} experience domains."
                ),
            )
        )

    async def process_standalone(self, **kwargs: Any) -> Snapshot[CaseMemorySnapshot]:
        raise NotImplementedError("CaseMemoryModule should be driven by RetrievalPolicySnapshot.")

    @staticmethod
    def records_from_snapshot(snapshot: CaseMemorySnapshot) -> tuple[CaseMemoryRecord, ...]:
        return tuple(
            CaseMemoryRecord(
                case_id=hit.case_id,
                domain=hit.domain,
                problem_pattern=hit.problem_pattern,
                user_state_pattern=hit.user_state_pattern,
                risk_markers=hit.risk_markers,
                track_tags=hit.track_tags,
                regime_tags=hit.regime_tags,
                intervention_ordering=tuple(step.action_label for step in hit.intervention_steps),
                outcome_label=hit.outcome.outcome_label,
                delayed_signal_count=hit.outcome.delayed_signal_count,
                escalation_observed=hit.outcome.escalation_observed,
                repair_observed=hit.outcome.repair_observed,
                confidence=hit.outcome.confidence,
                relevance_score=hit.relevance_score,
                description=hit.description,
                continuum_profile_id=(
                    hit.continuum_location.profile_id if hit.continuum_location is not None else None
                ),
                continuum_band_id=(
                    hit.continuum_location.band_id if hit.continuum_location is not None else None
                ),
                continuum_position=(
                    hit.continuum_location.position if hit.continuum_location is not None else 0.0
                ),
                continuum_update_frequency=(
                    hit.continuum_location.update_frequency if hit.continuum_location is not None else 0.0
                ),
                reconstruction_source=(
                    hit.continuum_location.reconstruction_source if hit.continuum_location is not None else "direct"
                ),
            )
            for hit in snapshot.hits
        )

