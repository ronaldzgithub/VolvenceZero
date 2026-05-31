"""Certified Capability Profile ("AI resume") contracts.

A capability profile is the buyer-facing, comparable, auditable resume
attached to a persona-market listing. It composes EXISTING platform
readouts + an exam run + a launch license into:

* headline display indices (``iq_index`` / ``eq_index`` + grade), and
* an evidence-backed multi-axis rubric underneath.

Trust invariants:

* **Readout, not reward (R12 / OA-1).** Every score here is a *display*
  derived from readouts; nothing computed in this module is ever fed
  back as a learning / Face gradient. ``iq_index`` / ``eq_index`` are
  explicitly labelled derived indices, never opaque scalars — each axis
  carries its evidence refs.
* **Certified vs claimed split.** ``certified`` / ``observed`` fields are
  produced by the platform and are immutable to the lister; the
  ``claimed`` block is lister-authored and always marked self-reported.
* **Snapshot at publish.** A profile is frozen at certify time with a
  ``cert_version`` + ``content_hash``; if the listing's asset bundle
  changes, the profile is ``stale`` and must be re-certified.

``compose_capability_indices`` is a pure, versioned function so the same
inputs always yield the same profile and a profile can be re-derived /
audited from its evidence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

CERT_FORMULA_VERSION = "capability-profile.v1"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ProfileProvenanceTier(str, Enum):
    # Lister-authored, always shown as self-reported.
    CLAIMED = "claimed"
    # Produced by the platform exam / license — immutable to the lister.
    CERTIFIED = "certified"
    # Derived from runtime readouts / usage history — immutable.
    OBSERVED = "observed"


class CapabilityAxisId(str, Enum):
    REASONING_SKILL = "reasoning_skill"
    KNOWLEDGE_DEPTH = "knowledge_depth"
    RELATIONSHIP_EQ = "relationship_eq"
    RELIABILITY = "reliability"
    SAFETY = "safety"
    EXPERIENCE = "experience"


# Which tier each axis is sourced from (dominant source).
_AXIS_PROVENANCE: dict[CapabilityAxisId, ProfileProvenanceTier] = {
    CapabilityAxisId.REASONING_SKILL: ProfileProvenanceTier.CERTIFIED,
    CapabilityAxisId.KNOWLEDGE_DEPTH: ProfileProvenanceTier.CERTIFIED,
    CapabilityAxisId.RELATIONSHIP_EQ: ProfileProvenanceTier.OBSERVED,
    CapabilityAxisId.RELIABILITY: ProfileProvenanceTier.OBSERVED,
    CapabilityAxisId.SAFETY: ProfileProvenanceTier.CERTIFIED,
    CapabilityAxisId.EXPERIENCE: ProfileProvenanceTier.OBSERVED,
}


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _clamp01(x: float) -> float:
    if x != x:  # NaN guard
        return 0.0
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _log_scale(value: float, cap: float) -> float:
    """Diminishing-returns normaliser: 0 -> 0, cap -> ~1."""
    if cap <= 0:
        return 0.0
    return _clamp01(math.log1p(max(0.0, value)) / math.log1p(cap))


def display_index(unit: float) -> int:
    """Map a 0..1 composite to an IQ/EQ-style display band.

    0.0 -> 60, 0.5 -> ~105, 1.0 -> 150. This is an *AI capability index*,
    not a human IQ; the UI must label it as such and keep the underlying
    axis values visible.
    """
    return int(round(60 + _clamp01(unit) * 90))


def grade_for(composite_unit: float) -> str:
    u = _clamp01(composite_unit)
    if u >= 0.85:
        return "A"
    if u >= 0.70:
        return "B"
    if u >= 0.55:
        return "C"
    if u >= 0.40:
        return "D"
    return "F"


def percentile_band_for(composite_unit: float) -> str:
    """Heuristic band (no population calibration yet). Marked as such."""
    u = _clamp01(composite_unit)
    if u >= 0.85:
        return "top 10%"
    if u >= 0.70:
        return "top 25%"
    if u >= 0.55:
        return "top 50%"
    return "developing"


# ---------------------------------------------------------------------------
# Inputs (normalised readout signals)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapabilityReadoutInputs:
    """Normalised 0..1 signals fed to :func:`compose_capability_indices`.

    The caller (platform certify orchestrator) is responsible for pulling
    these from the exam run + readout snapshot and normalising to 0..1.
    Missing signals default to 0.5 (neutral) and lower ``data_completeness``.
    """

    # Exam (certified skill / IQ).
    exam_aggregate: float = 0.5  # ExamRunSpec.aggregate_score (0..1)
    license_granted: bool = False
    # Six-family signals (0..1; 0.5 == neutral default).
    f1_task: float = 0.5
    f2_interaction: float = 0.5
    f3_relationship: float = 0.5
    f4_learning: float = 0.5
    f5_abstraction: float = 0.5
    f6_safety: float = 0.5
    # Interlocutor 12-axis (EQ).
    interlocutor_trust: float = 0.5
    interlocutor_rapport: float = 0.5
    # Reliability signals.
    eval_pass_rate: float = 0.5
    regime_stability: float = 0.5
    tool_repeat_fail_rate: float = 0.0  # higher == worse
    # Factory LLM-judge (0..1; divide the 0..10 axes by 10).
    judge_fidelity: float = 0.5
    judge_stability: float = 0.5
    judge_safety: float = 0.5
    # Human soft signal.
    kindness_ratio: float = 0.5
    # Experience counts (raw; log-scaled internally).
    closed_scenes: int = 0
    regime_history_days: int = 0
    usage_turns: int = 0
    tenure_days: int = 0
    # 0..1 fraction of the above signals that were actually present.
    data_completeness: float = 0.5

    def experience_norm(self) -> float:
        return (
            _log_scale(self.closed_scenes, 200)
            + _log_scale(self.regime_history_days, 180)
            + _log_scale(self.usage_turns, 5000)
            + _log_scale(self.tenure_days, 365)
        ) / 4.0


# ---------------------------------------------------------------------------
# Composed structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AxisScore:
    axis: CapabilityAxisId
    value_0_100: float
    confidence: float = 0.5
    provenance: ProfileProvenanceTier = ProfileProvenanceTier.OBSERVED
    evidence_refs: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "axis": self.axis.value,
            "value_0_100": round(self.value_0_100, 2),
            "confidence": round(self.confidence, 2),
            "provenance": self.provenance.value,
            "evidence_refs": list(self.evidence_refs),
        }

    @staticmethod
    def from_json(data: Mapping[str, Any]) -> "AxisScore":
        return AxisScore(
            axis=CapabilityAxisId(str(data["axis"])),
            value_0_100=float(data.get("value_0_100", 0.0) or 0.0),
            confidence=float(data.get("confidence", 0.5) or 0.5),
            provenance=ProfileProvenanceTier(
                str(data.get("provenance", ProfileProvenanceTier.OBSERVED.value))
            ),
            evidence_refs=tuple(
                str(x) for x in (data.get("evidence_refs") or ())
            ),
        )


@dataclass(frozen=True)
class SkillScore:
    name: str
    score_0_100: float
    source_exam_run_id: str = ""
    rubric_breakdown: tuple[Mapping[str, Any], ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score_0_100": round(self.score_0_100, 2),
            "source_exam_run_id": self.source_exam_run_id,
            "rubric_breakdown": [dict(b) for b in self.rubric_breakdown],
        }

    @staticmethod
    def from_json(data: Mapping[str, Any]) -> "SkillScore":
        return SkillScore(
            name=str(data.get("name", "")),
            score_0_100=float(data.get("score_0_100", 0.0) or 0.0),
            source_exam_run_id=str(data.get("source_exam_run_id", "")),
            rubric_breakdown=tuple(
                dict(b) for b in (data.get("rubric_breakdown") or ())
            ),
        )


@dataclass(frozen=True)
class ComposedCapability:
    """Result of :func:`compose_capability_indices`."""

    iq_index: int
    eq_index: int
    overall_grade: str
    percentile_band: str
    composite_unit: float
    axes: tuple[AxisScore, ...]
    formula_version: str = CERT_FORMULA_VERSION


def compose_capability_indices(
    inputs: CapabilityReadoutInputs,
    *,
    evidence_refs: Mapping[str, Sequence[str]] | None = None,
) -> ComposedCapability:
    """Deterministically compose six axes + IQ/EQ display indices.

    ``evidence_refs`` optionally maps an axis value (``"reasoning_skill"``
    …) to the ids backing it (exam run id, readout snapshot hash, …) so
    the profile stays auditable. Pure: same inputs -> same output.
    """

    ev = {k: tuple(v) for k, v in (evidence_refs or {}).items()}
    conf = _clamp01(inputs.data_completeness)

    exp = inputs.experience_norm()
    trust_rapport = (
        _clamp01(inputs.interlocutor_trust) + _clamp01(inputs.interlocutor_rapport)
    ) / 2.0

    reasoning = 100.0 * _clamp01(
        0.50 * inputs.exam_aggregate
        + 0.30 * inputs.f1_task
        + 0.20 * inputs.f5_abstraction
    )
    knowledge = 100.0 * _clamp01(
        0.50 * inputs.exam_aggregate + 0.30 * inputs.f4_learning + 0.20 * exp
    )
    relationship = 100.0 * _clamp01(
        0.30 * inputs.f2_interaction
        + 0.30 * inputs.f3_relationship
        + 0.25 * trust_rapport
        + 0.15 * inputs.kindness_ratio
    )
    reliability = 100.0 * _clamp01(
        0.40 * inputs.eval_pass_rate
        + 0.30 * inputs.regime_stability
        + 0.30 * (1.0 - _clamp01(inputs.tool_repeat_fail_rate))
    )
    safety = 100.0 * _clamp01(
        0.40 * inputs.f6_safety
        + 0.30 * inputs.judge_safety
        + 0.30 * (1.0 if inputs.license_granted else 0.5)
    )
    experience = 100.0 * exp

    axis_values = {
        CapabilityAxisId.REASONING_SKILL: reasoning,
        CapabilityAxisId.KNOWLEDGE_DEPTH: knowledge,
        CapabilityAxisId.RELATIONSHIP_EQ: relationship,
        CapabilityAxisId.RELIABILITY: reliability,
        CapabilityAxisId.SAFETY: safety,
        CapabilityAxisId.EXPERIENCE: experience,
    }
    axes = tuple(
        AxisScore(
            # Round at construction so the dataclass is the canonical
            # (rounded) form and JSON round-trips are stable.
            axis=axis,
            value_0_100=round(value, 2),
            confidence=round(conf, 2),
            provenance=_AXIS_PROVENANCE[axis],
            evidence_refs=ev.get(axis.value, ()),
        )
        for axis, value in axis_values.items()
    )

    iq_unit = _clamp01(
        0.45 * (reasoning / 100.0)
        + 0.30 * (knowledge / 100.0)
        + 0.25 * (reliability / 100.0)
    )
    eq_unit = _clamp01(
        0.55 * (relationship / 100.0)
        + 0.25 * trust_rapport
        + 0.20 * inputs.f2_interaction
    )
    composite_unit = _clamp01(
        sum(value for value in axis_values.values()) / (100.0 * len(axis_values))
    )

    return ComposedCapability(
        iq_index=display_index(iq_unit),
        eq_index=display_index(eq_unit),
        overall_grade=grade_for(composite_unit),
        percentile_band=percentile_band_for(composite_unit),
        composite_unit=round(composite_unit, 4),
        axes=axes,
    )


# ---------------------------------------------------------------------------
# Experience + claimed (lister) blocks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperienceSummary:
    closed_scenes: int = 0
    regime_history_days: int = 0
    usage_turns: int = 0
    tenure_days: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "closed_scenes": self.closed_scenes,
            "regime_history_days": self.regime_history_days,
            "usage_turns": self.usage_turns,
            "tenure_days": self.tenure_days,
        }

    @staticmethod
    def from_json(data: Mapping[str, Any]) -> "ExperienceSummary":
        return ExperienceSummary(
            closed_scenes=int(data.get("closed_scenes", 0) or 0),
            regime_history_days=int(data.get("regime_history_days", 0) or 0),
            usage_turns=int(data.get("usage_turns", 0) or 0),
            tenure_days=int(data.get("tenure_days", 0) or 0),
        )


@dataclass(frozen=True)
class ClaimedResume:
    """Lister-authored, always rendered as self-reported."""

    headline_tagline: str = ""
    role_title: str = ""
    domains: tuple[str, ...] = ()
    highlights: tuple[str, ...] = ()
    sample_refs: tuple[str, ...] = ()
    recommended_use_cases: tuple[str, ...] = ()
    experience_narrative: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "headline_tagline": self.headline_tagline,
            "role_title": self.role_title,
            "domains": list(self.domains),
            "highlights": list(self.highlights),
            "sample_refs": list(self.sample_refs),
            "recommended_use_cases": list(self.recommended_use_cases),
            "experience_narrative": self.experience_narrative,
        }

    @staticmethod
    def from_json(data: Mapping[str, Any]) -> "ClaimedResume":
        if not isinstance(data, Mapping):
            return ClaimedResume()
        return ClaimedResume(
            headline_tagline=str(data.get("headline_tagline", "")),
            role_title=str(data.get("role_title", "")),
            domains=tuple(str(x) for x in (data.get("domains") or ())),
            highlights=tuple(str(x) for x in (data.get("highlights") or ())),
            sample_refs=tuple(str(x) for x in (data.get("sample_refs") or ())),
            recommended_use_cases=tuple(
                str(x) for x in (data.get("recommended_use_cases") or ())
            ),
            experience_narrative=str(data.get("experience_narrative", "")),
        )


# ---------------------------------------------------------------------------
# The profile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapabilityProfileSpec:
    """The certified capability profile attached to a listing."""

    profile_ref: str
    listing_ref: str
    ai_id: str = ""
    vertical: str = ""
    archetype: str = ""
    # Headline display.
    iq_index: int = 0
    eq_index: int = 0
    overall_grade: str = "F"
    percentile_band: str = "developing"
    # Evidence-backed detail.
    axes: tuple[AxisScore, ...] = ()
    skills: tuple[SkillScore, ...] = ()
    experience: ExperienceSummary = field(default_factory=ExperienceSummary)
    # Lister self-report.
    claimed: ClaimedResume = field(default_factory=ClaimedResume)
    # Certification metadata.
    cert_version: str = CERT_FORMULA_VERSION
    certified_at_ms: int = 0
    exam_run_id: str = ""
    readout_snapshot_hash: str = ""
    license_granted: bool = False
    # = listing.asset_bundle_hash at certify time; drives stale detection.
    content_hash: str = ""

    def is_stale(self, current_content_hash: str) -> bool:
        """A profile is stale once the listing's asset bundle changes."""
        if not self.content_hash:
            return False
        return self.content_hash != current_content_hash

    def to_json(self) -> dict[str, Any]:
        return {
            "profile_ref": self.profile_ref,
            "listing_ref": self.listing_ref,
            "ai_id": self.ai_id,
            "vertical": self.vertical,
            "archetype": self.archetype,
            "iq_index": self.iq_index,
            "eq_index": self.eq_index,
            "overall_grade": self.overall_grade,
            "percentile_band": self.percentile_band,
            "axes": [a.to_json() for a in self.axes],
            "skills": [s.to_json() for s in self.skills],
            "experience": self.experience.to_json(),
            "claimed": self.claimed.to_json(),
            "cert_version": self.cert_version,
            "certified_at_ms": self.certified_at_ms,
            "exam_run_id": self.exam_run_id,
            "readout_snapshot_hash": self.readout_snapshot_hash,
            "license_granted": self.license_granted,
            "content_hash": self.content_hash,
        }

    @staticmethod
    def from_json(data: Mapping[str, Any]) -> "CapabilityProfileSpec":
        return CapabilityProfileSpec(
            profile_ref=str(data.get("profile_ref", "")),
            listing_ref=str(data.get("listing_ref", "")),
            ai_id=str(data.get("ai_id", "")),
            vertical=str(data.get("vertical", "")),
            archetype=str(data.get("archetype", "")),
            iq_index=int(data.get("iq_index", 0) or 0),
            eq_index=int(data.get("eq_index", 0) or 0),
            overall_grade=str(data.get("overall_grade", "F")),
            percentile_band=str(data.get("percentile_band", "developing")),
            axes=tuple(AxisScore.from_json(a) for a in (data.get("axes") or ())),
            skills=tuple(SkillScore.from_json(s) for s in (data.get("skills") or ())),
            experience=ExperienceSummary.from_json(data.get("experience") or {}),
            claimed=ClaimedResume.from_json(data.get("claimed") or {}),
            cert_version=str(data.get("cert_version", CERT_FORMULA_VERSION)),
            certified_at_ms=int(data.get("certified_at_ms", 0) or 0),
            exam_run_id=str(data.get("exam_run_id", "")),
            readout_snapshot_hash=str(data.get("readout_snapshot_hash", "")),
            license_granted=bool(data.get("license_granted", False)),
            content_hash=str(data.get("content_hash", "")),
        )


def build_profile_from_composed(
    *,
    profile_ref: str,
    listing_ref: str,
    ai_id: str,
    vertical: str,
    archetype: str,
    composed: ComposedCapability,
    skills: Sequence[SkillScore],
    experience: ExperienceSummary,
    claimed: ClaimedResume,
    certified_at_ms: int,
    exam_run_id: str,
    readout_snapshot_hash: str,
    license_granted: bool,
    content_hash: str,
) -> CapabilityProfileSpec:
    """Assemble a :class:`CapabilityProfileSpec` from a composed result."""
    return CapabilityProfileSpec(
        profile_ref=profile_ref,
        listing_ref=listing_ref,
        ai_id=ai_id,
        vertical=vertical,
        archetype=archetype,
        iq_index=composed.iq_index,
        eq_index=composed.eq_index,
        overall_grade=composed.overall_grade,
        percentile_band=composed.percentile_band,
        axes=composed.axes,
        skills=tuple(skills),
        experience=experience,
        claimed=claimed,
        cert_version=composed.formula_version,
        certified_at_ms=certified_at_ms,
        exam_run_id=exam_run_id,
        readout_snapshot_hash=readout_snapshot_hash,
        license_granted=license_granted,
        content_hash=content_hash,
    )


__all__ = [
    "CERT_FORMULA_VERSION",
    "AxisScore",
    "CapabilityAxisId",
    "CapabilityProfileSpec",
    "CapabilityReadoutInputs",
    "ClaimedResume",
    "ComposedCapability",
    "ExperienceSummary",
    "ProfileProvenanceTier",
    "SkillScore",
    "build_profile_from_composed",
    "compose_capability_indices",
    "display_index",
    "grade_for",
    "percentile_band_for",
]
