"""Deterministic natural-language rendering of the personal conditioning readout.

State KV arm B-prime and the distillation teacher need the *same*
information that the latent conditioning path receives, presented as a
natural-language state statement. To keep that comparison honest and to
preserve the snapshot's privacy posture, this renderer consumes only the
typed readout published by ``PersonalConditioningModule`` -- the 16
labelled coordinates plus confidence. It never reads semantic records,
raw dialogue, or profile prose.

The rendering is a pure template (numeric value -> qualitative bucket ->
English clause). No LLM call, no learned mapping: this is a readout
presentation owned by the personal-conditioning owner (R8), analogous to
the regime module owning its ``llm_guidance`` prose.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from volvence_zero.personal_conditioning_contracts import (
    PERSONAL_CONDITIONING_VECTOR_LABELS,
)

# Display grouping for the frozen personal-conditioning.v1 coordinates.
# Keys are the protocol labels (exact enum match, not text matching);
# values are (group, human phrase). Group order below is fixed so the
# rendering is deterministic for a given readout.
_GROUP_ORDER: tuple[str, ...] = ("User", "Relationship", "Goals", "Boundaries")

_LABEL_PHRASES: Mapping[str, tuple[str, str]] = {
    "user_stability": ("User", "overall stability"),
    "user_overwhelm": ("User", "overwhelm-pattern strength"),
    "user_control": ("User", "control-seeking signal"),
    "relationship_trust": ("Relationship", "trust"),
    "relationship_continuity": ("Relationship", "continuity"),
    "relationship_repair_need": ("Relationship", "repair need"),
    "relationship_emotional_load": ("Relationship", "emotional load"),
    "relationship_attunement_gap": ("Relationship", "attunement gap"),
    "goal_alignment": ("Goals", "alignment"),
    "goal_value_conflict": ("Goals", "value conflict"),
    "goal_decision_readiness": ("Goals", "decision readiness"),
    "goal_reversibility_need": ("Goals", "reversibility need"),
    "boundary_compliance": ("Boundaries", "compliance"),
    "boundary_autonomy_risk": ("Boundaries", "autonomy risk"),
    "boundary_consent_clarity": ("Boundaries", "consent clarity"),
    "boundary_overreach_risk": ("Boundaries", "overreach risk"),
}


def _qualitative_level(value: float) -> str:
    """Bucket a [0, 1] coordinate into a qualitative rendering level."""

    if value < 0.34:
        return "low"
    if value < 0.67:
        return "moderate"
    return "high"


def render_personal_conditioning_statement(
    *,
    state_vector: Sequence[float],
    vector_labels: Sequence[str],
    confidence: float,
    is_cold_start: bool,
) -> str:
    """Render the typed readout as a natural-language state statement.

    Returns an empty string for cold-start or zero-confidence readouts:
    with no evidence there is nothing to state, matching the latent
    path's injection gate. Every coordinate is rendered with both its
    qualitative bucket and numeric value so the statement carries
    exactly the information content of the vector.
    """

    if tuple(vector_labels) != PERSONAL_CONDITIONING_VECTOR_LABELS:
        raise ValueError(
            "personal conditioning rendering requires the frozen "
            "personal-conditioning.v1 label contract."
        )
    if len(state_vector) != len(vector_labels):
        raise ValueError(
            "personal conditioning rendering requires one coordinate per label."
        )
    if any(not 0.0 <= value <= 1.0 for value in state_vector):
        raise ValueError(
            "personal conditioning rendering requires coordinates in [0, 1]."
        )
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            "personal conditioning rendering requires confidence in [0, 1]."
        )
    if is_cold_start and (
        confidence != 0.0 or any(value != 0.0 for value in state_vector)
    ):
        raise ValueError(
            "cold-start personal conditioning rendering requires zero "
            "confidence and an all-zero vector."
        )
    if is_cold_start or confidence == 0.0:
        return ""

    grouped: dict[str, list[str]] = {group: [] for group in _GROUP_ORDER}
    for label, value in zip(vector_labels, state_vector, strict=True):
        group, phrase = _LABEL_PHRASES[label]
        grouped[group].append(
            f"{phrase} {_qualitative_level(value)} ({value:.2f})"
        )

    lines = [
        "Current relational state estimate "
        f"(typed readout only, confidence {confidence:.2f}):"
    ]
    for group in _GROUP_ORDER:
        lines.append(f"- {group}: " + "; ".join(grouped[group]) + ".")
    return "\n".join(lines)


__all__ = ["render_personal_conditioning_statement"]
