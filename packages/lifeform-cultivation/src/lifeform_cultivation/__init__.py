"""lifeform-cultivation — autonomous industry-expert cultivation engine.

Public surface:

* :class:`CultivationSeed` / :class:`CultivationCurriculum` — the
  operator-supplied "rough persona" + autonomous study schedule.
* :class:`CultivationEngine` — runs the research -> ingest -> study ->
  reflect loop and returns a :class:`CultivationProgress` readout.
* :class:`CultivationSink` / :class:`SessionCultivationSink` — the
  kernel-facing edge (canonical session + ingestion surfaces only).
* :func:`assess_coherence` / :class:`CoherenceAssessment` — the
  school-coherence (R14 regime concentration) readout.

Boundary: this wheel never imports a vz-* kernel-owner wheel nor the
dlaas-platform-api wheel. Cognition stays kernel-owned; the engine only
sequences intake and reads published readouts.
"""

from __future__ import annotations

from lifeform_cultivation.coherence import (
    CoherenceAssessment,
    ProtocolCoherenceAssessment,
    assess_coherence,
    assess_protocol_coherence,
)
from lifeform_cultivation.curriculum import (
    CultivationCurriculum,
    CultivationSeed,
    build_charter_text,
    build_study_brief,
)
from lifeform_cultivation.engine import (
    CultivationEngine,
    CultivationProgress,
    CycleEvent,
)
from lifeform_cultivation.protocols import (
    IDENTITY_CORE_WEIGHT_FLOOR,
    build_identity_core_protocol,
    is_identity_core,
)
from lifeform_cultivation.sink import (
    CultivationSink,
    ResearchDoc,
    SessionCultivationSink,
    StudyTurn,
)

__all__ = [
    "CoherenceAssessment",
    "CultivationCurriculum",
    "CultivationEngine",
    "CultivationProgress",
    "CultivationSeed",
    "CultivationSink",
    "CycleEvent",
    "IDENTITY_CORE_WEIGHT_FLOOR",
    "ProtocolCoherenceAssessment",
    "ResearchDoc",
    "SessionCultivationSink",
    "StudyTurn",
    "assess_coherence",
    "assess_protocol_coherence",
    "build_charter_text",
    "build_identity_core_protocol",
    "build_study_brief",
    "is_identity_core",
]
