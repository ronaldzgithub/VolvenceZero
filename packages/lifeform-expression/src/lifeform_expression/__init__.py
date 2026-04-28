"""Lifeform expression layer — the lifeform's voice.

Public API:

* ``PromptPlanner`` / ``PromptPlan`` / ``SectionId`` / ``TurnIntent`` — read-only
  structured plan over kernel snapshot state.
* ``GroundedResponseSynthesizer`` — implements ``vz-runtime.ResponseSynthesizer``
  by rendering the plan deterministically.
* ``EtiquetteWatchdog`` / ``EtiquetteVerdict`` / ``SpeakVerdict`` — UX-only
  speak/wait/silent advisor; not a kernel modification gate.
"""

from __future__ import annotations

from lifeform_expression.etiquette_watchdog import (
    EtiquetteVerdict,
    EtiquetteWatchdog,
    SpeakVerdict,
)
from lifeform_expression.llm_synthesizer import LifeformLLMResponseSynthesizer
from lifeform_expression.prompt_planner import (
    PromptPlan,
    PromptPlanner,
    SectionId,
    TurnIntent,
)
from lifeform_expression.response_synthesizer import GroundedResponseSynthesizer

__all__ = (
    "EtiquetteVerdict",
    "EtiquetteWatchdog",
    "GroundedResponseSynthesizer",
    "LifeformLLMResponseSynthesizer",
    "PromptPlan",
    "PromptPlanner",
    "SectionId",
    "SpeakVerdict",
    "TurnIntent",
)
