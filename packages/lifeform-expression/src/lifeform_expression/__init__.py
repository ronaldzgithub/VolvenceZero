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
from lifeform_expression.grounded_decoder import (
    GroundedDecoder,
    GroundedDecoderConfig,
    UngroundedAssertionError,
)
from lifeform_expression.llm_synthesizer import LifeformLLMResponseSynthesizer
from lifeform_expression.scope_refuser import (
    CoveragePolicy,
    ScopeRefuser,
    ScopeRefuserConfig,
    ScopeRefusalDirective,
)
from lifeform_expression.style_prior_injector import (
    StyleHint,
    StylePriorInjector,
    StylePriorInjectorConfig,
    TokenIdAdapter,
)
from lifeform_expression.prompt_planner import (
    PromptPlan,
    PromptPlanner,
    SectionId,
    TurnIntent,
)
from lifeform_expression.reflection_hints import (
    lesson_hint_map,
    reflection_lesson_hint,
    reflection_tension_hint,
    tension_hint_map,
)
from lifeform_expression.response_synthesizer import GroundedResponseSynthesizer

__all__ = (
    "CoveragePolicy",
    "EtiquetteVerdict",
    "EtiquetteWatchdog",
    "GroundedDecoder",
    "GroundedDecoderConfig",
    "GroundedResponseSynthesizer",
    "LifeformLLMResponseSynthesizer",
    "PromptPlan",
    "PromptPlanner",
    "ScopeRefusalDirective",
    "ScopeRefuser",
    "ScopeRefuserConfig",
    "SectionId",
    "SpeakVerdict",
    "StyleHint",
    "StylePriorInjector",
    "StylePriorInjectorConfig",
    "TokenIdAdapter",
    "TurnIntent",
    "UngroundedAssertionError",
    "lesson_hint_map",
    "reflection_lesson_hint",
    "reflection_tension_hint",
    "tension_hint_map",
)
