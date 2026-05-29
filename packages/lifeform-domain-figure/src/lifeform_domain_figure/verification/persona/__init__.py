"""Persona-level verification harness for the figure-vertical.

Wave O closure: an automated pipeline that generates test questions
from a curated bundle's corpus, runs ablations across raw / bundle /
bundle+LoRA conditions, scores voice fidelity + cognition accuracy +
refusal precision deterministically, and emits a 4-gate verdict.

All scoring is repository-internal — no LLM judge, no Anthropic /
OpenAI calls. The bundle's own :attr:`retrieval_index.assertion_is_supported`
and :attr:`style_prior.top_words` are the ground truth.
"""

from lifeform_domain_figure.verification.persona.ablation import (
    run_ablation,
)
from lifeform_domain_figure.verification.persona.out_of_scope_set import (
    OUT_OF_SCOPE_REFUSAL_QUESTIONS,
)
from lifeform_domain_figure.verification.persona.question_generator import (
    generate_in_corpus_questions,
)
from lifeform_domain_figure.verification.persona.records import (
    AblationResult,
    CognitionScore,
    ConditionAggregate,
    GateResult,
    L3FaithfulnessScore,
    PersonaCondition,
    PersonaQuestionCategory,
    PersonaTestQuestion,
    PersonaVerdict,
    QuestionScore,
    RefusalScore,
    VoiceScore,
)
from lifeform_domain_figure.verification.persona.runtime_conditions import (
    with_condition,
)
from lifeform_domain_figure.verification.persona.scoring import (
    DEFAULT_TOP_WORDS_K,
    aggregate_scores,
    score_cognition,
    score_l3_faithfulness,
    score_question,
    score_refusal,
    score_voice,
)
from lifeform_domain_figure.verification.persona.verdict import (
    DEFAULT_VERDICT_THRESHOLDS,
    VerdictThresholds,
    build_persona_verdict,
)

__all__ = [
    "AblationResult",
    "CognitionScore",
    "ConditionAggregate",
    "DEFAULT_TOP_WORDS_K",
    "DEFAULT_VERDICT_THRESHOLDS",
    "GateResult",
    "L3FaithfulnessScore",
    "OUT_OF_SCOPE_REFUSAL_QUESTIONS",
    "PersonaCondition",
    "PersonaQuestionCategory",
    "PersonaTestQuestion",
    "PersonaVerdict",
    "QuestionScore",
    "RefusalScore",
    "VerdictThresholds",
    "VoiceScore",
    "aggregate_scores",
    "build_persona_verdict",
    "generate_in_corpus_questions",
    "run_ablation",
    "score_cognition",
    "score_l3_faithfulness",
    "score_question",
    "score_refusal",
    "score_voice",
    "with_condition",
]
