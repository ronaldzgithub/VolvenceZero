"""DLaaS platform-tier eval gate (Slice 6).

Public surface:

* :func:`attach_eval_routes` — wires every audience / exam / license
  endpoint onto an aiohttp app.
* :func:`build_grader_from_env` — the deployment seam: returns
  :class:`LLMRubricGrader` when the eval LLM env is configured
  (``EVAL_LLM_*`` falling back to ``PROTOCOL_LLM_*``), else the
  fail-closed :class:`DefaultRubricGrader`. Other wheels
  (cultivation, interview) use this factory to share the same judge
  configuration.
* :class:`LLMRubricGrader` — real LLM rubric judge (debt #13).
  Malformed judge output raises :class:`GraderResponseError`; there
  is no silent 0.5 fallback.
* :class:`DefaultRubricGrader` — deterministic fail-closed scorer
  used when no LLM judge backend is configured. Returns a flat
  rubric breakdown; aggregate falls below the pass threshold by
  default so license gates fail safe until a real grader is wired.
* :class:`LLMAudienceAnalyzer` / :func:`build_audience_analyzer_from_env`
  — real corpus-grounded audience analysis (debt #14). Shares the
  eval LLM env contract; without it the audience route keeps the
  honest passthrough (``evidence_stats.analyzer = "none"``).

All endpoints, graders, and analyzers are strictly readouts (R12 /
OA-1 / EVO-2): no learning signal flows back from these endpoints
into the kernel. The license gate blocks ``template.status →
published`` but never mutates kernel state.
"""

from __future__ import annotations

from dlaas_platform_eval.audience import (
    AssetCorpusError,
    AudienceAnalysisError,
    LLMAudienceAnalyzer,
    build_audience_analyzer_from_env,
    load_asset_corpus,
)
from dlaas_platform_eval.grader import (
    DefaultRubricGrader,
    GradedSubmission,
    RubricGrader,
)
from dlaas_platform_eval.llm_grader import (
    EvalLLMConfig,
    EvalLLMError,
    GraderResponseError,
    LLMRubricGrader,
    QuestionGenerationError,
    build_grader_from_env,
    resolve_eval_llm_config,
)
from dlaas_platform_eval.question_gen import (
    GeneratedQuestion,
    QuestionSource,
    generate_exam_questions,
)
from dlaas_platform_eval.routes import EVAL_BUNDLE_APP_KEY, attach_eval_routes

__all__ = (
    "AssetCorpusError",
    "AudienceAnalysisError",
    "DefaultRubricGrader",
    "EVAL_BUNDLE_APP_KEY",
    "LLMAudienceAnalyzer",
    "build_audience_analyzer_from_env",
    "load_asset_corpus",
    "EvalLLMConfig",
    "EvalLLMError",
    "GeneratedQuestion",
    "GradedSubmission",
    "GraderResponseError",
    "LLMRubricGrader",
    "QuestionGenerationError",
    "QuestionSource",
    "RubricGrader",
    "attach_eval_routes",
    "build_grader_from_env",
    "generate_exam_questions",
    "resolve_eval_llm_config",
)
