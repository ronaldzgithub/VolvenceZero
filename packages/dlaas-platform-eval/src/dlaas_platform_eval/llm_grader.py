"""LLM rubric judge for the DLaaS exam gate (debt #13 ACTIVE).

``LLMRubricGrader`` implements the :class:`~dlaas_platform_eval.grader.RubricGrader`
protocol with a real LLM judge against an OpenAI-compatible
``/chat/completions`` endpoint. Design constraints:

* ``RubricGrader.grade`` is **sync** by contract, while the platform's
  shared LLM client (``lifeform_core.OpenAiCompatJsonClient``) is
  async-only and lives in a wheel this package must not depend on.
  The network call here is therefore a synchronous stdlib
  ``urllib.request`` POST — no new third-party dependency, no event
  loop entanglement. Route handlers run it via ``asyncio.to_thread``.
* **Fail loudly**: malformed LLM output (non-JSON, missing criteria,
  wrong shape) raises :class:`GraderResponseError`. There is NO silent
  0.5 fallback inside this grader — callers decide how the error
  surfaces (the exam-run handlers turn it into a 502 ``grader_error``
  and mark the run failed).
* **R12 / OA-1**: grading is a readout. The grader receives
  (rubric, ai_response, reference_answer) and returns a score
  breakdown; nothing here writes to any kernel owner or learning
  state.

Env contract (resolved by :func:`resolve_eval_llm_config`):

================== =============================== =================
Eval-specific      Platform-wide fallback          Default
================== =============================== =================
EVAL_LLM_BASE_URL  PROTOCOL_LLM_BASE_URL           (required)
EVAL_LLM_API_KEY   PROTOCOL_LLM_API_KEY            (required)
EVAL_LLM_MODEL     PROTOCOL_LLM_MODEL              (required)
EVAL_LLM_TIMEOUT_S PROTOCOL_LLM_TIMEOUT_SECONDS    60
================== =============================== =================

Precedence: each ``EVAL_LLM_*`` var overrides its ``PROTOCOL_LLM_*``
sibling individually. Unlike ``lifeform_service.openai_compat_client``,
this wheel has no provider-preset table (presets are a
lifeform-service deployment concern), so ``base_url`` and ``model``
must be explicit. When the trio (base_url, api_key, model) is
incomplete the platform stays on the fail-closed
``DefaultRubricGrader`` (which logs its loud warning) — automation
can then never grant a license, by design.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from dlaas_platform_contracts import RubricEntry

from dlaas_platform_eval.grader import (
    DefaultRubricGrader,
    GradedSubmission,
    RubricGrader,
)
from dlaas_platform_eval.prompts import (
    GRADER_SYSTEM_PROMPT,
    GRADER_USER_TEMPLATE,
)

_LOG = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 60.0


class EvalLLMError(RuntimeError):
    """Base class for typed eval-LLM failures (never swallowed)."""


class GraderResponseError(EvalLLMError):
    """The LLM judge call failed or returned contract-violating output."""


class QuestionGenerationError(EvalLLMError):
    """The question-generation call failed or returned bad output."""


@dataclass(frozen=True)
class EvalLLMConfig:
    """Resolved OpenAI-compatible endpoint config for the eval gate."""

    base_url: str
    api_key: str
    model: str
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


def _env(name: str, fallback_name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if value:
        return value
    return (os.environ.get(fallback_name) or "").strip()


def resolve_eval_llm_config() -> EvalLLMConfig | None:
    """Resolve the eval LLM endpoint from env, or ``None``.

    ``EVAL_LLM_*`` takes precedence per-variable over the platform-wide
    ``PROTOCOL_LLM_*`` convention. Returns ``None`` (not an error) when
    any of base_url / api_key / model is missing — callers fall back to
    the fail-closed :class:`DefaultRubricGrader` or answer 503.
    """

    base_url = _env("EVAL_LLM_BASE_URL", "PROTOCOL_LLM_BASE_URL")
    api_key = _env("EVAL_LLM_API_KEY", "PROTOCOL_LLM_API_KEY")
    model = _env("EVAL_LLM_MODEL", "PROTOCOL_LLM_MODEL")
    if not (base_url and api_key and model):
        return None
    timeout_raw = _env("EVAL_LLM_TIMEOUT_S", "PROTOCOL_LLM_TIMEOUT_SECONDS")
    try:
        timeout = float(timeout_raw) if timeout_raw else DEFAULT_TIMEOUT_SECONDS
    except ValueError as exc:
        raise ValueError(
            f"EVAL_LLM_TIMEOUT_S must be numeric, got {timeout_raw!r}"
        ) from exc
    return EvalLLMConfig(
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        model=model,
        timeout_seconds=timeout,
    )


# Transport signature: (config, system_prompt=..., user_prompt=...) -> str
# (the assistant message content). Tests inject fakes; production uses
# ``chat_completion_text`` below.
LLMTransport = Callable[..., str]


def chat_completion_text(
    config: EvalLLMConfig, *, system_prompt: str, user_prompt: str
) -> str:
    """Synchronous OpenAI-compatible ``/chat/completions`` call.

    Stdlib-only (``urllib.request``) so the sync ``RubricGrader.grade``
    contract holds without a new dependency or a nested event loop.

    Raises:
        EvalLLMError: on HTTP / network / timeout failure or a
            response body that is not a chat-completions envelope.
    """

    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        f"{config.base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request, timeout=config.timeout_seconds
        ) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise EvalLLMError(
            f"eval LLM HTTP {exc.code} from {config.base_url}: {detail}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise EvalLLMError(
            f"eval LLM unreachable at {config.base_url}: {exc}"
        ) from exc
    try:
        envelope = json.loads(body)
        content = envelope["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise EvalLLMError(
            f"eval LLM returned a non-chat-completions body: {body[:500]!r}"
        ) from exc
    if not isinstance(content, str) or not content.strip():
        raise EvalLLMError("eval LLM returned empty message content")
    return content


def parse_strict_json_object(content: str) -> Mapping[str, Any]:
    """Parse the LLM message content as one JSON object.

    Tolerates a single markdown code fence around the object (a common
    chat-model framing) but nothing else — any other deviation raises.
    """

    text = content.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1 and text.endswith("```"):
            text = text[first_newline + 1 : -3].strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise json.JSONDecodeError("top-level JSON must be an object", text, 0)
    return data


class LLMRubricGrader:
    """Rubric judge backed by an OpenAI-compatible chat endpoint.

    Satisfies the :class:`RubricGrader` protocol. Per grade call it
    renders the centralized grader prompt (rubric criteria + reference
    answer + AI response), requests strict JSON
    ``{"scores": [{"criterion", "score", "justification"}]}``, and
    validates full criteria coverage with scores clamped to
    ``[0, max_score]``.

    R12 guard: pure readout — (rubric, response, reference) in, score
    breakdown out. No kernel owner is read or written.
    """

    def __init__(
        self,
        config: EvalLLMConfig,
        *,
        transport: LLMTransport | None = None,
    ) -> None:
        self._config = config
        # ``None`` means "resolve chat_completion_text at call time" so
        # tests can monkeypatch the module-level function.
        self._transport = transport
        _LOG.info(
            "LLMRubricGrader configured (model=%s, base_url=%s)",
            config.model,
            config.base_url,
        )

    @property
    def grader_label(self) -> str:
        return f"llm:{self._config.model}"

    def grade(
        self,
        *,
        rubric: tuple[RubricEntry, ...],
        ai_response: str,
        reference_answer: str,
    ) -> GradedSubmission:
        if not rubric:
            return GradedSubmission(weighted_score=0.0, rubric_breakdown=())
        user_prompt = GRADER_USER_TEMPLATE.format(
            rubric_json=json.dumps(
                [entry.to_json() for entry in rubric],
                ensure_ascii=False,
                indent=2,
            ),
            reference_answer=reference_answer or "(none provided)",
            ai_response=ai_response,
        )
        transport = self._transport or chat_completion_text
        try:
            content = transport(
                self._config,
                system_prompt=GRADER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except GraderResponseError:
            raise
        except EvalLLMError as exc:
            raise GraderResponseError(str(exc)) from exc
        return self._parse_graded_submission(rubric, content)

    def _parse_graded_submission(
        self, rubric: tuple[RubricEntry, ...], content: str
    ) -> GradedSubmission:
        try:
            data = parse_strict_json_object(content)
        except json.JSONDecodeError as exc:
            raise GraderResponseError(
                f"grader LLM output is not valid JSON: {exc}; "
                f"content={content[:300]!r}"
            ) from exc
        raw_scores = data.get("scores")
        if not isinstance(raw_scores, list):
            raise GraderResponseError(
                f"grader LLM output missing 'scores' list: {content[:300]!r}"
            )
        by_criterion: dict[str, Mapping[str, Any]] = {}
        for item in raw_scores:
            if not isinstance(item, Mapping):
                raise GraderResponseError(
                    f"grader LLM score entry is not an object: {item!r}"
                )
            criterion = str(item.get("criterion", "") or "")
            if not criterion:
                raise GraderResponseError(
                    f"grader LLM score entry missing 'criterion': {item!r}"
                )
            by_criterion[criterion] = item

        breakdown: list[dict[str, Any]] = []
        weighted_sum = 0.0
        total_weight = 0.0
        for entry in rubric:
            item = by_criterion.get(entry.criterion)
            if item is None:
                raise GraderResponseError(
                    f"grader LLM omitted rubric criterion "
                    f"{entry.criterion!r}; got {sorted(by_criterion)}"
                )
            try:
                raw_score = float(item.get("score"))
            except (TypeError, ValueError) as exc:
                raise GraderResponseError(
                    f"grader LLM score for {entry.criterion!r} is not "
                    f"numeric: {item.get('score')!r}"
                ) from exc
            score = min(max(raw_score, 0.0), entry.max_score)
            breakdown.append(
                {
                    "criterion": entry.criterion,
                    "score": score,
                    "max_score": entry.max_score,
                    "weight": entry.weight,
                    "justification": str(item.get("justification", "") or ""),
                    "grader_label": self.grader_label,
                }
            )
            if entry.max_score > 0:
                weighted_sum += entry.weight * (score / entry.max_score)
                total_weight += entry.weight
        weighted_score = (
            weighted_sum / total_weight if total_weight > 0 else 0.0
        )
        return GradedSubmission(
            weighted_score=weighted_score,
            rubric_breakdown=tuple(breakdown),
        )


def build_grader_from_env() -> RubricGrader:
    """Public factory: the eval-gate grader for this deployment.

    Returns :class:`LLMRubricGrader` when the env trio
    (``EVAL_LLM_BASE_URL`` / ``EVAL_LLM_API_KEY`` / ``EVAL_LLM_MODEL``,
    each falling back to its ``PROTOCOL_LLM_*`` sibling) is complete;
    otherwise the fail-closed :class:`DefaultRubricGrader`, which logs
    its loud warning and never grants licenses on its own.

    This is the seam other wheels (cultivation, interview) use to get
    the same judge configuration.
    """

    config = resolve_eval_llm_config()
    if config is None:
        return DefaultRubricGrader()
    return LLMRubricGrader(config)


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "EvalLLMConfig",
    "EvalLLMError",
    "GraderResponseError",
    "LLMRubricGrader",
    "QuestionGenerationError",
    "build_grader_from_env",
    "chat_completion_text",
    "parse_strict_json_object",
    "resolve_eval_llm_config",
]
