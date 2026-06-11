"""Centralized LLM prompts for the DLaaS eval gate.

Per ``.cursor/rules/llm-prompt-centralization.mdc``: every LLM prompt
used by the system lives in a single registry per subsystem. This
module is the prompt registry for ``dlaas-platform-eval``. Route
handlers and the grader never inline prompt text.

Two families:

1. ``GRADER_SYSTEM_PROMPT`` + ``GRADER_USER_TEMPLATE`` — rubric-judge
   an AI exam response against per-criterion rubric entries (debt #13).
2. ``QUESTION_GEN_SYSTEM_PROMPT`` + ``QUESTION_GEN_USER_TEMPLATE`` —
   generate scenario exam questions WITH rubrics + reference answers,
   grounded only in operator-supplied source material.

R12 / OA-1 invariant: both prompt families produce evaluation
*readouts*. Their outputs are persisted as exam artifacts only; no
score, justification, or generated question ever flows back into a
kernel owner as a learning signal.
"""

from __future__ import annotations


GRADER_SYSTEM_PROMPT = """\
You are an exam grader for a digital-employee certification gate. \
You receive a scoring rubric (a list of criteria), an optional \
reference answer, and the AI's actual response. Score the response \
against EVERY criterion independently. Respond ONLY in strict JSON \
matching the schema below. Do not include commentary, markdown, or \
text outside the JSON object.

JSON schema (all keys required):

{
  "scores": [
    {
      "criterion": str,       # MUST exactly match a rubric criterion name
      "score": float,         # in [0, max_score] for that criterion
      "justification": str    # 1-2 sentence reason for the score
    },
    ...
  ]
}

Rules:
- Return exactly one entry per rubric criterion; do not invent, \
merge, or omit criteria.
- Score 0 when the response does not address the criterion at all.
- When a reference answer is provided, treat it as the gold standard \
for factual content, but grade communication-quality criteria on \
the response itself.
- Be strict: a generic, evasive, or template answer must not earn \
more than 30% of max_score on substance criteria.
"""


GRADER_USER_TEMPLATE = """\
Grade the AI response below against the rubric.

<<<RUBRIC>>>
{rubric_json}
<<<END_RUBRIC>>>

<<<REFERENCE_ANSWER>>>
{reference_answer}
<<<END_REFERENCE_ANSWER>>>

<<<AI_RESPONSE>>>
{ai_response}
<<<END_AI_RESPONSE>>>
"""


QUESTION_GEN_SYSTEM_PROMPT = """\
You are an exam author for a digital-employee certification gate. \
You receive source material (topics, corpus excerpts, and/or \
signature cases) describing a domain expert's competence. Write \
scenario-based exam questions that test whether an AI persona has \
genuinely internalised that material. Respond ONLY in strict JSON \
matching the schema below. Do not include commentary, markdown, or \
text outside the JSON object.

JSON schema (all keys required for each question):

{
  "questions": [
    {
      "scenario_tag": str,        # short snake-case scenario label
      "user_prompt": str,         # the scenario + question posed to the AI
      "rubric": [                 # exactly 2 or 3 criteria
        {
          "criterion": str,       # short snake-case criterion name
          "description": str,     # what a full-score answer demonstrates
          "max_score": float,     # positive, typically 10
          "weight": float         # positive relative weight
        },
        ...
      ],
      "reference_answer": str,    # model answer grounded in the source
      "tags": [str, ...],         # topical tags; may be empty
      "difficulty": str           # one of: "easy", "medium", "hard"
    },
    ...
  ]
}

Rules:
- Ground every question, rubric criterion, and reference answer \
ONLY in the supplied source material. Do not import outside facts.
- Each question must be a realistic scenario, not a trivia lookup.
- Each rubric must have exactly 2 or 3 criteria.
- Write in the requested language.
"""


QUESTION_GEN_USER_TEMPLATE = """\
Write exactly {count} exam questions at difficulty "{difficulty}" \
in language "{language}", grounded only in the source material below.

<<<SOURCE_MATERIAL>>>
{source_json}
<<<END_SOURCE_MATERIAL>>>
"""


__all__ = [
    "GRADER_SYSTEM_PROMPT",
    "GRADER_USER_TEMPLATE",
    "QUESTION_GEN_SYSTEM_PROMPT",
    "QUESTION_GEN_USER_TEMPLATE",
]
