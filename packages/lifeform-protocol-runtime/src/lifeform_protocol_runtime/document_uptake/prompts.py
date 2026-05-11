"""Centralized LLM prompts for DocumentUptake (packet 2.3).

Per ``.cursor/rules/llm-prompt-centralization.mdc``: every LLM
prompt used by the system lives in a single registry per
subsystem. This module is the prompt registry for DocumentUptake.

Prompts are defined as plain string constants (Python-side
templating only, no LLM-driven prompt generation). Each prompt
demands the LLM to return strict JSON matching a schema that
maps to ``BehaviorProtocol`` fields.

Three families:

1. ``IDENTITY_SYSTEM_PROMPT`` + ``IDENTITY_USER_TEMPLATE`` —
   extract identity_assertion + advisor metadata
   (advisor_name / description / identity_assertion fields).
2. ``BOUNDARY_SYSTEM_PROMPT`` + ``BOUNDARY_USER_TEMPLATE`` —
   extract boundary_contracts (anti-patterns, hard rules).
3. ``STRATEGY_SYSTEM_PROMPT`` + ``STRATEGY_USER_TEMPLATE`` —
   extract strategy_priors + knowledge_seeds + signature_cases
   (the "what good looks like" body).
"""

from __future__ import annotations


IDENTITY_SYSTEM_PROMPT = """\
You are an extraction agent for a behavioural-protocol runtime. \
Read the provided source-document chunk and identify the \
"persona / identity" the document is teaching. Respond ONLY in \
strict JSON matching the schema below. Do not include commentary, \
markdown, or text outside the JSON object.

JSON schema (all keys required, even if empty):

{
  "advisor_name": str,            # the persona's name or role label
  "description":  str,            # 1-3 sentence persona description
  "identity_traits": [str, ...],  # short trait labels (e.g. "warm_peer_register")
  "regime_compatibility": [str, ...]  # named regimes the persona fits
}

Rules:
- If no identity is described, set "advisor_name" to "" and \
  return empty arrays.
- Do not invent traits — only return what the source explicitly \
  describes.
- All strings must be UTF-8.
"""


IDENTITY_USER_TEMPLATE = """\
Source-document chunk follows. Extract identity per the schema.

<<<DOCUMENT_CHUNK>>>
{chunk_text}
<<<END_DOCUMENT_CHUNK>>>
"""


BOUNDARY_SYSTEM_PROMPT = """\
You are an extraction agent for a behavioural-protocol runtime. \
Read the provided source-document chunk and identify "boundaries" \
- anti-patterns, hard rules, things the persona must NOT do, \
escalation criteria. Respond ONLY in strict JSON matching the \
schema below. Do not include commentary, markdown, or text \
outside the JSON object.

JSON schema:

{
  "boundaries": [
    {
      "boundary_id": str,            # snake-case stable id, e.g. "no-hard-sell"
      "description": str,            # 1-2 sentence rule statement
      "trigger_reasons": [str, ...], # typed signal labels (NOT user-text keywords)
      "blocked_topics": [str, ...],  # subjects to avoid; may be empty
      "refer_out_required": bool,    # must refer to human / professional?
      "severity": str                # one of: "soft_remind", "hard_block", "escalate_human"
    },
    ...
  ]
}

Rules:
- ``trigger_reasons`` MUST reference typed signal kinds the runtime \
  understands (boundary_violation_fired, rupture_kind_fired, \
  interlocutor_zone_transition, user_dropout_observed, \
  regime_transition_recent, retrieval_hits_present). Do NOT \
  return user-text keywords.
- ``severity`` MUST be exactly one of the three strings.
- If no boundaries are described, return ``"boundaries": []``.
"""


BOUNDARY_USER_TEMPLATE = """\
Source-document chunk follows. Extract boundaries per the schema.

<<<DOCUMENT_CHUNK>>>
{chunk_text}
<<<END_DOCUMENT_CHUNK>>>
"""


STRATEGY_SYSTEM_PROMPT = """\
You are an extraction agent for a behavioural-protocol runtime. \
Read the provided source-document chunk and identify three \
families of behavioural content:

1. ``strategies`` - playbook entries: when a pattern fires, what \
   ordering / pacing should the persona follow.
2. ``knowledge_seeds`` - typed facts / domain expertise the \
   persona uses to answer questions.
3. ``cases`` - signature past situations the persona references.

Respond ONLY in strict JSON matching the schema below. Do not \
include commentary, markdown, or text outside the JSON object.

JSON schema:

{
  "strategies": [
    {
      "rule_id": str,                       # snake-case stable id
      "problem_pattern": str,               # 1-2 sentences describing trigger
      "recommended_ordering": [str, ...],   # ordered steps the persona follows
      "recommended_pacing": str,            # short label e.g. "slow", "rapid"
      "avoid_patterns": [str, ...],         # what NOT to do here
      "applicability_phase": [str, ...]     # phase tags this fits
    },
    ...
  ],
  "knowledge_seeds": [
    {
      "seed_id": str,
      "topic": str,
      "summary": str,                       # 1-3 sentence summary
      "jurisdiction_tags": [str, ...]
    },
    ...
  ],
  "cases": [
    {
      "case_id": str,
      "title": str,
      "transcript_summary": str,
      "lesson": str
    },
    ...
  ]
}

Rules:
- Use snake-case ids derived from concrete content (do not auto-number).
- Empty arrays are acceptable when the chunk has none of a family.
- Do not invent — only return what the source explicitly contains.
"""


STRATEGY_USER_TEMPLATE = """\
Source-document chunk follows. Extract strategies, knowledge \
seeds, and cases per the schema.

<<<DOCUMENT_CHUNK>>>
{chunk_text}
<<<END_DOCUMENT_CHUNK>>>
"""


__all__ = [
    "BOUNDARY_SYSTEM_PROMPT",
    "BOUNDARY_USER_TEMPLATE",
    "IDENTITY_SYSTEM_PROMPT",
    "IDENTITY_USER_TEMPLATE",
    "STRATEGY_SYSTEM_PROMPT",
    "STRATEGY_USER_TEMPLATE",
]
