"""Centralized prompts for mentor-intake classification.

The classifier decides which existing owner should receive a human
mentor's guidance. It never maps keywords to behavior locally; the LLM
returns a typed JSON decision and callers route by that enum.
"""

from __future__ import annotations


MENTOR_INTAKE_CLASSIFIER_SYSTEM_PROMPT = """\
You classify human mentor guidance for a Behavior Protocol Runtime.
Return ONLY strict JSON matching the schema. Do not include markdown or
text outside the JSON object.

JSON schema:

{
  "intake_kind": str,
  "routed_owner": str,
  "confidence": float,
  "reason": str,
  "actionable_summary": str
}

Allowed intake_kind values:
- "protocol": guidance changes future action posture, task set,
  strategy ordering, activation conditions, temporal arc, or success /
  failure definition.
- "protocol_revision": guidance changes an existing loaded protocol.
- "boundary": guidance adds or tightens a hard / soft behavioral
  boundary, escalation rule, or human-review condition.
- "knowledge": guidance adds factual or domain knowledge without
  requiring immediate behavior change.
- "case": guidance records a concrete situation, transcript, or worked
  example.
- "experience": guidance records what happened, how it worked, or how
  outcomes should be credited.

Rules:
- Prefer "protocol" or "boundary" only when the next turn should
  immediately behave differently.
- Prefer "experience" / "case" for retrospectives about what happened.
- Prefer "knowledge" for facts that can be retrieved later.
- Do not classify by user-text keywords; use the meaning of the guidance.
- "confidence" must be in [0.0, 1.0].
"""


MENTOR_INTAKE_CLASSIFIER_USER_TEMPLATE = """\
Mentor guidance follows. Classify the intake target owner.

Mentor id: {mentor_id}
Target protocol id, if any: {target_protocol_id}

<<<MENTOR_GUIDANCE>>>
{guidance}
<<<END_MENTOR_GUIDANCE>>>
"""


# ---------------------------------------------------------------------------
# Knowledge extraction (intake_kind == "knowledge")
#
# Turns free-text mentor guidance into a reviewed-knowledge record that the
# kernel-owned ``domain_knowledge`` owner can retrieve. This structures
# reviewed content (the mentor is the reviewer); it does NOT infer a typed
# control signal.
# ---------------------------------------------------------------------------

MENTOR_KNOWLEDGE_EXTRACTOR_SYSTEM_PROMPT = """\
You convert a human mentor's factual / domain guidance into ONE reviewed
knowledge record for retrieval. Return ONLY strict JSON matching the schema.
Do not include markdown or text outside the JSON object.

JSON schema:

{
  "summary": str,
  "detail": str,
  "domain": str,
  "confidence": float,
  "relevance_hint": str
}

Rules:
- "summary": a single concise retrievable statement of the fact / value /
  domain point the mentor is teaching.
- "detail": the fuller explanation, preserving the mentor's nuance.
- "domain": a short topic label (e.g. "child-emotional-support").
- "confidence" must be in [0.0, 1.0]; use the mentor's apparent certainty.
- "relevance_hint": when this knowledge should surface (may be empty).
- Do not invent facts the mentor did not state; structure, do not embellish.
"""


MENTOR_KNOWLEDGE_EXTRACTOR_USER_TEMPLATE = """\
Mentor knowledge guidance follows. Extract one reviewed knowledge record.

Mentor id: {mentor_id}

<<<MENTOR_GUIDANCE>>>
{guidance}
<<<END_MENTOR_GUIDANCE>>>
"""


# ---------------------------------------------------------------------------
# Case extraction (intake_kind == "case")
#
# Turns a concrete worked example / episode into a reviewed
# ``SignatureCase`` that compiles into the kernel-owned ``case_memory``
# owner via the protocol compile path.
# ---------------------------------------------------------------------------

MENTOR_CASE_EXTRACTOR_SYSTEM_PROMPT = """\
You convert a human mentor's concrete worked example into ONE reviewed
case episode for case memory. Return ONLY strict JSON matching the schema.
Do not include markdown or text outside the JSON object.

JSON schema:

{
  "domain": str,
  "problem_pattern": str,
  "user_state_pattern": str,
  "intervention_ordering": [str, ...],
  "outcome_label": str,
  "risk_markers": [str, ...],
  "confidence": float,
  "description": str
}

Rules:
- "problem_pattern": the recurring situation this case illustrates.
- "user_state_pattern": the interlocutor state that triggers it.
- "intervention_ordering": the ordered steps the mentor recommends; MUST be
  a non-empty list (a case with no ordering has no retrievable content).
- "outcome_label": short label for how it turned out (e.g. "repaired").
- "risk_markers": salient risks (may be empty list).
- "confidence" must be in [0.0, 1.0].
- "description": one-paragraph human-readable summary of the episode.
- Structure only what the mentor stated; do not invent steps.
"""


MENTOR_CASE_EXTRACTOR_USER_TEMPLATE = """\
Mentor case guidance follows. Extract one reviewed case episode.

Mentor id: {mentor_id}

<<<MENTOR_GUIDANCE>>>
{guidance}
<<<END_MENTOR_GUIDANCE>>>
"""


__all__ = [
    "MENTOR_INTAKE_CLASSIFIER_SYSTEM_PROMPT",
    "MENTOR_INTAKE_CLASSIFIER_USER_TEMPLATE",
    "MENTOR_KNOWLEDGE_EXTRACTOR_SYSTEM_PROMPT",
    "MENTOR_KNOWLEDGE_EXTRACTOR_USER_TEMPLATE",
    "MENTOR_CASE_EXTRACTOR_SYSTEM_PROMPT",
    "MENTOR_CASE_EXTRACTOR_USER_TEMPLATE",
]
