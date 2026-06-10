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


__all__ = [
    "MENTOR_INTAKE_CLASSIFIER_SYSTEM_PROMPT",
    "MENTOR_INTAKE_CLASSIFIER_USER_TEMPLATE",
]
