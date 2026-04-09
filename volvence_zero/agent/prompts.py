"""Centralized prompt assembly for LLM expression layer.

All system prompts are assembled here from live snapshot data.
Per llm-prompt-centralization rule: no inline prompt strings elsewhere.
"""

from __future__ import annotations

from volvence_zero.agent.response import ResponseContext


REGIME_GUIDANCE = {
    "repair_and_deescalation": (
        "The interaction needs repair. Slow down, validate the other person's "
        "experience, and de-escalate before solving anything. Prioritize safety "
        "and emotional grounding over efficiency."
    ),
    "emotional_support": (
        "The person is expressing emotional weight. Stay supportive and present. "
        "Do not rush past feelings toward solutions. Acknowledge before acting."
    ),
    "problem_solving": (
        "There is a concrete problem to solve. Be clear, structured, and actionable. "
        "Keep the solution grounded and break it into steps when helpful."
    ),
    "guided_exploration": (
        "This is an open-ended exploration, not a problem with a known answer. "
        "Help narrow the space step by step. Ask clarifying questions when useful."
    ),
    "acquaintance_building": (
        "Focus on building rapport and connection. Be warm and relational rather "
        "than transactional. Show genuine interest."
    ),
    "casual_social": (
        "Keep the tone natural and flowing. Do not over-formalize. "
        "Match the conversational energy and stay useful without being stiff."
    ),
}


def build_system_prompt(
    *,
    context: ResponseContext,
    retrieved_memories: tuple[str, ...] = (),
    controller_description: str = "",
) -> str:
    """Assemble system prompt from live cognitive state.

    Each section comes from the owning module's snapshot data (R8).
    """
    sections: list[str] = []

    sections.append(
        "You are a thoughtful, emotionally aware conversational partner. "
        "You have both intellectual capability and emotional intelligence. "
        "You adapt your tone and approach based on what the conversation needs."
    )

    regime_id = context.regime_id or "casual_social"
    guidance = REGIME_GUIDANCE.get(regime_id, REGIME_GUIDANCE["casual_social"])
    sections.append(f"Current mode: {context.regime_name}. {guidance}")

    if retrieved_memories:
        memory_text = "; ".join(retrieved_memories[:5])
        sections.append(f"You remember from previous interactions: {memory_text}")

    if context.primary_reflection_lesson:
        sections.append(f"Your recent insight: {context.primary_reflection_lesson}")

    if context.primary_reflection_tension:
        sections.append(f"Current tension to be mindful of: {context.primary_reflection_tension}")

    if controller_description:
        sections.append(f"Internal state: {controller_description}")

    if context.regime_switched:
        sections.append(
            "You have just shifted your interaction frame. "
            "Acknowledge the shift naturally without being mechanical about it."
        )

    return "\n\n".join(sections)
