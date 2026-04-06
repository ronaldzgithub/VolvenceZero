from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentResponse:
    text: str
    regime_id: str | None
    abstract_action: str | None
    rationale: str


@dataclass(frozen=True)
class ResponseContext:
    regime_id: str | None
    regime_name: str
    abstract_action: str | None
    alert_count: int
    retrieved_memory_count: int


class ResponseSynthesizer:
    """Expression-layer synthesizer over structured runtime state."""

    def synthesize(
        self,
        *,
        context: ResponseContext,
    ) -> AgentResponse:
        memory_hint = ""
        if context.retrieved_memory_count:
            memory_hint = f" I am carrying forward {context.retrieved_memory_count} retrieved memory cues."

        if context.regime_id == "repair_and_deescalation":
            text = (
                "I want to slow this down a little and make sure I respond in a steady, repairing way. "
                "We can handle the immediate issue, but I want to keep the interaction safe and grounded."
            )
        elif context.regime_id == "emotional_support":
            text = (
                "I am hearing emotional weight in this, so I want to stay supportive first and not rush past it. "
                "We can still move toward something useful together."
            )
        elif context.regime_id == "problem_solving":
            text = (
                "I see a concrete problem-solving path here. "
                "I can help structure the next steps clearly and keep the solution actionable."
            )
        elif context.regime_id == "guided_exploration":
            text = (
                "This feels like a place for guided exploration rather than a rushed answer. "
                "I can help us narrow the space step by step."
            )
        else:
            text = (
                "I can stay with the current context and respond in a way that keeps both usefulness and continuity in view."
            )

        if context.alert_count:
            text += " I also notice some internal caution signals, so I will stay measured rather than over-commit."
        text += memory_hint

        rationale_parts = [f"regime={context.regime_id or 'none'}"]
        if context.abstract_action:
            rationale_parts.append(f"temporal={context.abstract_action}")
        if context.alert_count:
            rationale_parts.append(f"alerts={context.alert_count}")
        rationale = ", ".join(rationale_parts)

        return AgentResponse(
            text=text,
            regime_id=context.regime_id,
            abstract_action=context.abstract_action,
            rationale=f"Synthesized from {context.regime_name}; {rationale}.",
        )
