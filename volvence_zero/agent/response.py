from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from volvence_zero.runtime import Snapshot


@dataclass(frozen=True)
class AgentResponse:
    text: str
    regime_id: str | None
    abstract_action: str | None
    rationale: str


class ResponseSynthesizer:
    """Expression-layer synthesizer over structured runtime state."""

    def synthesize(
        self,
        *,
        user_input: str,
        active_snapshots: dict[str, Snapshot[Any]],
        shadow_snapshots: dict[str, Snapshot[Any]],
    ) -> AgentResponse:
        regime_snapshot = active_snapshots.get("regime") or shadow_snapshots.get("regime")
        temporal_snapshot = active_snapshots.get("temporal_abstraction") or shadow_snapshots.get(
            "temporal_abstraction"
        )
        evaluation_snapshot = active_snapshots.get("evaluation")
        memory_snapshot = active_snapshots.get("memory")

        regime_id = None
        regime_name = "current context"
        if regime_snapshot is not None and hasattr(regime_snapshot.value, "active_regime"):
            regime_id = regime_snapshot.value.active_regime.regime_id
            regime_name = regime_snapshot.value.active_regime.name

        abstract_action = None
        if temporal_snapshot is not None and hasattr(temporal_snapshot.value, "active_abstract_action"):
            abstract_action = temporal_snapshot.value.active_abstract_action

        alerts = ()
        if evaluation_snapshot is not None and hasattr(evaluation_snapshot.value, "alerts"):
            alerts = evaluation_snapshot.value.alerts

        memory_hint = ""
        if memory_snapshot is not None and hasattr(memory_snapshot.value, "retrieved_entries"):
            entries = memory_snapshot.value.retrieved_entries
            if entries:
                memory_hint = f" I am carrying forward {len(entries)} retrieved memory cues."

        if regime_id == "repair_and_deescalation":
            text = (
                "I want to slow this down a little and make sure I respond in a steady, repairing way. "
                "We can handle the immediate issue, but I want to keep the interaction safe and grounded."
            )
        elif regime_id == "emotional_support":
            text = (
                "I am hearing emotional weight in this, so I want to stay supportive first and not rush past it. "
                "We can still move toward something useful together."
            )
        elif regime_id == "problem_solving":
            text = (
                "I see a concrete problem-solving path here. "
                "I can help structure the next steps clearly and keep the solution actionable."
            )
        elif regime_id == "guided_exploration":
            text = (
                "This feels like a place for guided exploration rather than a rushed answer. "
                "I can help us narrow the space step by step."
            )
        else:
            text = (
                "I can stay with the current context and respond in a way that keeps both usefulness and continuity in view."
            )

        if alerts:
            text += " I also notice some internal caution signals, so I will stay measured rather than over-commit."
        text += memory_hint

        rationale_parts = [f"regime={regime_id or 'none'}"]
        if abstract_action:
            rationale_parts.append(f"temporal={abstract_action}")
        if alerts:
            rationale_parts.append(f"alerts={len(alerts)}")
        rationale = ", ".join(rationale_parts)

        return AgentResponse(
            text=text,
            regime_id=regime_id,
            abstract_action=abstract_action,
            rationale=f"Synthesized from {regime_name}; {rationale}.",
        )
