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
    regime_switched: bool
    abstract_action: str | None
    alert_count: int
    retrieved_memory_count: int
    temporal_switch_gate: float
    temporal_is_switching: bool
    reflection_lesson_count: int
    reflection_tension_count: int
    reflection_writeback_applied: bool
    primary_reflection_lesson: str | None
    primary_reflection_tension: str | None
    joint_schedule_action: str
    user_input: str = ""
    retrieved_memories: tuple[str, ...] = ()
    controller_description: str = ""


class ResponseSynthesizer:
    """Expression-layer synthesizer over structured runtime state.

    Base class uses fixed templates. Subclass ``LLMResponseSynthesizer``
    replaces templates with real LLM generation.
    """

    def synthesize(
        self,
        *,
        context: ResponseContext,
    ) -> AgentResponse:
        lesson_hint_map = {
            "promote_high_signal_memories": "I want to keep hold of the strongest signals rather than restart from scratch.",
            "reinforce_recent_high_credit_beliefs": "I want to lean on the parts of the interaction that have already proven stable.",
            "adjust_track_priority_from_session_feedback": "I want to rebalance between the task and the relationship instead of over-favoring one side.",
            "rebalance_temporal_prior_toward_memory": "I want to use continuity and recalled context more strongly in how I respond.",
            "rebalance_temporal_prior_toward_reflection": "I want the slower reflective layer to shape this reply more directly.",
            "rebalance_temporal_prior_toward_residual": "I want to stay closer to the immediate task signal while keeping the frame coherent.",
            "increase_controller_persistence_for_continuity": "I want to keep the same internal frame steady for longer instead of jumping too quickly.",
            "reduce_controller_persistence_for_faster_recovery": "I want to stay flexible enough to recover if the current frame is not the right one.",
            "allow_controller_switch_when_context_shifts": "I am allowing myself to change internal stance because the context has shifted.",
            "hold_controller_before_switching": "I want to hold the current stance a bit longer before I switch tracks.",
            "respect_metacontroller_runtime_guard": "I am keeping the response conservative because an internal guard was triggered.",
            "keep_controller_guard_signal_in_background": "I am keeping a background check on the controller while still moving forward.",
            "review_tension_before_auto_writeback": "I do not want to smooth over the remaining tension too quickly.",
        }
        tension_hint_map = {
            "cross_track_tension_high": "There is a strong mismatch between task pressure and relational stability right now.",
            "cross_track_alignment_drift": "I can feel some drift between the task frame and the relational frame.",
            "self_track_pressure_dominant": "The relational or emotional side currently needs more weight than the task side.",
            "world_track_pressure_dominant": "The task side is currently pressing harder than the relational side.",
            "relationship_stability_soft_drop": "I do not want to assume continuity is fully stable yet.",
            "warmth_signal_thin": "I want to add more warmth instead of sounding mechanically efficient.",
            "task_signal_diffuse": "The task signal is still a bit diffuse, so I should narrow it carefully.",
        }

        transition_hint = ""
        if context.temporal_is_switching or context.temporal_switch_gate >= 0.65:
            transition_hint = " I am deliberately shifting the internal control path rather than answering on autopilot."
        elif context.joint_schedule_action == "ssl-only":
            transition_hint = " I am keeping the current frame stable long enough to consolidate it before widening."
        elif context.joint_schedule_action == "full-cycle":
            transition_hint = " I have already run a deeper internal cycle, so I can shape this reply more deliberately."

        regime_shift_hint = ""
        if context.regime_switched:
            regime_shift_hint = " I am also changing the interaction frame instead of forcing the previous one to fit."

        memory_hint = ""
        if context.retrieved_memory_count:
            memory_hint = f" I am carrying forward {context.retrieved_memory_count} retrieved memory cues."

        reflection_hint = ""
        if context.reflection_writeback_applied:
            reflection_hint = (
                f" I am also carrying forward {context.reflection_lesson_count} reflected lesson"
                f"{'' if context.reflection_lesson_count == 1 else 's'} from the slow loop."
            )
        elif context.reflection_lesson_count:
            reflection_hint = (
                f" I can already see {context.reflection_lesson_count} slow-loop lesson"
                f"{'' if context.reflection_lesson_count == 1 else 's'} shaping how I respond."
            )

        tension_hint = ""
        if context.primary_reflection_tension is not None:
            tension_hint = f" {tension_hint_map.get(context.primary_reflection_tension, 'I want to keep an eye on the tensions that are still open rather than collapsing too quickly.')}"

        lesson_hint = ""
        if context.primary_reflection_lesson is not None:
            lesson_hint = f" {lesson_hint_map.get(context.primary_reflection_lesson, 'I am letting the slower reflective layer shape the reply rather than treating it as a no-op.')}"

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
        elif context.regime_id == "acquaintance_building":
            text = (
                "I want to keep this warm and relational rather than treating it like a cold transaction. "
                "We can build clarity without losing that sense of connection."
            )
        elif context.regime_id == "casual_social":
            text = (
                "I can keep this steady, natural, and continuous rather than over-formalizing it too early. "
                "That gives us room to stay useful without losing flow."
            )
        else:
            text = (
                "I can stay with the current context and respond in a way that keeps both usefulness and continuity in view."
            )

        if context.alert_count:
            text += " I also notice some internal caution signals, so I will stay measured rather than over-commit."
        text += transition_hint
        text += regime_shift_hint
        text += memory_hint
        text += reflection_hint
        text += lesson_hint
        text += tension_hint

        rationale_parts = [f"regime={context.regime_id or 'none'}"]
        if context.abstract_action:
            rationale_parts.append(f"temporal={context.abstract_action}")
        rationale_parts.append(f"switch_gate={context.temporal_switch_gate:.2f}")
        rationale_parts.append(f"joint={context.joint_schedule_action}")
        if context.alert_count:
            rationale_parts.append(f"alerts={context.alert_count}")
        if context.reflection_lesson_count:
            rationale_parts.append(f"reflection_lessons={context.reflection_lesson_count}")
        if context.primary_reflection_lesson is not None:
            rationale_parts.append(f"primary_lesson={context.primary_reflection_lesson}")
        if context.primary_reflection_tension is not None:
            rationale_parts.append(f"primary_tension={context.primary_reflection_tension}")
        if context.reflection_writeback_applied:
            rationale_parts.append("reflection_writeback=applied")
        rationale = ", ".join(rationale_parts)

        return AgentResponse(
            text=text,
            regime_id=context.regime_id,
            abstract_action=context.abstract_action,
            rationale=f"Synthesized from {context.regime_name}; {rationale}.",
        )


class LLMResponseSynthesizer(ResponseSynthesizer):
    """Expression-layer synthesizer backed by a real LLM.

    Uses ``OpenWeightResidualRuntime.generate()`` to produce text,
    with system prompt assembled from live cognitive state via
    ``volvence_zero.agent.prompts.build_system_prompt``.
    """

    def __init__(
        self,
        *,
        runtime: object,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> None:
        self._runtime = runtime
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature

    def synthesize(
        self,
        *,
        context: ResponseContext,
    ) -> AgentResponse:
        from volvence_zero.agent.prompts import build_system_prompt

        system_prompt = build_system_prompt(
            context=context,
            retrieved_memories=context.retrieved_memories,
            controller_description=context.controller_description,
        )

        user_input = context.user_input
        if not user_input:
            return super().synthesize(context=context)

        control_params: tuple[float, ...] = ()
        control_scale = 0.0

        result = self._runtime.generate(
            prompt=user_input,
            system_context=system_prompt,
            max_new_tokens=self._max_new_tokens,
            temperature=self._temperature,
            control_parameters=control_params,
            control_scale=control_scale,
        )

        generated_text = result.text.strip()
        if not generated_text:
            return super().synthesize(context=context)

        rationale_parts = [
            f"regime={context.regime_id or 'none'}",
            f"model={self._runtime.model_id}",
            f"tokens={result.token_count}",
        ]
        if context.abstract_action:
            rationale_parts.append(f"temporal={context.abstract_action}")
        rationale_parts.append(f"switch_gate={context.temporal_switch_gate:.2f}")

        return AgentResponse(
            text=generated_text,
            regime_id=context.regime_id,
            abstract_action=context.abstract_action,
            rationale=f"LLM generated; {', '.join(rationale_parts)}.",
        )
