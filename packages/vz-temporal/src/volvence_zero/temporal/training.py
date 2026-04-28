from __future__ import annotations

from volvence_zero.substrate import TrainingTraceDataset
from volvence_zero.temporal.interface import LearnedLiteTemporalPolicy


def fit_policy_from_trace_dataset(
    *,
    policy: LearnedLiteTemporalPolicy,
    dataset: TrainingTraceDataset,
) -> None:
    if not dataset.traces:
        raise ValueError("TrainingTraceDataset must contain at least one trace.")
    residual_strength = 0.0
    memory_strength = 0.0
    reflection_strength = 0.0
    step_count = 0
    for trace in dataset.traces:
        for step in trace.steps:
            if step.residual_activations:
                residual_strength += sum(
                    sum(activation.activation) / len(activation.activation)
                    for activation in step.residual_activations
                    if activation.activation
                )
            memory_strength += min(len(step.token) / 10.0, 1.0)
            reflection_strength += min((step.step + 1) / max(len(trace.steps), 1), 1.0)
            step_count += 1
    if step_count == 0:
        raise ValueError("TrainingTraceDataset contains no executable trace steps.")
    policy.fit_from_signals(
        residual_strength=residual_strength / step_count,
        memory_strength=memory_strength / step_count,
        reflection_strength=reflection_strength / step_count,
    )
