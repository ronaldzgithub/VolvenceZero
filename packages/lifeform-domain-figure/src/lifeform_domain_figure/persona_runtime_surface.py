"""Runtime bridge for figure persona verification.

The ``verification/`` package is a readout/gate layer and must not
import kernel owner modules directly. This module is the vertical-side
composition surface that owns the kernel-facing imports needed by the
persona ablation harness.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

from volvence_zero.agent.response import ResponseContext
from volvence_zero.application.types import (
    ResponseAssemblySnapshot,
    ResponseMode,
    RiskBand,
)
from volvence_zero.substrate import (
    SyntheticOpenWeightResidualRuntime,
    TransformersOpenWeightResidualRuntime,
    default_persona_lora_pool,
)


def build_persona_verification_runtime(
    *,
    runtime: str,
    qwen_model_id: str,
    device: str,
) -> Any:
    if runtime == "synthetic":
        return SyntheticOpenWeightResidualRuntime()
    if runtime != "transformers":
        raise ValueError(f"unknown persona verification runtime {runtime!r}")
    return TransformersOpenWeightResidualRuntime(
        model_id=qwen_model_id,
        device=device,
    )


def baseline_response_assembly() -> Any:
    return ResponseAssemblySnapshot(
        regime_id="default",
        regime_name="default",
        abstract_action="answer",
        response_mode=ResponseMode.SUPPORT,
        answer_depth_limit="medium",
        citation_mode="optional",
        clarification_required=False,
        refer_out_required=False,
        ordering_plan=(),
        knowledge_briefs=(),
        case_briefs=(),
        playbook_ordering=(),
        required_disclaimers=(),
        required_disclaimer_phrases=(),
        control_code=(),
        control_scale=0.0,
        max_questions=0,
        prompt_residue_summary="",
        prompt_residue_ratio=0.0,
        knowledge_hit_count=0,
        case_hit_count=0,
        playbook_rule_count=0,
        risk_band=RiskBand.LOW,
        description="persona-verification minimal assembly",
    )


def default_response_context(prompt: str) -> Any:
    return ResponseContext(
        regime_id="default",
        regime_name="default",
        regime_switched=False,
        abstract_action="answer",
        alert_count=0,
        temporal_switch_gate=0.0,
        temporal_is_switching=False,
        reflection_lesson_count=0,
        reflection_tension_count=0,
        reflection_writeback_applied=False,
        primary_reflection_lesson=None,
        primary_reflection_tension=None,
        joint_schedule_action="continue",
        user_input=prompt,
    )


def persona_lora_record_id(figure_id: str) -> str:
    pool = default_persona_lora_pool()
    if not pool.has(figure_id):
        return "absent"
    record = pool.lookup(figure_id)
    return record.record_id


def ensure_pool_has_bundle_lora(*, bundle: Any) -> str:
    artifact = getattr(bundle, "lora", None)
    if artifact is None:
        return "absent"
    pool = default_persona_lora_pool()
    figure_id = artifact.figure_id
    bundle_id = bundle.bundle_id
    if pool.has(figure_id):
        existing = pool.lookup(figure_id)
        if existing.source_bundle_id == bundle_id:
            return existing.record_id
        pool.deregister(figure_id)
    return pool.register(
        figure_id=figure_id,
        source_bundle_id=bundle_id,
        backend_id=artifact.backend_id,
        training_plan_hash=artifact.training_plan_hash,
        adapter_layers=artifact.adapter_layers,
        parameter_count=artifact.parameter_count,
        description=artifact.description,
        peft_checkpoint_dir=getattr(artifact, "peft_checkpoint_dir", ""),
    )


@contextlib.contextmanager
def temporarily_deregister_pool_record(*, figure_id: str) -> Iterator[None]:
    pool = default_persona_lora_pool()
    if not pool.has(figure_id):
        yield
        return
    cached = pool.lookup(figure_id)
    pool.deregister(figure_id)
    try:
        yield
    finally:
        pool.register(
            figure_id=cached.figure_id,
            source_bundle_id=cached.source_bundle_id,
            backend_id=cached.backend_id,
            training_plan_hash=cached.training_plan_hash,
            adapter_layers=cached.adapter_layers,
            parameter_count=cached.parameter_count,
            description=cached.description,
            peft_checkpoint_dir=getattr(cached, "peft_checkpoint_dir", ""),
        )


__all__ = [
    "baseline_response_assembly",
    "build_persona_verification_runtime",
    "default_response_context",
    "ensure_pool_has_bundle_lora",
    "persona_lora_record_id",
    "temporarily_deregister_pool_record",
]
