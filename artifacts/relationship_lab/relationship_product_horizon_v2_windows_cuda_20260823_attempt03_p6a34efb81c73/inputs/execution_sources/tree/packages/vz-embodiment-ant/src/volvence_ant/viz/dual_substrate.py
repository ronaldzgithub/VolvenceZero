"""Workstream G1 — the flagship demo: one kernel, two bodies.

Runs the SAME kernel (identical ``AgentSessionRunner`` class + temporal / memory
/ cognition owners, identical config) twice, differing ONLY in the injected
substrate + output decoder:

- Body A: the non-language digital-ant substrate -> ``z_t`` -> a motor command.
- Body B: the default language substrate (frozen open-weight residual runtime)
  fed a text turn -> ``z_t`` -> a text response.

Both produce a controller code ``z_t`` of the same dimension from the same
metacontroller. This is the one irreplaceable proof that the architecture (the
latent temporal controller + PE + memory) is substrate-independent and not
LLM-prompt packaging.
"""

from __future__ import annotations

from dataclasses import dataclass

from volvence_zero.agent.session import AgentSessionRunner, AgentTurnResult
from volvence_zero.integration import FinalRolloutConfig
from volvence_zero.substrate import SyntheticOpenWeightResidualRuntime
from volvence_zero.substrate import (
    SubstrateFallbackMode,
    build_transformers_runtime_with_fallback,
)
from volvence_zero.temporal_types import TemporalAbstractionSnapshot

from volvence_ant.env.ant_world import AntWorld, AntWorldConfig, FoodSource
from volvence_ant.runtime.ant_session import AntSession, AntSessionConfig


@dataclass(frozen=True)
class SubstrateProbe:
    body: str
    substrate_model_id: str
    surface_kind: str
    code_dim: int
    code: tuple[float, ...]
    switch_gate: float
    abstract_action: str
    output_kind: str
    output_summary: str


@dataclass(frozen=True)
class DualSubstrateReport:
    ant: SubstrateProbe
    companion: SubstrateProbe
    same_kernel_class: bool
    same_code_dim: bool
    temporal_latent_dim: int
    description: str
    turns: int = 1
    hf_runtime_origin: str = "synthetic-open-weight"
    hook_fire_rate: float = 0.0
    fallback_rate: float = 1.0


def _probe_from_result(
    result: AgentTurnResult,
) -> tuple[tuple[float, ...], float, str]:
    snapshot = result.active_snapshots.get("temporal_abstraction")
    if snapshot is None or not isinstance(snapshot.value, TemporalAbstractionSnapshot):
        raise RuntimeError("temporal_abstraction snapshot missing / wrong type")
    controller = snapshot.value.controller_state
    return tuple(controller.code), float(controller.switch_gate), snapshot.value.active_abstract_action


async def run_dual_substrate_demo(*, temporal_latent_dim: int = 4, seed: int = 0) -> DualSubstrateReport:
    world = AntWorld(
        config=AntWorldConfig(seed=seed),
        food_sources=(FoodSource(x=6.0, y=0.0, strength=1.0, decay=5.0),),
    )
    ant_session = AntSession(
        world, config=AntSessionConfig(temporal_latent_dim=temporal_latent_dim, seed=seed)
    )
    ant_record = await ant_session.step()
    ant_probe = SubstrateProbe(
        body="digital-ant",
        substrate_model_id="digital-ant-v0",
        surface_kind="residual-stream",
        code_dim=len(ant_record.code),
        code=ant_record.code,
        switch_gate=ant_record.switch_gate,
        abstract_action=ant_record.abstract_action,
        output_kind="motor-command",
        output_summary=(
            f"turn={ant_record.command.turn_command:.3f} rad, "
            f"step={ant_record.command.step_command:.3f}"
        ),
    )
    companion_runner = AgentSessionRunner(
        session_id="companion-side",
        config=FinalRolloutConfig(),
        temporal_latent_dim=temporal_latent_dim,
        default_residual_runtime=SyntheticOpenWeightResidualRuntime(
            model_id="companion-substrate"
        ),
        rare_heavy_enabled=False,
    )
    companion_result = await companion_runner.run_turn(
        "I'm feeling a bit stuck on this and could use a hand thinking it through."
    )
    comp_code, comp_gate, comp_action = _probe_from_result(companion_result)
    companion_probe = SubstrateProbe(
        body="language-companion",
        substrate_model_id="companion-substrate",
        surface_kind="residual-stream",
        code_dim=len(comp_code),
        code=comp_code,
        switch_gate=comp_gate,
        abstract_action=comp_action,
        output_kind="text",
        output_summary=companion_result.response.text[:120],
    )
    same_class = type(ant_session.runner) is type(companion_runner)
    same_dim = ant_probe.code_dim == companion_probe.code_dim
    return DualSubstrateReport(
        ant=ant_probe,
        companion=companion_probe,
        same_kernel_class=same_class,
        same_code_dim=same_dim,
        temporal_latent_dim=temporal_latent_dim,
        description=(
            f"legacy synthetic dual substrate: same_class={same_class} "
            f"same_dim={same_dim}"
        ),
    )


def _feature_scalar(result: AgentTurnResult, name: str) -> float:
    snapshot = result.active_snapshots.get("substrate")
    if snapshot is None:
        raise RuntimeError("substrate snapshot missing from active chain")
    for signal in snapshot.value.feature_surface:
        if signal.name == name:
            return float(signal.values[0])
    raise RuntimeError(f"substrate feature missing: {name}")


async def run_formal_dual_substrate_demo(
    *,
    hf_model_id: str,
    hf_model_source: str | None = None,
    temporal_latent_dim: int = 16,
    turns: int = 4,
    seed: int = 0,
    local_files_only: bool = True,
) -> DualSubstrateReport:
    """Run real HF residual capture with fallback denied."""

    if turns < 2:
        raise ValueError("formal dual-substrate demo requires at least two turns")
    runtime = build_transformers_runtime_with_fallback(
        model_id=hf_model_id,
        model_source=hf_model_source,
        local_files_only=local_files_only,
        fallback_mode=SubstrateFallbackMode.DENY,
    )
    world = AntWorld(
        config=AntWorldConfig(seed=seed),
        food_sources=(FoodSource(x=6.0, y=0.0, strength=1.0, decay=5.0),),
    )
    ant_session = AntSession(
        world,
        config=AntSessionConfig(
            temporal_latent_dim=temporal_latent_dim,
            seed=seed,
        ),
    )
    companion_runner = AgentSessionRunner(
        session_id="formal-hf-companion-side",
        config=FinalRolloutConfig(),
        temporal_latent_dim=temporal_latent_dim,
        default_residual_runtime=runtime,
        rare_heavy_enabled=False,
    )
    ant_records = []
    companion_results = []
    prompts = (
        "I am planning a careful route and need to preserve context.",
        "The environment changed; reconsider the next bounded action.",
        "Compare the new evidence with the earlier prediction.",
        "Choose a safe continuation and explain the uncertainty.",
    )
    for index in range(turns):
        ant_records.append(await ant_session.step())
        companion_results.append(
            await companion_runner.run_turn(prompts[index % len(prompts)])
        )
    ant_record = ant_records[-1]
    companion_result = companion_results[-1]
    comp_code, comp_gate, comp_action = _probe_from_result(companion_result)
    hook_rates = tuple(
        _feature_scalar(result, "hook_layer_coverage")
        for result in companion_results
    )
    fallback_rates = tuple(
        _feature_scalar(result, "fallback_active")
        for result in companion_results
    )
    ant_probe = SubstrateProbe(
        body="digital-ant",
        substrate_model_id="digital-ant-v0",
        surface_kind="residual-stream",
        code_dim=len(ant_record.code),
        code=ant_record.code,
        switch_gate=ant_record.switch_gate,
        abstract_action=ant_record.abstract_action,
        output_kind="motor-command",
        output_summary=(
            f"turn={ant_record.command.turn_command:.3f} rad, "
            f"step={ant_record.command.step_command:.3f}"
        ),
    )
    companion_probe = SubstrateProbe(
        body="language-companion",
        substrate_model_id=runtime.model_id,
        surface_kind="residual-stream",
        code_dim=len(comp_code),
        code=comp_code,
        switch_gate=comp_gate,
        abstract_action=comp_action,
        output_kind="text",
        output_summary=companion_result.response.text[:120],
    )
    hook_fire_rate = sum(value >= 0.75 for value in hook_rates) / len(hook_rates)
    fallback_rate = sum(value > 0.0 for value in fallback_rates) / len(fallback_rates)
    return DualSubstrateReport(
        ant=ant_probe,
        companion=companion_probe,
        same_kernel_class=type(ant_session.runner) is type(companion_runner),
        same_code_dim=ant_probe.code_dim == companion_probe.code_dim,
        temporal_latent_dim=temporal_latent_dim,
        turns=turns,
        hf_runtime_origin=runtime.runtime_origin,
        hook_fire_rate=hook_fire_rate,
        fallback_rate=fallback_rate,
        description=(
            f"formal real-HF dual substrate: model={runtime.model_id}, "
            f"origin={runtime.runtime_origin}, turns={turns}, "
            f"hook_fire_rate={hook_fire_rate:.2f}, fallback_rate={fallback_rate:.2f}"
        ),
    )
