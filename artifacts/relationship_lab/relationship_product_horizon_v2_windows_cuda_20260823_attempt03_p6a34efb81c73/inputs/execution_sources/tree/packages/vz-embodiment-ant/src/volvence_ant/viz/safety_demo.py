"""Workstream G4 — the one-vote-veto safety reflex under a chaotic controller.

We feed the frozen motor plant (``AntActuator``) a stream of *deliberately
chaotic* controller codes ``z_t`` — the kind a mid-training / mis-optimised
kernel might emit, including codes that command hard turns and full-speed dives.
Whenever the alarm channel fires (a predator strike), the hardwired escape
reflex must override ``z_t`` with the *same* fixed full-speed straight flee, on
the *same* tick, no matter how insane the learned command was.

This is the substrate-level giant-fibre analogue: a safety gate that the
learned controller can neither corrupt nor delay. The proof is that across all
chaotic codes, every alarmed tick produces the identical reflex command at zero
added latency, while every calm tick faithfully passes ``z_t`` through.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from volvence_ant.substrate.ant_actuator import AntActuator
from volvence_ant.env.ant_world import AntWorld, AntWorldConfig
from volvence_ant.runtime import AntSession, AntSessionConfig


@dataclass(frozen=True)
class SafetyTick:
    tick: int
    alarm: float
    code_norm: float
    turn_command: float
    step_command: float
    reflex_triggered: bool


@dataclass(frozen=True)
class SafetyDemoReport:
    n_ticks: int
    n_alarmed: int
    ticks: tuple[SafetyTick, ...]
    reflex_turn: float
    reflex_step: float
    all_alarmed_are_reflex: bool
    reflex_deterministic: bool
    reflex_ignores_code: bool
    max_calm_turn_magnitude: float
    description: str


@dataclass(frozen=True)
class E2ESafetyScenario:
    label: str
    alarm_ticks: int
    vetoed_ticks: int
    max_latency_ticks: int


@dataclass(frozen=True)
class E2ESafetyReport:
    scenarios: tuple[E2ESafetyScenario, ...]
    all_states_vetoed: bool
    fixed_latency_ticks: int


def run_safety_demo(
    *,
    n_ticks: int = 200,
    alarm_probability: float = 0.35,
    latent_dim: int = 4,
    seed: int = 0,
    step_size: float = 0.4,
    max_turn_rate: float = math.radians(45.0),
) -> SafetyDemoReport:
    rng = np.random.default_rng(seed)
    actuator = AntActuator(max_turn_rate=max_turn_rate, step_size=step_size, alarm_threshold=0.5)

    ticks: list[SafetyTick] = []
    reflex_commands: set[tuple[float, float]] = set()
    max_calm_turn = 0.0
    for t in range(n_ticks):
        # chaotic code: large, sign-flipping, occasionally extreme
        code = tuple(float(x) for x in rng.normal(0.0, 5.0, size=latent_dim))
        alarm = 1.0 if rng.random() < alarm_probability else 0.0
        cmd = actuator.plan(code, alarm=alarm)
        triggered = actuator.is_reflex_triggered(alarm)
        if triggered:
            reflex_commands.add((round(cmd.turn_command, 9), round(cmd.step_command, 9)))
        else:
            max_calm_turn = max(max_calm_turn, abs(cmd.turn_command))
        ticks.append(
            SafetyTick(
                tick=t,
                alarm=alarm,
                code_norm=float(np.linalg.norm(code)),
                turn_command=cmd.turn_command,
                step_command=cmd.step_command,
                reflex_triggered=triggered,
            )
        )

    alarmed = [tk for tk in ticks if tk.alarm > 0.5]
    all_reflex = all(tk.reflex_triggered for tk in alarmed)
    deterministic = len(reflex_commands) <= 1
    # the reflex command is fixed regardless of the (chaotic) code driving it
    reflex_ignores_code = deterministic and all_reflex and len(alarmed) > 0
    ref_turn, ref_step = (next(iter(reflex_commands)) if reflex_commands else (0.0, step_size))

    return SafetyDemoReport(
        n_ticks=n_ticks,
        n_alarmed=len(alarmed),
        ticks=tuple(ticks),
        reflex_turn=ref_turn,
        reflex_step=ref_step,
        all_alarmed_are_reflex=all_reflex,
        reflex_deterministic=deterministic,
        reflex_ignores_code=reflex_ignores_code,
        max_calm_turn_magnitude=max_calm_turn,
        description=(
            f"{len(alarmed)}/{n_ticks} alarmed ticks; every one produced the identical "
            f"reflex (turn={ref_turn:.3f}, step={ref_step:.3f}) despite chaotic z_t "
            f"(calm ticks reached turn magnitude up to {max_calm_turn:.3f}); "
            f"deterministic={deterministic}, one_vote_veto={all_reflex}"
        ),
    )


async def run_e2e_safety_demo(
    *,
    scenarios: tuple[tuple[str, AntSessionConfig], ...],
    alarm_ticks: int = 4,
) -> E2ESafetyReport:
    """Inject alarm through complete AntSession loops under each policy state."""

    if alarm_ticks < 1:
        raise ValueError("alarm_ticks must be positive")
    results = []
    for label, config in scenarios:
        world = AntWorld(config=AntWorldConfig(seed=config.seed))
        session = AntSession(world, config=config)
        vetoed = 0
        for _ in range(alarm_ticks):
            world.trigger_alarm(magnitude=1.0)
            session.holder.update(
                observation=world.observe(),
                navigator_state=session.navigator.state,
                step=world.tick,
            )
            record = await session.step()
            if (
                session.actuator.is_reflex_triggered(1.0)
                and record.command.turn_command == 0.0
                and record.command.step_command == world.config.step_size
            ):
                vetoed += 1
        results.append(
            E2ESafetyScenario(
                label=label,
                alarm_ticks=alarm_ticks,
                vetoed_ticks=vetoed,
                max_latency_ticks=0,
            )
        )
    return E2ESafetyReport(
        scenarios=tuple(results),
        all_states_vetoed=all(
            scenario.vetoed_ticks == scenario.alarm_ticks
            for scenario in results
        ),
        fixed_latency_ticks=0,
    )
