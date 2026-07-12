"""Capacity→gain ladder manifest (CP-24 / causal evidence).

This is a typed experiment *plan* generator, not a runner. It freezes the
factorial axes from the 12-month AGI uplift plan so GPU jobs can be scheduled
and audited without hand-maintained spreadsheets.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class CapacityLadderArm:
    arm_id: str
    temporal_latent_dim: int
    pe_critic_capacity: int
    cocoa_hidden: int
    backend_combo: str
    trace_turns: int
    substrate_label: str
    seed: int

    def __post_init__(self) -> None:
        if self.temporal_latent_dim < 3:
            raise ValueError("temporal_latent_dim must be >= 3")
        if self.pe_critic_capacity not in (1, 2, 4):
            raise ValueError("pe_critic_capacity must be one of 1, 2, 4")
        if self.cocoa_hidden not in (8, 32, 128):
            raise ValueError("cocoa_hidden must be one of 8, 32, 128")
        if self.trace_turns not in (500, 1000):
            raise ValueError("trace_turns must be one of 500, 1000")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


@dataclass(frozen=True)
class CapacityLadderManifest:
    schema_version: str
    arms: tuple[CapacityLadderArm, ...]
    description: str

    @property
    def arm_count(self) -> int:
        return len(self.arms)


DEFAULT_N_Z = (3, 16, 64, 256)
DEFAULT_PE_CRITIC_CAPACITY = (1, 2, 4)
DEFAULT_COCOA_HIDDEN = (8, 32, 128)
DEFAULT_BACKEND_COMBOS = (
    "runtime-only",
    "runtime+ssl",
    "runtime+ssl+internal-rl",
    "runtime+ssl+internal-rl+cms-torch",
)
DEFAULT_TRACE_TURNS = (500, 1000)
DEFAULT_SUBSTRATES = ("qwen-0.5b-screen", "qwen-1.5b-main", "qwen-7b-upper")
DEFAULT_SEEDS = (0, 1, 2)


def build_capacity_ladder_manifest(
    *,
    n_z_values: tuple[int, ...] = DEFAULT_N_Z,
    pe_critic_capacities: tuple[int, ...] = DEFAULT_PE_CRITIC_CAPACITY,
    cocoa_hidden_values: tuple[int, ...] = DEFAULT_COCOA_HIDDEN,
    backend_combos: tuple[str, ...] = DEFAULT_BACKEND_COMBOS,
    trace_turns: tuple[int, ...] = DEFAULT_TRACE_TURNS,
    substrates: tuple[str, ...] = DEFAULT_SUBSTRATES,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> CapacityLadderManifest:
    arms: list[CapacityLadderArm] = []
    for n_z, pe_cap, cocoa, combo, turns, substrate, seed in product(
        n_z_values,
        pe_critic_capacities,
        cocoa_hidden_values,
        backend_combos,
        trace_turns,
        substrates,
        seeds,
    ):
        arm_id = (
            f"nz{n_z}-pe{pe_cap}-cocoa{cocoa}-"
            f"{combo}-t{turns}-{substrate}-seed{seed}"
        )
        arms.append(
            CapacityLadderArm(
                arm_id=arm_id,
                temporal_latent_dim=n_z,
                pe_critic_capacity=pe_cap,
                cocoa_hidden=cocoa,
                backend_combo=combo,
                trace_turns=turns,
                substrate_label=substrate,
                seed=seed,
            )
        )
    return CapacityLadderManifest(
        schema_version="capacity-ladder-manifest.v1",
        arms=tuple(arms),
        description=(
            "CP-24 capacity-to-gain factorial manifest. Produces no claim by "
            "itself; each arm must be run on the evidence lane with matched "
            "substrate, judge, prompt/context budget, and artifact provenance."
        ),
    )


__all__ = [
    "CapacityLadderArm",
    "CapacityLadderManifest",
    "DEFAULT_BACKEND_COMBOS",
    "DEFAULT_COCOA_HIDDEN",
    "DEFAULT_N_Z",
    "DEFAULT_PE_CRITIC_CAPACITY",
    "DEFAULT_SEEDS",
    "DEFAULT_SUBSTRATES",
    "DEFAULT_TRACE_TURNS",
    "build_capacity_ladder_manifest",
]
