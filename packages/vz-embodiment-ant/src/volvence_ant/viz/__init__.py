"""Digital-ant demonstrations (Workstream G)."""

from __future__ import annotations

from volvence_ant.viz.bio_overlay import BioOverlayReport, build_bio_overlays
from volvence_ant.viz.colony_theater import (
    ColonyTheaterReport,
    TheaterAntFrame,
    TheaterArmReplay,
    TheaterRoundFrame,
    run_colony_theater,
    write_colony_theater_html,
)
from volvence_ant.viz.dashboard import LiveAntDashboard, write_replay_dashboard
from volvence_ant.viz.dual_substrate import (
    DualSubstrateReport,
    SubstrateProbe,
    run_dual_substrate_demo,
    run_formal_dual_substrate_demo,
)
from volvence_ant.viz.perturbation import (
    FormalPerturbationReport,
    PerturbationArm,
    PerturbationReport,
    run_perturbation_demo,
    run_formal_perturbation_demo,
)
from volvence_ant.viz.safety_demo import (
    E2ESafetyReport,
    E2ESafetyScenario,
    SafetyDemoReport,
    SafetyTick,
    run_e2e_safety_demo,
    run_safety_demo,
)

__all__ = [
    "ColonyTheaterReport",
    "TheaterAntFrame",
    "TheaterArmReplay",
    "TheaterRoundFrame",
    "run_colony_theater",
    "write_colony_theater_html",
    "DualSubstrateReport",
    "LiveAntDashboard",
    "write_replay_dashboard",
    "SubstrateProbe",
    "run_dual_substrate_demo",
    "run_formal_dual_substrate_demo",
    "FormalPerturbationReport",
    "PerturbationArm",
    "PerturbationReport",
    "run_perturbation_demo",
    "run_formal_perturbation_demo",
    "BioOverlayReport",
    "build_bio_overlays",
    "SafetyDemoReport",
    "E2ESafetyReport",
    "E2ESafetyScenario",
    "SafetyTick",
    "run_safety_demo",
    "run_e2e_safety_demo",
]
