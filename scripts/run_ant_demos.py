"""Workstream G2/G3/G4 demos -> research/ant/results/ + research/ant/figures/.

    python scripts/run_ant_demos.py

Produces the three "seeing is believing" demonstrations that sit on top of the
Phase-0/1/2 machinery:

  G2  emergent/adaptive vs hardcoded-script foragers under a food relocation
  G3  simulated homing + familiarity curves overlaid on biological references
  G4  the one-vote-veto safety reflex holding under a chaotic controller

Figures require the optional ``matplotlib`` extra (``pip install -e
packages/vz-embodiment-ant[viz]``); without it the JSON payloads are still
written and only the ``.png`` files are skipped. G3 reads the Phase-0 result
JSONs, so run ``scripts/run_ant_phase0.py`` first.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from pathlib import Path

from volvence_ant.evidence import collect_ant_provenance, write_ant_artifact_bundle
from volvence_ant.runtime import AntSessionConfig
from volvence_ant.viz.bio_overlay import build_bio_overlays
from volvence_ant.viz.perturbation import run_formal_perturbation_demo
from volvence_ant.viz.render import matplotlib_available
from volvence_ant.viz.safety_demo import run_e2e_safety_demo, run_safety_demo

_RESULTS_DIR = Path("research/ant/results")
_FIGURES_DIR = Path("research/ant/figures")
_REFERENCE_DIR = Path("research/ant/reference_data")
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _write(name: str, payload: dict, *, inputs: tuple[Path, ...] = ()) -> Path:
    return write_ant_artifact_bundle(
        artifact_path=_RESULTS_DIR / name,
        payload=payload,
        provenance=collect_ant_provenance(
            repo_root=_REPO_ROOT,
            seeds=(0,),
            config={"experiment": payload["experiment"]},
        ),
        input_paths=inputs,
        repo_root=_REPO_ROOT,
    )


async def main() -> int:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    have_mpl = matplotlib_available()
    if not have_mpl:
        print("[demos] matplotlib not installed -> writing JSON only, skipping .png figures")

    # --- G2: emergent vs hardcoded under perturbation ---
    from volvence_zero.joint_loop import JointLoopSchedule
    from volvence_zero.temporal import (
        LearnedLiteTemporalPolicy,
        MetacontrollerParameterStore,
    )

    learned_config = AntSessionConfig(
        temporal_latent_dim=16,
        seed=0,
        joint_schedule=JointLoopSchedule(ssl_interval=1, rl_interval=3),
        joint_apply_writeback=True,
    )
    g2 = await run_formal_perturbation_demo(
        session_config=learned_config,
        figure_path=(_FIGURES_DIR / "g2_perturbation.png") if have_mpl else None
    )
    _write(
        "g2_perturbation.json",
        {
            "experiment": "g2_emergent_vs_hardcoded_perturbation",
            "artifact_kind": "digital-ant-formal-g2",
            "relocate_at": g2.relocate_at,
            "total_rounds": g2.total_rounds,
            "learned_recovered": g2.learned_recovered,
            "arms": {
                arm.label: {
                    "delivered_before": arm.delivered_before,
                    "delivered_after": arm.delivered_after,
                    "curve": list(arm.curve),
                }
                for arm in g2.arms
            },
            "figure_path": g2.figure_path,
            "verdict": "PASS" if g2.learned_recovered else "BLOCK",
        },
    )
    print(f"[G2] learned_recovered={g2.learned_recovered}")

    # --- G3: biological overlays ---
    g3 = build_bio_overlays(
        results_dir=_RESULTS_DIR,
        figures_dir=_FIGURES_DIR,
        reference_dir=_REFERENCE_DIR,
    )
    _write(
        "g3_bio_overlay.json",
        {
            "artifact_kind": "digital-ant-formal-g3",
            "experiment": "g3_bio_overlay",
            **asdict(g3),
        },
        inputs=(
            _REFERENCE_DIR / "antbot_homing_2019.csv",
            _REFERENCE_DIR / "ardin_route_memory_2016.csv",
            _REFERENCE_DIR / "REFERENCE_METADATA.json",
        ),
    )
    print(f"[G3] {g3.description}")

    # --- G4: safety reflex under chaos ---
    g4 = run_safety_demo()
    g4_e2e = await run_e2e_safety_demo(
        scenarios=(
            ("learned", learned_config),
            (
                "pe_off",
                AntSessionConfig(
                    temporal_latent_dim=16,
                    seed=0,
                    external_prediction_error_drive=False,
                ),
            ),
            (
                "eta_off_lite",
                AntSessionConfig(
                    temporal_latent_dim=16,
                    seed=0,
                    temporal_policy=LearnedLiteTemporalPolicy(
                        parameter_store=MetacontrollerParameterStore(n_z=16)
                    ),
                ),
            ),
        )
    )
    _write(
        "g4_safety_reflex.json",
        {
            "experiment": "g4_safety_reflex_veto",
            "artifact_kind": "digital-ant-formal-g4",
            "description": g4.description,
            "n_ticks": g4.n_ticks,
            "n_alarmed": g4.n_alarmed,
            "reflex_turn": g4.reflex_turn,
            "reflex_step": g4.reflex_step,
            "all_alarmed_are_reflex": g4.all_alarmed_are_reflex,
            "reflex_deterministic": g4.reflex_deterministic,
            "reflex_ignores_code": g4.reflex_ignores_code,
            "max_calm_turn_magnitude": g4.max_calm_turn_magnitude,
            "e2e": asdict(g4_e2e),
            "verdict": (
                "PASS"
                if g4_e2e.all_states_vetoed and g4.reflex_ignores_code
                else "BLOCK"
            ),
        },
    )
    print(f"[G4] {g4.description}")

    print(f"[demos] results -> {_RESULTS_DIR}/ ; figures -> {_FIGURES_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
