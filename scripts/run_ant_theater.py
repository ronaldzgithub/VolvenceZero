"""Digital-ant colony theater — an intuitive side-by-side swarm animation.

Runs two colonies foraging the same world and writes a self-contained
(zero-dependency) HTML canvas animation you can open in any browser:

- left  : heuristic colony (hardcoded FSM foragers, ``FixedRuleAnt``)
- right : digital-life colony (kernel-driven ``AntSession`` bodies)

Half-way through, the food is relocated. The heatmap shows the pheromone
trail corridor self-organising; the ant dots show foraging (blue) vs carrying
food home (amber). Watch the rigid FSM keep marching at stale assumptions while
the learning controller re-senses and re-routes.

    python scripts/run_ant_theater.py [--n-ants 6] [--rounds 60] [--seed 0] [--open]

The digital-life arm drives one full kernel ``AntSession`` per body, so runtime
scales with ``n_ants * rounds`` (~0.5s per ant-tick). The defaults render in a
few minutes; raise them for a longer, smoother clip.

Output: research/ant/figures/digital_ant_theater.html
"""

from __future__ import annotations

import argparse
import asyncio
import webbrowser
from pathlib import Path

from volvence_ant.runtime import AntSessionConfig
from volvence_ant.viz import run_colony_theater

_FIGURES = Path(__file__).resolve().parents[1] / "research/ant/figures"


async def main(args: argparse.Namespace) -> int:
    from volvence_zero.joint_loop import JointLoopSchedule

    out_path = _FIGURES / "digital_ant_theater.html"
    session_config = AntSessionConfig(
        temporal_latent_dim=16,
        session_id=f"colony-theater:{args.seed}",
        seed=args.seed,
        joint_schedule=JointLoopSchedule(ssl_interval=1, rl_interval=3),
        joint_apply_writeback=True,
    )
    report = await run_colony_theater(
        n_ants=args.n_ants,
        rounds=args.rounds,
        relocate_at=args.relocate_at,
        seed=args.seed,
        session_config=session_config,
        out_path=out_path,
    )
    for arm in report.arms:
        final = arm.frames[-1]
        print(f"[theater] {arm.kind:<12} delivered={final.delivered}")
    print(f"[theater] html={report.html_path}")
    if args.open and report.html_path is not None:
        webbrowser.open(Path(report.html_path).resolve().as_uri())
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-ants", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--relocate-at", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--open", action="store_true", help="open the HTML in a browser")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args)))
