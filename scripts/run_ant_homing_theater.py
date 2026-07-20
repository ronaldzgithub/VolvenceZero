"""Digital-ant homing theater — the system's validated, watchable strength.

Foraging pellet-count is not where the kernel beats a hand-written FSM at toy
scale (matched-control records ``learned`` ≈ 0 vs ``fixed_rule`` > 0). Its
AntBot-scale strength is *navigation*: path integration lets a swarm wander far
and still steer a straight line home. This renders that side by side against a
matched no-compass ablation that drifts and gets lost, plus a route-familiarity
panel driven by the real kernel memory/PE chain.

    python scripts/run_ant_homing_theater.py [--n-ants 18] [--outbound 70]
        [--home 140] [--seed 0] [--no-route] [--open]

The navigation arms are pure frozen-substrate numpy (fast). The optional route
panel drives the real kernel, so it adds a minute or two; use --no-route to skip.

Output: research/ant/figures/digital_ant_homing_theater.html
"""

from __future__ import annotations

import argparse
import asyncio
import webbrowser
from pathlib import Path

from volvence_ant.viz import run_homing_theater

_FIGURES = Path(__file__).resolve().parents[1] / "research/ant/figures"


async def main(args: argparse.Namespace) -> int:
    out_path = _FIGURES / "digital_ant_homing_theater.html"
    report = await run_homing_theater(
        n_ants=args.n_ants,
        outbound_steps=args.outbound,
        home_steps=args.home,
        seed=args.seed,
        include_route=not args.no_route,
        out_path=out_path,
    )
    for arm in report.arms:
        print(
            f"[homing] {arm.kind:<16} return_rate={arm.return_rate:.0%} "
            f"norm_error={arm.mean_normalized_error:.4f} "
            f"antbot_scale={arm.passes_antbot_scale}"
        )
    if report.route is not None:
        r = report.route
        print(
            f"[homing] route novelty {r.first_exposure_novelty:.4f} -> "
            f"{r.last_exposure_novelty:.4f} (memory_off={r.memory_off_last_novelty:.4f}, "
            f"improved={r.familiarity_improved})"
        )
    print(f"[homing] html={report.html_path}")
    if args.open and report.html_path is not None:
        webbrowser.open(Path(report.html_path).resolve().as_uri())
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-ants", type=int, default=18)
    parser.add_argument("--outbound", type=int, default=70)
    parser.add_argument("--home", type=int, default=140)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-route", action="store_true", help="skip the kernel route panel")
    parser.add_argument("--open", action="store_true", help="open the HTML in a browser")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args)))
