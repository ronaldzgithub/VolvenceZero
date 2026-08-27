#!/usr/bin/env python3
"""Coding-lab Packet 3.5 runner: interventional junction calibration.

Commands mirror the Packet 2 runner conventions:

* ``smoke``  — scripted ConstraintAwareScriptedHand, development tier,
  zero API cost; validates the RCT machinery (trigger, realizations,
  ITT bookkeeping, table building).
* ``freeze-prereg`` — write the formal prereg JSON + its SHA-256; any
  formal run must reference exactly these frozen bytes.
* ``formal`` — API hand run bound to a frozen prereg (SHA verified);
  refuses to start without it.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import pathlib
import sys
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
for _package_src in sorted(_REPO_ROOT.glob("packages/*/src")):
    if str(_package_src) not in sys.path:
        sys.path.insert(0, str(_package_src))

from lifeform_domain_coding.lab.hands import APIHandConfig  # noqa: E402
from lifeform_domain_coding.lab.junctions import JUNCTION_ACTIONS  # noqa: E402
from lifeform_domain_coding.lab.tasks import ALL_CATEGORIES  # noqa: E402
from lifeform_evolution.coding_lab_interventional import (  # noqa: E402
    HAND_API,
    HAND_SCRIPTED,
    InterventionalConfig,
    run_interventional_calibration,
)


def _first_decision_state_keys() -> tuple[str, ...]:
    return tuple(
        f"{category}|reads=0|edited=0|tests=none" for category in sorted(ALL_CATEGORIES)
    )


def cmd_smoke(args: argparse.Namespace) -> int:
    run_id = args.run_id or f"packet35_smoke_{time.strftime('%Y%m%d_%H%M%S')}"
    config = InterventionalConfig(
        run_id=run_id,
        output_root=pathlib.Path(args.output_root),
        target_state_keys=_first_decision_state_keys(),
        chains=args.chains,
        episodes_per_chain=args.episodes_per_chain,
        hand_kind=HAND_SCRIPTED,
        assignment_seed=args.assignment_seed,
        control_weight=args.control_weight,
        resume=args.resume,
    )
    report = asyncio.run(run_interventional_calibration(config, evidence_tier="development"))
    print(json.dumps({
        "run_id": report["run_id"],
        "evidence_tier": report["evidence_tier"],
        "episodes": len(report["episodes"]),
        "assignment_summary": report["assignment_summary"],
        "interventional_expert_actions": report["interventional_expert_actions"],
        "expert_disagreements": report["expert_disagreements"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


def cmd_freeze_prereg(args: argparse.Namespace) -> int:
    prereg = {
        "prereg_id": f"coding-lab-packet35-{int(time.time())}",
        "packet": "coding-lab-packet-3.5",
        "estimand": (
            "intention-to-treat episode pass rate per (state_key, assigned_action) "
            "cell, randomized uniformly at the first target-state junction"
        ),
        "arms": ["control"] + list(JUNCTION_ACTIONS),
        "control_weight": args.control_weight,
        "chains": args.chains,
        "episodes_per_chain": args.episodes_per_chain,
        "env_seed": args.env_seed,
        "assignment_seed": args.assignment_seed,
        "convention_ids": list(args.conventions or ()),
        "target_state_keys": list(args.target_state_keys),
        "min_action_support": 5,
        "min_pass_rate_margin": 0.10,
        "hand": {
            "kind": "api",
            "base_url": args.base_url,
            "model": args.model,
            "api_key_env": args.api_key_env,
            "temperature": 0.0,
        },
        "honest_endgames": [
            "interventional_experts_confirm_observational",
            "interventional_experts_contradict_observational (finding, not failure)",
            "insufficient_cell_support_no_expert_map",
        ],
        "notes": (
            "ITT is causal for assigned action given reached state; state "
            "reachability stays observational. No capability claim."
        ),
    }
    path = pathlib.Path(args.output)
    if path.exists():
        raise FileExistsError(f"prereg already exists: {path}")
    raw = json.dumps(prereg, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
    path.write_text(raw, encoding="utf-8")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="utf-8")
    print(json.dumps({"prereg_path": str(path), "sha256": digest}, ensure_ascii=False))
    return 0


def cmd_formal(args: argparse.Namespace) -> int:
    prereg_path = pathlib.Path(args.prereg)
    raw = prereg_path.read_text(encoding="utf-8")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    expected = (
        pathlib.Path(args.prereg + ".sha256").read_text(encoding="utf-8").strip()
    )
    if digest != expected:
        raise ValueError(
            f"prereg SHA mismatch: computed {digest}, frozen {expected}; refusing to run"
        )
    prereg = json.loads(raw)
    hand = prereg["hand"]
    config = InterventionalConfig(
        run_id=args.run_id,
        output_root=pathlib.Path(args.output_root),
        target_state_keys=tuple(prereg["target_state_keys"]),
        env_seed=int(prereg["env_seed"]),
        chains=int(prereg["chains"]),
        episodes_per_chain=int(prereg["episodes_per_chain"]),
        hand_kind=HAND_API,
        api_hand_config=APIHandConfig(
            base_url=str(hand["base_url"]),
            model=str(hand["model"]),
            api_key_env=str(hand["api_key_env"]),
            temperature=float(hand["temperature"]),
        ),
        convention_ids=tuple(prereg["convention_ids"]),
        assignment_seed=int(prereg["assignment_seed"]),
        control_weight=float(prereg["control_weight"]),
        resume=args.resume,
    )
    report = asyncio.run(
        run_interventional_calibration(
            config, evidence_tier="formal", prereg_sha256=digest
        )
    )
    print(json.dumps({
        "run_id": report["run_id"],
        "evidence_tier": report["evidence_tier"],
        "prereg_sha256": digest,
        "episodes": len(report["episodes"]),
        "assignment_summary": report["assignment_summary"],
        "interventional_expert_actions": report["interventional_expert_actions"],
        "expert_disagreements": report["expert_disagreements"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="coding-lab Packet 3.5 interventional runner")
    commands = parser.add_subparsers(dest="command", required=True)

    smoke = commands.add_parser("smoke")
    smoke.add_argument("--output-root", default="artifacts/coding_lab")
    smoke.add_argument("--run-id", default="")
    smoke.add_argument("--chains", type=int, default=4)
    smoke.add_argument("--episodes-per-chain", type=int, default=8)
    smoke.add_argument("--assignment-seed", type=int, default=20260827)
    smoke.add_argument("--control-weight", type=float, default=0.2)
    smoke.add_argument("--resume", action="store_true")
    smoke.set_defaults(func=cmd_smoke)

    freeze = commands.add_parser("freeze-prereg")
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--chains", type=int, required=True)
    freeze.add_argument("--episodes-per-chain", type=int, required=True)
    freeze.add_argument("--env-seed", type=int, default=20260812)
    freeze.add_argument("--assignment-seed", type=int, required=True)
    freeze.add_argument("--control-weight", type=float, default=0.2)
    freeze.add_argument("--conventions", nargs="*", default=["convention_export_all"])
    freeze.add_argument("--target-state-keys", nargs="+", required=True)
    freeze.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    freeze.add_argument("--model", default="qwen3-coder-next")
    freeze.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    freeze.set_defaults(func=cmd_freeze_prereg)

    formal = commands.add_parser("formal")
    formal.add_argument("--prereg", required=True)
    formal.add_argument("--run-id", required=True)
    formal.add_argument("--output-root", default="artifacts/coding_lab")
    formal.add_argument("--resume", action="store_true")
    formal.set_defaults(func=cmd_formal)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
