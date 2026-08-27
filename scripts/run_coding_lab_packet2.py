#!/usr/bin/env python3
"""Coding-lab Packet 2: memory injection vs long-context steelman.

Three subcommands:

* ``freeze-prereg`` — freeze the formal run's design (arms, N, M,
  budgets, dual-gate thresholds, hand lineage) into a JSON whose SHA-256
  is the run's identity. Refuses to overwrite.
* ``smoke``       — instrument calibration with the memory-aware
  scripted hand: injects a KNOWN effect and verifies the measurement
  spine detects it (brain-vs-stateless slope gap > 0), the expected
  null stays null (brain-vs-steelman ~ 0 because the needle exists in
  both contexts), and the scaling gate has teeth. NOT evidence.
* ``formal``      — the preregistered API-hand run (requires the frozen
  prereg, a matching hand config and the API key; refuses otherwise).

Verdict semantics follow the frozen plan: quality gate = chain-paired
slope difference (brain - steelman) with chain-bootstrap CI lower bound
above the preregistered minimum effect; scaling gate = brain/steelman
context-token ratio <= 0.10 (asymmetry is evidence, not matched).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import pathlib
import shutil
import sys
import time
from typing import Any

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
for _pkg in ("lifeform-evolution", "lifeform-domain-coding"):
    sys.path.insert(0, str(_REPO_ROOT / "packages" / _pkg / "src"))

from lifeform_domain_coding.lab.episode import EpisodeBudget  # noqa: E402
from lifeform_domain_coding.lab.hands import (  # noqa: E402
    APIHandConfig,
    MemoryAwareScriptedHand,
    OpenAICompatHand,
)
from lifeform_evolution.coding_lab_arms import (  # noqa: E402
    ALL_ARMS,
    ARM_BRAIN,
    ARM_STATELESS,
    ARM_STEELMAN,
    ArmChainConfig,
    ArmEpisodeRow,
    paired_slope_gap,
    run_chain_arm,
    scaling_gate,
    scaling_structure,
    wall_seconds_by_arm,
)

_SMOKE_NEEDLES = {"fix_bug": "round_half_up"}


def _preflight_disk(output_root: pathlib.Path, *, minimum_bytes: int = 2 * 1024**3) -> None:
    usage = shutil.disk_usage(output_root)
    if usage.free < minimum_bytes:
        raise SystemExit(
            f"preflight: only {usage.free} bytes free under {output_root!s} "
            f"(need >= {minimum_bytes}); C3 disk-death lesson: fail before running"
        )


async def _run_all_arms(
    *,
    run_dir: pathlib.Path,
    chains: int,
    episodes: int,
    env_seed: int,
    digest_char_budget: int,
    hand_builder: Any,
    budget: EpisodeBudget,
    convention_ids: tuple[str, ...] = (),
    resume: bool = False,
) -> tuple[ArmEpisodeRow, ...]:
    """Run every (chain, arm) cell, checkpointing per cell.

    Same discipline as Packet 0 calibration resume: a cell with a
    committed ``rows.json`` is loaded as-is; an interrupted cell is
    wiped and rerun whole (arms are independent given the frozen hand,
    so cell granularity keeps determinism).
    """

    rows: list[ArmEpisodeRow] = []
    for chain_index in range(chains):
        config = ArmChainConfig(
            env_seed=env_seed,
            chain_index=chain_index,
            episodes=episodes,
            brain_digest_char_budget=digest_char_budget,
            budget=budget,
            convention_ids=convention_ids,
        )
        for arm in ALL_ARMS:
            arm_root = run_dir / arm / f"chain-{chain_index:02d}"
            checkpoint = arm_root / "rows.json"
            if resume and checkpoint.is_file():
                for item in json.loads(checkpoint.read_text(encoding="utf-8")):
                    item["invariant_violations"] = tuple(item["invariant_violations"])
                    rows.append(ArmEpisodeRow(**item))
                continue
            if arm_root.exists():
                shutil.rmtree(arm_root)
            cell_rows = await run_chain_arm(
                arm=arm,
                config=config,
                arm_root=arm_root,
                hand_factory=hand_builder,
            )
            checkpoint.write_text(
                json.dumps(
                    [row.__dict__ for row in cell_rows],
                    ensure_ascii=False,
                    indent=1,
                    default=list,
                ),
                encoding="utf-8",
            )
            rows.extend(cell_rows)
    return tuple(rows)


def _report_common(rows: tuple[ArmEpisodeRow, ...]) -> dict[str, Any]:
    quality_vs_steelman = paired_slope_gap(rows, arm_a=ARM_BRAIN, arm_b=ARM_STEELMAN)
    quality_vs_stateless = paired_slope_gap(rows, arm_a=ARM_BRAIN, arm_b=ARM_STATELESS)
    scaling = scaling_gate(rows)
    pass_rates = {
        arm: (
            sum(1 for row in rows if row.arm == arm and row.passed)
            / max(1, sum(1 for row in rows if row.arm == arm))
        )
        for arm in ALL_ARMS
    }
    return {
        "quality_brain_vs_steelman": quality_vs_steelman,
        "quality_brain_vs_stateless": quality_vs_stateless,
        "scaling": scaling,
        "pass_rates": pass_rates,
        "mean_wall_seconds_by_arm": wall_seconds_by_arm(rows),
        "episodes": [row.__dict__ for row in rows],
    }


def _write_report(run_dir: pathlib.Path, report: dict[str, Any]) -> None:
    (run_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def cmd_freeze_prereg(args: argparse.Namespace) -> int:
    prereg_path = pathlib.Path(args.prereg_path)
    if prereg_path.exists():
        raise SystemExit(f"prereg already exists (frozen designs are immutable): {prereg_path!s}")
    prereg = {
        "prereg_id": f"coding-lab-packet2-{int(time.time())}",
        "arms": list(ALL_ARMS),
        "chains": args.chains,
        "episodes_per_chain": args.episodes_per_chain,
        "env_seed": args.env_seed,
        "digest_char_budget": args.digest_char_budget,
        # Difficulty knob must match the Packet 0 calibration that put
        # THIS hand inside the oracle band.
        "convention_ids": [
            item.strip() for item in args.conventions.split(",") if item.strip()
        ],
        # Three frozen gates, matching the actual claim: memory helps
        # (brain > stateless), structured memory is not worse than
        # context stuffing (non-inferiority vs steelman), and it scales
        # (token/latency ratios).
        "memory_gate": {
            "statistic": "chain-paired OLS slope gap (brain - stateless) of pass indicator",
            "min_effect": 0.0,
            "rule": "chain-bootstrap 5% lower bound > min_effect",
        },
        "quality_gate": {
            "statistic": "chain-paired OLS slope gap (brain - steelman) of pass indicator",
            "min_effect": args.quality_min_effect,
            "rule": "chain-bootstrap 5% lower bound > min_effect (non-inferiority margin)",
        },
        "scaling_gate": {"max_token_ratio": 0.10, "max_latency_ratio": 0.50},
        "hand": {
            "kind": "api",
            "base_url": args.api_base_url,
            "model": args.api_model,
            "api_key_env": args.api_key_env,
            "temperature": 0.0,
            "extra_body": json.loads(args.api_extra_body_json) if args.api_extra_body_json else {},
        },
        "preconditions": [
            "Packet 0 calibration re-run with THIS frozen hand passed the oracle band",
            "forecast-skill instrument fix or scope note recorded (see Packet 1 diagnostic)",
        ],
        "frozen_at_unix": int(time.time()),
    }
    blob = json.dumps(prereg, ensure_ascii=False, indent=2, sort_keys=True)
    prereg_path.parent.mkdir(parents=True, exist_ok=True)
    prereg_path.write_text(blob + "\n", encoding="utf-8")
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    (prereg_path.with_suffix(".sha256")).write_text(digest + "\n", encoding="utf-8")
    print(json.dumps({"prereg": str(prereg_path), "sha256": digest}))
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    run_dir = pathlib.Path(args.output_root) / args.run_id
    if run_dir.exists():
        raise SystemExit(f"run dir already exists: {run_dir!s}")
    run_dir.mkdir(parents=True)
    _preflight_disk(run_dir)

    def hand_builder(chain, chain_index):
        return MemoryAwareScriptedHand(
            needles_by_category=dict(_SMOKE_NEEDLES),
            tasks_by_id={task.task_id: task for task in chain},
            episode_index_by_task_id={task.task_id: i for i, task in enumerate(chain)},
            hand_seed=args.hand_seed + 101 * chain_index,
            invariant_sabotage_rate=0.0,
            acceptance_sabotage_rate=args.acceptance_sabotage_rate,
        )

    rows = asyncio.run(
        _run_all_arms(
            run_dir=run_dir,
            chains=args.chains,
            episodes=args.episodes_per_chain,
            env_seed=args.env_seed,
            digest_char_budget=args.digest_char_budget,
            hand_builder=hand_builder,
            budget=EpisodeBudget(max_steps=24, max_wall_seconds=600.0),
        )
    )
    common = _report_common(rows)
    structure = scaling_structure(rows)
    verdicts = {
        # Known-effect direction: the needle is reachable through the brain
        # digest but not through the empty stateless context.
        "instrument_detects_known_effect": (
            common["quality_brain_vs_stateless"]["mean_gap"] > 0
        ),
        # Expected null: the needle also lives in the steelman transcript,
        # so a scripted hand cannot differentiate the two context forms.
        "expected_null_brain_vs_steelman": (
            abs(common["quality_brain_vs_steelman"]["mean_gap"]) < 0.5
        ),
        # Toy scale cannot separate on the production threshold (0.10 mean
        # ratio, frozen for formal); the smoke asserts the structural teeth.
        "scaling_structure": structure["passed"],
    }
    common["scaling_structure"] = structure
    report = {
        "packet": "coding-lab-packet-2-smoke",
        "run_id": args.run_id,
        "scope": (
            "machinery calibration with an injected known effect; NOT evidence "
            "of memory value (see module docstring)"
        ),
        "needles": _SMOKE_NEEDLES,
        "verdicts": verdicts,
        **common,
    }
    _write_report(run_dir, report)
    print(json.dumps({"run_id": args.run_id, "verdicts": verdicts}, ensure_ascii=False))
    print(f"report: {run_dir / 'report.json'}")
    return 0 if all(verdicts.values()) else 2


def cmd_formal(args: argparse.Namespace) -> int:
    prereg_path = pathlib.Path(args.prereg_path)
    if not prereg_path.is_file():
        raise SystemExit(f"formal run requires a frozen prereg: {prereg_path!s}")
    blob = prereg_path.read_text(encoding="utf-8").rstrip("\n")
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    expected_digest = prereg_path.with_suffix(".sha256").read_text(encoding="utf-8").strip()
    if digest != expected_digest:
        raise SystemExit("prereg content does not match its frozen sha256; refusing to run")
    prereg = json.loads(blob)
    hand_config = APIHandConfig(
        base_url=prereg["hand"]["base_url"],
        model=prereg["hand"]["model"],
        api_key_env=prereg["hand"]["api_key_env"],
        temperature=float(prereg["hand"]["temperature"]),
        extra_body=dict(prereg["hand"]["extra_body"]),
    )
    run_dir = pathlib.Path(args.output_root) / args.run_id
    if run_dir.exists() and not args.resume:
        raise SystemExit(
            f"run dir already exists: {run_dir!s} (use --resume to continue)"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    # Single-writer lock: two racing instances under --resume wipe each
    # other's in-flight cell directories (2026-08-13 v2 crash: worktree
    # lost mv_app mid-episode).
    lock_path = run_dir / ".formal.lock"
    if lock_path.is_file():
        holder = int(lock_path.read_text(encoding="utf-8").strip() or "0")
        try:
            os.kill(holder, 0)
        except (OSError, ValueError):
            # POSIX raises ProcessLookupError for a dead pid; Windows raises
            # OSError (WinError 87) from the failed OpenProcess. Both mean
            # the recorded holder is gone and the lock is stale.
            lock_path.unlink()
        else:
            raise SystemExit(f"another formal instance (pid {holder}) holds {lock_path!s}")
    lock_path.write_text(str(os.getpid()), encoding="utf-8")
    _preflight_disk(run_dir)

    def hand_builder(chain, chain_index):
        del chain, chain_index
        return OpenAICompatHand(hand_config)

    rows = asyncio.run(
        _run_all_arms(
            run_dir=run_dir,
            chains=int(prereg["chains"]),
            episodes=int(prereg["episodes_per_chain"]),
            env_seed=int(prereg["env_seed"]),
            digest_char_budget=int(prereg["digest_char_budget"]),
            hand_builder=hand_builder,
            budget=EpisodeBudget(max_steps=24, max_wall_seconds=900.0),
            convention_ids=tuple(prereg["convention_ids"]),
            resume=args.resume,
        )
    )
    common = _report_common(rows)
    quality = common["quality_brain_vs_steelman"]
    memory = common["quality_brain_vs_stateless"]
    verdicts = {
        "memory_gate": (
            memory["bootstrap_ci_lower_5pct"] > float(prereg["memory_gate"]["min_effect"])
        ),
        "quality_gate": (
            quality["bootstrap_ci_lower_5pct"] > float(prereg["quality_gate"]["min_effect"])
        ),
        "scaling_gate": common["scaling"]["passed"],
    }
    report = {
        "packet": "coding-lab-packet-2-formal",
        "run_id": args.run_id,
        "prereg_sha256": digest,
        "verdicts": verdicts,
        **common,
    }
    _write_report(run_dir, report)
    print(json.dumps({"run_id": args.run_id, "verdicts": verdicts}, ensure_ascii=False))
    return 0 if all(verdicts.values()) else 2


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze-prereg")
    freeze.add_argument("--prereg-path", required=True)
    freeze.add_argument("--chains", type=int, default=8)
    freeze.add_argument("--episodes-per-chain", type=int, default=10)
    freeze.add_argument("--env-seed", type=int, default=20260812)
    freeze.add_argument("--digest-char-budget", type=int, default=4_000)
    freeze.add_argument("--quality-min-effect", type=float, default=0.0)
    freeze.add_argument("--api-base-url", required=True)
    freeze.add_argument("--api-model", required=True)
    freeze.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    freeze.add_argument("--api-extra-body-json", default="")
    freeze.add_argument(
        "--conventions",
        default="",
        help="逗号分隔的 house 约定 id；必须与通过频带的 Packet 0 标定一致。",
    )
    freeze.set_defaults(func=cmd_freeze_prereg)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--run-id", default=f"coding_lab_packet2_smoke_{int(time.time())}")
    smoke.add_argument("--output-root", default=str(_REPO_ROOT / "artifacts" / "coding_lab"))
    smoke.add_argument("--chains", type=int, default=3)
    smoke.add_argument("--episodes-per-chain", type=int, default=8)
    smoke.add_argument("--env-seed", type=int, default=20260812)
    smoke.add_argument("--digest-char-budget", type=int, default=4_000)
    smoke.add_argument("--hand-seed", type=int, default=11)
    smoke.add_argument("--acceptance-sabotage-rate", type=float, default=0.6)
    smoke.set_defaults(func=cmd_smoke)

    formal = subparsers.add_parser("formal")
    formal.add_argument("--prereg-path", required=True)
    formal.add_argument("--run-id", default=f"coding_lab_packet2_formal_{int(time.time())}")
    formal.add_argument("--output-root", default=str(_REPO_ROOT / "artifacts" / "coding_lab"))
    formal.add_argument(
        "--resume",
        action="store_true",
        help="按 (chain, arm) 格续跑：已提交 rows.json 的格直接加载，中断格整格重跑。",
    )
    formal.set_defaults(func=cmd_formal)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
