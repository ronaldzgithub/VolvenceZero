#!/usr/bin/env python3
"""Coding-lab Packet 3.6 runner: episode-outcome intervention-timing gate.

Commands mirror the Packet 3.5 runner conventions:

* ``smoke`` — scripted hand, development tier, zero API cost; validates the
  four-arm machinery against a synthetic certified surface.
* ``freeze-prereg`` — derive the certified steering surface from the frozen
  Packet 3.5 formal report (its SHA-256 is embedded), freeze gates/config.
* ``formal`` — API-hand run bound to the frozen prereg (SHA verified).
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
from lifeform_evolution.coding_lab_packet36 import (  # noqa: E402
    CertifiedCell,
    HAND_API,
    HAND_SCRIPTED,
    Packet36Config,
    derive_advisor_cells,
    derive_certified_cells,
    run_packet36,
)


def _sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(report: dict) -> dict:
    return {
        "run_id": report["run_id"],
        "evidence_tier": report["evidence_tier"],
        "pass_rates_by_arm": report["pass_rates_by_arm"],
        "verdicts": report["verdicts"],
        "timing_vs_noop": {
            "mean_gap": report["contrasts"]["timing_vs_noop"]["mean_gap"],
            "lower_5pct": report["contrasts"]["timing_vs_noop"]["bootstrap_ci_lower_5pct"],
        },
        "timing_vs_random": {
            "mean_gap": report["contrasts"]["timing_vs_random"]["mean_gap"],
            "lower_5pct": report["contrasts"]["timing_vs_random"]["bootstrap_ci_lower_5pct"],
        },
        "table_vs_always": {
            "mean_gap": report["contrasts"]["table_vs_always"]["mean_gap"],
            "lower_5pct": report["contrasts"]["table_vs_always"]["bootstrap_ci_lower_5pct"],
        },
        "binding_gates_pass": report["binding_gates_pass"],
        "mechanism": report["mechanism"],
    }


def _smoke_cells() -> tuple[CertifiedCell, ...]:
    """Synthetic certified surface for machinery smoke (dev tier only)."""

    return (
        CertifiedCell(
            state_key="fix_bug|reads=1|edited=0|tests=none",
            category="fix_bug",
            expert_action="edit",
            expert_itt_pass_rate=1.0,
            natural_control_pass_rate=0.8,
        ),
        CertifiedCell(
            state_key="refactor_alias|reads=2|edited=0|tests=none",
            category="refactor_alias",
            expert_action="investigate",
            expert_itt_pass_rate=1.0,
            natural_control_pass_rate=1.0,
        ),
    )


def cmd_smoke(args: argparse.Namespace) -> int:
    run_id = args.run_id or f"packet36_smoke_{time.strftime('%Y%m%d_%H%M%S')}"
    config = Packet36Config(
        run_id=run_id,
        output_root=pathlib.Path(args.output_root),
        certified_cells=_smoke_cells(),
        chains=args.chains,
        episodes_per_chain=args.episodes_per_chain,
        hand_kind=HAND_SCRIPTED,
        gate_seed=args.gate_seed,
        resume=args.resume,
    )
    report = asyncio.run(run_packet36(config, evidence_tier="development"))
    print(json.dumps(_summary(report), ensure_ascii=False, sort_keys=True))
    return 0


def cmd_freeze_prereg(args: argparse.Namespace) -> int:
    calibration_path = pathlib.Path(args.calibration_report)
    calibration_sha = _sha256_file(calibration_path)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if args.design == "v2-advisor":
        cells = derive_advisor_cells(calibration)
        positive = [c for c in cells if c.credited_gain >= args.table_gate_min_gain]
        negative = [c for c in cells if c.credited_gain < 0.0]
        if not positive or not negative:
            raise ValueError(
                "v2-advisor prereg requires at least one positive-gain and one "
                f"negative-gain priced cell (got {len(positive)} positive / "
                f"{len(negative)} negative); T2 exit decision says NO-GO"
            )
        binding_gates = ["avoidance_timing_gate", "outcome_timing_gate"]
        gates_doc = {
            "avoidance_timing_gate": (
                "table_gate - always_on chain-bootstrap 5% lower bound > 0 "
                "(declining causally priced harmful advice)"
            ),
            "outcome_timing_gate": "table_gate - noop chain-bootstrap 5% lower bound > 0",
            "intervention_gate": (
                "always_on - noop: DIRECTIONAL REPORT ONLY in v2 (the advisor is "
                "fallible by design; its net sign is a finding, not a gate)"
            ),
            "placement_gate": "table_gate - random_gate: directional report only",
        }
    else:
        cells = derive_certified_cells(calibration)
        binding_gates = ["outcome_timing_gate", "intervention_gate"]
        gates_doc = {
            "outcome_timing_gate": "table_gate - noop chain-bootstrap 5% lower bound > 0",
            "placement_gate": (
                "table_gate - random_gate: DIRECTIONAL REPORT ONLY. Pre-computed "
                "power: rate-matched random captures ~half the certified effect "
                "(expected gap ~= 0.018), below reliable detection at this N."
            ),
            "intervention_gate": "always_on - noop chain-bootstrap 5% lower bound > 0",
            "table_vs_always": "REPORT-ONLY (expected null on a two-cell surface)",
        }
    prereg = {
        "prereg_id": f"coding-lab-packet36-{int(time.time())}",
        "packet": "coding-lab-packet-3.6",
        "design": args.design,
        "estimand": (
            "chain-paired episode pass-rate gaps between matched arms "
            "(noop / always_on / random_gate / table_gate) on the oracle verdict"
        ),
        "calibration_report_path": str(calibration_path).replace("\\", "/"),
        "calibration_report_sha256": calibration_sha,
        "certified_cells": [
            {
                "state_key": cell.state_key,
                "category": cell.category,
                "expert_action": cell.expert_action,
                "expert_itt_pass_rate": cell.expert_itt_pass_rate,
                "natural_control_pass_rate": cell.natural_control_pass_rate,
                "credited_gain": cell.credited_gain,
            }
            for cell in cells
        ],
        "chains": args.chains,
        "episodes_per_chain": args.episodes_per_chain,
        "env_seed": args.env_seed,
        "gate_seed": args.gate_seed,
        "random_gate_steer_probability": args.random_gate_probability,
        "table_gate_min_gain": args.table_gate_min_gain,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": args.gate_seed,
        "convention_ids": list(args.conventions or ()),
        "gates": gates_doc,
        "binding_gates": binding_gates,
        "hand": {
            "kind": "api",
            "base_url": args.base_url,
            "model": args.model,
            "api_key_env": args.api_key_env,
            "temperature": 0.0,
        },
        "honest_endgames": [
            "all_gates_pass",
            "timing_no_outcome_gain (gate FAIL, sealed as-is)",
            "advisor_uniformly_good_or_bad (v2: no avoidance headroom realized)",
        ],
        "claim_boundary": (
            "Action-level intervention timing driven by interventional credit, "
            "measured on oracle episode pass rate. In the v2-advisor design the "
            "timing claim is 'knowing when to trust a fallible advisor', priced "
            "causally per cell. Residual-level Steerable is NOT lifted."
        ),
    }
    path = pathlib.Path(args.output)
    if path.exists():
        raise FileExistsError(f"prereg already exists: {path}")
    raw = json.dumps(prereg, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
    path.write_text(raw, encoding="utf-8")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="utf-8")
    print(json.dumps({"prereg_path": str(path), "sha256": digest,
                      "certified_cells": len(cells)}, ensure_ascii=False))
    return 0


def cmd_formal(args: argparse.Namespace) -> int:
    prereg_path = pathlib.Path(args.prereg)
    raw = prereg_path.read_text(encoding="utf-8")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    expected = pathlib.Path(args.prereg + ".sha256").read_text(encoding="utf-8").strip()
    if digest != expected:
        raise ValueError(
            f"prereg SHA mismatch: computed {digest}, frozen {expected}; refusing to run"
        )
    prereg = json.loads(raw)
    calibration_path = pathlib.Path(prereg["calibration_report_path"])
    observed_calibration = _sha256_file(calibration_path)
    if observed_calibration != prereg["calibration_report_sha256"]:
        raise ValueError("calibration report drifted from the prereg-frozen SHA-256")
    cells = tuple(
        CertifiedCell(
            state_key=cell["state_key"],
            category=cell["category"],
            expert_action=cell["expert_action"],
            expert_itt_pass_rate=float(cell["expert_itt_pass_rate"]),
            natural_control_pass_rate=float(cell["natural_control_pass_rate"]),
        )
        for cell in prereg["certified_cells"]
    )
    hand = prereg["hand"]
    config = Packet36Config(
        run_id=args.run_id,
        output_root=pathlib.Path(args.output_root),
        certified_cells=cells,
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
        gate_seed=int(prereg["gate_seed"]),
        random_gate_steer_probability=float(prereg["random_gate_steer_probability"]),
        table_gate_min_gain=float(prereg["table_gate_min_gain"]),
        bootstrap_resamples=int(prereg["bootstrap_resamples"]),
        bootstrap_seed=int(prereg["bootstrap_seed"]),
        binding_gates=tuple(prereg["binding_gates"]),
        resume=args.resume,
    )
    report = asyncio.run(
        run_packet36(
            config,
            evidence_tier="formal",
            prereg_sha256=digest,
            calibration_report_sha256=observed_calibration,
        )
    )
    print(json.dumps(_summary(report), ensure_ascii=False, sort_keys=True))
    return 0 if report["binding_gates_pass"] else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="coding-lab Packet 3.6 episode-outcome gate runner"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    smoke = commands.add_parser("smoke")
    smoke.add_argument("--output-root", default="artifacts/coding_lab")
    smoke.add_argument("--run-id", default="")
    smoke.add_argument("--chains", type=int, default=4)
    smoke.add_argument("--episodes-per-chain", type=int, default=8)
    smoke.add_argument("--gate-seed", type=int, default=20260828)
    smoke.add_argument("--resume", action="store_true")
    smoke.set_defaults(func=cmd_smoke)

    freeze = commands.add_parser("freeze-prereg")
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--calibration-report", required=True)
    freeze.add_argument("--design", choices=("v1-expert", "v2-advisor"), default="v1-expert")
    freeze.add_argument("--chains", type=int, required=True)
    freeze.add_argument("--episodes-per-chain", type=int, required=True)
    freeze.add_argument("--env-seed", type=int, default=20260812)
    freeze.add_argument("--gate-seed", type=int, required=True)
    freeze.add_argument("--random-gate-probability", type=float, default=0.5)
    freeze.add_argument("--table-gate-min-gain", type=float, default=0.05)
    freeze.add_argument("--conventions", nargs="*", default=["convention_export_all"])
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
