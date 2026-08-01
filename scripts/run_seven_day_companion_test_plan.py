#!/usr/bin/env python3
"""Run the isolated seven-day simulated-companion test plan on Apple MPS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from companion_test_plan_common import (
    exclusive_mps_lock,
    execution_environment,
    mps_payload,
    print_json,
    require_mps,
    run_plan_command,
)


PLAN_ID = "seven-day-companion-simulated-mps.v1"
STAGES = ("status", "preflight", "smoke", "formal", "audit", "all")


def _require_file(path: Path, *, label: str) -> Path:
    target = path.resolve()
    if not target.is_file():
        raise FileNotFoundError(f"{label} does not exist: {target}")
    return target


def _runner_command(
    *,
    python: Path,
    execution_root: Path,
    preregistration: Path,
    stage: str,
    output_dir: Path | None,
    host: str,
    port: int,
    startup_timeout_s: float,
    resume: bool,
) -> tuple[str, ...]:
    runner = _require_file(
        execution_root / "scripts/run_seven_day_companion_formal.py",
        label="seven-day runner",
    )
    argv = [
        str(python),
        str(runner),
        "--repo-root",
        str(execution_root),
        "--preregistration",
        str(preregistration),
        "--device",
        "mps",
        "--host",
        host,
        "--port",
        str(port),
        "--startup-timeout-s",
        str(startup_timeout_s),
    ]
    if stage == "preflight":
        argv.append("--preflight-only")
    elif stage == "smoke":
        if output_dir is None:
            raise ValueError("seven-day smoke requires --output-dir")
        argv.extend(("--output-dir", str(output_dir), "--smoke-one-run"))
    elif stage == "formal":
        if output_dir is None:
            raise ValueError("seven-day formal requires --output-dir")
        argv.extend(("--output-dir", str(output_dir), "--execute"))
    else:
        raise ValueError(f"unsupported seven-day runner stage: {stage}")
    if resume:
        argv.append("--resume")
    return tuple(argv)


def _audit_command(
    *,
    python: Path,
    execution_root: Path,
    preregistration: Path,
    output_dir: Path,
    audit_output_dir: Path,
) -> tuple[str, ...]:
    runner = _require_file(
        execution_root / "scripts/audit_seven_day_companion_formal.py",
        label="seven-day independent auditor",
    )
    return (
        str(python),
        str(runner),
        "--execution-root",
        str(execution_root),
        "--preregistration",
        str(preregistration),
        "--output-dir",
        str(output_dir),
        "--report-name",
        str((audit_output_dir / "independent_audit.json").relative_to(output_dir)),
    )


def _status(*, preregistration: Path, output_dir: Path | None) -> dict[str, object]:
    payload = json.loads(preregistration.read_text(encoding="utf-8"))
    formal = payload.get("formal_run")
    if not isinstance(formal, dict):
        raise ValueError("seven-day preregistration lacks formal_run")
    expected = formal.get("run_count")
    if isinstance(expected, bool) or not isinstance(expected, int):
        raise ValueError("seven-day preregistration run_count is invalid")
    completed = 0
    independent_audit = False
    if output_dir is not None and output_dir.is_dir():
        completed = len(tuple((output_dir / "runs").glob("*.json")))
        independent_audit = (output_dir / "audit/independent_audit.json").is_file()
    return {
        "plan_id": PLAN_ID,
        "claim_scope": payload.get("claim_scope"),
        "execution_device": formal.get("execution_device"),
        "expected_runs": expected,
        "completed_run_files": completed,
        "matrix_complete": completed == expected,
        "independent_audit_present": independent_audit,
        "human_rating_required_for_human_anchor": True,
        "production_promotion_authorized": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--execution-root", type=Path, default=Path.cwd())
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--audit-output-dir", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--mps-lock",
        type=Path,
        default=Path("artifacts/.companion-evidence-mps.lock"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--startup-timeout-s", type=float, default=600.0)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    execution_root = args.execution_root.resolve()
    preregistration = _require_file(args.preregistration, label="preregistration")
    output_dir = args.output_dir.resolve() if args.output_dir is not None else None
    if args.stage == "status":
        print_json(_status(preregistration=preregistration, output_dir=output_dir))
        return 0
    if args.stage in {"smoke", "formal", "audit", "all"} and output_dir is None:
        raise ValueError(f"seven-day {args.stage} requires --output-dir")
    if args.stage == "audit":
        if output_dir is None or not output_dir.is_dir():
            raise FileNotFoundError("seven-day audit requires an existing --output-dir")
        audit_output = (
            args.audit_output_dir.resolve()
            if args.audit_output_dir is not None
            else output_dir / "audit"
        )
        audit_output.mkdir(parents=True, exist_ok=True)
        command = _audit_command(
            python=args.python.resolve(),
            execution_root=execution_root,
            preregistration=preregistration,
            output_dir=output_dir,
            audit_output_dir=audit_output,
        )
        return run_plan_command(
            command,
            execution_root=execution_root,
            environment=execution_environment(execution_root),
        )

    environment = execution_environment(execution_root)
    with exclusive_mps_lock(args.mps_lock, plan_id=PLAN_ID):
        mps = require_mps()
        stages = ("preflight", "formal") if args.stage == "all" else (args.stage,)
        for stage in stages:
            command = _runner_command(
                python=args.python.resolve(),
                execution_root=execution_root,
                preregistration=preregistration,
                stage=stage,
                output_dir=output_dir,
                host=args.host,
                port=args.port,
                startup_timeout_s=args.startup_timeout_s,
                resume=args.resume and stage == "formal",
            )
            return_code = run_plan_command(
                command,
                execution_root=execution_root,
                environment=environment,
            )
            if return_code != 0:
                return return_code
    if args.stage == "all":
        if output_dir is None:
            raise RuntimeError("seven-day all lost its output directory")
        audit_output = (
            args.audit_output_dir.resolve()
            if args.audit_output_dir is not None
            else output_dir / "audit"
        )
        audit_output.mkdir(parents=True, exist_ok=True)
        return_code = run_plan_command(
            _audit_command(
                python=args.python.resolve(),
                execution_root=execution_root,
                preregistration=preregistration,
                output_dir=output_dir,
                audit_output_dir=audit_output,
            ),
            execution_root=execution_root,
            environment=environment,
        )
        if return_code != 0:
            return return_code
    print_json({"plan_id": PLAN_ID, "stage": args.stage, "mps": mps_payload(mps)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
