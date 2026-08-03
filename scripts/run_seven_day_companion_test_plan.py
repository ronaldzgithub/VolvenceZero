#!/usr/bin/env python3
"""Run every preregistered seven-day companion campaign on Apple MPS.

The preregistration schema is the dispatch SSOT.  This control plane owns
device/lock/process sequencing only; each campaign-specific runner and auditor
continues to own its intervention, readouts, and verdict.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

from companion_test_plan_common import (
    exclusive_mps_lock,
    execution_environment,
    mps_payload,
    print_json,
    require_mps,
    run_plan_command,
)


CONTINUITY_SCHEMAS = frozenset(
    {
        # v1/v2 remain readable so the stopped 2026-08-02 campaign can still
        # be audited as halted.  The campaign-specific runner rejects them
        # for execution because only v3/v4 contain the N+1 contract.
        "seven-day-companion-simulated.v1",
        "seven-day-companion-simulated.v2",
        "seven-day-companion-simulated.v3",
        "seven-day-companion-simulated.v4",
    }
)
GATE1_SCHEMA = "gate1-seven-day-companion-prereg.v2"
GATE_SUITE_SCHEMA = "companion-gate-suite-seven-day-prereg.v2"
GATE_SUITE_IDS = (4, 5, 6, 7, 9, 10)
SCIENTIFIC_NEGATIVE_EXIT = 2
FORMAL_RESULT_EXITS = frozenset({0, SCIENTIFIC_NEGATIVE_EXIT})
STAGES = ("status", "preflight", "smoke", "formal", "audit", "all")


@dataclass(frozen=True)
class SevenDayCampaign:
    key: str
    plan_id: str
    gate_ids: tuple[int, ...]
    runner_relative_path: str
    auditor_relative_path: str
    smoke_flag: str
    evaluation_relative_path: str
    audit_evaluation_sha_field: str
    audit_report_name: str
    audit_schema_version: str
    audit_pass_field: str
    default_port: int
    gate_id: int | None = None

    @property
    def audit_relative_path(self) -> Path:
        return Path("audit") / self.audit_report_name


def _require_file(path: Path, *, label: str) -> Path:
    target = path.resolve()
    if not target.is_file():
        raise FileNotFoundError(f"{label} does not exist: {target}")
    return target


def _load_preregistration(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"seven-day preregistration is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("seven-day preregistration root must be an object")
    return payload


def _campaign_from_preregistration(
    payload: Mapping[str, object],
) -> SevenDayCampaign:
    schema = payload.get("schema_version")
    if schema in CONTINUITY_SCHEMAS:
        return SevenDayCampaign(
            key="continuity",
            plan_id="seven-day-companion-gates-8-11-mps.v1",
            gate_ids=(8, 11),
            runner_relative_path="scripts/run_seven_day_companion_formal.py",
            auditor_relative_path="scripts/audit_seven_day_companion_formal.py",
            smoke_flag="--smoke-one-run",
            evaluation_relative_path="ablation_results.json",
            audit_evaluation_sha_field="ablation_results_sha256",
            audit_report_name="independent_audit.json",
            audit_schema_version="seven-day-companion-independent-audit.v1",
            audit_pass_field="passed",
            default_port=18765,
        )
    if schema == GATE1_SCHEMA:
        return SevenDayCampaign(
            key="gate1",
            plan_id="seven-day-companion-gate-1-mps.v1",
            gate_ids=(1,),
            runner_relative_path="scripts/run_seven_day_gate1_formal.py",
            auditor_relative_path="scripts/audit_seven_day_gate1_formal.py",
            smoke_flag="--smoke-one-pair",
            evaluation_relative_path="gate1_evaluation.json",
            audit_evaluation_sha_field="gate1_evaluation_sha256",
            audit_report_name="gate1_independent_audit.json",
            audit_schema_version="gate1-seven-day-independent-audit.v1",
            audit_pass_field="audit_passed",
            default_port=18779,
            gate_id=1,
        )
    if schema == GATE_SUITE_SCHEMA:
        gate_id = payload.get("gate_id")
        if isinstance(gate_id, bool) or gate_id not in GATE_SUITE_IDS:
            raise ValueError("seven-day gate-suite preregistration has invalid gate_id")
        assert isinstance(gate_id, int)
        return SevenDayCampaign(
            key=f"gate{gate_id}",
            plan_id=f"seven-day-companion-gate-{gate_id}-mps.v1",
            gate_ids=(gate_id,),
            runner_relative_path="scripts/run_seven_day_gate_suite_formal.py",
            auditor_relative_path="scripts/audit_seven_day_gate_suite_formal.py",
            smoke_flag="--smoke-one-pair",
            evaluation_relative_path=f"gate{gate_id}_evaluation.json",
            audit_evaluation_sha_field="evaluation_sha256",
            audit_report_name=f"gate{gate_id}_independent_audit.json",
            audit_schema_version="companion-gate-suite-independent-audit.v1",
            audit_pass_field="audit_passed",
            default_port=18780,
            gate_id=gate_id,
        )
    raise ValueError(f"unsupported seven-day preregistration schema: {schema!r}")


def _formal_contract(payload: Mapping[str, object]) -> Mapping[str, object]:
    formal = payload.get("formal_run")
    if not isinstance(formal, Mapping):
        raise ValueError("seven-day preregistration lacks formal_run")
    return formal


def _require_mps_preregistration(payload: Mapping[str, object]) -> None:
    device = _formal_contract(payload).get("execution_device")
    if device != "mps":
        raise ValueError(
            "the MPS control plane requires a hardware-specific preregistration "
            f"with execution_device='mps'; got {device!r}"
        )


def _string_list(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"seven-day {field} must be a non-empty list")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"seven-day {field} contains an invalid value")
    if len(set(value)) != len(value):
        raise ValueError(f"seven-day {field} contains duplicate values")
    return tuple(value)


def _seed_list(value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("seven-day paraphrase_seeds must be a non-empty list")
    if not all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in value):
        raise ValueError("seven-day paraphrase_seeds contains an invalid value")
    if len(set(value)) != len(value):
        raise ValueError("seven-day paraphrase_seeds contains duplicate values")
    return tuple(value)


def _expected_run_identities(
    payload: Mapping[str, object],
) -> dict[str, tuple[str, int, str]]:
    formal = _formal_contract(payload)
    scenarios = _string_list(payload.get("scenario_ids"), field="scenario_ids")
    seeds = _seed_list(formal.get("paraphrase_seeds"))
    arms = _string_list(formal.get("arm_schedule"), field="arm_schedule")
    expected: dict[str, tuple[str, int, str]] = {}
    for arm in arms:
        for scenario in scenarios:
            for seed in seeds:
                case_id = f"{scenario}:seed-{seed}"
                name = hashlib.sha256(f"{case_id}\0{arm}".encode("utf-8")).hexdigest() + ".json"
                expected[name] = (scenario, seed, arm)
    declared = formal.get("run_count")
    if isinstance(declared, bool) or not isinstance(declared, int) or declared != len(expected):
        raise ValueError("seven-day formal_run.run_count differs from scenario/seed/arm matrix")
    return expected


def _valid_completed_runs(
    *, output_dir: Path, expected: Mapping[str, tuple[str, int, str]]
) -> tuple[set[str], set[str], set[str]]:
    run_root = output_dir / "runs"
    actual_paths = {path.name: path for path in run_root.glob("*.json") if path.is_file()}
    valid: set[str] = set()
    invalid: set[str] = set()
    for name, (scenario, seed, arm) in expected.items():
        path = actual_paths.get(name)
        if path is None:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            invalid.add(name)
            continue
        if not isinstance(payload, dict) or (
            payload.get("schema_version") != "seven-day-companion-run.v1"
            or payload.get("scenario_id") != scenario
            or payload.get("paraphrase_seed") != seed
            or payload.get("arm_label") != arm
        ):
            invalid.add(name)
            continue
        valid.add(name)
    unexpected = set(actual_paths).difference(expected)
    return valid, invalid, unexpected


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _halt_state(
    *,
    preregistration: Mapping[str, object],
    output_dir: Path,
    completed_run_count: int,
    expected_run_count: int,
) -> dict[str, object] | None:
    path = output_dir / "halt_record.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("seven-day halt record is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError("seven-day halt record root must be an object")
    halted_preregistration = payload.get("halted_preregistration")
    preserved_state = payload.get("preserved_state")
    discipline = payload.get("discipline_attestation")
    resumption = payload.get("resumption_policy")
    if (
        payload.get("schema_version") != "seven-day-companion-halt-record.v1"
        or not isinstance(halted_preregistration, dict)
        or halted_preregistration.get("sha256")
        != _canonical_sha256(preregistration)
        or not isinstance(preserved_state, dict)
        or preserved_state.get("complete_run_envelopes_preserved")
        != completed_run_count
        or preserved_state.get("expected_run_count") != expected_run_count
        or not isinstance(discipline, dict)
        or discipline.get("effect_claim_allowed") is not False
        or discipline.get("production_promotion_authorized") is not False
        or not isinstance(resumption, dict)
        or resumption.get("resume_as_is_authorized") is not False
    ):
        raise ValueError("seven-day halt record contract drift")
    halt_class = payload.get("halt_class")
    if not isinstance(halt_class, str) or not halt_class:
        raise ValueError("seven-day halt record lacks halt_class")
    return {
        "halt_record_present": True,
        "halt_class": halt_class,
        "resume_as_is_authorized": False,
        "halted_at_unix_ms": payload.get("halted_at_unix_ms"),
    }


def _audit_state(
    *,
    campaign: SevenDayCampaign,
    preregistration: Mapping[str, object],
    output_dir: Path,
    expected_run_count: int,
    audit_output_dir: Path | None = None,
) -> tuple[bool, bool, str | None]:
    candidates = tuple(
        dict.fromkeys(
            (
                *(
                    (audit_output_dir / campaign.audit_report_name,)
                    if audit_output_dir is not None
                    else ()
                ),
                output_dir / campaign.audit_relative_path,
                output_dir / campaign.audit_report_name,
            )
        )
    )
    existing = tuple(path for path in candidates if path.is_file())
    if not existing:
        return False, False, None
    try:
        report_bytes = tuple(path.read_bytes() for path in existing)
    except OSError:
        return True, False, "independent audit cannot be read"
    if len(report_bytes) > 1 and any(
        content != report_bytes[0] for content in report_bytes[1:]
    ):
        return True, False, "multiple independent audit reports differ"
    path = existing[0]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return True, False, "independent audit is not valid JSON"
    if not isinstance(payload, dict):
        return True, False, "independent audit root is not an object"
    counts = payload.get("counts")
    evaluation_sha256 = _file_sha256(
        output_dir / campaign.evaluation_relative_path
    )
    valid = (
        payload.get("schema_version") == campaign.audit_schema_version
        and payload.get(campaign.audit_pass_field) is True
        and payload.get("preregistration_sha256") == _canonical_sha256(preregistration)
        and isinstance(counts, dict)
        and counts.get("runs") == expected_run_count
        and payload.get("claim_scope") == preregistration.get("claim_scope")
        and evaluation_sha256 is not None
        and payload.get(campaign.audit_evaluation_sha_field)
        == evaluation_sha256
    )
    if campaign.gate_id not in {None, 1}:
        valid = valid and payload.get("gate_id") == campaign.gate_id
    return True, bool(valid), None if valid else "independent audit contract drift"


def _status(
    *,
    preregistration: Path,
    output_dir: Path | None,
    audit_output_dir: Path | None = None,
) -> dict[str, object]:
    payload = _load_preregistration(preregistration)
    campaign = _campaign_from_preregistration(payload)
    formal = _formal_contract(payload)
    expected = _expected_run_identities(payload)
    valid: set[str] = set()
    invalid: set[str] = set()
    unexpected: set[str] = set()
    evaluation_present = False
    audit_present = False
    audit_valid = False
    audit_error: str | None = None
    halt: dict[str, object] | None = None
    if output_dir is not None and output_dir.is_dir():
        valid, invalid, unexpected = _valid_completed_runs(output_dir=output_dir, expected=expected)
        evaluation_present = (output_dir / campaign.evaluation_relative_path).is_file()
        audit_present, audit_valid, audit_error = _audit_state(
            campaign=campaign,
            preregistration=payload,
            output_dir=output_dir,
            expected_run_count=len(expected),
            audit_output_dir=audit_output_dir,
        )
        halt = _halt_state(
            preregistration=payload,
            output_dir=output_dir,
            completed_run_count=len(valid),
            expected_run_count=len(expected),
        )
    matrix_complete = len(valid) == len(expected) and not invalid and not unexpected
    if halt is not None and matrix_complete:
        raise ValueError("halted seven-day matrix cannot also be complete")
    analysis_allowed = (
        halt is None and matrix_complete and evaluation_present and audit_valid
    )
    run_state = (
        "halted"
        if halt is not None
        else "complete"
        if matrix_complete
        else "running"
        if output_dir is not None and output_dir.is_dir()
        else "not-started"
    )
    status: dict[str, object] = {
        "plan_id": campaign.plan_id,
        "campaign": campaign.key,
        "gate_ids": campaign.gate_ids,
        "preregistration_schema": payload.get("schema_version"),
        "claim_scope": payload.get("claim_scope"),
        "execution_device": formal.get("execution_device"),
        "expected_runs": len(expected),
        "completed_valid_run_files": len(valid),
        "invalid_run_files": len(invalid),
        "unexpected_run_files": len(unexpected),
        "run_state": run_state,
        "matrix_complete": matrix_complete,
        "evaluation_present": evaluation_present,
        "independent_audit_present": audit_present,
        "independent_audit_valid": audit_valid,
        "analysis_allowed": analysis_allowed,
        "human_rating_required_for_human_anchor": True,
        "production_promotion_authorized": False,
    }
    if audit_error is not None:
        status["audit_error"] = audit_error
    if halt is not None:
        status.update(halt)
    return status


def _runner_command(
    *,
    campaign: SevenDayCampaign,
    python: Path,
    execution_root: Path,
    preregistration: Path,
    stage: str,
    output_dir: Path | None,
    smoke_evidence_root: Path | None,
    host: str,
    port: int,
    startup_timeout_s: float,
    resume: bool,
) -> tuple[str, ...]:
    runner = _require_file(
        execution_root / campaign.runner_relative_path,
        label=f"{campaign.key} seven-day runner",
    )
    argv = [
        str(python),
        str(runner),
    ]
    if campaign.key.startswith("gate") and campaign.gate_id not in {None, 1}:
        argv.extend(("--gate", str(campaign.gate_id)))
    argv.extend(
        (
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
        )
    )
    if stage == "preflight":
        argv.append("--preflight-only")
    elif stage == "smoke":
        if output_dir is None:
            raise ValueError("seven-day smoke requires --output-dir")
        argv.extend(("--output-dir", str(output_dir), campaign.smoke_flag))
    elif stage == "formal":
        if output_dir is None or smoke_evidence_root is None:
            raise ValueError(
                "seven-day formal requires output and smoke evidence roots"
            )
        argv.extend(
            (
                "--output-dir",
                str(output_dir),
                "--smoke-evidence-root",
                str(smoke_evidence_root),
                "--execute",
            )
        )
    else:
        raise ValueError(f"unsupported seven-day runner stage: {stage}")
    if resume:
        argv.append("--resume")
    return tuple(argv)


def _audit_command(
    *,
    campaign: SevenDayCampaign,
    python: Path,
    execution_root: Path,
    preregistration: Path,
    output_dir: Path,
    audit_output_dir: Path,
) -> tuple[str, ...]:
    runner = _require_file(
        execution_root / campaign.auditor_relative_path,
        label=f"{campaign.key} seven-day independent auditor",
    )
    try:
        report_name = (audit_output_dir / campaign.audit_report_name).relative_to(output_dir)
    except ValueError as exc:
        raise ValueError("--audit-output-dir must be inside --output-dir") from exc
    argv = [str(python), str(runner)]
    if campaign.key.startswith("gate") and campaign.gate_id not in {None, 1}:
        argv.extend(("--gate", str(campaign.gate_id)))
    argv.extend(
        (
            "--execution-root",
            str(execution_root),
            "--preregistration",
            str(preregistration),
            "--output-dir",
            str(output_dir),
            "--report-name",
            str(report_name),
        )
    )
    return tuple(argv)


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
    parser.add_argument("--port", type=int)
    parser.add_argument("--startup-timeout-s", type=float, default=600.0)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    execution_root = args.execution_root.resolve()
    preregistration = _require_file(args.preregistration, label="preregistration")
    preregistration_payload = _load_preregistration(preregistration)
    campaign = _campaign_from_preregistration(preregistration_payload)
    _require_mps_preregistration(preregistration_payload)
    output_dir = args.output_dir.resolve() if args.output_dir is not None else None
    smoke_output_dir = (
        output_dir.with_name(f"{output_dir.name}_smoke")
        if output_dir is not None
        else None
    )
    requested_audit_output = (
        args.audit_output_dir.resolve()
        if args.audit_output_dir is not None
        else None
    )
    if args.stage == "status":
        print_json(
            _status(
                preregistration=preregistration,
                output_dir=output_dir,
                audit_output_dir=requested_audit_output,
            )
        )
        return 0
    if args.stage in {"smoke", "formal", "audit", "all"} and output_dir is None:
        raise ValueError(f"seven-day {args.stage} requires --output-dir")
    if (
        args.stage in {"formal", "all"}
        and output_dir is not None
        and output_dir.is_dir()
        and _status(
            preregistration=preregistration,
            output_dir=output_dir,
        )["run_state"]
        == "halted"
    ):
        raise RuntimeError(
            "seven-day output is formally halted and cannot be resumed as-is; "
            "create a new preregistration and output root"
        )
    environment = execution_environment(execution_root)
    audit_output = (
        requested_audit_output
        if requested_audit_output is not None
        else (output_dir / "audit" if output_dir is not None else None)
    )

    if args.stage == "audit":
        if output_dir is None or not output_dir.is_dir() or audit_output is None:
            raise FileNotFoundError("seven-day audit requires an existing --output-dir")
        command = _audit_command(
            campaign=campaign,
            python=args.python.resolve(),
            execution_root=execution_root,
            preregistration=preregistration,
            output_dir=output_dir,
            audit_output_dir=audit_output,
        )
        audit_output.mkdir(parents=True, exist_ok=True)
        return run_plan_command(
            command,
            execution_root=execution_root,
            environment=environment,
        )

    port = campaign.default_port if args.port is None else args.port
    formal_return_code = 0
    with exclusive_mps_lock(args.mps_lock, plan_id=campaign.plan_id):
        mps = require_mps()
        environment["VZ_COMPANION_MPS_LOCK_HELD"] = "1"
        environment["VZ_COMPANION_MPS_LOCK_PATH"] = str(
            args.mps_lock.resolve()
        )
        stages = (
            ("preflight", "smoke", "formal")
            if args.stage == "all"
            else (args.stage,)
        )
        for stage in stages:
            stage_output_dir = (
                smoke_output_dir if stage == "smoke" else output_dir
            )
            return_code = run_plan_command(
                _runner_command(
                    campaign=campaign,
                    python=args.python.resolve(),
                    execution_root=execution_root,
                    preregistration=preregistration,
                    stage=stage,
                    output_dir=stage_output_dir,
                    smoke_evidence_root=smoke_output_dir,
                    host=args.host,
                    port=port,
                    startup_timeout_s=args.startup_timeout_s,
                    resume=args.resume and stage in {"smoke", "formal"},
                ),
                execution_root=execution_root,
                environment=environment,
            )
            if stage == "formal":
                formal_return_code = return_code
                if return_code not in FORMAL_RESULT_EXITS:
                    return return_code
            elif return_code != 0:
                return return_code

    if args.stage == "all":
        if output_dir is None or audit_output is None:
            raise RuntimeError("seven-day all lost its output directory")
        audit_command = _audit_command(
            campaign=campaign,
            python=args.python.resolve(),
            execution_root=execution_root,
            preregistration=preregistration,
            output_dir=output_dir,
            audit_output_dir=audit_output,
        )
        audit_output.mkdir(parents=True, exist_ok=True)
        audit_return_code = run_plan_command(
            audit_command,
            execution_root=execution_root,
            environment=environment,
        )
        if audit_return_code != 0:
            return audit_return_code
        print_json(
            _status(
                preregistration=preregistration,
                output_dir=output_dir,
                audit_output_dir=audit_output,
            )
        )
        return formal_return_code

    print_json(
        {
            "plan_id": campaign.plan_id,
            "campaign": campaign.key,
            "stage": args.stage,
            "scientific_exit_code": formal_return_code,
            "mps": mps_payload(mps),
        }
    )
    return formal_return_code


if __name__ == "__main__":
    raise SystemExit(main())
