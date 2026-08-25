"""Gate 2 longitudinal admission for the frozen v35 open-loop selector.

This module is deliberately an evidence readout.  It never fits a selector,
installs a control basis, applies residual control, or mutates runtime owner
state.  Its first job is to reject longitudinal sources that cannot support a
selector-aligned matched outcome claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from volvence_zero.agent.gate11_longitudinal_source import (
    GATE11_LONGITUDINAL_SOURCE_SEEDS,
    build_gate11_longitudinal_source_plans,
    load_gate11_longitudinal_source_records,
    validate_gate11_longitudinal_source_prefix,
)
from volvence_zero.internal_rl import selector_artifact_from_payload


GATE2_LONGITUDINAL_ADMISSION_SCHEMA_VERSION = (
    "gate2-longitudinal-open-loop-readout-admission.v1"
)
GATE2_LONGITUDINAL_INPUT_SCHEMA_VERSION = (
    "gate2-v35-open-loop-selector-input.v1"
)
GATE2_LONGITUDINAL_OUTCOME_SCHEMA_VERSION = (
    "gate2-v35-open-loop-selector-outcome.v1"
)
GATE2_V35_SELECTOR_FINGERPRINT = (
    "ef360e0e72e00d235e7fc0df39b249178e080bf2065c6443dad801dfd77f4293"
)
GATE2_V35_CONTROL_BASIS_FINGERPRINT = (
    "326aecddc8d0b7e81161568121d457267d8473c22dc74c11b0dc1396b4d9761b"
)
GATE2_LONGITUDINAL_INPUT_FILENAME = "selector_readout_inputs.jsonl"
GATE2_LONGITUDINAL_OUTCOME_FILENAME = "selector_matched_outcomes.jsonl"
GATE2_LONGITUDINAL_REQUIRED_FILES = (
    "manifest.yaml",
    "predictions.jsonl",
    "outcomes.jsonl",
    "prediction_errors.jsonl",
    "segments.jsonl",
    "credit.jsonl",
    "state_diff.jsonl",
    "action_selection.jsonl",
    "ablation_results.json",
    "promotion_verdict.json",
    "rollback_evidence.json",
    "report.md",
)


@dataclass(frozen=True)
class Gate2LongitudinalReadoutContract:
    selector_fingerprint: str = GATE2_V35_SELECTOR_FINGERPRINT
    control_basis_fingerprint: str = (
        GATE2_V35_CONTROL_BASIS_FINGERPRINT
    )
    selector_input_dim: int = 8076
    selector_action_count: int = 22
    min_settled_transition_count: int = 500
    min_consumer_session_count: int = 2


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_bytes(row).decode("utf-8") + "\n")


def _git_output(*args: str) -> str:
    try:
        result = subprocess.run(
            ("git", *args),
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field} must be finite")
    return resolved


def _load_companion_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            raise ValueError(
                f"blank Gate 2 companion row at {path}:{line_number}"
            )
        payload = json.loads(line)
        transition_id = payload.get("transition_id")
        if not isinstance(transition_id, str) or not transition_id:
            raise ValueError(
                f"Gate 2 companion row lacks transition_id at "
                f"{path}:{line_number}"
            )
        if transition_id in rows:
            raise ValueError(
                f"duplicate Gate 2 companion transition {transition_id!r}"
            )
        actual_digest = payload.pop("record_sha256", None)
        expected_digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
        payload["record_sha256"] = actual_digest
        if actual_digest != expected_digest:
            raise ValueError(
                f"Gate 2 companion digest mismatch at {path}:{line_number}"
            )
        rows[transition_id] = payload
    return rows


def load_gate2_v35_selector_bundle(
    path: str | Path,
    *,
    contract: Gate2LongitudinalReadoutContract | None = None,
) -> dict[str, object]:
    """Load the frozen selector wrapper and fail on any lineage drift."""

    active_contract = contract or Gate2LongitudinalReadoutContract()
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "eta-gate2-selector-artifact.v1":
        raise ValueError(
            "Gate 2 longitudinal readout requires one frozen v1 selector "
            "artifact wrapper"
        )
    artifact_payload = payload.get("artifact")
    if not isinstance(artifact_payload, dict):
        raise ValueError("Gate 2 selector wrapper lacks artifact payload")
    artifact = selector_artifact_from_payload(artifact_payload)
    if artifact.model_fingerprint != active_contract.selector_fingerprint:
        raise ValueError(
            "Gate 2 v35 selector fingerprint drift: "
            f"expected={active_contract.selector_fingerprint}, "
            f"actual={artifact.model_fingerprint}"
        )
    if artifact.input_dim != active_contract.selector_input_dim:
        raise ValueError(
            "Gate 2 v35 selector input dimension drift: "
            f"expected={active_contract.selector_input_dim}, "
            f"actual={artifact.input_dim}"
        )
    if artifact.action_count != active_contract.selector_action_count:
        raise ValueError(
            "Gate 2 v35 selector action count drift: "
            f"expected={active_contract.selector_action_count}, "
            f"actual={artifact.action_count}"
        )
    basis_fingerprint = payload.get("control_basis_fingerprint")
    if basis_fingerprint != active_contract.control_basis_fingerprint:
        raise ValueError(
            "Gate 2 v35 control basis fingerprint drift: "
            f"expected={active_contract.control_basis_fingerprint}, "
            f"actual={basis_fingerprint}"
        )
    return {
        "wrapper_schema_version": payload["schema_version"],
        "selector_schema_version": artifact_payload["schema_version"],
        "selector_model_kind": artifact_payload["model_kind"],
        "selector_fingerprint": artifact.model_fingerprint,
        "selector_input_dim": artifact.input_dim,
        "selector_action_count": artifact.action_count,
        "control_basis_fingerprint": basis_fingerprint,
        "fit_split": payload.get("fit_split"),
        "run_id": payload.get("run_id"),
        "run_seed": payload.get("run_seed"),
        "artifact_sha256": _sha256_file(source),
    }


def _validate_readout_input(
    row: Mapping[str, Any],
    *,
    transition_id: str,
    contract: Gate2LongitudinalReadoutContract,
) -> None:
    if row.get("schema_version") != GATE2_LONGITUDINAL_INPUT_SCHEMA_VERSION:
        raise ValueError(
            f"{transition_id} has unsupported selector readout input schema"
        )
    if row.get("selector_fingerprint") != contract.selector_fingerprint:
        raise ValueError(f"{transition_id} selector fingerprint drifted")
    if (
        row.get("control_basis_fingerprint")
        != contract.control_basis_fingerprint
    ):
        raise ValueError(f"{transition_id} control basis fingerprint drifted")
    state_features = row.get("state_features")
    if not isinstance(state_features, list) or len(state_features) != (
        contract.selector_input_dim
    ):
        raise ValueError(
            f"{transition_id} selector state dimension mismatch"
        )
    for index, value in enumerate(state_features):
        _finite_number(
            value,
            field=f"{transition_id}.state_features[{index}]",
        )
    if (
        row.get("capture_source") != "real"
        or row.get("fallback_active") is not False
        or row.get("substrate_mutation_applied") is not False
    ):
        raise ValueError(
            f"{transition_id} selector input is not a mutation-free real "
            "capture"
        )


def _validate_matched_outcome(
    row: Mapping[str, Any],
    *,
    transition_id: str,
    contract: Gate2LongitudinalReadoutContract,
) -> None:
    if row.get("schema_version") != GATE2_LONGITUDINAL_OUTCOME_SCHEMA_VERSION:
        raise ValueError(
            f"{transition_id} has unsupported selector outcome schema"
        )
    if row.get("selector_fingerprint") != contract.selector_fingerprint:
        raise ValueError(f"{transition_id} outcome selector drifted")
    if (
        row.get("control_basis_fingerprint")
        != contract.control_basis_fingerprint
    ):
        raise ValueError(f"{transition_id} outcome basis drifted")
    action_index = row.get("selected_action_index")
    if (
        isinstance(action_index, bool)
        or not isinstance(action_index, int)
        or not 0 <= action_index < contract.selector_action_count
    ):
        raise ValueError(f"{transition_id} selected action index is invalid")
    for field in (
        "selected_realized_delta",
        "zero_realized_delta",
        "permutation_null_mean",
    ):
        _finite_number(row.get(field), field=f"{transition_id}.{field}")
    if row.get("outcome_chain") != (
        "isolated-residual-forward->realized-continuation-nll-readout"
    ):
        raise ValueError(
            f"{transition_id} outcome is not from the isolated matched lane"
        )
    if row.get("typed_pe_credit_executed") is not False:
        raise ValueError(
            f"{transition_id} readout must not claim typed PE/credit execution"
        )
    if row.get("source_fixed_outcome_reused") is not False:
        raise ValueError(
            f"{transition_id} outcome reused the fixed source result"
        )


def assess_gate2_longitudinal_readout_admission(
    *,
    records_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    readout_inputs_by_seed: Mapping[
        int, Mapping[str, Mapping[str, Any]]
    ],
    matched_outcomes_by_seed: Mapping[
        int, Mapping[str, Mapping[str, Any]]
    ],
    selector_lineage: Mapping[str, object],
    contract: Gate2LongitudinalReadoutContract | None = None,
) -> dict[str, object]:
    """Assess source and companion completeness without executing selector."""

    active_contract = contract or Gate2LongitudinalReadoutContract()
    seed_results: list[dict[str, object]] = []
    total_transition_count = 0
    total_missing_input_count = 0
    total_missing_outcome_count = 0
    for seed in sorted(records_by_seed):
        records = tuple(records_by_seed[seed])
        transition_ids = tuple(
            str(record.get("transition_id", "")) for record in records
        )
        if any(not transition_id for transition_id in transition_ids):
            raise ValueError(f"Gate 2 source seed {seed} has missing ids")
        if len(set(transition_ids)) != len(transition_ids):
            raise ValueError(f"Gate 2 source seed {seed} has duplicate ids")
        settled_count = sum(record.get("settled") is True for record in records)
        real_count = sum(
            record.get("substrate", {}).get("capture_source") == "real"
            for record in records
        )
        fallback_count = sum(
            record.get("substrate", {}).get("fallback_active") is not False
            for record in records
        )
        mutation_count = sum(
            record.get("substrate", {}).get("mutation_applied") is not False
            for record in records
        )
        intervals = {
            record.get("longitudinal", {}).get(
                "consumer_session_boundary_interval"
            )
            for record in records
        }
        if len(intervals) != 1:
            raise ValueError(
                f"Gate 2 source seed {seed} has session interval drift"
            )
        session_interval = next(iter(intervals), None)
        if (
            isinstance(session_interval, bool)
            or not isinstance(session_interval, int)
            or session_interval < 1
        ):
            raise ValueError(
                f"Gate 2 source seed {seed} has invalid session interval"
            )
        consumer_session_count = math.ceil(
            len(records) / session_interval
        )
        inputs = readout_inputs_by_seed.get(seed, {})
        outcomes = matched_outcomes_by_seed.get(seed, {})
        extra_inputs = set(inputs).difference(transition_ids)
        extra_outcomes = set(outcomes).difference(transition_ids)
        if extra_inputs or extra_outcomes:
            raise ValueError(
                f"Gate 2 companion rows do not join source seed {seed}: "
                f"extra_inputs={sorted(extra_inputs)!r}, "
                f"extra_outcomes={sorted(extra_outcomes)!r}"
            )
        missing_input_ids = [
            transition_id
            for transition_id in transition_ids
            if transition_id not in inputs
        ]
        missing_outcome_ids = [
            transition_id
            for transition_id in transition_ids
            if transition_id not in outcomes
        ]
        for transition_id in transition_ids:
            if transition_id in inputs:
                _validate_readout_input(
                    inputs[transition_id],
                    transition_id=transition_id,
                    contract=active_contract,
                )
            if transition_id in outcomes:
                _validate_matched_outcome(
                    outcomes[transition_id],
                    transition_id=transition_id,
                    contract=active_contract,
                )
        source_gates = {
            "settled_transition_count": (
                settled_count
                >= active_contract.min_settled_transition_count
            ),
            "real_substrate_rate": real_count == len(records),
            "fallback_rate": fallback_count == 0,
            "frozen_substrate_mutation_count": mutation_count == 0,
            "cross_session_count": (
                consumer_session_count
                >= active_contract.min_consumer_session_count
            ),
        }
        source_admitted = all(source_gates.values())
        readout_ready = (
            source_admitted
            and not missing_input_ids
            and not missing_outcome_ids
        )
        seed_results.append(
            {
                "seed": seed,
                "transition_count": len(records),
                "settled_transition_count": settled_count,
                "real_substrate_rate": (
                    real_count / len(records) if records else 0.0
                ),
                "fallback_count": fallback_count,
                "frozen_substrate_mutation_count": mutation_count,
                "consumer_session_boundary_interval": session_interval,
                "consumer_session_count": consumer_session_count,
                "source_gates": source_gates,
                "source_admitted": source_admitted,
                "readout_input_count": len(inputs),
                "matched_outcome_count": len(outcomes),
                "missing_readout_input_count": len(missing_input_ids),
                "missing_matched_outcome_count": len(missing_outcome_ids),
                "readout_ready": readout_ready,
            }
        )
        total_transition_count += len(records)
        total_missing_input_count += len(missing_input_ids)
        total_missing_outcome_count += len(missing_outcome_ids)
    source_admitted = bool(seed_results) and all(
        bool(result["source_admitted"]) for result in seed_results
    )
    readout_ready = source_admitted and all(
        bool(result["readout_ready"]) for result in seed_results
    )
    admission_status = (
        "readout-ready"
        if readout_ready
        else "capture-required"
        if source_admitted
        else "source-rejected"
    )
    return {
        "schema_version": GATE2_LONGITUDINAL_ADMISSION_SCHEMA_VERSION,
        "contract": asdict(active_contract),
        "selector_lineage": dict(selector_lineage),
        "seed_results": seed_results,
        "seed_count": len(seed_results),
        "total_transition_count": total_transition_count,
        "source_admitted": source_admitted,
        "missing_readout_input_count": total_missing_input_count,
        "missing_matched_outcome_count": total_missing_outcome_count,
        "readout_ready": readout_ready,
        "admission_status": admission_status,
        "inherited_gate2_evidence_level": "causal-supported",
        "longitudinal_verdict": "not-supported",
        "promotion_allowed": False,
        "selector_executed": False,
        "substrate_control_applied": False,
        "validation_delta_computed": False,
    }


def load_and_assess_gate2_longitudinal_readout(
    *,
    source_root: str | Path,
    selector_artifact_path: str | Path,
    companion_root: str | Path | None = None,
    contract: Gate2LongitudinalReadoutContract | None = None,
) -> tuple[dict[str, object], dict[str, str]]:
    """Load the immutable source and return admission plus input hashes."""

    active_contract = contract or Gate2LongitudinalReadoutContract()
    root = Path(source_root)
    companion = Path(companion_root) if companion_root is not None else root
    selector_path = Path(selector_artifact_path)
    selector_lineage = load_gate2_v35_selector_bundle(
        selector_path,
        contract=active_contract,
    )
    records_by_seed: dict[int, Sequence[Mapping[str, Any]]] = {}
    inputs_by_seed: dict[int, Mapping[str, Mapping[str, Any]]] = {}
    outcomes_by_seed: dict[int, Mapping[str, Mapping[str, Any]]] = {}
    input_hashes = {
        str(selector_path): _sha256_file(selector_path),
    }
    for seed in GATE11_LONGITUDINAL_SOURCE_SEEDS:
        seed_root = root / f"seed_{seed}"
        companion_seed_root = companion / f"seed_{seed}"
        transitions_path = seed_root / "transitions.jsonl"
        records = load_gate11_longitudinal_source_records(transitions_path)
        plans = build_gate11_longitudinal_source_plans(seed)
        validate_gate11_longitudinal_source_prefix(
            records=records,
            plans=plans,
        )
        input_path = (
            companion_seed_root / GATE2_LONGITUDINAL_INPUT_FILENAME
        )
        outcome_path = (
            companion_seed_root / GATE2_LONGITUDINAL_OUTCOME_FILENAME
        )
        records_by_seed[seed] = records
        inputs_by_seed[seed] = _load_companion_records(input_path)
        outcomes_by_seed[seed] = _load_companion_records(outcome_path)
        input_hashes[str(transitions_path)] = _sha256_file(transitions_path)
        if input_path.is_file():
            input_hashes[str(input_path)] = _sha256_file(input_path)
        if outcome_path.is_file():
            input_hashes[str(outcome_path)] = _sha256_file(outcome_path)
    assessment = assess_gate2_longitudinal_readout_admission(
        records_by_seed=records_by_seed,
        readout_inputs_by_seed=inputs_by_seed,
        matched_outcomes_by_seed=outcomes_by_seed,
        selector_lineage=selector_lineage,
        contract=active_contract,
    )
    return assessment, input_hashes


def export_gate2_longitudinal_readout_admission(
    *,
    source_root: str | Path,
    selector_artifact_path: str | Path,
    output_dir: str | Path,
    companion_root: str | Path | None = None,
) -> dict[str, object]:
    """Export a fail-closed 12-file Gate 2 admission evidence bundle."""

    target = Path(output_dir)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(
            f"Gate 2 longitudinal admission target is not empty: {target}"
        )
    target.mkdir(parents=True, exist_ok=True)
    assessment, input_hashes_before = (
        load_and_assess_gate2_longitudinal_readout(
            source_root=source_root,
            selector_artifact_path=selector_artifact_path,
            companion_root=companion_root,
        )
    )
    seed_results = tuple(assessment["seed_results"])
    manifest = {
        "schema_version": GATE2_LONGITUDINAL_ADMISSION_SCHEMA_VERSION,
        "run_kind": "read-only-admission",
        "source_root": str(Path(source_root)),
        "selector_artifact_path": str(Path(selector_artifact_path)),
        "companion_root": (
            str(Path(companion_root))
            if companion_root is not None
            else None
        ),
        "contract": assessment["contract"],
        "selector_lineage": assessment["selector_lineage"],
        "preregistered_plan": (
            ".cursor/plans/"
            "gate-2-longitudinal-v35-readout-admission_20260730.plan.md"
        ),
        "required_files": list(GATE2_LONGITUDINAL_REQUIRED_FILES),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_status": _git_output("status", "--short"),
    }
    _write_json(target / "manifest.yaml", manifest)
    _write_jsonl(
        target / "predictions.jsonl",
        tuple(
            {
                "seed": result["seed"],
                "expected_transition_count": 500,
                "expected_readout_input_dim": 8076,
                "expected_action_count": 22,
                "expected_matched_outcomes": (
                    "selected/zero/permutation-null realized continuation"
                ),
            }
            for result in seed_results
        ),
    )
    _write_jsonl(
        target / "outcomes.jsonl",
        tuple(
            {
                "seed": result["seed"],
                "source_admitted": result["source_admitted"],
                "readout_ready": result["readout_ready"],
                "transition_count": result["transition_count"],
                "readout_input_count": result["readout_input_count"],
                "matched_outcome_count": result["matched_outcome_count"],
            }
            for result in seed_results
        ),
    )
    _write_jsonl(
        target / "prediction_errors.jsonl",
        tuple(
            {
                "seed": result["seed"],
                "missing_readout_input_count": result[
                    "missing_readout_input_count"
                ],
                "missing_matched_outcome_count": result[
                    "missing_matched_outcome_count"
                ],
                "gap_kind": (
                    "selector-aligned-evidence-contract"
                    if not result["readout_ready"]
                    else "none"
                ),
            }
            for result in seed_results
        ),
    )
    _write_jsonl(
        target / "segments.jsonl",
        tuple(
            {
                "seed": result["seed"],
                "consumer_session_boundary_interval": result[
                    "consumer_session_boundary_interval"
                ],
                "consumer_session_count": result["consumer_session_count"],
                "cross_session_gate": result["source_gates"][
                    "cross_session_count"
                ],
            }
            for result in seed_results
        ),
    )
    _write_jsonl(
        target / "credit.jsonl",
        tuple(
            {
                "seed": result["seed"],
                "credit_status": "not-evaluated",
                "reason": (
                    "selector-aligned matched outcomes are required before "
                    "credit or validation delta can be computed"
                ),
            }
            for result in seed_results
        ),
    )
    _write_jsonl(
        target / "action_selection.jsonl",
        tuple(
            {
                "seed": result["seed"],
                "selector_fingerprint": GATE2_V35_SELECTOR_FINGERPRINT,
                "selection_status": (
                    "ready-not-executed"
                    if result["readout_ready"]
                    else "blocked-missing-readout-contract"
                ),
                "selector_executed": False,
                "substrate_control_applied": False,
            }
            for result in seed_results
        ),
    )
    _write_json(target / "ablation_results.json", assessment)
    verdict = {
        "schema_version": GATE2_LONGITUDINAL_ADMISSION_SCHEMA_VERSION,
        "admission_status": assessment["admission_status"],
        "source_admitted": assessment["source_admitted"],
        "readout_ready": assessment["readout_ready"],
        "inherited_gate2_evidence_level": assessment[
            "inherited_gate2_evidence_level"
        ],
        "longitudinal_verdict": assessment["longitudinal_verdict"],
        "promotion_allowed": False,
        "validation_delta_computed": False,
        "next_action": (
            "capture-fullwidth-readout-inputs-and-matched-outcomes"
            if assessment["admission_status"] == "capture-required"
            else "run-preregistered-longitudinal-readout"
            if assessment["admission_status"] == "readout-ready"
            else "repair-source-provenance"
        ),
    }
    _write_json(target / "promotion_verdict.json", verdict)
    assessment_after, input_hashes_after = (
        load_and_assess_gate2_longitudinal_readout(
            source_root=source_root,
            selector_artifact_path=selector_artifact_path,
            companion_root=companion_root,
        )
    )
    inputs_unchanged = input_hashes_before == input_hashes_after
    assessment_reproducible = assessment == assessment_after
    rollback = {
        "schema_version": GATE2_LONGITUDINAL_ADMISSION_SCHEMA_VERSION,
        "read_only": True,
        "source_and_selector_hashes_before": input_hashes_before,
        "source_and_selector_hashes_after": input_hashes_after,
        "source_and_selector_unchanged": inputs_unchanged,
        "assessment_reproducible": assessment_reproducible,
        "selector_updated": False,
        "substrate_control_applied": False,
        "runtime_owner_state_written": False,
        "rollback": "delete this admission artifact directory",
    }
    if not inputs_unchanged or not assessment_reproducible:
        raise RuntimeError(
            "Gate 2 longitudinal admission violated read-only reproducibility"
        )
    _write_json(target / "rollback_evidence.json", rollback)
    _write_jsonl(
        target / "state_diff.jsonl",
        (
            {
                "source_and_selector_unchanged": inputs_unchanged,
                "selector_updated": False,
                "substrate_control_applied": False,
                "runtime_owner_state_written": False,
            },
        ),
    )
    report = (
        "# Gate 2 longitudinal v35 readout admission\n\n"
        f"- source admitted: `{assessment['source_admitted']}`\n"
        f"- admission status: `{assessment['admission_status']}`\n"
        f"- settled transitions: `{assessment['total_transition_count']}`\n"
        f"- missing 8076-d readout inputs: "
        f"`{assessment['missing_readout_input_count']}`\n"
        f"- missing selector-aligned outcomes: "
        f"`{assessment['missing_matched_outcome_count']}`\n"
        f"- selector executed: `{assessment['selector_executed']}`\n"
        f"- substrate control applied: "
        f"`{assessment['substrate_control_applied']}`\n"
        f"- longitudinal verdict: "
        f"`{assessment['longitudinal_verdict']}`\n\n"
        "The source-scale and real-substrate gates are assessed separately "
        "from selector readiness. Fixed source task outcomes are not treated "
        "as outcomes of an unexecuted selector.\n"
    )
    (target / "report.md").write_text(report, encoding="utf-8")
    freeze_manifest = {
        name: _sha256_file(target / name)
        for name in GATE2_LONGITUDINAL_REQUIRED_FILES
    }
    _write_json(target / "freeze_manifest.json", freeze_manifest)
    return verdict
