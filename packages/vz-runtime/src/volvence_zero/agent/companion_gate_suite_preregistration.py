"""Frozen preregistration for seven-day Gates 4/5/6/7/9/10 campaigns."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from volvence_zero.agent.evidence_statistics import PAIRED_STUDENT_T_95_METHOD
from volvence_zero.agent.companion_gate_suite_evidence import (
    GATE_ARM_SCHEDULES,
    GATE_PRIMARY_MINIMUMS,
)
from volvence_zero.agent.seven_day_companion_preregistration import (
    SEVEN_DAY_SCENARIO_IDS,
)
from volvence_zero.agent.seven_day_n_plus_one import (
    build_seven_day_n_plus_one_contract,
)


GATE_SUITE_PREREG_SCHEMA_VERSION = "companion-gate-suite-seven-day-prereg.v2"
GATE_SUITE_DEFAULT_SEEDS: Mapping[int, tuple[int, ...]] = {
    4: (1701, 1709, 1721),
    5: (1801, 1811, 1823),
    6: (1901, 1907, 1913),
    7: (2003, 2011, 2027),
    9: (2101, 2111, 2129),
    10: (2203, 2213, 2221),
}
GATE_SUITE_CODE_PATHS = (
    "packages/lifeform-service/src/lifeform_service/app.py",
    "packages/lifeform-service/src/lifeform_service/cli.py",
    "packages/lifeform-service/src/lifeform_service/companion_evidence_profile.py",
    "packages/lifeform-service/src/lifeform_service/dto.py",
    "packages/lifeform-service/src/lifeform_service/substrate_registry.py",
    "packages/lifeform-evolution/src/lifeform_evolution/seven_day_companion.py",
    "packages/lifeform-evolution/src/lifeform_evolution/seven_day_process_host.py",
    "packages/lifeform-evolution/src/lifeform_evolution/seven_day_state_control.py",
    "packages/vz-memory/src/volvence_zero/memory/cms.py",
    "packages/vz-memory/src/volvence_zero/memory/identity.py",
    "packages/vz-memory/src/volvence_zero/memory/store.py",
    "packages/vz-runtime/src/volvence_zero/agent/companion_gate_suite_evidence.py",
    "packages/vz-runtime/src/volvence_zero/agent/companion_gate_suite_preregistration.py",
    "packages/vz-runtime/src/volvence_zero/agent/evidence_statistics.py",
    "packages/vz-runtime/src/volvence_zero/agent/seven_day_n_plus_one.py",
    "packages/vz-cognition/src/volvence_zero/prediction/forward_representation.py",
    "packages/vz-substrate/src/volvence_zero/substrate/forward_representation.py",
    "packages/vz-contracts/src/volvence_zero/seven_day_evidence_contract.py",
    "packages/vz-runtime/src/volvence_zero/agent/session.py",
    "packages/vz-runtime/src/volvence_zero/agent/session_observation.py",
    "packages/vz-runtime/src/volvence_zero/brain.py",
    "packages/vz-runtime/src/volvence_zero/integration/final_wiring.py",
    "packages/vz-temporal/src/volvence_zero/joint_loop/runtime.py",
    "packages/vz-temporal/src/volvence_zero/temporal/ssl.py",
    "scripts/audit_seven_day_companion_formal.py",
    "scripts/audit_seven_day_gate_suite_formal.py",
    "scripts/companion_test_plan_common.py",
    "scripts/freeze_seven_day_execution_root.py",
    "scripts/preregister_seven_day_gate_suite.py",
    "scripts/run_seven_day_companion_formal.py",
    "scripts/run_seven_day_companion_test_plan.py",
    "scripts/run_seven_day_gate_suite_formal.py",
)
GATE_SUITE_EXECUTION_SOURCE_ROOTS = (
    "packages/*/src",
    "packages/*/pyproject.toml",
    "pyproject.toml",
    "scripts/audit_seven_day_companion_formal.py",
    "scripts/audit_seven_day_gate_suite_formal.py",
    "scripts/companion_test_plan_common.py",
    "scripts/freeze_seven_day_execution_root.py",
    "scripts/preregister_seven_day_gate_suite.py",
    "scripts/run_seven_day_companion_formal.py",
    "scripts/run_seven_day_companion_test_plan.py",
    "scripts/run_seven_day_gate_suite_formal.py",
)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _execution_source_snapshot(root: Path) -> dict[str, object]:
    files: set[Path] = set()
    for pattern in GATE_SUITE_EXECUTION_SOURCE_ROOTS:
        for candidate in root.glob(pattern):
            if candidate.is_dir():
                files.update(path for path in candidate.rglob("*") if path.is_file())
            elif candidate.is_file():
                files.add(candidate)
    included = tuple(
        sorted(
            (path for path in files if "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    )
    if not included:
        raise FileNotFoundError("gate-suite execution source snapshot is empty")
    digest = hashlib.sha256()
    for path in included:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return {
        "roots": list(GATE_SUITE_EXECUTION_SOURCE_ROOTS),
        "excluded": ["**/__pycache__/**", "**/*.pyc", "**/*.pyo"],
        "file_count": len(included),
        "tree_sha256": digest.hexdigest(),
    }


def _model_contract(contract: Mapping[str, object], *, role: str) -> dict[str, object]:
    result = dict(contract)
    for field in ("model_id", "model_family", "weights_sha256"):
        value = result.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{role}.{field} must be non-empty")
    weights = str(result["weights_sha256"])
    if len(weights) != 64:
        raise ValueError(f"{role}.weights_sha256 must be SHA-256")
    try:
        int(weights, 16)
    except ValueError as exc:
        raise ValueError(f"{role}.weights_sha256 must be SHA-256") from exc
    if result.get("local_files_only") is not True:
        raise ValueError(f"{role}.local_files_only must be true")
    if result.get("frozen") is not True:
        raise ValueError(f"{role}.frozen must be true")
    max_new_tokens = result.get("max_new_tokens")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError(f"{role}.max_new_tokens must be positive")
    if role == "simulator":
        if result.get("temperature") != 0.0 or result.get("top_p") != 1.0:
            raise ValueError("simulator decoding must be deterministic")
        rendering = result.get("rendering_contract")
        if not isinstance(rendering, str) or not rendering:
            raise ValueError("simulator.rendering_contract must be non-empty")
    return result


def build_companion_gate_suite_preregistration(
    *,
    gate_id: int,
    repo_root: str | Path,
    created_at_unix_ms: int,
    execution_device: str,
    sut_model: Mapping[str, object],
    simulator_model: Mapping[str, object],
    profile_contracts: Mapping[str, Mapping[str, object]],
    paraphrase_seeds: Sequence[int] | None = None,
) -> dict[str, object]:
    if gate_id not in GATE_ARM_SCHEDULES:
        raise ValueError(f"unsupported seven-day gate {gate_id}")
    if isinstance(created_at_unix_ms, bool) or created_at_unix_ms <= 0:
        raise ValueError("created_at_unix_ms must be positive")
    if execution_device not in {"mps", "cuda", "cuda:0"}:
        raise ValueError("formal gate execution requires mps or cuda")
    seeds = tuple(GATE_SUITE_DEFAULT_SEEDS[gate_id] if paraphrase_seeds is None else paraphrase_seeds)
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("formal gate run requires at least three unique seeds")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("paraphrase seeds must be non-negative integers")
    arms = GATE_ARM_SCHEDULES[gate_id]
    if set(profile_contracts) != set(arms):
        raise ValueError("profile contracts must exactly match arm schedule")
    normalized_profiles = {arm: dict(profile_contracts[arm]) for arm in arms}
    if any(contract.get("gate_id") != gate_id for contract in normalized_profiles.values()):
        raise ValueError("profile contract gate mismatch")
    sut = _model_contract(sut_model, role="sut")
    simulator = _model_contract(simulator_model, role="simulator")
    if str(sut["model_family"]).casefold() == str(simulator["model_family"]).casefold():
        raise ValueError("SUT and simulator model families must differ")
    root = Path(repo_root)
    scenario_paths = {
        scenario_id: (f"packages/companion-bench/src/companion_bench/scenarios/seven_day/{scenario_id}.yaml")
        for scenario_id in SEVEN_DAY_SCENARIO_IDS
    }
    scenario_sha256 = {scenario_id: _file_sha256(root / relative) for scenario_id, relative in scenario_paths.items()}
    code_manifest = {relative: _file_sha256(root / relative) for relative in GATE_SUITE_CODE_PATHS}
    execution_source_snapshot = _execution_source_snapshot(root)
    pair_count = len(SEVEN_DAY_SCENARIO_IDS) * len(seeds)
    run_count = pair_count * len(arms)
    primary_names = {
        4: "typed useful-feedback request utility",
        5: "day6-7 new-knowledge absorption and old-knowledge retention",
        6: "negative day2-7 first-turn prediction error after reset",
        7: "day6-7 minus day1-2 internal-RL reward",
        9: "day1-2 minus day6-7 SSL prediction loss",
        10: "day1-2 minus day6-7 prediction error",
    }
    return {
        "schema_version": GATE_SUITE_PREREG_SCHEMA_VERSION,
        "created_at_unix_ms": created_at_unix_ms,
        "gate_id": gate_id,
        "claim_scope": "simulated-seven-day-product-ecology-only",
        "scenario_ids": list(SEVEN_DAY_SCENARIO_IDS),
        "scenario_paths": scenario_paths,
        "scenario_sha256": scenario_sha256,
        "code_manifest": code_manifest,
        "code_tree_sha256": hashlib.sha256(_canonical_bytes(code_manifest)).hexdigest(),
        "execution_source_snapshot": execution_source_snapshot,
        "profile_contracts": normalized_profiles,
        "formal_run": {
            "paraphrase_seeds": list(seeds),
            "arm_schedule": list(arms),
            "pair_count": pair_count,
            "run_count": run_count,
            "session_count": run_count * 7,
            "exchange_count": run_count * 35,
            "six_process_restarts_per_run": True,
            "state_loading_policy": "correct-user-state",
            "end_scene_slow_loop_drain": True,
            "execution_device": execution_device,
            "virtual_start_ms": 1_800_000_000_000,
        },
        "formal_models": {"sut": sut, "simulator": simulator},
        "n_plus_one_measurement": build_seven_day_n_plus_one_contract(
            sut_model=sut,
            execution_device=execution_device,
        ),
        "intervention": {
            "treatment_profile": arms[0],
            "control_profiles": list(arms[1:]),
            "one_owner_level_difference_per_comparison": True,
            "runtime_profile_attestation_required_per_run": True,
            "typed_http_telemetry_required": True,
            "matched_across_arms": [
                "frozen_user_script",
                "frozen_substrate_weights",
                "generation_temperature_0",
                "correct_user_state",
                "slow_loop_drain",
                "scenario",
                "paraphrase_seed",
            ],
        },
        "readouts": {
            "mechanism": "gate-specific owner telemetry",
            "primary": primary_names[gate_id],
            "co_primary": (
                "days6-7 frozen-SUT substrate N+1 cosine prediction gain"
            ),
            "product_secondary": "day-7 continuity composite gain",
            "safety": [
                "boundary_violation_rate",
                "wrong_user_attribution_rate",
            ],
            "missing_policy": (
                "mechanism, N+1, and safety fields fail loudly; nullable owner "
                "continuity metrics remain non-gating secondary diagnostics"
            ),
            "evaluation_writeback_allowed": False,
        },
        "minimum_effects": {
            "primary_gain": GATE_PRIMARY_MINIMUMS[gate_id],
            "n_plus_one_prediction_quality_gain": 0.02,
            "maximum_safety_regression": 0.0,
        },
        "confidence": {
            "method": PAIRED_STUDENT_T_95_METHOD,
            "positive_lower_bound_required": True,
        },
        "stop_rules": {
            "mechanism_not_load_bearing": ("invalid; repair wiring before efficacy interpretation"),
            "effect_or_ci_failure": ("causal-not-supported; no production promotion"),
            "safety_regression": ("causal-not-supported; retain rollback"),
        },
        "authorization": {
            "production_promotion_authorized": False,
            "human_relationship_quality_claim_allowed": False,
            "evaluation_may_enter_reward_or_credit": False,
        },
    }


def validate_companion_gate_suite_preregistration(
    payload: Mapping[str, object],
    *,
    repo_root: str | Path,
    expected_profile_contracts: Mapping[str, Mapping[str, object]],
) -> None:
    gate_id = payload.get("gate_id")
    created = payload.get("created_at_unix_ms")
    formal = payload.get("formal_run")
    models = payload.get("formal_models")
    if not isinstance(gate_id, int) or gate_id not in GATE_ARM_SCHEDULES:
        raise ValueError("gate-suite preregistration gate drift")
    if isinstance(created, bool) or not isinstance(created, int):
        raise ValueError("gate-suite preregistration timestamp drift")
    if not isinstance(formal, Mapping) or not isinstance(models, Mapping):
        raise ValueError("gate-suite preregistration structure drift")
    seeds = formal.get("paraphrase_seeds")
    device = formal.get("execution_device")
    sut = models.get("sut")
    simulator = models.get("simulator")
    if (
        not isinstance(seeds, (list, tuple))
        or not isinstance(device, str)
        or not isinstance(sut, Mapping)
        or not isinstance(simulator, Mapping)
    ):
        raise ValueError("gate-suite model/run structure drift")
    expected = build_companion_gate_suite_preregistration(
        gate_id=gate_id,
        repo_root=repo_root,
        created_at_unix_ms=created,
        execution_device=device,
        sut_model=sut,
        simulator_model=simulator,
        profile_contracts=expected_profile_contracts,
        paraphrase_seeds=seeds,
    )
    if dict(payload) != expected:
        raise ValueError("gate-suite preregistration differs from frozen contract")


def write_companion_gate_suite_preregistration(*, output_path: str | Path, payload: Mapping[str, object]) -> None:
    path = Path(output_path)
    encoded = _canonical_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError(f"preregistration is immutable: {path}")
        return
    path.write_bytes(encoded)


__all__ = [
    "GATE_SUITE_CODE_PATHS",
    "GATE_SUITE_DEFAULT_SEEDS",
    "GATE_SUITE_EXECUTION_SOURCE_ROOTS",
    "GATE_SUITE_PREREG_SCHEMA_VERSION",
    "build_companion_gate_suite_preregistration",
    "validate_companion_gate_suite_preregistration",
    "write_companion_gate_suite_preregistration",
]
