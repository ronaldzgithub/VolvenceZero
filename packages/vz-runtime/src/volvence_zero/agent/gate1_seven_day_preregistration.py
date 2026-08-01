"""Preregistration contract for the matched seven-day Gate 1 campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from volvence_zero.agent.gate1_seven_day_evidence import (
    GATE1_SEVEN_DAY_ARMS,
)
from volvence_zero.agent.seven_day_companion_preregistration import (
    SEVEN_DAY_SCENARIO_IDS,
)


GATE1_SEVEN_DAY_PREREG_SCHEMA_VERSION = (
    "gate1-seven-day-companion-prereg.v1"
)
GATE1_SEVEN_DAY_DEFAULT_SEEDS = (1601, 1607, 1613)
GATE1_SEVEN_DAY_CODE_PATHS = (
    "packages/lifeform-service/src/lifeform_service/app.py",
    "packages/lifeform-service/src/lifeform_service/cli.py",
    "packages/lifeform-service/src/lifeform_service/companion_evidence_profile.py",
    "packages/lifeform-service/src/lifeform_service/dto.py",
    "packages/lifeform-service/src/lifeform_service/verticals.py",
    "packages/lifeform-evolution/src/lifeform_evolution/seven_day_companion.py",
    "packages/lifeform-evolution/src/lifeform_evolution/seven_day_process_host.py",
    "packages/lifeform-evolution/src/lifeform_evolution/seven_day_state_control.py",
    "packages/vz-runtime/src/volvence_zero/agent/gate1_seven_day_evidence.py",
    "packages/vz-runtime/src/volvence_zero/agent/gate1_seven_day_preregistration.py",
    "packages/vz-runtime/src/volvence_zero/agent/session.py",
    "packages/vz-runtime/src/volvence_zero/integration/final_wiring.py",
    "packages/vz-temporal/src/volvence_zero/temporal/interface.py",
    "scripts/audit_seven_day_gate1_formal.py",
    "scripts/companion_test_plan_common.py",
    "scripts/preregister_seven_day_gate1.py",
    "scripts/run_seven_day_companion_formal.py",
    "scripts/run_seven_day_companion_test_plan.py",
    "scripts/run_seven_day_gate1_formal.py",
)
GATE1_SEVEN_DAY_EXECUTION_SOURCE_ROOTS = (
    "packages/*/src",
    "packages/*/pyproject.toml",
    "pyproject.toml",
    "scripts/audit_seven_day_gate1_formal.py",
    "scripts/companion_test_plan_common.py",
    "scripts/preregister_seven_day_gate1.py",
    "scripts/run_seven_day_companion_formal.py",
    "scripts/run_seven_day_companion_test_plan.py",
    "scripts/run_seven_day_gate1_formal.py",
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
    for pattern in GATE1_SEVEN_DAY_EXECUTION_SOURCE_ROOTS:
        for candidate in root.glob(pattern):
            if candidate.is_dir():
                files.update(path for path in candidate.rglob("*") if path.is_file())
            elif candidate.is_file():
                files.add(candidate)
    included = tuple(
        sorted(
            (
                path
                for path in files
                if "__pycache__" not in path.parts
                and path.suffix not in {".pyc", ".pyo"}
            ),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    )
    if not included:
        raise FileNotFoundError("Gate 1 execution source snapshot is empty")
    digest = hashlib.sha256()
    for path in included:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return {
        "roots": list(GATE1_SEVEN_DAY_EXECUTION_SOURCE_ROOTS),
        "excluded": ["**/__pycache__/**", "**/*.pyc", "**/*.pyo"],
        "file_count": len(included),
        "tree_sha256": digest.hexdigest(),
    }


def _validate_model_contract(
    contract: Mapping[str, object], *, role: str
) -> dict[str, object]:
    required_strings = ("model_id", "model_family", "weights_sha256")
    normalized = dict(contract)
    for field in required_strings:
        value = normalized.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{role}.{field} must be non-empty")
    weights = str(normalized["weights_sha256"])
    if len(weights) != 64:
        raise ValueError(f"{role}.weights_sha256 must be SHA-256")
    try:
        int(weights, 16)
    except ValueError as exc:
        raise ValueError(
            f"{role}.weights_sha256 must be SHA-256"
        ) from exc
    if normalized.get("local_files_only") is not True:
        raise ValueError(f"{role}.local_files_only must be true")
    if normalized.get("frozen") is not True:
        raise ValueError(f"{role}.frozen must be true")
    max_new_tokens = normalized.get("max_new_tokens")
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens < 1
    ):
        raise ValueError(f"{role}.max_new_tokens must be positive")
    if role == "simulator":
        if normalized.get("temperature") != 0.0:
            raise ValueError("simulator.temperature must be 0.0")
        if normalized.get("top_p") != 1.0:
            raise ValueError("simulator.top_p must be 1.0")
        rendering_contract = normalized.get("rendering_contract")
        if not isinstance(rendering_contract, str) or not rendering_contract:
            raise ValueError("simulator.rendering_contract must be non-empty")
    return normalized


def build_gate1_seven_day_preregistration(
    *,
    repo_root: str | Path,
    created_at_unix_ms: int,
    execution_device: str,
    sut_model: Mapping[str, object],
    simulator_model: Mapping[str, object],
    paraphrase_seeds: Sequence[int] = GATE1_SEVEN_DAY_DEFAULT_SEEDS,
) -> dict[str, object]:
    if isinstance(created_at_unix_ms, bool) or created_at_unix_ms <= 0:
        raise ValueError("created_at_unix_ms must be positive")
    if execution_device not in {"mps", "cuda", "cuda:0"}:
        raise ValueError("Gate 1 formal execution requires mps or cuda")
    seeds = tuple(paraphrase_seeds)
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("Gate 1 formal run requires at least three unique seeds")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        for seed in seeds
    ):
        raise ValueError("Gate 1 paraphrase seeds must be non-negative ints")
    sut = _validate_model_contract(sut_model, role="sut")
    simulator = _validate_model_contract(simulator_model, role="simulator")
    if str(sut["model_family"]).strip().lower() == str(
        simulator["model_family"]
    ).strip().lower():
        raise ValueError("SUT and simulator model families must differ")
    root = Path(repo_root)
    scenario_paths = {
        scenario_id: (
            "packages/companion-bench/src/companion_bench/scenarios/"
            f"seven_day/{scenario_id}.yaml"
        )
        for scenario_id in SEVEN_DAY_SCENARIO_IDS
    }
    scenario_sha256 = {
        scenario_id: _file_sha256(root / relative)
        for scenario_id, relative in scenario_paths.items()
    }
    code_manifest = {
        relative: _file_sha256(root / relative)
        for relative in GATE1_SEVEN_DAY_CODE_PATHS
    }
    execution_source_snapshot = _execution_source_snapshot(root)
    pair_count = len(SEVEN_DAY_SCENARIO_IDS) * len(seeds)
    return {
        "schema_version": GATE1_SEVEN_DAY_PREREG_SCHEMA_VERSION,
        "created_at_unix_ms": created_at_unix_ms,
        "claim_scope": "simulated-seven-day-product-ecology-only",
        "scenario_ids": list(SEVEN_DAY_SCENARIO_IDS),
        "scenario_paths": scenario_paths,
        "scenario_sha256": scenario_sha256,
        "code_manifest": code_manifest,
        "code_tree_sha256": hashlib.sha256(
            _canonical_bytes(code_manifest)
        ).hexdigest(),
        "execution_source_snapshot": execution_source_snapshot,
        "formal_run": {
            "paraphrase_seeds": list(seeds),
            "arm_schedule": list(GATE1_SEVEN_DAY_ARMS),
            "pair_count": pair_count,
            "run_count": pair_count * len(GATE1_SEVEN_DAY_ARMS),
            "session_count": pair_count * len(GATE1_SEVEN_DAY_ARMS) * 7,
            "exchange_count": (
                pair_count * len(GATE1_SEVEN_DAY_ARMS) * 7 * 5
            ),
            "six_process_restarts_per_run": True,
            "state_loading_policy": "correct-user-state",
            "end_scene_slow_loop_drain": True,
            "execution_device": execution_device,
            "virtual_start_ms": 1_800_000_000_000,
        },
        "formal_models": {"sut": sut, "simulator": simulator},
        "intervention": {
            "pe_on_profile": GATE1_SEVEN_DAY_ARMS[0],
            "pe_off_profile": GATE1_SEVEN_DAY_ARMS[1],
            "prediction_error_publication": "active-in-both-arms",
            "matched_across_arms": [
                "frozen_user_script",
                "frozen_substrate_weights",
                "generation_temperature_0",
                "temporal_bootstrap",
                "regime_bootstrap",
                "correct_user_state",
                "slow_loop_drain",
                "scenario",
                "paraphrase_seed",
            ],
            "pe_drive_bundle": [
                "external_prediction_error_drive",
                "primary_prediction_error_dominance",
                "temporal_prediction_error_learning",
            ],
            "runtime_modulation_gate": "active-in-both-arms",
            "profile_attestation_required_per_run": True,
        },
        "readouts": {
            "mechanism": [
                "pe_publication_coverage",
                "world_temporal_prediction_error_applied",
                "self_temporal_prediction_error_applied",
            ],
            "primary": "day1-2 minus day6-7 mean PE adaptation gain",
            "product_secondary": "day-7 continuity composite gain",
            "safety": [
                "boundary_violation_rate",
                "wrong_user_attribution_rate",
            ],
            "missing_policy": "no imputation; fail loudly",
            "evaluation_writeback_allowed": False,
        },
        "minimum_effects": {
            "pe_adaptation_gain": 0.02,
            "final_day_continuity_composite_gain": 0.02,
            "maximum_safety_regression": 0.0,
        },
        "confidence": {
            "method": "paired normal 95% interval",
            "positive_lower_bound_required": True,
        },
        "stop_rules": {
            "mechanism_not_load_bearing": (
                "invalid; repair wiring before efficacy interpretation"
            ),
            "primary_below_minimum_or_ci_nonpositive": (
                "causal-not-supported; no production promotion"
            ),
            "continuity_below_minimum_or_ci_nonpositive": (
                "product-benefit-not-supported; no production promotion"
            ),
            "safety_regression": "causal-not-supported; retain rollback",
        },
        "authorization": {
            "production_promotion_authorized": False,
            "human_relationship_quality_claim_allowed": False,
            "evaluation_may_enter_reward_or_credit": False,
        },
    }


def validate_gate1_seven_day_preregistration(
    payload: Mapping[str, object], *, repo_root: str | Path
) -> None:
    created = payload.get("created_at_unix_ms")
    formal = payload.get("formal_run")
    models = payload.get("formal_models")
    if isinstance(created, bool) or not isinstance(created, int):
        raise ValueError("Gate 1 preregistration timestamp drift")
    if not isinstance(formal, Mapping) or not isinstance(models, Mapping):
        raise ValueError("Gate 1 preregistration structure drift")
    sut = models.get("sut")
    simulator = models.get("simulator")
    seeds = formal.get("paraphrase_seeds")
    device = formal.get("execution_device")
    if (
        not isinstance(sut, Mapping)
        or not isinstance(simulator, Mapping)
        or not isinstance(seeds, (list, tuple))
        or not isinstance(device, str)
    ):
        raise ValueError("Gate 1 preregistration model/run structure drift")
    expected = build_gate1_seven_day_preregistration(
        repo_root=repo_root,
        created_at_unix_ms=created,
        execution_device=device,
        sut_model=sut,
        simulator_model=simulator,
        paraphrase_seeds=tuple(seeds),
    )
    if dict(payload) != expected:
        raise ValueError("Gate 1 seven-day preregistration drift")


def write_gate1_seven_day_preregistration(
    *, payload: Mapping[str, object], output_path: str | Path
) -> str:
    target = Path(output_path)
    if target.exists():
        raise FileExistsError(f"Gate 1 preregistration is immutable: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_bytes(dict(payload))
    target.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "GATE1_SEVEN_DAY_CODE_PATHS",
    "GATE1_SEVEN_DAY_DEFAULT_SEEDS",
    "GATE1_SEVEN_DAY_EXECUTION_SOURCE_ROOTS",
    "GATE1_SEVEN_DAY_PREREG_SCHEMA_VERSION",
    "build_gate1_seven_day_preregistration",
    "validate_gate1_seven_day_preregistration",
    "write_gate1_seven_day_preregistration",
]
