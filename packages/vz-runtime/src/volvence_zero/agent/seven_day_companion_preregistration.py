"""Frozen preregistration for the seven-day simulated companion experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from volvence_zero.agent.seven_day_companion_evidence import (
    SEVEN_DAY_ALL_ARMS,
    SEVEN_DAY_METRICS,
    SEVEN_DAY_PREREG_SCHEMA_VERSION,
)


SEVEN_DAY_SCENARIO_IDS = (
    "F1-seven-day-warmth-researcher",
    "F2-seven-day-repair-researcher",
    "F1-seven-day-warmth-nurse",
    "F2-seven-day-repair-nurse",
    "F1-seven-day-warmth-designer",
    "F2-seven-day-repair-designer",
)
SEVEN_DAY_FORMAL_SEEDS = (1501,)
SEVEN_DAY_PREREG_CODE_PATHS = (
    "packages/lifeform-service/src/lifeform_service/app.py",
    "packages/lifeform-evolution/src/lifeform_evolution/seven_day_companion.py",
    "packages/lifeform-evolution/src/lifeform_evolution/seven_day_process_host.py",
    "packages/lifeform-evolution/src/lifeform_evolution/seven_day_state_control.py",
    "packages/companion-bench/src/companion_bench/seven_day_driver.py",
    "packages/companion-bench/src/companion_bench/user_simulator.py",
    "packages/vz-cognition/src/volvence_zero/evaluation/relationship_continuity.py",
    "packages/vz-runtime/src/volvence_zero/agent/seven_day_companion_evidence.py",
    "packages/vz-runtime/src/volvence_zero/agent/gate811_simulated_capture.py",
    "packages/vz-runtime/src/volvence_zero/agent/seven_day_companion_preregistration.py",
    "scripts/run_seven_day_companion_formal.py",
)
SEVEN_DAY_EXECUTION_SOURCE_ROOTS = (
    "packages/*/src",
    "packages/*/pyproject.toml",
    "pyproject.toml",
    "scripts/preregister_seven_day_companion_simulated.py",
    "scripts/run_seven_day_companion_formal.py",
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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _execution_source_snapshot(root: Path) -> dict[str, object]:
    files: set[Path] = set()
    for pattern in SEVEN_DAY_EXECUTION_SOURCE_ROOTS:
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
        raise FileNotFoundError("seven-day execution source snapshot is empty")
    digest = hashlib.sha256()
    for path in included:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return {
        "roots": list(SEVEN_DAY_EXECUTION_SOURCE_ROOTS),
        "excluded": ["**/__pycache__/**", "**/*.pyc", "**/*.pyo"],
        "file_count": len(included),
        "tree_sha256": digest.hexdigest(),
    }


def build_seven_day_companion_preregistration(
    *,
    repo_root: str | Path,
    created_at_unix_ms: int,
) -> dict[str, object]:
    """Build the immutable simulated-longitudinal protocol."""

    if isinstance(created_at_unix_ms, bool) or created_at_unix_ms <= 0:
        raise ValueError("created_at_unix_ms must be positive")
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
        for relative in SEVEN_DAY_PREREG_CODE_PATHS
    }
    execution_source_snapshot = _execution_source_snapshot(root)
    return {
        "schema_version": SEVEN_DAY_PREREG_SCHEMA_VERSION,
        "created_at_unix_ms": created_at_unix_ms,
        "claim_scope": "simulated-user-real-lifecycle-only",
        "scenario_ids": list(SEVEN_DAY_SCENARIO_IDS),
        "scenario_paths": scenario_paths,
        "scenario_sha256": scenario_sha256,
        "code_manifest": code_manifest,
        "code_tree_sha256": hashlib.sha256(
            _canonical_bytes(code_manifest)
        ).hexdigest(),
        "execution_source_snapshot": execution_source_snapshot,
        "formal_run": {
            "paraphrase_seeds": list(SEVEN_DAY_FORMAL_SEEDS),
            "arm_schedule": list(SEVEN_DAY_ALL_ARMS),
            "case_count": len(SEVEN_DAY_SCENARIO_IDS)
            * len(SEVEN_DAY_FORMAL_SEEDS),
            "run_count": len(SEVEN_DAY_SCENARIO_IDS)
            * len(SEVEN_DAY_FORMAL_SEEDS)
            * len(SEVEN_DAY_ALL_ARMS),
            "session_count": len(SEVEN_DAY_SCENARIO_IDS)
            * len(SEVEN_DAY_FORMAL_SEEDS)
            * len(SEVEN_DAY_ALL_ARMS)
            * 7,
            "exchange_count": len(SEVEN_DAY_SCENARIO_IDS)
            * len(SEVEN_DAY_FORMAL_SEEDS)
            * len(SEVEN_DAY_ALL_ARMS)
            * 7
            * 5,
            "single_writer_per_user_namespace": True,
            "six_process_restarts_per_run": True,
            "virtual_calendar_day_gap_ms": 86_400_000,
            "virtual_start_ms": 1_800_000_000_000,
            "execution_device": "mps",
        },
        "formal_models": {
            "sut": {
                "model_id": "HuggingFaceTB/SmolLM2-360M-Instruct",
                "model_family": "smollm",
                "weights_sha256": "f2a59a2f8f71f33baf444c37bf0ea1901211e237d2b970fcb91335c278b4b9ad",
                "local_files_only": True,
                "frozen": True,
                "max_new_tokens": 96,
            },
            "simulator": {
                "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
                "model_family": "qwen",
                "weights_sha256": "fb8c44c48b8359fdd306cdc5f473d7c04d88955013f0dd8549f266e248194da4",
                "local_files_only": True,
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": 12,
                "rendering_contract": (
                    "typed-FSM substantive draft plus closed-list local-LLM "
                    "style-opener selection"
                ),
            },
        },
        "source_requirements": {
            "simulator_backend": "real-llm-or-local-open-weight",
            "sut_backend": "real-frozen-substrate",
            "deterministic_fake_allowed_in_formal": False,
            "simulator_and_sut_model_families_must_differ": True,
            "judge_optional_secondary_only": True,
            "judge_family_must_differ_when_present": True,
            "model_and_adapter_fingerprint_exact_across_arms": True,
            "frozen_user_script_exact_across_arms": True,
            "typed_fsm_substantive_content_cannot_be_rewritten_by_simulator": True,
        },
        "interventions": {
            "correct-user-state": "load same-user owner snapshots in order",
            "stateless": "load no prior owner snapshot at each new day",
            "swapped-user-state": (
                "load a matched donor user's owner snapshots only"
            ),
            "shuffled-history": (
                "load same-user owner snapshots in preregistered shuffled order"
            ),
            "shuffled_history_source_days_after_day_1_to_6": [
                1,
                1,
                2,
                1,
                4,
                3,
            ],
            "sleep-consolidation": "drain the end-scene slow loop",
            "no-sleep": "do not drain the end-scene slow loop",
            "only_manipulated_variables": [
                "state_loading_policy",
                "end_scene_slow_loop_drain",
            ],
            "state_archive_and_loaded_copy_sha256_required": True,
            "measurement_checkpoint_excluded_from_state_intervention": True,
            "measurement_checkpoint_preserved_sha256_required": True,
            "state_files_are_archived_without_deletion": True,
            "daily_console_probe_policy": (
                "keep first sorted pending memory proposal; delete second sorted "
                "pending memory proposal using content_inaccurate"
            ),
        },
        "readouts": {
            "daily_owner_metrics": list(SEVEN_DAY_METRICS),
            "primary_state": "day-7 direction-normalized continuity composite",
            "primary_sleep": "day-2..7 cold-start continuity composite",
            "callback": "typed callback-opportunity callback_hit_rate",
            "fsm_probe_pass_rate": "secondary when typed scorer is present",
            "llm_judge": "secondary-only",
            "missing_metric_policy": "no imputation; fail metric-coverage gate",
            "console_metrics_source": "public relationship-memory action API",
            "evaluation_writeback_allowed": False,
        },
        "minimum_effects": {
            "final_day_continuity_composite_gain": 0.02,
            "callback_hit_rate_gain": 0.02,
            "cold_start_continuity_composite_gain": 0.02,
        },
        "confidence": {
            "method": "paired normal 95% interval",
            "lower_bound_must_exceed_zero": True,
            "all_preregistered_cases_required": True,
        },
        "kill_conditions": {
            "correct_not_better_than_stateless": (
                "shrink continuity claim to typed owner-metric behavior only"
            ),
            "sleep_not_better_than_no_sleep": (
                "do not make next-day consolidation product claim"
            ),
            "missing_owner_metric": "no causal verdict; repair instrumentation",
            "arm_matching_drift": "abort run before analysis",
            "model_family_overlap": "abort run before analysis",
        },
        "authorization": {
            "simulated_result_is_real_user_product_value": False,
            "human_anchor_claim_allowed_without_ratings": False,
            "production_promotion_authorized": False,
            "evaluation_may_enter_reward_or_credit": False,
        },
    }


def validate_seven_day_companion_preregistration(
    payload: Mapping[str, object],
    *,
    repo_root: str | Path,
) -> None:
    created = payload.get("created_at_unix_ms")
    if isinstance(created, bool) or not isinstance(created, int):
        raise ValueError("seven-day preregistration timestamp drift")
    expected = build_seven_day_companion_preregistration(
        repo_root=repo_root,
        created_at_unix_ms=created,
    )
    if dict(payload) != expected:
        raise ValueError("seven-day companion preregistration drift")


def write_seven_day_companion_preregistration(
    *,
    payload: Mapping[str, object],
    output_path: str | Path,
) -> str:
    target = Path(output_path)
    if target.exists():
        raise FileExistsError(
            f"seven-day preregistration is immutable: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical_bytes(dict(payload))
    target.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


__all__ = [
    "SEVEN_DAY_FORMAL_SEEDS",
    "SEVEN_DAY_PREREG_CODE_PATHS",
    "SEVEN_DAY_EXECUTION_SOURCE_ROOTS",
    "SEVEN_DAY_SCENARIO_IDS",
    "build_seven_day_companion_preregistration",
    "validate_seven_day_companion_preregistration",
    "write_seven_day_companion_preregistration",
]
