"""Frozen preregistration for the seven-day simulated companion experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from volvence_zero.agent.seven_day_companion_evidence import (
    SEVEN_DAY_ALL_ARMS,
    SEVEN_DAY_CHARACTER_STACK_PREREG_SCHEMA_VERSION,
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
    "packages/lifeform-service/src/lifeform_service/character_packages.py",
    "packages/lifeform-service/src/lifeform_service/cli.py",
    "packages/lifeform-service/src/lifeform_service/session_manager.py",
    "packages/lifeform-evolution/src/lifeform_evolution/seven_day_companion.py",
    "packages/lifeform-evolution/src/lifeform_evolution/seven_day_process_host.py",
    "packages/lifeform-evolution/src/lifeform_evolution/seven_day_state_control.py",
    "packages/companion-bench/src/companion_bench/seven_day_driver.py",
    "packages/companion-bench/src/companion_bench/user_simulator.py",
    "packages/vz-cognition/src/volvence_zero/evaluation/relationship_continuity.py",
    "packages/vz-runtime/src/volvence_zero/agent/seven_day_companion_evidence.py",
    "packages/vz-runtime/src/volvence_zero/agent/gate811_simulated_capture.py",
    "packages/vz-runtime/src/volvence_zero/agent/seven_day_companion_preregistration.py",
    "packages/vz-substrate/src/volvence_zero/substrate/common_adapter_bundle.py",
    "packages/vz-substrate/src/volvence_zero/substrate/residual_backend.py",
    "packages/lifeform-domain-character/src/lifeform_domain_character/character_package.py",
    "scripts/audit_gate811_simulated_capture.py",
    "scripts/audit_seven_day_companion_formal.py",
    "scripts/companion_test_plan_common.py",
    "scripts/freeze_seven_day_execution_root.py",
    "scripts/run_seven_day_companion_test_plan.py",
    "scripts/run_seven_day_companion_formal.py",
)
SEVEN_DAY_EXECUTION_SOURCE_ROOTS = (
    "packages/*/src",
    "packages/*/pyproject.toml",
    "pyproject.toml",
    "scripts/audit_gate811_simulated_capture.py",
    "scripts/audit_seven_day_companion_formal.py",
    "scripts/companion_test_plan_common.py",
    "scripts/freeze_seven_day_execution_root.py",
    "scripts/preregister_seven_day_companion_simulated.py",
    "scripts/run_gate811_simulated_capture.py",
    "scripts/run_seven_day_companion_test_plan.py",
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


def _require_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"seven-day {field} must be an object")
    return value


def _require_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"seven-day {field} must be a non-empty string")
    return value


def _validate_relative_artifact(
    *,
    root: Path,
    locator: object,
    expected_sha256: object,
    field: str,
) -> str:
    relative = Path(_require_string(locator, field=f"{field}.locator"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"seven-day {field}.locator must stay under repo_root")
    digest = _require_string(expected_sha256, field=f"{field}.sha256")
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _file_sha256(path)
    if actual != digest:
        raise ValueError(f"seven-day {field} artifact digest drift")
    return relative.as_posix()


def _validated_runtime_stack(
    *,
    repo_root: Path,
    runtime_stack: Mapping[str, object],
) -> dict[str, object]:
    required = {
        "mode",
        "vertical",
        "selected_character_id",
        "wiring_level",
        "sut_model_family",
        "sut_max_new_tokens",
        "common_adapter",
        "character_manifests",
    }
    if set(runtime_stack) != required:
        raise ValueError(
            "seven-day character runtime_stack fields drift; "
            f"missing={sorted(required - set(runtime_stack))}, "
            f"extra={sorted(set(runtime_stack) - required)}"
        )
    if runtime_stack.get("mode") != "base+common-adapter+character-package":
        raise ValueError("seven-day runtime_stack mode drift")
    if runtime_stack.get("wiring_level") != "active":
        raise ValueError("seven-day character runtime_stack must be ACTIVE")
    vertical = _require_string(runtime_stack.get("vertical"), field="runtime_stack.vertical")
    selected_character_id = _require_string(
        runtime_stack.get("selected_character_id"),
        field="runtime_stack.selected_character_id",
    )
    sut_model_family = _require_string(
        runtime_stack.get("sut_model_family"),
        field="runtime_stack.sut_model_family",
    ).lower()
    if sut_model_family == "smollm":
        raise ValueError("seven-day SUT and frozen simulator families must differ")
    sut_max_new_tokens = runtime_stack.get("sut_max_new_tokens")
    if isinstance(sut_max_new_tokens, bool) or not isinstance(sut_max_new_tokens, int) or sut_max_new_tokens <= 0:
        raise ValueError("seven-day runtime_stack.sut_max_new_tokens must be positive")

    common = _require_mapping(
        runtime_stack.get("common_adapter"),
        field="runtime_stack.common_adapter",
    )
    common_required = {
        "locator",
        "sha256",
        "bundle_id",
        "common_adapter_version",
        "compatibility_fingerprint",
        "base_model_id",
        "base_model_weights_sha256",
    }
    if set(common) != common_required:
        raise ValueError("seven-day common_adapter contract fields drift")
    common_locator = _validate_relative_artifact(
        root=repo_root,
        locator=common.get("locator"),
        expected_sha256=common.get("sha256"),
        field="runtime_stack.common_adapter",
    )
    normalized_common = {
        "locator": common_locator,
        "sha256": _require_string(common.get("sha256"), field="common_adapter.sha256"),
        "bundle_id": _require_string(common.get("bundle_id"), field="common_adapter.bundle_id"),
        "common_adapter_version": _require_string(
            common.get("common_adapter_version"),
            field="common_adapter.common_adapter_version",
        ),
        "compatibility_fingerprint": _require_string(
            common.get("compatibility_fingerprint"),
            field="common_adapter.compatibility_fingerprint",
        ),
        "base_model_id": _require_string(common.get("base_model_id"), field="common_adapter.base_model_id"),
        "base_model_weights_sha256": _require_string(
            common.get("base_model_weights_sha256"),
            field="common_adapter.base_model_weights_sha256",
        ),
    }
    raw_manifests = runtime_stack.get("character_manifests")
    if not isinstance(raw_manifests, list) or not raw_manifests:
        raise ValueError("seven-day character_manifests must be a non-empty list")
    manifest_required = {
        "locator",
        "sha256",
        "package_id",
        "character_id",
        "prefix_package_id",
        "artifact_files",
    }
    manifests: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, raw_manifest in enumerate(raw_manifests):
        manifest = _require_mapping(
            raw_manifest,
            field=f"runtime_stack.character_manifests[{index}]",
        )
        if set(manifest) != manifest_required:
            raise ValueError("seven-day character manifest contract fields drift")
        character_id = _require_string(manifest.get("character_id"), field="character_manifest.character_id")
        if character_id in seen:
            raise ValueError(f"duplicate seven-day character_id {character_id!r}")
        seen.add(character_id)
        raw_artifact_files = manifest.get("artifact_files")
        if not isinstance(raw_artifact_files, list) or not raw_artifact_files:
            raise ValueError("seven-day ACTIVE character manifest must freeze nested artifact_files")
        artifact_files: list[dict[str, str]] = []
        for artifact_index, raw_artifact in enumerate(raw_artifact_files):
            artifact = _require_mapping(
                raw_artifact,
                field=(f"runtime_stack.character_manifests[{index}].artifact_files[{artifact_index}]"),
            )
            if set(artifact) != {"locator", "sha256"}:
                raise ValueError("seven-day nested character artifact fields drift")
            artifact_files.append(
                {
                    "locator": _validate_relative_artifact(
                        root=repo_root,
                        locator=artifact.get("locator"),
                        expected_sha256=artifact.get("sha256"),
                        field=(f"runtime_stack.character_manifests[{index}].artifact_files[{artifact_index}]"),
                    ),
                    "sha256": _require_string(
                        artifact.get("sha256"),
                        field="character_manifest.artifact_file.sha256",
                    ),
                }
            )
        manifests.append(
            {
                "locator": _validate_relative_artifact(
                    root=repo_root,
                    locator=manifest.get("locator"),
                    expected_sha256=manifest.get("sha256"),
                    field=f"runtime_stack.character_manifests[{index}]",
                ),
                "sha256": _require_string(manifest.get("sha256"), field="character_manifest.sha256"),
                "package_id": _require_string(
                    manifest.get("package_id"),
                    field="character_manifest.package_id",
                ),
                "character_id": character_id,
                "prefix_package_id": _require_string(
                    manifest.get("prefix_package_id"),
                    field="character_manifest.prefix_package_id",
                ),
                "artifact_files": artifact_files,
            }
        )
    if selected_character_id not in seen:
        raise ValueError("seven-day selected_character_id has no frozen character manifest")
    return {
        "mode": "base+common-adapter+character-package",
        "vertical": vertical,
        "selected_character_id": selected_character_id,
        "wiring_level": "active",
        "sut_model_family": sut_model_family,
        "sut_max_new_tokens": sut_max_new_tokens,
        "common_adapter": normalized_common,
        "character_manifests": manifests,
    }


def _execution_source_snapshot(
    root: Path,
    *,
    extra_paths: tuple[str, ...] = (),
) -> dict[str, object]:
    files: set[Path] = set()
    source_roots = tuple(dict.fromkeys((*SEVEN_DAY_EXECUTION_SOURCE_ROOTS, *extra_paths)))
    for pattern in source_roots:
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
        "roots": list(source_roots),
        "excluded": ["**/__pycache__/**", "**/*.pyc", "**/*.pyo"],
        "file_count": len(included),
        "tree_sha256": digest.hexdigest(),
    }


def build_seven_day_companion_preregistration(
    *,
    repo_root: str | Path,
    created_at_unix_ms: int,
    runtime_stack: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the immutable simulated-longitudinal protocol."""

    if isinstance(created_at_unix_ms, bool) or created_at_unix_ms <= 0:
        raise ValueError("created_at_unix_ms must be positive")
    root = Path(repo_root)
    normalized_runtime_stack = (
        _validated_runtime_stack(
            repo_root=root,
            runtime_stack=runtime_stack,
        )
        if runtime_stack is not None
        else None
    )
    scenario_paths = {
        scenario_id: (f"packages/companion-bench/src/companion_bench/scenarios/seven_day/{scenario_id}.yaml")
        for scenario_id in SEVEN_DAY_SCENARIO_IDS
    }
    scenario_sha256 = {scenario_id: _file_sha256(root / relative) for scenario_id, relative in scenario_paths.items()}
    code_manifest = {relative: _file_sha256(root / relative) for relative in SEVEN_DAY_PREREG_CODE_PATHS}
    runtime_artifact_paths: tuple[str, ...] = ()
    if normalized_runtime_stack is not None:
        common = _require_mapping(
            normalized_runtime_stack["common_adapter"],
            field="runtime_stack.common_adapter",
        )
        raw_manifests = normalized_runtime_stack["character_manifests"]
        assert isinstance(raw_manifests, list)
        artifact_paths = [str(common["locator"])]
        for manifest in raw_manifests:
            assert isinstance(manifest, Mapping)
            artifact_paths.append(str(manifest["locator"]))
            raw_files = manifest["artifact_files"]
            assert isinstance(raw_files, list)
            artifact_paths.extend(str(item["locator"]) for item in raw_files if isinstance(item, Mapping))
        runtime_artifact_paths = tuple(artifact_paths)
    execution_source_snapshot = _execution_source_snapshot(
        root,
        extra_paths=runtime_artifact_paths,
    )
    payload: dict[str, object] = {
        "schema_version": (
            SEVEN_DAY_CHARACTER_STACK_PREREG_SCHEMA_VERSION
            if normalized_runtime_stack is not None
            else SEVEN_DAY_PREREG_SCHEMA_VERSION
        ),
        "created_at_unix_ms": created_at_unix_ms,
        "claim_scope": "simulated-user-real-lifecycle-only",
        "scenario_ids": list(SEVEN_DAY_SCENARIO_IDS),
        "scenario_paths": scenario_paths,
        "scenario_sha256": scenario_sha256,
        "code_manifest": code_manifest,
        "code_tree_sha256": hashlib.sha256(_canonical_bytes(code_manifest)).hexdigest(),
        "execution_source_snapshot": execution_source_snapshot,
        "formal_run": {
            "paraphrase_seeds": list(SEVEN_DAY_FORMAL_SEEDS),
            "arm_schedule": list(SEVEN_DAY_ALL_ARMS),
            "case_count": len(SEVEN_DAY_SCENARIO_IDS) * len(SEVEN_DAY_FORMAL_SEEDS),
            "run_count": len(SEVEN_DAY_SCENARIO_IDS) * len(SEVEN_DAY_FORMAL_SEEDS) * len(SEVEN_DAY_ALL_ARMS),
            "session_count": len(SEVEN_DAY_SCENARIO_IDS) * len(SEVEN_DAY_FORMAL_SEEDS) * len(SEVEN_DAY_ALL_ARMS) * 7,
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
        "formal_models": _formal_models(normalized_runtime_stack),
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
            "swapped-user-state": ("load a matched donor user's owner snapshots only"),
            "shuffled-history": ("load same-user owner snapshots in preregistered shuffled order"),
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
            "correct_not_better_than_stateless": ("shrink continuity claim to typed owner-metric behavior only"),
            "sleep_not_better_than_no_sleep": ("do not make next-day consolidation product claim"),
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
    if normalized_runtime_stack is not None:
        payload["runtime_stack"] = normalized_runtime_stack
    return payload


def _formal_models(
    runtime_stack: Mapping[str, object] | None,
) -> dict[str, object]:
    if runtime_stack is None:
        return {
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
                "rendering_contract": ("typed-FSM substantive draft plus closed-list local-LLM style-opener selection"),
            },
        }
    common = _require_mapping(runtime_stack.get("common_adapter"), field="runtime_stack.common_adapter")
    return {
        "sut": {
            "model_id": common["base_model_id"],
            "model_family": runtime_stack["sut_model_family"],
            "weights_sha256": common["base_model_weights_sha256"],
            "local_files_only": True,
            "frozen": True,
            "max_new_tokens": runtime_stack["sut_max_new_tokens"],
        },
        "simulator": {
            "model_id": "HuggingFaceTB/SmolLM2-360M-Instruct",
            "model_family": "smollm",
            "weights_sha256": "f2a59a2f8f71f33baf444c37bf0ea1901211e237d2b970fcb91335c278b4b9ad",
            "local_files_only": True,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": 12,
            "rendering_contract": ("typed-FSM substantive draft plus closed-list local-LLM style-opener selection"),
        },
    }


def seven_day_source_attestation_contract(
    preregistration: Mapping[str, object],
) -> dict[str, str]:
    """Return the exact matched source/runtime fields required on every run."""

    formal_models = _require_mapping(preregistration.get("formal_models"), field="formal_models")
    sut = _require_mapping(formal_models.get("sut"), field="formal_models.sut")
    simulator = _require_mapping(formal_models.get("simulator"), field="formal_models.simulator")
    contract = {
        "simulator_model_id": _require_string(simulator.get("model_id"), field="simulator.model_id"),
        "simulator_model_family": _require_string(simulator.get("model_family"), field="simulator.model_family"),
        "sut_model_id": _require_string(sut.get("model_id"), field="sut.model_id"),
        "sut_model_family": _require_string(sut.get("model_family"), field="sut.model_family"),
    }
    runtime_stack = preregistration.get("runtime_stack")
    if runtime_stack is None:
        fingerprint_payload: dict[str, object] = {
            "sut_model_id": sut.get("model_id"),
            "sut_weights_sha256": sut.get("weights_sha256"),
            "adapter": "none",
        }
    else:
        stack = _require_mapping(runtime_stack, field="runtime_stack")
        common = _require_mapping(stack.get("common_adapter"), field="runtime_stack.common_adapter")
        manifests = stack.get("character_manifests")
        if not isinstance(manifests, list):
            raise ValueError("seven-day character_manifests must be a list")
        selected_character_id = stack.get("selected_character_id")
        selected = next(
            (
                _require_mapping(item, field="character_manifest")
                for item in manifests
                if isinstance(item, Mapping) and item.get("character_id") == selected_character_id
            ),
            None,
        )
        if selected is None:
            raise ValueError("selected seven-day character manifest is missing")
        fingerprint_payload = {
            "sut_model_id": sut.get("model_id"),
            "sut_weights_sha256": sut.get("weights_sha256"),
            "common_adapter_bundle_sha256": common.get("sha256"),
            "common_adapter_bundle_id": common.get("bundle_id"),
            "common_adapter_version": common.get("common_adapter_version"),
            "compatibility_fingerprint": common.get("compatibility_fingerprint"),
            "character_manifest_sha256": selected.get("sha256"),
            "character_manifest_package_id": selected.get("package_id"),
            "character_id": selected_character_id,
            "character_prefix_package_id": selected.get("prefix_package_id"),
            "character_wiring_level": stack.get("wiring_level"),
        }
        contract.update(
            {
                "common_adapter_bundle_id": _require_string(common.get("bundle_id"), field="common_adapter.bundle_id"),
                "common_adapter_version": _require_string(
                    common.get("common_adapter_version"),
                    field="common_adapter.common_adapter_version",
                ),
                "common_adapter_compatibility_fingerprint": _require_string(
                    common.get("compatibility_fingerprint"),
                    field="common_adapter.compatibility_fingerprint",
                ),
                "character_manifest_package_id": _require_string(
                    selected.get("package_id"),
                    field="character_manifest.package_id",
                ),
                "character_id": _require_string(selected_character_id, field="selected_character_id"),
                "character_prefix_package_id": _require_string(
                    selected.get("prefix_package_id"),
                    field="character_manifest.prefix_package_id",
                ),
                "character_wiring_level": _require_string(
                    stack.get("wiring_level"), field="runtime_stack.wiring_level"
                ),
            }
        )
    contract["model_and_adapter_fingerprint"] = hashlib.sha256(_canonical_bytes(fingerprint_payload)).hexdigest()
    return contract


def validate_seven_day_companion_preregistration(
    payload: Mapping[str, object],
    *,
    repo_root: str | Path,
) -> None:
    created = payload.get("created_at_unix_ms")
    if isinstance(created, bool) or not isinstance(created, int):
        raise ValueError("seven-day preregistration timestamp drift")
    runtime_stack = payload.get("runtime_stack")
    if runtime_stack is not None and not isinstance(runtime_stack, Mapping):
        raise ValueError("seven-day runtime_stack must be an object")
    expected = build_seven_day_companion_preregistration(
        repo_root=repo_root,
        created_at_unix_ms=created,
        runtime_stack=runtime_stack,
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
        raise FileExistsError(f"seven-day preregistration is immutable: {target}")
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
    "seven_day_source_attestation_contract",
    "validate_seven_day_companion_preregistration",
    "write_seven_day_companion_preregistration",
]
