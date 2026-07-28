from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean
from typing import Any

from volvence_zero.agent.eta_proof_benchmark import (
    ETAProofPaperSuiteAggregateReport,
    build_eta_open_weight_paper_suite_manifest,
)
from volvence_zero.agent.paper_suite import PaperProfileSpec, PaperSuiteManifest


ETA_GATE2_SCHEMA_VERSION = "eta-gate2-residual-causal.v1"
ETA_GATE2_MIN_CAUSAL_DELTA = 0.02
ETA_GATE2_REQUIRED_FILES = (
    "manifest.yaml",
    "predictions.jsonl",
    "outcomes.jsonl",
    "prediction_errors.jsonl",
    "segments.jsonl",
    "credit.jsonl",
    "state_diff.jsonl",
    "ablation_results.json",
    "promotion_verdict.json",
    "rollback_evidence.json",
    "report.md",
)
ETA_GATE2_REQUIRED_MANIFEST_KEYS = (
    "git_sha",
    "substrate_fingerprint",
    "model_and_adapter_ids",
    "wiring_levels",
    "seed_schedule",
    "scenario_split",
    "cohort_scope",
    "prompt_and_context_budget",
    "metric_version",
    "judge_or_human_protocol",
)


def default_eta_gate2_residual_profiles() -> tuple[str, ...]:
    return (
        "full-internal-rl",
        "full-zero-control",
        "full-shuffled-control",
        "full-reversed-control",
    )


def build_eta_gate2_residual_manifest(
    *,
    suite_tier: str = "ci-smoke",
) -> PaperSuiteManifest:
    base = build_eta_open_weight_paper_suite_manifest(suite_tier=suite_tier)
    profiles = tuple(
        PaperProfileSpec(
            profile_label=profile_label,
            role=(
                "candidate"
                if profile_label == "full-internal-rl"
                else "matched-control"
            ),
            description=(
                "Same policy, optimizer, frozen substrate, scenarios, and seed "
                f"schedule with residual control mode {profile_label}."
            ),
        )
        for profile_label in default_eta_gate2_residual_profiles()
    )
    case_groups = base.case_groups + (
        (
            "residual_control_modes",
            ("identity", "zero", "shuffled", "reversed"),
        ),
        (
            "gate2_preregistered_thresholds",
            (
                f"min_causal_delta={ETA_GATE2_MIN_CAUSAL_DELTA}",
                "min_hook_coverage=0.75",
                "max_fallback_rate=0.0",
            ),
        ),
    )
    return replace(
        base,
        suite_id=f"eta-gate2-residual-causal-{suite_tier}",
        version=base.version + 1,
        profiles=profiles,
        case_groups=case_groups,
        artifact_expectations=ETA_GATE2_REQUIRED_FILES,
        description=(
            "Gate 2 matched residual intervention suite. The identity arm is "
            "compared with zero, deterministic shuffled, and reversed U_t on "
            "the same frozen substrate and schedule."
        ),
    )


def _write_json(path: Path, payload: Any) -> Path:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    serialized = "\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True)
        for row in rows
    )
    path.write_text(f"{serialized}\n" if serialized else "", encoding="utf-8")
    return path


def _load_benchmark_payloads(
    *,
    report: ETAProofPaperSuiteAggregateReport,
    output_dir: Path,
) -> tuple[dict[str, Any], ...]:
    payloads: list[dict[str, Any]] = []
    for index, run_summary in enumerate(report.run_summaries, start=1):
        path = output_dir / f"eta_run_{index:02d}_benchmark.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"Gate 2 bundle requires per-run benchmark artifact {path}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a JSON object")
        payloads.append(
            {
                "run_id": run_summary.run_id,
                "run_seed": run_summary.run_seed,
                "benchmark": payload,
            }
        )
    return tuple(payloads)


def _runtime_descriptor(
    report: ETAProofPaperSuiteAggregateReport,
) -> dict[str, str]:
    return dict(report.provenance.runtime_descriptor)


def _descriptor_fingerprint(descriptor: dict[str, str]) -> str:
    digest = sha256(
        json.dumps(descriptor, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"runtime-descriptor-sha256:{digest}"


def _metric_map(profile: dict[str, Any]) -> dict[str, float]:
    metrics = profile.get("metric_means")
    if not isinstance(metrics, list):
        raise ValueError("profile report requires metric_means list")
    return {str(name): float(value) for name, value in metrics}


def _mean_or_zero(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _mean_abs(values: list[float]) -> float:
    return _mean_or_zero([abs(value) for value in values])


def _close_vectors(
    left: list[float],
    right: list[float],
    *,
    tolerance: float = 1e-9,
) -> bool:
    return len(left) == len(right) and all(
        abs(left_value - right_value) <= tolerance
        for left_value, right_value in zip(left, right, strict=True)
    )


def _flatten_evidence(
    payloads: tuple[dict[str, Any], ...],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, list[float]],
]:
    predictions: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    prediction_errors: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    credit: list[dict[str, Any]] = []
    state_diff: list[dict[str, Any]] = []
    metric_samples: dict[str, list[float]] = {}
    for run in payloads:
        benchmark = run["benchmark"]
        profiles = benchmark.get("profile_reports")
        if not isinstance(profiles, list):
            raise ValueError("benchmark report requires profile_reports list")
        for profile in profiles:
            profile_label = str(profile["profile_label"])
            for metric_name, metric_value in _metric_map(profile).items():
                metric_samples.setdefault(
                    f"{profile_label}:{metric_name}",
                    [],
                ).append(metric_value)
            episodes = profile.get("episode_reports")
            if not isinstance(episodes, list):
                raise ValueError("profile report requires episode_reports list")
            for episode in episodes:
                episode_key = {
                    "run_id": run["run_id"],
                    "run_seed": run["run_seed"],
                    "profile_label": profile_label,
                    "case_id": episode["case_id"],
                    "split": episode["split"],
                }
                segments.append(
                    {
                        **episode_key,
                        "switch_sparsity": episode["switch_sparsity"],
                        "mean_persistence": episode["mean_persistence"],
                        "completed_subgoals": episode["completed_subgoals"],
                        "completed_family_ids": episode[
                            "completed_family_ids"
                        ],
                        "evidence_scope": (
                            "episode-level segment summary; per-step beta "
                            "boundaries remain in owner trace"
                        ),
                    }
                )
                credit.append(
                    {
                        **episode_key,
                        "delayed_credit_assignment_count": episode[
                            "delayed_credit_assignment_count"
                        ],
                        "credit_alignment": episode["credit_alignment"],
                        "terminal_credit_coverage": episode[
                            "terminal_credit_coverage"
                        ],
                    }
                )
                records = episode.get("intervention_records")
                if not isinstance(records, list) or not records:
                    raise ValueError(
                        "Gate 2 episode requires intervention_records"
                    )
                for record in records:
                    lineage = {
                        **episode_key,
                        "step_index": record["step_index"],
                    }
                    before = [
                        float(value)
                        for value in record["control_before_ablation"]
                    ]
                    applied = [
                        float(value) for value in record["applied_control"]
                    ]
                    downstream = [
                        float(value) for value in record["downstream_effect"]
                    ]
                    predictions.append(
                        {
                            **lineage,
                            "residual_control_mode": record[
                                "residual_control_mode"
                            ],
                            "decoder_output": record["decoder_output"],
                            "control_before_ablation": before,
                            "applied_control": applied,
                            "backend_name": record["backend_name"],
                        }
                    )
                    outcomes.append(
                        {
                            **lineage,
                            "downstream_effect": downstream,
                            "downstream_effect_magnitude": record[
                                "downstream_effect_magnitude"
                            ],
                            "reward": record["reward"],
                            "proof_terminal_success": record[
                                "proof_terminal_success"
                            ],
                        }
                    )
                    residual_transport_error = _mean_abs(
                        [
                            applied_value - downstream_value
                            for applied_value, downstream_value in zip(
                                applied[:3],
                                downstream,
                                strict=True,
                            )
                        ]
                    )
                    prediction_errors.append(
                        {
                            **lineage,
                            "measurement_kind": "residual-transport-error",
                            "value": residual_transport_error,
                            "is_primary_prediction_error_owner_signal": False,
                            "note": (
                                "This file is present for the unified bundle "
                                "contract; Gate 1 raw PE is not claimed here."
                            ),
                        }
                    )
                    state_diff.append(
                        {
                            **lineage,
                            "residual_control_mode": record[
                                "residual_control_mode"
                            ],
                            "control_before_ablation": before,
                            "applied_control": applied,
                            "downstream_effect": downstream,
                            "ablation_l1_delta": _mean_abs(
                                [
                                    before_value - applied_value
                                    for before_value, applied_value in zip(
                                        before,
                                        applied,
                                        strict=True,
                                    )
                                ]
                            ),
                            "replacement_effect_delta": record[
                                "replacement_effect_delta"
                            ],
                            "switch_gate": record["switch_gate"],
                            "active_family_id": record["active_family_id"],
                        }
                    )
    return (
        predictions,
        outcomes,
        prediction_errors,
        segments,
        credit,
        state_diff,
        metric_samples,
    )


def _transformation_gates(
    predictions: list[dict[str, Any]],
) -> dict[str, bool]:
    modes: dict[str, list[dict[str, Any]]] = {}
    for row in predictions:
        modes.setdefault(str(row["residual_control_mode"]), []).append(row)
    identity = modes.get("identity", [])
    zero = modes.get("zero", [])
    shuffled = modes.get("shuffled", [])
    reversed_rows = modes.get("reversed", [])
    return {
        "identity_exact": bool(identity)
        and all(
            _close_vectors(
                row["control_before_ablation"],
                row["applied_control"],
            )
            for row in identity
        ),
        "zero_exact": bool(zero)
        and all(
            all(abs(float(value)) <= 1e-12 for value in row["applied_control"])
            for row in zero
        ),
        "shuffle_permutation_nonidentity": bool(shuffled)
        and all(
            sorted(row["control_before_ablation"])
            == sorted(row["applied_control"])
            and not _close_vectors(
                row["control_before_ablation"],
                row["applied_control"],
            )
            for row in shuffled
        ),
        "reverse_exact": bool(reversed_rows)
        and all(
            _close_vectors(
                list(reversed(row["control_before_ablation"])),
                row["applied_control"],
            )
            for row in reversed_rows
        ),
    }


def _profile_metric_means(
    metric_samples: dict[str, list[float]],
    metric_name: str,
) -> dict[str, float]:
    return {
        profile_label: _mean_or_zero(
            metric_samples.get(f"{profile_label}:{metric_name}", [])
        )
        for profile_label in default_eta_gate2_residual_profiles()
    }


def _build_ablation_results(
    *,
    predictions: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    metric_samples: dict[str, list[float]],
) -> dict[str, Any]:
    transformations = _transformation_gates(predictions)
    strong_success = _profile_metric_means(
        metric_samples,
        "heldout_strong_success_rate",
    )
    terminal_success = _profile_metric_means(
        metric_samples,
        "heldout_terminal_success_rate",
    )
    hook_coverage = _profile_metric_means(
        metric_samples,
        "real_open_weight_hook_coverage",
    )
    fallback_rate = _profile_metric_means(
        metric_samples,
        "real_open_weight_fallback_rate",
    )
    protocol_valid = _profile_metric_means(
        metric_samples,
        "real_open_weight_intervention_protocol_valid",
    )
    effect_magnitude: dict[str, float] = {}
    for profile_label in default_eta_gate2_residual_profiles():
        values = [
            float(row["downstream_effect_magnitude"])
            for row in outcomes
            if row["profile_label"] == profile_label
        ]
        effect_magnitude[profile_label] = _mean_or_zero(values)
    controls = tuple(
        profile
        for profile in default_eta_gate2_residual_profiles()
        if profile != "full-internal-rl"
    )
    strongest_control_success = max(
        strong_success[profile] for profile in controls
    )
    strongest_control_terminal = max(
        terminal_success[profile] for profile in controls
    )
    identity_success_delta = (
        strong_success["full-internal-rl"] - strongest_control_success
    )
    identity_terminal_delta = (
        terminal_success["full-internal-rl"] - strongest_control_terminal
    )
    identity_effect_delta_vs_zero = (
        effect_magnitude["full-internal-rl"]
        - effect_magnitude["full-zero-control"]
    )
    backend_names = {
        str(row["backend_name"]) for row in predictions
    }
    real_backend = bool(backend_names) and all(
        name.startswith("open-weight:")
        or name.startswith("transformers-open-weight:")
        for name in backend_names
    )
    mechanism_gates = {
        **transformations,
        "real_open_weight_backend": real_backend,
        "hook_coverage": (
            hook_coverage["full-internal-rl"] >= 0.75
            if real_backend
            else False
        ),
        "fallback_rate_zero": (
            fallback_rate["full-internal-rl"] <= 0.0
            if real_backend
            else False
        ),
        "prefix_intervention_protocol": (
            protocol_valid["full-internal-rl"] >= 1.0
            if real_backend
            else False
        ),
    }
    causal_gates = {
        "identity_effect_beats_zero": (
            identity_effect_delta_vs_zero >= ETA_GATE2_MIN_CAUSAL_DELTA
        ),
        "identity_strong_success_beats_controls": (
            identity_success_delta >= ETA_GATE2_MIN_CAUSAL_DELTA
        ),
        "identity_terminal_success_not_worse": (
            identity_terminal_delta >= 0.0
        ),
    }
    return {
        "schema_version": ETA_GATE2_SCHEMA_VERSION,
        "preregistered_min_causal_delta": ETA_GATE2_MIN_CAUSAL_DELTA,
        "profile_strong_success": strong_success,
        "profile_terminal_success": terminal_success,
        "profile_downstream_effect_magnitude": effect_magnitude,
        "profile_hook_coverage": hook_coverage,
        "profile_fallback_rate": fallback_rate,
        "profile_intervention_protocol_valid": protocol_valid,
        "identity_strong_success_delta_vs_best_control": (
            identity_success_delta
        ),
        "identity_terminal_success_delta_vs_best_control": (
            identity_terminal_delta
        ),
        "identity_effect_delta_vs_zero": identity_effect_delta_vs_zero,
        "mechanism_gates": mechanism_gates,
        "causal_gates": causal_gates,
        "all_mechanism_gates_passed": all(mechanism_gates.values()),
        "all_causal_gates_passed": all(causal_gates.values()),
        "backend_names": sorted(backend_names),
    }


def _promotion_verdict(ablation: dict[str, Any]) -> dict[str, Any]:
    mechanism_passed = bool(ablation["all_mechanism_gates_passed"])
    causal_passed = mechanism_passed and bool(
        ablation["all_causal_gates_passed"]
    )
    if causal_passed:
        status = "causal-supported"
    elif mechanism_passed:
        status = "mechanism-supported"
    else:
        status = "wiring-ready"
    return {
        "schema_version": ETA_GATE2_SCHEMA_VERSION,
        "status": status,
        "gate_scope": "Gate 2 residual intervention causal packet",
        "thesis_status": "not-evaluated",
        "promotion_allowed": causal_passed,
        "mechanism_gates": ablation["mechanism_gates"],
        "causal_gates": ablation["causal_gates"],
        "kill_conditions": [
            gate
            for gate, passed in {
                **ablation["mechanism_gates"],
                **ablation["causal_gates"],
            }.items()
            if not passed
        ],
        "note": (
            "This packet cannot emit thesis-retained. Gate 2 longitudinal "
            "coverage, semantic no-label evidence, and Gates 1/3-10 remain "
            "separate prerequisites."
        ),
    }


def export_eta_gate2_residual_bundle(
    report: ETAProofPaperSuiteAggregateReport,
    *,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    payloads = _load_benchmark_payloads(report=report, output_dir=target)
    (
        predictions,
        outcomes,
        prediction_errors,
        segments,
        credit,
        state_diff,
        metric_samples,
    ) = _flatten_evidence(payloads)
    descriptor = _runtime_descriptor(report)
    route_ids = [
        value
        for name, values in report.manifest.case_groups
        if name == "route_ids"
        for value in values
    ]
    manifest = {
        "schema_version": ETA_GATE2_SCHEMA_VERSION,
        "git_sha": report.provenance.git_sha,
        "substrate_fingerprint": _descriptor_fingerprint(descriptor),
        "substrate_fingerprint_verification": (
            "runtime-descriptor-only; weights digest not exposed by runtime"
        ),
        "model_and_adapter_ids": {
            "model_id": descriptor.get("open_weight_model_id", "unknown"),
            "adapter_ids": [],
        },
        "wiring_levels": {
            "residual_capture": "ACTIVE",
            "residual_intervention": "ACTIVE",
            "metacontroller": "ACTIVE",
            "artifact_promotion": "SHADOW",
        },
        "seed_schedule": list(report.manifest.seed_schedule),
        "scenario_split": {
            "route_ids": route_ids,
            "split_owner": "default_eta_proof_cases",
        },
        "cohort_scope": (
            "hierarchical proof cases; no per-user longitudinal cohort"
        ),
        "prompt_and_context_budget": {
            "max_prefix_steps": descriptor.get(
                "open_weight_max_prefix_steps",
                "unknown",
            ),
            "prefix_alignment": "capture-and-intervention-same-prefix",
        },
        "metric_version": ETA_GATE2_SCHEMA_VERSION,
        "judge_or_human_protocol": (
            "typed automatic metrics; no LLM judge or human rating"
        ),
        "suite_id": report.manifest.suite_id,
        "profiles": list(default_eta_gate2_residual_profiles()),
        "required_files": list(ETA_GATE2_REQUIRED_FILES),
    }
    missing_manifest_keys = [
        key for key in ETA_GATE2_REQUIRED_MANIFEST_KEYS if key not in manifest
    ]
    if missing_manifest_keys:
        raise ValueError(
            "Gate 2 manifest missing required keys: "
            + ", ".join(missing_manifest_keys)
        )
    ablation = _build_ablation_results(
        predictions=predictions,
        outcomes=outcomes,
        metric_samples=metric_samples,
    )
    verdict = _promotion_verdict(ablation)
    rollback = {
        "schema_version": ETA_GATE2_SCHEMA_VERSION,
        "rollback_target": "residual_control_mode=identity",
        "configuration_default_is_identity": True,
        "owner_state_mutated_by_mode_switch": False,
        "full_chain_rollback_drill_executed": False,
        "passed": False,
        "note": (
            "Mode rollback is immediate and does not mutate the frozen "
            "substrate. The #92 full-chain artifact/user-state drill is not "
            "part of this Gate 2 packet."
        ),
    }
    report_markdown = "\n".join(
        (
            "# ETA Gate 2 残差因果证据",
            "",
            f"- 判定：`{verdict['status']}`",
            f"- substrate：`{manifest['model_and_adapter_ids']['model_id']}`",
            f"- seeds：`{manifest['seed_schedule']}`",
            (
                "- identity 对 strongest control 的 strong-success 差："
                f"`{ablation['identity_strong_success_delta_vs_best_control']:.6f}`"
            ),
            (
                "- identity 对 zero 的真实 downstream-effect 差："
                f"`{ablation['identity_effect_delta_vs_zero']:.6f}`"
            ),
            (
                "- mechanism gates："
                f"`{ablation['mechanism_gates']}`"
            ),
            f"- causal gates：`{ablation['causal_gates']}`",
            "",
            "本包只判 Gate 2 的残差注入机制与 matched-control 因果差。",
            "它不声称 Gate 2 longitudinal 已完成，也不产生 thesis-retained。",
            "",
        )
    )
    paths = (
        _write_json(target / "manifest.yaml", manifest),
        _write_jsonl(target / "predictions.jsonl", predictions),
        _write_jsonl(target / "outcomes.jsonl", outcomes),
        _write_jsonl(
            target / "prediction_errors.jsonl",
            prediction_errors,
        ),
        _write_jsonl(target / "segments.jsonl", segments),
        _write_jsonl(target / "credit.jsonl", credit),
        _write_jsonl(target / "state_diff.jsonl", state_diff),
        _write_json(target / "ablation_results.json", ablation),
        _write_json(target / "promotion_verdict.json", verdict),
        _write_json(target / "rollback_evidence.json", rollback),
        target / "report.md",
    )
    paths[-1].write_text(report_markdown, encoding="utf-8")
    missing_files = [
        name for name in ETA_GATE2_REQUIRED_FILES
        if not (target / name).is_file()
    ]
    if missing_files:
        raise RuntimeError(
            "Gate 2 bundle export missed required files: "
            + ", ".join(missing_files)
        )
    return paths
