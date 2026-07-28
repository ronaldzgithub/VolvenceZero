from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean
from typing import Any

from volvence_zero.agent.eta_proof_benchmark import (
    ETA_COUNTERFACTUAL_TARGET_PREFIX_EXPECTED,
    ETA_CONTINUATION_PE_REWARD_SCALE,
    ETA_CONTINUATION_PE_TRAINING_SIGNAL,
    ETA_GATE2_EXPECTED_VALUE_CASE_CORPUS,
    ETAProofPaperSuiteAggregateReport,
    build_eta_open_weight_paper_suite_manifest,
    eta_gate2_expected_value_routes,
    eta_gate2_expected_value_validation_routes,
    eta_gate2_independent_training_routes,
)
from volvence_zero.agent.paper_suite import PaperProfileSpec, PaperSuiteManifest


ETA_GATE2_SCHEMA_VERSION = "eta-gate2-residual-causal.v30"
ETA_GATE2_MIN_CAUSAL_DELTA = 0.02
ETA_GATE2_MIN_MEASURABLE_EFFECT = 1e-6
ETA_GATE2_REQUIRED_FILES = (
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
ETA_GATE2_REQUIRED_MANIFEST_KEYS = (
    "git_sha",
    "substrate_fingerprint",
    "model_and_adapter_ids",
    "wiring_levels",
    "seed_schedule",
    "scenario_split",
    "cohort_scope",
    "prompt_and_context_budget",
    "counterfactual_action_selector",
    "counterfactual_target",
    "residual_capture",
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
    expanded_route_ids = tuple(
        route.case_id for route in eta_gate2_expected_value_routes()
    )
    independent_route_ids = tuple(
        route.case_id
        for route in eta_gate2_independent_training_routes()
    )
    validation_route_ids = tuple(
        route.case_id
        for route in eta_gate2_expected_value_validation_routes()
    )
    base_case_groups = tuple(
        ("route_ids", expanded_route_ids)
        if name == "route_ids"
        else (name, values)
        for name, values in base.case_groups
    )
    case_groups = base_case_groups + (
        ("case_corpus", (ETA_GATE2_EXPECTED_VALUE_CASE_CORPUS,)),
        ("independent_training_route_ids", independent_route_ids),
        ("validation_route_ids", validation_route_ids),
        ("validation_frozen_before_run", ("true",)),
        ("training_route_count", ("16",)),
        ("development_routes_unchanged", ("true",)),
        (
            "training_signal",
            (ETA_CONTINUATION_PE_TRAINING_SIGNAL,),
        ),
        (
            "continuation_pe_reward_scale",
            (str(ETA_CONTINUATION_PE_REWARD_SCALE),),
        ),
        ("latent_unit_clamp", ("true",)),
        ("real_residual_ssl_bootstrap", ("true",)),
        ("real_residual_activation_width", ("896",)),
        ("causal_action_head_active", ("true",)),
        ("causal_action_head_state_dim", ("12",)),
        ("residual_actuator_dim", ("3",)),
        ("continuation_counterfactual_grid", ("true",)),
        (
            "counterfactual_action_selector_diagnostic",
            ("true",),
        ),
        (
            "counterfactual_action_selector_input",
            ("full-layer-coordinate-mean-latest-trend",),
        ),
        (
            "counterfactual_action_selector_encoder",
            ("train-only-standardized-linear-kernel",),
        ),
        (
            "counterfactual_action_selector_head",
            ("train-only-dual-ridge-ladder-0.1-1-10",),
        ),
        (
            "counterfactual_action_selector_model_selection",
            ("train-route-cv-audit-delta-then-audit-regret",),
        ),
        (
            "counterfactual_action_selector_cv",
            ("route-grouped-4fold",),
        ),
        (
            "counterfactual_action_selector_live_injection",
            ("disabled",),
        ),
        (
            "counterfactual_target_mode",
            (ETA_COUNTERFACTUAL_TARGET_PREFIX_EXPECTED,),
        ),
        ("counterfactual_target_sample_count", ("4",)),
        ("counterfactual_audit_sample_count", ("4",)),
        ("counterfactual_sampling_temperature", ("0.8",)),
        ("counterfactual_sampling_max_new_tokens", ("4",)),
        (
            "counterfactual_sampling_seed_protocol",
            ("sha256-case-prefix-cohort-index",),
        ),
        (
            "counterfactual_reward_normalization",
            ("per-prefix-centered-range-v1",),
        ),
        ("confirmation_split_locked", ("false",)),
        (
            "residual_control_modes",
            ("identity", "zero", "shuffled", "reversed"),
        ),
        (
            "gate2_preregistered_thresholds",
            (
                f"min_causal_delta={ETA_GATE2_MIN_CAUSAL_DELTA}",
                (
                    "min_measurable_residual_effect="
                    f"{ETA_GATE2_MIN_MEASURABLE_EFFECT}"
                ),
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


def _continuation_metrics_by_split(
    *,
    outcomes: list[dict[str, Any]],
    split_names: tuple[str, ...],
) -> tuple[
    dict[str, dict[str, float | None]],
    dict[str, dict[str, int]],
]:
    means: dict[str, dict[str, float | None]] = {}
    counts: dict[str, dict[str, int]] = {}
    for split_name in split_names:
        split_means: dict[str, float | None] = {}
        split_counts: dict[str, int] = {}
        for profile_label in default_eta_gate2_residual_profiles():
            values = [
                float(row["continuation_mean_nll"])
                for row in outcomes
                if row["profile_label"] == profile_label
                and row["split"] == split_name
                and row["continuation_mean_nll"] is not None
            ]
            split_counts[profile_label] = len(values)
            split_means[profile_label] = (
                _mean_or_zero(values) if values else None
            )
        means[split_name] = split_means
        counts[split_name] = split_counts
    return means, counts


def _identity_nll_delta_vs_best_control(
    *,
    profile_means: dict[str, float | None],
) -> float | None:
    identity_nll = profile_means["full-internal-rl"]
    control_nlls = [
        profile_means[profile]
        for profile in default_eta_gate2_residual_profiles()
        if profile != "full-internal-rl"
    ]
    if identity_nll is None or any(
        value is None for value in control_nlls
    ):
        return None
    return min(float(value) for value in control_nlls) - float(identity_nll)


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
    list[dict[str, Any]],
    dict[str, list[float]],
]:
    predictions: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    prediction_errors: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    credit: list[dict[str, Any]] = []
    state_diff: list[dict[str, Any]] = []
    action_selection: list[dict[str, Any]] = []
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
            selection_records = profile.get(
                "action_selection_records",
                [],
            )
            if not isinstance(selection_records, list):
                raise ValueError(
                    "profile report action_selection_records must be a list"
                )
            for record in selection_records:
                action_selection.append(
                    {
                        "run_id": run["run_id"],
                        "run_seed": run["run_seed"],
                        "profile_label": profile_label,
                        **record,
                    }
                )
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
                            "continuation_text": record[
                                "continuation_text"
                            ],
                            "continuation_token_count": record[
                                "continuation_token_count"
                            ],
                            "continuation_mean_nll": record[
                                "continuation_mean_nll"
                            ],
                            "continuation_geometric_mean_probability": record[
                                "continuation_geometric_mean_probability"
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
        action_selection,
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
    informative_shuffled = [
        row
        for row in shuffled
        if len(set(row["control_before_ablation"])) > 1
    ]
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
        "shuffle_permutation_nonidentity": bool(informative_shuffled)
        and all(
            sorted(row["control_before_ablation"])
            == sorted(row["applied_control"])
            for row in shuffled
        ),
        "shuffle_informative_steps_nonidentity": all(
            not _close_vectors(
                row["control_before_ablation"],
                row["applied_control"],
            )
            for row in informative_shuffled
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
    confirmation_split_locked: bool = False,
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
    shared_checkpoint_used = _profile_metric_means(
        metric_samples,
        "shared_policy_checkpoint_used",
    )
    shared_fingerprint_match = _profile_metric_means(
        metric_samples,
        "shared_policy_fingerprint_match_at_eval_start",
    )
    bootstrap_init_used = _profile_metric_means(
        metric_samples,
        "bootstrap_init_used",
    )
    causal_action_head_active = _profile_metric_means(
        metric_samples,
        "causal_action_head_active",
    )
    causal_action_head_update_step = _profile_metric_means(
        metric_samples,
        "causal_action_head_update_step",
    )
    causal_action_head_state_dim = _profile_metric_means(
        metric_samples,
        "causal_action_head_state_dim",
    )
    continuation_pe_training_count = _profile_metric_means(
        metric_samples,
        "continuation_pe_training_transition_count",
    )
    continuation_pe_training_mean_raw_delta = _profile_metric_means(
        metric_samples,
        "continuation_pe_training_mean_raw_delta",
    )
    continuation_pe_training_positive_rate = _profile_metric_means(
        metric_samples,
        "continuation_pe_training_positive_rate",
    )
    continuation_pe_structure_frozen = _profile_metric_means(
        metric_samples,
        "continuation_pe_structure_frozen",
    )
    continuation_pe_policy_changed = _profile_metric_means(
        metric_samples,
        "continuation_pe_policy_changed",
    )
    continuation_counterfactual_grid_used = _profile_metric_means(
        metric_samples,
        "continuation_counterfactual_grid_used",
    )
    continuation_counterfactual_unique_score_count = (
        _profile_metric_means(
            metric_samples,
            "continuation_counterfactual_unique_score_count",
        )
    )
    continuation_counterfactual_prefix_count = _profile_metric_means(
        metric_samples,
        "continuation_counterfactual_prefix_count",
    )
    continuation_counterfactual_oracle_mean_raw_delta = (
        _profile_metric_means(
            metric_samples,
            "continuation_counterfactual_oracle_mean_raw_delta",
        )
    )
    continuation_counterfactual_oracle_positive_rate = (
        _profile_metric_means(
            metric_samples,
            "continuation_counterfactual_oracle_positive_rate",
        )
    )
    continuation_counterfactual_best_fixed_mean_raw_delta = (
        _profile_metric_means(
            metric_samples,
            "continuation_counterfactual_best_fixed_mean_raw_delta",
        )
    )
    continuation_counterfactual_oracle_vs_fixed_gap = (
        _profile_metric_means(
            metric_samples,
            "continuation_counterfactual_oracle_vs_fixed_gap",
        )
    )
    continuation_counterfactual_diagnostics_by_split = {
        split: {
            metric: _profile_metric_means(
                metric_samples,
                (
                    f"continuation_counterfactual_{split}_"
                    f"{metric}"
                ),
            )
            for metric in (
                "prefix_count",
                "oracle_mean_raw_delta",
                "oracle_positive_rate",
                "best_fixed_mean_raw_delta",
                "oracle_vs_fixed_gap",
            )
        }
        for split in (
            "eval",
            "heldout",
            "validation",
            "confirmation",
        )
    }
    selector_metric_names = (
        "counterfactual_selector_input_dim",
        "counterfactual_selector_latent_dim",
        "counterfactual_selector_ridge_strength",
        "counterfactual_selector_model_candidate_count",
        "counterfactual_selector_explained_variance_ratio",
        "counterfactual_selector_chance_top3_rate",
        "counterfactual_selector_injection_gate_passed",
        "counterfactual_selector_eval_updates_after_fit",
        *tuple(
            f"counterfactual_selector_{split}_{metric}"
            for split in (
                "train",
                "eval",
                "heldout",
                "validation",
                "confirmation",
            )
            for metric in (
                "count",
                "mean_selected_raw_delta",
                "mean_oracle_raw_delta",
                "mean_oracle_regret",
                "top1_rate",
                "top3_rate",
                "selected_positive_rate",
                "audit_available_rate",
                "mean_audit_selected_raw_delta",
                "mean_audit_oracle_raw_delta",
                "mean_audit_oracle_regret",
                "audit_selected_positive_rate",
            )
        ),
    )
    counterfactual_selector_metrics = {
        metric_name: _profile_metric_means(
            metric_samples,
            metric_name,
        )
        for metric_name in selector_metric_names
    }
    effect_magnitude_all_splits: dict[str, float] = {}
    heldout_effect_magnitude: dict[str, float] = {}
    continuation_mean_nll_all_splits: dict[str, float | None] = {}
    (
        continuation_mean_nll_by_split,
        continuation_score_count_by_split,
    ) = _continuation_metrics_by_split(
        outcomes=outcomes,
        split_names=(
            "train",
            "eval",
            "heldout",
            "validation",
            "confirmation",
        ),
    )
    continuation_mean_nll = continuation_mean_nll_by_split["heldout"]
    continuation_score_count = continuation_score_count_by_split["heldout"]
    continuation_score_count_all_splits: dict[str, int] = {}
    for profile_label in default_eta_gate2_residual_profiles():
        values = [
            float(row["downstream_effect_magnitude"])
            for row in outcomes
            if row["profile_label"] == profile_label
        ]
        effect_magnitude_all_splits[profile_label] = _mean_or_zero(
            values
        )
        heldout_values = [
            float(row["downstream_effect_magnitude"])
            for row in outcomes
            if row["profile_label"] == profile_label
            and row["split"] == "heldout"
        ]
        heldout_effect_magnitude[profile_label] = _mean_or_zero(
            heldout_values
        )
        all_continuation_values = [
            float(row["continuation_mean_nll"])
            for row in outcomes
            if row["profile_label"] == profile_label
            and row["continuation_mean_nll"] is not None
        ]
        continuation_score_count_all_splits[profile_label] = len(
            all_continuation_values
        )
        continuation_mean_nll_all_splits[profile_label] = (
            _mean_or_zero(all_continuation_values)
            if all_continuation_values
            else None
        )
    controls = tuple(
        profile
        for profile in default_eta_gate2_residual_profiles()
        if profile != "full-internal-rl"
    )
    shared_policy_checkpoint_matched = all(
        shared_checkpoint_used[profile] >= 1.0
        and shared_fingerprint_match[profile] >= 1.0
        for profile in controls
    )
    continuation_pe_training_isolated = (
        continuation_pe_training_count["full-internal-rl"] > 0.0
        and all(
            continuation_pe_training_count[profile] == 0.0
            for profile in controls
        )
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
        heldout_effect_magnitude["full-internal-rl"]
        - heldout_effect_magnitude["full-zero-control"]
    )
    heldout_identity_predictions = [
        row
        for row in predictions
        if row["profile_label"] == "full-internal-rl"
        and row["split"] == "heldout"
    ]
    heldout_identity_control_exposure = _mean_or_zero(
        [
            1.0
            if any(
                abs(float(value))
                >= ETA_GATE2_MIN_MEASURABLE_EFFECT
                for value in row["applied_control"]
            )
            else 0.0
            for row in heldout_identity_predictions
        ]
    )
    eval_continuation_scores_available = all(
        continuation_score_count_by_split["eval"][profile] > 0
        for profile in default_eta_gate2_residual_profiles()
    )
    confirmation_continuation_scores_available = all(
        continuation_score_count_by_split["confirmation"][profile] > 0
        for profile in default_eta_gate2_residual_profiles()
    )
    heldout_identity_nll_delta = _identity_nll_delta_vs_best_control(
        profile_means=continuation_mean_nll_by_split["heldout"],
    )
    eval_identity_nll_delta = _identity_nll_delta_vs_best_control(
        profile_means=continuation_mean_nll_by_split["eval"],
    )
    confirmation_identity_nll_delta = (
        _identity_nll_delta_vs_best_control(
            profile_means=continuation_mean_nll_by_split["confirmation"],
        )
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
        "identity_effect_is_measurable_vs_zero": (
            identity_effect_delta_vs_zero
            >= ETA_GATE2_MIN_MEASURABLE_EFFECT
        ),
        "heldout_identity_control_exposure": (
            heldout_identity_control_exposure >= 0.75
        ),
        "shared_policy_checkpoint_matched": (
            shared_policy_checkpoint_matched
        ),
        "continuation_pe_training_isolated_to_identity": (
            continuation_pe_training_isolated
            if real_backend
            else False
        ),
        "real_residual_ssl_bootstrap_used": (
            bootstrap_init_used["full-internal-rl"] >= 1.0
            if real_backend
            else False
        ),
        "causal_action_head_updated": (
            causal_action_head_active["full-internal-rl"] >= 1.0
            and causal_action_head_update_step["full-internal-rl"] > 0.0
        ),
        "observation_state_separate_from_actuator": (
            causal_action_head_state_dim["full-internal-rl"]
            > max(
                (
                    len(row["applied_control"])
                    for row in predictions
                    if row["profile_label"] == "full-internal-rl"
                ),
                default=0,
            )
        ),
        "continuation_pe_updates_policy_only": (
            continuation_pe_structure_frozen["full-internal-rl"] >= 1.0
            and continuation_pe_policy_changed["full-internal-rl"] >= 1.0
        ),
        "continuation_counterfactual_grid_used": (
            continuation_counterfactual_grid_used[
                "full-internal-rl"
            ]
            >= 1.0
        ),
    }
    causal_gates = {
        "eval_continuation_scores_available_all_arms": (
            eval_continuation_scores_available
        ),
        "identity_eval_nll_beats_controls": (
            eval_identity_nll_delta is not None
            and eval_identity_nll_delta > 0.0
        ),
        "fresh_confirmation_scores_available_all_arms": (
            confirmation_continuation_scores_available
        ),
        "fresh_confirmation_split_locked": (
            confirmation_split_locked
            and confirmation_continuation_scores_available
        ),
        "identity_confirmation_nll_beats_controls": (
            confirmation_identity_nll_delta is not None
            and confirmation_identity_nll_delta
            >= ETA_GATE2_MIN_CAUSAL_DELTA
        ),
    }
    selector_gates = {
        "train_grouped_cv_predictions_available": (
            counterfactual_selector_metrics[
                "counterfactual_selector_train_count"
            ]["full-internal-rl"]
            > 0.0
        ),
        "frozen_eval_predictions_available": (
            counterfactual_selector_metrics[
                "counterfactual_selector_eval_count"
            ]["full-internal-rl"]
            > 0.0
        ),
        "frozen_heldout_predictions_available": (
            counterfactual_selector_metrics[
                "counterfactual_selector_heldout_count"
            ]["full-internal-rl"]
            > 0.0
        ),
        "frozen_validation_predictions_available": (
            counterfactual_selector_metrics[
                "counterfactual_selector_validation_count"
            ]["full-internal-rl"]
            > 0.0
        ),
        "train_independent_audit_available": (
            counterfactual_selector_metrics[
                "counterfactual_selector_train_audit_available_rate"
            ]["full-internal-rl"]
            >= 1.0
        ),
        "validation_independent_audit_available": (
            counterfactual_selector_metrics[
                (
                    "counterfactual_selector_validation_"
                    "audit_available_rate"
                )
            ]["full-internal-rl"]
            >= 1.0
        ),
        "no_eval_updates_after_fit": (
            counterfactual_selector_metrics[
                "counterfactual_selector_eval_updates_after_fit"
            ]["full-internal-rl"]
            == 0.0
        ),
        "selector_ready_for_shadow_injection": (
            counterfactual_selector_metrics[
                "counterfactual_selector_injection_gate_passed"
            ]["full-internal-rl"]
            >= 1.0
        ),
    }
    return {
        "schema_version": ETA_GATE2_SCHEMA_VERSION,
        "preregistered_min_causal_delta": ETA_GATE2_MIN_CAUSAL_DELTA,
        "measurement_resolution_floor": (
            ETA_GATE2_MIN_MEASURABLE_EFFECT
        ),
        "profile_strong_success": strong_success,
        "profile_terminal_success": terminal_success,
        "profile_downstream_effect_magnitude": (
            heldout_effect_magnitude
        ),
        "profile_downstream_effect_magnitude_all_splits": (
            effect_magnitude_all_splits
        ),
        "profile_continuation_mean_nll": continuation_mean_nll,
        "profile_continuation_mean_nll_by_split": (
            continuation_mean_nll_by_split
        ),
        "profile_continuation_mean_nll_all_splits": (
            continuation_mean_nll_all_splits
        ),
        "profile_continuation_score_count": continuation_score_count,
        "profile_continuation_score_count_by_split": (
            continuation_score_count_by_split
        ),
        "profile_continuation_score_count_all_splits": (
            continuation_score_count_all_splits
        ),
        "profile_hook_coverage": hook_coverage,
        "profile_fallback_rate": fallback_rate,
        "profile_intervention_protocol_valid": protocol_valid,
        "profile_shared_checkpoint_used": shared_checkpoint_used,
        "profile_bootstrap_init_used": bootstrap_init_used,
        "profile_causal_action_head_active": (
            causal_action_head_active
        ),
        "profile_causal_action_head_update_step": (
            causal_action_head_update_step
        ),
        "profile_causal_action_head_state_dim": (
            causal_action_head_state_dim
        ),
        "profile_shared_fingerprint_match_at_eval_start": (
            shared_fingerprint_match
        ),
        "profile_continuation_pe_training_count": (
            continuation_pe_training_count
        ),
        "profile_continuation_pe_training_mean_raw_delta": (
            continuation_pe_training_mean_raw_delta
        ),
        "profile_continuation_pe_training_positive_rate": (
            continuation_pe_training_positive_rate
        ),
        "profile_continuation_pe_structure_frozen": (
            continuation_pe_structure_frozen
        ),
        "profile_continuation_pe_policy_changed": (
            continuation_pe_policy_changed
        ),
        "profile_continuation_counterfactual_grid_used": (
            continuation_counterfactual_grid_used
        ),
        "profile_continuation_counterfactual_unique_score_count": (
            continuation_counterfactual_unique_score_count
        ),
        "profile_continuation_counterfactual_prefix_count": (
            continuation_counterfactual_prefix_count
        ),
        "profile_continuation_counterfactual_oracle_mean_raw_delta": (
            continuation_counterfactual_oracle_mean_raw_delta
        ),
        "profile_continuation_counterfactual_oracle_positive_rate": (
            continuation_counterfactual_oracle_positive_rate
        ),
        "profile_continuation_counterfactual_best_fixed_mean_raw_delta": (
            continuation_counterfactual_best_fixed_mean_raw_delta
        ),
        "profile_continuation_counterfactual_oracle_vs_fixed_gap": (
            continuation_counterfactual_oracle_vs_fixed_gap
        ),
        "profile_continuation_counterfactual_diagnostics_by_split": (
            continuation_counterfactual_diagnostics_by_split
        ),
        "profile_counterfactual_selector_metrics": (
            counterfactual_selector_metrics
        ),
        "identity_strong_success_delta_vs_best_control": (
            identity_success_delta
        ),
        "identity_terminal_success_delta_vs_best_control": (
            identity_terminal_delta
        ),
        "identity_effect_delta_vs_zero": identity_effect_delta_vs_zero,
        "heldout_identity_control_exposure": (
            heldout_identity_control_exposure
        ),
        "identity_continuation_nll_delta_vs_best_control": (
            heldout_identity_nll_delta
            if heldout_identity_nll_delta is not None
            else 0.0
        ),
        "identity_eval_nll_delta_vs_best_control": (
            eval_identity_nll_delta
        ),
        "identity_confirmation_nll_delta_vs_best_control": (
            confirmation_identity_nll_delta
        ),
        "mechanism_gates": mechanism_gates,
        "causal_gates": causal_gates,
        "selector_gates": selector_gates,
        "selector_injection_allowed": all(selector_gates.values()),
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
        "selector_gates": ablation["selector_gates"],
        "selector_injection_allowed": ablation[
            "selector_injection_allowed"
        ],
        "selector_blockers": [
            gate
            for gate, passed in ablation["selector_gates"].items()
            if not passed
        ],
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
            "coverage, a fresh locked confirmation split, semantic no-label "
            "evidence, and Gates 1/3-10 remain separate prerequisites."
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
        action_selection,
        metric_samples,
    ) = _flatten_evidence(payloads)
    descriptor = _runtime_descriptor(report)
    confirmation_split_locked = next(
        (
            values[0].lower() == "true"
            for name, values in report.manifest.case_groups
            if name == "confirmation_split_locked"
        ),
        False,
    )
    route_ids = [
        value
        for name, values in report.manifest.case_groups
        if name == "route_ids"
        for value in values
    ]
    case_groups = dict(report.manifest.case_groups)
    independent_training_route_ids = case_groups.get(
        "independent_training_route_ids",
        (),
    )
    validation_route_ids = case_groups.get(
        "validation_route_ids",
        (),
    )
    validation_frozen_values = case_groups.get(
        "validation_frozen_before_run",
        (),
    )
    training_route_count_values = case_groups.get(
        "training_route_count",
        (),
    )
    development_routes_unchanged_values = case_groups.get(
        "development_routes_unchanged",
        (),
    )
    residual_activation_width_values = case_groups.get(
        "real_residual_activation_width",
        (),
    )
    if len(training_route_count_values) != 1:
        raise ValueError(
            "Gate 2 manifest requires exactly one training_route_count"
        )
    if len(development_routes_unchanged_values) != 1:
        raise ValueError(
            "Gate 2 manifest requires exactly one "
            "development_routes_unchanged value"
        )
    if len(residual_activation_width_values) != 1:
        raise ValueError(
            "Gate 2 manifest requires exactly one "
            "real_residual_activation_width value"
        )
    if len(validation_frozen_values) != 1:
        raise ValueError(
            "Gate 2 manifest requires exactly one "
            "validation_frozen_before_run value"
        )
    validation_frozen = (
        validation_frozen_values[0].lower() == "true"
    )
    if not validation_route_ids or not validation_frozen:
        raise ValueError(
            "Gate 2 v30 requires frozen validation routes before execution"
        )
    if any(
        route_id not in route_ids for route_id in validation_route_ids
    ):
        raise ValueError(
            "Gate 2 validation_route_ids must be included in route_ids"
        )
    residual_activation_width = int(
        residual_activation_width_values[0]
    )
    if descriptor.get("primary_backend") == "transformers-open-weight":
        observed_activation_width = int(
            descriptor.get("open_weight_activation_width", "0")
        )
        if observed_activation_width != residual_activation_width:
            raise ValueError(
                "Gate 2 residual activation width mismatch: "
                f"manifest={residual_activation_width}, "
                f"runtime={observed_activation_width}"
            )
    training_route_count = int(training_route_count_values[0])
    development_routes_unchanged = (
        development_routes_unchanged_values[0].lower() == "true"
    )
    if training_route_count != sum(
        route_id.startswith("train-") for route_id in route_ids
    ):
        raise ValueError(
            "Gate 2 training_route_count does not match route_ids"
        )
    if not development_routes_unchanged:
        raise ValueError(
            "Gate 2 causal packet requires unchanged development routes"
        )
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
            "independent_training_route_ids": list(
                independent_training_route_ids
            ),
            "validation_route_ids": list(validation_route_ids),
            "validation_frozen_before_run": validation_frozen,
            "training_route_count": training_route_count,
            "development_routes_unchanged": (
                development_routes_unchanged
            ),
            "split_owner": descriptor.get(
                "case_corpus",
                ETA_GATE2_EXPECTED_VALUE_CASE_CORPUS,
            ),
            "development_heldout_status": "reused-during-gate-development",
            "causal_confirmation_split": None,
            "confirmation_split_locked": confirmation_split_locked,
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
        "counterfactual_action_selector": {
            "diagnostic_active": case_groups[
                "counterfactual_action_selector_diagnostic"
            ][0].lower()
            == "true",
            "input": case_groups[
                "counterfactual_action_selector_input"
            ][0],
            "encoder": case_groups[
                "counterfactual_action_selector_encoder"
            ][0],
            "head": case_groups[
                "counterfactual_action_selector_head"
            ][0],
            "model_selection": case_groups[
                "counterfactual_action_selector_model_selection"
            ][0],
            "cross_validation": case_groups[
                "counterfactual_action_selector_cv"
            ][0],
            "live_injection": case_groups[
                "counterfactual_action_selector_live_injection"
            ][0],
        },
        "counterfactual_target": {
            "mode": case_groups["counterfactual_target_mode"][0],
            "target_sample_count": int(
                case_groups["counterfactual_target_sample_count"][0]
            ),
            "audit_sample_count": int(
                case_groups["counterfactual_audit_sample_count"][0]
            ),
            "sampling_temperature": float(
                case_groups[
                    "counterfactual_sampling_temperature"
                ][0]
            ),
            "sampling_max_new_tokens": int(
                case_groups[
                    "counterfactual_sampling_max_new_tokens"
                ][0]
            ),
            "sampling_seed_protocol": case_groups[
                "counterfactual_sampling_seed_protocol"
            ][0],
            "audit_role": (
                "readout-and-train-route-cv-model-selection-only"
            ),
        },
        "residual_capture": {
            "activation_width": residual_activation_width,
            "compression_mode": "exact-up-to-configured-width",
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
        confirmation_split_locked=confirmation_split_locked,
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
                "- identity 对 development-heldout best control 的 "
                "continuation-NLL 改善："
                f"`{ablation['identity_continuation_nll_delta_vs_best_control']:.6f}`"
            ),
            (
                "- identity 对 eval best control 的 continuation-NLL 改善："
                f"`{ablation['identity_eval_nll_delta_vs_best_control']}`"
            ),
            (
                "- identity 对 fresh confirmation best control 的 "
                "continuation-NLL 改善："
                f"`{ablation['identity_confirmation_nll_delta_vs_best_control']}`"
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
            f"- selector gates：`{ablation['selector_gates']}`",
            (
                "- selector shadow injection allowed："
                f"`{ablation['selector_injection_allowed']}`"
            ),
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
        _write_jsonl(
            target / "action_selection.jsonl",
            action_selection,
        ),
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
