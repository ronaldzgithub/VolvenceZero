from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean
from typing import Any

from volvence_zero.agent.eta_proof_benchmark import (
    ETA_COUNTERFACTUAL_TARGET_ENVIRONMENT_OUTCOME,
    ETA_CONTINUATION_PE_REWARD_SCALE,
    ETA_CONTINUATION_PE_TRAINING_SIGNAL,
    ETA_GATE2_RECENT_K2_FRESH_CASE_CORPUS,
    ETA_GATE2_SHADOW_FRESH_CASE_CORPUS,
    ETAProofPaperSuiteAggregateReport,
    build_eta_open_weight_paper_suite_manifest,
    eta_gate2_expected_value_validation_routes,
    eta_gate2_independent_training_routes,
    eta_gate2_recent_k2_confirmation_routes,
    eta_gate2_recent_k2_fresh_routes,
    eta_gate2_recent_k2_fresh_validation_routes,
    eta_gate2_selector_confirmation_routes,
    eta_gate2_selector_fresh_validation_routes,
    eta_gate2_shadow_confirmation_routes,
    eta_gate2_shadow_fresh_routes,
    eta_gate2_shadow_fresh_validation_routes,
)
from volvence_zero.agent.paper_suite import PaperProfileSpec, PaperSuiteManifest
from volvence_zero.substrate import TRAIN_TRANSITION_PCA_CONTROL_BASIS_MODE


ETA_GATE2_SCHEMA_VERSION = "eta-gate2-residual-causal.v36"
ETA_GATE2_RECENT_K2_FORMAL_SCHEMA_VERSION = (
    "eta-gate2-residual-causal.v37"
)
ETA_GATE2_RECENT_K_DIAGNOSTIC_SCHEMA_VERSION = (
    "eta-gate2-v36-recent-k-diagnostic.v1"
)
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
    "counterfactual_outcomes.jsonl",
    "selector_artifact.json",
    "shadow_closed_loop.jsonl",
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
        route.case_id for route in eta_gate2_shadow_fresh_routes()
    )
    independent_route_ids = tuple(
        route.case_id
        for route in eta_gate2_independent_training_routes()
    )
    validation_route_ids = tuple(
        route.case_id
        for route in eta_gate2_shadow_fresh_validation_routes()
    )
    confirmation_route_ids = tuple(
        route.case_id
        for route in eta_gate2_shadow_confirmation_routes()
    )
    superseded_validation_route_ids = tuple(
        route.case_id
        for route in (
            eta_gate2_expected_value_validation_routes()
            + eta_gate2_selector_fresh_validation_routes()
            + eta_gate2_selector_confirmation_routes()
        )
    )
    base_case_groups = tuple(
        ("route_ids", expanded_route_ids)
        if name == "route_ids"
        else (name, values)
        for name, values in base.case_groups
    )
    case_groups = base_case_groups + (
        ("case_corpus", (ETA_GATE2_SHADOW_FRESH_CASE_CORPUS,)),
        ("independent_training_route_ids", independent_route_ids),
        ("validation_route_ids", validation_route_ids),
        ("confirmation_route_ids", confirmation_route_ids),
        (
            "superseded_validation_route_ids",
            superseded_validation_route_ids,
        ),
        ("selector_signal_gate", ("selector-vs-permutation-null-v1",)),
        ("shadow_observation_gate", ("shadow-closed-loop-v1",)),
        ("shadow_closed_loop_arm", ("true",)),
        ("validation_frozen_before_run", ("true",)),
        ("confirmation_split_locked", ("true",)),
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
            (ETA_COUNTERFACTUAL_TARGET_ENVIRONMENT_OUTCOME,),
        ),
        ("counterfactual_target_sample_count", ("1",)),
        ("counterfactual_audit_sample_count", ("1",)),
        ("counterfactual_sampling_temperature", ("0.0",)),
        ("counterfactual_sampling_max_new_tokens", ("0",)),
        (
            "counterfactual_sampling_seed_protocol",
            ("not-applicable-deterministic-environment-forward",),
        ),
        (
            "counterfactual_reward_normalization",
            ("pe-action-credit-per-prefix-centered-range-v1",),
        ),
        (
            "counterfactual_outcome_chain",
            (
                "z-candidate->residual-forward->realized-continuation-nll"
                "->prediction-error->action-credit",
            ),
        ),
        (
            "counterfactual_primary_target",
            (
                "realized-next-segment-teacher-forced-"
                "nll-improvement-vs-zero-control",
            ),
        ),
        (
            "counterfactual_audit_surface",
            (
                "subsequent-realized-segment-teacher-forced-"
                "nll-improvement-vs-zero-control",
            ),
        ),
        ("fresh_validation_split_locked", ("true",)),
        (
            "control_basis_mode",
            (TRAIN_TRANSITION_PCA_CONTROL_BASIS_MODE,),
        ),
        ("control_basis_fit_split", ("train",)),
        ("control_basis_rank", ("3",)),
        (
            "control_basis_state_coordinate",
            ("hook-layer-mean-last-token-hidden",),
        ),
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
        repeat_count=3,
        seed_schedule=(0, 1, 2),
        profiles=profiles,
        case_groups=case_groups,
        artifact_expectations=ETA_GATE2_REQUIRED_FILES,
        description=(
            "Gate 2 matched residual intervention suite. The identity arm is "
            "compared with zero, deterministic shuffled, and reversed U_t on "
            "the same frozen substrate and schedule."
        ),
    )


def build_eta_gate2_recent_k_diagnostic_manifest(
    *,
    committed_control_window: int,
    suite_tier: str = "ci-smoke",
) -> PaperSuiteManifest:
    if committed_control_window not in {1, 2}:
        raise ValueError(
            "Gate 2 v36 recent-k diagnostic only permits k=1 or k=2"
        )
    base = build_eta_gate2_residual_manifest(suite_tier=suite_tier)
    baseline_profiles = tuple(
        profile
        for profile in base.profiles
        if profile.profile_label == base.baseline_label
    )
    if len(baseline_profiles) != 1:
        raise ValueError(
            "Gate 2 recent-k diagnostic requires exactly one baseline "
            "profile"
        )
    return replace(
        base,
        suite_id=(
            "eta-gate2-v36-recent-k"
            f"{committed_control_window}-diagnostic-{suite_tier}"
        ),
        version=base.version + 1,
        repeat_count=1,
        seed_schedule=(0,),
        profiles=baseline_profiles,
        case_groups=base.case_groups
        + (
            (
                "shadow_committed_control_window",
                (str(committed_control_window),),
            ),
            ("evidence_claim_scope", ("development-diagnostic-only",)),
            ("observed_v36_routes_reused", ("true",)),
        ),
        description=(
            "Development-only replay of the locked v36 closed-loop routes "
            f"with recent-k={committed_control_window}. This manifest cannot "
            "produce a formal promotion or SHADOW admission verdict."
        ),
    )


def build_eta_gate2_recent_k2_formal_manifest(
    *,
    suite_tier: str = "ci-smoke",
) -> PaperSuiteManifest:
    base = build_eta_gate2_residual_manifest(suite_tier=suite_tier)
    baseline_profiles = tuple(
        profile
        for profile in base.profiles
        if profile.profile_label == base.baseline_label
    )
    if len(baseline_profiles) != 1:
        raise ValueError(
            "Gate 2 v37 formal requires exactly one inherited-causal "
            "baseline profile"
        )
    route_ids = tuple(
        route.case_id for route in eta_gate2_recent_k2_fresh_routes()
    )
    validation_route_ids = tuple(
        route.case_id
        for route in eta_gate2_recent_k2_fresh_validation_routes()
    )
    confirmation_route_ids = tuple(
        route.case_id
        for route in eta_gate2_recent_k2_confirmation_routes()
    )
    superseded_route_ids = tuple(
        route.case_id
        for route in (
            eta_gate2_expected_value_validation_routes()
            + eta_gate2_selector_fresh_validation_routes()
            + eta_gate2_selector_confirmation_routes()
            + eta_gate2_shadow_fresh_validation_routes()
            + eta_gate2_shadow_confirmation_routes()
        )
    )
    replacements = {
        "route_ids": route_ids,
        "case_corpus": (ETA_GATE2_RECENT_K2_FRESH_CASE_CORPUS,),
        "validation_route_ids": validation_route_ids,
        "confirmation_route_ids": confirmation_route_ids,
        "superseded_validation_route_ids": superseded_route_ids,
        "shadow_observation_gate": (
            "recent-k2-shadow-closed-loop-v2",
        ),
    }
    case_groups = tuple(
        (name, replacements.get(name, values))
        for name, values in base.case_groups
    ) + (
        (
            "evidence_schema_version",
            (ETA_GATE2_RECENT_K2_FORMAL_SCHEMA_VERSION,),
        ),
        ("shadow_committed_control_window", ("2",)),
        ("causal_packet_source", ("inherited-v35-locked",)),
        ("formal_claim_scope", ("shadow-admission-only",)),
        ("recent_k_development_selection_locked", ("true",)),
    )
    return replace(
        base,
        suite_id=f"eta-gate2-recent-k2-formal-{suite_tier}",
        version=base.version + 1,
        profiles=baseline_profiles,
        case_groups=case_groups,
        description=(
            "Gate 2 v37 formal recent-k=2 closed-loop SHADOW admission "
            "suite on fresh validation and locked confirmation routes. "
            "The v35 causal packet is inherited and not rerun."
        ),
    )


def export_eta_gate2_recent_k_diagnostic_bundle(
    report: ETAProofPaperSuiteAggregateReport,
    *,
    output_dir: str | Path,
    source_v36_artifact: str,
) -> tuple[Path, ...]:
    case_groups = dict(report.manifest.case_groups)
    claim_scope = case_groups.get("evidence_claim_scope", ())
    observed_route_reuse = case_groups.get(
        "observed_v36_routes_reused",
        (),
    )
    window_values = case_groups.get(
        "shadow_committed_control_window",
        (),
    )
    if (
        claim_scope != ("development-diagnostic-only",)
        or observed_route_reuse != ("true",)
        or len(window_values) != 1
    ):
        raise ValueError(
            "Gate 2 recent-k exporter requires an explicitly non-formal "
            "observed-route diagnostic manifest"
        )
    committed_control_window = int(window_values[0])
    if committed_control_window not in {1, 2}:
        raise ValueError(
            "Gate 2 recent-k exporter only permits committed control "
            "windows 1 and 2"
        )
    benchmark = report.reference_benchmark_report
    if benchmark is None:
        raise ValueError(
            "Gate 2 recent-k diagnostic requires a benchmark report"
        )
    profiles = tuple(
        profile
        for profile in benchmark.profile_reports
        if profile.profile_label == report.manifest.baseline_label
    )
    if len(profiles) != 1:
        raise ValueError(
            "Gate 2 recent-k diagnostic requires exactly one baseline "
            "profile report"
        )
    profile = profiles[0]
    rows = [
        {
            "profile_label": profile.profile_label,
            **asdict(record),
        }
        for record in profile.shadow_closed_loop_records
    ]
    if not rows:
        raise ValueError(
            "Gate 2 recent-k diagnostic produced no closed-loop records"
        )
    if any(
        row["committed_control_window"] != committed_control_window
        for row in rows
    ):
        raise ValueError(
            "Gate 2 recent-k records do not match the manifest window"
        )
    selector_artifact = profile.selector_artifact_payload
    if selector_artifact is None:
        raise ValueError(
            "Gate 2 recent-k diagnostic requires the frozen selector "
            "artifact payload"
        )

    trajectory_totals: dict[tuple[str, str, str], float] = {}
    split_step_deltas: dict[str, list[float]] = {}
    split_aggregate_norms: dict[str, list[float]] = {}
    split_active_counts: dict[str, list[int]] = {}
    for row in rows:
        split = str(row["split"])
        arm = str(row["arm"])
        trajectory_key = (split, str(row["case_id"]), arm)
        trajectory_totals[trajectory_key] = (
            trajectory_totals.get(trajectory_key, 0.0)
            + float(row["realized_delta"])
        )
        if arm != "selector":
            continue
        split_step_deltas.setdefault(split, []).append(
            float(row["realized_delta"])
        )
        aggregate_control = tuple(
            float(value) for value in row["aggregate_control"]
        )
        split_aggregate_norms.setdefault(split, []).append(
            sum(value * value for value in aggregate_control) ** 0.5
        )
        split_active_counts.setdefault(split, []).append(
            int(row["active_control_count"])
        )

    split_metrics: dict[str, dict[str, float]] = {}
    for split in ("train", "eval", "heldout", "validation", "confirmation"):
        route_ids = sorted(
            {
                case_id
                for row_split, case_id, _arm in trajectory_totals
                if row_split == split
            }
        )
        selector_zero: list[float] = []
        selector_permutation: list[float] = []
        for case_id in route_ids:
            selector = trajectory_totals.get(
                (split, case_id, "selector")
            )
            zero = trajectory_totals.get(
                (split, case_id, "zero-control")
            )
            permutation = trajectory_totals.get(
                (split, case_id, "permutation-null")
            )
            if selector is None or zero is None or permutation is None:
                raise ValueError(
                    "Gate 2 recent-k diagnostic requires complete "
                    f"three-arm trajectories for {split}/{case_id}"
                )
            selector_zero.append(selector - zero)
            selector_permutation.append(selector - permutation)
        step_deltas = split_step_deltas.get(split, [])
        aggregate_norms = split_aggregate_norms.get(split, [])
        active_counts = split_active_counts.get(split, [])
        split_metrics[split] = {
            "trajectory_count": float(len(route_ids)),
            "selector_minus_zero_mean": _mean_or_zero(selector_zero),
            "selector_minus_permutation_mean": _mean_or_zero(
                selector_permutation
            ),
            "selected_step_realized_delta_mean": _mean_or_zero(
                step_deltas
            ),
            "selector_aggregate_norm_mean": _mean_or_zero(
                aggregate_norms
            ),
            "selector_aggregate_norm_max": (
                max(aggregate_norms) if aggregate_norms else 0.0
            ),
            "active_control_count_max": float(
                max(active_counts) if active_counts else 0
            ),
        }

    formal_split_names = ("train", "validation", "confirmation")
    development_gate_passed = all(
        split_metrics[split][metric_name]
        >= ETA_GATE2_MIN_MEASURABLE_EFFECT
        for split in formal_split_names
        for metric_name in (
            "selector_minus_zero_mean",
            "selector_minus_permutation_mean",
            "selected_step_realized_delta_mean",
        )
    )
    selector_fingerprint = str(
        selector_artifact["artifact"]["model_fingerprint"]
    )
    basis_fingerprint = str(
        selector_artifact["control_basis_fingerprint"]
    )
    diagnostic = {
        "schema_version": (
            ETA_GATE2_RECENT_K_DIAGNOSTIC_SCHEMA_VERSION
        ),
        "claim_scope": "development-diagnostic-only",
        "formal_promotion_or_shadow_admission_allowed": False,
        "observed_v36_routes_reused": True,
        "source_v36_artifact": source_v36_artifact,
        "committed_control_window": committed_control_window,
        "selection_rule": (
            "prefer-k1-if-all-formal-split-metrics-positive-else-k2-"
            "if-all-positive-else-stop-recent-k"
        ),
        "development_gate_passed": development_gate_passed,
        "split_metrics": split_metrics,
        "record_count": len(rows),
        "selector_fingerprint": selector_fingerprint,
        "control_basis_fingerprint": basis_fingerprint,
        "live_injection": "disabled",
        "next_formal_requirement": (
            "preregister fresh validation and locked confirmation routes, "
            "then run 3 seeds with the single selected k"
        ),
    }
    manifest_payload = {
        "schema_version": (
            ETA_GATE2_RECENT_K_DIAGNOSTIC_SCHEMA_VERSION
        ),
        "suite_id": report.manifest.suite_id,
        "suite_version": report.manifest.version,
        "seed_schedule": list(report.manifest.seed_schedule),
        "repeat_count": report.manifest.repeat_count,
        "case_groups": {
            name: list(values)
            for name, values in report.manifest.case_groups
        },
        "claim_scope": "development-diagnostic-only",
        "source_v36_artifact": source_v36_artifact,
    }
    report_markdown = "\n".join(
        (
            "# Gate 2 v36 recent-k 开发诊断",
            "",
            f"- k: `{committed_control_window}`",
            "- 证据范围：`development-diagnostic-only`",
            "- 旧 v36 routes：已观察，仅用于根因选择",
            (
                "- development gate: "
                f"`{'PASS' if development_gate_passed else 'FAIL'}`"
            ),
            "- formal promotion / SHADOW admission：`不允许`",
            "- live injection：`disabled`",
            "",
            "正式结论必须使用预注册 fresh routes 与 3 seeds。",
            "",
        )
    )
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths = (
        _write_json(target / "diagnostic_manifest.json", manifest_payload),
        _write_json(
            target / "recent_k_diagnostic.json",
            diagnostic,
        ),
        _write_json(
            target / "selector_artifact.json",
            selector_artifact,
        ),
        _write_jsonl(target / "shadow_closed_loop.jsonl", rows),
    )
    report_path = target / "report.md"
    report_path.write_text(report_markdown, encoding="utf-8")
    return (*paths, report_path)


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


def _oracle_permutation_null_by_split(
    counterfactual_outcomes: list[dict[str, Any]],
    *,
    split_names: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    """Cross-cohort oracle-vs-permutation-null diagnostics per split.

    For every prefix we take the candidate that maximizes the target-cohort
    action credit and evaluate that same candidate on the independent audit
    cohort. Under the no-signal null (exchangeable candidates), the audit
    value of the target-argmax candidate equals the audit mean over all
    candidates, so `transfer_excess_over_null_mean` estimates how much of the
    observed oracle survives an independent re-measurement. A target-cohort
    oracle that does not transfer is max-of-noise selection bias, not
    evidence of a reachable solution (known-debts #92 Gate 2, 2026-07-29).
    """
    grid: dict[tuple[str, str], list[tuple[int, float, float]]] = {}
    for row in counterfactual_outcomes:
        if row["profile_label"] != "full-internal-rl":
            continue
        prefix_key = str(row["observation_id"]).rsplit(":", 1)[0]
        grid.setdefault(
            (str(row["split"]), prefix_key),
            [],
        ).append(
            (
                int(row["candidate_index"]),
                float(row["action_credit"]),
                float(row["audit_action_credit"]),
            )
        )
    diagnostics: dict[str, dict[str, float]] = {}
    for split_name in split_names:
        oracle_target: list[float] = []
        transfer_audit: list[float] = []
        permutation_null_audit: list[float] = []
        for (split, _prefix), candidates in grid.items():
            if split != split_name or len(candidates) < 2:
                continue
            ordered = sorted(candidates)
            target_credits = [credit for _, credit, _ in ordered]
            audit_credits = [credit for _, _, credit in ordered]
            best_index = max(
                range(len(target_credits)),
                key=lambda index: target_credits[index],
            )
            oracle_target.append(target_credits[best_index])
            transfer_audit.append(audit_credits[best_index])
            permutation_null_audit.append(
                mean(audit_credits)
            )
        diagnostics[split_name] = {
            "prefix_count": float(len(oracle_target)),
            "observed_oracle_target_mean": _mean_or_zero(oracle_target),
            "transfer_oracle_audit_mean": _mean_or_zero(transfer_audit),
            "permutation_null_audit_mean": _mean_or_zero(
                permutation_null_audit
            ),
            "transfer_excess_over_null_mean": (
                _mean_or_zero(transfer_audit)
                - _mean_or_zero(permutation_null_audit)
            ),
        }
    return diagnostics


def _selection_step_index(example_id: str) -> int:
    try:
        _case_prefix, step_text = example_id.rsplit(":prefix-", 1)
        return int(step_text)
    except ValueError as exc:
        raise ValueError(
            "counterfactual selector example_id must end with ':prefix-N', "
            f"got {example_id!r}"
        ) from exc


def _selector_permutation_null_by_split(
    action_selection: list[dict[str, Any]],
    counterfactual_outcomes: list[dict[str, Any]],
    *,
    split_names: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    """Selector-vs-permutation-null diagnostics per split.

    The v35 reachable-solution claim is conditional action value, not
    marginal action value. For each selector prediction we compare the
    selected candidate's independent audit credit with the mean audit credit
    of all candidates for the same (split, route, prefix) grid row. Under
    an exchangeable no-signal null, a train-fit selector should equal that
    per-prefix candidate mean on frozen validation/confirmation.
    """

    audit_grid: dict[tuple[str, str, int], dict[int, float]] = {}
    for row in counterfactual_outcomes:
        if row["profile_label"] != "full-internal-rl":
            continue
        key = (
            str(row["split"]),
            str(row["case_id"]),
            int(row["step_index"]),
        )
        candidate_index = int(row["candidate_index"])
        candidates = audit_grid.setdefault(key, {})
        audit_credit = float(row["audit_action_credit"])
        existing_credit = candidates.get(candidate_index)
        if (
            existing_credit is not None
            and abs(existing_credit - audit_credit) > 1e-9
        ):
            raise ValueError(
                "selector permutation-null grid contains conflicting "
                f"audit credit for candidate index {candidate_index} at "
                f"{key!r}: {existing_credit!r} != {audit_credit!r}"
            )
        candidates[candidate_index] = audit_credit

    diagnostics: dict[str, dict[str, float]] = {}
    for split_name in split_names:
        selected_audit: list[float] = []
        permutation_null_audit: list[float] = []
        selected_positive = 0
        input_selection_count = 0
        missing_grid = 0
        missing_selected_candidate = 0
        selected_audit_mismatch = 0
        for row in action_selection:
            if row["profile_label"] != "full-internal-rl":
                continue
            if str(row["split"]) != split_name:
                continue
            audit_value = row.get("audit_selected_raw_delta")
            if audit_value is None:
                continue
            input_selection_count += 1
            key = (
                split_name,
                str(row["group_id"]),
                _selection_step_index(str(row["example_id"])),
            )
            candidate_audits = audit_grid.get(key)
            if not candidate_audits:
                missing_grid += 1
                continue
            selected_index = int(row["selected_action_index"])
            selected_grid_audit = candidate_audits.get(selected_index)
            if selected_grid_audit is None:
                missing_selected_candidate += 1
                continue
            selected = float(audit_value)
            if abs(selected - selected_grid_audit) > 1e-9:
                selected_audit_mismatch += 1
                continue
            selected_audit.append(selected)
            permutation_null_audit.append(mean(candidate_audits.values()))
            if selected > 0.0:
                selected_positive += 1
        count = len(selected_audit)
        diagnostics[split_name] = {
            "input_selection_count": float(input_selection_count),
            "selection_count": float(count),
            "selected_audit_mean": _mean_or_zero(selected_audit),
            "selected_positive_rate": (
                selected_positive / count if count else 0.0
            ),
            "permutation_null_audit_mean": _mean_or_zero(
                permutation_null_audit
            ),
            "selected_excess_over_null_mean": (
                _mean_or_zero(selected_audit)
                - _mean_or_zero(permutation_null_audit)
            ),
            "missing_counterfactual_grid_count": float(missing_grid),
            "missing_selected_candidate_count": float(
                missing_selected_candidate
            ),
            "selected_audit_lineage_mismatch_count": float(
                selected_audit_mismatch
            ),
        }
    return diagnostics


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
    counterfactual_outcomes: list[dict[str, Any]] = []
    selector_artifacts: list[dict[str, Any]] = []
    shadow_closed_loop: list[dict[str, Any]] = []
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
            outcome_records = profile.get(
                "counterfactual_outcome_records",
                [],
            )
            if not isinstance(outcome_records, list):
                raise ValueError(
                    "profile report counterfactual_outcome_records must "
                    "be a list"
                )
            for record in outcome_records:
                counterfactual_outcomes.append(
                    {
                        "run_id": run["run_id"],
                        "run_seed": run["run_seed"],
                        "profile_label": profile_label,
                        **record,
                    }
                )
            selector_artifact = profile.get(
                "selector_artifact_payload"
            )
            if selector_artifact is not None:
                if not isinstance(selector_artifact, dict):
                    raise ValueError(
                        "profile report selector_artifact_payload must be "
                        "an object or null"
                    )
                if (
                    selector_artifact.get("run_id") != run["run_id"]
                    or selector_artifact.get("run_seed")
                    != run["run_seed"]
                ):
                    raise ValueError(
                        "selector artifact run lineage does not match its "
                        "paper-suite run"
                    )
                selector_artifacts.append(selector_artifact)
            shadow_records = profile.get(
                "shadow_closed_loop_records",
                [],
            )
            if not isinstance(shadow_records, list):
                raise ValueError(
                    "profile report shadow_closed_loop_records must be a list"
                )
            for record in shadow_records:
                if (
                    record.get("run_id") != run["run_id"]
                    or record.get("run_seed") != run["run_seed"]
                ):
                    raise ValueError(
                        "closed-loop SHADOW record run lineage does not "
                        "match its paper-suite run"
                    )
                shadow_closed_loop.append(
                    {
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
        counterfactual_outcomes,
        selector_artifacts,
        shadow_closed_loop,
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


def _shadow_closed_loop_diagnostics(
    *,
    rows: list[dict[str, Any]],
    selector_artifacts: list[dict[str, Any]],
    split_names: tuple[str, ...],
) -> tuple[dict[str, dict[str, float]], bool, bool]:
    expected_arms = {
        "selector",
        "zero-control",
        "permutation-null",
    }
    artifact_by_run: dict[tuple[str, int], dict[str, Any]] = {}
    artifact_provenance_valid = True
    for artifact in selector_artifacts:
        run_key = (
            str(artifact.get("run_id", "")),
            int(artifact.get("run_seed", -1)),
        )
        nested = artifact.get("artifact")
        if (
            not run_key[0]
            or run_key[1] < 0
            or artifact.get("fit_split") != "train"
            or not artifact.get("control_basis_fingerprint")
            or not isinstance(nested, dict)
            or not nested.get("model_fingerprint")
            or run_key in artifact_by_run
        ):
            artifact_provenance_valid = False
            continue
        artifact_by_run[run_key] = artifact

    grouped_steps: dict[
        tuple[str, int, str, int],
        dict[str, dict[str, Any]],
    ] = {}
    trajectory_totals: dict[
        tuple[str, int, str, str],
        float,
    ] = {}
    for row in rows:
        if row.get("profile_label") != "full-internal-rl":
            continue
        split = str(row["split"])
        run_seed = int(row["run_seed"])
        case_id = str(row["case_id"])
        step_index = int(row["step_index"])
        arm = str(row["arm"])
        step_key = (split, run_seed, case_id, step_index)
        arms = grouped_steps.setdefault(step_key, {})
        if arm in arms:
            artifact_provenance_valid = False
        arms[arm] = row
        trajectory_key = (split, run_seed, case_id, arm)
        trajectory_totals[trajectory_key] = (
            trajectory_totals.get(trajectory_key, 0.0)
            + float(row["realized_delta"])
        )
        artifact = artifact_by_run.get(
            (str(row["run_id"]), run_seed)
        )
        nested = artifact.get("artifact") if artifact is not None else None
        if (
            artifact is None
            or not isinstance(nested, dict)
            or row.get("selector_fingerprint")
            != nested.get("model_fingerprint")
            or row.get("control_basis_fingerprint")
            != artifact.get("control_basis_fingerprint")
            or not row.get("runtime_descriptor_fingerprint")
            or row.get("side_effect_free") is not True
        ):
            artifact_provenance_valid = False

    diagnostics: dict[str, dict[str, float]] = {}
    for split_name in split_names:
        split_steps = {
            key: arms
            for key, arms in grouped_steps.items()
            if key[0] == split_name
        }
        complete_step_count = sum(
            set(arms) == expected_arms
            for arms in split_steps.values()
        )
        trajectory_keys = {
            (split, seed, case_id)
            for split, seed, case_id, _arm in trajectory_totals
            if split == split_name
        }
        selector_zero_by_trajectory = []
        selector_permutation_by_trajectory = []
        selector_zero_by_seed: dict[int, list[float]] = {}
        selector_permutation_by_seed: dict[int, list[float]] = {}
        for split, seed, case_id in sorted(trajectory_keys):
            selector = trajectory_totals.get(
                (split, seed, case_id, "selector")
            )
            zero = trajectory_totals.get(
                (split, seed, case_id, "zero-control")
            )
            permutation = trajectory_totals.get(
                (split, seed, case_id, "permutation-null")
            )
            if selector is None or zero is None or permutation is None:
                continue
            selector_zero = selector - zero
            selector_permutation = selector - permutation
            selector_zero_by_trajectory.append(selector_zero)
            selector_permutation_by_trajectory.append(
                selector_permutation
            )
            selector_zero_by_seed.setdefault(seed, []).append(
                selector_zero
            )
            selector_permutation_by_seed.setdefault(seed, []).append(
                selector_permutation
            )
        selector_zero_seed_means = tuple(
            _mean_or_zero(values)
            for _seed, values in sorted(selector_zero_by_seed.items())
        )
        selector_permutation_seed_means = tuple(
            _mean_or_zero(values)
            for _seed, values in sorted(
                selector_permutation_by_seed.items()
            )
        )
        selector_step_deltas = [
            float(arms["selector"]["realized_delta"])
            for arms in split_steps.values()
            if "selector" in arms
        ]
        selector_step_deltas_by_seed: dict[int, list[float]] = {}
        for (_split, seed, _case_id, _step_index), arms in (
            split_steps.items()
        ):
            selector_row = arms.get("selector")
            if selector_row is not None:
                selector_step_deltas_by_seed.setdefault(seed, []).append(
                    float(selector_row["realized_delta"])
                )
        selector_step_seed_means = tuple(
            _mean_or_zero(values)
            for _seed, values in sorted(
                selector_step_deltas_by_seed.items()
            )
        )
        first_half_deltas = []
        second_half_deltas = []
        max_step_by_trajectory: dict[tuple[int, str], int] = {}
        for _split, seed, case_id, step_index in split_steps:
            key = (seed, case_id)
            max_step_by_trajectory[key] = max(
                max_step_by_trajectory.get(key, -1),
                step_index,
            )
        for (
            _split,
            seed,
            case_id,
            step_index,
        ), arms in split_steps.items():
            selector_row = arms.get("selector")
            if selector_row is None:
                continue
            midpoint = (
                max_step_by_trajectory[(seed, case_id)] + 1
            ) / 2.0
            target = (
                first_half_deltas
                if step_index < midpoint
                else second_half_deltas
            )
            target.append(float(selector_row["realized_delta"]))
        diagnostics[split_name] = {
            "record_count": float(
                sum(len(arms) for arms in split_steps.values())
            ),
            "step_count": float(len(split_steps)),
            "complete_step_count": float(complete_step_count),
            "trajectory_count": float(
                len(selector_zero_by_trajectory)
            ),
            "seed_count": float(len(selector_zero_seed_means)),
            "positive_seed_count": float(
                sum(
                    value >= ETA_GATE2_MIN_MEASURABLE_EFFECT
                    for value in selector_zero_seed_means
                )
            ),
            "selector_zero_positive_seed_count": float(
                sum(
                    value >= ETA_GATE2_MIN_MEASURABLE_EFFECT
                    for value in selector_zero_seed_means
                )
            ),
            "selector_permutation_positive_seed_count": float(
                sum(
                    value >= ETA_GATE2_MIN_MEASURABLE_EFFECT
                    for value in selector_permutation_seed_means
                )
            ),
            "selected_step_positive_seed_count": float(
                sum(
                    value >= ETA_GATE2_MIN_MEASURABLE_EFFECT
                    for value in selector_step_seed_means
                )
            ),
            "selector_minus_zero_mean": _mean_or_zero(
                selector_zero_by_trajectory
            ),
            "selector_minus_permutation_mean": _mean_or_zero(
                selector_permutation_by_trajectory
            ),
            "selected_step_realized_delta_mean": _mean_or_zero(
                selector_step_deltas
            ),
            "first_half_selected_delta_mean": _mean_or_zero(
                first_half_deltas
            ),
            "second_half_selected_delta_mean": _mean_or_zero(
                second_half_deltas
            ),
        }
    expected_runs = {
        (str(row["run_id"]), int(row["run_seed"]))
        for row in rows
        if row.get("profile_label") == "full-internal-rl"
    }
    artifact_lineage_valid = (
        artifact_provenance_valid
        and set(artifact_by_run) == expected_runs
    )
    artifact_provenance_valid = (
        artifact_lineage_valid and len(expected_runs) == 3
    )
    return (
        diagnostics,
        artifact_provenance_valid,
        artifact_lineage_valid,
    )


def _build_ablation_results(
    *,
    predictions: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    metric_samples: dict[str, list[float]],
    counterfactual_outcomes: list[dict[str, Any]],
    action_selection: list[dict[str, Any]] | None = None,
    selector_artifacts: list[dict[str, Any]] | None = None,
    shadow_closed_loop: list[dict[str, Any]] | None = None,
    confirmation_split_locked: bool = False,
    inherited_causal_promotion: bool = False,
    schema_version: str = ETA_GATE2_SCHEMA_VERSION,
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
    environment_outcome_target_active = _profile_metric_means(
        metric_samples,
        "counterfactual_environment_outcome_target_active",
    )
    environment_application_count = _profile_metric_means(
        metric_samples,
        "counterfactual_environment_application_count",
    )
    environment_pe_credit_transition_count = _profile_metric_means(
        metric_samples,
        "counterfactual_environment_pe_credit_transition_count",
    )
    self_nll_target_active = _profile_metric_means(
        metric_samples,
        "counterfactual_self_nll_target_active",
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
        "environment_outcome_target_active": (
            environment_outcome_target_active["full-internal-rl"] >= 1.0
        ),
        "environment_forward_observed": (
            environment_application_count["full-internal-rl"] > 0.0
        ),
        "environment_outcome_reaches_pe_credit": (
            environment_pe_credit_transition_count["full-internal-rl"]
            == continuation_pe_training_count["full-internal-rl"]
            and environment_pe_credit_transition_count[
                "full-internal-rl"
            ]
            > 0.0
        ),
        "self_nll_excluded_from_selector_target": (
            self_nll_target_active["full-internal-rl"] == 0.0
        ),
    }
    environment_target_is_active = (
        environment_outcome_target_active["full-internal-rl"] >= 1.0
    )
    if environment_target_is_active:
        eval_audit_selected_delta = counterfactual_selector_metrics[
            "counterfactual_selector_eval_mean_audit_selected_raw_delta"
        ]["full-internal-rl"]
        confirmation_count = counterfactual_selector_metrics[
            "counterfactual_selector_confirmation_count"
        ]["full-internal-rl"]
        confirmation_audit_available = counterfactual_selector_metrics[
            (
                "counterfactual_selector_confirmation_"
                "audit_available_rate"
            )
        ]["full-internal-rl"]
        confirmation_audit_selected_delta = (
            counterfactual_selector_metrics[
                (
                    "counterfactual_selector_confirmation_"
                    "mean_audit_selected_raw_delta"
                )
            ]["full-internal-rl"]
        )
        causal_gates = {
            "eval_environment_outcome_audit_positive": (
                eval_audit_selected_delta > 0.0
            ),
            "fresh_confirmation_environment_audit_available": (
                confirmation_count > 0.0
                and confirmation_audit_available >= 1.0
            ),
            "fresh_confirmation_split_locked": (
                confirmation_split_locked
                and confirmation_count > 0.0
                and confirmation_audit_available >= 1.0
            ),
            "confirmation_environment_outcome_audit_positive": (
                confirmation_audit_selected_delta
                >= ETA_GATE2_MIN_MEASURABLE_EFFECT
            ),
        }
    else:
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
            all(
                counterfactual_selector_metrics[
                    f"counterfactual_selector_{split}_count"
                ]["full-internal-rl"]
                > 0.0
                and counterfactual_selector_metrics[
                    (
                        f"counterfactual_selector_{split}_"
                        "audit_available_rate"
                    )
                ]["full-internal-rl"]
                >= 1.0
                and counterfactual_selector_metrics[
                    (
                        f"counterfactual_selector_{split}_"
                        "mean_audit_selected_raw_delta"
                    )
                ]["full-internal-rl"]
                > 0.0
                for split in (
                    "train",
                    "eval",
                    "heldout",
                    "validation",
                )
            )
        ),
    }
    oracle_permutation_null_by_split = _oracle_permutation_null_by_split(
        counterfactual_outcomes,
        split_names=(
            "train",
            "eval",
            "heldout",
            "validation",
            "confirmation",
        ),
    )
    selector_permutation_null_by_split = (
        _selector_permutation_null_by_split(
            action_selection or [],
            counterfactual_outcomes,
            split_names=(
                "train",
                "eval",
                "heldout",
                "validation",
                "confirmation",
            ),
        )
    )
    signal_gates = {
        "train_selector_grid_present": (
            selector_permutation_null_by_split["train"]["selection_count"]
            > 0.0
        ),
        "validation_selector_grid_present": (
            selector_permutation_null_by_split[
                "validation"
            ]["selection_count"]
            > 0.0
        ),
        "confirmation_selector_grid_present": (
            selector_permutation_null_by_split[
                "confirmation"
            ]["selection_count"]
            > 0.0
        ),
        "train_selector_lineage_complete": (
            selector_permutation_null_by_split["train"][
                "input_selection_count"
            ]
            == selector_permutation_null_by_split["train"][
                "selection_count"
            ]
            and selector_permutation_null_by_split["train"][
                "missing_counterfactual_grid_count"
            ]
            == 0.0
            and selector_permutation_null_by_split["train"][
                "missing_selected_candidate_count"
            ]
            == 0.0
            and selector_permutation_null_by_split["train"][
                "selected_audit_lineage_mismatch_count"
            ]
            == 0.0
        ),
        "validation_selector_lineage_complete": (
            selector_permutation_null_by_split["validation"][
                "input_selection_count"
            ]
            == selector_permutation_null_by_split["validation"][
                "selection_count"
            ]
            and selector_permutation_null_by_split["validation"][
                "missing_counterfactual_grid_count"
            ]
            == 0.0
            and selector_permutation_null_by_split["validation"][
                "missing_selected_candidate_count"
            ]
            == 0.0
            and selector_permutation_null_by_split["validation"][
                "selected_audit_lineage_mismatch_count"
            ]
            == 0.0
        ),
        "confirmation_selector_lineage_complete": (
            selector_permutation_null_by_split["confirmation"][
                "input_selection_count"
            ]
            == selector_permutation_null_by_split["confirmation"][
                "selection_count"
            ]
            and selector_permutation_null_by_split["confirmation"][
                "missing_counterfactual_grid_count"
            ]
            == 0.0
            and selector_permutation_null_by_split["confirmation"][
                "missing_selected_candidate_count"
            ]
            == 0.0
            and selector_permutation_null_by_split["confirmation"][
                "selected_audit_lineage_mismatch_count"
            ]
            == 0.0
        ),
        "train_selector_exceeds_permutation_null": (
            selector_permutation_null_by_split[
                "train"
            ]["selected_excess_over_null_mean"]
            >= ETA_GATE2_MIN_MEASURABLE_EFFECT
        ),
        "validation_selector_exceeds_permutation_null": (
            selector_permutation_null_by_split[
                "validation"
            ]["selected_excess_over_null_mean"]
            >= ETA_GATE2_MIN_MEASURABLE_EFFECT
        ),
        "confirmation_selector_exceeds_permutation_null": (
            selector_permutation_null_by_split[
                "confirmation"
            ]["selected_excess_over_null_mean"]
            >= ETA_GATE2_MIN_MEASURABLE_EFFECT
        ),
    }
    reachable_solution_evidence = all(signal_gates.values())
    (
        shadow_closed_loop_by_split,
        shadow_artifact_provenance_valid,
        shadow_artifact_lineage_valid,
    ) = _shadow_closed_loop_diagnostics(
        rows=shadow_closed_loop or [],
        selector_artifacts=selector_artifacts or [],
        split_names=(
            "train",
            "eval",
            "heldout",
            "validation",
            "confirmation",
        ),
    )
    formal_shadow_splits = (
        "train",
        "validation",
        "confirmation",
    )
    shadow_single_seed_stoploss_passed = (
        shadow_artifact_lineage_valid
        and all(
            shadow_closed_loop_by_split[split]["step_count"] > 0.0
            and shadow_closed_loop_by_split[split][
                "complete_step_count"
            ]
            == shadow_closed_loop_by_split[split]["step_count"]
            and shadow_closed_loop_by_split[split]["seed_count"] == 1.0
            and shadow_closed_loop_by_split[split][
                "selector_zero_positive_seed_count"
            ]
            == 1.0
            and shadow_closed_loop_by_split[split][
                "selector_permutation_positive_seed_count"
            ]
            == 1.0
            and shadow_closed_loop_by_split[split][
                "selected_step_positive_seed_count"
            ]
            == 1.0
            and shadow_closed_loop_by_split[split][
                "selector_minus_zero_mean"
            ]
            >= ETA_GATE2_MIN_MEASURABLE_EFFECT
            and shadow_closed_loop_by_split[split][
                "selector_minus_permutation_mean"
            ]
            >= ETA_GATE2_MIN_MEASURABLE_EFFECT
            and shadow_closed_loop_by_split[split][
                "selected_step_realized_delta_mean"
            ]
            >= ETA_GATE2_MIN_MEASURABLE_EFFECT
            for split in formal_shadow_splits
        )
    )
    shadow_gates = {
        "selector_artifact_provenance_complete": (
            shadow_artifact_provenance_valid
        ),
        **{
            f"{split}_closed_loop_records_complete": (
                shadow_closed_loop_by_split[split]["step_count"] > 0.0
                and shadow_closed_loop_by_split[split][
                    "complete_step_count"
                ]
                == shadow_closed_loop_by_split[split]["step_count"]
            )
            for split in formal_shadow_splits
        },
        **{
            f"{split}_selector_beats_zero": (
                shadow_closed_loop_by_split[split][
                    "selector_minus_zero_mean"
                ]
                >= ETA_GATE2_MIN_MEASURABLE_EFFECT
            )
            for split in formal_shadow_splits
        },
        **{
            f"{split}_selector_beats_zero_all_seeds": (
                shadow_closed_loop_by_split[split]["seed_count"] == 3.0
                and shadow_closed_loop_by_split[split][
                    "selector_zero_positive_seed_count"
                ]
                == 3.0
            )
            for split in formal_shadow_splits
        },
        **{
            f"{split}_selector_beats_permutation_null": (
                shadow_closed_loop_by_split[split][
                    "selector_minus_permutation_mean"
                ]
                >= ETA_GATE2_MIN_MEASURABLE_EFFECT
            )
            for split in formal_shadow_splits
        },
        **{
            f"{split}_selector_beats_permutation_all_seeds": (
                shadow_closed_loop_by_split[split]["seed_count"] == 3.0
                and shadow_closed_loop_by_split[split][
                    "selector_permutation_positive_seed_count"
                ]
                == 3.0
            )
            for split in formal_shadow_splits
        },
        **{
            f"{split}_selected_step_distribution_positive": (
                shadow_closed_loop_by_split[split][
                    "selected_step_realized_delta_mean"
                ]
                >= ETA_GATE2_MIN_MEASURABLE_EFFECT
            )
            for split in formal_shadow_splits
        },
        **{
            f"{split}_selected_step_positive_all_seeds": (
                shadow_closed_loop_by_split[split]["seed_count"] == 3.0
                and shadow_closed_loop_by_split[split][
                    "selected_step_positive_seed_count"
                ]
                == 3.0
            )
            for split in formal_shadow_splits
        },
    }
    shadow_observation_passed = all(shadow_gates.values())
    return {
        "schema_version": schema_version,
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
        "profile_environment_outcome_target_active": (
            environment_outcome_target_active
        ),
        "profile_environment_application_count": (
            environment_application_count
        ),
        "profile_environment_pe_credit_transition_count": (
            environment_pe_credit_transition_count
        ),
        "profile_self_nll_target_active": self_nll_target_active,
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
        "oracle_permutation_null_by_split": (
            oracle_permutation_null_by_split
        ),
        "selector_permutation_null_by_split": (
            selector_permutation_null_by_split
        ),
        "mechanism_gates": mechanism_gates,
        "causal_gates": causal_gates,
        "selector_gates": selector_gates,
        "signal_gates": signal_gates,
        "reachable_solution_evidence": reachable_solution_evidence,
        "shadow_closed_loop_by_split": shadow_closed_loop_by_split,
        "shadow_gates": shadow_gates,
        "shadow_observation_passed": shadow_observation_passed,
        "shadow_artifact_lineage_valid": (
            shadow_artifact_lineage_valid
        ),
        "shadow_single_seed_stoploss_passed": (
            shadow_single_seed_stoploss_passed
        ),
        "causal_promotion_inherited": inherited_causal_promotion,
        "selector_injection_allowed": all(selector_gates.values()),
        "all_mechanism_gates_passed": all(mechanism_gates.values()),
        "all_causal_gates_passed": all(causal_gates.values()),
        "backend_names": sorted(backend_names),
    }


def _promotion_verdict(ablation: dict[str, Any]) -> dict[str, Any]:
    mechanism_passed = bool(ablation["all_mechanism_gates_passed"])
    reachable_solution_evidence = bool(
        ablation["reachable_solution_evidence"]
    )
    causal_packet_revalidated = (
        mechanism_passed
        and reachable_solution_evidence
        and bool(ablation["all_causal_gates_passed"])
    )
    causal_passed = (
        causal_packet_revalidated
        or bool(ablation.get("causal_promotion_inherited", False))
    )
    if causal_passed:
        status = "causal-supported"
    elif mechanism_passed:
        status = "mechanism-supported"
    else:
        status = "wiring-ready"
    return {
        "schema_version": ablation["schema_version"],
        "status": status,
        "gate_scope": (
            "Gate 2 recent-k2 closed-loop SHADOW admission"
            if ablation["schema_version"]
            == ETA_GATE2_RECENT_K2_FORMAL_SCHEMA_VERSION
            else "Gate 2 residual intervention causal packet"
        ),
        "thesis_status": "not-evaluated",
        "promotion_allowed": causal_passed,
        "shadow_admission_allowed": ablation[
            "shadow_observation_passed"
        ],
        "counterfactual_action_selector_live_injection": "disabled",
        "causal_packet_revalidated": causal_packet_revalidated,
        "mechanism_gates": ablation["mechanism_gates"],
        "causal_gates": ablation["causal_gates"],
        "selector_gates": ablation["selector_gates"],
        "signal_gates": ablation["signal_gates"],
        "reachable_solution_evidence": reachable_solution_evidence,
        "shadow_gates": ablation["shadow_gates"],
        "shadow_observation_passed": ablation[
            "shadow_observation_passed"
        ],
        "shadow_single_seed_stoploss_passed": ablation[
            "shadow_single_seed_stoploss_passed"
        ],
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
                **ablation["signal_gates"],
                **ablation["shadow_gates"],
            }.items()
            if not passed
        ],
        "note": (
            "This packet cannot emit thesis-retained. Gate 2 longitudinal "
            "coverage, a fresh locked confirmation split, semantic no-label "
            "evidence, and Gates 1/3-10 remain separate prerequisites. "
            "When reachable_solution_evidence is false the train-fit "
            "selector does not survive the independent-audit permutation "
            "null on frozen validation/confirmation, so the verdict records "
            "no-reachable-solution-evidence and causal promotion is refused "
            "regardless of other gates."
            " v36 shadow_observation_passed is an independent admission "
            "gate for later runtime SHADOW wiring and never revokes the "
            "inherited v35 causal promotion."
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
        counterfactual_outcomes,
        selector_artifacts,
        shadow_closed_loop,
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
    schema_version = case_groups.get(
        "evidence_schema_version",
        (ETA_GATE2_SCHEMA_VERSION,),
    )[0]
    if schema_version not in {
        ETA_GATE2_SCHEMA_VERSION,
        ETA_GATE2_RECENT_K2_FORMAL_SCHEMA_VERSION,
    }:
        raise ValueError(
            f"Unsupported Gate 2 evidence schema {schema_version!r}"
        )
    independent_training_route_ids = case_groups.get(
        "independent_training_route_ids",
        (),
    )
    validation_route_ids = case_groups.get(
        "validation_route_ids",
        (),
    )
    confirmation_route_ids = case_groups.get(
        "confirmation_route_ids",
        (),
    )
    superseded_validation_route_ids = case_groups.get(
        "superseded_validation_route_ids",
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
            "Gate 2 v31 requires frozen validation routes before execution"
        )
    if any(
        route_id not in route_ids for route_id in validation_route_ids
    ):
        raise ValueError(
            "Gate 2 validation_route_ids must be included in route_ids"
        )
    if confirmation_split_locked and (
        not confirmation_route_ids
        or any(route_id not in route_ids for route_id in confirmation_route_ids)
    ):
        raise ValueError(
            "Gate 2 locked confirmation_route_ids must be included in route_ids"
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
        "schema_version": schema_version,
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
            "confirmation_route_ids": list(confirmation_route_ids),
            "superseded_validation_route_ids": list(
                superseded_validation_route_ids
            ),
            "validation_frozen_before_run": validation_frozen,
            "training_route_count": training_route_count,
            "development_routes_unchanged": (
                development_routes_unchanged
            ),
            "split_owner": descriptor.get(
                "case_corpus",
                ETA_GATE2_SHADOW_FRESH_CASE_CORPUS,
            ),
            "development_heldout_status": "reused-during-gate-development",
            "causal_confirmation_split": list(confirmation_route_ids),
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
            "shadow_closed_loop_arm": case_groups[
                "shadow_closed_loop_arm"
            ][0],
            "shadow_observation_gate": case_groups[
                "shadow_observation_gate"
            ][0],
            **(
                {
                    "committed_control_window": int(
                        case_groups[
                            "shadow_committed_control_window"
                        ][0]
                    )
                }
                if "shadow_committed_control_window" in case_groups
                else {}
            ),
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
                "subsequent-realized-segment-transfer-readout-and-"
                "train-route-cv-model-selection-only"
            ),
            "outcome_chain": case_groups.get(
                "counterfactual_outcome_chain",
                ("legacy-continuation-target",),
            )[0],
            "primary_target": case_groups.get(
                "counterfactual_primary_target",
                ("legacy-continuation-target",),
            )[0],
            "audit_surface": case_groups.get(
                "counterfactual_audit_surface",
                ("legacy-continuation-audit",),
            )[0],
        },
        "residual_capture": {
            "activation_width": residual_activation_width,
            "compression_mode": "exact-up-to-configured-width",
        },
        "metric_version": schema_version,
        "judge_or_human_protocol": (
            "typed automatic metrics; no LLM judge or human rating"
        ),
        "suite_id": report.manifest.suite_id,
        "claim_scope": case_groups.get(
            "formal_claim_scope",
            ("causal-packet",),
        )[0],
        "causal_packet_source": case_groups.get(
            "causal_packet_source",
            ("current-packet",),
        )[0],
        "counterfactual_action_selector_live_injection": (
            case_groups[
                "counterfactual_action_selector_live_injection"
            ][0]
        ),
        "profiles": [
            profile.profile_label for profile in report.manifest.profiles
        ],
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
        action_selection=action_selection,
        counterfactual_outcomes=counterfactual_outcomes,
        selector_artifacts=selector_artifacts,
        shadow_closed_loop=shadow_closed_loop,
        confirmation_split_locked=confirmation_split_locked,
        inherited_causal_promotion=(
            descriptor.get("primary_backend")
            == "transformers-open-weight"
        ),
        schema_version=schema_version,
    )
    verdict = _promotion_verdict(ablation)
    rollback = {
        "schema_version": schema_version,
        "rollback_target": "residual_control_mode=identity",
        "configuration_default_is_identity": True,
        "owner_state_mutated_by_mode_switch": False,
        "shadow_closed_loop_side_effect_free": bool(shadow_closed_loop)
        and all(
            row.get("side_effect_free") is True
            for row in shadow_closed_loop
        ),
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
            f"- signal gates：`{ablation['signal_gates']}`",
            f"- shadow gates：`{ablation['shadow_gates']}`",
            (
                "- shadow observation passed："
                f"`{ablation['shadow_observation_passed']}`"
            ),
            (
                "- single-seed stop-loss passed："
                f"`{ablation['shadow_single_seed_stoploss_passed']}`"
            ),
            (
                "- reachable solution evidence（selector 过置换零假设）："
                f"`{ablation['reachable_solution_evidence']}`"
            ),
            (
                "- oracle permutation-null diagnostics："
                f"`{ablation['oracle_permutation_null_by_split']}`"
            ),
            (
                "- selector permutation-null diagnostics："
                f"`{ablation['selector_permutation_null_by_split']}`"
            ),
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
        _write_jsonl(
            target / "counterfactual_outcomes.jsonl",
            counterfactual_outcomes,
        ),
        _write_json(
            target / "selector_artifact.json",
            {
                "schema_version": "eta-gate2-selector-artifacts.v1",
                "artifacts": selector_artifacts,
            },
        ),
        _write_jsonl(
            target / "shadow_closed_loop.jsonl",
            shadow_closed_loop,
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
