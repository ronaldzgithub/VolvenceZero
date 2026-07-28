from __future__ import annotations

from dataclasses import replace
import json

from volvence_zero.agent.eta_gate2_residual_evidence import (
    ETA_GATE2_REQUIRED_FILES,
    ETA_GATE2_REQUIRED_MANIFEST_KEYS,
    _build_ablation_results,
    _promotion_verdict,
    build_eta_gate2_residual_manifest,
    export_eta_gate2_residual_bundle,
)
from volvence_zero.agent.eta_proof_benchmark import (
    ETA_CONTINUATION_PE_TRAINING_SIGNAL,
    ETA_COUNTERFACTUAL_TARGET_ENVIRONMENT_OUTCOME,
    ETA_COUNTERFACTUAL_TARGET_PREFIX_EXPECTED,
    ETA_GATE2_EXPECTED_VALUE_CASE_CORPUS,
    ETAOpenWeightRuntimeConfig,
    _counterfactual_selector_metric_rows,
    default_eta_proof_cases,
    eta_gate2_expanded_cases,
    eta_gate2_expected_value_cases,
    eta_gate2_expected_value_validation_routes,
    eta_gate2_independent_training_routes,
    run_eta_internal_rl_paper_suite,
    run_eta_internal_rl_proof_benchmark,
)
from volvence_zero.internal_rl import CounterfactualActionSelection
from volvence_zero.substrate import build_builtin_transformers_runtime


def test_eta_residual_controls_record_actual_matched_interventions() -> None:
    cases = default_eta_proof_cases()
    train_case = next(case for case in cases if case.split == "train")
    heldout_case = next(
        case for case in cases if case.split == "heldout"
    )
    report = run_eta_internal_rl_proof_benchmark(
        cases=(train_case, heldout_case),
        profile_labels=(
            "full-internal-rl",
            "full-zero-control",
            "full-shuffled-control",
            "full-reversed-control",
        ),
        backend_label="trace",
        train_epochs=1,
    )

    records_by_profile = {
        profile.profile_label: tuple(
            record
            for episode in profile.episode_reports
            for record in episode.intervention_records
        )
        for profile in report.profile_reports
    }
    assert all(records_by_profile.values())
    profiles_by_label = {
        profile.profile_label: profile
        for profile in report.profile_reports
    }
    baseline_profile = profiles_by_label["full-internal-rl"]
    baseline_records = {
        (episode.case_id, record.step_index): record
        for episode in baseline_profile.episode_reports
        for record in episode.intervention_records
    }
    for profile_label in (
        "full-zero-control",
        "full-shuffled-control",
        "full-reversed-control",
    ):
        profile = profiles_by_label[profile_label]
        metrics = dict(profile.metric_means)
        assert profile.training_update_count == 0
        assert metrics["shared_policy_checkpoint_used"] == 1.0
        assert (
            metrics["shared_policy_fingerprint_match_at_eval_start"]
            == 1.0
        )
        for episode in profile.episode_reports:
            for record in episode.intervention_records:
                baseline = baseline_records[
                    (episode.case_id, record.step_index)
                ]
                assert (
                    record.control_before_ablation
                    == baseline.control_before_ablation
                )

    for record in records_by_profile["full-internal-rl"]:
        assert record.residual_control_mode == "identity"
        assert record.applied_control == record.control_before_ablation

    for record in records_by_profile["full-zero-control"]:
        assert record.residual_control_mode == "zero"
        assert record.applied_control == tuple(
            0.0 for _ in record.control_before_ablation
        )

    for record in records_by_profile["full-shuffled-control"]:
        assert record.residual_control_mode == "shuffled"
        assert sorted(record.applied_control) == sorted(
            record.control_before_ablation
        )
        assert record.applied_control != record.control_before_ablation

    for record in records_by_profile["full-reversed-control"]:
        assert record.residual_control_mode == "reversed"
        assert record.applied_control == tuple(
            reversed(record.control_before_ablation)
        )


def test_eta_proof_source_text_does_not_expose_split_labels() -> None:
    for case in eta_gate2_expected_value_cases():
        source_tokens = set(case.source_text.lower().split())
        assert source_tokens.isdisjoint(
            {"train", "eval", "heldout", "validation"}
        )


def test_eta_gate2_freezes_expected_value_validation_corpus() -> None:
    default_cases = default_eta_proof_cases()
    expanded_cases = eta_gate2_expanded_cases()
    expected_value_cases = eta_gate2_expected_value_cases()

    assert sum(case.split == "train" for case in default_cases) == 3
    assert sum(case.split == "train" for case in expanded_cases) == 16
    assert tuple(
        case for case in expanded_cases if case.split != "train"
    ) == tuple(case for case in default_cases if case.split != "train")
    assert tuple(
        case for case in expected_value_cases
        if case.split in {"eval", "heldout"}
    ) == tuple(case for case in default_cases if case.split != "train")
    assert sum(
        case.split == "validation" for case in expected_value_cases
    ) == 4

    manifest = build_eta_gate2_residual_manifest(suite_tier="ci-smoke")
    case_groups = dict(manifest.case_groups)
    assert case_groups["case_corpus"] == (
        ETA_GATE2_EXPECTED_VALUE_CASE_CORPUS,
    )
    assert case_groups["route_ids"] == tuple(
        case.case_id for case in expected_value_cases
    )
    independent_route_ids = tuple(
        route.case_id
        for route in eta_gate2_independent_training_routes()
    )
    assert case_groups["independent_training_route_ids"] == (
        independent_route_ids
    )
    assert case_groups["training_route_count"] == ("16",)
    assert case_groups["development_routes_unchanged"] == ("true",)
    assert case_groups["validation_frozen_before_run"] == ("true",)
    assert case_groups["validation_route_ids"] == tuple(
        route.case_id
        for route in eta_gate2_expected_value_validation_routes()
    )
    assert case_groups[
        "counterfactual_action_selector_diagnostic"
    ] == ("true",)
    assert case_groups[
        "counterfactual_action_selector_live_injection"
    ] == ("disabled",)
    assert case_groups["real_residual_activation_width"] == ("896",)
    assert case_groups["counterfactual_target_mode"] == (
        ETA_COUNTERFACTUAL_TARGET_ENVIRONMENT_OUTCOME,
    )
    assert case_groups["counterfactual_target_sample_count"] == ("1",)
    assert case_groups["counterfactual_audit_sample_count"] == ("1",)
    assert case_groups["counterfactual_outcome_chain"] == (
        "z-candidate->residual-forward->realized-continuation-nll"
        "->prediction-error->action-credit",
    )
    assert case_groups["counterfactual_audit_surface"] == (
        "subsequent-realized-segment-teacher-forced-"
        "nll-improvement-vs-zero-control",
    )
    assert case_groups["counterfactual_primary_target"] == (
        "realized-next-segment-teacher-forced-"
        "nll-improvement-vs-zero-control",
    )
    assert not any(
        case.split == "confirmation" for case in expected_value_cases
    )


def test_eta_gate2_independent_training_text_avoids_development_content_words() -> None:
    stop_words = {
        "a",
        "an",
        "and",
        "before",
        "from",
        "in",
        "the",
        "to",
        "was",
        "while",
        "with",
    }

    def content_words(source_text: str) -> set[str]:
        return {
            token.strip(".,:;!?").lower()
            for token in source_text.split()
            if token.strip(".,:;!?").lower() not in stop_words
        }

    development_words = set().union(
        *(
            content_words(case.source_text)
            for case in default_eta_proof_cases()
            if case.split != "train"
        )
    )
    independent_words = set().union(
        *(
            content_words(route.source_text)
            for route in eta_gate2_independent_training_routes()
        )
    )

    assert development_words
    assert independent_words
    assert independent_words.isdisjoint(development_words)


def test_eta_gate2_v31_validation_text_is_lexically_fresh() -> None:
    stop_words = {
        "a",
        "across",
        "after",
        "an",
        "and",
        "as",
        "before",
        "from",
        "in",
        "the",
        "to",
        "was",
        "while",
        "with",
    }

    def content_words(source_text: str) -> set[str]:
        return {
            token.strip(".,:;!?").lower()
            for token in source_text.split()
            if token.strip(".,:;!?").lower() not in stop_words
        }

    existing_words = set().union(
        *(
            content_words(case.source_text)
            for case in eta_gate2_expanded_cases()
        )
    )
    validation_words = set().union(
        *(
            content_words(route.source_text)
            for route in eta_gate2_expected_value_validation_routes()
        )
    )

    assert validation_words
    assert validation_words.isdisjoint(existing_words)


def test_eta_gate2_bundle_exports_frozen_v31_file_contract(
    tmp_path,
) -> None:
    manifest = build_eta_gate2_residual_manifest(suite_tier="ci-smoke")
    manifest = replace(
        manifest,
        suite_kind="eta-gate2-trace-contract-smoke",
        case_groups=tuple(
            ("backend_labels", ("trace",))
            if name == "backend_labels"
            else ("training_signal", ("proof-reward",))
            if name == "training_signal"
            else (name, values)
            for name, values in manifest.case_groups
        ),
    )
    report = run_eta_internal_rl_paper_suite(
        manifest=manifest,
        output_dir=tmp_path,
    )

    paths = export_eta_gate2_residual_bundle(
        report,
        output_dir=tmp_path,
    )

    assert {path.name for path in paths} == set(
        ETA_GATE2_REQUIRED_FILES
    )
    manifest_payload = json.loads(
        (tmp_path / "manifest.yaml").read_text(encoding="utf-8")
    )
    assert set(ETA_GATE2_REQUIRED_MANIFEST_KEYS).issubset(
        manifest_payload
    )
    assert (
        manifest_payload["scenario_split"][
            "development_heldout_status"
        ]
        == "reused-during-gate-development"
    )
    assert manifest_payload["scenario_split"][
        "independent_training_route_ids"
    ] == [
        route.case_id
        for route in eta_gate2_independent_training_routes()
    ]
    assert manifest_payload["scenario_split"][
        "training_route_count"
    ] == 16
    assert manifest_payload["scenario_split"][
        "development_routes_unchanged"
    ]
    assert manifest_payload["scenario_split"][
        "validation_frozen_before_run"
    ]
    assert manifest_payload["scenario_split"][
        "validation_route_ids"
    ] == [
        route.case_id
        for route in eta_gate2_expected_value_validation_routes()
    ]
    assert manifest_payload["counterfactual_action_selector"] == {
        "diagnostic_active": True,
        "input": "full-layer-coordinate-mean-latest-trend",
        "encoder": "train-only-standardized-linear-kernel",
        "head": "train-only-dual-ridge-ladder-0.1-1-10",
        "model_selection": (
            "train-route-cv-audit-delta-then-audit-regret"
        ),
        "cross_validation": "route-grouped-4fold",
        "live_injection": "disabled",
    }
    assert manifest_payload["counterfactual_target"] == {
        "mode": ETA_COUNTERFACTUAL_TARGET_ENVIRONMENT_OUTCOME,
        "target_sample_count": 1,
        "audit_sample_count": 1,
        "sampling_temperature": 0.0,
        "sampling_max_new_tokens": 0,
        "sampling_seed_protocol": (
            "not-applicable-deterministic-environment-forward"
        ),
        "audit_role": (
            "subsequent-realized-segment-transfer-readout-and-"
            "train-route-cv-model-selection-only"
        ),
        "outcome_chain": (
            "z-candidate->residual-forward->realized-continuation-nll"
            "->prediction-error->action-credit"
        ),
        "primary_target": (
            "realized-next-segment-teacher-forced-"
            "nll-improvement-vs-zero-control"
        ),
        "audit_surface": (
            "subsequent-realized-segment-teacher-forced-"
            "nll-improvement-vs-zero-control"
        ),
    }
    assert manifest_payload["residual_capture"] == {
        "activation_width": 896,
        "compression_mode": "exact-up-to-configured-width",
    }
    assert (
        tmp_path / "action_selection.jsonl"
    ).read_text(encoding="utf-8") == ""
    assert (
        manifest_payload["scenario_split"]["causal_confirmation_split"]
        is None
    )
    assert not manifest_payload["scenario_split"][
        "confirmation_split_locked"
    ]
    ablation = json.loads(
        (tmp_path / "ablation_results.json").read_text(encoding="utf-8")
    )
    assert all(
        ablation["mechanism_gates"][gate]
        for gate in (
            "identity_exact",
            "zero_exact",
            "shuffle_permutation_nonidentity",
            "reverse_exact",
            "shared_policy_checkpoint_matched",
        )
    )
    verdict = json.loads(
        (tmp_path / "promotion_verdict.json").read_text(encoding="utf-8")
    )
    assert verdict["status"] == "wiring-ready"
    assert verdict["thesis_status"] == "not-evaluated"


def test_eta_gate2_bundle_rejects_residual_width_provenance_mismatch(
    tmp_path,
) -> None:
    manifest = build_eta_gate2_residual_manifest(suite_tier="ci-smoke")
    manifest = replace(
        manifest,
        suite_kind="eta-gate2-width-mismatch-smoke",
        case_groups=tuple(
            ("backend_labels", ("trace",))
            if name == "backend_labels"
            else ("training_signal", ("proof-reward",))
            if name == "training_signal"
            else (name, values)
            for name, values in manifest.case_groups
        ),
    )
    report = run_eta_internal_rl_paper_suite(
        manifest=manifest,
        output_dir=tmp_path,
    )
    report = replace(
        report,
        provenance=replace(
            report.provenance,
            runtime_descriptor=tuple(
                (
                    key,
                    "transformers-open-weight"
                    if key == "primary_backend"
                    else "8"
                    if key == "open_weight_activation_width"
                    else value,
                )
                for key, value in (
                    *report.provenance.runtime_descriptor,
                    ("open_weight_activation_width", "8"),
                )
            ),
        ),
    )

    try:
        export_eta_gate2_residual_bundle(
            report,
            output_dir=tmp_path,
        )
    except ValueError as exc:
        assert "residual activation width mismatch" in str(exc)
    else:
        raise AssertionError("expected residual width mismatch to fail")


def test_eta_gate2_causal_verdict_requires_eval_and_fresh_confirmation() -> None:
    profiles = (
        "full-internal-rl",
        "full-zero-control",
        "full-shuffled-control",
        "full-reversed-control",
    )
    outcomes = []
    for split in ("eval", "heldout", "confirmation"):
        for profile in profiles:
            outcomes.append(
                {
                    "profile_label": profile,
                    "split": split,
                    "continuation_mean_nll": (
                        0.95 if profile == "full-internal-rl" else 1.0
                    ),
                    "downstream_effect_magnitude": 0.0,
                }
            )

    ablation = _build_ablation_results(
        predictions=[],
        outcomes=outcomes,
        metric_samples={},
        counterfactual_outcomes=[],
        confirmation_split_locked=True,
    )

    assert ablation["identity_eval_nll_delta_vs_best_control"] > 0.04
    assert (
        ablation["identity_confirmation_nll_delta_vs_best_control"]
        > 0.04
    )
    assert all(ablation["causal_gates"].values())

    without_confirmation = _build_ablation_results(
        predictions=[],
        outcomes=[
            row for row in outcomes if row["split"] != "confirmation"
        ],
        metric_samples={},
        counterfactual_outcomes=[],
        confirmation_split_locked=True,
    )
    assert not without_confirmation["causal_gates"][
        "fresh_confirmation_scores_available_all_arms"
    ]
    assert not without_confirmation["causal_gates"][
        "identity_confirmation_nll_beats_controls"
    ]

    unlocked_confirmation = _build_ablation_results(
        predictions=[],
        outcomes=outcomes,
        metric_samples={},
        counterfactual_outcomes=[],
    )
    assert not unlocked_confirmation["causal_gates"][
        "fresh_confirmation_split_locked"
    ]


def _counterfactual_grid_rows(
    *,
    split: str,
    prefix_count: int,
    audit_tracks_target: bool,
) -> list[dict]:
    target_credits = (0.0, 0.01, 0.002, -0.004)
    if audit_tracks_target:
        # Same candidate wins on the independent audit cohort: real signal.
        audit_credits = (0.0, 0.008, 0.001, -0.003)
    else:
        # Target argmax (index 1) lands below the audit candidate mean:
        # the target oracle is max-of-noise selection bias.
        audit_credits = (0.0, -0.009, 0.006, 0.006)
    rows: list[dict] = []
    for prefix in range(prefix_count):
        for candidate_index in range(len(target_credits)):
            rows.append(
                {
                    "profile_label": "full-internal-rl",
                    "split": split,
                    "observation_id": (
                        f"route-a:{split}-0:prefix-{prefix}"
                        f":candidate-{candidate_index}"
                    ),
                    "candidate_index": candidate_index,
                    "action_credit": target_credits[candidate_index],
                    "audit_action_credit": audit_credits[
                        candidate_index
                    ],
                }
            )
    return rows


def test_eta_gate2_signal_gates_reject_max_of_noise_oracle() -> None:
    rows = [
        *_counterfactual_grid_rows(
            split="train",
            prefix_count=3,
            audit_tracks_target=False,
        ),
        *_counterfactual_grid_rows(
            split="validation",
            prefix_count=2,
            audit_tracks_target=False,
        ),
    ]
    ablation = _build_ablation_results(
        predictions=[],
        outcomes=[],
        metric_samples={},
        counterfactual_outcomes=rows,
    )

    train_diagnostics = ablation["oracle_permutation_null_by_split"][
        "train"
    ]
    # The trap this gate exists for: target-cohort oracle looks positive...
    assert train_diagnostics["observed_oracle_target_mean"] > 0.0
    # ...but the argmax candidate does not survive independent audit.
    assert (
        train_diagnostics["transfer_excess_over_null_mean"] < 0.0
    )
    assert not ablation["signal_gates"][
        "train_oracle_transfer_exceeds_permutation_null"
    ]
    assert not ablation["signal_gates"][
        "validation_oracle_transfer_exceeds_permutation_null"
    ]
    assert ablation["reachable_solution_evidence"] is False

    verdict = _promotion_verdict(ablation)
    assert verdict["promotion_allowed"] is False
    assert verdict["reachable_solution_evidence"] is False
    assert (
        "train_oracle_transfer_exceeds_permutation_null"
        in verdict["kill_conditions"]
    )


def test_eta_gate2_reachable_solution_evidence_gates_causal_promotion() -> None:
    def _ablation_with_grid(*, audit_tracks_target: bool) -> dict:
        rows = [
            *_counterfactual_grid_rows(
                split="train",
                prefix_count=3,
                audit_tracks_target=audit_tracks_target,
            ),
            *_counterfactual_grid_rows(
                split="validation",
                prefix_count=2,
                audit_tracks_target=audit_tracks_target,
            ),
        ]
        return _build_ablation_results(
            predictions=[],
            outcomes=[],
            metric_samples={},
            counterfactual_outcomes=rows,
        )

    transferable = _ablation_with_grid(audit_tracks_target=True)
    assert all(transferable["signal_gates"].values())
    assert transferable["reachable_solution_evidence"] is True

    noise = _ablation_with_grid(audit_tracks_target=False)
    assert noise["reachable_solution_evidence"] is False

    # With mechanism + causal gates all green, promotion must still hinge on
    # reachable-solution evidence.
    other_gates_green = {
        "all_mechanism_gates_passed": True,
        "all_causal_gates_passed": True,
        "mechanism_gates": {"probe": True},
        "causal_gates": {"probe": True},
    }
    promoted = _promotion_verdict({**transferable, **other_gates_green})
    assert promoted["status"] == "causal-supported"
    assert promoted["promotion_allowed"] is True

    refused = _promotion_verdict({**noise, **other_gates_green})
    assert refused["status"] == "mechanism-supported"
    assert refused["promotion_allowed"] is False
    assert any(
        gate in refused["kill_conditions"]
        for gate in noise["signal_gates"]
    )


def test_eta_real_residual_records_observed_continuation_nll() -> None:
    cases = default_eta_proof_cases()
    train_cases = tuple(
        case for case in cases if case.split == "train"
    )[:2]
    heldout_case = next(
        case for case in cases if case.split == "heldout"
    )
    runtime = build_builtin_transformers_runtime(
        model_id="eta-continuation-score-test",
    )

    report = run_eta_internal_rl_proof_benchmark(
        cases=(*train_cases, heldout_case),
        profile_labels=(
            "full-internal-rl",
            "full-zero-control",
            "full-shuffled-control",
            "full-reversed-control",
        ),
        backend_label="transformers-open-weight",
        train_epochs=1,
        open_weight_runtime=runtime,
        open_weight_config=ETAOpenWeightRuntimeConfig(
            require_real_backend=False,
            max_prefix_steps=4,
        ),
        training_signal=ETA_CONTINUATION_PE_TRAINING_SIGNAL,
        latent_unit_clamp=True,
        real_residual_ssl_bootstrap=True,
        causal_action_head_active=True,
        causal_action_head_state_dim=12,
        continuation_counterfactual_grid=True,
        counterfactual_action_selector_diagnostic=True,
        counterfactual_target_mode=(
            ETA_COUNTERFACTUAL_TARGET_PREFIX_EXPECTED
        ),
        counterfactual_target_sample_count=1,
        counterfactual_audit_sample_count=1,
        counterfactual_sampling_max_new_tokens=1,
    )

    for profile in report.profile_reports:
        records = tuple(
            record
            for episode in profile.episode_reports
            for record in episode.intervention_records
        )
        scored = tuple(
            record
            for record in records
            if record.continuation_mean_nll is not None
        )
        assert scored
        assert all(record.continuation_token_count > 0 for record in scored)
        assert all(record.continuation_mean_nll > 0.0 for record in scored)
        assert all(
            0.0
            < record.continuation_geometric_mean_probability
            < 1.0
            for record in scored
            if record.continuation_geometric_mean_probability is not None
        )
    profiles = {
        profile.profile_label: profile
        for profile in report.profile_reports
    }
    baseline_metrics = dict(
        profiles["full-internal-rl"].metric_means
    )
    assert (
        baseline_metrics[
            "continuation_pe_training_transition_count"
        ]
        > 0.0
    )
    assert baseline_metrics["bootstrap_init_used"] == 1.0
    assert baseline_metrics["causal_action_head_active"] == 1.0
    assert baseline_metrics["causal_action_head_state_dim"] == 12.0
    assert baseline_metrics["causal_action_head_update_step"] > 0.0
    assert baseline_metrics["continuation_pe_structure_frozen"] == 1.0
    assert baseline_metrics["continuation_pe_policy_changed"] == 1.0
    assert (
        baseline_metrics["continuation_counterfactual_grid_used"]
        == 1.0
    )
    assert (
        baseline_metrics["continuation_counterfactual_prefix_count"]
        > 0.0
    )
    assert (
        profiles["full-internal-rl"].training_transition_count
        == baseline_metrics[
            "continuation_counterfactual_prefix_count"
        ]
    )
    assert (
        baseline_metrics[
            "continuation_counterfactual_oracle_mean_raw_delta"
        ]
        >= baseline_metrics[
            "continuation_counterfactual_best_fixed_mean_raw_delta"
        ]
    )
    assert baseline_metrics[
        "counterfactual_selector_train_count"
    ] > 0.0
    assert baseline_metrics[
        "counterfactual_selector_heldout_count"
    ] > 0.0
    assert baseline_metrics[
        "counterfactual_selector_eval_updates_after_fit"
    ] == 0.0
    assert baseline_metrics[
        "counterfactual_selector_train_audit_available_rate"
    ] == 1.0
    assert baseline_metrics[
        "counterfactual_selector_heldout_audit_available_rate"
    ] == 1.0
    assert baseline_metrics[
        "counterfactual_generated_continuation_count"
    ] > 0.0
    assert profiles[
        "full-internal-rl"
    ].action_selection_records
    assert all(
        not profile.action_selection_records
        for label, profile in profiles.items()
        if label != "full-internal-rl"
    )
    assert (
        baseline_metrics[
            "continuation_counterfactual_heldout_prefix_count"
        ]
        > 0.0
    )
    for profile_label in (
        "full-zero-control",
        "full-shuffled-control",
        "full-reversed-control",
    ):
        metrics = dict(profiles[profile_label].metric_means)
        assert (
            metrics[
                "continuation_pe_training_transition_count"
            ]
            == 0.0
        )


def test_eta_environment_outcome_target_uses_pe_credit_not_self_nll() -> None:
    cases = default_eta_proof_cases()
    train_case = next(case for case in cases if case.split == "train")
    heldout_case = next(
        case for case in cases if case.split == "heldout"
    )
    runtime = build_builtin_transformers_runtime(
        model_id="eta-environment-outcome-target-test",
    )

    report = run_eta_internal_rl_proof_benchmark(
        cases=(train_case, heldout_case),
        profile_labels=("full-internal-rl",),
        backend_label="transformers-open-weight",
        train_epochs=1,
        open_weight_runtime=runtime,
        open_weight_config=ETAOpenWeightRuntimeConfig(
            require_real_backend=False,
            max_prefix_steps=4,
        ),
        training_signal=ETA_CONTINUATION_PE_TRAINING_SIGNAL,
        latent_unit_clamp=True,
        continuation_counterfactual_grid=True,
        counterfactual_target_mode=(
            ETA_COUNTERFACTUAL_TARGET_ENVIRONMENT_OUTCOME
        ),
    )

    profile = report.profile_reports[0]
    metrics = dict(profile.metric_means)
    assert metrics["continuation_pe_training_transition_count"] > 0.0
    assert metrics["counterfactual_generated_continuation_count"] == 0.0
    assert metrics["counterfactual_environment_outcome_target_active"] == 1.0
    assert metrics["counterfactual_environment_application_count"] > 0.0
    assert (
        metrics[
            "counterfactual_environment_pe_credit_transition_count"
        ]
        == metrics["continuation_pe_training_transition_count"]
    )
    assert metrics["counterfactual_self_nll_target_active"] == 0.0
    records = profile.counterfactual_outcome_records
    assert records
    assert all(
        record.outcome_chain
        == (
            "residual-forward->environment-outcome->prediction-error"
            "->action-credit"
        )
        for record in records
    )
    assert any(record.action_credit > 0.0 for record in records)
    assert any(record.action_credit < 0.0 for record in records)
    assert all(
        record.target_signature != record.audit_target_signature
        for record in records
    )


def test_eta_selector_injection_requires_positive_audit_on_every_frozen_split() -> None:
    def selection(
        split: str,
        *,
        audit_selected_raw_delta: float,
    ) -> CounterfactualActionSelection:
        return CounterfactualActionSelection(
            example_id=f"{split}:example",
            group_id=f"{split}:group",
            split=split,
            prediction_source="test-frozen-selector",
            selected_action_index=1,
            oracle_action_index=1,
            predicted_top3_action_indices=(1, 2, 3),
            selected_raw_delta=0.1,
            oracle_raw_delta=0.1,
            oracle_regret=0.0,
            top1_match=True,
            top3_match=True,
            selected_positive=True,
            audit_selected_raw_delta=audit_selected_raw_delta,
            audit_oracle_raw_delta=max(
                audit_selected_raw_delta,
                0.1,
            ),
            audit_oracle_regret=max(
                0.0,
                0.1 - audit_selected_raw_delta,
            ),
            audit_selected_positive=(
                audit_selected_raw_delta > 0.0
            ),
            model_fingerprint="test-selector",
        )

    metrics = dict(
        _counterfactual_selector_metric_rows(
            selections=(
                selection("train", audit_selected_raw_delta=0.1),
                selection("eval", audit_selected_raw_delta=-0.01),
                selection("heldout", audit_selected_raw_delta=0.1),
                selection("validation", audit_selected_raw_delta=0.1),
            ),
            input_dim=4,
            latent_dim=4,
            action_count=22,
            ridge_strength=1.0,
            model_candidate_count=3,
            explained_variance_ratio=1.0,
        )
    )

    assert metrics["counterfactual_selector_injection_gate_passed"] == 0.0
