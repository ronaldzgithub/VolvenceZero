from __future__ import annotations

from dataclasses import replace
import json

import pytest

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
    ETA_GATE2_SHADOW_FRESH_CASE_CORPUS,
    ETAOpenWeightRuntimeConfig,
    _counterfactual_selector_metric_rows,
    _install_learned_control_basis,
    default_eta_proof_cases,
    eta_gate2_expanded_cases,
    eta_gate2_expected_value_cases,
    eta_gate2_expected_value_validation_routes,
    eta_gate2_independent_training_routes,
    eta_gate2_selector_confirmation_routes,
    eta_gate2_selector_fresh_cases,
    eta_gate2_selector_fresh_validation_routes,
    eta_gate2_shadow_confirmation_routes,
    eta_gate2_shadow_fresh_cases,
    eta_gate2_shadow_fresh_validation_routes,
    run_eta_internal_rl_paper_suite,
    run_eta_internal_rl_proof_benchmark,
)
from volvence_zero.internal_rl import CounterfactualActionSelection
from volvence_zero.substrate import (
    TRAIN_TRANSITION_PCA_CONTROL_BASIS_MODE,
    build_builtin_transformers_runtime,
)


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
    for case in (
        eta_gate2_selector_fresh_cases()
        + eta_gate2_shadow_fresh_cases()
    ):
        source_tokens = set(case.source_text.lower().split())
        assert source_tokens.isdisjoint(
            {"train", "eval", "heldout", "validation", "confirmation"}
        )


def test_eta_gate2_freezes_expected_value_validation_corpus() -> None:
    default_cases = default_eta_proof_cases()
    expanded_cases = eta_gate2_expanded_cases()
    expected_value_cases = eta_gate2_expected_value_cases()
    selector_fresh_cases = eta_gate2_selector_fresh_cases()
    shadow_fresh_cases = eta_gate2_shadow_fresh_cases()

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
    assert sum(
        case.split == "validation" for case in selector_fresh_cases
    ) == 4
    assert sum(
        case.split == "confirmation" for case in selector_fresh_cases
    ) == 4
    assert sum(
        case.split == "validation" for case in shadow_fresh_cases
    ) == 4
    assert sum(
        case.split == "confirmation" for case in shadow_fresh_cases
    ) == 4
    assert not {
        case.case_id for case in expected_value_cases
        if case.split == "validation"
    } & {
        case.case_id for case in selector_fresh_cases
        if case.split in {"validation", "confirmation"}
    }
    assert not {
        case.case_id for case in selector_fresh_cases
        if case.split in {"validation", "confirmation"}
    } & {
        case.case_id for case in shadow_fresh_cases
        if case.split in {"validation", "confirmation"}
    }

    manifest = build_eta_gate2_residual_manifest(suite_tier="ci-smoke")
    case_groups = dict(manifest.case_groups)
    assert case_groups["case_corpus"] == (
        ETA_GATE2_SHADOW_FRESH_CASE_CORPUS,
    )
    assert case_groups["route_ids"] == tuple(
        case.case_id for case in shadow_fresh_cases
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
        for route in eta_gate2_shadow_fresh_validation_routes()
    )
    assert case_groups["confirmation_route_ids"] == tuple(
        route.case_id
        for route in eta_gate2_shadow_confirmation_routes()
    )
    assert case_groups["superseded_validation_route_ids"] == tuple(
        route.case_id
        for route in (
            eta_gate2_expected_value_validation_routes()
            + eta_gate2_selector_fresh_validation_routes()
            + eta_gate2_selector_confirmation_routes()
        )
    )
    assert case_groups["selector_signal_gate"] == (
        "selector-vs-permutation-null-v1",
    )
    assert case_groups["shadow_observation_gate"] == (
        "shadow-closed-loop-v1",
    )
    assert case_groups["shadow_closed_loop_arm"] == ("true",)
    assert manifest.repeat_count == 3
    assert manifest.seed_schedule == (0, 1, 2)
    assert case_groups["confirmation_split_locked"] == ("true",)
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
    assert case_groups["control_basis_mode"] == (
        TRAIN_TRANSITION_PCA_CONTROL_BASIS_MODE,
    )
    assert case_groups["control_basis_fit_split"] == ("train",)
    assert case_groups["control_basis_rank"] == ("3",)
    assert case_groups["control_basis_state_coordinate"] == (
        "hook-layer-mean-last-token-hidden",
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


def test_eta_gate2_v35_fresh_splits_are_lexically_fresh() -> None:
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
            for route in eta_gate2_selector_fresh_validation_routes()
        )
    )
    confirmation_words = set().union(
        *(
            content_words(route.source_text)
            for route in eta_gate2_selector_confirmation_routes()
        )
    )
    superseded_validation_words = set().union(
        *(
            content_words(route.source_text)
            for route in eta_gate2_expected_value_validation_routes()
        )
    )

    assert validation_words
    assert confirmation_words
    assert validation_words.isdisjoint(existing_words)
    assert confirmation_words.isdisjoint(existing_words)
    assert validation_words.isdisjoint(superseded_validation_words)
    assert confirmation_words.isdisjoint(superseded_validation_words)

    prior_words = (
        existing_words
        | validation_words
        | confirmation_words
        | superseded_validation_words
    )
    v36_validation_words = set().union(
        *(
            content_words(route.source_text)
            for route in eta_gate2_shadow_fresh_validation_routes()
        )
    )
    v36_confirmation_words = set().union(
        *(
            content_words(route.source_text)
            for route in eta_gate2_shadow_confirmation_routes()
        )
    )
    assert v36_validation_words
    assert v36_confirmation_words
    assert v36_validation_words.isdisjoint(prior_words)
    assert v36_confirmation_words.isdisjoint(prior_words)
    assert v36_validation_words.isdisjoint(v36_confirmation_words)


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
            else ("control_basis_mode", ("fixed-sinusoid-v1",))
            if name == "control_basis_mode"
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
        for route in eta_gate2_shadow_fresh_validation_routes()
    ]
    assert manifest_payload["scenario_split"][
        "confirmation_route_ids"
    ] == [
        route.case_id
        for route in eta_gate2_shadow_confirmation_routes()
    ]
    assert manifest_payload["scenario_split"][
        "superseded_validation_route_ids"
    ] == [
        route.case_id
        for route in (
            eta_gate2_expected_value_validation_routes()
            + eta_gate2_selector_fresh_validation_routes()
            + eta_gate2_selector_confirmation_routes()
        )
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
        "shadow_closed_loop_arm": "true",
        "shadow_observation_gate": "shadow-closed-loop-v1",
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
    assert manifest_payload["scenario_split"][
        "causal_confirmation_split"
    ] == [
        route.case_id
        for route in eta_gate2_shadow_confirmation_routes()
    ]
    assert manifest_payload["scenario_split"][
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
            else ("control_basis_mode", ("fixed-sinusoid-v1",))
            if name == "control_basis_mode"
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


def test_eta_gate2_learned_control_basis_fit_is_train_only_and_installs() -> None:
    runtime = build_builtin_transformers_runtime(activation_width=48)
    assert runtime.control_basis_provenance == "fixed-sinusoid-v1"
    cases = default_eta_proof_cases()
    config = ETAOpenWeightRuntimeConfig(require_real_backend=False)

    descriptor = _install_learned_control_basis(
        runtime=runtime,
        cases=cases,
        open_weight_config=config,
    )

    assert descriptor["control_basis_mode"] == (
        TRAIN_TRANSITION_PCA_CONTROL_BASIS_MODE
    )
    assert descriptor["control_basis_fit_split"] == "train"
    assert descriptor["control_basis_fingerprint"]
    assert int(descriptor["control_basis_transition_count"]) >= 4
    fit_route_ids = descriptor["control_basis_fit_route_ids"].split(",")
    train_case_ids = {case.case_id for case in cases if case.split == "train"}
    assert set(fit_route_ids) == train_case_ids
    assert runtime.control_basis_provenance == (
        descriptor["control_basis_provenance"]
    )
    assert runtime.control_basis_provenance.startswith(
        TRAIN_TRANSITION_PCA_CONTROL_BASIS_MODE
    )
    # Deterministic artifact: refitting the same corpus reproduces the
    # same fingerprint.
    second = _install_learned_control_basis(
        runtime=runtime,
        cases=cases,
        open_weight_config=config,
    )
    assert second["control_basis_fingerprint"] == (
        descriptor["control_basis_fingerprint"]
    )


def test_eta_gate2_learned_control_basis_requires_open_weight_runtime(
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

    try:
        run_eta_internal_rl_paper_suite(
            manifest=manifest,
            output_dir=tmp_path,
        )
    except ValueError as exc:
        assert "requires an open-weight runtime" in str(exc)
    else:
        raise AssertionError(
            "expected learned control-basis mode without an open-weight "
            "runtime to fail loudly"
        )


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
                    "case_id": "route-a",
                    "step_index": prefix,
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


def _action_selection_rows(
    *,
    split: str,
    prefix_count: int,
    selected_index: int = 1,
    selected_audit_raw_delta: float,
) -> list[dict]:
    return [
        {
            "profile_label": "full-internal-rl",
            "split": split,
            "group_id": "route-a",
            "example_id": f"route-a:prefix-{prefix}",
            "selected_action_index": selected_index,
            "audit_selected_raw_delta": selected_audit_raw_delta,
        }
        for prefix in range(prefix_count)
    ]


def test_eta_gate2_signal_gates_reject_selector_max_of_noise() -> None:
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
        *_counterfactual_grid_rows(
            split="confirmation",
            prefix_count=2,
            audit_tracks_target=False,
        ),
    ]
    action_selection = [
        *_action_selection_rows(
            split="train",
            prefix_count=3,
            selected_audit_raw_delta=-0.009,
        ),
        *_action_selection_rows(
            split="validation",
            prefix_count=2,
            selected_audit_raw_delta=-0.009,
        ),
        *_action_selection_rows(
            split="confirmation",
            prefix_count=2,
            selected_audit_raw_delta=-0.009,
        ),
    ]
    ablation = _build_ablation_results(
        predictions=[],
        outcomes=[],
        metric_samples={},
        action_selection=action_selection,
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
        "train_selector_exceeds_permutation_null"
    ]
    assert not ablation["signal_gates"][
        "validation_selector_exceeds_permutation_null"
    ]
    assert not ablation["signal_gates"][
        "confirmation_selector_exceeds_permutation_null"
    ]
    selector_diagnostics = ablation[
        "selector_permutation_null_by_split"
    ]["train"]
    assert selector_diagnostics["selected_excess_over_null_mean"] < 0.0
    assert ablation["reachable_solution_evidence"] is False

    verdict = _promotion_verdict(ablation)
    assert verdict["promotion_allowed"] is False
    assert verdict["reachable_solution_evidence"] is False
    assert (
        "train_selector_exceeds_permutation_null"
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
            *_counterfactual_grid_rows(
                split="confirmation",
                prefix_count=2,
                audit_tracks_target=audit_tracks_target,
            ),
        ]
        selected_audit = 0.008 if audit_tracks_target else -0.009
        action_selection = [
            *_action_selection_rows(
                split="train",
                prefix_count=3,
                selected_audit_raw_delta=selected_audit,
            ),
            *_action_selection_rows(
                split="validation",
                prefix_count=2,
                selected_audit_raw_delta=selected_audit,
            ),
            *_action_selection_rows(
                split="confirmation",
                prefix_count=2,
                selected_audit_raw_delta=selected_audit,
            ),
        ]
        return _build_ablation_results(
            predictions=[],
            outcomes=[],
            metric_samples={},
            action_selection=action_selection,
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


def test_eta_gate2_selector_signal_gate_requires_exact_candidate_lineage() -> None:
    rows = [
        *_counterfactual_grid_rows(
            split="train",
            prefix_count=1,
            audit_tracks_target=True,
        ),
        *_counterfactual_grid_rows(
            split="validation",
            prefix_count=1,
            audit_tracks_target=True,
        ),
        *_counterfactual_grid_rows(
            split="confirmation",
            prefix_count=1,
            audit_tracks_target=True,
        ),
    ]
    action_selection = [
        *_action_selection_rows(
            split="train",
            prefix_count=1,
            selected_audit_raw_delta=0.008,
        ),
        *_action_selection_rows(
            split="validation",
            prefix_count=1,
            selected_audit_raw_delta=0.008,
        ),
        *_action_selection_rows(
            split="confirmation",
            prefix_count=1,
            selected_audit_raw_delta=0.008,
        ),
    ]
    # A positive value copied from another candidate must not be accepted
    # as the selected candidate's independent audit measurement.
    action_selection[1]["audit_selected_raw_delta"] = 0.006

    ablation = _build_ablation_results(
        predictions=[],
        outcomes=[],
        metric_samples={},
        action_selection=action_selection,
        counterfactual_outcomes=rows,
    )

    validation = ablation["selector_permutation_null_by_split"][
        "validation"
    ]
    assert validation["input_selection_count"] == 1.0
    assert validation["selection_count"] == 0.0
    assert validation["selected_audit_lineage_mismatch_count"] == 1.0
    assert not ablation["signal_gates"][
        "validation_selector_lineage_complete"
    ]
    assert ablation["reachable_solution_evidence"] is False


def test_eta_gate2_selector_signal_gate_rejects_missing_grid_rows() -> None:
    rows = [
        *_counterfactual_grid_rows(
            split="train",
            prefix_count=1,
            audit_tracks_target=True,
        ),
        *_counterfactual_grid_rows(
            split="validation",
            prefix_count=1,
            audit_tracks_target=True,
        ),
        *_counterfactual_grid_rows(
            split="confirmation",
            prefix_count=1,
            audit_tracks_target=True,
        ),
    ]
    action_selection = [
        *_action_selection_rows(
            split="train",
            prefix_count=1,
            selected_audit_raw_delta=0.008,
        ),
        *_action_selection_rows(
            split="validation",
            prefix_count=1,
            selected_audit_raw_delta=0.008,
        ),
        *_action_selection_rows(
            split="confirmation",
            prefix_count=1,
            selected_audit_raw_delta=0.008,
        ),
        {
            **_action_selection_rows(
                split="validation",
                prefix_count=1,
                selected_audit_raw_delta=0.008,
            )[0],
            "group_id": "missing-route",
            "example_id": "missing-route:prefix-0",
        },
    ]

    ablation = _build_ablation_results(
        predictions=[],
        outcomes=[],
        metric_samples={},
        action_selection=action_selection,
        counterfactual_outcomes=rows,
    )

    validation = ablation["selector_permutation_null_by_split"][
        "validation"
    ]
    assert validation["input_selection_count"] == 2.0
    assert validation["selection_count"] == 1.0
    assert validation["missing_counterfactual_grid_count"] == 1.0
    assert not ablation["signal_gates"][
        "validation_selector_lineage_complete"
    ]
    assert ablation["reachable_solution_evidence"] is False


def test_eta_gate2_selector_signal_gate_rejects_conflicting_grid_rows() -> None:
    rows = _counterfactual_grid_rows(
        split="train",
        prefix_count=1,
        audit_tracks_target=True,
    )
    rows.append(
        {
            **rows[1],
            "audit_action_credit": 0.25,
        }
    )

    with pytest.raises(ValueError, match="conflicting audit credit"):
        _build_ablation_results(
            predictions=[],
            outcomes=[],
            metric_samples={},
            action_selection=[],
            counterfactual_outcomes=rows,
        )


def _shadow_gate_fixture() -> tuple[list[dict], list[dict]]:
    rows = []
    artifacts = []
    for seed in range(3):
        run_id = f"shadow-run-{seed}"
        selector_fingerprint = f"selector-{seed}"
        basis_fingerprint = "basis-fingerprint"
        artifacts.append(
            {
                "schema_version": "eta-gate2-selector-artifact.v1",
                "run_id": run_id,
                "run_seed": seed,
                "fit_split": "train",
                "control_basis_fingerprint": basis_fingerprint,
                "artifact": {
                    "model_fingerprint": selector_fingerprint,
                },
            }
        )
        for split in ("train", "validation", "confirmation"):
            for arm, delta in (
                ("zero-control", 0.0),
                ("selector", 0.03),
                ("permutation-null", 0.01),
            ):
                rows.append(
                    {
                        "profile_label": "full-internal-rl",
                        "run_id": run_id,
                        "run_seed": seed,
                        "split": split,
                        "case_id": f"{split}-route",
                        "step_index": 0,
                        "arm": arm,
                        "realized_delta": delta,
                        "selector_fingerprint": selector_fingerprint,
                        "control_basis_fingerprint": basis_fingerprint,
                        "runtime_descriptor_fingerprint": "runtime-fp",
                        "side_effect_free": True,
                    }
                )
    return rows, artifacts


def test_eta_gate2_shadow_observation_gate_is_independent_of_v35_promotion() -> None:
    rows, artifacts = _shadow_gate_fixture()
    passed = _build_ablation_results(
        predictions=[],
        outcomes=[],
        metric_samples={},
        action_selection=[],
        counterfactual_outcomes=[],
        selector_artifacts=artifacts,
        shadow_closed_loop=rows,
        inherited_causal_promotion=True,
    )

    assert passed["shadow_observation_passed"] is True
    assert all(passed["shadow_gates"].values())
    verdict = _promotion_verdict(passed)
    assert verdict["promotion_allowed"] is True
    assert verdict["shadow_observation_passed"] is True

    failed_rows = [
        {
            **row,
            "realized_delta": -0.03,
        }
        if row["split"] == "confirmation" and row["arm"] == "selector"
        else row
        for row in rows
    ]
    failed = _build_ablation_results(
        predictions=[],
        outcomes=[],
        metric_samples={},
        action_selection=[],
        counterfactual_outcomes=[],
        selector_artifacts=artifacts,
        shadow_closed_loop=failed_rows,
        inherited_causal_promotion=True,
    )
    failed_verdict = _promotion_verdict(failed)

    assert failed["shadow_observation_passed"] is False
    assert failed_verdict["promotion_allowed"] is True
    assert failed_verdict["shadow_observation_passed"] is False
    assert (
        "confirmation_selector_beats_zero"
        in failed_verdict["kill_conditions"]
    )


def test_eta_gate2_closed_loop_shadow_emits_frozen_side_effect_free_records() -> None:
    runtime = build_builtin_transformers_runtime(activation_width=48)
    all_cases = eta_gate2_shadow_fresh_cases()
    train_cases = tuple(
        case for case in all_cases if case.split == "train"
    )[:2]
    validation_case = next(
        case for case in all_cases if case.split == "validation"
    )
    confirmation_case = next(
        case for case in all_cases if case.split == "confirmation"
    )
    cases = (*train_cases, validation_case, confirmation_case)
    config = ETAOpenWeightRuntimeConfig(
        require_real_backend=False,
        activation_width=48,
        max_prefix_steps=6,
    )
    basis = _install_learned_control_basis(
        runtime=runtime,
        cases=cases,
        open_weight_config=config,
    )

    report = run_eta_internal_rl_proof_benchmark(
        cases=cases,
        profile_labels=("full-internal-rl",),
        backend_label="transformers-open-weight",
        train_epochs=1,
        open_weight_runtime=runtime,
        open_weight_config=config,
        training_signal=ETA_CONTINUATION_PE_TRAINING_SIGNAL,
        latent_unit_clamp=True,
        real_residual_ssl_bootstrap=True,
        causal_action_head_active=True,
        causal_action_head_state_dim=12,
        continuation_counterfactual_grid=True,
        counterfactual_action_selector_diagnostic=True,
        counterfactual_target_mode=(
            ETA_COUNTERFACTUAL_TARGET_ENVIRONMENT_OUTCOME
        ),
        shadow_closed_loop_arm=True,
        evidence_run_id="closed-loop-test",
        evidence_run_seed=0,
        control_basis_fingerprint_value=basis[
            "control_basis_fingerprint"
        ],
    )

    profile = report.profile_reports[0]
    artifact = profile.selector_artifact_payload
    records = profile.shadow_closed_loop_records
    assert artifact is not None
    assert artifact["fit_split"] == "train"
    assert artifact["control_basis_fingerprint"] == basis[
        "control_basis_fingerprint"
    ]
    assert records
    assert all(record.side_effect_free for record in records)
    assert {
        record.arm for record in records
    } == {"selector", "zero-control", "permutation-null"}
    step_arms: dict[tuple[str, int], set[str]] = {}
    for record in records:
        step_arms.setdefault(
            (record.case_id, record.step_index),
            set(),
        ).add(record.arm)
        assert record.selector_fingerprint == artifact["artifact"][
            "model_fingerprint"
        ]
        assert record.control_basis_fingerprint == basis[
            "control_basis_fingerprint"
        ]
    assert all(
        arms == {"selector", "zero-control", "permutation-null"}
        for arms in step_arms.values()
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
            "residual-forward->realized-continuation-nll->prediction-error"
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
