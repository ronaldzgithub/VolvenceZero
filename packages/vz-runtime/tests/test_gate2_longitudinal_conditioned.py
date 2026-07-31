from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from volvence_zero.agent.gate2_longitudinal_capture import (
    Gate2CandidateControlContract,
)
from volvence_zero.agent.gate2_longitudinal_conditioned import (
    GATE2_CONDITIONED_EVALUATION_SEEDS,
    GATE2_CONDITIONED_SOURCE_SCHEMA_VERSION,
    GATE2_RELATIONSHIP_PROFILES,
    build_gate2_conditioned_source_plans,
    build_gate2_conditioned_preregistration,
    build_gate2_conditioned_training_example,
    gate2_conditioned_permutation_action_index,
    gate2_conditioned_permuted_profile_id,
    gate2_conditioned_plan_digest,
    gate2_relationship_readouts,
    summarize_gate2_conditioned_seed,
    validate_gate2_conditioned_preregistration,
)
from volvence_zero.internal_rl import residual_action_state_vector
from volvence_zero.substrate import (
    ResidualActivation,
    ResidualSequenceStep,
    SubstrateSnapshot,
    SurfaceKind,
)


def _snapshot() -> SubstrateSnapshot:
    steps = tuple(
        ResidualSequenceStep(
            step=step,
            token=f"token-{step}",
            feature_surface=(),
            residual_activations=(
                ResidualActivation(
                    layer_index=20,
                    activation=(0.1 + step, -0.2, 0.3),
                    step=step,
                ),
            ),
            description="conditioned selector test",
        )
        for step in range(3)
    )
    return SubstrateSnapshot(
        model_id="test",
        is_frozen=True,
        surface_kind=SurfaceKind.RESIDUAL_STREAM,
        token_logits=(),
        feature_surface=(),
        residual_activations=steps[-1].residual_activations,
        residual_sequence=steps,
        unavailable_fields=(),
        description="conditioned selector test",
    )


@dataclass(frozen=True)
class _Score:
    mean_negative_log_likelihood: float


class _Runtime:
    def score_continuation(
        self,
        *,
        source_text: str,
        continuation_text: str,
        applied_control: tuple[float, float, float],
        track_scale: tuple[float, float, float],
    ) -> _Score:
        del source_text, continuation_text, track_scale
        return _Score(2.0 - applied_control[0])


def _candidate_contract() -> Gate2CandidateControlContract:
    controls = tuple((index / 100.0, 0.0, 0.0) for index in range(22))
    return Gate2CandidateControlContract(
        controls=controls,
        source_sha256="source",
        mapping_fingerprint="mapping",
    )


def test_source_registry_is_fresh_typed_and_deterministic() -> None:
    first = build_gate2_conditioned_source_plans(seed=1301, count=510)
    second = build_gate2_conditioned_source_plans(seed=1301, count=510)
    other_seed = build_gate2_conditioned_source_plans(seed=1313, count=510)

    assert first == second
    assert gate2_conditioned_plan_digest(first) == gate2_conditioned_plan_digest(
        second
    )
    assert gate2_conditioned_plan_digest(first) != gate2_conditioned_plan_digest(
        other_seed
    )
    assert len({plan.transition_id for plan in first}) == 510
    assert all(
        plan.schema_version == GATE2_CONDITIONED_SOURCE_SCHEMA_VERSION
        for plan in first
    )
    assert {plan.relationship_profile_id for plan in first} == {
        profile.profile_id for profile in GATE2_RELATIONSHIP_PROFILES
    }
    assert len({plan.prediction_turn for plan in first[:4]}) == 1
    assert len({plan.settlement_turn for plan in first[:4]}) == 4


def test_relationship_profiles_compile_through_real_owner_without_collapse() -> None:
    readouts = gate2_relationship_readouts()

    assert set(readouts) == {
        profile.profile_id for profile in GATE2_RELATIONSHIP_PROFILES
    }
    assert all(not readout.is_cold_start for readout in readouts.values())
    assert all(readout.confidence > 0.0 for readout in readouts.values())
    assert len({readout.readout for readout in readouts.values()}) == len(readouts)


def test_condition_and_action_permutations_are_preregistered_and_balanced() -> None:
    schedules = {
        seed: tuple(
            gate2_conditioned_permutation_action_index(
                seed=seed,
                global_index=index,
            )
            for index in range(510)
        )
        for seed in GATE2_CONDITIONED_EVALUATION_SEEDS
    }
    for schedule in schedules.values():
        counts = tuple(schedule.count(index) for index in range(22))
        assert min(counts) == 23
        assert max(counts) == 24
    for seed in GATE2_CONDITIONED_EVALUATION_SEEDS:
        for profile in GATE2_RELATIONSHIP_PROFILES:
            assert gate2_conditioned_permuted_profile_id(
                seed=seed,
                relationship_profile_id=profile.profile_id,
            ) != profile.profile_id


def test_training_example_appends_owner_state_and_scores_all_actions() -> None:
    plan = build_gate2_conditioned_source_plans(seed=1291, count=1)[0]
    readout = gate2_relationship_readouts()[plan.relationship_profile_id]
    example = build_gate2_conditioned_training_example(
        plan=plan,
        snapshot=_snapshot(),
        relationship_readout=readout,
        runtime=_Runtime(),
        candidate_contract=_candidate_contract(),
    )

    assert len(example.state_features) == len(
        residual_action_state_vector(_snapshot())
    ) + len(readout.readout)
    assert len(example.candidate_raw_deltas) == 22
    assert example.candidate_raw_deltas[0] == pytest.approx(0.0)
    assert example.candidate_raw_deltas[-1] == pytest.approx(0.21)


def test_single_seed_stoploss_requires_conditioned_effect() -> None:
    rows = tuple(
        {
            "consumer_session_index": index // 10,
            "selected_minus_action_permutation": 0.03,
            "selected_minus_zero": 0.03,
            "selected_minus_condition_permutation": 0.0,
        }
        for index in range(510)
    )
    summary = summarize_gate2_conditioned_seed(seed=1301, rows=rows)

    assert summary["complete"] is True
    assert summary["gates"][
        "selector_minus_condition_permutation_at_least_0_02"
    ] is False
    assert summary["single_seed_stoploss_passed"] is False


def test_preregistration_binds_fresh_plans_code_and_candidate_mapping() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    payload = build_gate2_conditioned_preregistration(
        repo_root=repository_root,
        candidate_artifact_path=(
            "artifacts/"
            "eta_gate2_residual_causal_v35_selector_null_fresh_"
            "fullwidth896_qwen25_05b_cpu_1seed_20260729/"
            "counterfactual_outcomes.jsonl"
        ),
        substrate_fingerprint="frozen-test-substrate",
    )

    validate_gate2_conditioned_preregistration(
        payload,
        repo_root=repository_root,
    )
    assert payload["evaluation"]["seeds"] == [1301, 1313, 1327]
    assert payload["forbidden_reuse"]["historical_routes_reused"] is False
    assert payload["mechanism"]["conditioned_input_shape"] == 8090
