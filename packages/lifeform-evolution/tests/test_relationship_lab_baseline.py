from __future__ import annotations

import hashlib
import json

import pytest

from lifeform_domain_emogpt.lab import RelationshipAction, canonical_json, sha256_json
from lifeform_evolution.relationship_lab_baseline import (
    SCHEMA_CONSTRAINED_DECODING_MODE,
    HFStatelessRelationshipActionPolicy,
    StatelessActionCompletion,
    action_choice_schema_path,
    build_canonical_action_json_token_constraint,
    freeze_stateless_baseline_attestation,
    run_stateless_baseline,
    stateless_prompt_path,
    write_stateless_baseline_run,
)
from lifeform_evolution.relationship_lab_gate0 import (
    Gate0CalibrationConfig,
    GateCheckStatus,
    run_relationship_gate0_calibration,
)


class _AlwaysStayPolicy:
    model_id = "frozen-fake-policy"
    weights_sha256 = sha256_json("fake-weights")
    prompt_sha256 = sha256_json("fake-prompt")
    generation_config_sha256 = sha256_json("fake-generation")

    def choose(self, *, current_input: str, seed: int) -> StatelessActionCompletion:
        assert current_input
        assert seed >= 0
        return StatelessActionCompletion(
            raw_output='{"action_id":"stay_present_without_probe"}',
            chosen_action_id=RelationshipAction.STAY_PRESENT_WITHOUT_PROBE,
            prompt_tokens=40,
            completion_tokens=8,
        )


class _InvalidOutputPolicy(_AlwaysStayPolicy):
    def choose(self, *, current_input: str, seed: int) -> StatelessActionCompletion:
        del current_input, seed
        return StatelessActionCompletion(
            raw_output="I would stay.",
            chosen_action_id=None,
            prompt_tokens=40,
            completion_tokens=4,
        )


def test_hf_policy_requires_cached_positive_chunked_prefill() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        HFStatelessRelationshipActionPolicy(
            prefill_chunk_size=True,
            generation_use_cache=True,
        )
    with pytest.raises(ValueError, match="generation_use_cache=True"):
        HFStatelessRelationshipActionPolicy(
            prefill_chunk_size=2048,
            generation_use_cache=False,
        )


def test_hf_policy_forwards_frozen_chunked_prefill_to_generate() -> None:
    import torch

    class RecordingModel:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] | None = None

        def generate(self, **kwargs: object):
            self.kwargs = kwargs
            return torch.tensor([[11, 12, 13]])

    class FakeTokenizer:
        eos_token_id = 0

        @staticmethod
        def decode(_token_ids, *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens
            return '{"action_id":"stay_present_without_probe"}'

    policy = object.__new__(HFStatelessRelationshipActionPolicy)
    policy._device = "cpu"
    policy._temperature = 0.0
    policy._top_p = 1.0
    policy._max_new_tokens = 64
    policy._prefill_chunk_size = 2048
    policy._generation_use_cache = True
    policy._schema_constraint = None
    policy._torch = torch
    policy._tokenizer = FakeTokenizer()
    model = RecordingModel()
    policy._model = model
    policy._encode_contextual_messages = lambda *, messages: {
        "input_ids": torch.tensor([[11, 12]])
    }

    completion = policy.choose_from_messages(
        messages=({"role": "user", "content": "public input"},),
        seed=17,
    )

    assert model.kwargs is not None
    assert model.kwargs["use_cache"] is True
    assert model.kwargs["prefill_chunk_size"] == 2048
    assert model.kwargs["max_new_tokens"] == 64
    assert "prefix_allowed_tokens_fn" not in model.kwargs
    assert completion.chosen_action_id is RelationshipAction.STAY_PRESENT_WITHOUT_PROBE


class _CharacterActionTokenizer:
    eos_token_id = 0

    def __call__(self, text: str, *, add_special_tokens: bool):
        assert add_special_tokens is False
        return {"input_ids": [ord(character) + 1 for character in text]}

    @staticmethod
    def decode(token_ids, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        ids = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
        return "".join(chr(token - 1) for token in ids if token != 0)


@pytest.mark.parametrize(
    ("candidate_index", "expected_action"),
    tuple(
        enumerate(
            (
                RelationshipAction.STAY_PRESENT_WITHOUT_PROBE,
                RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION,
                RelationshipAction.NEUTRAL_NOOP,
            )
        )
    ),
)
def test_schema_constrained_generation_is_strict_valid_for_all_actions(
    candidate_index: int,
    expected_action: RelationshipAction,
) -> None:
    import torch

    tokenizer = _CharacterActionTokenizer()
    constraint = build_canonical_action_json_token_constraint(tokenizer)
    candidate_tokens = constraint.candidate_token_ids[candidate_index]

    class TrieFollowingModel:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] | None = None

        def generate(self, **kwargs: object):
            self.kwargs = kwargs
            input_ids = kwargs["input_ids"]
            prefix_fn = kwargs["prefix_allowed_tokens_fn"]
            prompt = [int(token) for token in input_ids[0].tolist()]
            generated: list[int] = []
            for token in (*candidate_tokens, constraint.eos_token_id):
                allowed = prefix_fn(0, torch.tensor([*prompt, *generated]))
                assert token in allowed
                generated.append(token)
            return torch.tensor([[*prompt, *generated]])

    policy = object.__new__(HFStatelessRelationshipActionPolicy)
    policy._device = "cpu"
    policy._temperature = 0.0
    policy._top_p = 1.0
    policy._max_new_tokens = constraint.maximum_completion_tokens
    policy._prefill_chunk_size = None
    policy._generation_use_cache = None
    policy._schema_constraint = constraint
    policy._torch = torch
    policy._tokenizer = tokenizer
    model = TrieFollowingModel()
    policy._model = model
    policy._encode_contextual_messages = lambda *, messages: {
        "input_ids": torch.tensor([[901, 902]])
    }

    completion = policy.choose_from_messages(
        messages=({"role": "user", "content": "public input"},),
        seed=19,
    )

    assert completion.raw_output == canonical_json({"action_id": expected_action.value})
    assert completion.chosen_action_id is expected_action
    assert model.kwargs is not None
    assert model.kwargs["eos_token_id"] == tokenizer.eos_token_id
    assert callable(model.kwargs["prefix_allowed_tokens_fn"])


def test_schema_constraint_payload_binds_mode_candidates_and_token_sequences() -> None:
    constraint = build_canonical_action_json_token_constraint(_CharacterActionTokenizer())
    payload = constraint.generation_config_payload()

    assert payload["mode"] == SCHEMA_CONSTRAINED_DECODING_MODE
    assert payload["canonical_candidates"] == [
        canonical_json({"action_id": RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value}),
        canonical_json(
            {"action_id": RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION.value}
        ),
        canonical_json({"action_id": RelationshipAction.NEUTRAL_NOOP.value}),
    ]
    assert payload["candidate_token_ids"] == [
        list(sequence) for sequence in constraint.candidate_token_ids
    ]
    assert payload["terminal_eos_token_id"] == _CharacterActionTokenizer.eos_token_id
    assert sha256_json({"legacy": True}) != sha256_json(
        {"legacy": True, "schema_constrained_decoding": payload}
    )


def test_stateless_runner_is_matched_and_excludes_heldout() -> None:
    run = run_stateless_baseline(_AlwaysStayPolicy())
    assert len(run.decisions) == 24
    assert run.valid_decisions == 24
    assert run.correct_decisions == 12
    assert run.context_tokens_total == 960
    assert {item.split.value for item in run.decisions} == {"train", "validation"}
    grouped: dict[tuple[str, int], list] = {}
    for decision in run.decisions:
        grouped.setdefault((decision.pair_id, decision.seed), []).append(decision)
    assert grouped
    for rows in grouped.values():
        assert len(rows) == 2
        assert rows[0].current_input_sha256 == rows[1].current_input_sha256
        assert rows[0].raw_output == rows[1].raw_output
        assert rows[0].chosen_action_id == rows[1].chosen_action_id
        assert rows[0].correct != rows[1].correct


def test_stateless_run_freezes_ledger_backed_attestation_and_closes_gate0(
    tmp_path,
) -> None:
    run = run_stateless_baseline(_AlwaysStayPolicy())
    attestation = freeze_stateless_baseline_attestation(
        run,
        frozen_at_iso="2026-08-19T02:00:00+00:00",
    )
    assert attestation.decision_ledger_sha256 == run.decision_ledger_sha256
    assert attestation.valid_decisions == len(run.decisions)
    report = run_relationship_gate0_calibration(
        config=Gate0CalibrationConfig(samples_per_action=64),
        baseline=attestation,
        created_at_iso="2026-08-19T02:01:00+00:00",
    )
    assert report.gate0_passed

    ledger, summary, frozen = write_stateless_baseline_run(
        run,
        output_dir=tmp_path,
        frozen_at_iso="2026-08-19T02:00:00+00:00",
    )
    assert hashlib.sha256(ledger.read_bytes()).hexdigest() == (run.decision_ledger_sha256)
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 24
    summary_payload = json.loads(summary.read_text(encoding="utf-8"))
    assert summary_payload["decision_ledger_sha256"] == run.decision_ledger_sha256
    frozen_payload = json.loads(frozen.read_text(encoding="utf-8"))
    assert frozen_payload["artifact_id"] == attestation.artifact_id


def test_invalid_structured_outputs_cannot_close_baseline_tooth() -> None:
    run = run_stateless_baseline(_InvalidOutputPolicy())
    assert run.valid_decisions == 0
    assert run.correct_decisions == 0
    attestation = freeze_stateless_baseline_attestation(
        run,
        frozen_at_iso="2026-08-19T02:00:00+00:00",
    )
    report = run_relationship_gate0_calibration(
        config=Gate0CalibrationConfig(samples_per_action=64),
        baseline=attestation,
        created_at_iso="2026-08-19T02:01:00+00:00",
    )
    assert report.machinery_ready
    assert not report.gate0_passed
    statuses = {check.check_id: check.status for check in report.checks}
    assert statuses["frozen_baseline_non_saturation"] is GateCheckStatus.FAIL


def test_stateless_prompt_and_schema_are_dedicated_assets() -> None:
    prompt = stateless_prompt_path().read_text(encoding="utf-8")
    schema = json.loads(action_choice_schema_path().read_text(encoding="utf-8"))
    assert "user history" in prompt
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]["action_id"]["enum"]) == {
        "stay_present_without_probe",
        "respect_space_with_return_option",
        "neutral_noop",
    }
