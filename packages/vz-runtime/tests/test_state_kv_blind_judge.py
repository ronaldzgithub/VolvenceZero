"""Blind-judge guard acceptance.

Exercised with a stub model so the properties that make the votes admissible
are tested without loading two checkpoints:

* the cross-family rule is enforced at construction, not documented;
* a judge with a fixed positional preference scores exactly chance, which is
  what the control arm's claim depends on;
* material that would make the comparison unlike is refused up front.
"""

from __future__ import annotations

import torch

import pytest

from volvence_zero.state_kv_blind_judge import (
    JudgeMaterial,
    JudgeMaterialKind,
    LocalEmbeddingBlindJudge,
    LocalTransformersBlindJudge,
)


class _StubTokenizer:
    """Tensor-only outputs, like a real ``return_tensors="pt"`` tokenizer.

    The prompt text is handed to the stub model out of band rather than smuggled
    into the encoding, so the judge keeps seeing only tensors -- its strictness
    about that is the behaviour real HF tokenizers rely on.
    """

    def __init__(self) -> None:
        self.last_text = ""

    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, object]:
        del return_tensors
        self.last_text = text
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return {"A": [10], "B": [11]}.get(text.strip(), [0])


class _StubModel:
    """Returns logits chosen by a callable over the prompt text."""

    def __init__(self, score_fn, tokenizer: _StubTokenizer) -> None:
        self._score_fn = score_fn
        self._tokenizer = tokenizer

    def to(self, device: str) -> "_StubModel":
        del device
        return self

    def eval(self) -> "_StubModel":
        return self

    def __call__(self, **kwargs: object) -> object:
        del kwargs
        logits = torch.full((1, 3, 12), -10.0)
        score_a, score_b = self._score_fn(self._tokenizer.last_text)
        logits[0, -1, 10] = score_a
        logits[0, -1, 11] = score_b
        return type("Out", (), {"logits": logits})()


class _EmbeddingTokenizer:
    def __call__(
        self,
        text: str,
        return_tensors: str = "pt",
        truncation: bool = True,
        max_length: int = 512,
    ) -> dict[str, object]:
        del return_tensors, truncation, max_length
        token = 1 if "grief" in text.lower() else 2
        return {
            "input_ids": torch.tensor([[token]]),
            "attention_mask": torch.tensor([[1]]),
        }


class _EmbeddingModel:
    def to(self, device: str) -> "_EmbeddingModel":
        del device
        return self

    def eval(self) -> "_EmbeddingModel":
        return self

    def __call__(self, **kwargs: object) -> object:
        token = int(kwargs["input_ids"][0, 0])  # type: ignore[index]
        vector = torch.tensor([[[1.0, 0.0] if token == 1 else [0.0, 1.0]]])
        return type("Out", (), {"last_hidden_state": vector})()


def _materials(kind: str = JudgeMaterialKind.SESSION_HISTORY):
    return (
        JudgeMaterial(user_id="u1", summary="Grieving a pet.", material_kind=kind),
        JudgeMaterial(user_id="u2", summary="Starting a new job.", material_kind=kind),
    )


def _judge(score_fn, **overrides):
    tokenizer = _StubTokenizer()
    kwargs = {
        "judge_model_id": "stub/judge",
        "substrate_model_id": "stub/substrate",
        "materials": _materials(),
        "model": _StubModel(score_fn, tokenizer),
        "tokenizer": tokenizer,
        "judge_family": "llama",
        "substrate_family": "qwen2",
    }
    kwargs.update(overrides)
    return LocalTransformersBlindJudge(**kwargs)


def _embedding_judge(**overrides):
    kwargs = {
        "judge_model_id": "stub/embedder",
        "substrate_model_id": "stub/substrate",
        "materials": (
            JudgeMaterial(
                user_id="u1",
                summary="grief support state",
                material_kind=JudgeMaterialKind.RENDERED_STATE,
            ),
            JudgeMaterial(
                user_id="u2",
                summary="planning structure state",
                material_kind=JudgeMaterialKind.RENDERED_STATE,
            ),
        ),
        "model": _EmbeddingModel(),
        "tokenizer": _EmbeddingTokenizer(),
        "judge_family": "xlm-roberta",
        "substrate_family": "qwen2",
    }
    kwargs.update(overrides)
    return LocalEmbeddingBlindJudge(**kwargs)


def test_same_family_judge_is_refused() -> None:
    with pytest.raises(ValueError, match="cross-family judge rule violated"):
        _judge(lambda text: (0.0, -1.0), judge_family="qwen2")


def test_position_biased_judge_lands_exactly_at_chance() -> None:
    """Always-pick-A must not become a signal.

    Order symmetrization is what lets the control arm collapse to chance
    instead of inheriting the judge's positional preference.
    """

    judge = _judge(lambda text: (0.0, -5.0))  # always prefers whatever is "A"
    votes = [
        judge.match(response_text="a reply", candidate_user_ids=("u1", "u2"))
        for _ in range(10)
    ]
    # Both orderings cancel: the score is 0, so the tie-break is deterministic
    # and the judge carries no information either way.
    assert judge.tie_count == 10
    assert set(votes) == {"u2"}


def test_content_sensitive_judge_recovers_the_right_user() -> None:
    def score(text: str) -> tuple[float, float]:
        # Prefer whichever slot holds the grieving summary.
        slot_a = text.index("Person A:")
        slot_b = text.index("Person B:")
        grief = text.index("Grieving a pet.")
        return (0.0, -3.0) if slot_a < grief < slot_b else (-3.0, 0.0)

    judge = _judge(score)
    assert (
        judge.match(response_text="I am sorry.", candidate_user_ids=("u1", "u2"))
        == "u1"
    )
    assert (
        judge.match(response_text="I am sorry.", candidate_user_ids=("u2", "u1"))
        == "u1"
    )
    assert judge.tie_count == 0


def test_embedding_judge_matches_by_content_similarity() -> None:
    judge = _embedding_judge()

    assert (
        judge.match(response_text="offer grief support", candidate_user_ids=("u1", "u2"))
        == "u1"
    )
    assert (
        judge.match(response_text="make a structured plan", candidate_user_ids=("u1", "u2"))
        == "u2"
    )
    assert judge.tie_count == 0


def test_mixed_material_kinds_are_refused() -> None:
    mixed = (
        JudgeMaterial(
            user_id="u1", summary="History.", material_kind=JudgeMaterialKind.SESSION_HISTORY
        ),
        JudgeMaterial(
            user_id="u2", summary="State.", material_kind=JudgeMaterialKind.RENDERED_STATE
        ),
    )
    with pytest.raises(ValueError, match="same kind of material"):
        _judge(lambda text: (0.0, -1.0), materials=mixed)


def test_empty_summary_is_refused() -> None:
    with pytest.raises(ValueError, match="empty summary"):
        JudgeMaterial(
            user_id="u1", summary="   ", material_kind=JudgeMaterialKind.RENDERED_STATE
        )


def test_unknown_material_kind_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown judge material kind"):
        JudgeMaterial(user_id="u1", summary="x", material_kind="vibes")


def test_empty_response_is_a_substrate_failure_not_a_coin_flip() -> None:
    judge = _judge(lambda text: (0.0, -1.0))
    with pytest.raises(ValueError, match="empty response"):
        judge.match(response_text="   ", candidate_user_ids=("u1", "u2"))


def test_unknown_candidate_is_refused() -> None:
    judge = _judge(lambda text: (0.0, -1.0))
    with pytest.raises(ValueError, match="no judge material"):
        judge.match(response_text="hi", candidate_user_ids=("u1", "ghost"))


def test_judge_provenance_is_reportable() -> None:
    judge = _judge(lambda text: (0.0, -1.0))
    judge.match(response_text="hi", candidate_user_ids=("u1", "u2"))
    payload = judge.as_json_dict()
    assert payload["judge_family"] == "llama"
    assert payload["substrate_family"] == "qwen2"
    assert payload["order_symmetrized"] is True
    assert payload["greedy"] is True
    assert payload["decision_count"] == 1
    assert payload["material_kind"] == JudgeMaterialKind.SESSION_HISTORY
