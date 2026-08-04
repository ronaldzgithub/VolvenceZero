"""Tests for the differentiable steered action scorer (ETA Eq.3 distortion).

The scorer is fully dependency-injected, so these run against a tiny real
torch stack instead of a downloaded checkpoint. The properties under test are
the ones the rate-distortion evidence depends on: gradient reaches the control
delta and never the frozen base, the injected delta is norm-capped, and the
joint-training validity control leaves the shared model pristine.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from volvence_zero.substrate.steered_action_scoring import (  # noqa: E402
    SteeredActionOption,
    TransformersSteeredActionScorer,
)

HIDDEN_SIZE = 8
BLOCK_COUNT = 4
INJECTION_LAYER = 1


class _Block(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(hidden_states))


class _TinyCausalLM(torch.nn.Module):
    def __init__(self, *, vocab_size: int, hidden_size: int, blocks: int) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.blocks = torch.nn.ModuleList(
            _Block(hidden_size) for _ in range(blocks)
        )
        self.final_norm = torch.nn.LayerNorm(hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def get_output_embeddings(self) -> torch.nn.Linear:
        return self.lm_head

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool = False,
        logits_to_keep: int = 0,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        hidden = self.embed(input_ids)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        slice_indices = (
            slice(-logits_to_keep, None) if logits_to_keep else slice(None)
        )
        return SimpleNamespace(logits=self.lm_head(hidden[:, slice_indices, :]))


class _WordTokenizer:
    """Whitespace tokenizer with stable ids, enough for the scorer contract."""

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 0
        self.pad_token = "<pad>"
        self.eos_token = "<pad>"
        self._ids: dict[str, int] = {}

    @property
    def vocab_size(self) -> int:
        return 128

    def _token_id(self, word: str) -> int:
        if word not in self._ids:
            # 0 is padding, so real tokens start at 1.
            self._ids[word] = 1 + len(self._ids)
        return self._ids[word]

    def __call__(
        self,
        texts,
        *,
        add_special_tokens: bool = True,
        return_tensors: str | None = None,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
    ):
        del add_special_tokens, return_tensors, padding, truncation
        if isinstance(texts, str):
            return {"input_ids": [self._token_id(w) for w in texts.split()]}
        limit = max_length or 16
        sequences = [
            [self._token_id(word) for word in text.split()][:limit] or [1]
            for text in texts
        ]
        width = max(len(sequence) for sequence in sequences)
        return {
            "input_ids": torch.tensor(
                [
                    sequence + [self.pad_token_id] * (width - len(sequence))
                    for sequence in sequences
                ],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [
                    [1] * len(sequence) + [0] * (width - len(sequence))
                    for sequence in sequences
                ],
                dtype=torch.long,
            ),
        }


_OPTIONS = (
    SteeredActionOption(action_id="alpha", surface_text="alpha"),
    SteeredActionOption(action_id="beta", surface_text="beta"),
    SteeredActionOption(action_id="gamma", surface_text="gamma"),
)

_TEXTS = ("route one segment two", "route three segment four")


def _build_model() -> _TinyCausalLM:
    torch.manual_seed(11)
    model = _TinyCausalLM(
        vocab_size=128, hidden_size=HIDDEN_SIZE, blocks=BLOCK_COUNT
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _build_scorer(
    *,
    model: _TinyCausalLM | None = None,
    joint_training: bool = False,
    action_options: tuple[SteeredActionOption, ...] = _OPTIONS,
    injection_layer_index: int = INJECTION_LAYER,
    control_norm_ratio: float = 0.25,
) -> TransformersSteeredActionScorer:
    resolved = model if model is not None else _build_model()
    return TransformersSteeredActionScorer(
        torch_module=torch,
        model=resolved,
        tokenizer=_WordTokenizer(),
        block_modules=tuple(resolved.blocks),
        final_norm_module=resolved.final_norm,
        injection_layer_index=injection_layer_index,
        hidden_size=HIDDEN_SIZE,
        device="cpu",
        model_id="tiny-test-lm",
        action_options=action_options,
        control_norm_ratio=control_norm_ratio,
        joint_training=joint_training,
    )


def test_frozen_arm_sends_gradient_to_the_control_delta_only() -> None:
    model = _build_model()
    scorer = _build_scorer(model=model)
    deltas = torch.full((2, HIDDEN_SIZE), 0.05, requires_grad=True)

    nll = scorer.action_nll(
        source_texts=_TEXTS, control_deltas=deltas, action_indices=(0, 2)
    )
    nll.sum().backward()

    assert nll.shape == (2,)
    assert nll.dtype is torch.float64
    assert deltas.grad is not None
    assert float(deltas.grad.abs().sum()) > 0.0
    assert all(parameter.grad is None for parameter in model.parameters())


def _make_scorer(
    *,
    model: _TinyCausalLM,
    tokenizer: _WordTokenizer,
    prefix_cache: bool,
    joint_training: bool = False,
) -> TransformersSteeredActionScorer:
    return TransformersSteeredActionScorer(
        torch_module=torch,
        model=model,
        tokenizer=tokenizer,
        block_modules=tuple(model.blocks),
        final_norm_module=model.final_norm,
        injection_layer_index=INJECTION_LAYER,
        hidden_size=HIDDEN_SIZE,
        device="cpu",
        model_id="tiny-test-lm",
        action_options=_OPTIONS,
        control_norm_ratio=0.25,
        joint_training=joint_training,
        prefix_cache=prefix_cache,
    )


def test_prefix_cache_matches_full_forward_frozen_arm() -> None:
    # The cached hot path must be numerically identical to the reference full
    # forward, across repeated updates that reuse the cached prefix.
    model = _build_model()
    tokenizer = _WordTokenizer()
    full = _make_scorer(model=model, tokenizer=tokenizer, prefix_cache=False)
    cached = _make_scorer(model=model, tokenizer=tokenizer, prefix_cache=True)

    torch.manual_seed(5)
    for _ in range(3):
        base = torch.randn(2, HIDDEN_SIZE)
        deltas_full = base.clone().requires_grad_(True)
        deltas_cached = base.clone().requires_grad_(True)

        nll_full = full.action_nll(
            source_texts=_TEXTS, control_deltas=deltas_full, action_indices=(0, 1)
        )
        nll_cached = cached.action_nll(
            source_texts=_TEXTS,
            control_deltas=deltas_cached,
            action_indices=(0, 1),
        )
        assert torch.allclose(nll_full, nll_cached, atol=1e-6)

        nll_full.sum().backward()
        nll_cached.sum().backward()
        assert deltas_full.grad is not None
        assert deltas_cached.grad is not None
        assert torch.allclose(deltas_full.grad, deltas_cached.grad, atol=1e-6)

    # One prefix entry per token surface, reused across the three updates.
    assert cached.prefix_cache_enabled
    assert len(cached._prefix_cache) == 1


def test_prefix_cache_matches_full_forward_joint_arm() -> None:
    # Separate but identical models so upper-block gradients stay independent.
    model_full = _build_model()
    model_cached = _build_model()
    tokenizer = _WordTokenizer()
    full = _make_scorer(
        model=model_full,
        tokenizer=tokenizer,
        prefix_cache=False,
        joint_training=True,
    )
    cached = _make_scorer(
        model=model_cached,
        tokenizer=tokenizer,
        prefix_cache=True,
        joint_training=True,
    )

    torch.manual_seed(7)
    base = torch.randn(2, HIDDEN_SIZE)
    deltas_full = base.clone().requires_grad_(True)
    deltas_cached = base.clone().requires_grad_(True)

    nll_full = full.action_nll(
        source_texts=_TEXTS, control_deltas=deltas_full, action_indices=(0, 2)
    )
    nll_cached = cached.action_nll(
        source_texts=_TEXTS, control_deltas=deltas_cached, action_indices=(0, 2)
    )
    assert torch.allclose(nll_full, nll_cached, atol=1e-6)

    nll_full.sum().backward()
    nll_cached.sum().backward()

    # Gradient reaches the trainable upper blocks identically. INJECTION_LAYER
    # is 1, so blocks 2 and 3 plus the final norm are the joint parameters.
    for index in (2, 3):
        grad_full = model_full.blocks[index].linear.weight.grad
        grad_cached = model_cached.blocks[index].linear.weight.grad
        assert grad_full is not None
        assert grad_cached is not None
        assert torch.allclose(grad_full, grad_cached, atol=1e-6)
    # Lower (frozen) blocks below the injection point never receive gradient.
    assert model_cached.blocks[0].linear.weight.grad is None


def test_prefix_cache_baseline_matches_full_forward() -> None:
    model = _build_model()
    tokenizer = _WordTokenizer()
    full = _make_scorer(model=model, tokenizer=tokenizer, prefix_cache=False)
    cached = _make_scorer(model=model, tokenizer=tokenizer, prefix_cache=True)

    baseline_full = full.baseline_action_nll(
        source_texts=_TEXTS, action_indices=(0, 1)
    )
    baseline_cached = cached.baseline_action_nll(
        source_texts=_TEXTS, action_indices=(0, 1)
    )
    for got, expected in zip(baseline_cached, baseline_full, strict=True):
        assert abs(got - expected) < 1e-6


def test_controlled_readout_matches_autograd_path_and_cache_can_be_released() -> None:
    model = _build_model()
    tokenizer = _WordTokenizer()
    scorer = _make_scorer(model=model, tokenizer=tokenizer, prefix_cache=True)
    deltas = torch.full((2, HIDDEN_SIZE), 0.05)

    differentiable = scorer.action_nll(
        source_texts=_TEXTS,
        control_deltas=deltas.clone().requires_grad_(True),
        action_indices=(0, 2),
    )
    controlled = scorer.controlled_action_nll(
        source_texts=_TEXTS,
        control_deltas=deltas,
        action_indices=(0, 2),
    )

    assert tuple(float(value) for value in differentiable.detach()) == pytest.approx(
        controlled,
        abs=1e-6,
    )
    assert len(scorer._prefix_cache) == 1
    scorer.clear_prefix_cache()
    assert scorer._prefix_cache == {}


def test_frozen_arm_refuses_to_score_a_thawed_base_parameter() -> None:
    model = _build_model()
    scorer = _build_scorer(model=model)
    model.blocks[3].linear.weight.requires_grad_(True)

    with pytest.raises(RuntimeError, match="R2 frozen basis"):
        scorer.action_nll(
            source_texts=_TEXTS,
            control_deltas=torch.zeros((2, HIDDEN_SIZE)),
            action_indices=(0, 1),
        )


def test_injected_delta_is_capped_at_the_probe_derived_norm() -> None:
    scorer = _build_scorer()
    direction = torch.zeros((2, HIDDEN_SIZE))
    direction[:, 0] = 1.0
    at_cap = direction * scorer.control_norm_cap
    far_past_cap = direction * (scorer.control_norm_cap * 1_000.0)

    nll_at_cap = scorer.action_nll(
        source_texts=_TEXTS, control_deltas=at_cap, action_indices=(0, 1)
    )
    nll_past_cap = scorer.action_nll(
        source_texts=_TEXTS,
        control_deltas=far_past_cap,
        action_indices=(0, 1),
    )

    assert scorer.control_norm_cap > 0.0
    assert torch.allclose(nll_at_cap, nll_past_cap, atol=1e-6)


def test_baseline_nll_is_detached_and_finite() -> None:
    model = _build_model()
    scorer = _build_scorer(model=model)

    baseline = scorer.baseline_action_nll(
        source_texts=_TEXTS, action_indices=(0, 1)
    )

    assert len(baseline) == 2
    assert all(value == value and value < float("inf") for value in baseline)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_candidate_projection_matches_full_logits_restricted_softmax() -> None:
    # The scorer projects only the candidate rows of the output embedding from
    # the captured post-norm hidden state instead of materialising the full
    # [batch, seq, vocab] logits. This must be numerically identical to the
    # pre-refactor path: full logits at the last real token, then a restricted
    # softmax over the candidate action tokens. Both computations must share
    # one tokenizer instance so lazily-assigned token ids line up.
    model = _build_model()
    tokenizer = _WordTokenizer()
    scorer = TransformersSteeredActionScorer(
        torch_module=torch,
        model=model,
        tokenizer=tokenizer,
        block_modules=tuple(model.blocks),
        final_norm_module=model.final_norm,
        injection_layer_index=INJECTION_LAYER,
        hidden_size=HIDDEN_SIZE,
        device="cpu",
        model_id="tiny-test-lm",
        action_options=_OPTIONS,
        control_norm_ratio=0.25,
    )
    action_indices = (0, 2)

    scorer_nll = scorer.baseline_action_nll(
        source_texts=_TEXTS, action_indices=action_indices
    )

    prompt_suffix = "\nNext move:"
    candidate_ids = [
        tokenizer(" " + option.surface_text, add_special_tokens=False)[
            "input_ids"
        ][0]
        for option in _OPTIONS
    ]
    encoded = tokenizer(
        [text + prompt_suffix for text in _TEXTS],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=96,
    )
    with torch.no_grad():
        full_logits = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        ).logits
    lengths = encoded["attention_mask"].sum(dim=-1) - 1
    rows = torch.arange(len(_TEXTS))
    last_logits = full_logits[rows, lengths]
    candidate_logits = last_logits[:, torch.tensor(candidate_ids)]
    log_probs = torch.log_softmax(candidate_logits.to(torch.float32), dim=-1)
    reference = [
        -float(log_probs[index, action_indices[index]])
        for index in range(len(_TEXTS))
    ]

    assert len(scorer_nll) == len(reference)
    for got, expected in zip(scorer_nll, reference, strict=True):
        assert abs(got - expected) < 1e-6


def test_joint_arm_thaws_only_the_blocks_above_the_injection_layer() -> None:
    model = _build_model()
    scorer = _build_scorer(model=model, joint_training=True)

    trainable = {id(parameter) for parameter in scorer.trainable_parameters()}
    expected = {
        id(parameter)
        for index in range(INJECTION_LAYER + 1, BLOCK_COUNT)
        for parameter in model.blocks[index].parameters()
    } | {id(parameter) for parameter in model.final_norm.parameters()}
    below = {
        id(parameter)
        for index in range(INJECTION_LAYER + 1)
        for parameter in model.blocks[index].parameters()
    }

    assert trainable == expected
    assert not (trainable & below)
    assert all(
        not parameter.requires_grad for parameter in model.lm_head.parameters()
    )


def test_joint_arm_reset_restores_pristine_weights() -> None:
    model = _build_model()
    scorer = _build_scorer(model=model, joint_training=True)
    pristine = model.blocks[3].linear.weight.detach().clone()
    with torch.no_grad():
        model.blocks[3].linear.weight.add_(1.0)
    assert not torch.allclose(model.blocks[3].linear.weight, pristine)

    scorer.reset_joint_parameters()

    assert torch.allclose(model.blocks[3].linear.weight, pristine)


def test_restore_and_freeze_returns_the_shared_model_to_the_frozen_contract() -> None:
    model = _build_model()
    scorer = _build_scorer(model=model, joint_training=True)
    pristine = model.final_norm.weight.detach().clone()
    with torch.no_grad():
        model.final_norm.weight.add_(0.5)

    scorer.restore_and_freeze()

    assert scorer.joint_training is False
    assert scorer.trainable_parameters() == ()
    assert torch.allclose(model.final_norm.weight, pristine)
    assert all(
        not parameter.requires_grad for parameter in model.parameters()
    )


def test_joint_only_controls_reject_a_frozen_scorer() -> None:
    scorer = _build_scorer()

    with pytest.raises(RuntimeError, match="only valid on a joint-training"):
        scorer.reset_joint_parameters()
    with pytest.raises(RuntimeError, match="only valid on a joint-training"):
        scorer.restore_and_freeze()


def test_action_index_lookup_fails_loudly_on_an_unknown_action() -> None:
    scorer = _build_scorer()

    assert scorer.action_index("beta") == 1
    with pytest.raises(KeyError, match="Unknown action_id"):
        scorer.action_index("delta")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        (
            {"action_options": (_OPTIONS[0],)},
            "at least two action options",
        ),
        (
            {
                "action_options": (
                    SteeredActionOption(action_id="a", surface_text="alpha"),
                    SteeredActionOption(action_id="a", surface_text="beta"),
                )
            },
            "duplicate action_id",
        ),
        (
            {
                "action_options": (
                    SteeredActionOption(action_id="a", surface_text="alpha one"),
                    SteeredActionOption(action_id="b", surface_text="alpha two"),
                )
            },
            "first tokens collide",
        ),
        ({"injection_layer_index": BLOCK_COUNT}, "out of range"),
        ({"control_norm_ratio": 0.0}, "control_norm_ratio must be in"),
        ({"control_norm_ratio": 2.5}, "control_norm_ratio must be in"),
    ),
)
def test_scorer_construction_rejects_invalid_configuration(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _build_scorer(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("source_texts", "delta_shape", "action_indices", "message"),
    (
        ((), (0, HIDDEN_SIZE), (), "source_texts must be non-empty"),
        (_TEXTS, (2, HIDDEN_SIZE), (0,), "for 1 action indices"),
        (_TEXTS, (2, HIDDEN_SIZE + 1), (0, 1), "control_deltas must be"),
        (_TEXTS, (3, HIDDEN_SIZE), (0, 1), "control_deltas must be"),
        (_TEXTS, (2, HIDDEN_SIZE), (0, 7), "outside the 3-way action"),
    ),
)
def test_scoring_rejects_shape_and_vocabulary_mismatches(
    source_texts: tuple[str, ...],
    delta_shape: tuple[int, int],
    action_indices: tuple[int, ...],
    message: str,
) -> None:
    scorer = _build_scorer()

    with pytest.raises(ValueError, match=message):
        scorer.action_nll(
            source_texts=source_texts,
            control_deltas=torch.zeros(delta_shape),
            action_indices=action_indices,
        )
