"""Held-out discipline for the State-KV prefix distillation material.

The prefix generator is fit against the frozen substrate's own text arm. That
is only evidence about a carrier if the evaluation material was never part of
the fit. These tests pin the two hold-outs the trainer claims in its manifest,
so a later edit to either pool cannot quietly turn the P3 verdict into a
measurement of memorisation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "train_state_kv_prefix.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "train_state_kv_prefix", _SCRIPT_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load the prefix trainer from {_SCRIPT_PATH}")
_TRAINER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TRAINER)

PERSONAS = _TRAINER.PERSONAS
P2_PROBE_SENTENCES = _TRAINER.P2_PROBE_SENTENCES
PROBE_SENTENCES = _TRAINER.PROBE_SENTENCES
TRAIN_AXIS_LIMIT = _TRAINER.TRAIN_AXIS_LIMIT
TRAIN_PROBE_SENTENCES = _TRAINER.TRAIN_PROBE_SENTENCES


def test_training_probes_are_disjoint_from_evaluation_probes() -> None:
    evaluation = {
        sentence
        for _, sentence in (*PROBE_SENTENCES, *P2_PROBE_SENTENCES)
    }

    assert evaluation.isdisjoint(TRAIN_PROBE_SENTENCES)
    _TRAINER._assert_probe_holdout()


def test_probe_holdout_check_actually_fires(monkeypatch) -> None:
    stolen = PROBE_SENTENCES[0][1]
    monkeypatch.setattr(
        _TRAINER, "TRAIN_PROBE_SENTENCES", (*TRAIN_PROBE_SENTENCES, stolen)
    )

    with pytest.raises(ValueError, match="overlap"):
        _TRAINER._assert_probe_holdout()


def test_p2_probe_holdout_check_actually_fires(monkeypatch) -> None:
    stolen = P2_PROBE_SENTENCES[0][1]
    monkeypatch.setattr(
        _TRAINER, "TRAIN_PROBE_SENTENCES", (*TRAIN_PROBE_SENTENCES, stolen)
    )

    with pytest.raises(ValueError, match="overlap"):
        _TRAINER._assert_probe_holdout()


def _primary_projection(state, centre, primary) -> float:
    """Least-squares coordinate of ``state`` along the persona axis.

    ``u = +-1`` is exactly an evaluation persona, so this is the number the
    training envelope has to stay inside.
    """

    energy = sum(value * value for value in primary)
    return (
        sum(
            (value - c) * p
            for value, c, p in zip(state, centre, primary, strict=True)
        )
        / energy
    )


def test_sampled_states_stay_inside_the_persona_envelope() -> None:
    centre, primary, secondary = _TRAINER._persona_axes()
    states = _TRAINER._sample_states(count=256, seed=11)

    assert len(states) == 256
    for state in states:
        assert all(0.0 <= value <= 1.0 for value in state)
    # The second factor must not leak into the first, or a state drawn at
    # |u| <= 0.8 could still project onto an evaluation persona.
    assert sum(
        s * p for s, p in zip(secondary, primary, strict=True)
    ) == pytest.approx(0.0, abs=1e-9)
    for state in states:
        assert abs(_primary_projection(state, centre, primary)) < 1.0


def test_evaluation_personas_are_not_in_the_training_set() -> None:
    states = set(_TRAINER._sample_states(count=256, seed=11))

    for _, vector, _, _ in PERSONAS:
        assert vector not in states


def test_persona_axes_span_the_evaluation_pair() -> None:
    centre, primary, secondary = _TRAINER._persona_axes()
    vector_a, vector_b = PERSONAS[0][1], PERSONAS[1][1]

    reconstructed_a = tuple(c + p for c, p in zip(centre, primary, strict=True))
    reconstructed_b = tuple(c - p for c, p in zip(centre, primary, strict=True))

    assert reconstructed_a == pytest.approx(vector_a)
    assert reconstructed_b == pytest.approx(vector_b)
    # The second axis must not be a copy of the first, or boundary risk could
    # never vary independently of relationship distress in the material.
    assert secondary != primary


def test_training_axis_limit_leaves_a_real_margin() -> None:
    assert 0.0 < TRAIN_AXIS_LIMIT < 1.0


def test_route_targets_are_deterministic_state_sensitive_distributions() -> None:
    torch = pytest.importorskip("torch")
    state_a = torch.tensor(PERSONAS[0][1], dtype=torch.float32)
    state_b = torch.tensor(PERSONAS[1][1], dtype=torch.float32)

    target_a = _TRAINER._slot_route_target(
        torch=torch, state=state_a, slots=4, temperature=0.18
    )
    target_again = _TRAINER._slot_route_target(
        torch=torch, state=state_a, slots=4, temperature=0.18
    )
    target_b = _TRAINER._slot_route_target(
        torch=torch, state=state_b, slots=4, temperature=0.18
    )

    assert float(target_a.sum()) == pytest.approx(1.0)
    assert torch.all(target_a > 0.0)
    assert torch.equal(target_a, target_again)
    assert not torch.allclose(target_a, target_b)


def test_route_loss_rewards_attention_matching_the_state_route() -> None:
    torch = pytest.importorskip("torch")
    target = torch.tensor([0.8, 0.15, 0.05], dtype=torch.float32)

    def attention(slot_distribution):
        row = torch.tensor(
            [*slot_distribution, 0.2, 0.1], dtype=torch.float32
        )
        row = row / row.sum()
        return row.reshape(1, 1, 1, 5).repeat(1, 2, 1, 1)

    matched = _TRAINER._slot_attention_route_loss_from_attentions(
        torch=torch,
        attentions=(attention([0.8, 0.15, 0.05]),),
        slots=3,
        target=target,
    )
    uniform = _TRAINER._slot_attention_route_loss_from_attentions(
        torch=torch,
        attentions=(attention([1.0, 1.0, 1.0]),),
        slots=3,
        target=target,
    )
    wrong = _TRAINER._slot_attention_route_loss_from_attentions(
        torch=torch,
        attentions=(attention([0.05, 0.15, 0.8]),),
        slots=3,
        target=target,
    )

    assert float(matched) < float(uniform)
    assert float(uniform) < float(wrong)


def test_routed_attention_objective_is_enabled_by_default() -> None:
    assert _TRAINER.DEFAULT_ROUTE_WEIGHT > 0.0
    assert _TRAINER.DEFAULT_ROUTE_TEMPERATURE > 0.0


def test_state_strategy_target_is_legible_and_state_sensitive() -> None:
    persona_a = _TRAINER._state_strategy_target(
        state=PERSONAS[0][1], probe="今天特别累"
    )
    persona_b = _TRAINER._state_strategy_target(
        state=PERSONAS[1][1], probe="今天特别累"
    )

    assert persona_a != persona_b
    assert "压力很高" in persona_a
    assert "控制感偏低" in persona_a
    assert "稳定" in persona_b
    assert "决策准备度" in persona_b


def test_state_strategy_is_the_default_training_target() -> None:
    assert _TRAINER.TARGET_SOURCE_STATE_STRATEGY in _TRAINER.TARGET_SOURCES
    assert _TRAINER.TARGET_SOURCE_TEACHER in _TRAINER.TARGET_SOURCES
