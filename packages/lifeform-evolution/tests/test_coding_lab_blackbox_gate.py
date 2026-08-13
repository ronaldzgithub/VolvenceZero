"""Packet 2.5 黑盒择时 gate 机制测试。

用合成 JunctionRecord 构造一个有真实结构的 bandit 表（某类状态下
investigate 优于 edit），验证：表构建、特征、split、以及 REINFORCE
gate 能在 held-out 键上恢复该结构并过判词；同时覆盖语料太薄的
fail-loudly 路径。
"""

from __future__ import annotations

import pytest

from lifeform_domain_coding.lab.junctions import JunctionRecord
from lifeform_evolution.coding_lab_blackbox_gate import (
    FEATURE_DIM,
    build_bandit_table,
    featurize,
    fit_and_judge_blackbox_gate,
    split_state_keys,
)


def _record(
    *,
    category: str,
    reads_bucket: int,
    has_edited: bool,
    test_state: str,
    action: str,
    passed: bool,
    index: int,
) -> JunctionRecord:
    key = (
        f"{category}|reads={reads_bucket}|edited={int(has_edited)}|"
        f"tests={test_state}"
    )
    return JunctionRecord(
        junction_id=f"j{index:04d}",
        state_key=key,
        state_text=f"[coding junction] {key}",
        action_taken=action,
        episode_passed=passed,
        decisions_to_end=1,
        provenance=f"fixture#{index}",
        trajectory_sha256=f"{index:064d}",
        category=category,
        reads_bucket=reads_bucket,
        has_edited=has_edited,
        test_state=test_state,
    )


def _structured_corpus() -> tuple[JunctionRecord, ...]:
    """20 state keys; at reads=0 states investigate wins, edit loses."""

    records: list[JunctionRecord] = []
    index = 0
    for cat_index in range(20):
        category = f"cat{cat_index:02d}"
        for copy in range(4):
            records.append(
                _record(
                    category=category,
                    reads_bucket=0,
                    has_edited=False,
                    test_state="none",
                    action="investigate",
                    passed=copy < 3,  # 75% pass
                    index=index,
                )
            )
            index += 1
            records.append(
                _record(
                    category=category,
                    reads_bucket=0,
                    has_edited=False,
                    test_state="none",
                    action="edit",
                    passed=copy < 1,  # 25% pass
                    index=index,
                )
            )
            index += 1
    return tuple(records)


def test_bandit_table_counts_terminal_outcomes() -> None:
    records = _structured_corpus()
    table = build_bandit_table(records)
    assert len(table) == 20
    cells = {cell.action: cell for cell in next(iter(table.values()))}
    assert cells["investigate"].trials == 4
    assert cells["investigate"].pass_rate == pytest.approx(0.75)
    assert cells["edit"].pass_rate == pytest.approx(0.25)


def test_featurize_shape_and_determinism() -> None:
    record = _structured_corpus()[0]
    features = featurize(record)
    assert len(features) == FEATURE_DIM
    assert features == featurize(record)
    assert sum(features[:8]) == pytest.approx(1.0)  # one category bucket


def test_split_partitions_deterministically() -> None:
    table = build_bandit_table(_structured_corpus())
    train_a, eval_a = split_state_keys(table)
    train_b, eval_b = split_state_keys(table)
    assert train_a == train_b and eval_a == eval_b
    assert len(train_a) + len(eval_a) == len(table)
    assert train_a and eval_a


def test_gate_learns_structure_and_passes_verdict() -> None:
    verdict = fit_and_judge_blackbox_gate(
        _structured_corpus(),
        seed=7,
        restarts=2,
        updates=200,
        bootstrap_samples=500,
    )
    # investigate 优于 edit 的结构在所有状态键上一致，held-out 上
    # 学到的 gate 必须显著超过 uniform（uniform = 0.5 期望）。
    assert verdict.beats_uniform
    assert verdict.policy_expected_pass > verdict.uniform_expected_pass
    assert verdict.uplift_ci_lower_5pct > 0


def test_thin_corpus_fails_loudly() -> None:
    records = _structured_corpus()[:8]  # 一两个状态键
    with pytest.raises(ValueError, match="bandit table too thin"):
        fit_and_judge_blackbox_gate(records, restarts=1, updates=10)
