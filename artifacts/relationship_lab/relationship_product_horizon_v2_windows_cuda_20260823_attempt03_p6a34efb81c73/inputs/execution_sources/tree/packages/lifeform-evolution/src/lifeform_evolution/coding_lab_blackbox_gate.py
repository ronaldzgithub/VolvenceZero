"""Packet 2.5：层 A 黑盒择时 gate（Learnable 轴不依赖白盒的直接检验）。

从 coding-lab 轨迹日志离线构建 (协议状态, 下一步动作类) → episode 终局
的 bandit 表，用 REINFORCE 学一个 softmax gate。三条结构约束：

1. **特征全黑盒**：只用 junction owner 发布的结构化协议状态
   （category / reads_bucket / has_edited / test_state），不碰残差、
   不碰白盒模型；
2. **信用只来自 episode 终局**：reward 是该 (状态, 动作) 单元下真实
   分支的 oracle 通过率——oracle 是环境结算，不是 evaluation/judge
   readout（R12 无泄漏）；
3. **只有策略在学**：bandit 表冻结后训练期间不再更新，动作采样掩码
   到有观测支撑的动作上（无支撑的动作没有反事实证据，不许乐观外推）。

判词：held-out 状态键上，学到的 gate 的期望通过率显著高于
uniform-over-supported 基线（bootstrap CI 下界 > 0），并报告 vs 全局
众数动作基线。语料不足如实 FAIL。
"""

from __future__ import annotations

import hashlib
import statistics

from collections import Counter, defaultdict
from dataclasses import dataclass
from random import Random
from typing import Any

from lifeform_domain_coding.lab.junctions import JUNCTION_ACTIONS, JunctionRecord

_CATEGORY_BUCKETS = 8
FEATURE_DIM = _CATEGORY_BUCKETS + 1 + 1 + 3
_TEST_STATES = ("none", "failed", "passed")


@dataclass(frozen=True)
class BanditCell:
    state_key: str
    action: str
    trials: int
    passes: int

    @property
    def pass_rate(self) -> float:
        return self.passes / self.trials


@dataclass(frozen=True)
class BlackboxGateVerdict:
    train_state_keys: int
    eval_state_keys: int
    policy_expected_pass: float
    uniform_expected_pass: float
    modal_expected_pass: float
    uplift_vs_uniform: float
    uplift_ci_lower_5pct: float
    beats_uniform: bool
    restarts: int
    selected_restart: int


def build_bandit_table(
    records: tuple[JunctionRecord, ...],
) -> dict[str, tuple[BanditCell, ...]]:
    """Freeze (state, action) → terminal-outcome cells from logs."""

    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    features_seen: dict[str, JunctionRecord] = {}
    for record in records:
        if record.action_taken not in JUNCTION_ACTIONS:
            continue
        cell = counts[(record.state_key, record.action_taken)]
        cell[0] += 1
        cell[1] += int(record.episode_passed)
        features_seen.setdefault(record.state_key, record)
    table: dict[str, list[BanditCell]] = defaultdict(list)
    for (state_key, action), (trials, passes) in sorted(counts.items()):
        table[state_key].append(
            BanditCell(state_key=state_key, action=action, trials=trials, passes=passes)
        )
    return {key: tuple(cells) for key, cells in table.items()}


def featurize(record: JunctionRecord) -> tuple[float, ...]:
    """Blackbox protocol-state features; no text semantics, no residuals."""

    digest = hashlib.sha256(record.category.encode("utf-8")).digest()
    bucket = digest[0] % _CATEGORY_BUCKETS
    category_onehot = [0.0] * _CATEGORY_BUCKETS
    category_onehot[bucket] = 1.0
    test_onehot = [0.0] * len(_TEST_STATES)
    test_onehot[_TEST_STATES.index(record.test_state)] = 1.0
    return (
        *category_onehot,
        record.reads_bucket / 3.0,
        1.0 if record.has_edited else 0.0,
        *test_onehot,
    )


def state_features(
    records: tuple[JunctionRecord, ...],
) -> dict[str, tuple[float, ...]]:
    features: dict[str, tuple[float, ...]] = {}
    for record in records:
        if record.state_key not in features:
            features[record.state_key] = featurize(record)
    return features


def split_state_keys(
    table: dict[str, tuple[BanditCell, ...]],
    *,
    eval_fraction: float = 0.3,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Deterministic content-addressed split, mirroring the corpus split."""

    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be in (0, 1)")
    threshold = int(eval_fraction * 2**32)
    train: list[str] = []
    evaluation: list[str] = []
    for key in sorted(table):
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big")
        (evaluation if bucket < threshold else train).append(key)
    return tuple(train), tuple(evaluation)


def _expected_pass(
    policy_probs: dict[str, float], cells: tuple[BanditCell, ...]
) -> float:
    """Probability-weighted terminal pass rate over supported actions."""

    supported = {cell.action: cell.pass_rate for cell in cells}
    mass = sum(policy_probs.get(action, 0.0) for action in supported)
    if mass <= 0.0:
        return statistics.fmean(supported.values())
    return (
        sum(policy_probs.get(action, 0.0) * rate for action, rate in supported.items())
        / mass
    )


def _uniform_pass(cells: tuple[BanditCell, ...]) -> float:
    return statistics.fmean(cell.pass_rate for cell in cells)


def _modal_action(table: dict[str, tuple[BanditCell, ...]], keys: tuple[str, ...]) -> str:
    counter: Counter[str] = Counter()
    for key in keys:
        for cell in table[key]:
            counter[cell.action] += cell.trials
    return counter.most_common(1)[0][0]


def _modal_pass(cells: tuple[BanditCell, ...], modal: str) -> float:
    supported = {cell.action: cell.pass_rate for cell in cells}
    if modal in supported:
        return supported[modal]
    return statistics.fmean(supported.values())


def _policy_probs_for_state(
    policy: Any,
    torch: Any,
    features: tuple[float, ...],
    cells: tuple[BanditCell, ...],
) -> dict[str, float]:
    feature_tensor = torch.tensor([features], dtype=torch.float32)
    logits = policy.logits(feature_tensor)[0]
    supported_indices = [
        JUNCTION_ACTIONS.index(cell.action) for cell in cells
    ]
    mask = torch.full((len(JUNCTION_ACTIONS),), float("-inf"))
    for index in supported_indices:
        mask[index] = 0.0
    probs = torch.softmax(logits + mask, dim=-1)
    return {
        action: float(probs[index])
        for index, action in enumerate(JUNCTION_ACTIONS)
        if index in supported_indices
    }


class _BlackboxGatePolicy:
    """Softmax gate over blackbox protocol features (S3-E `_GatePolicy` 骨架).

    No free bias toward any action: the bias vector stays zero and only
    the weight matrix learns, so every preference is earned from
    terminal credit.
    """

    def __init__(self, *, torch: Any, feature_dim: int, n_actions: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self._torch = torch
        self._W = (
            torch.randn(feature_dim, n_actions, generator=generator) * 0.01
        ).requires_grad_(True)

    def parameters(self) -> tuple[Any, ...]:
        return (self._W,)

    def logits(self, features: Any) -> Any:
        return features @ self._W


def _train_policy(
    *,
    torch: Any,
    table: dict[str, tuple[BanditCell, ...]],
    features: dict[str, tuple[float, ...]],
    train_keys: tuple[str, ...],
    seed: int,
    updates: int,
    batch_size: int,
    learning_rate: float,
    entropy_coef: float,
) -> tuple[_BlackboxGatePolicy, float]:
    policy = _BlackboxGatePolicy(
        torch=torch,
        feature_dim=FEATURE_DIM,
        n_actions=len(JUNCTION_ACTIONS),
        seed=seed,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    rng = Random(seed)
    reward_baseline = 0.0
    for update in range(updates):
        losses = []
        entropies = []
        for _ in range(batch_size):
            key = train_keys[rng.randrange(len(train_keys))]
            cells = table[key]
            feature_tensor = torch.tensor([features[key]], dtype=torch.float32)
            logits = policy.logits(feature_tensor)[0]
            mask = torch.full((len(JUNCTION_ACTIONS),), float("-inf"))
            index_by_action = {}
            for cell in cells:
                action_index = JUNCTION_ACTIONS.index(cell.action)
                mask[action_index] = 0.0
                index_by_action[action_index] = cell
            log_probs = torch.log_softmax(logits + mask, dim=-1)
            probs = log_probs.exp()
            sampled = int(torch.multinomial(probs, 1).item())
            reward = index_by_action[sampled].pass_rate
            advantage = reward - reward_baseline
            reward_baseline = 0.98 * reward_baseline + 0.02 * reward
            losses.append(-log_probs[sampled] * advantage)
            finite = probs[probs > 0]
            entropies.append(-(finite * finite.log()).sum())
        loss = torch.stack(losses).mean() - entropy_coef * torch.stack(entropies).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del update
    train_expected = statistics.fmean(
        _expected_pass(
            _policy_probs_for_state(policy, torch, features[key], table[key]),
            table[key],
        )
        for key in train_keys
    )
    return policy, train_expected


def fit_and_judge_blackbox_gate(
    records: tuple[JunctionRecord, ...],
    *,
    seed: int = 20260813,
    restarts: int = 4,
    updates: int = 300,
    batch_size: int = 16,
    learning_rate: float = 0.05,
    entropy_coef: float = 0.01,
    eval_fraction: float = 0.3,
    bootstrap_samples: int = 2000,
    min_train_keys: int = 8,
    min_eval_keys: int = 4,
) -> BlackboxGateVerdict:
    """Offline-bandit REINFORCE fit + held-out verdict (fail loudly if thin)."""

    import torch  # noqa: PLC0415 — heavyweight optional dependency

    table = build_bandit_table(records)
    features = state_features(records)
    train_keys, eval_keys = split_state_keys(table, eval_fraction=eval_fraction)
    if len(train_keys) < min_train_keys or len(eval_keys) < min_eval_keys:
        raise ValueError(
            f"bandit table too thin: {len(train_keys)} train / "
            f"{len(eval_keys)} eval state keys (need >= {min_train_keys}/"
            f"{min_eval_keys}); collect more trajectories"
        )

    best_policy: _BlackboxGatePolicy | None = None
    best_train = float("-inf")
    selected_restart = 0
    for restart in range(restarts):
        policy, train_expected = _train_policy(
            torch=torch,
            table=table,
            features=features,
            train_keys=train_keys,
            seed=seed * 1000 + restart,
            updates=updates,
            batch_size=batch_size,
            learning_rate=learning_rate,
            entropy_coef=entropy_coef,
        )
        if train_expected > best_train:
            best_train = train_expected
            best_policy = policy
            selected_restart = restart
    assert best_policy is not None

    modal = _modal_action(table, train_keys)
    policy_scores: list[float] = []
    uniform_scores: list[float] = []
    modal_scores: list[float] = []
    for key in eval_keys:
        cells = table[key]
        probs = _policy_probs_for_state(best_policy, torch, features[key], cells)
        policy_scores.append(_expected_pass(probs, cells))
        uniform_scores.append(_uniform_pass(cells))
        modal_scores.append(_modal_pass(cells, modal))

    uplifts = [p - u for p, u in zip(policy_scores, uniform_scores, strict=True)]
    rng = Random(seed)
    means: list[float] = []
    for _ in range(bootstrap_samples):
        draw = [uplifts[rng.randrange(len(uplifts))] for _ in uplifts]
        means.append(statistics.fmean(draw))
    means.sort()
    ci_lower = means[int(0.05 * len(means))]

    return BlackboxGateVerdict(
        train_state_keys=len(train_keys),
        eval_state_keys=len(eval_keys),
        policy_expected_pass=round(statistics.fmean(policy_scores), 4),
        uniform_expected_pass=round(statistics.fmean(uniform_scores), 4),
        modal_expected_pass=round(statistics.fmean(modal_scores), 4),
        uplift_vs_uniform=round(statistics.fmean(uplifts), 4),
        uplift_ci_lower_5pct=round(ci_lower, 4),
        beats_uniform=statistics.fmean(uplifts) > 0 and ci_lower > 0,
        restarts=restarts,
        selected_restart=selected_restart,
    )


__all__ = [
    "FEATURE_DIM",
    "BanditCell",
    "BlackboxGateVerdict",
    "build_bandit_table",
    "featurize",
    "fit_and_judge_blackbox_gate",
    "split_state_keys",
    "state_features",
]
