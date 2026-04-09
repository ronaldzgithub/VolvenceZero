# U01 — Phase 1: RL 闭环打通

> Status: draft
> Last updated: 2026-04-09
> Phase owner: `internal_rl` + `joint_loop`
> 差距族: A（RL 策略优化器缺失）
> 影响需求: R4, R9, R13, R14
> 前置条件: P00–P09 收敛包完成
> 预估周期: 2–3 周

## 1. Phase 目标

让 Internal RL 从"能收集 rollout 和计算 reward"变为"能更新策略参数并观测到行为改善"。

这是整个提升计划的**最高优先级**——没有策略优化器，系统就是一个只能观察但无法学习的骨架。SSL-RL 交替只走了 SSL 半程（R13），内部控制无法真正学习（R4），信用信号无法驱动参数更新（R9），regime 选择无法通过 RL 训练改进（R14）。

## 2. 设计依据

### 2.1 ETA Internal RL 算法（Appendix B.5）

```
观测 = 残差流激活 e_{t,l}
动作 = 控制器代码 z_t（而非 token a_t）
环境 = 原始环境 + 冻结自回归模型 + metacontroller 的解码器和切换单元
```

1. 自监督训练 metacontroller（Eq.3），学习切换单元和有意义的控制器代码空间 ← **当前已有**
2. 丢弃非因果编码器，替换为因果抽象动作策略 π(z_t | e_{1:t}) ← **`CausalZPolicy` 已有**
3. 二值化切换门 ← **`beta_binary` 已有**
4. 用 RL（PPO/GRPO）训练 π，保持其他模块冻结 ← **缺失，本 Phase 核心交付**

### 2.2 当前代码基础

已有可复用的基础设施：

- `InternalRLSandbox.run_rollout()` → 收集 `ZRollout`（含 `ZTransition` 序列）
- `ZTransition` 已含 `log_prob`、`reward`、`policy_score`
- `CausalZPolicy` 已有 `step()` 和 `export_parameters()`
- `OptimizationReport` 已定义了 `average_reward`、`baseline_reward`、`mean_advantage`、`surrogate_objective`、`clip_fraction`、`kl_penalty`
- `DualTrackOptimizationReport` 分离 task/relationship
- `ETANLJointLoop.run_joint_cycle()` 已调用 `_sandbox.run_rollout()` 并产出 `JointCycleReport`

**缺口**：`OptimizationReport` 的字段目前由 `_compute_optimization_report()` 用统计量填充，但没有真正的策略梯度计算和参数更新。

## 3. 步骤分解

### U01.1 — Z-space 策略优化器

**owner**: `internal_rl`
**位置**: 新增 `volvence_zero/internal_rl/optimizer.py`
**修改**: `volvence_zero/internal_rl/sandbox.py`

**内容**：

1. 实现 `ZSpacePolicyOptimizer` 类，接收 `ZRollout` 序列，输出参数更新
2. 优化算法：PPO-clip 变体
   - 从 `ZTransition` 中提取 `(observation_signature, latent_code, log_prob, reward)`
   - 计算 GAE（Generalized Advantage Estimation）
   - PPO surrogate objective：`L_clip = min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)`
   - KL penalty：`D_KL(π_new || π_old)` 作为软约束
3. 参数更新作用目标：
   - `MetacontrollerParameterStore.track_weights`（每 track 独立权重）
   - `MetacontrollerParameterStore.persistence`
   - `MetacontrollerParameterStore.learning_rate`
4. 支持 mini-batch 采样：从多条 rollout 中采样 transition batch
5. 更新后导出 `CausalPolicyParameters` 并生成 `SelfModificationRecord`

**约束**：

- clip ratio ε ≤ 0.2
- 单步 KL 超过 `max_kl` 时跳过更新
- 参数变动幅度受 `ModificationGate.ONLINE` 约束
- 每次优化后自动保存 `CausalPolicyCheckpoint`

**接口定义**：

```python
@dataclass(frozen=True)
class PolicyUpdateResult:
    track: Track
    surrogate_objective: float
    clip_fraction: float
    kl_divergence: float
    mean_advantage: float
    parameter_delta_norm: float
    checkpoint: CausalPolicyCheckpoint
    modification_record: SelfModificationRecord

class ZSpacePolicyOptimizer:
    def optimize(
        self,
        *,
        rollouts: tuple[ZRollout, ...],
        policy: CausalZPolicy,
        track: Track,
        n_epochs: int = 3,
        batch_size: int = 16,
        clip_ratio: float = 0.2,
        max_kl: float = 0.05,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> PolicyUpdateResult: ...
```

**验收**：
- 给定固定 rollout 数据，优化器能产出参数更新
- 更新后 `surrogate_objective` > 0
- `SelfModificationRecord` 正确记录 old/new hash

### U01.2 — Joint loop 集成

**owner**: `joint_loop`
**位置**: `volvence_zero/joint_loop/runtime.py`

**内容**：

1. 在 `ETANLJointLoop.run_joint_cycle()` 中，RL 阶段调用 `ZSpacePolicyOptimizer`
2. 将 `DualTrackRollout` 拆分为双轨，分别优化
3. 将 `PolicyUpdateResult` 合并回 `JointCycleReport`
4. 验证逻辑：连续 N 个 cycle 的 `total_reward` 趋势
5. 回滚逻辑：如果 `kl_divergence` 连续超标或 `total_reward` 持续下降，恢复到上一个 checkpoint

**修改的数据流**：

```
现有: run_rollout → compute_optimization_report → (统计量填充)
目标: run_rollout → optimize → apply_update → compute_report → (真实优化结果)
```

**约束**：

- 优化只在 `rl_interval` 对齐的 cycle 执行
- 优化前保存 checkpoint，优化后验证，失败回滚
- `schedule_telemetry` 中记录优化事件

**验收**：
- `ETANLJointLoop` 连续运行 10 个 cycle，RL 阶段的 `surrogate_objective` 非零
- `JointCycleReport.total_reward` 在 5 个 cycle 后展现非随机趋势
- 回滚在 KL 超标时自动触发

### U01.3 — 信用信号→参数更新闭环

**owner**: `credit` + `internal_rl`
**位置**: `volvence_zero/credit/gate.py`, `volvence_zero/internal_rl/optimizer.py`

**内容**：

1. 信用 reward shaping：
   - `derive_abstract_action_credit()` 产出的 `CreditRecord` 作为 RL 的稀疏 bonus
   - bonus 权重可配，默认 0.1（不让稀疏信号主导短期学习）
2. 门控检查集成：
   - 策略更新前调用 `ModificationGate.ONLINE` 检查
   - 检查 `has_blocking_writeback()` 确认无冲突写回
   - 门控 DENY 时记录 `SelfModificationRecord`（decision=BLOCK）但不更新参数
3. 信用→reward 映射：
   - `session_level_credits` 中的 abstract-action 级信用聚合为 per-rollout reward bonus
   - 双轨独立：WORLD track credit 只加到 task rollout，SELF track credit 只加到 relationship rollout
4. 审计路径：
   - 每次策略更新的 `RuntimeAdaptationAudit` 含 credit-shaped reward 的明细

**约束**：

- 信用 bonus 不改变 reward 的符号（不把负 reward 翻转为正）
- 门控 DENY 不阻塞 SSL 训练，只阻塞策略参数更新

**验收**：
- 有 credit bonus 的 rollout vs 无 bonus 的 rollout，优化器能区分并产出不同的更新
- 门控 BLOCK 时参数不变，audit 有记录
- 审计链完整：credit → reward shaping → optimize → modification record

## 4. 数据契约变更

| Schema | 变更类型 | 说明 |
|--------|---------|------|
| `OptimizationReport` | 语义升级 | 字段含义从"统计量填充"变为"真实优化结果" |
| `JointCycleReport` | 字段扩展 | 新增 `policy_update_applied: bool`, `policy_kl_divergence: float` |
| `CreditSnapshot` | 不变 | 只是被消费方式变了（从记录变为参与 reward shaping） |
| `CausalPolicyCheckpoint` | 不变 | 已有，增加保存频率 |

新增 schema：

| Schema | 位置 | 说明 |
|--------|------|------|
| `PolicyUpdateResult` | `internal_rl/optimizer.py` | 单次优化结果 |
| `ZSpacePolicyOptimizer` | `internal_rl/optimizer.py` | 策略优化器 |

## 5. 退出条件

Phase 1 视为完成，当且仅当以下全部满足：

1. `ZSpacePolicyOptimizer.optimize()` 能接收 rollout 并产出参数更新
2. `ETANLJointLoop` 中 RL 阶段调用优化器，报告中 `surrogate_objective` 非零
3. 连续 10 个 joint cycle 中 `total_reward` 展现可观测的学习趋势
4. 策略更新受门控约束，BLOCK 时参数不变且有 audit
5. 信用 bonus 参与 reward shaping，路径可审计
6. 每次更新前后自动保存 checkpoint

## 6. 回滚触发与回滚动作

### 回滚触发

- `kl_divergence` 连续 3 个 cycle 超过 `max_kl`
- `total_reward` 连续 5 个 cycle 下降
- 任何 `SelfModificationRecord` 中 `is_reversible == False`
- 门控系统出现异常（ONLINE gate 失效）

### 回滚动作

1. 恢复 `CausalPolicyCheckpoint` 到最近的安全检查点
2. 重置 `ZSpacePolicyOptimizer` 内部状态
3. 关闭 RL 优化阶段（只保留 rollout 收集），退化为 Phase 0 行为
4. 保留所有 `JointCycleReport` 和 `SelfModificationRecord` 用于诊断

## 7. 需同步更新的文档

| 文档 | 更新内容 |
|------|---------|
| `docs/specs/temporal-abstraction.md` | Internal RL 部分：从"接口存在"更新为"优化器可运行" |
| `docs/specs/credit-and-self-modification.md` | 信用→参数更新路径：从"信号记录"更新为"信号驱动" |
| `docs/specs/multi-timescale-learning.md` | SSL-RL 交替：从"骨架"更新为"RL 半程可运行" |
| `docs/DATA_CONTRACT.md` | 新增 `PolicyUpdateResult` schema |
| `docs/implementation/01_package_registry.md` | P08 接线级别说明更新 |

## 8. 风险与缓解

| 风险 | 级别 | 缓解 |
|------|------|------|
| PPO 在低维 z-space 不收敛 | 高 | 先用合成环境验证；备选 REINFORCE baseline |
| CausalZPolicy 的 policy_score/log_prob 计算不精确 | 中 | 对比优化前后的 log_prob 一致性；必要时重构 log_prob |
| reward 信号过于稀疏导致学不动 | 中 | 先用 dense proxy reward 验证，逐步切换到 sparse |
| 优化器与 joint loop 的生命周期管理冲突 | 低 | 优化器无状态，每次调用接收 rollout 返回结果 |

## 9. 测试策略

1. **单元测试**：`ZSpacePolicyOptimizer` 在合成 rollout 数据上能产出更新
2. **集成测试**：`ETANLJointLoop` 10 cycle 运行不崩溃，报告字段完整
3. **回归测试**：确保 `test_phase2_eta_nl.py` 现有用例不受影响
4. **行为测试**：在合成环境中验证 reward 趋势，确认不是随机波动

## 10. 参考

- `docs/next_gen_emogpt.md` — Appendix B.5 Internal RL
- `docs/specs/temporal-abstraction.md` — Internal RL spec
- `docs/specs/credit-and-self-modification.md` — 信用→参数路径
- `volvence_zero/internal_rl/sandbox.py` — 现有 rollout 基础设施
- `volvence_zero/joint_loop/runtime.py` — 现有 joint loop
