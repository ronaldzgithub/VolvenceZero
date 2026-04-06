# 时间抽象与内部控制 Spec

> Status: draft
> Last updated: 2026-04-06
> 对应需求: R3, R4

## 要解决的问题

如何让系统在 token 之上的抽象层级做决策和学习，实现子目标级控制？

## 关键不变量

- 实时行为可通过内部状态转换引导，而非仅通过表面文本损失
- 抽象动作可组合、可训练、无需详尽手动标签
- 冻结基础模型是发现时间抽象的前提（ETA rate-distortion 证明）
- 内部控制空间维度低于原始 token 动作空间

## 工程挑战

- 实现 metacontroller：从残差流中发现时间抽象动作
- 实现切换单元：稀疏切换、组合泛化
- 实现 Internal RL：在控制器代码空间（而非 token 空间）执行强化学习
- 将抽象动作与产品级行为（regime、策略）对齐

## 算法候选

来自 `docs/next_gen_emogpt.md`：

### Metacontroller 架构（ETA 附录 B.3）

```
残差流 e_{1:T}
    │
    ├──→ 内部序列嵌入器 → s(e_{1:T})  [全局嵌入，训练时非因果]
    │
    ├──→ 编码器 (GRU) → μ_t, Σ_t → z̃_t ~ N(μ_t, Σ_t)
    │         │
    │         └──→ 切换单元 → β_t ∈ [0,1]
    │                  │
    │                  └──→ z_t = β_t ⊙ z̃_t + (1-β_t) ⊙ z_{t-1}  (Eq.2)
    │
    └──→ 解码器 (FFN) → U_t = Decoder(z_t)
              │
              └──→ 残差流控制: e_{t,l} ← e_{t,l} + U_t · e_{t,l}  (Eq.1)
```

### 切换单元

- `β_t ≈ 0`：保持当前控制器（继续执行当前抽象动作）
- `β_t ≈ 1`：切换到新控制器（开始新的抽象动作）
- 训练后自发学会准二值、稀疏切换行为，切换时刻对齐子目标边界

### Internal RL（ETA 附录 B.5）

| 概念 | 定义 |
|------|------|
| 观测 | 残差流激活 `e_{t,l}` |
| 动作 | 控制器代码 `z_t`（低维，`n_z < n_e`） |
| 环境 | 原始环境 + 冻结自回归模型 + 解码器 + 切换单元 |
| 策略 | 因果抽象动作策略 `π(z_t | e_{1:t})` |

**优势**：动作空间降维、时间尺度压缩、信用分配简化、探索效率提升。

### 自监督训练目标（ETA Eq.3）

```
L(φ) = Σ_{(o,a)~D*} Σ_t [
    -ln p_{θ,φ}(a_t | o_{1:t}, z_{1:t})     // 动作预测损失
    + α · D_KL(N(μ_t, Σ_t) || N(0, I))      // 先验匹配正则化
]
```

### CMS 增强 Metacontroller（NL×ETA 附录 C.2）

用 CMS 替换 GRU 编码器，获得多时间尺度记忆：
- 高频层：每步更新，跟踪当前子目标执行进度
- 中频层：每 c_2 步更新，记忆近期子目标序列模式
- 低频层：每 c_3 步更新，保存跨 episode 的策略偏好

## 接口契约

**消费的输入**：
- `substrate` 快照：当前可实现的 substrate surface；当前阶段优先消费 `feature_surface`，只有在 hook 可用时才消费 `residual_activations`
- `memory` 快照：相关记忆上下文
- `reflection` 快照：策略沉淀（控制器参数更新）

**产出的输出**：
- `temporal_abstraction` 快照：`TemporalAbstractionSnapshot`
  - 控制器状态（`z_t`, `β_t`, `steps_since_switch`）
  - 当前抽象动作的语义描述
  - 控制器参数哈希

**当前实现口径**：

- P08 先固定接口和状态 contract，不承诺完整 ETA 训练闭环
- 当前实现已支持 `placeholder` / `heuristic` / `learned-lite` 三类 temporal policy
- `learned-lite` 当前仍是最小可训练控制器，不等同于 full ETA metacontroller 或因果 `π(z_t | e_{1:t})`
- 第二阶段 runtime 已补充一个独立的参数化 causal z-policy sandbox，支持 dual-track rollout、checkpoint/rollback 和 trajectory-level clipped surrogate objective；当前仍是训练沙箱，不是正式 runtime owner
- `learned-lite temporal` 与 causal z-policy 当前共享同一控制器参数 store：internal RL 更新和 checkpoint/restore 会直接影响 temporal controller 的参数视图
- 当前 `TemporalModule` 默认以 `learned-lite` 作为 runtime owner policy，并可通过 owner API 导出 machine-readable metacontroller runtime state；这条导出链不改变 `temporal_abstraction` 公共 snapshot schema
- 后续可平滑替换为 learned-lite 或 full learned policy，而不改变 snapshot schema

**快照 schema**：见 `docs/DATA_CONTRACT.md` 3.2 节

## 与其他能力域的关系

| 关系 | 能力域 | 说明 |
|------|--------|------|
| 依赖 | 契约式运行时（5.5）| 通过快照发布控制器状态 |
| 依赖 | 多时间尺度学习（5.1）| 在 online-fast 时间尺度运行 |
| 被依赖 | 双轨学习（5.4）| 提供 z_task / z_rel 控制器代码 |
| 被依赖 | 认知 Regime（5.8）| 控制器切换与 regime 切换对齐 |
| 被依赖 | 信用分配（5.6）| 提供抽象动作级信用分配的基础 |
| 协作 | 评估体系（5.7）| F5 抽象质量评估 |

## 变更日志

- 2026-04-06: 补充 learned-lite temporal policy 的当前实现口径，并记录 runtime-visible metacontroller owner state
- 2026-03-25: 初始版本，从 SYSTEM_DESIGN.md 和 next_gen_emogpt.md 提取
