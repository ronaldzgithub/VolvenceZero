# U02 — Phase 2: CMS 升维

> Status: draft
> Last updated: 2026-04-09
> Phase owner: `memory` (CMSMemoryCore)
> 差距族: B（CMS 向量→MLP 跨越）
> 影响需求: R1, R2, R5
> 前置条件: P00–P09 收敛包完成
> 可与 Phase 1 并行
> 预估周期: 2–3 周

## 1. Phase 目标

将 CMS 从 `dim=3` 的向量级模拟升级为参数化 MLP 层级，使其有足够的表达容量承载真实的多时间尺度知识存储。

当前 CMS 在概念层面正确实现了 NL 的核心设计——多频率带、动量更新、节奏门控、抗遗忘回流。但向量级表示的容量极其有限（3 个浮点数 × 3 个带 = 9 个参数），与 NL 设计中"每个频率层配备独立 MLP 知识存储"（附录 A.5）的要求差距巨大。

## 2. 设计依据

### 2.1 NL CMS 形式化（Appendix A.5）

```
y_t = MLP^(ν_K)(MLP^(ν_{K-1})(... MLP^(ν_1)(x_t)))

W^(ν_i)_{t+1} = W^(ν_i)_t - η^(i) · ∇ε(W^(ν_i)_t; x_t)   如果 t ≡ 0 (mod c^(i))
W^(ν_i)_{t+1} = W^(ν_i)_t                                    否则
```

每层 MLP 有独立参数 `W^(ν_i)`，独立学习率 `η^(i)`，独立更新节奏 `c^(i)`。

### 2.2 CMS 三种变体

| 变体 | 特点 | 优先级 |
|------|------|--------|
| 顺序 CMS | MLP 串联，初始状态通过最低频率层反向传播连接 | 默认 |
| 嵌套 CMS | 第 `i+1` 层初始状态由第 `i` 层元学习 | 后续开启 |
| 独立 CMS | 各 MLP 独立，通过聚合函数合并 | 后续开启 |

### 2.3 当前代码基础

`CMSMemoryCore` 已有：

- 三频带架构：`_online_fast`、`_session_medium`、`_background_slow`
- 节奏门控：`_integrate_signal_gradient()` 实现 `c^(i)` 间隔触发
- 梯度风格更新：`_gradient_update()` 的 error → momentum → apply
- 抗遗忘回流：`_apply_anti_forgetting()` 慢→快
- Encoder feedback：`observe_encoder_feedback()` 接收 metacontroller 信号
- Checkpoint/restore：`CMSCheckpointState`

所有这些机制可以平移到 MLP 级——更新规则不变，只是 `current`/`target` 从向量变为 MLP 参数。

## 3. 步骤分解

### U02.1 — MLP 级 CMS band

**owner**: `memory`
**位置**: `volvence_zero/memory/cms.py`

**内容**：

1. 实现 `CMSBandMLP` 类——每个频带的 MLP 知识存储：
   - 2 层 MLP：`y = x + W_1 · σ(W_2 · x)`（与 Hope 的 `M_α(x)` 对齐）
   - 输入维度 `d_in`，隐藏维度 `d_hidden`，输出维度 `d_out = d_in`
   - 残差连接确保初始化时为恒等映射
   - 默认配置：`d_in=16, d_hidden=32`（总参数 ~1K/band）

2. 将 `CMSMemoryCore` 内部表示从 `tuple[float, ...]` 升级为 `CMSBandMLP`：
   - `_online_fast` → `CMSBandMLP(cadence=1, lr=online_lr)`
   - `_session_medium` → `CMSBandMLP(cadence=session_cadence, lr=session_lr)`
   - `_background_slow` → `CMSBandMLP(cadence=background_cadence, lr=background_lr)`

3. 更新机制升级：
   - `_gradient_update()` → MLP 参数级 SGD with momentum
   - 信号 `signal` 作为 MLP 的目标输出，计算 MSE loss 并反向传播
   - 保持每 band 独立动量

4. 抗遗忘回流升级：
   - 慢 band MLP 参数部分回流到快 band（通过学习率加权混合）
   - 回流强度仍由 `anti_forgetting` 参数控制

5. 兼容性：
   - 新增配置参数 `mode: Literal["vector", "mlp"] = "vector"`
   - `mode="vector"` 时保持现有行为，所有测试不受影响
   - `mode="mlp"` 时使用 MLP 实现

**接口定义**：

```python
class CMSBandMLP:
    def __init__(
        self,
        *,
        d_in: int = 16,
        d_hidden: int = 32,
        learning_rate: float = 0.1,
        momentum_beta: float = 0.9,
    ) -> None: ...

    def forward(self, x: tuple[float, ...]) -> tuple[float, ...]: ...
    def update(self, *, target: tuple[float, ...], signal: tuple[float, ...]) -> None: ...
    def export_params(self) -> tuple[tuple[float, ...], ...]: ...
    def restore_params(self, params: tuple[tuple[float, ...], ...]) -> None: ...
    def parameter_count(self) -> int: ...
```

**约束**：

- 总参数量 < 5K per band（三个 band 合计 < 15K）
- 在线 band 单步更新延迟 < 2ms（CPU）
- `CMSState` / `CMSCheckpointState` 格式向后兼容（`mode="vector"` 不变）
- MLP 实现不依赖 PyTorch/TensorFlow——用纯 Python tuple 算术（与现有代码风格一致）

**验收**：
- `mode="vector"` 时所有现有测试通过
- `mode="mlp"` 时 `observe_substrate()` + `reflect_lessons()` 正常运行
- MLP band 的 `forward()` 输出在多轮 `update()` 后收敛到目标信号
- 三个 band 展现出不同的更新节奏（online 每步更新，session 每 `c_2` 步，background 每 `c_3` 步）

### U02.2 — CMS 变体支持

**owner**: `memory`
**位置**: `volvence_zero/memory/cms.py`

**内容**：

1. 顺序 CMS（默认）：
   - MLP 串联：`y_t = MLP_slow(MLP_medium(MLP_fast(x_t)))`
   - 初始状态通过最低频率层连接
   - 与当前 `observe_substrate()` 的数据流一致

2. 独立 CMS：
   - 各 MLP 独立处理输入
   - 聚合函数：`y_t = Agg(MLP_fast(x_t), MLP_medium(x_t), MLP_slow(x_t))`
   - 聚合方式：加权平均（权重可配）
   - 适合快速实验，不依赖 band 间的串联顺序

3. 嵌套 CMS（高级）：
   - 第 `i+1` 层 MLP 初始状态由第 `i` 层元学习
   - 上下文结束时重新初始化高频层
   - 实现高阶 in-context learning

4. 配置接口：

```python
class CMSVariant(str, Enum):
    SEQUENTIAL = "sequential"
    INDEPENDENT = "independent"
    NESTED = "nested"
```

**约束**：

- 默认仍为 `SEQUENTIAL`
- 三种变体共享 `CMSBandMLP` 底层
- 变体切换不影响 checkpoint 格式（参数可互相加载）

**验收**：
- `SEQUENTIAL` 变体的行为与升级前一致
- `INDEPENDENT` 变体三个 band 可独立运行和更新
- `NESTED` 变体高频层在上下文切换后能快速适应

### U02.3 — Encoder feedback MLP 级集成

**owner**: `memory` + `temporal`
**位置**: `volvence_zero/memory/cms.py`, `volvence_zero/temporal/interface.py`

**内容**：

1. CMS 接收 metacontroller encoder 输出：
   - `observe_encoder_feedback()` 升级为 MLP 级
   - Encoder 信号作为 `CMSBandMLP` 的额外训练目标
   - 仅影响 online-fast 和 session-medium band

2. CMS 接收 action family observation：
   - 新增 `observe_family_signal()`
   - action family 的 `latent_centroid`/`stability`/`support` 作为 session-medium band 信号
   - 帮助中频 band 积累 family 级模式

3. Temporal module 侧协调：
   - `TemporalModule.process()` 在产出快照后，向 CMS 发送 encoder feedback
   - 通过 post-propagate hook 或 `run_final_wiring_turn` 中的协调逻辑实现
   - 保持快照隔离：CMS 接收的是 temporal 快照的公开信息，不直接访问 temporal 内部

**约束**：

- encoder feedback 不阻塞 temporal 的 snapshot 产出
- family signal 只在有 `DiscoveredActionFamily` 时发送
- 新信号不改变 background-slow band 的更新逻辑（background 仍由反思驱动）

**验收**：
- 有 encoder feedback 时，online-fast band 的更新比无 feedback 时更快收敛
- action family observation 后，session-medium band 的表示包含 family 相关结构
- 各 band 的更新节奏不受新信号干扰（仍严格遵循 cadence 间隔）

## 4. 数据契约变更

| Schema | 变更类型 | 说明 |
|--------|---------|------|
| `CMSBandState` | 扩展 | 新增 `mode: str = "vector"`, `mlp_param_count: int = 0` |
| `CMSState` | 扩展 | 新增 `variant: str = "sequential"` |
| `CMSCheckpointState` | 扩展 | 新增 `mlp_params: tuple[tuple[tuple[float,...],...],...] = ()` |

新增 schema：

| Schema | 位置 | 说明 |
|--------|------|------|
| `CMSBandMLP` | `memory/cms.py` | MLP 级 band 实现 |
| `CMSVariant` | `memory/cms.py` | CMS 变体枚举 |

所有扩展字段有默认值，不破坏现有 `mode="vector"` 的行为。

## 5. 退出条件

Phase 2 视为完成，当且仅当以下全部满足：

1. `CMSMemoryCore(mode="mlp")` 三个 band 均为 MLP 实现，参数量可配
2. 三个 band 展现不同更新节奏和知识内容分化
3. 抗遗忘回流在 MLP 模式下正常工作（慢 band 知识可回流到快 band）
4. 至少两种 CMS 变体（`SEQUENTIAL` + `INDEPENDENT`）可运行
5. Encoder feedback 和 family observation 信号在 MLP 模式下正常集成
6. 所有现有测试在 `mode="vector"` 下通过（向后兼容）
7. MLP 模式下新增测试覆盖核心路径

## 6. 回滚触发与回滚动作

### 回滚触发

- MLP 模式下 CMS 更新延迟 > 5ms（CPU 单步）
- MLP 模式下知识表示不收敛（loss 不下降）
- MLP 模式导致下游模块（memory、dual_track）的快照内容退化
- 抗遗忘回流在 MLP 模式下导致参数发散

### 回滚动作

1. 切换 `mode` 回 `"vector"`
2. 从 `CMSCheckpointState` 恢复向量级状态
3. 禁用 MLP 相关的 encoder feedback / family observation
4. 保留 MLP 模式的实验数据用于分析

## 7. 需同步更新的文档

| 文档 | 更新内容 |
|------|---------|
| `docs/specs/multi-timescale-learning.md` | CMS 实现深度：从向量级更新为 MLP 级 |
| `docs/specs/continuum-memory.md` | 记忆容量：从 dim=3 更新为可配 MLP |
| `docs/DATA_CONTRACT.md` | `CMSBandState`/`CMSState`/`CMSCheckpointState` 的扩展字段 |

## 8. 风险与缓解

| 风险 | 级别 | 缓解 |
|------|------|------|
| 纯 Python MLP 计算性能不足 | 中 | 轻量维度（16x32）+ 延迟预算检查；未来可切换到 NumPy |
| MLP 参数初始化导致恒等映射偏移 | 中 | 使用残差连接 `y = x + W1·σ(W2·x)`，W1 初始化为零 |
| CMS 变体选择过多增加维护负担 | 低 | 默认 SEQUENTIAL 不变，其他变体延后到 Phase 3 需要时再激活 |
| MLP checkpoint 体积膨胀 | 低 | 每 band < 5K 参数，序列化后 < 50KB |

## 9. 测试策略

1. **单元测试**：`CMSBandMLP` 的 forward/update/export/restore 正确性
2. **等价测试**：`mode="vector"` 下所有 `test_memory_store.py` 测试通过
3. **MLP 收敛测试**：给定固定信号序列，MLP band 的输出向目标收敛
4. **多频率测试**：三个 band 在不同 cadence 下独立更新，互不干扰
5. **抗遗忘测试**：快 band 遗忘旧知识后，从慢 band 回流恢复
6. **性能测试**：MLP 模式单步更新延迟测量

## 10. 参考

- `docs/next_gen_emogpt.md` — Appendix A.5 CMS, A.7 Hope 架构
- `docs/specs/multi-timescale-learning.md` — R1/R2/R13
- `docs/specs/continuum-memory.md` — R5/R6
- `volvence_zero/memory/cms.py` — 现有 CMS 实现
